"""The user-facing DSL: ``JitContext`` with the ``jit`` and ``aot``
decorators.

A decorated function is *registered* in its context - the registration
records the function and binds a callable handle
(``_RegisteredFunction``) to the decorated name - and parsed (astgen)
only when it is first used.  A Python-side call goes through the
handle, which at call time

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. solves the concrete spy types of the parameters (in jit mode from
   the marshaled types of the arguments plus type-parameter unification,
   in aot mode from the annotations),
3. marshals every argument to the parameter types,
4. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see
``interp``); calling it is compiled to a native ``call``.
"""

import ctypes
from types import FunctionType
from typing import Any, dataclass_transform

from typing_extensions import override

from . import astgen, mir
from .builtins import AsValue
from .errors import CompileError, SpyError, TypeMismatchError
from .fn import FunctionEntry, FunctionValue, LazyJitFunction, LazyJitFunctionInstance
from .interp import FunctionResolver, HirRunner
from .lower import NativeFn, compile_module
from .type import (
    BoolType,
    FloatType,
    FormalArg,
    FunctionCallInfo,
    IntType,
    PointerType,
    Type,
    Value,
    function_call_info,
    int_range,
    to_mir_type,
    type_str,
    value_type,
)
from .type import (
    FunctionType as SpyFunctionType,
)
from .type import (
    StructType as SpyStructType,
)

# ---------------------------------------------------------------------------
# Python values of struct types: a struct instance is a ctypes.Structure
# subclass instance whose memory follows the LLVM layout of the struct (so
# native functions may read and - through a pointer parameter - modify it
# in place).
# ---------------------------------------------------------------------------


class StructInstance(ctypes.Structure):
    """The common base of the Python classes of struct instances (see
    ``StructType._py_cls``).  Every instance knows its spy struct type
    through the ``__spy_struct_type__`` attribute of its class."""


_CTYPE_INT = {
    (8, True): ctypes.c_int8,
    (8, False): ctypes.c_uint8,
    (16, True): ctypes.c_int16,
    (16, False): ctypes.c_uint16,
    (32, True): ctypes.c_int32,
    (32, False): ctypes.c_uint32,
    (64, True): ctypes.c_int64,
    (64, False): ctypes.c_uint64,
}


def ctypes_of(type: Type) -> type:
    """The ctypes field type mirroring the LLVM layout of a spy type."""
    match type:
        case BoolType():
            return ctypes.c_bool
        case IntType():
            ct = _CTYPE_INT.get((type.bits, type.signed))
            if ct is None:
                raise CompileError(f"integer type {type_str(type)} has no ctypes mapping")
            return ct
        case FloatType():
            return ctypes.c_float if type.bits == 32 else ctypes.c_double
        case SpyStructType():
            return type._py_cls
        case _:
            raise CompileError(
                f"type {type_str(type)} cannot be stored in a struct field"
            )


def _coerce_py(value: object, target: Type, what: str) -> object:
    """Convert one Python value to the representation the ctypes boundary
    of ``target`` expects (range checks included), for the argument of a
    struct constructor or a struct field."""
    match target:
        case BoolType():
            if not isinstance(value, bool):
                raise TypeMismatchError(
                    f"type mismatch: a {type(value).__name__} value cannot be "
                    f"passed as {what} (expected bool)"
                )
            return bool(value)
        case IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeMismatchError(
                    f"type mismatch: a {type(value).__name__} value cannot be "
                    f"passed as {what} (expected {type_str(target)})"
                )
            lo, hi = int_range(target)
            if not lo <= value <= hi:
                raise TypeMismatchError(
                    f"integer {value} is out of range for type {type_str(target)} "
                    f"in {what}"
                )
            return int(value)
        case FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeMismatchError(
                    f"type mismatch: a {type(value).__name__} value cannot be "
                    f"passed as {what} (expected {type_str(target)})"
                )
            try:
                return float(value)
            except OverflowError:
                raise TypeMismatchError(
                    f"cannot convert {value} to {type_str(target)} in {what}"
                ) from None
        case PointerType():
            if not isinstance(value, str):
                raise TypeMismatchError(
                    f"type mismatch: a {type(value).__name__} value cannot be "
                    f"passed as {what} (expected a string)"
                )
            return value.encode('utf-8')
        case SpyStructType():
            if not isinstance(value, target._py_cls):
                raise TypeMismatchError(
                    f"type mismatch: a {type(value).__name__} value cannot be "
                    f"passed as {what} (expected a {target.name} struct)"
                )
            return value
        case _:
            raise TypeMismatchError(
                f"type {type_str(target)} is not supported as {what}"
            )


def _sanitize(name: str) -> str:
    return ''.join(c if (c.isalnum() or c in '_.') else '_' for c in name)


def symbol_of(fn_name: str, arg_types: tuple[Type, ...]) -> str:
    """The native symbol name of one specialization.  ``fn_name`` must
    be the context-unique name allocated at registration (see
    ``JitContext._allocate_name``); the suffix of ``type_str`` strings
    distinguishes the specializations of one function."""
    parts = ['spy', _sanitize(fn_name)]
    parts.extend(_sanitize(type_str(t)) for t in arg_types)
    return '.'.join(parts)


def _types_str(arg_types: tuple[Type, ...]) -> str:
    return ', '.join(type_str(t) for t in arg_types)


def _candidate_type(fn_name: str, param_name: str, value: object) -> Type:
    """The spy type a provided Python argument marshals to."""
    if isinstance(value, AsValue):
        return value.type
    t = value_type(value)
    if t is None:
        raise TypeMismatchError(
            f"cannot pass a {type(value).__name__} value as the '{param_name}' argument "
            f"of {fn_name} (not a spy value)"
        )
    return t


def _marshal(fn_name: str, param_name: str, value: object, target: Type) -> object:
    """Convert one Python argument to the native calling convention of
    ``target`` (returns the value handed to the ctypes entry point)."""
    if isinstance(value, AsValue):
        if value.type != target:
            raise TypeMismatchError(
                f"type mismatch: spy.as(..., {type_str(value.type)}) cannot be passed "
                f"as the '{param_name}' argument of {fn_name} (expected "
                f"{type_str(target)})"
            )
        value = value.value

    what = f"the '{param_name}' argument of {fn_name}"
    if isinstance(target, PointerType) and isinstance(target.elem, SpyStructType):
        # a struct *pointer* parameter (the ``self`` of a ``ptr_self``
        # method): the Python instance is passed by reference - the
        # callee sees (and may modify) its native memory
        value = _coerce_py(value, target.elem, what)
        return ctypes.addressof(value)  # type: ignore[arg-type]
    if isinstance(target, SpyStructType):
        # a by-value struct parameter: the Python-facing entry is a
        # pointer-form thunk (see ``lower._ctypes_thunk``) that loads
        # the struct and calls the by-value function, so the instance is
        # passed by reference and never modified
        value = _coerce_py(value, target, what)
        return ctypes.addressof(value)  # type: ignore[arg-type]
    return _coerce_py(value, target, what)


def _bound_method(handle: Any) -> Any:
    """The Python method of a struct instance: ``bar.hkm()`` binds the
    instance as the first argument of the method's registration
    handle."""

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        return handle(self, *args, **kwargs)

    method.__name__ = handle.__name__
    method.__qualname__ = (
        f'{handle._method.name}.{handle.__name__}' if handle._method else handle.__name__
    )
    return method


class _RegisteredFunction:
    """The registration handle of a spy function.  The decorated name
    in a module (or an enclosing factory) scope binds to this handle, so
    it is both the object Python-side calls go through and the object a
    spy body references when it uses the function.  Registration itself
    never parses the function: the function entry - whose construction
    parses the body - is created at the first use and cached in
    ``_entry`` (see ``entry``).

    A function decorated *inside a struct class* is a *method*: the
    struct (``JitContext.struct``) marks its handle with the owning
    struct type (``_method``).  Its first parameter is then ``self``,
    typed by the struct instead of by an annotation, and Python-side
    calls go through the instance (``bar.hkm()``); ``ptr_self`` decides
    whether ``self`` is passed by value or by pointer.
    """

    __slots__ = ('_context', '_entry', '_fn', '_kind', '_method', '_name_base', '_ptr_self')

    def __init__(
        self,
        context: 'JitContext',
        fn: FunctionType,
        kind: str,
        name_base: str,
        ptr_self: bool = False,
    ) -> None:
        self._context = context
        self._fn = fn
        self._kind = kind
        # the context-unique base name of the native symbols (see
        # ``JitContext._allocate_name``)
        self._name_base = name_base
        # whether the first parameter of a method is passed by pointer
        # (``@aot(ptr_self=True)``; meaningful only for methods)
        self._ptr_self = ptr_self
        # the owning struct type, when the function is the method of a
        # struct (see ``JitContext.struct``)
        self._method: SpyStructType | None = None
        # the function entry, created at the first use
        self._entry: FunctionEntry | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._method is not None:
            # a method called from Python: the first argument is the
            # struct instance (``bar.hkm()`` calls ``handle(bar)``)
            if len(args) == 0:
                raise TypeError(
                    f'{self._fn.__name__}() is a spy method of {self._method.name}: '
                    'call it on a struct instance, e.g. instance.hkm()'
                )
            instance = args[0]
            if not isinstance(instance, self._method._py_cls):
                raise TypeError(
                    f'{self._fn.__name__}() is a spy method of '
                    f'{self._method.name}: its first argument must be a '
                    f'{self._method.name} instance'
                )
            return self._context._dispatch_method(
                self, instance, args[1:], kwargs
            )
        return self._context._dispatch(self.entry(), args, kwargs)

    def entry(self) -> FunctionEntry:
        """The entry of this registered function, creating it on first
        use.

        Registration never parses.  The first use - a Python-side call
        or a reference from inside a spy body - creates the entry here:
        the body is parsed (astgen) and, for an aot function, its fixed
        signature is resolved from the annotations at the same time.
        The entry is cached in ``_entry``.
        """
        entry = self._entry
        if entry is not None:
            return entry
        fn_ir = astgen.parse_function(self._fn)
        if self._kind == 'aot':
            if self._method is None:
                # resolve the fixed signature from the annotations
                params = fn_ir.params
                formal, ret = astgen.solve_call_types(fn_ir, 'aot', (None,) * len(params))
                args = tuple(FormalArg(params[i].name, formal[i]) for i in range(len(params)))
            else:
                args, ret = self._method_args(fn_ir)
            entry = FunctionValue(self._fn, fn_ir, args, ret)
        elif self._kind == 'jit':
            entry = LazyJitFunction(self._fn, fn_ir)
        else:
            raise ValueError(f'unknown function kind {self._kind!r}')
        entry.context = self._context
        entry.name_base = self._name_base
        self._entry = entry
        return entry

    def _method_args(
        self, fn_ir: astgen.FunctionIR
    ) -> tuple[tuple[FormalArg, ...], Type | None]:
        """The signature of an ``aot`` method: its first parameter is
        ``self``, typed by the owning struct (by value, or by pointer
        with ``ptr_self``); the other parameters are annotated like any
        aot parameter; the return type is the declared one, or None when
        it is inferred from the body (a method may return nothing)."""
        assert self._method is not None
        params = fn_ir.params
        if len(params) == 0:
            raise CompileError(
                f'method {self._fn.__name__} of {self._method.name} must take '
                "a 'self' parameter"
            )
        self_type: Type = self._method
        if self._ptr_self:
            self_type = PointerType(self._method)
        args = [FormalArg(params[0].name, self_type)]
        for param in params[1:]:
            args.append(FormalArg(param.name, astgen.annotation_type(fn_ir, param)))
        if fn_ir.ret_annotation is not None:
            ret = astgen.return_annotation_type(fn_ir)
        else:
            # no return annotation: the return type is inferred from the
            # body (a body that never returns a value is a void method)
            ret = None
        return tuple(args), ret

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def __repr__(self) -> str:
        return f'<spy {self._kind} function {self._fn.__name__}>'

def _native_spec(entry: FunctionEntry, arg_types: tuple[Type, ...]) -> NativeFn | None:
    """The compiled native function of one specialization, or None
    when it is not (yet) compiled.

    An aot function has exactly one specialization, fixed by its
    annotations and lowered at its first use: its native function is
    stored on the value (the value also keeps the MIR, in
    ``mir_fn``).  A jit specialization stores it - together with its
    call lowering plan - in a :class:`LazyJitFunctionInstance` (see
    ``LazyJitFunction.specs``)."""
    if isinstance(entry, FunctionValue):
        return entry.native_fn
    instance = entry.specs.get(arg_types)
    return instance.native_fn if instance is not None else None


def _spec_function_type(
    entry: FunctionEntry, arg_types: tuple[Type, ...], ret: Type
) -> SpyFunctionType:
    """The spy function type of one specialization: its (concrete)
    argument types bound to the names of its parameters, and its
    logical return type.  The call lowering plan of the specialization
    is derived from this type (see ``type.function_call_info``)."""
    params = entry.hir.params
    return SpyFunctionType(
        tuple(FormalArg(params[i].name, arg_types[i]) for i in range(len(arg_types))),
        ret,
    )


def _recursion_ret_type_error(entry: FunctionEntry, arg_types: tuple[Type, ...]) -> str:
    """Explain why the return type of a recursive specialization
    cannot be determined from its annotations."""
    fn_ir = entry.hir
    ret_ann = fn_ir.ret_annotation
    name = entry.fn.__name__
    if ret_ann is None:
        return (
            f"recursive function {name} requires a return type annotation: "
            'the return type of a recursive call must be known while '
            'the body is being compiled'
        )
    for type_param in fn_ir.type_params:
        if ret_ann is type_param:
            return (
                f"cannot determine the return type of the recursive function {name}: "
                f"type parameter {type_param.__name__} appears only in the return "
                'annotation, not on any parameter'
            )
    return (
        f"cannot determine the return type of the recursive function {name}: "
        f'the return annotation {ret_ann!r} is not a spy type'
    )


def _build_instance_class(cls: type, desc: SpyStructType) -> type:
    """The Python class of the struct instances: a ctypes.Structure
    subclass whose memory follows the LLVM layout of the struct (the
    same layout the native code compiles against)."""
    fields = [(f.name, ctypes_of(f.type)) for f in desc.fields]
    py_cls = type(
        cls.__name__,
        (StructInstance,),
        {'_fields_': fields, '__module__': cls.__module__, '__doc__': cls.__doc__},
    )
    py_cls.__spy_struct_type__ = desc  # type: ignore[attr-defined]
    return py_cls

def _copy_memory(instance: StructInstance, field: str, value: Any) -> None:
    """Copy the native memory of a nested struct instance into a
    struct field of another instance."""
    view = getattr(instance, field)
    ctypes.memmove(ctypes.addressof(view), ctypes.addressof(value), ctypes.sizeof(view))

def _write_fields(
    desc: SpyStructType,
    instance: StructInstance,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Write Python arguments into the fields of a fresh struct
    instance (in declaration order; keyword arguments by field
    name)."""
    fields = desc.fields
    bound: list[Any] = []
    names = [f.name for f in fields]
    if len(args) > len(fields):
        raise TypeError(
            f'{desc.name}() takes at most {len(fields)} arguments '
            f'({len(args)} given)'
        )
    for i, value in enumerate(args):
        bound.append((fields[i], value))
    for key, value in kwargs.items():
        if key not in names:
            raise TypeError(f"{desc.name}() got an unexpected keyword argument '{key}'")
        if key in [b[0].name for b in bound]:
            raise TypeError(f"{desc.name}() got multiple values for argument '{key}'")
        bound.append((fields[names.index(key)], value))
    if len(bound) != len(fields):
        missing = [n for n in names if n not in [b[0].name for b in bound]]
        raise TypeError(f"{desc.name}() missing required argument '{missing[0]}'")
    for field, value in bound:
        converted = _coerce_py(value, field.type, f"field '{field.name}' of {desc.name}")
        if isinstance(field.type, SpyStructType):
            # copy the memory of the nested struct into the field
            _copy_memory(instance, field.name, converted)
        else:
            setattr(instance, field.name, converted)

class JitContext(FunctionResolver):
    """A cache of compiled spy functions; functions decorated by the
    same context may call each other (as native calls)."""

    def __init__(self) -> None:
        # raw function -> its registration handle (see
        # ``_RegisteredFunction``); registration never parses, the
        # function entry is created at the first use
        self._entries: dict[FunctionType, _RegisteredFunction] = {}
        # parsed HIR of plain Python functions that are only ever
        # inlined (inlined functions are not registered);
        # a registered function's HIR is parsed with its entry and
        # cached on the entry instead
        self._inline_fn_hir_cache: dict[FunctionType, astgen.FunctionIR] = {}
        # allocated base names (``spy.<name>.<types>``) -> their owner
        # (the raw function object); keeps the native symbols of
        # different functions apart even when they happen to share a
        # ``__name__``
        self._name_owners: dict[str, FunctionType] = {}
        # the MIR functions defined by the module under construction
        # (see ``ensure_spec``), keyed like the per-function
        # ``mir_cache``; None outside a build
        self._module: dict[tuple[FunctionEntry, tuple[Type, ...]], mir.Function] | None = None

    # -- decorators ----------------------------------------------------------

    def jit(self, fn: FunctionType | None = None, *, ptr_self: bool = False):
        """``@cache.jit()``: compile lazily at the first call, using the
        marshaled types of the arguments (annotations have no effect on
        the chosen argument types; type parameters like ``T`` unify the
        parameters, and a declared return annotation fixes the return
        type of the specialization).  ``ptr_self`` is only meaningful for
        a method defined inside a struct class (see ``struct``)."""
        if fn is not None:
            return self._register(fn, 'jit', ptr_self)
        return lambda f: self._register(f, 'jit', ptr_self)

    def aot(self, fn: FunctionType | None = None, *, ptr_self: bool = False):
        """``@cache.aot()``: compile lazily at the first use from the
        (concrete) type annotations, which are required for every
        parameter (except the ``self`` of a method) and for the return
        type (except that a method without a return annotation infers it
        from its body).  Unlike a ``jit`` function an ``aot`` function
        has exactly one specialization, fixed by its annotations (a
        plain Python argument is still marshaled to the annotated type).
        ``ptr_self=True`` makes the ``self`` of a method a pointer
        parameter (``def hkm(self: spy.ptr(Bar))``), so that the method
        may modify the struct it is called on in place."""
        if fn is not None:
            return self._register(fn, 'aot', ptr_self)
        return lambda f: self._register(f, 'aot', ptr_self)

    @dataclass_transform()
    def struct[T](self):
        """``@cache.struct()``: turn a class whose annotations declare
        spy-typed fields into a spy struct type (bound to the class
        name).  Methods decorated with ``@cache.aot()``/``@cache.jit()``
        inside the class are the spy methods of the struct; a plain
        (undecorated) method is inlined on call; a ``def __init__``
        becomes the user constructor (default: arguments are written
        into the fields in declaration order)."""

        def wrapper(cls):
            return self._struct(cls)

        return wrapper
    # -- struct types ----------------------------------------------------------

    def _struct(self, cls: type) -> SpyStructType:
        """Build the spy struct type of ``cls``: the fields from the
        annotations (in declaration order), the spy methods from the
        class members, and the Python class of the instances (a ctypes
        layout mirroring the LLVM struct)."""
        annotations = getattr(cls, '__annotations__', {})
        desc = SpyStructType(cls.__name__)
        for name, annotation in annotations.items():
            if not isinstance(annotation, Type):
                raise CompileError(
                    f"field '{name}' of struct {cls.__name__} has no spy type: "
                    f'{annotation!r}'
                )
            desc.add_field(name, annotation)
        self._check_struct_cycles(desc, [desc])
        desc._py_cls = _build_instance_class(cls, desc)
        desc.__module__ = cls.__module__
        # collect the spy methods out of the class; a method that is not
        # decorated is a plain function and is inlined when called
        for name, value in list(cls.__dict__.items()):
            if name.startswith('__') and name != '__init__':
                # ``__init__`` is the (optional) user constructor; other
                # special members of the source class are ignored
                continue
            if name in ('__module__', '__qualname__', '__doc__', '__annotations__'):
                continue
            if isinstance(value, _RegisteredFunction):
                if value._context is not self:
                    raise CompileError(
                        f"method {value.__name__} of struct {cls.__name__} is "
                        'registered in another JitContext'
                    )
                if value._entry is not None:
                    raise CompileError(
                        f'method {value.__name__} of struct {cls.__name__} was '
                        'used before its struct was defined'
                    )
                if name == '__init__':
                    # a constructor always receives the result pointer
                    value._ptr_self = True
                else:
                    # Python-side method calls go through the instance
                    # (``bar.hkm()``), whose class carries a bound wrapper
                    setattr(desc._py_cls, name, _bound_method(value))
                value._method = desc
                desc.methods[name] = value
            elif isinstance(value, FunctionType):
                if name == '__init__':
                    # a plain ``__init__`` is registered as a jit-style
                    # method (its arguments have no annotations to solve
                    # from), with ``self`` bound to the result pointer
                    value = self._register(value, 'jit', True)
                    value._method = desc
                    desc.methods[name] = value
                else:
                    desc.methods[name] = value
                    setattr(desc._py_cls, name, value)
            else:
                raise CompileError(
                    f"unsupported member '{name}' of struct class {cls.__name__}: "
                    'struct classes may only contain annotated fields and spy '
                    'methods'
                )
        self._build_constructor(desc)
        return desc

    def _check_struct_cycles(self, desc: SpyStructType, stack: list[SpyStructType]) -> None:
        """Reject recursive struct definitions (a struct cannot contain a
        struct value of its own type)."""
        for field in desc.fields:
            if isinstance(field.type, SpyStructType):
                if field.type in stack:
                    names = ' -> '.join(t.name for t in stack + [field.type])
                    raise CompileError(f'recursive struct definition: {names}')
                self._check_struct_cycles(field.type, stack + [field.type])

    def _build_constructor(self, desc: SpyStructType) -> None:
        """The Python-side constructor ``Foo(a, b)`` of a struct: create
        the native instance and run the constructor (the ``__init__``
        method, or the default field-wise constructor) on it."""

        def construct(*args: Any, **kwargs: Any) -> Any:
            py_cls = desc._py_cls
            instance = py_cls()
            if desc.methods.get('__init__') is not None:
                # a user constructor: run it with ``self`` bound to the
                # native memory of the new instance
                handle = desc.methods['__init__']
                self._dispatch_method(handle, instance, args, kwargs)
                return instance
            # the default constructor: arguments are the fields
            _write_fields(desc, instance, args, kwargs)
            return instance

        desc._py_init = construct

    def _dispatch_method(
        self,
        handle: _RegisteredFunction,
        instance: StructInstance,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """A Python-side call of a spy method on a struct instance
        (``bar.hkm()``): bind the arguments to the parameters after
        ``self``, marshal the instance according to ``ptr_self`` and
        call the compiled method."""
        entry = handle.entry()
        fn_ir = entry.hir
        name = entry.fn.__name__
        assert handle._method is not None
        desc = handle._method
        params = fn_ir.params
        if len(params) == 0:
            raise TypeError(f'{name}() is missing its self parameter')
        self_type: Type = PointerType(desc) if handle._ptr_self else desc

        rest = params[1:]
        names = [p.name for p in rest]
        if len(args) > len(rest):
            raise TypeError(
                f'{name}() takes {len(rest)} positional arguments but {len(args)} were given'
            )
        present: dict[str, Any] = {}
        for i, value in enumerate(args):
            present[names[i]] = value
        for key, value in kwargs.items():
            if key not in names:
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
            if key in present:
                raise TypeError(f"{name}() got multiple values for argument '{key}'")
            present[key] = value

        if isinstance(entry, FunctionValue):
            formal = tuple(a.type for a in entry.args)
        else:
            # a jit method: solve the formal types from the marshaled
            # arguments; ``self`` is pinned to its pointer/value type
            provided: list[Type | None] = [self_type]
            for param in rest:
                if param.name in present:
                    provided.append(_candidate_type(name, param.name, present[param.name]))
                else:
                    provided.append(None)
            formal, _ = astgen.solve_call_types(fn_ir, 'jit', tuple(provided))

        marshaled = [_marshal(name, 'self', instance, self_type)]
        for i, param in enumerate(rest):
            if param.name in present:
                value = present[param.name]
            else:
                if not param.has_default:
                    raise TypeError(f"{name}() missing required argument '{param.name}'")
                value = param.default_value
            marshaled.append(_marshal(name, param.name, value, formal[i + 1]))
        spec = self.ensure_spec(entry, formal)
        return spec.call(*marshaled)

    # -- Python-side calls ---------------------------------------------------

    def _dispatch(self, entry: FunctionEntry, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Bind Python arguments to the formal parameters of ``entry``
        and call the (possibly just compiled) specialization; this is
        what the registration handle of a spy function forwards to."""
        fn_ir = entry.hir
        name = entry.fn.__name__
        params = fn_ir.params
        param_names = [p.name for p in params]

        if len(args) > len(params):
            raise TypeError(
                f"{name}() takes {len(params)} positional arguments but {len(args)} were given"
            )
        present: dict[str, Any] = {}
        for i, value in enumerate(args):
            present[param_names[i]] = value
        for key, value in kwargs.items():
            if key not in param_names:
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
            if key in present:
                raise TypeError(f"{name}() got multiple values for argument '{key}'")
            present[key] = value

        provided: list[Type | None] = []
        for param in params:
            if param.name in present:
                provided.append(_candidate_type(name, param.name, present[param.name]))
            else:
                provided.append(None)
        formal, _ = astgen.solve_call_types(fn_ir, entry.kind, tuple(provided))

        marshaled: list[Any] = []
        for i, param in enumerate(params):
            if param.name in present:
                value = present[param.name]
            else:
                if not param.has_default:
                    raise TypeError(f"{name}() missing required argument '{param.name}'")
                value = param.default_value
            marshaled.append(_marshal(name, param.name, value, formal[i]))

        spec = self.ensure_spec(entry, formal)
        return spec.call(*marshaled)

    # -- registry ------------------------------------------------------------

    @override
    def hir_of_plain_fn(self, fn: FunctionType) -> astgen.FunctionIR:
        # the HIR of a registered function is parsed when its entry is
        # created (at its first use) and cached on the entry; a plain
        # function (which is only ever inlined) is parsed and cached by
        # the context
        assert fn not in self._entries
        ir = self._inline_fn_hir_cache.get(fn)
        if ir is None:
            ir = astgen.parse_function(fn)
            self._inline_fn_hir_cache[fn] = ir
        return ir

    def ensure_spec(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> NativeFn:
        """Compile (or look up) the specialization of ``entry`` for
        ``arg_types`` and return its native function.

        Compilation happens module-at-a-time: a fresh function and every
        function it depends on that is not compiled yet are lowered
        together into one LLVM module (``define``s); specializations
        compiled in earlier modules are referenced as external symbols.
        """
        native = _native_spec(entry, arg_types)
        if native is not None:
            return native
        message: str | None = None
        if not isinstance(entry, FunctionValue):
            # a jit function may be retried through its wrapper, so its
            # failures are cached; an aot function simply recompiles on
            # its next call
            message = entry.failed.get(arg_types)
        if message is not None:
            raise CompileError(message)
        assert self._module is None, 'internal error: nested module compilation'

        self._module = {}
        try:
            self._compile_mir(entry, arg_types)
            return self._compile_module()[(entry, arg_types)]
        except SpyError as e:
            if not isinstance(entry, FunctionValue):
                entry.failed[arg_types] = str(e)
            raise CompileError(
                f"error while compiling {entry.fn.__name__}({_types_str(arg_types)}): {e}"
            ) from e
        finally:
            self._module = None

    def _compile_mir(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> mir.Function:
        """Produce (and cache) the typed MIR of one specialization.

        The function is created - with an empty body - and registered
        (in the cache of its entry and in the module under
        construction) *before* its body is run, so a call the body
        makes to it - recursion, direct or mutual - resolves to the
        very function being typed, whose signature is already fixed;
        running the body fills it in.  Calls to other functions may
        compile (MIR-wise) further functions, which are scheduled into
        the module under construction as well (see :meth:`ensure_spec`);
        the whole module is lowered at once.
        """
        key = (entry, arg_types)
        if isinstance(entry, FunctionValue):
            # an aot function has exactly one specialization: ``mir_fn``
            # is its cache slot
            fn = entry.mir_fn
        else:
            fn = entry.mir_cache.get(arg_types)
        if fn is not None:
            # the MIR is already being (or has been) typed, but no native
            # function is registered yet - the module build that produces
            # it is the one under construction (a recursive call, or a
            # build that never finished); the current build references
            # the function, so it is defined here too
            module = self._module
            if module is None:
                raise CompileError(
                    f'internal error: cached MIR of '
                    f'{entry.fn.__name__}({_types_str(arg_types)}) is reused '
                    'outside of a module build'
                )
            module[key] = fn
            return fn

        fn_ir = entry.hir
        assert len(arg_types) == len(fn_ir.params)
        # a declared return type - concrete, or naming a type parameter
        # bound by the parameters - fixes the return type of the
        # specialization (a recursive function needs it: its calls are
        # typed while its body is still being compiled); without one the
        # return type is inferred from the return sites.  An aot
        # function carries its (fixed) return type on its entry; a jit
        # function declares one by its return annotation (see
        # ``astgen.solve_call_types``)
        if isinstance(entry, FunctionValue):
            ret_hint = entry.ret
        else:
            ret_hint = astgen.solve_call_types(fn_ir, 'jit', arg_types)[1]
        args = tuple(
            mir.FormalArg(fn_ir.params[i].name, to_mir_type(arg_types[i]))
            for i in range(len(arg_types))
        )
        fn = mir.Function(
            symbol_of(entry.name_base, arg_types),
            args,
            to_mir_type(ret_hint) if ret_hint is not None else None,
            [],
        )
        module = self._module
        if module is None:
            raise CompileError(
                f"internal error: {entry.fn.__name__}({_types_str(arg_types)}) "
                'is compiled outside of a module build'
            )
        if isinstance(entry, FunctionValue):
            entry.mir_fn = fn
        else:
            entry.mir_cache[arg_types] = fn
        module[key] = fn
        try:
            runner = HirRunner(self)
            ret = runner.run_function(fn, fn_ir, arg_types, ret_hint)
            # the logical spy return type of the specialization (see
            # ``run_function``), kept with its MIR so that callers of a
            # later build can re-derive its function type without reading
            # the MIR types back
            fn.spy_ret = ret  # type: ignore[attr-defined]
            return fn
        except BaseException:
            # the body typing failed: drop the partially-filled function
            # so that a retry starts from scratch (functions completed
            # earlier in this build stay cached)
            if isinstance(entry, FunctionValue):
                entry.mir_fn = None
            else:
                entry.mir_cache.pop(arg_types, None)
            raise

    def _compile_module(self) -> dict[tuple[FunctionEntry, tuple[Type, ...]], NativeFn]:
        """Lower the module under construction (see :meth:`ensure_spec`)
        into one native module and register its specializations."""
        assert self._module is not None
        natives = compile_module(list(self._module.values()), self._extern_symbols())
        by_name = {native.name: native for native in natives}
        result: dict[tuple[FunctionEntry, tuple[Type, ...]], NativeFn] = {}
        for key, mir_fn in self._module.items():
            native = by_name[mir_fn.name]
            entry, arg_types = key
            if isinstance(entry, FunctionValue):
                # an aot function has exactly one specialization: its
                # single native function is stored on the value (see
                # ``_native_spec``)
                entry.native_fn = native
            else:
                # a jit specialization also records the call lowering
                # plan of its signature, so that spy functions of later
                # modules can call it without recomputing it
                ret = getattr(mir_fn, 'spy_ret', None)
                assert ret is not None, f'internal error: {entry.fn.__name__} is not typed'
                ft = _spec_function_type(entry, arg_types, ret)
                entry.specs[arg_types] = LazyJitFunctionInstance(native, function_call_info(ft))
            result[key] = native
        return result

    def _logical_spy_ret(
        self, entry: FunctionEntry, arg_types: tuple[Type, ...]
    ) -> Type:
        """The logical spy return type of one specialization: the type
        the callers of the specialization see.  For a specialization
        whose body has been typed it is the type the body inferred (or
        declared) - recorded on its MIR function when it was typed (see
        ``_compile_mir``); while a function is still being typed - a
        recursive call - it can only be its declared type."""
        fn = entry.mir_fn if isinstance(entry, FunctionValue) else entry.mir_cache.get(arg_types)
        if fn is not None:
            ret = getattr(fn, 'spy_ret', None)
            if ret is not None:
                return ret
        if isinstance(entry, FunctionValue):
            declared = entry.ret
        else:
            declared = astgen.solve_call_types(entry.hir, 'jit', arg_types)[1]
        if declared is None:
            raise CompileError(_recursion_ret_type_error(entry, arg_types))
        return declared

    @override
    def resolve_call(
        self, entry: FunctionEntry, arg_types: tuple[Type, ...]
    ) -> tuple[mir.Value, Type, FunctionCallInfo]:
        """The callable value of one callee specialization as seen from
        inside a compiled function: a :class:`mir.Function` of the
        module under construction - already compiled, or still being
        compiled when the call is a recursive one - or a
        :class:`mir.Symbol` of a specialization compiled in an earlier
        module.

        Together with it the *logical spy return type* of the
        specialization (all the interpreter's type checks happen on it)
        and its *call lowering plan* (a :class:`FunctionCallInfo`,
        derived from the specialization's spy function type - see
        ``type.function_call_info``): the interpreter emits the
        ``mir.Call`` by following the plan."""
        native = _native_spec(entry, arg_types)
        if native is not None:
            # native function exists: external symbol call
            fn_type = mir.FunctionType(native.arg_types, native.ret_type)
            callee: mir.Value = mir.Symbol(native.name, fn_type)
            ret = self._logical_spy_ret(entry, arg_types)
            if isinstance(entry, FunctionValue):
                info = function_call_info(entry.type())
            else:
                # a jit specialization recorded its plan when it was
                # registered (see ``_compile_module``)
                instance = entry.specs.get(arg_types)
                assert instance is not None, 'internal error: unregistered jit spec'
                info = instance.call_info
            return callee, ret, info
        fn = self._compile_mir(entry, arg_types)
        ret = self._logical_spy_ret(entry, arg_types)
        info = function_call_info(_spec_function_type(entry, arg_types, ret))
        return fn, ret, info

    @override
    def resolve_global(self, value: Any) -> Value | None:
        """The spy value a global object referenced inside a function
        body resolves to (see :class:`FunctionResolver`): a function
        registered in this context - reached as the raw function object
        or through the ``_RegisteredFunction`` its decorated name binds
        to - becomes its function entry (creating the entry of a
        function that is not used yet), and anything else returns
        ``None`` (the object stays a plain compile-time Python value)."""
        if isinstance(value, _RegisteredFunction):
            # a handle always belongs to the context that registered
            # its function
            if value._context is not self:
                return value._context.resolve_global(value._fn)
            value = value._fn
        if not isinstance(value, FunctionType):
            return None
        if value in self._entries:
            return self._entries[value].entry()
        return None

    @override
    def resolve_method(self, struct: SpyStructType, name: str) -> tuple[Any, bool] | None:
        """The spy method ``name`` of ``struct`` as seen from inside a
        compiled function body (see :class:`FunctionResolver`): the
        entry of the registered ``@aot``/``@jit`` method, or the plain
        Python function of an undecorated method (which the interpreter
        inlines), together with its ``ptr_self`` flag.  ``None`` when
        the struct has no such method."""
        method = struct.methods.get(name)
        if method is None:
            return None
        if isinstance(method, _RegisteredFunction):
            if method._method is None:
                method._method = struct
                if name == '__init__':
                    method._ptr_self = True
            if method._context is not self:
                raise CompileError(
                    f"cannot call method {method.__name__} of struct "
                    f'{struct.name} from another JitContext'
                )
            return method.entry(), method._ptr_self
        if isinstance(method, FunctionType):
            return method, False
        raise CompileError(
            f'method {name} of struct {struct.name} is not a spy method'
        )

    def _register(
        self, fn: FunctionType, kind: str, ptr_self: bool = False
    ) -> _RegisteredFunction:
        if not isinstance(fn, FunctionType):
            raise TypeError('spy decorators can only be applied to plain Python functions')
        if fn in self._entries:
            raise ValueError(f'function {fn.__name__} is already registered in this JitContext')
        # registration never parses (astgen happens only when the
        # function is used): it records the registration handle with the
        # context-unique base name of the native symbols; the function
        # entry - whose construction parses the body - is created at the
        # first use (see ``_RegisteredFunction.entry``)
        registered = _RegisteredFunction(self, fn, kind, self._allocate_name(fn), ptr_self)
        self._entries[fn] = registered
        return registered

    def _allocate_name(self, fn: FunctionType) -> str:
        """The context-unique base name of the native symbols of ``fn``.

        Two different functions registered in the same context may share
        a ``__name__`` (e.g. same-named functions of two modules, or two
        instances of a nested function); their symbols must still not
        collide, because extern calls are linked by name.  The plain
        ``__name__`` is kept when it is free; otherwise the
        module-qualified name is used, then numbered suffixes."""
        candidates: list[str] = [_sanitize(fn.__name__)]
        qualified = _sanitize(f'{fn.__module__}.{fn.__qualname__}')
        if qualified not in candidates:
            candidates.append(qualified)
        name: str | None = None
        for candidate in candidates:
            if candidate not in self._name_owners:
                name = candidate
                break
        if name is None:
            n = 2
            while True:
                candidate = f'{qualified}.{n}'
                if candidate not in self._name_owners:
                    name = candidate
                    break
                n += 1
        assert name is not None
        self._name_owners[name] = fn
        return name

    def _extern_symbols(self) -> dict[str, NativeFn]:
        """The addresses of the native functions compiled in *earlier*
        modules, keyed by their symbol names: everything a module under
        construction may reference from outside (see
        :meth:`_compile_module`).  Every registered function of the
        context contributes its compiled specializations (a function
        that was never used has none)."""
        externs: dict[str, NativeFn] = {}
        for registered in self._entries.values():
            entry = registered._entry
            if entry is None:
                continue
            if isinstance(entry, FunctionValue):
                native = entry.native_fn
                if native is not None:
                    externs[native.name] = native
            else:
                for instance in entry.specs.values():
                    externs[instance.native_fn.name] = instance.native_fn
        return externs

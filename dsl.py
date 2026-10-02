"""The user-facing DSL: the ``func`` and ``struct`` decorators.

A function decorated with ``func`` is *registered* in its host context - the
registration binds a callable handle (``_RegisteredFn``) to the decorated
name - and parsed (astgen) only when it is first used.  A Python-side call
goes through the handle, which at call time

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. specializes the signature: the declared generic type parameters are
   solved from the marshaled argument types, and each parameter then
   takes its (substituted) annotation, the marshaled type of the
   argument provided for it, or the spy type of its default value, in
   that order,
3. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

A *class* decorated with ``struct`` declares a spy struct type of the same
name: the annotated attributes of its body are the fields, in declaration
order, and the functions of its body are its methods (a decorated one is a
registered function, compiled into a native call, and a plain one is inlined
at its call sites).  The decorated name stands for that struct: a spy body
annotates with it, constructs it (``Foo(a, b)`` - the arguments fill the
fields in place, positional ones in declaration order and keyword ones by
name; a custom ``__init__`` is not supported) and calls its methods on a
value of it (``x.m()``, the object passed as the method's ``self``).  A class
with type parameters (``class Foo[T]``) declares a struct *template*:
``Foo[i32]`` names one specialization of it, a construction of the bare
template (``Foo(...)``) takes the arguments of the specialization from the
type of the location it is built into, and a method call carries the
specialization's type arguments into the method (``x.m()`` behaves like
``typeof(x).m(x)``, see ``interp``).  A struct is a compile-time type only:
Python-side construction is not supported yet.

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see ``interp``);
calling it is compiled to a native ``call``.  An *undecorated* plain
Python function reached the same way is inlined instead.  The ``spy.*``
builtins (``spy.typeof``, ``spy.compile_log``) are evaluated at compile
time.
"""

import ctypes
import types as pytypes
from dataclasses import dataclass
from typing import Any, Literal, NoDefault, TypeVar, cast, dataclass_transform, override

from . import astgen, mir, sval
from .builtins import spy_as, spy_compile_log, spy_typeof
from .errors import CompileError, SpyError
from .fn import (
    AnyValue,
    ArgList,
    Backend,
    CallSignature,
    FunctionValue,
    NativeFn,
    PartialReturnSignature,
    RawArgList,
    SpecializedComptimeArg,
    SymbolTable,
)
from .interp import Analyser
from .lower import LLVMBackend, to_ctype
from .sval import CompileContext, MirLowerCache, StructDecl
from .target import TargetInfo
from .util import FrozenArraySet, frozendict

# the ``spy.*`` builtins, by the name the interpreter knows them by
_BUILTINS: dict[Any, str] = {
    spy_typeof: 'typeof',
    spy_compile_log: 'compile_log',
    spy_as: 'as',
}


@dataclass(frozen=True)
class FnMetadata:
    sfv: bool  # self by value
    extern: bool
    linkname: str | None
    # an undecorated struct method: it is inlined at its call sites like a
    # plain Python function instead of being compiled into a native call
    inline: bool = False
    # the exceptions the function may raise: ``None`` (the default) means it
    # raises nothing, ``"infer"`` that they are inferred from the body, and a
    # tuple of spy struct classes the exceptions it may raise (in error-code
    # order)
    exceptions: tuple[type, ...] | Literal["infer"] | None = None
    # the calling convention (see ``sval.FunctionType.callconv``): ``'default'``
    # is the spy one, any other value names a C one
    callconv: str = 'default'
    # whether the function may panic; passed through only
    may_panic: bool = False


@dataclass(frozen=True, slots=True)
class FuncTypeMetadata:
    """The ``@func_type(...)`` declaration of a spy function *type* (see
    ``_FuncTypeDecl``)."""

    callconv: str
    may_panic: bool
    exceptions: tuple[type, ...] | Literal["infer"] | None


@dataclass(frozen=True)
class StructMetadata:
    extern_c: bool


# the metadata of an undecorated method (see ``_RegisteredClass.get_entry``)
_INLINE_META = FnMetadata(sfv=False, extern=False, linkname=None, inline=True)


def _normalize_exceptions(
    exceptions: type | tuple[type, ...] | Literal["infer"] | None,
) -> tuple[type, ...] | Literal["infer"] | None:
    """The ``exceptions`` declaration of ``@func``/``@func_type``: a single
    exception type is taken as the one-element tuple, so
    ``@func(exceptions=ErrorA)`` and ``@func(exceptions=(ErrorA,))`` mean the
    same thing; ``None`` and ``"infer"`` are kept as they are."""
    if exceptions is None:
        return None
    if isinstance(exceptions, str):
        return 'infer'
    if isinstance(exceptions, (tuple, list)):
        return tuple(exceptions)
    return (exceptions,)


def _to_py_arg(value: sval.AnyValue) -> Any:
    """The Python value a marshaled spy value is passed to the native
    function as."""
    match value:
        case sval.AsValue():
            return value.value
        case sval.Int():
            return value.value
        case sval.Float():
            return value.value
        case sval.Void():
            return None
        case sval.Null():
            return None
        case _:
            return value


def _call_multi_value(
    native_fn: NativeFn,
    py_args: list[Any],
    ret_spec: sval.RetSpec,
    mir_lower_cache: sval.MirLowerCache,
) -> tuple[Any, ...]:
    """Call a native artifact that returns several values from Python.  The
    lowered function returns one result directly and delivers every other
    one through a caller-provided result pointer, so a ctypes buffer is
    allocated for each of those pointers and the values are gathered into a
    tuple, in declaration order.  A nested ``tuple[...]`` result is gathered
    into a tuple of its own, so the Python value mirrors the annotation.  A
    zero-sized result is ``None``.

    A by-value aggregate result would need the Python-entry thunk's trailing
    out pointer, which this path does not build; like passing an aggregate
    from Python, that is not supported yet."""
    by_value = sval.ret_returned_type(ret_spec)
    if by_value is not None:
        mir_ret = by_value.to_mir_type(mir_lower_cache)
        if isinstance(mir_ret, (mir.StructType, mir.ArrayType)):
            raise SpyError(
                'cannot call a function that returns several values from Python '
                'when one result is a by-value aggregate yet'
            )
    buffers: list[Any] = []
    call_args: list[Any] = list(py_args)
    for leaf in sval.iter_ret_leaves(ret_spec):
        if not leaf.via_result_ptr:
            continue
        mir_type = leaf.type.to_mir_type(mir_lower_cache)
        assert mir_type is not None and not leaf.type.is_zst()
        buffer = to_ctype(mir_type)()
        buffers.append(buffer)
        call_args.append(ctypes.c_void_p(ctypes.addressof(buffer)))
    result = native_fn.call(*call_args)
    pending = iter(buffers)

    def leaf_value(leaf: sval.RetValue) -> Any:
        if leaf.type.get_unit_value() is not None:
            return None
        if leaf.via_result_ptr:
            buffer = next(pending)
            # a scalar buffer reads back through ``.value``; an aggregate one
            # stays the ctypes object (Python-side struct values are not
            # supported yet)
            return getattr(buffer, 'value', buffer)
        return result

    # regroup the leaves into the (possibly nested) tuple of the annotation
    assert isinstance(ret_spec, sval.RetTuple)
    groups: list[list[Any]] = [[]]
    work: list[sval.RetSpec | None] = list(reversed(ret_spec.values))
    while work:
        node = work.pop()
        if node is None:
            nested = tuple(groups.pop())
            groups[-1].append(nested)
            continue
        match node:
            case sval.RetValue():
                groups[-1].append(leaf_value(node))
            case sval.RetTuple(values=values):
                work.append(None)
                work.extend(reversed(values))
                groups.append([])
    return tuple(groups[0])


_INT_LITERAL_BITS = 64

class _RegisteredFn:
    def __init__(self, fn, cls, meta: FnMetadata, context: _Context) -> None:
        self.fn = fn
        self.cls = cls
        self.meta = meta
        self.entry: FunctionValue | None = None
        self.context = context

        # The type parameters of an enclosing context, such as a generic
        # class: the Python type parameter object an annotation evaluates to
        # -> its spy value (see ``astgen.parse_function``)
        self.context_type_vars: dict[TypeVar, sval.Value] = {}

    def with_context(self, context: _Context) -> _RegisteredFn:
        # the handle of the same Python function in ``context``: its compiled
        # artifacts (and its native symbols) belong to that context, so a
        # handle resolved from another context must be re-bound there (see
        # ``_Context.resolve_global``).  The enclosing context's type
        # parameters carry over, so that a method still resolves the struct's
        # annotations in the scope it was written in.
        clone = _RegisteredFn(self.fn, self.cls, self.meta, context)
        clone.context_type_vars = dict(self.context_type_vars)
        return clone

    def __call__(self, *args, **kwds):
        entry = self.get_entry()
        arglist = entry.hir.signature.bind_arg_pos(
            RawArgList(
                tuple(sval.as_value(a, self.context) for a in args),
                frozendict((k, sval.as_value(v, self.context)) for k, v in kwds.items()),
            ),
            lambda e: e,
        )
        arg_types: ArgList[sval.Type | None] = arglist.map(lambda a: sval.type_of(a, _INT_LITERAL_BITS))
        call_sig, ret_sig = entry.hir.signature.specialize(
            arg_types, self.context.mir_lower_cache,
        )

        analyser = Analyser(self.context, self.context.mir_lower_cache)
        analyser.analyse_function(entry, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self.context._symbol_table, self.context.backend)

        instance = entry.specs[call_sig]
        # a function whose value form ctypes cannot call directly has a
        # Python-entry thunk (see ``fn._fn_thunk``); call that instead
        native_fn = instance.wrapper_fn or instance.native_fn
        assert native_fn is not None
        ret_sig = instance.ret_sig
        assert ret_sig is not None
        if len(ret_sig.exceptions) > 0:
            raise SpyError(
                'calling a function that may raise from Python is not supported yet'
            )

        # the native call takes the arguments of the *lowered* signature:
        # a zero-sized (compile-time) parameter is not passed
        py_args = [
            _to_py_arg(value)
            for (_, sig_arg), value in zip(call_sig.positional, arglist.positional)
            if not isinstance(sig_arg, SpecializedComptimeArg)
        ]
        if ret_sig.is_single_value():
            return native_fn.call(*py_args)
        # no exception part here (a raising function is rejected above): the
        # values alone, which never include the zero-sized empty error union
        value_spec = ret_sig.ret_type_spec
        assert value_spec is not None
        return _call_multi_value(native_fn, py_args, value_spec, self.context.mir_lower_cache)

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(
                self.fn, self.context, self.cls,
                self.meta.sfv, self.context_type_vars, self.meta.exceptions,
                self.meta.callconv, self.meta.may_panic,
            )
            self.entry = FunctionValue(self.fn.__qualname__, hir, force_inline=self.meta.inline)
        return self.entry

class _RegisteredClass(StructDecl):
    """One class decorated with ``@struct()``, bound to its name in place of
    the class itself: the declaration of one spy struct.  The struct type the
    class names is built from the class body - the annotated class attributes
    are the fields, in declaration order, and the functions of the class body
    are its methods - and is cached on the handle (``get_entry``)."""

    def __init__(self, cls: type, context: _Context, meta: StructMetadata) -> None:
        self.cls = cls
        self.context = context
        self.meta = meta
        self.entry: sval.StructTypeHead | None = None
        # the declared generic type parameters of the class, by the Python
        # type parameter object their annotations evaluate to (see
        # ``get_entry``)
        self.class_type_vars: dict[TypeVar, sval.Value] = {}

    def get_entry(self) -> sval.StructTypeHead:
        if self.entry is None:
            if '__init__' in self.cls.__dict__:
                raise CompileError(
                    f'struct {self.cls.__name__} cannot declare __init__: custom '
                    f'constructors are not supported; a construction initializes '
                    f'the fields directly (a keyword argument names one)'
                )
            # the declared generic type parameters (PEP 695 ``[T]``): one spy
            # ``sval.TypeVar`` per parameter, which the annotations of the
            # class and of its methods name
            generic_args: list[sval.TypeVar] = []
            # the declared default of every type parameter (``[C: bool =
            # Literal[False]]``): a use that leaves the trailing arguments out
            # takes them (see ``sval.StructTypeHead.specialize``)
            generic_defaults: list[sval.AnyValue | None] = []
            for type_param in getattr(self.cls, '__type_params__', ()):
                if not isinstance(type_param, TypeVar):
                    raise CompileError(
                        f'unsupported type parameter {type_param!r} of struct '
                        f'{self.cls.__name__}'
                    )
                spy_type = sval.TypeVar(type_param.__name__)
                generic_args.append(spy_type)
                self.class_type_vars[type_param] = spy_type
                declared_default = getattr(type_param, '__default__', NoDefault)
                generic_defaults.append(
                    None if declared_default is NoDefault
                    else sval.as_value(declared_default, self.context)
                )
            head = sval.StructTypeHead(
                self.cls.__name__,
                tuple(generic_args),
                modifiers=sval.StructModifiers(extern_c=self.meta.extern_c),
                generic_defaults=tuple(generic_defaults),
            )
            # the head is bound before the class body is read: a field may
            # name the struct itself, or one of its methods ``self``
            self.entry = head
            # the annotations are evaluated lazily by Python, in the
            # annotation scope of the class: a subscripted struct template in
            # one of them (``inner: Pair[T]``) evaluates to an application
            # that resolves against the class' type parameters here
            for name, annotation in self.cls.__annotations__.items():
                type = sval.as_value(annotation, self.context, self.class_type_vars)
                if type is None or not isinstance(type, sval.Type):
                    raise CompileError(f'cannot convert annotation {annotation!r} to a value')
                # the value the class body assigned to the annotated attribute
                # is the field's default, which a construction leaves it at when
                # it is not given (``None`` - no value written - is the null
                # value, so it marks a default like any other; a field with no
                # default has no entry in the class body at all)
                default: sval.AnyValue | None = None
                if name in self.cls.__dict__:
                    default = sval.as_value(
                        self.cls.__dict__[name], self.context, self.class_type_vars,
                    )
                head.add_field(name, type, default)
            # the ``self`` of every method is the struct *template*: a
            # specialization whose type arguments are the struct's own
            # parameters, so that a call substitutes the arguments of the
            # specialization the method was resolved on into it (see
            # ``interp``)
            template = head.specialize(tuple(generic_args))
            for name, value in self.cls.__dict__.items():
                if isinstance(value, _RegisteredFn):
                    # a registered method: ``self`` is the struct it belongs
                    # to (the method is parsed with that type, see astgen).  A
                    # handle registered in another context is re-bound here,
                    # so that the method is parsed and compiled with this
                    # context's struct type and type parameters
                    if value.context is not self.context:
                        value = value.with_context(self.context)
                    value.cls = template
                    value.context_type_vars = self.class_type_vars
                    head.methods[name] = value
                elif isinstance(value, pytypes.FunctionType):
                    # an undecorated method: inlined at its call sites like a
                    # plain function.  It is wrapped like a registered one so
                    # that it is parsed lazily with the struct as its ``self``
                    # type and the struct's type parameters in scope
                    method = _RegisteredFn(value, template, _INLINE_META, self.context)
                    method.context_type_vars = self.class_type_vars
                    head.methods[name] = method
        return self.entry

    def __getitem__(self, key: Any) -> sval.StructTypeApplication:
        """``Foo[i32]``: an application of the struct template to generic
        arguments.  Python evaluates an annotation lazily, in the annotation
        scope of the annotated function or class, so this is what a
        subscripted struct *annotation* evaluates to; the arguments name the
        type parameters of that scope, which ``__getitem__`` does not see -
        ``sval.as_value`` turns the application into the struct
        specialization once it is given the scope (see
        :class:`sval.StructTypeApplication`).  A ``Foo[i32]`` used as an
        expression inside a body is the HIR's ``hir.Subscript`` instead,
        resolved by the interpreter.

        A future *class-name method access* (``Foo[i32].m(x)``) resolves the
        same specialization through this path (see ``interp``)."""
        args = key if isinstance(key, tuple) else (key,)
        return sval.StructTypeApplication(self, args)

    def as_spy_value(self) -> sval.AnyValue:
        """The spy value of this class: the struct type it declares, built in
        the context that resolves it (see ``_Context.resolve_global``).  The
        name of a *generic* struct stands for its template, which a call has to
        specialize."""
        entry = self.get_entry()
        if len(entry.generic_args) == 0:
            return entry.specialize(())
        return entry

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise SpyError(
            f'struct {self.cls.__name__} cannot be constructed from Python yet: '
            'it is a compile-time type only'
        )


class _FuncTypeDecl:
    """One class decorated with ``@func_type()``: the declaration of a spy
    function *type*.

    Like a ``@struct()`` class, the decorated name is a handle the context it
    is resolved in turns into an actual spy value - here a
    ``sval.FunctionType`` built from the annotations of the class' ``__call__``
    (the receiver is dropped).  The declaration is resolved lazily because its
    annotations name objects of the resolving context (a ``std`` struct, whose
    type is the context's own copy).

    A function type is dynamically sized: a parameter may use it directly (a
    DST argument is passed by reference), while a local or a struct field has
    to name a pointer to it (``ConstPtr[...]``).
    """

    def __init__(self, cls: type, meta: FuncTypeMetadata, context: _Context) -> None:
        self.cls = cls
        self.meta = meta
        self.context = context
        self.entry: sval.FunctionType | None = None

    def with_context(self, context: _Context) -> _FuncTypeDecl:
        # this context's copy of the declaration (see ``_Context._local_func_type``)
        return _FuncTypeDecl(self.cls, self.meta, context)

    def as_spy_value(self) -> sval.FunctionType:
        """The function type this declaration names, built in this handle's
        context."""
        if self.entry is not None:
            return self.entry
        call = self.cls.__call__
        code = getattr(call, '__code__', None)
        if code is None:
            raise CompileError(
                f'function type {self.cls.__name__} must declare a Python __call__'
            )
        count = code.co_argcount
        names = code.co_varnames[:count]
        annotations = call.__annotations__
        defaults = call.__defaults__ if call.__defaults__ is not None else ()
        offset = count - len(defaults)
        args: list[sval.FormalArg] = []
        for i, name in enumerate(names):
            if i == 0:
                # the receiver of the protocol method is not a parameter of the
                # function type
                continue
            if name not in annotations:
                raise CompileError(
                    f'every parameter of function type {self.cls.__name__} must be annotated'
                )
            arg_type = sval.as_value(annotations[name], self.context)
            if not isinstance(arg_type, sval.Type):
                raise CompileError(
                    f'cannot use {annotations[name]!r} as the type of parameter '
                    f'{name!r} of function type {self.cls.__name__}'
                )
            default: sval.AnyValue | None = None
            if i >= offset:
                default = sval.as_value(defaults[i - offset], self.context)
            args.append(sval.FormalArg(name, arg_type, default))
        if 'return' not in annotations or annotations['return'] is None:
            ret: sval.AnyValue = sval.VoidType()
        else:
            ret = sval.as_value(annotations['return'], self.context)
        if not isinstance(ret, sval.Type):
            raise CompileError(
                f'cannot use {annotations.get("return")!r} as the return type of '
                f'function type {self.cls.__name__}'
            )
        if self.meta.exceptions == 'infer':
            raise CompileError(
                f'the exceptions of function type {self.cls.__name__} cannot be inferred'
            )
        exceptions: list[sval.Type] = []
        if self.meta.exceptions is not None:
            for exception in self.meta.exceptions:
                value = sval.as_value(exception, self.context)
                if not isinstance(value, sval.StructType):
                    raise CompileError(
                        f'cannot use {exception!r} as an exception of function type '
                        f'{self.cls.__name__}: an exception must be a spy struct'
                    )
                exceptions.append(value)
        if self.meta.callconv != 'default' and len(exceptions) > 0:
            raise CompileError(
                f'a non-default-callconv function type may not declare exceptions: '
                f'{self.cls.__name__}'
            )
        self.entry = sval.FunctionType(
            tuple(args), ret, FrozenArraySet(exceptions), self.meta.callconv, self.meta.may_panic,
        )
        return self.entry

    def __or__(self, other: Any) -> sval.TaggedUnionApplication:
        return sval.union_application(self, other)

    def __ror__(self, other: Any) -> sval.TaggedUnionApplication:
        return sval.union_application(other, self)


class _Context(CompileContext):
    def __init__(self, backend: Backend, target: TargetInfo | None = None) -> None:
        self.backend = backend
        # the parameters of the compile target this context compiles for (the
        # pointer size, which the layout of every type depends on)
        target_info = TargetInfo() if target is None else target
        self._fn_anotation_cache: dict[Any, _RegisteredFn] = {}
        self._cls_annotation_cache: dict[type, _RegisteredClass] = {}
        # the ``@func_type()`` declarations, by the class that carries them
        self._func_type_cache: dict[type, _FuncTypeDecl] = {}
        # the inline entries of the undecorated Python functions reached
        # from a spy body, by function object
        self._inline_cache: dict[Any, FunctionValue] = {}
        self._symbol_table = SymbolTable()
        # the MIR-mirror interning table of this context, shared by every
        # analysis it runs and bound to the target it compiles for (see
        # ``sval.MirLowerCache``)
        self.mir_lower_cache = sval.MirLowerCache(target_info)
        # the ``std`` types the type rules need, resolved lazily on the first
        # request (see ``sval.SpecialTypes``)
        self._special_types: sval.SpecialTypes | None = None

    def _local_fn(self, handle: _RegisteredFn) -> _RegisteredFn:
        # this context's handle of the Python function ``handle`` names: a
        # context registers one handle per Python function, so a handle that
        # belongs to another context is re-bound here and cached, so that the
        # function is compiled (and its native symbols named) in this context
        existing = self._fn_anotation_cache.get(handle.fn)
        if existing is None:
            existing = handle.with_context(self)
            self._fn_anotation_cache[handle.fn] = existing
        return existing

    def _local_class(self, handle: _RegisteredClass) -> _RegisteredClass:
        # likewise for a struct class: the struct type this context declares
        # for it, with this context's own method handles (see
        # ``_RegisteredClass.get_entry``)
        existing = self._cls_annotation_cache.get(handle.cls)
        if existing is None:
            existing = _RegisteredClass(handle.cls, self, handle.meta)
            self._cls_annotation_cache[handle.cls] = existing
        return existing

    def _local_func_type(self, handle: _FuncTypeDecl) -> _FuncTypeDecl:
        # likewise for a function-type declaration: its annotations resolve in
        # this context, so that a ``std`` struct it names is this context's copy
        existing = self._func_type_cache.get(handle.cls)
        if existing is None:
            existing = handle.with_context(self)
            self._func_type_cache[handle.cls] = existing
        return existing

    @override
    def resolve_global(self, value: Any) -> AnyValue | None:
        match value:
            case _RegisteredFn():
                # a handle of another context is re-bound here, so that a spy
                # body only ever reaches this context's functions and structs
                handle = value if value.context is self else self._local_fn(value)
                return handle.get_entry()
            case _RegisteredClass():
                handle = value if value.context is self else self._local_class(value)
                return handle.as_spy_value()
            case _FuncTypeDecl():
                handle = value if value.context is self else self._local_func_type(value)
                return handle.as_spy_value()
            case pytypes.FunctionType():
                builtin = _BUILTINS.get(value)
                if builtin is not None:
                    return sval.BuiltinFn(builtin)
                # an undecorated plain Python function: it is inlined where
                # it is called (it contributes no native specialization)
                entry = self._inline_cache.get(value)
                if entry is None:
                    hir = astgen.parse_function(value, self)
                    entry = FunctionValue(value.__qualname__, hir, force_inline=True)
                    self._inline_cache[value] = entry
                return entry
            case _:
                # any other object stays a plain compile-time Python value
                return None

    @override
    def target_info(self) -> TargetInfo:
        return self.mir_lower_cache.target

    def _struct_head(self, handle: Any) -> sval.StructTypeHead:
        # the head this context declares for a ``std`` struct: the handle is
        # re-bound to this context (see ``resolve_global``), so the struct type
        # rules reach is this context's own
        head = self.resolve_global(handle)
        assert isinstance(head, sval.StructTypeHead)
        return head

    @override
    def special_types(self) -> sval.SpecialTypes:
        if self._special_types is None:
            from . import std
            self._special_types = sval.SpecialTypes(
                self._struct_head(std.slice),
                self._struct_head(std.SlicePtr),
                self._struct_head(std.ConstSlicePtr),
            )
        return self._special_types

    @override
    def mir_cache(self) -> MirLowerCache:
        return self.mir_lower_cache

    def _resolve_call(
        self, fn: FunctionValue, call_sig: CallSignature, ret_sig: PartialReturnSignature
    ):
        analyser = Analyser(self, self.mir_lower_cache)
        analyser.analyse_function(fn, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self._symbol_table, self.backend)

    def func(self, sfv: bool = False, extern: bool = False, linkname: str | None = None, exceptions: type | tuple[type, ...] | Literal["infer"] | None = None, callconv: str = 'default', may_panic: bool = False):
        meta = FnMetadata(
            sfv=sfv, extern=extern, linkname=linkname,
            exceptions=_normalize_exceptions(exceptions),
            callconv=callconv, may_panic=may_panic,
        )

        def wrapper[T](fn: T) -> T:
            if fn in self._fn_anotation_cache:
                return self._fn_anotation_cache[fn]
            result = _RegisteredFn(fn, None, meta, self)
            self._fn_anotation_cache[fn] = result
            return cast(T, result)
        return wrapper

    def func_type(self, callconv: str = 'default', may_panic: bool = False, exceptions: type | tuple[type, ...] | Literal["infer"] | None = None):
        """Declare a spy function *type*: the decorated ``Protocol`` names the
        signature through its ``__call__`` (whose receiver is dropped), and
        the decorated name is the handle that resolves to a
        ``sval.FunctionType`` (see ``_FuncTypeDecl``)."""
        meta = FuncTypeMetadata(
            callconv=callconv, may_panic=may_panic,
            exceptions=_normalize_exceptions(exceptions),
        )

        def wrapper[T](cls: type[T]) -> type[T]:
            if cls in self._func_type_cache:
                return cast(type[T], self._func_type_cache[cls])
            result = _FuncTypeDecl(cls, meta, self)
            self._func_type_cache[cls] = result
            return cast(type[T], result)
        return wrapper

    @dataclass_transform()
    def struct(self, extern_c: bool = False):
        meta = StructMetadata(extern_c=extern_c)

        def wrapper[T](cls: type[T]) -> type[T]:
            if cls in self._cls_annotation_cache:
                return self._cls_annotation_cache[cls]
            result = _RegisteredClass(cls, self, meta)
            self._cls_annotation_cache[cls] = result
            return cast(type[T], result)
        return wrapper

_GLOBAL_CONTEXT = _Context(LLVMBackend())

func = _GLOBAL_CONTEXT.func
struct = _GLOBAL_CONTEXT.struct
func_type = _GLOBAL_CONTEXT.func_type

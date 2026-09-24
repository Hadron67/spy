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
from typing import Any, TypeVar, cast, dataclass_transform, override

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
    RawArgList,
    ReturnSignature,
    SpecializedComptimeArg,
    SymbolTable,
)
from .interp import Analyser
from .lower import LLVMBackend, to_ctype
from .sval import AsSpyValue, GlobalResolver, StructDecl
from .util import frozendict

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


@dataclass(frozen=True)
class StructMetadata:
    extern_c: bool


# the metadata of an undecorated method (see ``_RegisteredClass.get_entry``)
_INLINE_META = FnMetadata(sfv=False, extern=False, linkname=None, inline=True)


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
    ret_sig: ReturnSignature,
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
    by_value = ret_sig.returned_type()
    if by_value is not None:
        mir_ret = by_value.to_mir_type()
        if isinstance(mir_ret, (mir.StructType, mir.ArrayType)):
            raise SpyError(
                'cannot call a function that returns several values from Python '
                'when one result is a by-value aggregate yet'
            )
    buffers: list[Any] = []
    call_args: list[Any] = list(py_args)
    for leaf in sval.iter_ret_leaves(ret_sig.ret_spec):
        if not leaf.via_result_ptr:
            continue
        mir_type = leaf.type.to_mir_type()
        assert mir_type is not None and not isinstance(mir_type, mir.VoidType)
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
    assert isinstance(ret_sig.ret_spec, sval.RetTuple)
    groups: list[list[Any]] = [[]]
    work: list[sval.RetSpec | None] = list(reversed(ret_sig.ret_spec.values))
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

class _RegisteredFn(AsSpyValue):
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

    def __call__(self, *args, **kwds):
        entry = self.get_entry()
        arglist = entry.hir.signature.bind_arg_pos(
            RawArgList(
                tuple(sval.as_value(a) for a in args),
                frozendict((k, sval.as_value(v)) for k, v in kwds.items()),
            ),
            lambda e: e,
        )
        arg_types: ArgList[sval.Type | None] = arglist.map(lambda a: sval.type_of(a, _INT_LITERAL_BITS))
        call_sig, ret_sig = entry.hir.signature.specialize(arg_types)

        analyser = Analyser(self.context)
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

        # the native call takes the arguments of the *lowered* signature:
        # a zero-sized (compile-time) parameter is not passed
        py_args = [
            _to_py_arg(value)
            for (_, sig_arg), value in zip(call_sig.positional, arglist.positional)
            if not isinstance(sig_arg, SpecializedComptimeArg)
        ]
        if ret_sig.is_single_value():
            return native_fn.call(*py_args)
        return _call_multi_value(native_fn, py_args, ret_sig)

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(
                self.fn, self.cls, self.meta.sfv, self.context_type_vars
            )
            self.entry = FunctionValue(self.fn.__qualname__, hir, force_inline=self.meta.inline)
        return self.entry

    @override
    def as_spy_value(self) -> sval.AnyValue:
        return self.get_entry()

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
            for type_param in getattr(self.cls, '__type_params__', ()):
                if not isinstance(type_param, TypeVar):
                    raise CompileError(
                        f'unsupported type parameter {type_param!r} of struct '
                        f'{self.cls.__name__}'
                    )
                spy_type = sval.TypeVar(type_param.__name__)
                generic_args.append(spy_type)
                self.class_type_vars[type_param] = spy_type
            head = sval.StructTypeHead(
                self.cls.__name__,
                tuple(generic_args),
                modifiers=sval.StructModifiers(extern_c=self.meta.extern_c),
            )
            # the head is bound before the class body is read: a field may
            # name the struct itself, or one of its methods ``self``
            self.entry = head
            # the annotations are evaluated lazily by Python, in the
            # annotation scope of the class: a subscripted struct template in
            # one of them (``inner: Pair[T]``) evaluates to an application
            # that resolves against the class' type parameters here
            for name, annotation in self.cls.__annotations__.items():
                type = sval.as_value(annotation, self.class_type_vars, resolver=self.context)
                if type is None or not isinstance(type, sval.Type):
                    raise CompileError(f'cannot convert annotation {annotation!r} to a value')
                head.add_field(name, type)
            # the ``self`` of every method is the struct *template*: a
            # specialization whose type arguments are the struct's own
            # parameters, so that a call substitutes the arguments of the
            # specialization the method was resolved on into it (see
            # ``interp``)
            template = head.specialize(tuple(generic_args))
            for name, value in self.cls.__dict__.items():
                if isinstance(value, _RegisteredFn):
                    # a registered method: ``self`` is the struct it belongs
                    # to (the method is parsed with that type, see astgen)
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
        head = self.get_entry()
        args = key if isinstance(key, tuple) else (key,)
        return sval.StructTypeApplication(head, args)

    @override
    def as_spy_value(self) -> sval.AnyValue:
        """The spy value of this class: the struct type it declares (see
        ``sval.as_value``, which asks for it).  The name of a *generic* struct
        stands for its template, which a call has to specialize."""
        entry = self.get_entry()
        if len(entry.generic_args) == 0:
            return entry.specialize(())
        return entry

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise SpyError(
            f'struct {self.cls.__name__} cannot be constructed from Python yet: '
            'it is a compile-time type only'
        )


class _Context(GlobalResolver):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._fn_anotation_cache: dict[Any, _RegisteredFn] = {}
        self._cls_annotation_cache: dict[type, _RegisteredClass] = {}
        # the inline entries of the undecorated Python functions reached
        # from a spy body, by function object
        self._inline_cache: dict[Any, FunctionValue] = {}
        self._symbol_table = SymbolTable()

    @override
    def resolve_global(self, value: Any) -> AnyValue | None:
        match value:
            case _RegisteredFn():
                return value.get_entry()
            case _RegisteredClass():
                return value.as_spy_value()
            case pytypes.FunctionType():
                builtin = _BUILTINS.get(value)
                if builtin is not None:
                    return sval.BuiltinFn(builtin)
                # an undecorated plain Python function: it is inlined where
                # it is called (it contributes no native specialization)
                entry = self._inline_cache.get(value)
                if entry is None:
                    hir = astgen.parse_function(value)
                    entry = FunctionValue(value.__qualname__, hir, force_inline=True)
                    self._inline_cache[value] = entry
                return entry
            case _:
                # any other object stays a plain compile-time Python value
                return None

    def _resolve_call(
        self, fn: FunctionValue, call_sig: CallSignature, ret_sig: ReturnSignature | None
    ):
        analyser = Analyser(self)
        analyser.analyse_function(fn, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self._symbol_table, self.backend)

    def func(self, sfv: bool = False, extern: bool = False, linkname: str | None = None):
        meta = FnMetadata(sfv=sfv, extern=extern, linkname=linkname)

        def wrapper[T](fn: T) -> T:
            if fn in self._fn_anotation_cache:
                return self._fn_anotation_cache[fn]
            result = _RegisteredFn(fn, None, meta, self)
            self._fn_anotation_cache[fn] = result
            return cast(T, result)
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

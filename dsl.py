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
annotates with it, constructs it (``Foo(a, b)`` - the struct's ``__init__``
runs if it has one, and its fields are filled otherwise) and calls its
methods on a value of it (``x.m()``, the object passed as the method's
``self``).  A struct is a compile-time type only: Python-side construction is
not supported yet, and neither are generic structs (the fields of a generic
struct are declared on its :class:`~spy.sval.StructTypeHead`, which a call has
to specialize first).

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see ``interp``);
calling it is compiled to a native ``call``.  An *undecorated* plain
Python function reached the same way is inlined instead.  The ``spy.*``
builtins (``spy.typeof``, ``spy.compile_log``) are evaluated at compile
time.
"""

import types as pytypes
from dataclasses import dataclass
from typing import Any, cast, dataclass_transform, override

from . import astgen, sval
from .builtins import spy_as, spy_compile_log, spy_typeof
from .errors import CompileError, SpyError
from .fn import (
    AnyValue,
    ArgList,
    Backend,
    CallSignature,
    FunctionResolver,
    FunctionValue,
    RawArgList,
    ReturnSignature,
    SpecializedComptimeArg,
    SymbolTable,
)
from .interp import Analyser
from .lower import LLVMBackend
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


@dataclass(frozen=True)
class StructMetadata:
    extern_c: bool


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
        case _:
            return value

_INT_LITERAL_BITS = 64

class _RegisteredFn:
    def __init__(self, fn, cls, meta: FnMetadata, context: _Context) -> None:
        self.fn = fn
        self.cls = cls
        self.meta = meta
        self.entry: FunctionValue | None = None
        self.context = context

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

        # the native call takes the arguments of the *lowered* signature:
        # a zero-sized (compile-time) parameter is not passed
        py_args = [
            _to_py_arg(value)
            for (_, sig_arg), value in zip(call_sig.positional, arglist.positional)
            if not isinstance(sig_arg, SpecializedComptimeArg)
        ]
        return native_fn.call(*py_args)

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(self.fn, self.cls, self.meta.sfv)
            self.entry = FunctionValue(self.fn.__qualname__, hir)
        return self.entry

class _RegisteredClass:
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

    def get_entry(self) -> sval.StructTypeHead:
        if self.entry is None:
            head = sval.StructTypeHead(
                self.cls.__name__,
                modifiers=sval.StructModifiers(extern_c=self.meta.extern_c),
            )
            # the head is bound before the class body is read: a field may
            # name the struct itself, or one of its methods ``self``
            self.entry = head
            for name, annotation in self.cls.__annotations__.items():
                head.add_field(name, self._spy_type(annotation, f"the field '{name}'"))
            struct = head.specialize(())
            for name, value in self.cls.__dict__.items():
                if isinstance(value, _RegisteredFn):
                    # a registered method: ``self`` is the struct it belongs
                    # to (the method is parsed with that type, see astgen)
                    value.cls = struct
                    head.methods[name] = value
                elif isinstance(value, pytypes.FunctionType):
                    # an undecorated method: a plain Python function, inlined
                    # at its call sites like any other
                    head.methods[name] = value
        return self.entry

    def as_spy_value(self) -> sval.AnyValue:
        """The spy value of this class: the struct type it declares (see
        ``sval.as_value``, which asks for it).  The name of a *generic* struct
        stands for its template, which a call has to specialize."""
        entry = self.get_entry()
        if len(entry.generic_args) == 0:
            return entry.specialize(())
        return entry

    def _spy_type(self, annotation: Any, what: str) -> sval.Type:
        """The spy type one field annotation of the class denotes: another
        struct class resolves to the struct it declares, anything else through
        the ordinary conversion (``sval.as_value``)."""
        type: sval.AnyValue | None = self.context.resolve_global(annotation)
        if type is None:
            try:
                type = sval.as_value(annotation)
            except Exception as e:
                raise CompileError(
                    f'cannot use {annotation!r} as {what} of struct {self.cls.__name__}: {e}'
                ) from e
        if not isinstance(type, sval.Type):
            raise CompileError(
                f'cannot use {annotation!r} as {what} of struct {self.cls.__name__}: '
                'a field needs a type'
            )
        return type

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise SpyError(
            f'struct {self.cls.__name__} cannot be constructed from Python yet: '
            'it is a compile-time type only'
        )


class _Context(FunctionResolver):
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

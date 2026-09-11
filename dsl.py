"""The user-facing DSL: the ``func`` decorator.

A decorated function is *registered* in its host context - the
registration binds a callable handle (``_RegisteredFn``) to the decorated
name - and parsed (astgen) only when it is first used.  A Python-side call
goes through the handle, which at call time

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. solves the concrete spy types of the parameters from the marshaled
   types of the arguments plus type-parameter unification,
3. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see ``interp``);
calling it is compiled to a native ``call``.  An *undecorated* plain
Python function reached the same way is inlined instead.  The ``spy.*``
builtins (``spy.typeof``, ``spy.compile_log``) are evaluated at compile
time.
"""

import types as pytypes
from dataclasses import dataclass
from typing import Any, cast, override

from . import astgen, sval
from .builtins import spy_as, spy_compile_log, spy_typeof
from .fn import (
    AnyValue,
    ArgList,
    Backend,
    FunctionResolver,
    FunctionValue,
    RawArgList,
    ReturnSignature,
    SpecializedCallSignature,
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
    repr: str | None


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
        native_fn = instance.native_fn
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
            hir = astgen.parse_function(self.fn, self.cls)
            self.entry = FunctionValue(self.fn.__qualname__, hir)
        return self.entry


class _Context(FunctionResolver):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._anotation_cache: dict[Any, _RegisteredFn] = {}
        # the inline entries of the undecorated Python functions reached
        # from a spy body, by function object
        self._inline_cache: dict[Any, FunctionValue] = {}
        self._symbol_table = SymbolTable()

    @override
    def resolve_global(self, value: Any) -> AnyValue | None:
        match value:
            case _RegisteredFn():
                return value.get_entry()
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
        self, fn: FunctionValue, call_sig: SpecializedCallSignature, ret_sig: ReturnSignature | None
    ):
        analyser = Analyser(self)
        analyser.analyse_function(fn, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self._symbol_table, self.backend)

    def func(self, sfv: bool = False, extern: bool = False, linkname: str | None = None):
        meta = FnMetadata(sfv=sfv, extern=extern, linkname=linkname)

        def wrapper[T](fn: T) -> T:
            if fn in self._anotation_cache:
                return self._anotation_cache[fn]
            result = _RegisteredFn(fn, None, meta, self)
            self._anotation_cache[fn] = result
            return cast(T, result)
        return wrapper


_GLOBAL_CONTEXT = _Context(LLVMBackend())


def func(*, sfv: bool = False, extern: bool = False, linkname: str | None = None):
    def decorator[T](fn: T) -> T:
        return cast(T, _GLOBAL_CONTEXT.func(sfv=sfv, extern=extern, linkname=linkname)(fn))
    return decorator

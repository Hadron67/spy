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

import types as pytypes
from dataclasses import dataclass
from typing import cast, override

from spy.lower import LLVMBackend

from . import astgen, mir, sval
from .fn import (
    Any,
    AnyValue,
    ArgList,
    Backend,
    FunctionResolver,
    FunctionValue,
    RawArgList,
    ReturnSignature,
    SpecializedCallSignature,
)
from .interp import Analyser
from .util import frozendict


@dataclass(frozen=True)
class FnMetadata:
    sfv: bool # self by value
    extern: bool
    linkname: str | None

@dataclass(frozen=True)
class StructMetadata:
    repr: str | None

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
            RawArgList(tuple(sval.as_value(a) for a in args), frozendict((k, sval.as_value(v)) for k, v in kwds.items())),
            lambda e: e,
        )
        arg_types: ArgList[sval.Type | None] = arglist.map(lambda a: sval.type_of(a))
        call_sig, ret_sig = entry.hir.signature.specialize(arg_types)

        analyser = Analyser(self.context)
        analyser.analyse_function(entry, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self.context.backend)

        assert call_sig in entry.specs
        instance = entry.specs[call_sig]
        assert instance.native_fn is not None and instance.ret_sig is not None
        # TODO

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(self.fn, self.cls)
            self.entry = FunctionValue(self.fn.__qualname__, hir)
        return self.entry

class _Context(FunctionResolver):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._anotation_cache: dict[Any, _RegisteredFn] = {}

    @override
    def resolve_global(self, value: Any) -> AnyValue:
        match value:
            case _RegisteredFn():
                return value.get_entry()
            case pytypes.FunctionType():
                return cast(AnyValue, self.func()(value))
            case _:
                raise ValueError(f"Unexpected global value: {value}")

    def _resolve_call(self, fn: FunctionValue, call_sig: SpecializedCallSignature, ret_sig: ReturnSignature | None):
        analyser = Analyser(self)
        analyser.analyse_function(fn, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self.backend)

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

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

from dataclasses import dataclass
from typing import cast

from . import astgen, sval
from .fn import FunctionValue, RawArgList
from .util import frozendict


@dataclass(frozen=True)
class FnMetadata:
    sfv: bool # self by value
    extern: bool
    linkname: str | None

@dataclass(frozen=True)
class StructMetadata:
    repr: str | None

class RegisteredFn:
    def __init__(self, fn, cls, meta: FnMetadata) -> None:
        self.fn = fn
        self.cls = cls
        self.meta = meta
        self.entry: FunctionValue | None = None

    def __call__(self, *args, **kwds):
        entry = self.get_entry()
        arglist = entry.hir.signature.bind_arg_pos(
            RawArgList(tuple(sval.as_value(a) for a in args), frozendict((k, sval.as_value(v)) for k, v in kwds.items())),
            lambda e: e,
        )

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(self.fn, self.cls)
            self.entry = FunctionValue(self.fn.__qualname__, hir)
        return self.entry

class _Context:
    def __init__(self) -> None:
        pass
    def func(*, sfv: bool = False, extern: bool = False, linkname: str | None = None):
        meta = FnMetadata(sfv=sfv, extern=extern, linkname=linkname)
        def wrapper[T](fn: T) -> T:
            return cast(T, RegisteredFn(fn, None, meta))
        return wrapper

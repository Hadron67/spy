"""The ``spy.*`` builtins that appear inside function bodies.

At the Python level these are ordinary functions; the compile-time
interpreter recognizes them by object identity and evaluates them while
running the HIR (``spy.compile_log``).  Calling them from plain Python
raises an error: ``spy.compile_log`` only makes sense during compilation,
and ``spy.as`` is meant to build typed arguments at the call boundary (it
coerces the value to a typed spy constant the marshal layer understands).
"""

from typing import Any, cast

from .errors import SpyError
from .sval import PointerType, SpecialTypeKind, Type, coerce_const


def spy_as[T](value: T, type: type[T]) -> T:
    if not isinstance(type, Type):
        raise TypeError(f'spy.as requires a spy type, got {type!r}')
    if isinstance(type, PointerType) or type.classify() == SpecialTypeKind.DST:
        # a pointer (or a dynamically-sized function value): the value is the
        # raw address it carries
        from . import glue
        return cast(T, glue.pointer_value(value, type))
    return cast(T, coerce_const(cast(Any, value), type))


def spy_compile_log(*args: Any, **kwargs: Any) -> None:
    raise SpyError(
        'spy.compile_log may only be called from inside a spy function, '
        'where it prints at compile time'
    )

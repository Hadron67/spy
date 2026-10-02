"""The ``spy.*`` builtins that appear inside function bodies.

At the Python level these are ordinary functions; the compile-time
interpreter recognizes them by object identity and evaluates them while
running the HIR (``spy.typeof``, ``spy.compile_log``).  Calling them from
plain Python raises an error: ``spy.typeof`` and ``spy.compile_log`` only
make sense during compilation, and ``spy.as`` is meant to build typed
arguments at the call boundary (it returns an :class:`AsValue` that the
marshal layer understands).
"""

from typing import Any, cast

from .errors import SpyError
from .sval import AsValue, Type


def spy_typeof(value: Any) -> None:
    raise SpyError(
        'spy.typeof may only be called from inside a spy function, '
        'where it is evaluated at compile time'
    )


def spy_as[T](value: T, type: type[T]) -> T:
    if not isinstance(type, Type):
        raise TypeError(f'spy.as requires a spy type, got {type!r}')
    return cast(T, AsValue(value, type))


def spy_compile_log(*args: Any) -> None:
    raise SpyError(
        'spy.compile_log may only be called from inside a spy function, '
        'where it prints at compile time'
    )

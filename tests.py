"""Integration tests for the spy JIT (``symlat.spy``).

The functions under test are defined inside :func:`make_samples`, which
registers them into a :class:`JitContext` the way a user would, with
the ordinary ``@cache.jit()``/``@cache.aot()`` decorators.  Bodies that
call other spy functions (``foo`` calling ``add_aot``/``add_inline``,
``is_even`` and ``is_odd`` calling each other, ...) resolve those names
through the enclosing factory scope, exactly like closures in user
code.  Every test builds a fresh context in ``setUp``, so no module
globals are touched.
"""

import io
from contextlib import redirect_stdout
from typing import Any, Protocol, Self
from unittest import TestCase

from spy.dsl import func

from . import (
    CompileError,
    TypeMismatchError,
    astgen,
    compile_log,
    f64,
    hir,
    i32,
    mir,
    sval,
    u32,
    u64,
    void,
)
from . import as_ as spy_as
from . import bool as spy_bool
from . import typeof as spy_typeof

# ---------------------------------------------------------------------------
# functions under test (the example of spy/instructions.md)
# ---------------------------------------------------------------------------

class Numeric(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mod__(self, other: Self, /) -> Self: ...
    def __lt__(self, other: Self, /) -> spy_bool: ...
    def __gt__(self, other: Self, /) -> spy_bool: ...
    def __le__(self, other: Self, /) -> spy_bool: ...
    def __ge__(self, other: Self, /) -> spy_bool: ...

@func()
def smoke_test(a: i32, b: i32) -> i32:
    return a + b


all_tests = []

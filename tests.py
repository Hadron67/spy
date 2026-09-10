"""Integration tests for the spy JIT (``symlat.spy``).

The functions under test are defined at module level and registered with
the ordinary ``@spy.func()`` decorator; a function body may call the
other registered functions by name (they are module globals, resolved by
the compile-time interpreter) exactly like a user would.  The
undecorated ``add_inline`` is deliberately left unregistered: it stays a
plain Python function and is inlined at its call sites.

Only function calls are exercised here (the struct features are not).
The global host context caches specializations, so the tests share the
compiled functions; a test that needs a fresh compilation calls a
function no earlier test has compiled.
"""

import io
from contextlib import redirect_stdout
from typing import Protocol, Self
from unittest import TestCase

from spy.dsl import func

from . import (
    CompileError,
    compile_log,
    f32,
    f64,
    i32,
    u64,
)
from . import as_ as spy_as
from . import bool as spy_bool
from . import typeof as spy_typeof

# ---------------------------------------------------------------------------
# functions under test
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


@func()
def add[T: Numeric](a: T, b: T) -> T:
    return a + b


@func()
def add_u64(a: u64, b: u64) -> u64:
    return a + b


@func()
def sub(a: i32, b: i32) -> i32:
    return a - b


@func()
def mul(a: i32, b: i32) -> i32:
    return a * b


@func()
def mod(a: i32, b: i32) -> i32:
    return a % b


@func()
def scale(a: f64, b: f64) -> f64:
    return a * b + 1.0


@func()
def add_default[T: Numeric](a: T, b: T = 0) -> T:
    return a + b


def add_inline[T: Numeric](a: T, b: T) -> T:
    compile_log("add_inline was compiled")
    return a + b


def abs_inline(n: i32) -> i32:
    if n < 0:
        return -n
    else:
        return n


def clamp_inline(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def call_abs(n: i32) -> i32:
    return abs_inline(n)


@func()
def call_clamp(n: i32) -> i32:
    return clamp_inline(n)


@func()
def call_abs_plus_one(n: i32) -> i32:
    return abs_inline(n) + 1


@func()
def call_add[T: Numeric](a: T, b: T) -> T:
    return add(a, b)


@func()
def call_inline[T: Numeric](a: T, b: T) -> T:
    return add_inline(a, b)


@func()
def call_inline_log(a: i32, b: i32) -> i32:
    return add_inline(a, b)


@func()
def use_default[T: Numeric](a: T) -> T:
    return add_default(a)


@func()
def accumulate(a: i32, b: i32) -> i32:
    a += b
    return a


@func()
def sign(n: i32) -> i32:
    if n > 0:
        return 1
    else:
        return -1


@func()
def clamped(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def le(a: i32, b: i32) -> spy_bool:
    return a <= b


@func()
def eq(a: i32, b: i32) -> spy_bool:
    return a == b


@func()
def is_i32(a) -> spy_bool:
    return spy_typeof(a) == i32


@func()
def is_u64(a) -> spy_bool:
    return spy_typeof(a) == u64


@func()
def nothing(a: i32) -> None:
    pass


@func()
def fact(n: i32) -> i32:
    if n <= 1:
        return 1
    return n * fact(n - 1)


@func()
def is_even(n: i32) -> spy_bool:
    if n == 0:
        return True
    return is_odd(n - 1)


@func()
def is_odd(n: i32) -> spy_bool:
    if n == 0:
        return False
    return is_even(n - 1)


@func()
def inc(a: i32) -> i32:
    return a + 1


@func()
def twice_inc(a: i32) -> i32:
    return inc(inc(a))


@func()
def id_f32(a: f32) -> f32:
    return a


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SpyFunctionCallTest(TestCase):
    """Basic function calls: arithmetic, generics, calls between spy
    functions, inlining of plain functions, compile-time dispatch and
    recursion."""

    def test_smoke(self) -> None:
        self.assertEqual(smoke_test(2, 3), 5)
        self.assertEqual(smoke_test(-4, 1), -3)

    def test_arithmetic(self) -> None:
        self.assertEqual(sub(7, 12), -5)
        self.assertEqual(mul(6, 7), 42)
        self.assertEqual(mod(17, 5), 2)

    def test_generic_int(self) -> None:
        # a plain Python int marshals to the default signed 32-bit type
        self.assertEqual(add(2, 3), 5)

    def test_generic_float(self) -> None:
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)

    def test_float_promotion(self) -> None:
        self.assertAlmostEqual(scale(3.0, 0.5), 2.5)

    def test_non_default_integer_type(self) -> None:
        # ``spy.as_`` binds an argument to an explicit spy type
        self.assertEqual(add_u64(spy_as(2**63 - 1, u64), spy_as(2, u64)), 2**63 + 1)

    def test_default_argument(self) -> None:
        self.assertEqual(add_default(41), 41)
        self.assertEqual(add_default(40, 2), 42)

    def test_call_spy_function(self) -> None:
        self.assertEqual(call_add(20, 22), 42)

    def test_inline_plain_function(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline(20, 22), 42)

    def test_inline_with_runtime_branches(self) -> None:
        # the inlined body has a runtime ``if`` whose branches both return
        # (the ``Block``/``Break`` shape of an inlined return)
        self.assertEqual(call_abs(-5), 5)
        self.assertEqual(call_abs(5), 5)
        # a branch returns, the other falls through to the trailing return
        self.assertEqual(call_clamp(42), 42)
        self.assertEqual(call_clamp(101), 100)
        # the inlined result is consumed by the caller's continuation
        self.assertEqual(call_abs_plus_one(-5), 6)

    def test_call_with_default(self) -> None:
        self.assertEqual(use_default(7), 7)

    def test_augmented_assignment(self) -> None:
        self.assertEqual(accumulate(5, 6), 11)

    def test_runtime_if_both_return(self) -> None:
        self.assertEqual(sign(5), 1)
        self.assertEqual(sign(-5), -1)

    def test_runtime_if_fallthrough(self) -> None:
        self.assertEqual(clamped(42), 42)
        self.assertEqual(clamped(101), 100)

    def test_comparisons(self) -> None:
        self.assertTrue(le(1, 2))
        self.assertFalse(le(2, 1))
        self.assertTrue(eq(3, 3))
        self.assertFalse(eq(3, 4))

    def test_comptime_typeof(self) -> None:
        self.assertTrue(is_i32(1))
        self.assertFalse(is_i32(1.0))
        self.assertTrue(is_u64(spy_as(1, u64)))

    def test_void_function(self) -> None:
        self.assertIsNone(nothing(3))

    def test_recursion(self) -> None:
        self.assertEqual(fact(5), 120)
        self.assertEqual(fact(0), 1)
        self.assertEqual(fact(10), 3628800)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(is_even(10))
        self.assertFalse(is_even(7))
        self.assertTrue(is_odd(7))

    def test_cross_module_call(self) -> None:
        # ``inc`` is compiled on its own first, so the compilation of
        # ``twice_inc`` references it as an external symbol of an earlier
        # module (the modular-compilation path)
        self.assertEqual(inc(10), 11)
        self.assertEqual(twice_inc(10), 12)

    def test_f32(self) -> None:
        self.assertAlmostEqual(id_f32(spy_as(0.5, f32)), 0.5)

    def test_specialization_is_cached(self) -> None:
        # repeated calls reuse the compiled specialization
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)

    def test_compile_error_on_unsupported_expression(self) -> None:
        with self.assertRaises(CompileError):
            div(1, 2)


@func()
def div(a: i32, b: i32) -> i32:
    return a / b # pyright: ignore[reportReturnType]


class SpyCompileLogTest(TestCase):
    def test_compile_log_prints_at_compile_time(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline_log(1, 2), 3)
        self.assertIn('add_inline was compiled', out.getvalue())


all_tests = [SpyFunctionCallTest, SpyCompileLogTest]

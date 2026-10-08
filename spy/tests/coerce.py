from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i8,
    i32,
    u8,
    u32,
    u64,
)
from ..compiler.dsl import func
from ..compiler.syntax import (
    Option,
    comptime,
    ref,
)
from ..std.core import coerce

# ---------------------------------------------------------------------------
# ``std.core.coerce(value, T)``: materialize ``value`` as the spy type ``T`` -
# the same conversion a store into a location of ``T`` performs.  A
# compile-time value is converted in Python and stays compile-time; a runtime
# value gets the numeric conversion ``T`` needs (widening or narrowing)
# ---------------------------------------------------------------------------


@func()
def coerce_widen(x: u32) -> u64:
    return coerce(x, u64)


@func()
def coerce_narrow(x: i32) -> i8:
    return coerce(x, i8)


@func()
def coerce_int_to_float(x: i32) -> f64:
    return coerce(x, f64)


@func()
def coerce_comptime_fold() -> u64:
    # a compile-time value folds in Python, so the comparison is a compile-time
    # bool: the ``else`` branch (whose coerce does not type check) is dead and
    # is never typed
    comptime()
    x: f64 = 1.0
    if coerce(x, f64) == 1.0:
        return 1
    return coerce(None, u64)


@func()
def coerce_option_present(x: i32) -> Option[i32]:
    return coerce(x, Option[i32])  # pyright: ignore[reportArgumentType]


@func()
def coerce_option_absent() -> Option[i32]:
    return coerce(None, Option[i32])  # pyright: ignore[reportArgumentType]


@func()
def coerce_pointer_to_int(x: i32) -> u8:
    # an address has no numeric conversion to an integer
    p = ref(x)
    return coerce(p, u8)


class SpyCoerceTest(TestCase):
    """``std.core.coerce(value, T)`` materializes a value as the spy type ``T``,
    folding a compile-time value in Python and emitting the numeric conversion
    a runtime one needs."""

    def test_widen(self) -> None:
        self.assertEqual(coerce_widen(0xFFFFFFFF), 0xFFFFFFFF)

    def test_narrow(self) -> None:
        self.assertEqual(coerce_narrow(100), 100)
        # the high bits are dropped: 128 truncates to the i8 value -128
        self.assertEqual(coerce_narrow(128), -128)

    def test_int_to_float(self) -> None:
        self.assertEqual(coerce_int_to_float(3), 3.0)

    def test_comptime_fold(self) -> None:
        self.assertEqual(coerce_comptime_fold(), 1)

    def test_option(self) -> None:
        self.assertEqual(coerce_option_present(7), 7)
        self.assertIsNone(coerce_option_absent())

    def test_incompatible_value(self) -> None:
        with self.assertRaises(CompileError):
            coerce_pointer_to_int(1)


all_tests = [
    SpyCoerceTest,
]

from unittest import TestCase

from ..compiler import (
    CompileError,
    f32,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
)
from ..compiler import bool as spy_bool
from ..compiler.dsl import func
from ..compiler.syntax import closure, comptime
from ..std.core import UnwindException, catch_unwind
from ..std.int import IntCastError, int_cast, truncate, try_int_cast, valid_range

# ---------------------------------------------------------------------------
# ``std.int``: ``truncate`` narrows an integer to a same-signedness type with
# fewer bits; ``try_int_cast`` casts between integer types, checking the range
# only when the conversion may not fit (and raising ``IntCastError`` otherwise);
# ``int_cast`` panics instead of raising.
# ---------------------------------------------------------------------------

# -- try_int_cast: conversions that never need a range check ---------------


@func(exceptions=IntCastError)
def cast_identity(x: i32) -> i32:
    return try_int_cast(x, i32)


@func(exceptions=IntCastError)
def cast_u8_to_u16(x: u8) -> u16:
    return try_int_cast(x, u16)


@func(exceptions=IntCastError)
def cast_u8_to_i16(x: u8) -> i16:
    return try_int_cast(x, i16)


@func(exceptions=IntCastError)
def cast_i8_to_i16(x: i8) -> i16:
    return try_int_cast(x, i16)


@func(exceptions=IntCastError)
def cast_i32_to_i64(x: i32) -> i64:
    return try_int_cast(x, i64)


# -- try_int_cast: narrowing, same signedness ------------------------------


@func(exceptions=IntCastError)
def cast_u32_to_u8(x: u32) -> u8:
    return try_int_cast(x, u8)


@func(exceptions=IntCastError)
def cast_i32_to_i8(x: i32) -> i8:
    return try_int_cast(x, i8)


# -- try_int_cast: signedness change ---------------------------------------


@func(exceptions=IntCastError)
def cast_u8_to_i8(x: u8) -> i8:
    return try_int_cast(x, i8)


@func(exceptions=IntCastError)
def cast_i8_to_u8(x: i8) -> u8:
    return try_int_cast(x, u8)


@func(exceptions=IntCastError)
def cast_i32_to_u16(x: i32) -> u16:
    return try_int_cast(x, u16)


@func(exceptions=IntCastError)
def cast_i8_to_u16(x: i8) -> u16:
    return try_int_cast(x, u16)


# -- try_int_cast: range checks (does the value fit?) ----------------------


@func()
def fits_u32_u8(x: u32) -> spy_bool:
    try:
        _ = try_int_cast(x, u8)
    except IntCastError:
        return False
    return True


@func()
def fits_i32_i8(x: i32) -> spy_bool:
    try:
        _ = try_int_cast(x, i8)
    except IntCastError:
        return False
    return True


@func()
def fits_u8_i8(x: u8) -> spy_bool:
    try:
        _ = try_int_cast(x, i8)
    except IntCastError:
        return False
    return True


@func()
def fits_i8_u8(x: i8) -> spy_bool:
    try:
        _ = try_int_cast(x, u8)
    except IntCastError:
        return False
    return True


@func()
def fits_i32_u16(x: i32) -> spy_bool:
    try:
        _ = try_int_cast(x, u16)
    except IntCastError:
        return False
    return True


@func()
def fits_i8_u16(x: i8) -> spy_bool:
    try:
        _ = try_int_cast(x, u16)
    except IntCastError:
        return False
    return True


# -- truncate --------------------------------------------------------------


@func()
def trunc_u32_u8(x: u32) -> u8:
    return truncate(x, u8)


@func()
def trunc_i32_i8(x: i32) -> i8:
    return truncate(x, i8)


@func()
def trunc_fold() -> u8:
    comptime()
    x: u32 = 0x1FF
    return truncate(x, u8)


@func()
def trunc_widen(x: u8) -> u16:
    return truncate(x, u16)


@func()
def trunc_sign_mismatch(x: u32) -> i8:
    return truncate(x, i8)


@func()
def trunc_non_integer(x: f32) -> u16:
    return truncate(x, u16)


# -- valid_range -----------------------------------------------------------


@func()
def range_i8() -> spy_bool:
    comptime()
    lo, hi = valid_range(8, True)
    return lo == -128 and hi == 127


@func()
def range_u8() -> spy_bool:
    comptime()
    lo, hi = valid_range(8, False)
    return lo == 0 and hi == 255


@func()
def range_i32() -> spy_bool:
    comptime()
    lo, hi = valid_range(32, True)
    return lo == -2147483648 and hi == 2147483647


# -- int_cast --------------------------------------------------------------


@func()
def int_cast_in_range(x: u32) -> u8:
    return int_cast(x, u8)


@func()
def int_cast_panics(x: u32) -> spy_bool:
    @closure(inline=False)
    def body() -> u8:
        return int_cast(x, u8)

    try:
        catch_unwind(body)
    except UnwindException:
        return True
    return False


class SpyTryIntCastTest(TestCase):
    """``std.int.try_int_cast``."""

    def test_identity(self) -> None:
        self.assertEqual(cast_identity(42), 42)
        self.assertEqual(cast_identity(-7), -7)

    def test_safe_widening(self) -> None:
        self.assertEqual(cast_u8_to_u16(200), 200)
        self.assertEqual(cast_u8_to_i16(200), 200)
        self.assertEqual(cast_i8_to_i16(-100), -100)
        self.assertEqual(cast_i32_to_i64(-5), -5)

    def test_narrowing_same_signedness(self) -> None:
        self.assertEqual(cast_u32_to_u8(255), 255)
        self.assertEqual(cast_u32_to_u8(0), 0)
        self.assertEqual(cast_i32_to_i8(127), 127)
        self.assertEqual(cast_i32_to_i8(-128), -128)

    def test_signedness_change(self) -> None:
        self.assertEqual(cast_u8_to_i8(127), 127)
        self.assertEqual(cast_i8_to_u8(0), 0)
        self.assertEqual(cast_i8_to_u8(127), 127)
        self.assertEqual(cast_i32_to_u16(65535), 65535)
        self.assertEqual(cast_i8_to_u16(127), 127)

    def test_range_checks(self) -> None:
        self.assertTrue(fits_u32_u8(255))
        self.assertFalse(fits_u32_u8(256))
        self.assertTrue(fits_i32_i8(127))
        self.assertTrue(fits_i32_i8(-128))
        self.assertFalse(fits_i32_i8(128))
        self.assertFalse(fits_i32_i8(-129))
        self.assertTrue(fits_u8_i8(127))
        self.assertFalse(fits_u8_i8(128))
        self.assertTrue(fits_i8_u8(0))
        self.assertTrue(fits_i8_u8(127))
        self.assertFalse(fits_i8_u8(-1))
        self.assertTrue(fits_i32_u16(65535))
        self.assertFalse(fits_i32_u16(65536))
        self.assertFalse(fits_i32_u16(-1))
        self.assertTrue(fits_i8_u16(0))
        self.assertTrue(fits_i8_u16(127))
        self.assertFalse(fits_i8_u16(-1))


class SpyTruncateTest(TestCase):
    """``std.int.truncate``."""

    def test_runtime_truncation(self) -> None:
        self.assertEqual(trunc_u32_u8(0x1FF), 0xFF)
        self.assertEqual(trunc_u32_u8(200), 200)
        self.assertEqual(trunc_i32_i8(0x1FF), -1)

    def test_comptime_fold(self) -> None:
        self.assertEqual(trunc_fold(), 0xFF)

    def test_errors(self) -> None:
        with self.assertRaises(CompileError):
            trunc_widen(1)
        with self.assertRaises(CompileError):
            trunc_sign_mismatch(1)
        with self.assertRaises(CompileError):
            trunc_non_integer(1.0)


class SpyValidRangeTest(TestCase):
    """``std.int.valid_range``."""

    def test_ranges(self) -> None:
        self.assertTrue(range_i8())
        self.assertTrue(range_u8())
        self.assertTrue(range_i32())


class SpyIntCastTest(TestCase):
    """``std.int.int_cast``."""

    def test_in_range(self) -> None:
        self.assertEqual(int_cast_in_range(200), 200)

    def test_out_of_range_panics(self) -> None:
        self.assertTrue(int_cast_panics(256))
        self.assertFalse(int_cast_panics(255))


all_tests = [
    SpyTryIntCastTest,
    SpyTruncateTest,
    SpyValidRangeTest,
    SpyIntCastTest,
]

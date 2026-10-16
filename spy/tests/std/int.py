from unittest import TestCase

from ...compiler import (
    CompileError,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from ...compiler import bool as spy_bool
from ...compiler.dsl import func
from ...compiler.syntax import closure, comptime
from ...std.core import (
    UnwindException,
    catch_unwind,
    float_to_int_unchecked,
    int_to_float_unchecked,
)
from ...std.int import (
    FloatCastError,
    IntCastError,
    float_to_int,
    int_cast,
    int_to_float,
    truncate,
    try_float_to_int,
    try_int_cast,
    try_int_to_float,
    valid_range,
)

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


# -- int_to_float_unchecked / float_to_int_unchecked (no checks) -----------


@func()
def unchecked_i2f_u8(x: u8) -> f64:
    return int_to_float_unchecked(x, f64)


@func()
def unchecked_i2f_i32_f32(x: i32) -> f32:
    return int_to_float_unchecked(x, f32)


@func()
def unchecked_i2f_fold() -> f64:
    comptime()
    x: i64 = 5
    return int_to_float_unchecked(x, f64)


@func()
def unchecked_f2i_i32(x: f64) -> i32:
    return float_to_int_unchecked(x, i32)


@func()
def unchecked_f2i_u64(x: f64) -> u64:
    return float_to_int_unchecked(x, u64)


@func()
def unchecked_f2i_fold() -> i32:
    comptime()
    x: f64 = 3.9
    return float_to_int_unchecked(x, i32)


@func()
def unchecked_f2i_fold_negative() -> i32:
    comptime()
    x: f64 = -3.9
    return float_to_int_unchecked(x, i32)


# -- int_to_float: exactness checks ----------------------------------------


@func()
def to_float_exact(x: i32) -> f64:
    # the whole i32 range is exactly representable in f64: no check
    return int_to_float(x, f64)


@func()
def to_float_u32_f32(x: u32) -> f32:
    return int_to_float(x, f32)


@func()
def to_float_u64_f64(x: u64) -> f64:
    return int_to_float(x, f64)


@func()
def to_float_fold(x: u64) -> f64:
    comptime()
    v: u64 = 9007199254740992  # 2**53, the last exact integer of f64
    return int_to_float(v, f64)


@func(exceptions=FloatCastError)
def try_to_float_u64_f64(x: u64) -> f64:
    return try_int_to_float(x, f64)


@func()
def fits_float_u64(x: u64) -> spy_bool:
    try:
        _ = try_int_to_float(x, f64)
    except FloatCastError:
        return False
    return True


@func()
def to_float_panics(x: u64) -> spy_bool:
    @closure(inline=False)
    def body() -> f64:
        return int_to_float(x, f64)

    try:
        catch_unwind(body)
    except UnwindException:
        return True
    return False


# -- float_to_int: range checks (truncating towards zero) ------------------


@func()
def to_int_i32(x: f64) -> i32:
    return float_to_int(x, i32)


@func()
def to_int_u8(x: f64) -> u8:
    return float_to_int(x, u8)


@func(exceptions=FloatCastError)
def try_to_int_i32(x: f64) -> i32:
    return try_float_to_int(x, i32)


@func()
def fits_int_i32(x: f64) -> spy_bool:
    try:
        _ = try_float_to_int(x, i32)
    except FloatCastError:
        return False
    return True


@func()
def to_int_panics(x: f64) -> spy_bool:
    @closure(inline=False)
    def body() -> i32:
        return float_to_int(x, i32)

    try:
        catch_unwind(body)
    except UnwindException:
        return True
    return False


class SpyUncheckedFloatConversionTest(TestCase):
    """``std.core.int_to_float_unchecked``/``float_to_int_unchecked`` convert
    without any check (``float_to_int_unchecked`` truncates towards zero)."""

    def test_int_to_float(self) -> None:
        self.assertEqual(unchecked_i2f_u8(200), 200.0)
        self.assertEqual(unchecked_i2f_i32_f32(-5), -5.0)

    def test_int_to_float_fold(self) -> None:
        self.assertEqual(unchecked_i2f_fold(), 5.0)

    def test_float_to_int(self) -> None:
        self.assertEqual(unchecked_f2i_i32(3), 3)
        self.assertEqual(unchecked_f2i_i32(-3), -3)
        self.assertEqual(unchecked_f2i_u64(42), 42)

    def test_float_to_int_truncates_towards_zero(self) -> None:
        self.assertEqual(unchecked_f2i_i32(3.9), 3)
        self.assertEqual(unchecked_f2i_i32(-3.9), -3)

    def test_float_to_int_fold(self) -> None:
        self.assertEqual(unchecked_f2i_fold(), 3)
        self.assertEqual(unchecked_f2i_fold_negative(), -3)


class SpyIntToFloatTest(TestCase):
    """``std.int.try_int_to_float``/``int_to_float`` reject an integer that is
    not exactly representable in the target float."""

    def test_exact_conversions(self) -> None:
        self.assertEqual(to_float_exact(42), 42.0)
        self.assertEqual(to_float_exact(-7), -7.0)
        self.assertEqual(to_float_u32_f32(16777216), 16777216.0)  # 2**24

    def test_exact_boundary(self) -> None:
        self.assertEqual(to_float_u64_f64(9007199254740992), 9007199254740992.0)  # 2**53
        self.assertTrue(fits_float_u64(9007199254740992))

    def test_inexact_panics(self) -> None:
        self.assertTrue(to_float_panics(9007199254740993))  # 2**53 + 1
        self.assertFalse(to_float_panics(9007199254740992))

    def test_try_raises_out_of_range(self) -> None:
        self.assertEqual(try_to_float_u64_f64(9007199254740992), 9007199254740992.0)
        self.assertTrue(fits_float_u64(9007199254740992))
        self.assertFalse(fits_float_u64(9007199254740993))

    def test_comptime_fold(self) -> None:
        self.assertEqual(to_float_fold(0), 9007199254740992.0)


class SpyFloatToIntTest(TestCase):
    """``std.int.try_float_to_int``/``float_to_int`` reject a float whose
    truncation does not fit the target integer."""

    def test_truncating(self) -> None:
        self.assertEqual(to_int_i32(3.9), 3)
        self.assertEqual(to_int_i32(-3.9), -3)
        self.assertEqual(to_int_u8(200.7), 200)

    def test_range(self) -> None:
        self.assertTrue(fits_int_i32(2147483647.0))  # 2**31 - 1
        self.assertTrue(fits_int_i32(-2147483648.0))  # -2**31
        self.assertFalse(fits_int_i32(2147483648.0))  # 2**31
        self.assertFalse(fits_int_i32(-2147483649.0))  # -2**31 - 1

    def test_non_finite_fails(self) -> None:
        self.assertFalse(fits_int_i32(float('nan')))
        self.assertFalse(fits_int_i32(float('inf')))
        self.assertFalse(fits_int_i32(float('-inf')))

    def test_out_of_range_panics(self) -> None:
        self.assertTrue(to_int_panics(2147483648.0))
        self.assertFalse(to_int_panics(2147483647.0))

    def test_try_raises_out_of_range(self) -> None:
        self.assertEqual(try_to_int_i32(7.5), 7)
        self.assertFalse(fits_int_i32(2147483648.0))


all_tests = [
    SpyTryIntCastTest,
    SpyTruncateTest,
    SpyValidRangeTest,
    SpyIntCastTest,
    SpyUncheckedFloatConversionTest,
    SpyIntToFloatTest,
    SpyFloatToIntTest,
]

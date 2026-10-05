from unittest import TestCase

from ..compiler import (
    CompileError,
    f32,
    f64,
    i1,
    i32,
    u1,
    u8,
    u32,
    u64,
)
from ..compiler import as_ as spy_as
from ..compiler import bool as spy_bool
from ..compiler.dsl import func
from ..compiler.syntax import comptime, ref
from ..std.core import bitcast

# ---------------------------------------------------------------------------
# ``std.core.bitcast(value, T)``: reinterpret the bits of a scalar (a bool, an
# integer or a float) as another scalar of the same width.  A compile-time
# value is reinterpreted in Python and stays compile-time; a runtime value is
# ``mir.BitCast`` - a real ``llvm.bitcast`` between an integer and a float, and
# a no-op when the two share the LLVM type (a signedness change, or ``bool``
# against its ``i1``/``u1`` counterpart)
# ---------------------------------------------------------------------------


@func()
def bitcast_f32_to_u32(x: f32) -> u32:
    return bitcast(x, u32)


@func()
def bitcast_u32_to_f32(x: u32) -> f32:
    return bitcast(x, f32)


@func()
def bitcast_f64_to_u64(x: f64) -> u64:
    return bitcast(x, u64)


@func()
def bitcast_u64_to_f64(x: u64) -> f64:
    return bitcast(x, f64)


@func()
def bitcast_i32_to_u32(x: i32) -> u32:
    # the same width, only the sign changes: the bits are kept as they are
    return bitcast(x, u32)


@func()
def bitcast_u32_to_i32(x: u32) -> i32:
    return bitcast(x, i32)


@func()
def bitcast_bool_to_u1(b: spy_bool) -> i32:
    # a boolean is a one-bit integer: True is the bit 1, False the bit 0
    return bitcast(b, u1)


@func()
def bitcast_bool_to_i1(b: spy_bool) -> i32:
    # ... and a *signed* one-bit integer reads the bit 1 as -1 (sign extension)
    return bitcast(b, i1)


@func()
def bitcast_u1_roundtrip(b: spy_bool) -> spy_bool:
    return bitcast(bitcast(b, u1), spy_bool)


@func()
def bitcast_i1_roundtrip(b: spy_bool) -> spy_bool:
    return bitcast(bitcast(b, i1), spy_bool)


@func()
def bitcast_bool_identity(b: spy_bool) -> spy_bool:
    # the same type: no instruction at all
    return bitcast(b, spy_bool)


@func()
def bitcast_comptime_fold() -> u64:
    # the bitcast of a compile-time value folds in Python, so the comparison is
    # a compile-time bool: the ``else`` branch (whose bitcast does not size
    # check) is dead and is never typed
    comptime()
    x: f64 = 1.0
    if bitcast(x, u64) == 0x3FF0000000000000:
        return 1
    return bitcast(x, u8)


@func()
def bitcast_size_mismatch(x: i32) -> f64:
    return bitcast(x, f64)


@func()
def bitcast_bool_size_mismatch(b: spy_bool) -> u8:
    return bitcast(b, u8)


@func()
def bitcast_pointer(x: i32) -> u64:
    return bitcast(ref(x), u64)


class SpyBitcastTest(TestCase):
    """``std.core.bitcast`` reinterprets a scalar as another scalar of the same
    width (bool/integer/float), at compile time when the value is known and
    through ``mir.BitCast`` otherwise."""

    def test_float_and_integer(self) -> None:
        self.assertEqual(bitcast_f32_to_u32(spy_as(1.0, f32)), 0x3F800000)
        self.assertAlmostEqual(bitcast_u32_to_f32(spy_as(0x3F800000, u32)), 1.0)
        self.assertEqual(bitcast_f64_to_u64(1.0), 0x3FF0000000000000)
        self.assertAlmostEqual(bitcast_u64_to_f64(spy_as(0x3FF0000000000000, u64)), 1.0)

    def test_integer_signedness(self) -> None:
        # the same width, only the sign differs: the bits are reinterpreted
        self.assertEqual(bitcast_i32_to_u32(-1), 0xFFFFFFFF)
        self.assertEqual(bitcast_u32_to_i32(spy_as(0xFFFFFFFF, u32)), -1)

    def test_bool(self) -> None:
        self.assertEqual(bitcast_bool_to_u1(True), 1)
        self.assertEqual(bitcast_bool_to_u1(False), 0)
        self.assertEqual(bitcast_bool_to_i1(True), -1)
        self.assertEqual(bitcast_bool_to_i1(False), 0)
        self.assertTrue(bitcast_bool_identity(True))
        self.assertFalse(bitcast_bool_identity(False))
        self.assertTrue(bitcast_u1_roundtrip(True))
        self.assertFalse(bitcast_u1_roundtrip(False))
        self.assertTrue(bitcast_i1_roundtrip(True))
        self.assertFalse(bitcast_i1_roundtrip(False))

    def test_comptime_fold(self) -> None:
        self.assertEqual(bitcast_comptime_fold(), 1)

    def test_size_mismatch(self) -> None:
        with self.assertRaises(CompileError):
            bitcast_size_mismatch(1)
        with self.assertRaises(CompileError):
            bitcast_bool_size_mismatch(True)

    def test_unsupported_type(self) -> None:
        # only scalars are supported: a pointer is rejected
        with self.assertRaises(CompileError):
            bitcast_pointer(1)


all_tests = [
    SpyBitcastTest,
]

import math
import struct
from unittest import TestCase

from ...compiler import CompileError, f32, f64
from ...compiler import as_ as spy_as
from ...compiler.dsl import _Context, func
from ...compiler.lower import LLVMBackend
from ...std.math import sin
from ...std.math.sin import sin as software_sin

# ---------------------------------------------------------------------------
# ``std.math.sin``: the sine of an ``f32``/``f64``, dispatched at compile time
# by the host's compile-time option ``sin_mode`` - ``llvm`` (the LLVM
# intrinsic), ``libc`` (the C ``sinf``/``sin``) or ``software`` (the pure spy
# port of musl, also the default when the option is unset).
#
# The mode is read while the function is compiled and the specialization is
# cached per host context, so each mode is exercised in a context of its own.
# ---------------------------------------------------------------------------


def _f64_ulp(a: float, b: float) -> int:
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0
    ia = struct.unpack('<q', struct.pack('<d', a))[0]
    ib = struct.unpack('<q', struct.pack('<d', b))[0]
    return abs(ia - ib)


def _f32_ulp(a: float, b: float) -> int:
    """The distance in f32 ULPs between two Python floats that hold f32 values."""
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0
    ia = struct.unpack('<i', struct.pack('<f', a))[0]
    ib = struct.unpack('<i', struct.pack('<f', b))[0]
    return abs(ia - ib)


def _to_f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


_F64_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 3.141592653589793,
    1.5707963267948966, 0.7853981633974483, 2.0 * 3.141592653589793,
    1e-30, 1e-8, 1e-300, 1e16, 1e17, 1e20, 1e100, 1e300, -1e300,
    100.0, 999999.0, 123456789.123, 2.0**20, 2.0**30,
]

_F32_VALUES = [
    0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 0.7853982, 2.0, 3.0, 3.1415927,
    6.0, 100.0, 1234.5678, 1e-4, 1e-30, 1e7, 1e8, -1e8,
    1e9, -1e9, 1e10, 1e20, 1e30, 3.4e38,
]


# -- software implementation: the port itself, tested directly ---------------


@func()
def sw_f64(x: f64) -> f64:
    return software_sin(x)


@func()
def sw_f32(x: f32) -> f32:
    return software_sin(x)


# -- one host context per mode ----------------------------------------------


def _context_with_mode(mode: bytes | None) -> _Context:
    ctx = _Context(LLVMBackend())
    if mode is not None:
        ctx.set_compile_option(b'sin_mode', mode)
    return ctx


_CTX_LLVM = _context_with_mode(b'llvm')
_CTX_LIBC = _context_with_mode(b'libc')
_CTX_SOFTWARE = _context_with_mode(b'software')
_CTX_DEFAULT = _context_with_mode(None)
_CTX_BOGUS = _context_with_mode(b'bogus')


@_CTX_LLVM.func()
def mode_llvm[T](x: T) -> T:
    return sin(x)


@_CTX_LIBC.func()
def mode_libc[T](x: T) -> T:
    return sin(x)


@_CTX_SOFTWARE.func()
def mode_software[T](x: T) -> T:
    return sin(x)


@_CTX_DEFAULT.func()
def mode_default[T](x: T) -> T:
    return sin(x)


@_CTX_BOGUS.func()
def mode_bogus(x: f64) -> f64:
    return sin(x)


class SpySoftwareSinTest(TestCase):
    """The pure spy ``sin`` (the musl port): both ``f32`` and ``f64`` are
    correctly rounded (well within 1 ULP of the C library)."""

    def test_f64(self) -> None:
        for v in _F64_VALUES:
            self.assertLessEqual(_f64_ulp(sw_f64(v), math.sin(v)), 1, msg=repr(v))

    def test_f32(self) -> None:
        for v in _F32_VALUES:
            want = _to_f32(math.sin(_to_f32(v)))
            self.assertLessEqual(_f32_ulp(float(sw_f32(spy_as(v, f32))), want), 1, msg=repr(v))

    def test_negative_zero(self) -> None:
        # sin(-0) is -0
        self.assertEqual(math.copysign(1.0, sw_f64(-0.0)), -1.0)


class SpySinModeTest(TestCase):
    """``sin_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        for v in _F64_VALUES:
            self.assertLessEqual(_f64_ulp(mode_llvm(v), math.sin(v)), 2, msg=repr(v))
        self.assertLessEqual(
            _f32_ulp(float(mode_llvm(spy_as(0.5, f32))), _to_f32(math.sin(0.5))), 2
        )

    def test_libc(self) -> None:
        for v in _F64_VALUES:
            self.assertLessEqual(_f64_ulp(mode_libc(v), math.sin(v)), 2, msg=repr(v))
        self.assertLessEqual(
            _f32_ulp(float(mode_libc(spy_as(0.5, f32))), _to_f32(math.sin(0.5))), 2
        )

    def test_software(self) -> None:
        for v in _F64_VALUES:
            self.assertLessEqual(_f64_ulp(mode_software(v), math.sin(v)), 1, msg=repr(v))
        self.assertLessEqual(
            _f32_ulp(float(mode_software(spy_as(0.5, f32))), _to_f32(math.sin(0.5))), 1
        )

    def test_default_is_software(self) -> None:
        # the software implementation is exact, so the default (an unset option)
        # has to match it bit for bit
        for v in _F64_VALUES:
            self.assertEqual(mode_default(v), sw_f64(v), msg=repr(v))
        self.assertEqual(mode_default(0.5), mode_software(0.5))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            mode_bogus(0.5)


all_tests = [
    SpySoftwareSinTest,
    SpySinModeTest,
]

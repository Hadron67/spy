import math
import struct
from collections.abc import Callable
from unittest import TestCase, skipUnless

from llvmlite.binding import llvm_version_info

from ...compiler import CompileError, f32, f64
from ...compiler import as_ as spy_as
from ...compiler.dsl import _Context, func
from ...compiler.lower import LLVMBackend
from ...std.math import cbrt, cos, sin, sqrt, tan
from ...std.math.cbrt import cbrt as software_cbrt
from ...std.math.sqrt import sqrt as software_sqrt
from ...std.math.trig import cos as software_cos
from ...std.math.trig import sin as software_sin
from ...std.math.trig import tan as software_tan

# ---------------------------------------------------------------------------
# ``std.math.sin``/``cos``/``tan``/``sqrt``/``cbrt``: the mathematical functions
# of an ``f32``/``f64``, each dispatched at compile time by the host's
# compile-time option
# ``sin_mode``/``cos_mode``/``tan_mode``/``sqrt_mode``/``cbrt_mode`` - ``llvm``
# (the LLVM intrinsic), ``libc`` (the C ``sinf``/``sin``/...) or ``software``
# (the pure spy port of musl in ``std.math.trig``/``std.math.sqrt``/
# ``std.math.cbrt``, also the default when the option is unset).
#
# The mode is read while the function is compiled and the specialization is
# cached per host context, so each mode is exercised in a context of its own.
#
# ``llvm.cbrt`` is an LLVM intrinsic only from LLVM 21 on; on an older LLVM a
# call to it is left as an unresolved external, so the ``llvm`` mode of ``cbrt``
# can only be exercised there.
# ---------------------------------------------------------------------------


_LLVM_HAS_CBRT = llvm_version_info >= (21,)


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
    -2.0, -2.5, -3.0, -4.0, -5.0, 2.0, 3.0, 4.0, 5.0, -0.7853981633974483,
    1e-30, 1e-8, 1e-300, 1e16, 1e17, 1e20, 1e100, 1e300, -1e300,
    100.0, 999999.0, 123456789.123, 2.0**20, 2.0**30,
]

_F32_VALUES = [
    0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 0.7853982, 2.0, 3.0, 3.1415927,
    6.0, 100.0, 1234.5678, 1e-4, 1e-30, 1e7, 1e8, -1e8,
    1e9, -1e9, 1e10, 1e20, 1e30, 3.4e38,
    -2.0, -2.5, -3.0, -4.0, -5.0, -7.0,
]

# ``tan`` is a ratio, so its error is measured relatively (an absolute/ULP
# distance is meaningless next to its poles); its values stay away from an odd
# multiple of pi/2.
_TAN_F64_VALUES = [
    0.0, -0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
    0.7853981633974483, 1e-8, 100.0, -100.0, 1e8, 1e16, 1e100, 1e300,
]

_TAN_F32_VALUES = [
    0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
    0.7853982, 1e-4, 100.0, -100.0, 1e8,
]

# ``sqrt`` only takes non-negative arguments (a negative one is a NaN), and it
# is correctly rounded, so the values cover the whole range: the two zeros, a
# subnormal, the smallest normal, several magnitudes, and the powers of two on
# either side of the ``f64``/``f32`` significand width.
_SQRT_F64_VALUES = [
    0.0, -0.0, 1.0, 0.5, 0.1, 2.0, 3.0, 4.0, 5.0, 0.7853981633974483,
    1.5707963267948966, 3.141592653589793, 1e-8, 1e-30, 1e-300, 5e-324,
    2.2250738585072014e-308, 2.0**20, 2.0**30, 2.0**52, 2.0**53, 1e16, 1e20,
    1e100, 1e300, 100.0, 999999.0, 123456789.123,
]

_SQRT_F32_VALUES = [
    0.0, 1.0, 0.5, 0.1, 2.0, 3.0, 0.7853982, 1e-4, 1e-30, 1.4e-45,
    1.1754944e-38, 2.0**12, 2.0**23, 2.0**24, 1e7, 1e8, 1e30, 3.4e38,
]

# ``cbrt`` is odd (negative arguments are computed, not a NaN), so both signs are
# covered, together with the two zeros, the exact cubes, a subnormal, the
# smallest normal and several magnitudes.
_CBRT_F64_VALUES = [
    0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 27.0, -27.0, 0.5, -0.5, 0.1, -0.1,
    2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 0.7853981633974483,
    3.141592653589793, 1e-8, -1e-8, 1e-30, 1e-300, 5e-324,
    2.2250738585072014e-308, 2.0**20, 2.0**30, 2.0**52, 2.0**53, 1e16, 1e20,
    1e100, 1e300, -1e300, 100.0, -100.0, 999999.0, 123456789.123,
]

_CBRT_F32_VALUES = [
    0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 27.0, -27.0, 0.5, -0.5, 0.1,
    2.0, -2.0, 3.0, -3.0, 0.7853982, 1e-4, -1e-4, 1e-30, 1.4e-45,
    1.1754944e-38, 2.0**12, 2.0**23, 2.0**24, 1e7, -1e7, 1e8, 1e30, 3.4e38,
]


# -- software implementation: the port itself, tested directly ---------------


@func()
def sw_sin_f64(x: f64) -> f64:
    return software_sin(x)


@func()
def sw_sin_f32(x: f32) -> f32:
    return software_sin(x)


@func()
def sw_cos_f64(x: f64) -> f64:
    return software_cos(x)


@func()
def sw_cos_f32(x: f32) -> f32:
    return software_cos(x)


@func()
def sw_tan_f64(x: f64) -> f64:
    return software_tan(x)


@func()
def sw_tan_f32(x: f32) -> f32:
    return software_tan(x)


@func()
def sw_sqrt_f64(x: f64) -> f64:
    return software_sqrt(x)


@func()
def sw_sqrt_f32(x: f32) -> f32:
    return software_sqrt(x)


@func()
def sw_cbrt_f64(x: f64) -> f64:
    return software_cbrt(x)


@func()
def sw_cbrt_f32(x: f32) -> f32:
    return software_cbrt(x)


# -- one host context per mode (the option names cover every function) -------


def _context_with_mode(mode: bytes | None) -> _Context:
    ctx = _Context(LLVMBackend())
    if mode is not None:
        ctx.set_compile_option(b'sin_mode', mode)
        ctx.set_compile_option(b'cos_mode', mode)
        ctx.set_compile_option(b'tan_mode', mode)
        ctx.set_compile_option(b'sqrt_mode', mode)
        ctx.set_compile_option(b'cbrt_mode', mode)
    return ctx


_CTX_LLVM = _context_with_mode(b'llvm')
_CTX_LIBC = _context_with_mode(b'libc')
_CTX_SOFTWARE = _context_with_mode(b'software')
_CTX_DEFAULT = _context_with_mode(None)
_CTX_BOGUS = _context_with_mode(b'bogus')


@_CTX_LLVM.func()
def libm_sin_llvm[T](x: T) -> T:
    return sin(x)


@_CTX_LLVM.func()
def libm_cos_llvm[T](x: T) -> T:
    return cos(x)


@_CTX_LLVM.func()
def libm_tan_llvm[T](x: T) -> T:
    return tan(x)


@_CTX_LLVM.func()
def libm_sqrt_llvm[T](x: T) -> T:
    return sqrt(x)


@_CTX_LLVM.func()
def libm_cbrt_llvm[T](x: T) -> T:
    return cbrt(x)


@_CTX_LIBC.func()
def libm_sin_libc[T](x: T) -> T:
    return sin(x)


@_CTX_LIBC.func()
def libm_cos_libc[T](x: T) -> T:
    return cos(x)


@_CTX_LIBC.func()
def libm_tan_libc[T](x: T) -> T:
    return tan(x)


@_CTX_LIBC.func()
def libm_sqrt_libc[T](x: T) -> T:
    return sqrt(x)


@_CTX_LIBC.func()
def libm_cbrt_libc[T](x: T) -> T:
    return cbrt(x)


@_CTX_SOFTWARE.func()
def libm_sin_software[T](x: T) -> T:
    return sin(x)


@_CTX_SOFTWARE.func()
def libm_cos_software[T](x: T) -> T:
    return cos(x)


@_CTX_SOFTWARE.func()
def libm_tan_software[T](x: T) -> T:
    return tan(x)


@_CTX_SOFTWARE.func()
def libm_sqrt_software[T](x: T) -> T:
    return sqrt(x)


@_CTX_SOFTWARE.func()
def libm_cbrt_software[T](x: T) -> T:
    return cbrt(x)


@_CTX_DEFAULT.func()
def libm_sin_default[T](x: T) -> T:
    return sin(x)


@_CTX_DEFAULT.func()
def libm_cos_default[T](x: T) -> T:
    return cos(x)


@_CTX_DEFAULT.func()
def libm_tan_default[T](x: T) -> T:
    return tan(x)


@_CTX_DEFAULT.func()
def libm_sqrt_default[T](x: T) -> T:
    return sqrt(x)


@_CTX_DEFAULT.func()
def libm_cbrt_default[T](x: T) -> T:
    return cbrt(x)


@_CTX_BOGUS.func()
def libm_sin_bogus(x: f64) -> f64:
    return sin(x)


@_CTX_BOGUS.func()
def libm_cos_bogus(x: f64) -> f64:
    return cos(x)


@_CTX_BOGUS.func()
def libm_tan_bogus(x: f64) -> f64:
    return tan(x)


@_CTX_BOGUS.func()
def libm_sqrt_bogus(x: f64) -> f64:
    return sqrt(x)


@_CTX_BOGUS.func()
def libm_cbrt_bogus(x: f64) -> f64:
    return cbrt(x)


class _TrigUlpTest(TestCase):
    """Shared comparison helpers (no tests of its own)."""

    def check_f64(
        self,
        fn: Callable[[float], float],
        expect: Callable[[float], float],
        tol: int,
        values: list[float] = _F64_VALUES,
    ) -> None:
        for v in values:
            self.assertLessEqual(_f64_ulp(fn(v), expect(v)), tol, msg=repr(v))

    def check_f32(
        self,
        fn: Callable[[f32], f32],
        expect: Callable[[float], float],
        tol: int,
        values: list[float] = _F32_VALUES,
    ) -> None:
        for v in values:
            want = _to_f32(expect(_to_f32(v)))
            self.assertLessEqual(_f32_ulp(float(fn(spy_as(v, f32))), want), tol, msg=repr(v))

    def assert_rel_close(self, got: float, want: float, tol: float, msg: str) -> None:
        if got == want or (math.isnan(got) and math.isnan(want)):
            return
        denom = abs(want) if want != 0.0 else 1.0
        self.assertLessEqual(abs(got - want), tol * denom, msg=msg)

    def check_tan_f64(self, fn: Callable[[float], float], tol: float) -> None:
        for v in _TAN_F64_VALUES:
            self.assert_rel_close(fn(v), math.tan(v), tol, repr(v))

    def check_tan_f32(self, fn: Callable[[f32], f32], tol: float) -> None:
        for v in _TAN_F32_VALUES:
            got = float(fn(spy_as(v, f32)))
            self.assert_rel_close(got, math.tan(_to_f32(v)), tol, repr(v))


class SpySoftwareTrigTest(_TrigUlpTest):
    """The pure spy ``sin``/``cos``/``tan`` (the musl port): both ``f32`` and
    ``f64`` are correctly rounded (well within 1 ULP of the C library); ``tan``
    is ``sin / cos`` of the same kernels."""

    def test_sin_f64(self) -> None:
        self.check_f64(sw_sin_f64, math.sin, 1)

    def test_sin_f32(self) -> None:
        self.check_f32(sw_sin_f32, math.sin, 1)

    def test_cos_f64(self) -> None:
        self.check_f64(sw_cos_f64, math.cos, 1)

    def test_cos_f32(self) -> None:
        self.check_f32(sw_cos_f32, math.cos, 1)

    def test_tan_f64(self) -> None:
        self.check_tan_f64(sw_tan_f64, 1e-14)

    def test_tan_f32(self) -> None:
        self.check_tan_f32(sw_tan_f32, 1e-5)

    def test_tan_is_sin_over_cos(self) -> None:
        for v in _TAN_F64_VALUES:
            self.assertEqual(sw_tan_f64(v), sw_sin_f64(v) / sw_cos_f64(v), msg=repr(v))

    def test_sin_of_negative_zero(self) -> None:
        self.assertEqual(math.copysign(1.0, sw_sin_f64(-0.0)), -1.0)

    def test_cos_of_zero(self) -> None:
        self.assertEqual(sw_cos_f64(0.0), 1.0)
        self.assertEqual(sw_cos_f64(-0.0), 1.0)

    def test_tan_of_zero(self) -> None:
        self.assertEqual(sw_tan_f64(0.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_tan_f64(-0.0)), -1.0)


class SpyTrigModeTest(_TrigUlpTest):
    """``sin_mode``/``cos_mode``/``tan_mode`` select the
    ``llvm``/``libc``/``software`` implementation at compile time; an unset
    option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_sin_llvm, math.sin, 2)
        self.check_f32(libm_sin_llvm, math.sin, 2)
        self.check_f64(libm_cos_llvm, math.cos, 2)
        self.check_f32(libm_cos_llvm, math.cos, 2)
        self.check_tan_f64(libm_tan_llvm, 1e-13)
        self.check_tan_f32(libm_tan_llvm, 1e-5)

    def test_libc(self) -> None:
        self.check_f64(libm_sin_libc, math.sin, 2)
        self.check_f32(libm_sin_libc, math.sin, 2)
        self.check_f64(libm_cos_libc, math.cos, 2)
        self.check_f32(libm_cos_libc, math.cos, 2)
        self.check_tan_f64(libm_tan_libc, 1e-13)
        self.check_tan_f32(libm_tan_libc, 1e-5)

    def test_software(self) -> None:
        self.check_f64(libm_sin_software, math.sin, 1)
        self.check_f32(libm_sin_software, math.sin, 1)
        self.check_f64(libm_cos_software, math.cos, 1)
        self.check_f32(libm_cos_software, math.cos, 1)
        self.check_tan_f64(libm_tan_software, 1e-14)
        self.check_tan_f32(libm_tan_software, 1e-5)

    def test_default_is_software(self) -> None:
        # the software implementation is exact, so the default (an unset option)
        # has to match it bit for bit
        for v in _F64_VALUES:
            self.assertEqual(libm_sin_default(v), sw_sin_f64(v), msg=repr(v))
            self.assertEqual(libm_cos_default(v), sw_cos_f64(v), msg=repr(v))
        for v in _TAN_F64_VALUES:
            self.assertEqual(libm_tan_default(v), sw_tan_f64(v), msg=repr(v))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_sin_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_cos_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_tan_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_sqrt_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_cbrt_bogus(0.5)


class SpySoftwareSqrtTest(_TrigUlpTest):
    """The pure spy ``sqrt`` (the musl port): both ``f32`` and ``f64`` are
    correctly rounded, and the special values (the two zeros, ``inf``, a NaN
    and a negative argument) come out as C's do."""

    def test_sqrt_f64(self) -> None:
        self.check_f64(sw_sqrt_f64, math.sqrt, 0, _SQRT_F64_VALUES)

    def test_sqrt_f32(self) -> None:
        self.check_f32(sw_sqrt_f32, math.sqrt, 1, _SQRT_F32_VALUES)

    def test_sqrt_of_zero(self) -> None:
        self.assertEqual(sw_sqrt_f64(0.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_sqrt_f64(-0.0)), -1.0)

    def test_sqrt_of_inf(self) -> None:
        self.assertEqual(sw_sqrt_f64(math.inf), math.inf)

    def test_sqrt_of_nan(self) -> None:
        self.assertTrue(math.isnan(sw_sqrt_f64(math.nan)))

    def test_sqrt_of_negative_is_nan(self) -> None:
        for v in (-1.0, -0.5, -5e-324, -1e300):
            self.assertTrue(math.isnan(sw_sqrt_f64(v)), msg=repr(v))
        for v in (-1.0, -0.5, -1e30):
            self.assertTrue(math.isnan(float(sw_sqrt_f32(spy_as(v, f32)))), msg=repr(v))


class SpySqrtModeTest(_TrigUlpTest):
    """``sqrt_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_sqrt_llvm, math.sqrt, 1, _SQRT_F64_VALUES)
        self.check_f32(libm_sqrt_llvm, math.sqrt, 1, _SQRT_F32_VALUES)

    def test_libc(self) -> None:
        self.check_f64(libm_sqrt_libc, math.sqrt, 1, _SQRT_F64_VALUES)
        self.check_f32(libm_sqrt_libc, math.sqrt, 1, _SQRT_F32_VALUES)

    def test_software(self) -> None:
        self.check_f64(libm_sqrt_software, math.sqrt, 0, _SQRT_F64_VALUES)
        self.check_f32(libm_sqrt_software, math.sqrt, 1, _SQRT_F32_VALUES)

    def test_default_is_software(self) -> None:
        # the software implementation is exact, so the default (an unset option)
        # has to match it bit for bit
        for v in _SQRT_F64_VALUES:
            self.assertEqual(libm_sqrt_default(v), sw_sqrt_f64(v), msg=repr(v))

    def test_negative_is_nan_in_every_mode(self) -> None:
        for fn in (libm_sqrt_llvm, libm_sqrt_libc, libm_sqrt_software, libm_sqrt_default):
            self.assertTrue(math.isnan(fn(-1.0)))


class SpySoftwareCbrtTest(_TrigUlpTest):
    """The pure spy ``cbrt`` (the musl port): both ``f32`` and ``f64`` are within
    1 ULP of the correctly rounded result (musl bounds the error at < 0.667 ULP),
    it is odd, and the special values (the two zeros, ``+-inf``, a NaN) come out
    as C's do."""

    def test_cbrt_f64(self) -> None:
        self.check_f64(sw_cbrt_f64, math.cbrt, 1, _CBRT_F64_VALUES)

    def test_cbrt_f32(self) -> None:
        self.check_f32(sw_cbrt_f32, math.cbrt, 1, _CBRT_F32_VALUES)

    def test_cbrt_of_exact_cubes(self) -> None:
        for v, want in ((8.0, 2.0), (27.0, 3.0), (-8.0, -2.0), (-27.0, -3.0),
                        (0.125, 0.5), (1e9, 1000.0)):
            self.assertEqual(sw_cbrt_f64(v), want, msg=repr(v))

    def test_cbrt_of_zero(self) -> None:
        self.assertEqual(sw_cbrt_f64(0.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_cbrt_f64(-0.0)), -1.0)

    def test_cbrt_of_inf(self) -> None:
        self.assertEqual(sw_cbrt_f64(math.inf), math.inf)
        self.assertEqual(sw_cbrt_f64(-math.inf), -math.inf)

    def test_cbrt_of_nan(self) -> None:
        self.assertTrue(math.isnan(sw_cbrt_f64(math.nan)))

    def test_cbrt_is_odd(self) -> None:
        for v in _CBRT_F64_VALUES:
            if v > 0.0:
                self.assertEqual(sw_cbrt_f64(-v), -sw_cbrt_f64(v), msg=repr(v))


class SpyCbrtModeTest(_TrigUlpTest):
    """``cbrt_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    @skipUnless(_LLVM_HAS_CBRT, 'llvm.cbrt requires LLVM >= 21')
    def test_llvm(self) -> None:
        self.check_f64(libm_cbrt_llvm, math.cbrt, 2, _CBRT_F64_VALUES)
        self.check_f32(libm_cbrt_llvm, math.cbrt, 2, _CBRT_F32_VALUES)
        self.assertEqual(libm_cbrt_llvm(-8.0), -2.0)

    def test_libc(self) -> None:
        self.check_f64(libm_cbrt_libc, math.cbrt, 2, _CBRT_F64_VALUES)
        self.check_f32(libm_cbrt_libc, math.cbrt, 2, _CBRT_F32_VALUES)

    def test_software(self) -> None:
        self.check_f64(libm_cbrt_software, math.cbrt, 1, _CBRT_F64_VALUES)
        self.check_f32(libm_cbrt_software, math.cbrt, 1, _CBRT_F32_VALUES)

    def test_default_is_software(self) -> None:
        # the software implementation is deterministic, so the default (an unset
        # option) has to match it bit for bit
        for v in _CBRT_F64_VALUES:
            self.assertEqual(libm_cbrt_default(v), sw_cbrt_f64(v), msg=repr(v))

    def test_negative_is_computed_in_every_mode(self) -> None:
        for fn in (libm_cbrt_libc, libm_cbrt_software, libm_cbrt_default):
            self.assertEqual(fn(-8.0), -2.0)


all_tests = [
    SpySoftwareTrigTest,
    SpyTrigModeTest,
    SpySoftwareSqrtTest,
    SpySqrtModeTest,
    SpySoftwareCbrtTest,
    SpyCbrtModeTest,
]

import math
import struct
from collections.abc import Callable
from unittest import TestCase, skipUnless

from llvmlite.binding import llvm_version_info

from ...compiler import CompileError, f32, f64
from ...compiler import as_ as spy_as
from ...compiler.dsl import Context, func
from ...compiler.lower import LLVMBackend
from ...std.math import (
    acos,
    asin,
    atan,
    atan2,
    cbrt,
    cos,
    exp,
    log,
    pow,
    sin,
    sqrt,
    tan,
)
from ...std.math.cbrt import cbrt as software_cbrt
from ...std.math.exp import exp as software_exp
from ...std.math.invtrig import acos as software_acos
from ...std.math.invtrig import asin as software_asin
from ...std.math.invtrig import atan as software_atan
from ...std.math.invtrig import atan2 as software_atan2
from ...std.math.log import log as software_log
from ...std.math.pow import pow as software_pow
from ...std.math.sqrt import sqrt as software_sqrt
from ...std.math.trig import cos as software_cos
from ...std.math.trig import sin as software_sin
from ...std.math.trig import tan as software_tan

# ---------------------------------------------------------------------------
# ``std.math.sin``/``cos``/``tan``/``asin``/``acos``/``atan``/``atan2``/
# ``sqrt``/``cbrt``/``pow``/``exp``/``log``: the mathematical functions of an
# ``f32``/``f64``, each dispatched at compile time by the host's compile-time
# option ``sin_mode``/``cos_mode``/``tan_mode``/``asin_mode``/``acos_mode``/
# ``atan_mode``/``atan2_mode``/``sqrt_mode``/``cbrt_mode``/``pow_mode``/
# ``exp_mode``/``log_mode`` - ``llvm``
# (the LLVM intrinsic), ``libc`` (the C ``sinf``/``sin``/...) or ``software``
# (the pure spy port of musl in ``std.math.trig``/``std.math.invtrig``/
# ``std.math.sqrt``/``std.math.cbrt``/``std.math.pow``/``std.math.exp``/
# ``std.math.log``, also the default when the option is unset).
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

# ``pow`` takes two arguments; the values stay where the result neither
# overflows nor underflows and a negative base always has an integer exponent
# (the special cases are exercised separately).
_POW_F64_VALUES = [
    (2.0, 10.0), (2.0, 0.5), (3.0, 3.0), (10.0, 3.5), (0.5, -2.0), (0.5, 2.5),
    (1.5, -1.5), (3.7, 2.3), (123456.789, -1.234), (1e10, 0.3), (1e-10, 2.5),
    (0.1, 0.1), (7.0, 7.0), (-2.0, 3.0), (-2.0, -3.0), (-1.5, 5.0), (2.5, 8.0),
    (0.9, 100.0), (1.1, 1000.0), (2.0, -10.0),
]

_POW_F32_VALUES = [
    (2.0, 10.0), (2.0, 0.5), (3.0, 3.0), (1.5, 2.5), (0.5, -2.0), (10.0, 3.3),
    (0.1, 5.0), (-2.0, 3.0), (7.0, 0.5), (1.25, 8.0), (2.0, -4.0), (0.8, 20.0),
]

# ``exp`` and ``log`` cover the ordinary range, the subnormal range and the
# values close to 1 where ``log`` uses its dedicated polynomial.
_EXP_F64_VALUES = [
    0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0,
    100.0, -100.0, 700.0, -700.0, 709.0, -745.0, 1e-8, -1e-8, 1e-16,
]

_EXP_F32_VALUES = [
    0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0,
    80.0, -80.0, 88.0, -88.0, 1e-4, -1e-4, 1e-8,
]

_LOG_F64_VALUES = [
    0.5, 1.0, 2.0, 10.0, 0.1, 1.0 - 2.0 ** -4,
    1.0 + float.fromhex('0x1.09p-4'), 1e-8, 1e-30, 1e-300, 5e-324,
    2.2250738585072014e-308, 1e300, 123456.789, 999999.0, 2.0 ** -20,
]

_LOG_F32_VALUES = [
    0.5, 1.0, 2.0, 10.0, 0.1, 0.9, 1.1, 1e-4, 1e-30, 1.4e-45,
    1.1754944e-38, 1e7, 1e30, 3.4e38,
]

# ``asin``/``acos`` take ``[-1, 1]``: the values cover both halves (the ``|x| = 0.5``
# split and the ``|x| = 0.975`` / ``2**-57`` / ``2**-26`` thresholds), the ends,
# the tiny and subnormal arguments that have their own fast path, and both signs.
_ASIN_ACOS_F64_VALUES = [
    -1.0, -0.9999999999, -0.99, -0.975, -0.9749, -0.9, -0.75, -0.5,
    -0.25, -0.1, -2.0 ** -26, -2.0 ** -56, -2.0 ** -57, -1e-8, -1e-20, -5e-324,
    0.0, -0.0, 5e-324, 1e-20, 1e-8, 2.0 ** -57, 2.0 ** -56, 2.0 ** -26, 0.1,
    0.25, 0.5, 0.75, 0.9, 0.9749, 0.975, 0.99, 0.9999999999, 1.0,
]

_ASIN_ACOS_F32_VALUES = [
    -1.0, -0.99, -0.975, -0.9, -0.75, -0.5, -0.1, -2.0 ** -26, -1e-8,
    -1.4e-45, 0.0, -0.0, 1.4e-45, 1e-8, 2.0 ** -26, 0.1, 0.5, 0.75, 0.9,
    0.975, 0.99, 1.0,
]

# ``atan`` is odd, so both signs are covered, together with the two zeros, the
# reduction boundaries (0.4375, 0.6875, 1.1875, 2.4375), the tiny/large cutoffs
# and the special values.
_ATAN_F64_VALUES = [
    -math.inf, -1e300, -2.0 ** 66, -1e16, -3.0, -2.5, -2.4375, -1.1875,
    -1.0, -0.7, -0.6875, -0.5, -0.4375, -0.4, -0.1, -2.0 ** -26,
    -2.0 ** -27, -5e-324, -0.0, 0.0, 5e-324, 2.0 ** -27, 2.0 ** -26, 0.1,
    0.4, 0.4375, 0.5, 0.6875, 0.7, 1.0, 1.1875, 2.4375, 2.5, 3.0,
    1e16, 2.0 ** 66, 1e300, math.inf,
]

_ATAN_F32_VALUES = [
    -math.inf, -3.4e38, -2.0 ** 27, -1e7, -3.0, -2.5, -2.4375, -1.1875,
    -1.0, -0.7, -0.6875, -0.5, -0.4375, -0.4, -0.1, -2.0 ** -12,
    -2.0 ** -13, -1.4e-45, -0.0, 0.0, 1.4e-45, 2.0 ** -13, 2.0 ** -12, 0.1,
    0.4, 0.4375, 0.5, 0.6875, 0.7, 1.0, 1.1875, 2.4375, 2.5, 3.0,
    1e7, 2.0 ** 26, 3.4e38, math.inf,
]

# ``atan2``: every quadrant, the signed zeros, the infinities, the `x == 1` fast
# path and the `|y/x| = 2**+-64` (f64) / `2**+-26` (f32) cutoffs.
_ATAN2_F64_VALUES = [
    (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
    (0.0, 1.0), (-0.0, 1.0), (0.0, -1.0), (-0.0, -1.0),
    (1.0, 0.0), (-1.0, 0.0), (1.0, -0.0), (-1.0, -0.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    (0.1, 0.2), (-0.1, 0.2), (0.1, -0.2), (-0.1, -0.2),
    (0.5, 1.0), (-0.5, 1.0), (0.5, -1.0), (-0.5, -1.0),
    (1.0, math.inf), (-1.0, math.inf), (1.0, -math.inf), (-1.0, -math.inf),
    (math.inf, 1.0), (-math.inf, 1.0), (math.inf, -1.0), (-math.inf, -1.0),
    (math.inf, math.inf), (-math.inf, math.inf), (math.inf, -math.inf),
    (-math.inf, -math.inf),
    (2.0 ** 65, 1.0), (-2.0 ** 65, 1.0), (2.0 ** 65, -1.0), (-2.0 ** 65, -1.0),
    (1.0, 2.0 ** 65), (-1.0, 2.0 ** 65), (1.0, -2.0 ** 65), (-1.0, -2.0 ** 65),
    (2.0 ** -65, 1.0), (-2.0 ** -65, 1.0), (2.0 ** -65, -1.0), (-2.0 ** -65, -1.0),
    (1e300, 1e300), (-1e300, 1e300), (1e300, -1e300), (-1e300, -1e300),
]

_ATAN2_F32_VALUES = [
    (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
    (0.0, 1.0), (-0.0, 1.0), (0.0, -1.0), (-0.0, -1.0),
    (1.0, 0.0), (-1.0, 0.0), (1.0, -0.0), (-1.0, -0.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    (0.1, 0.2), (-0.1, 0.2), (0.1, -0.2), (-0.1, -0.2),
    (0.5, 1.0), (-0.5, 1.0), (0.5, -1.0), (-0.5, -1.0),
    (1.0, math.inf), (-1.0, math.inf), (1.0, -math.inf), (-1.0, -math.inf),
    (math.inf, 1.0), (-math.inf, 1.0), (math.inf, -1.0), (-math.inf, -1.0),
    (math.inf, math.inf), (-math.inf, math.inf), (math.inf, -math.inf),
    (-math.inf, -math.inf),
    (2.0 ** 27, 1.0), (-2.0 ** 27, 1.0), (2.0 ** 27, -1.0), (-2.0 ** 27, -1.0),
    (1.0, 2.0 ** 27), (-1.0, 2.0 ** 27), (1.0, -2.0 ** 27), (-1.0, -2.0 ** 27),
    (2.0 ** -27, 1.0), (-2.0 ** -27, 1.0), (2.0 ** -27, -1.0), (-2.0 ** -27, -1.0),
    (3.4e38, 3.4e38), (-3.4e38, 3.4e38), (3.4e38, -3.4e38), (-3.4e38, -3.4e38),
]


# -- software implementation: the port itself, tested directly ---------------


@func()
def sw_asin_f64(x: f64) -> f64:
    return software_asin(x)


@func()
def sw_asin_f32(x: f32) -> f32:
    return software_asin(x)


@func()
def sw_acos_f64(x: f64) -> f64:
    return software_acos(x)


@func()
def sw_acos_f32(x: f32) -> f32:
    return software_acos(x)


@func()
def sw_atan_f64(x: f64) -> f64:
    return software_atan(x)


@func()
def sw_atan_f32(x: f32) -> f32:
    return software_atan(x)


@func()
def sw_atan2_f64(y: f64, x: f64) -> f64:
    return software_atan2(y, x)


@func()
def sw_atan2_f32(y: f32, x: f32) -> f32:
    return software_atan2(y, x)


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


@func()
def sw_pow_f64(x: f64, y: f64) -> f64:
    return software_pow(x, y)


@func()
def sw_pow_f32(x: f32, y: f32) -> f32:
    return software_pow(x, y)


@func()
def sw_exp_f64(x: f64) -> f64:
    return software_exp(x)


@func()
def sw_exp_f32(x: f32) -> f32:
    return software_exp(x)


@func()
def sw_log_f64(x: f64) -> f64:
    return software_log(x)


@func()
def sw_log_f32(x: f32) -> f32:
    return software_log(x)


# -- one host context per mode (the option names cover every function) -------


def _context_with_mode(mode: bytes | None) -> Context:
    ctx = Context(LLVMBackend())
    if mode is not None:
        ctx.set_compile_option(b'sin_mode', mode)
        ctx.set_compile_option(b'cos_mode', mode)
        ctx.set_compile_option(b'tan_mode', mode)
        ctx.set_compile_option(b'asin_mode', mode)
        ctx.set_compile_option(b'acos_mode', mode)
        ctx.set_compile_option(b'atan_mode', mode)
        ctx.set_compile_option(b'atan2_mode', mode)
        ctx.set_compile_option(b'sqrt_mode', mode)
        ctx.set_compile_option(b'cbrt_mode', mode)
        ctx.set_compile_option(b'pow_mode', mode)
        ctx.set_compile_option(b'exp_mode', mode)
        ctx.set_compile_option(b'log_mode', mode)
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


@_CTX_LLVM.func()
def libm_asin_llvm[T](x: T) -> T:
    return asin(x)


@_CTX_LIBC.func()
def libm_asin_libc[T](x: T) -> T:
    return asin(x)


@_CTX_SOFTWARE.func()
def libm_asin_software[T](x: T) -> T:
    return asin(x)


@_CTX_DEFAULT.func()
def libm_asin_default[T](x: T) -> T:
    return asin(x)


@_CTX_BOGUS.func()
def libm_asin_bogus(x: f64) -> f64:
    return asin(x)


@_CTX_LLVM.func()
def libm_acos_llvm[T](x: T) -> T:
    return acos(x)


@_CTX_LIBC.func()
def libm_acos_libc[T](x: T) -> T:
    return acos(x)


@_CTX_SOFTWARE.func()
def libm_acos_software[T](x: T) -> T:
    return acos(x)


@_CTX_DEFAULT.func()
def libm_acos_default[T](x: T) -> T:
    return acos(x)


@_CTX_BOGUS.func()
def libm_acos_bogus(x: f64) -> f64:
    return acos(x)


@_CTX_LLVM.func()
def libm_atan_llvm[T](x: T) -> T:
    return atan(x)


@_CTX_LIBC.func()
def libm_atan_libc[T](x: T) -> T:
    return atan(x)


@_CTX_SOFTWARE.func()
def libm_atan_software[T](x: T) -> T:
    return atan(x)


@_CTX_DEFAULT.func()
def libm_atan_default[T](x: T) -> T:
    return atan(x)


@_CTX_BOGUS.func()
def libm_atan_bogus(x: f64) -> f64:
    return atan(x)


@_CTX_LLVM.func()
def libm_atan2_llvm[T](y: T, x: T) -> T:
    return atan2(y, x)


@_CTX_LIBC.func()
def libm_atan2_libc[T](y: T, x: T) -> T:
    return atan2(y, x)


@_CTX_SOFTWARE.func()
def libm_atan2_software[T](y: T, x: T) -> T:
    return atan2(y, x)


@_CTX_DEFAULT.func()
def libm_atan2_default[T](y: T, x: T) -> T:
    return atan2(y, x)


@_CTX_BOGUS.func()
def libm_atan2_bogus(y: f64, x: f64) -> f64:
    return atan2(y, x)


@_CTX_LLVM.func()
def libm_pow_llvm[T](x: T, y: T) -> T:
    return pow(x, y)


@_CTX_LIBC.func()
def libm_pow_libc[T](x: T, y: T) -> T:
    return pow(x, y)


@_CTX_SOFTWARE.func()
def libm_pow_software[T](x: T, y: T) -> T:
    return pow(x, y)


@_CTX_DEFAULT.func()
def libm_pow_default[T](x: T, y: T) -> T:
    return pow(x, y)


@_CTX_BOGUS.func()
def libm_pow_bogus(x: f64, y: f64) -> f64:
    return pow(x, y)


@_CTX_LLVM.func()
def libm_exp_llvm[T](x: T) -> T:
    return exp(x)


@_CTX_LIBC.func()
def libm_exp_libc[T](x: T) -> T:
    return exp(x)


@_CTX_SOFTWARE.func()
def libm_exp_software[T](x: T) -> T:
    return exp(x)


@_CTX_DEFAULT.func()
def libm_exp_default[T](x: T) -> T:
    return exp(x)


@_CTX_BOGUS.func()
def libm_exp_bogus(x: f64) -> f64:
    return exp(x)


@_CTX_LLVM.func()
def libm_log_llvm[T](x: T) -> T:
    return log(x)


@_CTX_LIBC.func()
def libm_log_libc[T](x: T) -> T:
    return log(x)


@_CTX_SOFTWARE.func()
def libm_log_software[T](x: T) -> T:
    return log(x)


@_CTX_DEFAULT.func()
def libm_log_default[T](x: T) -> T:
    return log(x)


@_CTX_BOGUS.func()
def libm_log_bogus(x: f64) -> f64:
    return log(x)


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

    def check_atan2_f64(
        self,
        fn: Callable[[float, float], float],
        tol: int,
        values: list[tuple[float, float]] = _ATAN2_F64_VALUES,
    ) -> None:
        for y, x in values:
            self.assertLessEqual(_f64_ulp(fn(y, x), math.atan2(y, x)), tol, msg=repr((y, x)))

    def check_atan2_f32(
        self,
        fn: Callable[[f32, f32], f32],
        tol: int,
        values: list[tuple[float, float]] = _ATAN2_F32_VALUES,
    ) -> None:
        for y, x in values:
            want = _to_f32(math.atan2(_to_f32(y), _to_f32(x)))
            got = float(fn(spy_as(y, f32), spy_as(x, f32)))
            self.assertLessEqual(_f32_ulp(got, want), tol, msg=repr((y, x)))

    def check_pow_f64(
        self,
        fn: Callable[[float, float], float],
        tol: int,
        values: list[tuple[float, float]] = _POW_F64_VALUES,
    ) -> None:
        for x, y in values:
            self.assertLessEqual(_f64_ulp(fn(x, y), math.pow(x, y)), tol, msg=repr((x, y)))

    def check_pow_f32(
        self,
        fn: Callable[[f32, f32], f32],
        tol: int,
        values: list[tuple[float, float]] = _POW_F32_VALUES,
    ) -> None:
        for x, y in values:
            want = _to_f32(math.pow(_to_f32(x), _to_f32(y)))
            got = float(fn(spy_as(x, f32), spy_as(y, f32)))
            self.assertLessEqual(_f32_ulp(got, want), tol, msg=repr((x, y)))


class SpySoftwareTrigTest(_TrigUlpTest):
    """The pure spy ``sin``/``cos``/``tan`` (the musl port): both ``f32`` and
    ``f64`` are correctly rounded (well within 1 ULP of the C library)."""

    def test_sin_f64(self) -> None:
        self.check_f64(sw_sin_f64, math.sin, 1)

    def test_sin_f32(self) -> None:
        self.check_f32(sw_sin_f32, math.sin, 1)

    def test_cos_f64(self) -> None:
        self.check_f64(sw_cos_f64, math.cos, 1)

    def test_cos_f32(self) -> None:
        self.check_f32(sw_cos_f32, math.cos, 1)

    def test_tan_f64(self) -> None:
        self.check_tan_f64(sw_tan_f64, 1e-15)

    def test_tan_f32(self) -> None:
        self.check_tan_f32(sw_tan_f32, 2e-7)

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
        self.check_tan_f64(libm_tan_software, 1e-15)
        self.check_tan_f32(libm_tan_software, 2e-7)

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


class SpySoftwareAsinAcosTest(_TrigUlpTest):
    """The pure spy ``asin``/``acos`` (the musl port): both ``f32`` and ``f64``
    are within 1 ULP of the correctly rounded result, and the special values (the
    ends, the arguments past ``+-1``, a NaN and the zeros) come out as C's do."""

    def test_asin_f64(self) -> None:
        self.check_f64(sw_asin_f64, math.asin, 1, _ASIN_ACOS_F64_VALUES)

    def test_asin_f32(self) -> None:
        self.check_f32(sw_asin_f32, math.asin, 1, _ASIN_ACOS_F32_VALUES)

    def test_acos_f64(self) -> None:
        self.check_f64(sw_acos_f64, math.acos, 1, _ASIN_ACOS_F64_VALUES)

    def test_acos_f32(self) -> None:
        self.check_f32(sw_acos_f32, math.acos, 1, _ASIN_ACOS_F32_VALUES)

    def test_asin_f64_of_the_ends(self) -> None:
        self.assertLessEqual(_f64_ulp(sw_asin_f64(1.0), math.asin(1.0)), 1)
        self.assertLessEqual(_f64_ulp(sw_asin_f64(-1.0), math.asin(-1.0)), 1)

    def test_asin_f64_of_a_zero_keeps_its_sign(self) -> None:
        self.assertEqual(sw_asin_f64(0.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_asin_f64(-0.0)), -1.0)

    def test_acos_f64_of_the_ends(self) -> None:
        self.assertEqual(sw_acos_f64(1.0), 0.0)
        self.assertLessEqual(_f64_ulp(sw_acos_f64(-1.0), math.acos(-1.0)), 1)

    def test_acos_f64_of_a_zero(self) -> None:
        for v in (0.0, -0.0):
            self.assertLessEqual(_f64_ulp(sw_acos_f64(v), math.pi / 2), 1, msg=repr(v))

    def test_asin_is_odd(self) -> None:
        for v in _ASIN_ACOS_F64_VALUES:
            if v > 0.0:
                self.assertEqual(sw_asin_f64(-v), -sw_asin_f64(v), msg=repr(v))

    def test_out_of_range_is_nan(self) -> None:
        for fn, v in ((sw_asin_f64, 2.0), (sw_asin_f64, -2.0), (sw_asin_f64, 1e300),
                      (sw_acos_f64, 2.0), (sw_acos_f64, -2.0), (sw_acos_f64, 1e300),
                      (sw_asin_f64, math.inf), (sw_asin_f64, math.nan),
                      (sw_acos_f64, -math.inf), (sw_acos_f64, math.nan)):
            self.assertTrue(math.isnan(fn(v)), msg=repr(v))
        for fn in (sw_asin_f32, sw_acos_f32):
            for v in (2.0, -2.0, math.inf, math.nan):
                self.assertTrue(math.isnan(float(fn(spy_as(v, f32)))), msg=repr(v))


class SpyAsinAcosModeTest(_TrigUlpTest):
    """``asin_mode``/``acos_mode`` select the ``llvm``/``libc``/``software``
    implementation at compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_asin_llvm, math.asin, 2, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_asin_llvm, math.asin, 2, _ASIN_ACOS_F32_VALUES)
        self.check_f64(libm_acos_llvm, math.acos, 2, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_acos_llvm, math.acos, 2, _ASIN_ACOS_F32_VALUES)

    def test_libc(self) -> None:
        self.check_f64(libm_asin_libc, math.asin, 2, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_asin_libc, math.asin, 2, _ASIN_ACOS_F32_VALUES)
        self.check_f64(libm_acos_libc, math.acos, 2, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_acos_libc, math.acos, 2, _ASIN_ACOS_F32_VALUES)

    def test_software(self) -> None:
        self.check_f64(libm_asin_software, math.asin, 1, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_asin_software, math.asin, 1, _ASIN_ACOS_F32_VALUES)
        self.check_f64(libm_acos_software, math.acos, 1, _ASIN_ACOS_F64_VALUES)
        self.check_f32(libm_acos_software, math.acos, 1, _ASIN_ACOS_F32_VALUES)

    def test_default_is_software(self) -> None:
        # the software implementation is deterministic, so the default (an unset
        # option) has to match it bit for bit
        for v in _ASIN_ACOS_F64_VALUES:
            self.assertEqual(libm_asin_default(v), sw_asin_f64(v), msg=repr(v))
            self.assertEqual(libm_acos_default(v), sw_acos_f64(v), msg=repr(v))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_asin_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_acos_bogus(0.5)


class SpySoftwareAtanAtan2Test(_TrigUlpTest):
    """The pure spy ``atan``/``atan2`` (the musl port): both ``f32`` and
    ``f64`` are within 1 ULP of the correctly rounded result, and the special
    values (the signed zeros, the quadrants, the infinities and a NaN) come out
    as C's do."""

    def test_atan_f64(self) -> None:
        self.check_f64(sw_atan_f64, math.atan, 1, _ATAN_F64_VALUES)

    def test_atan_f32(self) -> None:
        self.check_f32(sw_atan_f32, math.atan, 1, _ATAN_F32_VALUES)

    def test_atan2_f64(self) -> None:
        self.check_atan2_f64(sw_atan2_f64, 1)

    def test_atan2_f32(self) -> None:
        self.check_atan2_f32(sw_atan2_f32, 1)

    def test_atan_of_a_zero_keeps_its_sign(self) -> None:
        self.assertEqual(sw_atan_f64(0.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_atan_f64(-0.0)), -1.0)
        self.assertEqual(math.copysign(1.0, float(sw_atan_f32(spy_as(-0.0, f32)))), -1.0)

    def test_atan_of_a_nan(self) -> None:
        self.assertTrue(math.isnan(sw_atan_f64(math.nan)))
        self.assertTrue(math.isnan(float(sw_atan_f32(spy_as(math.nan, f32)))))

    def test_atan_is_odd(self) -> None:
        for v in _ATAN_F64_VALUES:
            if v > 0.0 and math.isfinite(v):
                self.assertEqual(sw_atan_f64(-v), -sw_atan_f64(v), msg=repr(v))

    def test_atan2_of_the_signed_zeros(self) -> None:
        cases = [
            (0.0, 1.0, 0.0), (-0.0, 1.0, -0.0),
            (0.0, -1.0, math.pi), (-0.0, -1.0, -math.pi),
        ]
        for y, x, want in cases:
            got = sw_atan2_f64(y, x)
            self.assertEqual(got, want, msg=repr((y, x)))
            if want == 0.0:
                self.assertEqual(math.copysign(1.0, got), math.copysign(1.0, want), msg=repr((y, x)))
        for y, x, want in cases:
            got32 = float(sw_atan2_f32(spy_as(y, f32), spy_as(x, f32)))
            self.assertEqual(got32, _to_f32(want), msg=repr((y, x)))
            if want == 0.0:
                self.assertEqual(math.copysign(1.0, got32), math.copysign(1.0, want), msg=repr((y, x)))

    def test_atan2_of_zero_x(self) -> None:
        for y in (1.0, -1.0, 1e300, -1e300):
            self.assertLessEqual(
                _f64_ulp(sw_atan2_f64(y, 0.0), math.atan2(y, 0.0)), 1, msg=repr(y)
            )

    def test_atan2_of_infinities(self) -> None:
        for y, x in ((math.inf, math.inf), (-math.inf, math.inf),
                     (math.inf, -math.inf), (-math.inf, -math.inf),
                     (1.0, math.inf), (-1.0, math.inf),
                     (1.0, -math.inf), (-1.0, -math.inf),
                     (math.inf, 1.0), (-math.inf, 1.0),
                     (math.inf, -1.0), (-math.inf, -1.0)):
            self.assertLessEqual(
                _f64_ulp(sw_atan2_f64(y, x), math.atan2(y, x)), 1, msg=repr((y, x))
            )

    def test_atan2_of_a_nan(self) -> None:
        for y, x in ((math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan)):
            self.assertTrue(math.isnan(sw_atan2_f64(y, x)), msg=repr((y, x)))
        for y, x in ((math.nan, 1.0), (1.0, math.nan)):
            self.assertTrue(
                math.isnan(float(sw_atan2_f32(spy_as(y, f32), spy_as(x, f32)))),
                msg=repr((y, x)),
            )


class SpyAtanAtan2ModeTest(_TrigUlpTest):
    """``atan_mode``/``atan2_mode`` select the ``llvm``/``libc``/``software``
    implementation at compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_atan_llvm, math.atan, 2, _ATAN_F64_VALUES)
        self.check_f32(libm_atan_llvm, math.atan, 2, _ATAN_F32_VALUES)
        self.check_atan2_f64(libm_atan2_llvm, 2)
        self.check_atan2_f32(libm_atan2_llvm, 2)

    def test_libc(self) -> None:
        self.check_f64(libm_atan_libc, math.atan, 2, _ATAN_F64_VALUES)
        self.check_f32(libm_atan_libc, math.atan, 2, _ATAN_F32_VALUES)
        self.check_atan2_f64(libm_atan2_libc, 2)
        self.check_atan2_f32(libm_atan2_libc, 2)

    def test_software(self) -> None:
        self.check_f64(libm_atan_software, math.atan, 1, _ATAN_F64_VALUES)
        self.check_f32(libm_atan_software, math.atan, 1, _ATAN_F32_VALUES)
        self.check_atan2_f64(libm_atan2_software, 1)
        self.check_atan2_f32(libm_atan2_software, 1)

    def test_default_is_software(self) -> None:
        # the software implementation is deterministic, so the default (an unset
        # option) has to match it bit for bit
        for v in _ATAN_F64_VALUES:
            self.assertEqual(libm_atan_default(v), sw_atan_f64(v), msg=repr(v))
        for y, x in _ATAN2_F64_VALUES:
            self.assertEqual(libm_atan2_default(y, x), sw_atan2_f64(y, x), msg=repr((y, x)))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_atan_bogus(0.5)
        with self.assertRaises(CompileError):
            libm_atan2_bogus(0.5, 0.5)


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


class SpySoftwarePowTest(_TrigUlpTest):
    """The pure spy ``pow`` (the musl port): both ``f32`` and ``f64`` are within
    1 ULP of the correctly rounded result (musl bounds the error at < 0.54 ULP),
    and the special values (the zeros, ``+-inf``, a NaN, a negative base with a
    non-integer exponent, overflow and underflow) come out as C's do."""

    def test_pow_f64(self) -> None:
        self.check_pow_f64(sw_pow_f64, 1)

    def test_pow_f32(self) -> None:
        self.check_pow_f32(sw_pow_f32, 1)

    def test_exact_powers(self) -> None:
        for x, y, want in ((2.0, 10.0, 1024.0), (3.0, 4.0, 81.0), (5.0, 0.0, 1.0),
                           (0.5, 3.0, 0.125), (-8.0, 3.0, -512.0), (-2.0, 2.0, 4.0),
                           (16.0, 0.5, 4.0), (1000.0, 1.0, 1000.0)):
            self.assertEqual(sw_pow_f64(x, y), want, msg=repr((x, y)))

    def test_pow_of_zero_exponent(self) -> None:
        for x in (0.0, -0.0, 2.0, -3.0, math.inf, math.nan):
            self.assertEqual(sw_pow_f64(x, 0.0), 1.0, msg=repr(x))

    def test_pow_of_zero_base(self) -> None:
        self.assertEqual(sw_pow_f64(0.0, 2.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_pow_f64(-0.0, 3.0)), -1.0)

    def test_pow_of_negative_base(self) -> None:
        self.assertEqual(sw_pow_f64(-8.0, 3.0), -512.0)
        self.assertEqual(sw_pow_f64(-2.0, -3.0), -0.125)
        for x, y in ((-8.0, 0.5), (-2.0, 1.5), (-0.5, 0.25)):
            self.assertTrue(math.isnan(sw_pow_f64(x, y)), msg=repr((x, y)))

    def test_pow_of_infinite_exponent(self) -> None:
        self.assertEqual(sw_pow_f64(1.0, math.inf), 1.0)
        self.assertEqual(sw_pow_f64(-1.0, math.inf), 1.0)
        self.assertEqual(sw_pow_f64(2.0, math.inf), math.inf)
        self.assertEqual(sw_pow_f64(0.5, math.inf), 0.0)
        self.assertEqual(sw_pow_f64(2.0, -math.inf), 0.0)
        self.assertEqual(sw_pow_f64(0.5, -math.inf), math.inf)

    def test_pow_of_infinite_base(self) -> None:
        self.assertEqual(sw_pow_f64(math.inf, 2.0), math.inf)
        self.assertEqual(sw_pow_f64(math.inf, -2.0), 0.0)
        self.assertEqual(sw_pow_f64(math.inf, 0.5), math.inf)
        self.assertEqual(sw_pow_f64(-math.inf, 3.0), -math.inf)

    def test_pow_overflow_and_underflow(self) -> None:
        self.assertEqual(sw_pow_f64(1e200, 2.0), math.inf)
        self.assertEqual(sw_pow_f64(1e-200, 2.0), 0.0)
        self.assertEqual(sw_pow_f64(2.0, 3000.0), math.inf)
        self.assertEqual(sw_pow_f64(2.0, -3000.0), 0.0)
        self.assertEqual(sw_pow_f64(-1e200, 3.0), -math.inf)
        self.assertEqual(sw_pow_f64(-1e200, 2.0), math.inf)


class SpyPowModeTest(_TrigUlpTest):
    """``pow_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_pow_f64(libm_pow_llvm, 1)
        self.check_pow_f32(libm_pow_llvm, 1)

    def test_libc(self) -> None:
        self.check_pow_f64(libm_pow_libc, 1)
        self.check_pow_f32(libm_pow_libc, 1)

    def test_software(self) -> None:
        self.check_pow_f64(libm_pow_software, 1)
        self.check_pow_f32(libm_pow_software, 1)

    def test_default_is_software(self) -> None:
        # the software implementation is deterministic, so the default (an unset
        # option) has to match it bit for bit
        for x, y in _POW_F64_VALUES:
            self.assertEqual(libm_pow_default(x, y), sw_pow_f64(x, y), msg=repr((x, y)))

    def test_mixed_float_types(self) -> None:
        # two differing float types meet at their common type (f64 here)
        self.assertEqual(libm_pow_software(2.0, spy_as(3.0, f32)), 8.0)
        self.assertEqual(libm_pow_software(spy_as(2.0, f32), 3.0), 8.0)

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_pow_bogus(2.0, 3.0)


class SpySoftwareExpTest(_TrigUlpTest):
    """The pure spy ``exp`` (the musl port): both ``f32`` and ``f64`` are within
    1 ULP of the correctly rounded result, and the special values (the zeros,
    ``+-inf``, a NaN, overflow and underflow) come out as C's do."""

    def test_exp_f64(self) -> None:
        self.check_f64(sw_exp_f64, math.exp, 1, _EXP_F64_VALUES)

    def test_exp_f32(self) -> None:
        self.check_f32(sw_exp_f32, math.exp, 1, _EXP_F32_VALUES)

    def test_exp_of_zero(self) -> None:
        self.assertEqual(sw_exp_f64(0.0), 1.0)
        self.assertEqual(sw_exp_f64(-0.0), 1.0)

    def test_exp_of_inf_and_nan(self) -> None:
        self.assertEqual(sw_exp_f64(math.inf), math.inf)
        self.assertEqual(sw_exp_f64(-math.inf), 0.0)
        self.assertTrue(math.isnan(sw_exp_f64(math.nan)))

    def test_exp_overflow_and_underflow(self) -> None:
        self.assertEqual(sw_exp_f64(1000.0), math.inf)
        self.assertEqual(sw_exp_f64(-1000.0), 0.0)
        self.assertEqual(float(sw_exp_f32(spy_as(100.0, f32))), math.inf)
        self.assertEqual(float(sw_exp_f32(spy_as(-110.0, f32))), 0.0)

    def test_exp_f32_of_inf_and_nan(self) -> None:
        self.assertEqual(float(sw_exp_f32(spy_as(math.inf, f32))), math.inf)
        self.assertEqual(float(sw_exp_f32(spy_as(-math.inf, f32))), 0.0)
        self.assertTrue(math.isnan(float(sw_exp_f32(spy_as(math.nan, f32)))))


class SpyExpModeTest(_TrigUlpTest):
    """``exp_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_exp_llvm, math.exp, 1, _EXP_F64_VALUES)
        self.check_f32(libm_exp_llvm, math.exp, 1, _EXP_F32_VALUES)

    def test_libc(self) -> None:
        self.check_f64(libm_exp_libc, math.exp, 1, _EXP_F64_VALUES)
        self.check_f32(libm_exp_libc, math.exp, 1, _EXP_F32_VALUES)

    def test_software(self) -> None:
        self.check_f64(libm_exp_software, math.exp, 1, _EXP_F64_VALUES)
        self.check_f32(libm_exp_software, math.exp, 1, _EXP_F32_VALUES)

    def test_default_is_software(self) -> None:
        for v in _EXP_F64_VALUES:
            self.assertEqual(libm_exp_default(v), sw_exp_f64(v), msg=repr(v))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_exp_bogus(1.0)


class SpySoftwareLogTest(_TrigUlpTest):
    """The pure spy ``log`` (the musl port): both ``f32`` and ``f64`` are within
    1 ULP of the correctly rounded result, and the special values (the zeros,
    ``+-inf``, a NaN, a negative argument and subnormals) come out as C's do."""

    def test_log_f64(self) -> None:
        self.check_f64(sw_log_f64, math.log, 1, _LOG_F64_VALUES)

    def test_log_f32(self) -> None:
        self.check_f32(sw_log_f32, math.log, 1, _LOG_F32_VALUES)

    def test_log_of_one(self) -> None:
        self.assertEqual(sw_log_f64(1.0), 0.0)
        self.assertEqual(math.copysign(1.0, sw_log_f64(1.0)), 1.0)

    def test_log_of_zero(self) -> None:
        for v in (0.0, -0.0):
            self.assertEqual(sw_log_f64(v), -math.inf, msg=repr(v))

    def test_log_of_inf_and_nan(self) -> None:
        self.assertEqual(sw_log_f64(math.inf), math.inf)
        self.assertTrue(math.isnan(sw_log_f64(math.nan)))

    def test_log_of_negative_is_nan(self) -> None:
        for v in (-1.0, -0.5, -5e-324, -1e300):
            self.assertTrue(math.isnan(sw_log_f64(v)), msg=repr(v))
        for v in (-1.0, -0.5, -1e30):
            self.assertTrue(math.isnan(float(sw_log_f32(spy_as(v, f32)))), msg=repr(v))

    def test_log_of_subnormal(self) -> None:
        self.assertLessEqual(
            _f64_ulp(sw_log_f64(5e-324), math.log(5e-324)), 1
        )
        self.assertLessEqual(
            _f32_ulp(float(sw_log_f32(spy_as(1.4e-45, f32))), _to_f32(math.log(_to_f32(1.4e-45)))), 1
        )

    def test_log_f32_special_values(self) -> None:
        for v in (0.0, -0.0):
            self.assertEqual(float(sw_log_f32(spy_as(v, f32))), -math.inf, msg=repr(v))
        self.assertEqual(float(sw_log_f32(spy_as(math.inf, f32))), math.inf)
        self.assertTrue(math.isnan(float(sw_log_f32(spy_as(math.nan, f32)))))
        self.assertTrue(math.isnan(float(sw_log_f32(spy_as(-1.0, f32)))))


class SpyLogModeTest(_TrigUlpTest):
    """``log_mode`` selects the ``llvm``/``libc``/``software`` implementation at
    compile time; an unset option defaults to ``software``."""

    def test_llvm(self) -> None:
        self.check_f64(libm_log_llvm, math.log, 1, _LOG_F64_VALUES)
        self.check_f32(libm_log_llvm, math.log, 1, _LOG_F32_VALUES)

    def test_libc(self) -> None:
        self.check_f64(libm_log_libc, math.log, 1, _LOG_F64_VALUES)
        self.check_f32(libm_log_libc, math.log, 1, _LOG_F32_VALUES)

    def test_software(self) -> None:
        self.check_f64(libm_log_software, math.log, 1, _LOG_F64_VALUES)
        self.check_f32(libm_log_software, math.log, 1, _LOG_F32_VALUES)

    def test_default_is_software(self) -> None:
        for v in _LOG_F64_VALUES:
            self.assertEqual(libm_log_default(v), sw_log_f64(v), msg=repr(v))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            libm_log_bogus(1.0)


all_tests = [
    SpySoftwareTrigTest,
    SpyTrigModeTest,
    SpySoftwareAsinAcosTest,
    SpyAsinAcosModeTest,
    SpySoftwareSqrtTest,
    SpySqrtModeTest,
    SpySoftwareCbrtTest,
    SpyCbrtModeTest,
    SpySoftwarePowTest,
    SpyPowModeTest,
    SpySoftwareExpTest,
    SpyExpModeTest,
    SpySoftwareLogTest,
    SpyLogModeTest,
]

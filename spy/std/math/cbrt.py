"""The software (pure spy) cube root (``std.math``).

A port of musl's ``cbrt.c`` and ``cbrtf.c``.  The argument is reduced to a 5-bit
estimate ``t`` of ``cbrt(x)`` by folding the biased exponent into the bits and
dividing it by three (``cbrt(2**e * (1+m)) ~= 2**(e/3) * ...``); ``t`` is then
refined:

* ``f64``: a degree-4 polynomial estimates ``1/cbrt(x/t**3)`` to 23 bits, the
  estimate is rounded away from zero and a single Newton iteration reaches the
  result;
* ``f32``: the rough estimate is fed to two Newton iterations carried out in
  ``f64`` (musl's ``cbrtf`` does them in ``double_t``), the last being correctly
  rounded to ``f32``.

Where C needs a separate ``float`` and ``double`` copy, a single function with a
compile-time ``is_32`` flag serves both: only the branch of the requested float
type is ever typed and lowered.

The public entry point is :func:`cbrt`, generic over ``f32``/``f64``.  It is
what ``std.math`` calls when the compile-time option ``cbrt_mode`` is
``software`` (the default).

Unlike ``sqrt``, the result is not guaranteed to be correctly rounded: musl
bounds the error of both ``cbrt`` and ``cbrtf`` at less than 0.667 ULP.  ``cbrt``
is odd (``cbrt(-x) == -cbrt(x)``, so negative arguments are computed, not a
NaN), ``NaN``/``+-inf`` come out unchanged and ``cbrt(+-0)`` is ``+-0``.
"""

from ...compiler import f32, f64, typeof, u32, u64
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..core import Numeric, bitcast, coerce, compile_error

# ``cbrt.c`` (f64): the exponent bias folded into the bits for the rough
# estimate, and the polynomial approximating ``1/cbrt(r)`` with error < 2**-23.5.
_B1_F64 = 715094163
_B2_F64 = 696219795

_P0 = 1.87595182427177009643
_P1 = -1.88497979543377169875
_P2 = 1.621429720105354466140
_P3 = -0.758397934778766047437
_P4 = 0.145996192886612446982

# ``cbrtf.c`` (f32)
_B1_F32 = 709958130
_B2_F32 = 642849266

# the powers of two the subnormal arguments are scaled by (``0x1p54``/``0x1p24f``)
_SCALE_F64 = 18014398509481984.0
_SCALE_F32 = 16777216.0


@func()
def cbrt[T: Numeric](x: T) -> T:
    """The cube root of ``x`` (``f32`` or ``f64``), the pure spy implementation:
    a port of musl's ``cbrtf``/``cbrt``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        ix = bitcast(x, u32)
        hx = ix & 0x7FFFFFFF
        if hx >= 0x7F800000:
            # cbrt(NaN, +-inf) is itself
            return x + x
        if hx < 0x00800000:
            # zero or subnormal?
            if hx == 0:
                return x  # cbrt(+-0) is itself
            # scale it into the normal range first
            hx = ((bitcast(coerce(x, f32) * coerce(_SCALE_F32, f32), u32) & 0x7FFFFFFF) // 3) + _B2_F32
        else:
            hx = (hx // 3) + _B1_F32
        t = coerce(bitcast((ix & 0x80000000) | hx, f32), f64)

        # two Newton iterations carried out in double precision
        xd = coerce(x, f64)
        r = t * t * t
        t = t * (xd + xd + r) / (xd + r + r)
        r = t * t * t
        t = t * (xd + xd + r) / (xd + r + r)

        # rounding to 24 bits is perfect in round-to-nearest mode
        return coerce(t, typ)
    elif typ == f64:
        xd = coerce(x, f64)
        ix = bitcast(x, u64)
        hx = coerce(ix >> 32, u32) & 0x7FFFFFFF
        if hx >= 0x7FF00000:
            # cbrt(NaN, +-inf) is itself
            return x + x
        if hx < 0x00100000:
            # zero or subnormal?
            scaled = xd * coerce(_SCALE_F64, f64)
            hx = coerce(bitcast(scaled, u64) >> 32, u32) & 0x7FFFFFFF
            if hx == 0:
                return x  # cbrt(+-0) is itself
            hx = (hx // 3) + _B2_F64
        else:
            hx = (hx // 3) + _B1_F64
        u = (ix & 0x8000000000000000) | (coerce(hx, u64) << 32)
        t = bitcast(u, f64)

        # rough cbrt to 5 bits, refined to 23 bits by the polynomial
        r = (t * t) * (t / xd)
        t = t * ((_P0 + r * (_P1 + r * _P2)) + ((r * r) * r) * (_P3 + r * _P4))

        # round t away from zero to 23 bits
        u = (bitcast(t, u64) + 0x80000000) & 0xFFFFFFFFC0000000
        t = bitcast(u, f64)

        # one Newton iteration to 53 bits with error < 0.667 ulps
        s = t * t
        r = xd / s
        w = t + t
        r = (r - t) / (w + r)
        return coerce(t + t * r, typ)
    compile_error("only f32 or f64")

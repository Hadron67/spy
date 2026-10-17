"""The software (pure spy) inverse trigonometric functions (``std.math``).

A port of musl's ``asin.c``/``asinf.c`` and ``acos.c``/``acosf.c``.  Both rest on
the same rational approximation ``R`` of ``(asin(x)-x)/x**3``:

* ``asin`` is ``x + x*x**2*R(x**2)`` for ``|x| < 0.5`` and the reduction
  ``asin(x) = pi/2 - 2*asin(sqrt((1-|x|)/2))`` beyond it (with ``pi/2`` split as
  ``pio2_hi + pio2_lo`` so the last bits survive, and a ``pio4`` variant for
  ``|x| <= 0.975``);
* ``acos`` is ``pi/2 - asin(x)``, so it shares the coefficients and reduces
  ``|x| >= 0.5`` through the two branches of the same quadrant.

``R`` comes in an ``f64`` and a narrower ``f32`` coefficient set.  Like the
``f32`` kernels of ``std.math.trig``, an ``f32`` path evaluates the polynomial in
``f64`` with the ``f32`` coefficients and rounds once, at the end, to ``f32``
(the reduced argument is still rounded to ``f32`` where musl keeps a ``float``).

Where C needs a separate ``float`` and ``double`` copy, a single function
branching on the requested float type serves both: only the selected branch is
ever typed and lowered.

The public entry points are :func:`asin` and :func:`acos`, generic over
``f32``/``f64``.  They are what ``std.math`` calls when the compile-time option
``asin_mode``/``acos_mode`` is ``software`` (the default).

The one deviation from musl is its floating point environment handling: spy has
no ``fenv``, so the ``0x1p-120`` tail that flags an inexact ``asin(+-1)`` /
``acos(+-1)`` -- far below the result's ulp -- is dropped.
"""

from ...compiler import f32, f64, typeof, u32, u64
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..core import Numeric, bitcast, coerce, compile_error
from .sqrt import _invalid_f32, _invalid_f64
from .sqrt import sqrt as _sqrt

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# ``asin.c``/``acos.c`` (f64): the split ``pi/2`` and the coefficients of
# ``R(x**2)`` (their remez error is bounded by 2**-58.75)
_PIO2_HI = 1.57079632679489655800e+00
_PIO2_LO = 6.12323399573676603587e-17

_PS0 = 1.66666666666666657415e-01
_PS1 = -3.25565818622400915405e-01
_PS2 = 2.01212532134862925881e-01
_PS3 = -4.00555345006794114027e-02
_PS4 = 7.91534994289814532176e-04
_PS5 = 3.47933107596021167570e-05
_QS1 = -2.40339491173441421878e+00
_QS2 = 2.02094576023350569471e+00
_QS3 = -6.88283971605453293030e-01
_QS4 = 7.70381505559019352791e-02

# ``asinf.c``/``acosf.c`` (f32).  ``asinf``'s ``pio2`` is ``_PIO2_HI`` itself;
# ``acosf`` splits its own ``f32`` ``pi/2``.
_PIO2_HI_F = 1.5707962513e+00
_PIO2_LO_F = 7.5497894159e-08

# the coefficients of the ``f32`` ``R(x**2)``
_PS0F = 1.6666586697e-01
_PS1F = -4.2743422091e-02
_PS2F = -8.6563630030e-03
_QS1F = -7.0662963390e-01


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@func(inline=True)
def _fabs64(x: f64) -> f64:
    """``fabs(x)``: spy has no ``fabs`` builtin, so the sign bit is masked off by
    hand (``fabsf`` of an ``f32`` is the same, on the value promoted to
    ``f64``)."""
    return bitcast(bitcast(x, u64) & 0x7FFFFFFFFFFFFFFF, f64)


@func(inline=True)
def _clear_low32(x: f32) -> f32:
    """``SET_FLOAT_WORD(x, hx & 0xfffff000)``: an ``f32`` with its low 12 bits
    dropped."""
    return bitcast(bitcast(x, u32) & 0xFFFFF000, f32)


@func(inline=True)
def _clear_low64(x: f64) -> f64:
    """``SET_LOW_WORD(x, 0)``: an ``f64`` with its low 32 bits dropped."""
    return bitcast(bitcast(x, u64) & 0xFFFFFFFF00000000, f64)


@func()
def _r64(z: f64) -> f64:
    """``R(z)`` of ``asin.c``/``acos.c``: the rational approximation of
    ``(asin(x)-x)/x**3`` at ``z = x*x``."""
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    return p / q


@func()
def _r32(z: f64) -> f64:
    """``R(z)`` of ``asinf.c``/``acosf.c``: the same approximation with the
    ``f32`` coefficient set, evaluated in ``f64``."""
    p = z * (_PS0F + z * (_PS1F + z * _PS2F))
    q = 1.0 + z * _QS1F
    return p / q


# ---------------------------------------------------------------------------
# the public entry points
# ---------------------------------------------------------------------------


@func()
def asin[T: Numeric](x: T) -> T:
    """The arc sine of ``x`` in ``[-pi/2, pi/2]`` (``f32`` or ``f64``, the result
    of the same type), the pure spy implementation: a port of musl's
    ``asinf``/``asin``.  ``asin(+-1)`` is ``+-pi/2``, ``|x| > 1`` (and a NaN) is
    a NaN, and the sign of a zero survives."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        bits = bitcast(x, u32)
        ix = bits & 0x7FFFFFFF
        if ix >= 0x3F800000:
            if ix == 0x3F800000:
                # asin(+-1) = +-pi/2
                return coerce(coerce(x, f64) * _PIO2_HI, typ)
            return coerce(_invalid_f32(coerce(x, f32)), typ)
        if ix < 0x3F000000:
            # |x| < 0.5: a small |x| from 2**-126 to 2**-12 yields x itself, so
            # no underflow is raised
            if ix < 0x39800000 and ix >= 0x00800000:
                return x
            xd = coerce(x, f64)
            return coerce(xd + xd * _r32(xd * xd), typ)
        # 1 > |x| >= 0.5
        xd = coerce(x, f64)
        z = coerce((1.0 - _fabs64(xd)) * 0.5, f32)
        zf = coerce(z, f64)
        s = _sqrt(zf)
        res = _PIO2_HI - 2.0 * (s + s * _r32(zf))
        if (bits >> 31) != 0:
            res = -res
        return coerce(res, typ)
    elif typ == f64:
        bits = bitcast(x, u64)
        hx = coerce(bits >> 32, u32)
        ix = hx & 0x7FFFFFFF
        if ix >= 0x3FF00000:
            if ((ix - 0x3FF00000) | coerce(bits & 0xFFFFFFFF, u32)) == 0:
                # asin(+-1) = +-pi/2
                return coerce(coerce(x, f64) * _PIO2_HI, typ)
            return coerce(_invalid_f64(coerce(x, f64)), typ)
        if ix < 0x3FE00000:
            # |x| < 0.5: a small |x| from 2**-1022 to 2**-26 yields x itself, so
            # no underflow is raised
            if ix < 0x3E500000 and ix >= 0x00100000:
                return x
            xd = coerce(x, f64)
            return coerce(xd + xd * _r64(xd * xd), typ)
        # 1 > |x| >= 0.5
        xd = coerce(x, f64)
        z = (1.0 - _fabs64(xd)) * 0.5
        s = _sqrt(z)
        r = _r64(z)
        res: f64 = 0.0
        if ix >= 0x3FEF3333:
            # |x| > 0.975: the tail of pi/2 does not need the f+c split
            res = _PIO2_HI - (2.0 * (s + s * r) - _PIO2_LO)
        else:
            f = _clear_low64(s)
            c = (z - f * f) / (s + f)
            res = 0.5 * _PIO2_HI - (2.0 * s * r - (_PIO2_LO - 2.0 * c) - (0.5 * _PIO2_HI - 2.0 * f))
        if (hx >> 31) != 0:
            res = -res
        return coerce(res, typ)
    compile_error("only f32 or f64")


@func()
def acos[T: Numeric](x: T) -> T:
    """The arc cosine of ``x`` in ``[0, pi]`` (``f32`` or ``f64``, the result of
    the same type), the pure spy implementation: a port of musl's
    ``acosf``/``acos``.  ``acos(1)`` is ``0``, ``acos(-1)`` is ``pi``, ``|x| > 1``
    (and a NaN) is a NaN, and ``acos(+-0)`` is ``pi/2``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        bits = bitcast(x, u32)
        ix = bits & 0x7FFFFFFF
        if ix >= 0x3F800000:
            if ix == 0x3F800000:
                # acos(1) = 0, acos(-1) = pi
                if (bits >> 31) != 0:
                    return coerce(2.0 * _PIO2_HI_F, typ)
                return coerce(0.0, typ)
            return coerce(_invalid_f32(coerce(x, f32)), typ)
        if ix < 0x3F000000:
            # |x| < 0.5
            if ix <= 0x32800000:
                # |x| < 2**-26
                return coerce(_PIO2_HI_F, typ)
            xd = coerce(x, f64)
            return coerce(_PIO2_HI_F - (xd - (_PIO2_LO_F - xd * _r32(xd * xd))), typ)
        xd = coerce(x, f64)
        if (bits >> 31) != 0:
            # x < -0.5
            z = coerce((1.0 + xd) * 0.5, f32)
            zf = coerce(z, f64)
            s = _sqrt(zf)
            w = _r32(zf) * s - _PIO2_LO_F
            return coerce(2.0 * (_PIO2_HI_F - (s + w)), typ)
        # x > 0.5
        z = coerce((1.0 - xd) * 0.5, f32)
        zf = coerce(z, f64)
        s32 = _sqrt(z)
        s = coerce(s32, f64)
        df = coerce(_clear_low32(s32), f64)
        c = (zf - df * df) / (s + df)
        w = _r32(zf) * s + c
        return coerce(2.0 * (df + w), typ)
    elif typ == f64:
        bits = bitcast(x, u64)
        hx = coerce(bits >> 32, u32)
        ix = hx & 0x7FFFFFFF
        if ix >= 0x3FF00000:
            if ((ix - 0x3FF00000) | coerce(bits & 0xFFFFFFFF, u32)) == 0:
                # acos(1) = 0, acos(-1) = pi
                if (hx >> 31) != 0:
                    return coerce(2.0 * _PIO2_HI, typ)
                return coerce(0.0, typ)
            return coerce(_invalid_f64(coerce(x, f64)), typ)
        if ix < 0x3FE00000:
            # |x| < 0.5
            if ix <= 0x3C600000:
                # |x| < 2**-57
                return coerce(_PIO2_HI, typ)
            xd = coerce(x, f64)
            return coerce(_PIO2_HI - (xd - (_PIO2_LO - xd * _r64(xd * xd))), typ)
        xd = coerce(x, f64)
        if (hx >> 31) != 0:
            # x < -0.5
            z = (1.0 + xd) * 0.5
            s = _sqrt(z)
            w = _r64(z) * s - _PIO2_LO
            return coerce(2.0 * (_PIO2_HI - (s + w)), typ)
        # x > 0.5
        z = (1.0 - xd) * 0.5
        s = _sqrt(z)
        df = _clear_low64(s)
        c = (z - df * df) / (s + df)
        w = _r64(z) * s + c
        return coerce(2.0 * (df + w), typ)
    compile_error("only f32 or f64")

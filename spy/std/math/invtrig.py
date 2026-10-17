"""The software (pure spy) inverse trigonometric functions (``std.math``).

A port of musl's ``asin.c``/``asinf.c``, ``acos.c``/``acosf.c``,
``atan.c``/``atanf.c`` and ``atan2.c``/``atan2f.c``.

``asin`` and ``acos`` rest on the same rational approximation ``R`` of
``(asin(x)-x)/x**3``:

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

``atan`` reduces the argument to one of five intervals (``|x| < 0.4375`` and the
four ranges above it) and evaluates a degree-23 odd polynomial there, adding the
tabulated ``atan(0.5)``/``atan(1)``/``atan(1.5)``/``pi/2`` offset when needed.
``atan2`` handles the quadrants and the special values of ``(y, x)`` itself and
falls back to ``atan`` for the reduced ratio ``|y/x|``.

Where C needs a separate ``float`` and ``double`` copy, a single function
branching on the requested float type serves both: only the selected branch is
ever typed and lowered.

The public entry points are :func:`asin`, :func:`acos`, :func:`atan` and
:func:`atan2`, generic over ``f32``/``f64``.  They are what ``std.math`` calls
when the compile-time option ``asin_mode``/``acos_mode``/``atan_mode``/
``atan2_mode`` is ``software`` (the default).

The one deviation from musl is its floating point environment handling: spy has
no ``fenv``, so the ``0x1p-120`` tail that flags an inexact ``asin(+-1)`` /
``acos(+-1)`` and the one that makes ``atan(+-inf)`` inexact -- both far below
the result's ulp -- are dropped.
"""

from ...compiler import f32, f64, i32, typeof, u32, u64
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..core import Numeric, bitcast, coerce, compile_error
from ..float import float_truncate
from ..int import truncate
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

# ``atan.c``/``atanf.c``: ``atan(0.5)``/``atan(1)``/``atan(1.5)`` as a
# high/low pair.  Entry 3 is ``pi/2``, i.e. ``_PIO2_HI``/``_PIO2_LO`` for
# ``f64`` and ``_PIO2_HI_F``/``_PIO2_LO_F`` for ``f32``.
_ATANHI0 = 4.63647609000806093515e-01
_ATANHI1 = 7.85398163397448278999e-01
_ATANHI2 = 9.82793723247329054082e-01
_ATANLO0 = 2.26987774529616870924e-17
_ATANLO1 = 3.06161699786838301793e-17
_ATANLO2 = 1.39033110312309984516e-17

_ATANHI0_F32 = 4.6364760399e-01
_ATANHI1_F32 = 7.8539812565e-01
_ATANHI2_F32 = 9.8279368877e-01
_ATANLO0_F32 = 5.0121582440e-09
_ATANLO1_F32 = 3.7748947079e-08
_ATANLO2_F32 = 3.4473217170e-08

# the odd polynomial of ``atan.c`` (f64) and ``atanf.c`` (f32)
_AT0 = 3.33333333333329318027e-01
_AT1 = -1.99999999998764832476e-01
_AT2 = 1.42857142725034663711e-01
_AT3 = -1.11111104054623557880e-01
_AT4 = 9.09088713343650656196e-02
_AT5 = -7.69187620504482999495e-02
_AT6 = 6.66107313738753120669e-02
_AT7 = -5.83357013379057348645e-02
_AT8 = 4.97687799461593236017e-02
_AT9 = -3.65315727442169155270e-02
_AT10 = 1.62858201153657823623e-02

_AT0_F32 = 3.3333328366e-01
_AT1_F32 = -1.9999158382e-01
_AT2_F32 = 1.4253635705e-01
_AT3_F32 = -1.0648017377e-01
_AT4_F32 = 6.1687607318e-02

# ``atan2.c``/``atan2f.c``: ``pi`` and its low part (the ``f32`` one is
# negative).
_PI = 3.1415926535897931160e+00
_PI_LO = 1.2246467991473531772e-16
_PI_F32 = 3.1415927410e+00
_PI_LO_F32 = -8.7422776573e-08


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


@func(inline=True)
def _neg_f32(x: f32) -> f32:
    """``-x`` with the sign of a zero kept (spy lowers a unary ``-`` on a float
    as a subtraction from zero, which would turn ``-0.0`` into ``+0.0``)."""
    return bitcast(bitcast(x, u32) ^ 0x80000000, f32)


@func(inline=True)
def _neg_f64(x: f64) -> f64:
    """The ``f64`` counterpart of :func:`_neg_f32`."""
    return bitcast(bitcast(x, u64) ^ 0x8000000000000000, f64)


@func()
def _atan_hi(id: i32) -> f64:
    """``atanhi[id]`` of ``atan.c`` (entry 3 is ``pi/2``)."""
    res: f64 = 0.0
    if id == 0:
        res = _ATANHI0
    elif id == 1:
        res = _ATANHI1
    elif id == 2:
        res = _ATANHI2
    else:
        res = _PIO2_HI
    return res


@func()
def _atan_lo(id: i32) -> f64:
    """``atanlo[id]`` of ``atan.c``."""
    res: f64 = 0.0
    if id == 0:
        res = _ATANLO0
    elif id == 1:
        res = _ATANLO1
    elif id == 2:
        res = _ATANLO2
    else:
        res = _PIO2_LO
    return res


@func()
def _atan_hi_f32(id: i32) -> f64:
    """``atanhi[id]`` of ``atanf.c``, as a ``float`` value held in an
    ``f64``."""
    res: f64 = 0.0
    if id == 0:
        res = _ATANHI0_F32
    elif id == 1:
        res = _ATANHI1_F32
    elif id == 2:
        res = _ATANHI2_F32
    else:
        res = _PIO2_HI_F
    return res


@func()
def _atan_lo_f32(id: i32) -> f64:
    """``atanlo[id]`` of ``atanf.c``, as a ``float`` value held in an
    ``f64``."""
    res: f64 = 0.0
    if id == 0:
        res = _ATANLO0_F32
    elif id == 1:
        res = _ATANLO1_F32
    elif id == 2:
        res = _ATANLO2_F32
    else:
        res = _PIO2_LO_F
    return res


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
                return float_truncate(coerce(x, f64) * _PIO2_HI, typ)
            return coerce(_invalid_f32(coerce(x, f32)), typ)
        if ix < 0x3F000000:
            # |x| < 0.5: a small |x| from 2**-126 to 2**-12 yields x itself, so
            # no underflow is raised
            if ix < 0x39800000 and ix >= 0x00800000:
                return x
            xd = coerce(x, f64)
            return float_truncate(xd + xd * _r32(xd * xd), typ)
        # 1 > |x| >= 0.5
        xd = coerce(x, f64)
        z = float_truncate((1.0 - _fabs64(xd)) * 0.5, f32)
        zf = coerce(z, f64)
        s = _sqrt(zf)
        res = _PIO2_HI - 2.0 * (s + s * _r32(zf))
        if (bits >> 31) != 0:
            res = -res
        return float_truncate(res, typ)
    elif typ == f64:
        bits = bitcast(x, u64)
        hx = truncate(bits >> 32, u32)
        ix = hx & 0x7FFFFFFF
        if ix >= 0x3FF00000:
            if ((ix - 0x3FF00000) | truncate(bits & 0xFFFFFFFF, u32)) == 0:
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
            return float_truncate(_PIO2_HI_F - (xd - (_PIO2_LO_F - xd * _r32(xd * xd))), typ)
        xd = coerce(x, f64)
        if (bits >> 31) != 0:
            # x < -0.5
            z = float_truncate((1.0 + xd) * 0.5, f32)
            zf = coerce(z, f64)
            s = _sqrt(zf)
            w = _r32(zf) * s - _PIO2_LO_F
            return float_truncate(2.0 * (_PIO2_HI_F - (s + w)), typ)
        # x > 0.5
        z = float_truncate((1.0 - xd) * 0.5, f32)
        zf = coerce(z, f64)
        s32 = _sqrt(z)
        s = coerce(s32, f64)
        df = coerce(_clear_low32(s32), f64)
        c = (zf - df * df) / (s + df)
        w = _r32(zf) * s + c
        return float_truncate(2.0 * (df + w), typ)
    elif typ == f64:
        bits = bitcast(x, u64)
        hx = truncate(bits >> 32, u32)
        ix = hx & 0x7FFFFFFF
        if ix >= 0x3FF00000:
            if ((ix - 0x3FF00000) | truncate(bits & 0xFFFFFFFF, u32)) == 0:
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


@func()
def atan[T: Numeric](x: T) -> T:
    """The arc tangent of ``x`` in ``(-pi/2, pi/2)`` (``f32`` or ``f64``, the
    result of the same type), the pure spy implementation: a port of musl's
    ``atanf``/``atan``.  ``atan(+-0)`` keeps its sign, ``atan(+-inf)`` is
    ``+-pi/2``, and a NaN stays a NaN."""
    comptime()
    typ = typeof(x)
    xa: f64 = 0.0
    id: i32 = -1
    if typ == f32:
        bits = bitcast(x, u32)
        sign = bits >> 31
        ix = bits & 0x7FFFFFFF
        if ix >= 0x4C800000:
            # |x| >= 2**26 (or a NaN)
            if ix > 0x7F800000:
                return x
            if sign != 0:
                return coerce(-_PIO2_HI_F, typ)
            return coerce(_PIO2_HI_F, typ)
        xa = coerce(x, f64)
        id = -1
        if ix < 0x3EE00000:
            # |x| < 0.4375
            if ix < 0x39800000:
                # |x| < 2**-12
                return x
        else:
            xa = _fabs64(xa)
            if ix < 0x3F980000:
                # |x| < 1.1875
                if ix < 0x3F300000:
                    # 7/16 <= |x| < 11/16
                    id = 0
                    xa = (2.0 * xa - 1.0) / (2.0 + xa)
                else:
                    # 11/16 <= |x| < 19/16
                    id = 1
                    xa = (xa - 1.0) / (xa + 1.0)
            else:
                if ix < 0x401C0000:
                    # |x| < 2.4375
                    id = 2
                    xa = (xa - 1.5) / (1.0 + 1.5 * xa)
                else:
                    # 2.4375 <= |x| < 2**26
                    id = 3
                    xa = -1.0 / xa
        # the odd polynomial, split into its odd and even parts
        z = xa * xa
        w = z * z
        s1 = z * (_AT0_F32 + w * (_AT2_F32 + w * _AT4_F32))
        s2 = w * (_AT1_F32 + w * _AT3_F32)
        if id < 0:
            return float_truncate(xa - xa * (s1 + s2), typ)
        res = _atan_hi_f32(id) - ((xa * (s1 + s2) - _atan_lo_f32(id)) - xa)
        if sign != 0:
            res = -res
        return float_truncate(res, typ)
    elif typ == f64:
        bits = bitcast(x, u64)
        hx = truncate(bits >> 32, u32)
        sign = hx >> 31
        ix = hx & 0x7FFFFFFF
        lx = truncate(bits & 0xFFFFFFFF, u32)
        if ix >= 0x44100000:
            # |x| >= 2**66 (or a NaN)
            if ix > 0x7FF00000 or (ix == 0x7FF00000 and lx != 0):
                return x
            if sign != 0:
                return coerce(-_PIO2_HI, typ)
            return coerce(_PIO2_HI, typ)
        xa = coerce(x, f64)
        id = -1
        if ix < 0x3FDC0000:
            # |x| < 0.4375
            if ix < 0x3E400000:
                # |x| < 2**-27
                return x
        else:
            xa = _fabs64(xa)
            if ix < 0x3FF30000:
                # |x| < 1.1875
                if ix < 0x3FE60000:
                    # 7/16 <= |x| < 11/16
                    id = 0
                    xa = (2.0 * xa - 1.0) / (2.0 + xa)
                else:
                    # 11/16 <= |x| < 19/16
                    id = 1
                    xa = (xa - 1.0) / (xa + 1.0)
            else:
                if ix < 0x40038000:
                    # |x| < 2.4375
                    id = 2
                    xa = (xa - 1.5) / (1.0 + 1.5 * xa)
                else:
                    # 2.4375 <= |x| < 2**66
                    id = 3
                    xa = -1.0 / xa
        # the odd polynomial, split into its odd and even parts
        z = xa * xa
        w = z * z
        s1 = z * (_AT0 + w * (_AT2 + w * (_AT4 + w * (_AT6 + w * (_AT8 + w * _AT10)))))
        s2 = w * (_AT1 + w * (_AT3 + w * (_AT5 + w * (_AT7 + w * _AT9))))
        if id < 0:
            return coerce(xa - xa * (s1 + s2), typ)
        res = _atan_hi(id) - ((xa * (s1 + s2) - _atan_lo(id)) - xa)
        if sign != 0:
            res = -res
        return coerce(res, typ)
    compile_error("only f32 or f64")


@func()
def atan2[T: Numeric](y: T, x: T) -> T:
    """The arc tangent of ``y/x`` in ``[-pi, pi]``, using the signs of both
    arguments to select the quadrant (``f32`` or ``f64``, the result of their
    common type), the pure spy implementation: a port of musl's
    ``atan2f``/``atan2``.  The special values (the signed zeros, the
    infinities and a NaN) follow C's ``atan2``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        xb = bitcast(x, u32)
        yb = bitcast(y, u32)
        ix = xb & 0x7FFFFFFF
        iy = yb & 0x7FFFFFFF
        if ix > 0x7F800000 or iy > 0x7F800000:
            # a NaN in either argument
            return float_truncate(coerce(x, f64) + coerce(y, f64), typ)
        if xb == 0x3F800000:
            # x = 1.0
            return atan(y)
        m = ((yb >> 31) & 1) | ((xb >> 30) & 2)
        if iy == 0:
            # y = +-0
            if m == 0 or m == 1:
                return y
            elif m == 2:
                return coerce(_PI_F32, typ)
            else:
                return coerce(-_PI_F32, typ)
        if ix == 0:
            # x = +-0
            if (m & 1) != 0:
                return coerce(-_PIO2_HI_F, typ)
            return coerce(_PIO2_HI_F, typ)
        if ix == 0x7F800000:
            # x = +-inf
            if iy == 0x7F800000:
                # y = +-inf
                if m == 0:
                    return coerce(_PI_F32 / 4.0, typ)
                elif m == 1:
                    return coerce(-_PI_F32 / 4.0, typ)
                elif m == 2:
                    return coerce(3.0 * _PI_F32 / 4.0, typ)
                else:
                    return coerce(-3.0 * _PI_F32 / 4.0, typ)
            if m == 0:
                return coerce(0.0, typ)
            elif m == 1:
                return coerce(_neg_f32(coerce(0.0, f32)), typ)
            elif m == 2:
                return coerce(_PI_F32, typ)
            else:
                return coerce(-_PI_F32, typ)
        yd = coerce(y, f64)
        xd = coerce(x, f64)
        if ix + (26 << 23) < iy or iy == 0x7F800000:
            # |y/x| > 2**26 (or y = +-inf with a finite x)
            if (m & 1) != 0:
                return coerce(-_PIO2_HI_F, typ)
            return coerce(_PIO2_HI_F, typ)
        zf: f32 = coerce(0.0, f32)
        if (m & 2) != 0 and iy + (26 << 23) < ix:
            # |y/x| < 2**-26, x < 0: the result is pi, so no underflow
            zf = coerce(0.0, f32)
        else:
            ratio = float_truncate(_fabs64(yd / xd), f32)
            zf = atan(ratio)
        if m == 0:
            return coerce(zf, typ)
        elif m == 1:
            return coerce(_neg_f32(zf), typ)
        elif m == 2:
            return float_truncate(_PI_F32 - (coerce(zf, f64) - _PI_LO_F32), typ)
        else:
            return float_truncate((coerce(zf, f64) - _PI_LO_F32) - _PI_F32, typ)
    elif typ == f64:
        xb = bitcast(x, u64)
        yb = bitcast(y, u64)
        hx = truncate(xb >> 32, u32)
        hy = truncate(yb >> 32, u32)
        lx = truncate(xb & 0xFFFFFFFF, u32)
        ly = truncate(yb & 0xFFFFFFFF, u32)
        ix = hx & 0x7FFFFFFF
        iy = hy & 0x7FFFFFFF
        if (ix > 0x7FF00000 or (ix == 0x7FF00000 and lx != 0)
                or iy > 0x7FF00000 or (iy == 0x7FF00000 and ly != 0)):
            # a NaN in either argument
            return coerce(coerce(x, f64) + coerce(y, f64), typ)
        if hx == 0x3FF00000 and lx == 0:
            # x = 1.0 (the sign is part of ``hx``: x = -1 must not match)
            return atan(y)
        m = ((hy >> 31) & 1) | ((hx >> 30) & 2)
        if (iy | ly) == 0:
            # y = +-0
            if m == 0 or m == 1:
                return y
            elif m == 2:
                return coerce(_PI, typ)
            else:
                return coerce(-_PI, typ)
        if (ix | lx) == 0:
            # x = +-0
            if (m & 1) != 0:
                return coerce(-_PIO2_HI, typ)
            return coerce(_PIO2_HI, typ)
        if ix == 0x7FF00000:
            # x = +-inf
            if iy == 0x7FF00000:
                # y = +-inf
                if m == 0:
                    return coerce(_PI / 4.0, typ)
                elif m == 1:
                    return coerce(-_PI / 4.0, typ)
                elif m == 2:
                    return coerce(3.0 * _PI / 4.0, typ)
                else:
                    return coerce(-3.0 * _PI / 4.0, typ)
            if m == 0:
                return coerce(0.0, typ)
            elif m == 1:
                return coerce(_neg_f64(coerce(0.0, f64)), typ)
            elif m == 2:
                return coerce(_PI, typ)
            else:
                return coerce(-_PI, typ)
        yd = coerce(y, f64)
        xd = coerce(x, f64)
        if ix + (64 << 20) < iy or iy == 0x7FF00000:
            # |y/x| > 2**64 (or y = +-inf with a finite x)
            if (m & 1) != 0:
                return coerce(-_PIO2_HI, typ)
            return coerce(_PIO2_HI, typ)
        z64: f64 = 0.0
        if (m & 2) != 0 and iy + (64 << 20) < ix:
            # |y/x| < 2**-64, x < 0: the result is pi, so no underflow
            z64 = 0.0
        else:
            z64 = atan(_fabs64(yd / xd))
        if m == 0:
            return coerce(z64, typ)
        elif m == 1:
            return coerce(_neg_f64(z64), typ)
        elif m == 2:
            return coerce(_PI - (z64 - _PI_LO), typ)
        else:
            return coerce((z64 - _PI_LO) - _PI, typ)
    compile_error("only f32 or f64")

"""The software (pure spy) exponential function (``std.math``).

A port of musl's ``exp.c`` and ``expf.c`` (the Arm Limited implementations, MIT).

* ``f64`` reuses the ``exp`` kernel already present in :mod:`std.math.pow`:
  :func:`_exp_inline` is a direct port of ``exp.c`` and computes
  ``sign * exp(x + xtail)``; for ``exp`` the tail is zero and the sign is
  positive.  The only thing the wrapper adds is the NaN check, because the
  kernel only inspects the exponent field and would otherwise turn a NaN into
  ``+-inf``/``+-0``.
* ``f32`` follows ``expf.c``: the argument is reduced as
  ``x*N/ln2 = k + r`` against the shared 32-entry ``exp2f`` table, and
  ``2**(k/N) * 2**(r/N)`` is evaluated with the ``poly_scaled`` coefficients of
  ``exp2f_data.c``.  The computation is carried out in ``f64`` and rounded once,
  at the end, to ``f32``.

The public entry point is :func:`exp`, generic over ``f32``/``f64``.  It is what
``std.math`` calls when the compile-time option ``exp_mode`` is ``software``
(the default).

The one deviation from musl is its floating point environment handling: spy has
no ``fenv``, so ``__math_oflow``/``__math_uflow`` are the plain ``+inf``/``+0``
results, ``issignaling`` is always false and ``WANT_ROUNDING`` is 1 (so the tiny
argument path keeps ``1 + x``).  musl bounds the error of ``exp`` and ``expf``
at about 0.51 ULP, which the arithmetic below reproduces (it is the non-FMA
path).
"""

from ...compiler import f32, f64, typeof, u32, u64, usize
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..core import Numeric, bitcast, coerce, compile_error
from ..float import float_truncate
from .pow import (
    _EXP2F_C0,
    _EXP2F_C1,
    _EXP2F_C2,
    _exp2f_tab,
    _exp_inline,
    _hex,
)

# ``expf.c``: ``N/ln2``, the rounding shift and the ``poly_scaled``
# coefficients of ``exp2f_data.c`` (``N = 1 << EXP2F_TABLE_BITS = 32``).
_EXPF_INVLN2N = _hex(b'0x1.71547652b82fep0') * 32.0
_EXPF_SHIFT = _hex(b'0x1.8p52')
_EXPF_C0 = _EXP2F_C0 / 32.0 / 32.0 / 32.0
_EXPF_C1 = _EXP2F_C1 / 32.0 / 32.0
_EXPF_C2 = _EXP2F_C2 / 32.0

# the ``expf`` overflow / underflow thresholds and the top-12-bit tests
_EXPF_OFLOW = _hex(b'0x1.62e42ep6')
_EXPF_UFLOW = _hex(b'-0x1.9fe368p6')
_EXPF_88_TOP = 0x42B
_EXPF_INF_TOP = 0x7F8

_INF = float('inf')


@func()
def exp[T: Numeric](x: T) -> T:
    """``e ** x`` (``f32`` or ``f64``), the pure spy implementation: a port of
    musl's ``expf``/``exp``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        xf = coerce(x, f32)
        ix = bitcast(xf, u32)
        abstop = (ix >> 20) & 0x7FF
        if abstop >= _EXPF_88_TOP:
            # |x| >= 88 or x is an inf/NaN
            if ix == 0xFF800000:
                return coerce(0.0, typ)  # x is -inf
            if abstop >= _EXPF_INF_TOP:
                return coerce(xf + xf, typ)  # x is +inf or NaN
            if xf > _EXPF_OFLOW:
                return coerce(_INF, typ)
            if xf < _EXPF_UFLOW:
                return coerce(0.0, typ)
        xd = coerce(xf, f64)
        # x*N/ln2 = k + r with r in [-1/2, 1/2] and integer k
        z = _EXPF_INVLN2N * xd
        kdf = z + _EXPF_SHIFT
        ki = bitcast(kdf, u64)
        kd = kdf - _EXPF_SHIFT
        r = z - kd
        # exp(x) = 2**(k/N) * 2**(r/N) ~= s * (C0*r**3 + C1*r**2 + C2*r + 1)
        t = _exp2f_tab(coerce(ki % 32, usize)) + (ki << 47)
        s = bitcast(t, f64)
        r2 = r * r
        y = _EXPF_C2 * r + 1.0
        y = (_EXPF_C0 * r + _EXPF_C1) * r2 + y
        return float_truncate(y * s, typ)
    elif typ == f64:
        xd = coerce(x, f64)
        ix = bitcast(xd, u64)
        if (ix & 0x7FF0000000000000) == 0x7FF0000000000000 and (ix & 0x000FFFFFFFFFFFFF) != 0:
            return coerce(xd + xd, typ)  # NaN
        return coerce(_exp_inline(xd, 0.0, coerce(0, u32)), typ)
    compile_error("only f32 or f64")

"""The software (pure spy) square root (``std.math``).

A port of musl's ``sqrt.c`` and ``sqrtf.c``: the argument is reduced to
``x = 4**e * m`` with ``m`` in ``[1, 4)``, a 7-bit table estimate of
``1/sqrt(m)`` (``__rsqrt_tab``, musl's ``sqrt_data.c``) is refined by
Goldschmidt iterations, and a last step rounds the result to nearest.  Where C
needs a separate ``float`` and ``double`` copy, a single function with a
compile-time ``is_32`` flag serves both: only the branch of the requested float
type is ever typed and lowered.

The public entry point is :func:`sqrt`, generic over ``f32``/``f64``.  It is
what ``std.math`` calls when the compile-time option ``sqrt_mode`` is
``software`` (the default).

The one deviation from musl is its ``FENV_SUPPORT`` tail (adding a ``tiny``
value so a non-default rounding mode and the ``inexact`` flag see the right
result): spy has no floating point environment, and under the default
round-to-nearest mode that value is far below the result's ulp, so it never
changes the returned number.
"""

from ...compiler import (
    array,
    comptime,
    const_comptime,
    f32,
    f64,
    func,
    typeof,
    u16,
    u32,
    u64,
    usize,
)
from ..core import Numeric, bitcast, coerce, compile_error
from ..int import truncate


@func(inline=True)
def _mul32(a: u32, b: u32) -> u32:
    """``a * b * 2**-32``, the 32-bit fixed point product of musl's ``mul32``."""
    return truncate(coerce(a, u64) * coerce(b, u64) >> 32, u32)


@func()
def _mul64(a: u64, b: u64) -> u64:
    """``a * b * 2**-64``, the fixed point product of musl's ``mul64``: the
    high 64 bits of the full product, computed from the 32-bit halves."""
    ahi = a >> 32
    alo = a & 0xFFFFFFFF
    bhi = b >> 32
    blo = b & 0xFFFFFFFF
    return ahi * bhi + (ahi * blo >> 32) + (alo * bhi >> 32)


@func()
def _rsqrt_tab(index: usize) -> u32:
    """``__rsqrt_tab[index]`` (musl's ``sqrt_data.c``): the 7-bit estimate of
    ``1/sqrt(m)``, as the 16-bit entry zero-extended to ``u32``."""
    const_comptime()
    tab = array(
        coerce(0xB451, u16), 0xB2F0, 0xB196, 0xB044,
        0xAEF9, 0xADB6, 0xAC79, 0xAB43,
        0xAA14, 0xA8EB, 0xA7C8, 0xA6AA,
        0xA592, 0xA480, 0xA373, 0xA26B,
        0xA168, 0xA06A, 0x9F70, 0x9E7B,
        0x9D8A, 0x9C9D, 0x9BB5, 0x9AD1,
        0x99F0, 0x9913, 0x983A, 0x9765,
        0x9693, 0x95C4, 0x94F8, 0x9430,
        0x936B, 0x92A9, 0x91EA, 0x912E,
        0x9075, 0x8FBE, 0x8F0A, 0x8E59,
        0x8DAA, 0x8CFE, 0x8C54, 0x8BAC,
        0x8B07, 0x8A64, 0x89C4, 0x8925,
        0x8889, 0x87EE, 0x8756, 0x86C0,
        0x862B, 0x8599, 0x8508, 0x8479,
        0x83EC, 0x8361, 0x82D8, 0x8250,
        0x81C9, 0x8145, 0x80C2, 0x8040,
        0xFF02, 0xFD0E, 0xFB25, 0xF947,
        0xF773, 0xF5AA, 0xF3EA, 0xF234,
        0xF087, 0xEEE3, 0xED47, 0xEBB3,
        0xEA27, 0xE8A3, 0xE727, 0xE5B2,
        0xE443, 0xE2DC, 0xE17A, 0xE020,
        0xDECB, 0xDD7D, 0xDC34, 0xDAF1,
        0xD9B3, 0xD87B, 0xD748, 0xD61A,
        0xD4F1, 0xD3CD, 0xD2AD, 0xD192,
        0xD07B, 0xCF69, 0xCE5B, 0xCD51,
        0xCC4A, 0xCB48, 0xCA4A, 0xC94F,
        0xC858, 0xC764, 0xC674, 0xC587,
        0xC49D, 0xC3B7, 0xC2D4, 0xC1F4,
        0xC116, 0xC03C, 0xBF65, 0xBE90,
        0xBDBE, 0xBCEF, 0xBC23, 0xBB59,
        0xBA91, 0xB9CC, 0xB90A, 0xB84A,
        0xB78C, 0xB6D0, 0xB617, 0xB560,
    )
    return coerce(tab[index], u32)


@func()
def _invalid_f32(x: f32) -> f32:
    """``__math_invalidf``: the NaN a negative argument yields (``0/0``; in C it
    also raises the invalid exception)."""
    return (x - x) / (x - x)


@func()
def _invalid_f64(x: f64) -> f64:
    """``__math_invalid``: the ``f64`` counterpart of :func:`_invalid_f32`."""
    return (x - x) / (x - x)


@func()
def sqrt[T: Numeric](x: T) -> T:
    """The square root of ``x`` (``f32`` or ``f64``), the pure spy
    implementation: a port of musl's ``sqrtf``/``sqrt``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        ix = bitcast(x, u32)
        if ix - 0x00800000 >= 0x7F800000 - 0x00800000:
            if ix * 2 == 0:
                return x
            if ix == 0x7F800000:
                return x
            if ix > 0x7F800000:
                return coerce(_invalid_f32(coerce(x, f32)), typ)
            # x is subnormal: scale it into the normal range first
            ix = bitcast(coerce(x, f32) * coerce(8388608.0, f32), u32) - 0x0B800000
        m = (ix << 8) | 0x80000000
        if (ix & 0x00800000) != 0:
            m = (ix << 7) & 0x7FFFFFFF
        ey = ((ix >> 1) + 0x1FC00000) & 0x7F800000
        r = _rsqrt_tab(coerce((ix >> 17) % 128, usize)) << 16
        s = _mul32(m, r)
        d = _mul32(s, r)
        u = 0xC0000000 - d
        r = _mul32(r, u) << 1
        s = _mul32(s, u) << 1
        d = _mul32(s, r)
        u = 0xC0000000 - d
        s = _mul32(s, u)
        s = (s - 1) >> 6
        d0 = (m << 16) - s * s
        d1 = s - d0
        s = (s + (d1 >> 31)) & 0x007FFFFF
        return coerce(bitcast(s | ey, f32), typ)
    elif typ == f64:
        ix = bitcast(x, u64)
        top = ix >> 52
        if top - 0x001 >= 0x7FF - 0x001:
            if ix * 2 == 0:
                return x
            if ix == 0x7FF0000000000000:
                return x
            if ix > 0x7FF0000000000000:
                return coerce(_invalid_f64(coerce(x, f64)), typ)
            # x is subnormal: scale it into the normal range first
            ix = bitcast(coerce(x, f64) * 4503599627370496.0, u64)
            top = (ix >> 52) - 52
        m = (ix << 11) | 0x8000000000000000
        if (top & 1) != 0:
            m = m >> 1
        top = (top + 0x3FF) >> 1
        r = coerce(_rsqrt_tab(coerce((ix >> 46) % 128, usize)) << 16, u64)
        s = coerce(_mul32(truncate(m >> 32, u32), truncate(r, u32)), u64)
        d = coerce(_mul32(truncate(s, u32), truncate(r, u32)), u64)
        u = 0xC0000000 - d
        r = coerce(_mul32(truncate(r, u32), truncate(u, u32)) << 1, u64)
        s = coerce(_mul32(truncate(s, u32), truncate(u, u32)) << 1, u64)
        d = coerce(_mul32(truncate(s, u32), truncate(r, u32)), u64)
        u = 0xC0000000 - d
        r = coerce(_mul32(truncate(r, u32), truncate(u, u32)) << 1, u64)
        r = r << 32
        s = _mul64(m, r)
        d = _mul64(s, r)
        u = (coerce(0xC0000000, u64) << 32) - d
        s = _mul64(s, u)
        s = (s - 2) >> 9
        d0 = (m << 42) - s * s
        d1 = s - d0
        s = (s + (d1 >> 63)) & 0x000FFFFFFFFFFFFF
        return coerce(bitcast(s | (top << 52), f64), typ)
    compile_error("only f32 or f64")

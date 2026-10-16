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

from ...compiler import f32, f64, typeof, u16, u32, u64, usize
from ...compiler.dsl import func
from ...compiler.syntax import Comptime, array, comptime, const_comptime
from ..core import bitcast, coerce


@func(inline=True)
def _mul32(a: u32, b: u32) -> u32:
    """``a * b * 2**-32``, the 32-bit fixed point product of musl's ``mul32``."""
    return coerce(coerce(a, u64) * coerce(b, u64) >> 32, u32)


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
        coerce(0xB451, u16), coerce(0xB2F0, u16), coerce(0xB196, u16), coerce(0xB044, u16),
        coerce(0xAEF9, u16), coerce(0xADB6, u16), coerce(0xAC79, u16), coerce(0xAB43, u16),
        coerce(0xAA14, u16), coerce(0xA8EB, u16), coerce(0xA7C8, u16), coerce(0xA6AA, u16),
        coerce(0xA592, u16), coerce(0xA480, u16), coerce(0xA373, u16), coerce(0xA26B, u16),
        coerce(0xA168, u16), coerce(0xA06A, u16), coerce(0x9F70, u16), coerce(0x9E7B, u16),
        coerce(0x9D8A, u16), coerce(0x9C9D, u16), coerce(0x9BB5, u16), coerce(0x9AD1, u16),
        coerce(0x99F0, u16), coerce(0x9913, u16), coerce(0x983A, u16), coerce(0x9765, u16),
        coerce(0x9693, u16), coerce(0x95C4, u16), coerce(0x94F8, u16), coerce(0x9430, u16),
        coerce(0x936B, u16), coerce(0x92A9, u16), coerce(0x91EA, u16), coerce(0x912E, u16),
        coerce(0x9075, u16), coerce(0x8FBE, u16), coerce(0x8F0A, u16), coerce(0x8E59, u16),
        coerce(0x8DAA, u16), coerce(0x8CFE, u16), coerce(0x8C54, u16), coerce(0x8BAC, u16),
        coerce(0x8B07, u16), coerce(0x8A64, u16), coerce(0x89C4, u16), coerce(0x8925, u16),
        coerce(0x8889, u16), coerce(0x87EE, u16), coerce(0x8756, u16), coerce(0x86C0, u16),
        coerce(0x862B, u16), coerce(0x8599, u16), coerce(0x8508, u16), coerce(0x8479, u16),
        coerce(0x83EC, u16), coerce(0x8361, u16), coerce(0x82D8, u16), coerce(0x8250, u16),
        coerce(0x81C9, u16), coerce(0x8145, u16), coerce(0x80C2, u16), coerce(0x8040, u16),
        coerce(0xFF02, u16), coerce(0xFD0E, u16), coerce(0xFB25, u16), coerce(0xF947, u16),
        coerce(0xF773, u16), coerce(0xF5AA, u16), coerce(0xF3EA, u16), coerce(0xF234, u16),
        coerce(0xF087, u16), coerce(0xEEE3, u16), coerce(0xED47, u16), coerce(0xEBB3, u16),
        coerce(0xEA27, u16), coerce(0xE8A3, u16), coerce(0xE727, u16), coerce(0xE5B2, u16),
        coerce(0xE443, u16), coerce(0xE2DC, u16), coerce(0xE17A, u16), coerce(0xE020, u16),
        coerce(0xDECB, u16), coerce(0xDD7D, u16), coerce(0xDC34, u16), coerce(0xDAF1, u16),
        coerce(0xD9B3, u16), coerce(0xD87B, u16), coerce(0xD748, u16), coerce(0xD61A, u16),
        coerce(0xD4F1, u16), coerce(0xD3CD, u16), coerce(0xD2AD, u16), coerce(0xD192, u16),
        coerce(0xD07B, u16), coerce(0xCF69, u16), coerce(0xCE5B, u16), coerce(0xCD51, u16),
        coerce(0xCC4A, u16), coerce(0xCB48, u16), coerce(0xCA4A, u16), coerce(0xC94F, u16),
        coerce(0xC858, u16), coerce(0xC764, u16), coerce(0xC674, u16), coerce(0xC587, u16),
        coerce(0xC49D, u16), coerce(0xC3B7, u16), coerce(0xC2D4, u16), coerce(0xC1F4, u16),
        coerce(0xC116, u16), coerce(0xC03C, u16), coerce(0xBF65, u16), coerce(0xBE90, u16),
        coerce(0xBDBE, u16), coerce(0xBCEF, u16), coerce(0xBC23, u16), coerce(0xBB59, u16),
        coerce(0xBA91, u16), coerce(0xB9CC, u16), coerce(0xB90A, u16), coerce(0xB84A, u16),
        coerce(0xB78C, u16), coerce(0xB6D0, u16), coerce(0xB617, u16), coerce(0xB560, u16),
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
def sqrt[T](x: T) -> T:
    """The square root of ``x`` (``f32`` or ``f64``), the pure spy
    implementation: a port of musl's ``sqrtf``/``sqrt``."""
    comptime()
    typ = typeof(x)
    is_32: Comptime = typ == f32
    if is_32:
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
    s = coerce(_mul32(coerce(m >> 32, u32), coerce(r, u32)), u64)
    d = coerce(_mul32(coerce(s, u32), coerce(r, u32)), u64)
    u = 0xC0000000 - d
    r = coerce(_mul32(coerce(r, u32), coerce(u, u32)) << 1, u64)
    s = coerce(_mul32(coerce(s, u32), coerce(u, u32)) << 1, u64)
    d = coerce(_mul32(coerce(s, u32), coerce(r, u32)), u64)
    u = 0xC0000000 - d
    r = coerce(_mul32(coerce(r, u32), coerce(u, u32)) << 1, u64)
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

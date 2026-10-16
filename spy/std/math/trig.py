"""The software (pure spy) trigonometric functions (``std.math``).

A port of musl's ``sin.c``/``sinf.c``/``cos.c``/``cosf.c`` and the kernels they
build on (``__sin``/``__cos``/``__sindf``/``__cosdf``, ``__rem_pio2``/
``__rem_pio2f`` and ``__rem_pio2_large``).  Where C needs a separate ``float`` and
``double`` copy, a single function with a compile-time ``is_32`` flag serves both:
only the branch of the requested float type is ever typed and lowered.  ``sin``
and ``cos`` share the kernels and the argument reduction, so both live here.

The public entry points are :func:`sin` and :func:`cos`, generic over
``f32``/``f64``.  They are what ``std.math`` calls when the compile-time option
``sin_mode``/``cos_mode`` is ``software`` (the default).
"""

from typing import Literal

from ...compiler import bool, f32, f64, i32, typeof, u32, u64
from ...compiler.dsl import func
from ...compiler.syntax import Array, Comptime, array, comptime, const_comptime
from ..core import (
    bitcast,
    coerce,
    float_to_int_unchecked,
    int_to_float_unchecked,
    undefined,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# __sin (f64)
_DS1 = -1.66666666666666324348e-01
_DS2 = 8.33333333332248946124e-03
_DS3 = -1.98412698298579493134e-04
_DS4 = 2.75573137070700676789e-06
_DS5 = -2.50507602534068634195e-08
_DS6 = 1.58969099521155010221e-10

# __cos (f64)
_DC1 = 4.16666666666666019037e-02
_DC2 = -1.38888888888741095749e-03
_DC3 = 2.48015872894767294178e-05
_DC4 = -2.75573143513906633035e-07
_DC5 = 2.08757232129817482790e-09
_DC6 = -1.13596475577881948265e-11

# __sindf (f32; evaluated in double)
_FS1 = -0.166666666416265235595
_FS2 = 8.3333293858894631756e-03
_FS3 = -1.98393348360966317347e-04
_FS4 = 2.7183114939898219064e-06

# __cosdf (f32; evaluated in double)
_FC0 = -0.499999997251031003120
_FC1 = 4.16666233237390631894e-02
_FC2 = -1.38867637746099294692e-03
_FC3 = 2.43904487962774090654e-05

# powers of two used to split/assemble the 24-bit chunks
_P24 = 16777216.0  # 0x1p24
_M24 = 5.9604644775390625e-08  # 0x1p-24

# __rem_pio2 (f64)
_TOINT = 6755399441055744.0  # 1.5 / DBL_EPSILON
_PIO4 = 0.7853981633974483  # 0x1.921fb54442d18p-1
_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e00
_PIO2_1T = 6.07710050650619224932e-11
_PIO2_2 = 6.07710050630396597660e-11
_PIO2_2T = 2.02226624879595063154e-21
_PIO2_3 = 2.02226624871116645580e-21
_PIO2_3T = 8.47842766036889956997e-32

# __rem_pio2f (f32)
_PIO4F = 0.7853981852531433  # 0x1.921fb6p-1
_PIO2_1F = 1.57079631090164184570e00
_PIO2_1TF = 1.58932547735281966916e-08

# smaller multiples of pi/2 for sinf/cosf (f64; the `s1pio2..` of musl's sinf
# and the `c1pio2..` of cosf are the same values)
_PIO2 = 1.5707963267948966
_PI = 3.141592653589793
_THREE_PIO2 = 4.71238898038469
_TWO_PI = 6.283185307179586


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@func()
def _scalbn(x: f64, n: i32) -> f64:
    """``x * 2**n``, computed by repeated (exact) scaling by two.  ``n`` stays
    small in every use here, so the linear loop is cheap."""
    r = x
    k = n
    if k < 0:
        k = -k
        for _ in range(k):
            r = r / 2.0
    else:
        for _ in range(k):
            r = r * 2.0
    return r


@func()
def _sin_kernel(is_32: Comptime[bool], x: f64, y: f64, iy: i32) -> f64:
    """The sine kernel on ``[-pi/4, pi/4]``: ``__sin`` for ``f64`` (with the
    tail ``y``) and ``__sindf`` for ``f32`` (which ignores ``y``/``iy``)."""
    if is_32:
        z32 = x * x
        w32 = z32 * z32
        r32 = _FS3 + z32 * _FS4
        s32 = z32 * x
        return (x + s32 * (_FS1 + z32 * _FS2)) + s32 * w32 * r32
    z = x * x
    w = z * z
    r = _DS2 + z * (_DS3 + z * _DS4) + z * w * (_DS5 + z * _DS6)
    v = z * x
    if iy == 0:
        return x + v * (_DS1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * _DS1)


@func()
def _cos_kernel(is_32: Comptime[bool], x: f64, y: f64) -> f64:
    """The cosine kernel on ``[-pi/4, pi/4]``: ``__cos`` for ``f64`` and
    ``__cosdf`` for ``f32``."""
    if is_32:
        z32 = x * x
        w32 = z32 * z32
        r32 = _FC2 + z32 * _FC3
        return ((1.0 + z32 * _FC0) + w32 * _FC1) + (w32 * z32) * r32
    z = x * x
    w = z * z
    r = z * (_DC1 + z * (_DC2 + z * _DC3)) + w * w * (_DC4 + z * (_DC5 + z * _DC6))
    hz = 0.5 * z
    w2 = 1.0 - hz
    return w2 + (((1.0 - w2) - hz) + (z * r - x * y))


@func()
def _rem_pio2_large(x: Array[f64, Literal[3]], e0: i32, nx: i32, prec: i32) -> tuple[i32, f64, f64]:
    """The argument reduction of ``__rem_pio2_large``: the double-precision
    kernel shared by ``f32`` and ``f64``.  ``x`` holds the ``nx`` 24-bit chunks
    of the input (in ``[2**23, 2**24)``), ``e0`` the exponent of the first, and
    ``prec`` the precision (0 for single, 1 for double).  Returns
    ``(n & 7, y0, y1)`` with ``y0 + y1 = x - n*pi/2``."""
    const_comptime()
    ipio2 = array(
        coerce(0xA2F983, i32), 0x6E4E44, 0x1529FC,
        0x2757D1, 0xF534DD, 0xC0DB62,
        0x95993C, 0x439041, 0xFE5163,
        0xABDEBB, 0xC561B7, 0x246E3A,
        0x424DD2, 0xE00649, 0x2EEA09,
        0xD1921C, 0xFE1DEB, 0x1CB129,
        0xA73EE8, 0x8235F5, 0x2EBB44,
        0x84E99C, 0x7026B4, 0x5F7E41,
        0x3991D6, 0x398353, 0x39F49C,
        0x845F8B, 0xBDF928, 0x3B1FF8,
        0x97FFDE, 0x05980F, 0xEF2F11,
        0x8B5A0A, 0x6D1F6D, 0x367ECF,
        0x27CB09, 0xB74F46, 0x3F669E,
        0x5FEA2D, 0x7527BA, 0xC7EBE5,
        0xF17B3D, 0x0739F7, 0x8A5292,
        0xEA6BFB, 0x5FB11F, 0x8D5D08,
        0x560330, 0x46FC7B, 0x6BABF0,
        0xCFBC20, 0x9AF436, 0x1DA9E3,
        0x91615E, 0xE61B08, 0x659985,
        0x5F14A0, 0x68408D, 0xFFD880,
        0x4D7327, 0x310606, 0x1556CA,
        0x73A8C9, 0x60E27B, 0xC08C6B,
    )
    const_comptime()
    pio2 = array(
        coerce(1.57079625129699707031e00, f64),
        7.54978941586159635335e-08,
        5.39030252995776476554e-15,
        3.28200341580791294123e-22,
        1.27065575308067607349e-29,
        1.22933308981111328932e-36,
        2.73370053816464559624e-44,
        2.16741683877804819444e-51,
    )
    # the tables are compile-time constants: taking their address lowers each to
    # a private constant global, built once instead of on every call

    jk: i32 = 3
    if prec == 1 or prec == 2:
        jk = 4
    elif prec == 3:
        jk = 6
    jp: i32 = jk
    jx: i32 = nx - 1
    jv: i32 = 0
    if e0 - 3 > 0:
        jv = (e0 - 3) // 24
    q0 = e0 - 24 * (jv + 1)

    f: Array[f64, Literal[20]] = undefined()
    fq: Array[f64, Literal[20]] = undefined()
    q: Array[f64, Literal[20]] = undefined()
    iq: Array[i32, Literal[20]] = undefined()

    # loop counters, declared once so every block below shares one typed slot
    i: i32 = 0
    j: i32 = 0
    k: i32 = 0
    fw: f64 = 0.0

    # f[0 .. jx+jk] = ipio2[jv-jx .. jv+jk] (padding the negative indices with 0)
    jf = jv - jx
    m = jx + jk
    i = 0
    while i <= m:
        if jf < 0:
            f[i] = 0.0
        else:
            f[i] = int_to_float_unchecked(ipio2[jf], f64)
        jf = jf + 1
        i = i + 1

    # q[0 .. jk] = the first chunks of x*(2/pi)
    i = 0
    while i <= jk:
        fw = 0.0
        j = 0
        while j <= jx:
            fw = fw + x[j] * f[jx + i - j]
            j = j + 1
        q[i] = fw
        i = i + 1

    jz = jk
    n: i32 = 0
    ih: i32 = 0
    z: f64 = 0.0

    while True:
        # distill q[] into iq[] reversingly
        j = jz
        z = q[jz]
        i = 0
        while j > 0:
            fw = int_to_float_unchecked(float_to_int_unchecked(_M24 * z, i32), f64)
            iq[i] = float_to_int_unchecked(z - _P24 * fw, i32)
            z = q[j - 1] + fw
            i = i + 1
            j = j - 1

        # compute n
        z = _scalbn(z, q0)
        z = z - 8.0 * (z * 0.125 // 1.0)
        n = float_to_int_unchecked(z, i32)
        z = z - int_to_float_unchecked(n, f64)
        ih = 0
        if q0 > 0:
            ii: i32 = iq[jz - 1] >> (24 - q0)
            n = n + ii
            iq[jz - 1] = iq[jz - 1] - (ii << (24 - q0))
            ih = iq[jz - 1] >> (23 - q0)
        elif q0 == 0:
            ih = iq[jz - 1] >> 23
        elif z >= 0.5:
            ih = 2

        if ih > 0:
            # compute 1 - q
            n = n + 1
            carry: i32 = 0
            i = 0
            while i < jz:
                jj: i32 = iq[i]
                if carry == 0:
                    if jj != 0:
                        carry = 1
                        iq[i] = 0x1000000 - jj
                else:
                    iq[i] = 0xFFFFFF - jj
                i = i + 1
            if q0 > 0:
                if q0 == 1:
                    iq[jz - 1] = iq[jz - 1] & 0x7FFFFF
                elif q0 == 2:
                    iq[jz - 1] = iq[jz - 1] & 0x3FFFFF
            if ih == 2:
                z = 1.0 - z
                if carry != 0:
                    z = z - _scalbn(1.0, q0)

        # check if recomputation is needed
        need: bool = False
        if z == 0.0:
            acc: i32 = 0
            i = jz - 1
            while i >= jk:
                acc = acc | iq[i]
                i = i - 1
            if acc == 0:
                k = 1
                while iq[jk - k] == 0:
                    k = k + 1
                i = jz + 1
                while i <= jz + k:
                    f[jx + i] = int_to_float_unchecked(ipio2[jv + i], f64)
                    fw = 0.0
                    j = 0
                    while j <= jx:
                        fw = fw + x[j] * f[jx + i - j]
                        j = j + 1
                    q[i] = fw
                    i = i + 1
                jz = jz + k
                need = True
        if need:
            continue
        break

    # chop off zero terms
    if z == 0.0:
        jz = jz - 1
        q0 = q0 - 24
        while iq[jz] == 0:
            jz = jz - 1
            q0 = q0 - 24
    else:
        z = _scalbn(z, -q0)
        if z >= _P24:
            fw = int_to_float_unchecked(float_to_int_unchecked(_M24 * z, i32), f64)
            iq[jz] = float_to_int_unchecked(z - _P24 * fw, i32)
            jz = jz + 1
            q0 = q0 + 24
            iq[jz] = float_to_int_unchecked(fw, i32)
        else:
            iq[jz] = float_to_int_unchecked(z, i32)

    # convert the integer "bit" chunks to floating point
    fw = _scalbn(1.0, q0)
    i = jz
    while i >= 0:
        q[i] = fw * int_to_float_unchecked(iq[i], f64)
        fw = fw * _M24
        i = i - 1

    # pi/2 * q, chunk by chunk
    i = jz
    while i >= 0:
        fw = 0.0
        k = 0
        while k <= jp and k <= jz - i:
            fw = fw + pio2[k] * q[i + k]
            k = k + 1
        fq[jz - i] = fw
        i = i - 1

    # compress fq[] into y[]
    y0s: f64 = 0.0
    y1s: f64 = 0.0
    if prec == 0:
        fw = 0.0
        i = jz
        while i >= 0:
            fw = fw + fq[i]
            i = i - 1
        y0s = fw
        if ih != 0:
            y0s = -fw
        return n & 7, y0s, 0.0

    fw = 0.0
    i = jz
    while i >= 0:
        fw = fw + fq[i]
        i = i - 1
    y0s = fw
    if ih != 0:
        y0s = -fw
    fw = fq[0] - fw
    i = 1
    while i <= jz:
        fw = fw + fq[i]
        i = i + 1
    y1s = fw
    if ih != 0:
        y1s = -fw
    return n & 7, y0s, y1s


@func()
def _rem_pio2(is_32: Comptime[bool], bits: u64, xf: f64) -> tuple[i32, f64, f64]:
    """The argument reduction: ``x - n*pi/2`` in ``y0 + y1`` for ``f64``
    (``__rem_pio2``) and in ``y0`` for ``f32`` (``__rem_pio2f``, whose ``y1`` is
    always ``0``).  ``bits`` is the raw little-endian bit pattern of ``x`` (the
    32-bit one zero-extended for ``f32``) and ``xf`` its value as a double.
    Returns ``(n, y0, y1)``."""
    if is_32:
        ix = bits & 0x7FFFFFFF
        if ix < 0x4DC90FDB:
            fn32 = xf * _INVPIO2 + _TOINT - _TOINT
            n32 = float_to_int_unchecked(fn32, i32)
            yv: f64 = xf - fn32 * _PIO2_1F - fn32 * _PIO2_1TF
            if yv < -_PIO4F:
                n32 = n32 - 1
                fn32 = fn32 - 1.0
                yv = xf - fn32 * _PIO2_1F - fn32 * _PIO2_1TF
            elif yv > _PIO4F:
                n32 = n32 + 1
                fn32 = fn32 + 1.0
                yv = xf - fn32 * _PIO2_1F - fn32 * _PIO2_1TF
            return n32, yv, 0.0
        if ix >= 0x7F800000:
            return 0, xf - xf, 0.0
        sgn = bits >> 31
        e032 = coerce(ix >> 23, i32) - (0x7F + 23)
        u2 = coerce(ix, u32) - bitcast(e032 << 23, u32)
        tx32: Array[f64, Literal[3]] = undefined()
        tx32[0] = coerce(bitcast(u2, f32), f64)
        tx32[1] = 0.0
        tx32[2] = 0.0
        n2, ty0, _ = _rem_pio2_large(tx32, e032, 1, 0)
        if sgn != 0:
            return -n2, -ty0, 0.0
        return n2, ty0, 0.0

    sign = bits >> 63
    ix64 = (bits >> 32) & 0x7FFFFFFF

    medium = False
    if ix64 <= 0x400F6A7A:
        if (ix64 & 0xFFFFF) == 0x921FB:
            medium = True
        elif ix64 <= 0x4002D97C:
            if sign == 0:
                z: f64 = xf - _PIO2_1
                y0a: f64 = z - _PIO2_1T
                return 1, y0a, (z - y0a) - _PIO2_1T
            z = xf + _PIO2_1
            y0a = z + _PIO2_1T
            return -1, y0a, (z - y0a) + _PIO2_1T
        else:
            if sign == 0:
                z = xf - 2.0 * _PIO2_1
                y0a = z - 2.0 * _PIO2_1T
                return 2, y0a, (z - y0a) - 2.0 * _PIO2_1T
            z = xf + 2.0 * _PIO2_1
            y0a = z + 2.0 * _PIO2_1T
            return -2, y0a, (z - y0a) + 2.0 * _PIO2_1T
    elif ix64 <= 0x401C463B:
        if ix64 <= 0x4015FDBC:
            if ix64 == 0x4012D97C:
                medium = True
            elif sign == 0:
                z = xf - 3.0 * _PIO2_1
                y0a = z - 3.0 * _PIO2_1T
                return 3, y0a, (z - y0a) - 3.0 * _PIO2_1T
            else:
                z = xf + 3.0 * _PIO2_1
                y0a = z + 3.0 * _PIO2_1T
                return -3, y0a, (z - y0a) + 3.0 * _PIO2_1T
        elif ix64 == 0x401921FB:
            medium = True
        elif sign == 0:
            z = xf - 4.0 * _PIO2_1
            y0a = z - 4.0 * _PIO2_1T
            return 4, y0a, (z - y0a) - 4.0 * _PIO2_1T
        else:
            z = xf + 4.0 * _PIO2_1
            y0a = z + 4.0 * _PIO2_1T
            return -4, y0a, (z - y0a) + 4.0 * _PIO2_1T

    if medium or ix64 < 0x413921FB:
        # medium size: rint(x/(pi/2)) through the 1.5/eps trick
        fn = xf * _INVPIO2 + _TOINT - _TOINT
        n = float_to_int_unchecked(fn, i32)
        r = xf - fn * _PIO2_1
        w = fn * _PIO2_1T
        if r - w < -_PIO4:
            n = n - 1
            fn = fn - 1.0
            r = xf - fn * _PIO2_1
            w = fn * _PIO2_1T
        elif r - w > _PIO4:
            n = n + 1
            fn = fn + 1.0
            r = xf - fn * _PIO2_1
            w = fn * _PIO2_1T
        y0b = r - w
        ey = coerce((bitcast(y0b, u64) >> 52) & 0x7FF, i32)
        ex = coerce(ix64 >> 20, i32)
        if ex - ey > 16:
            # second round
            t = r
            w = fn * _PIO2_2
            r = t - w
            w = fn * _PIO2_2T - ((t - r) - w)
            y0b = r - w
            ey = coerce((bitcast(y0b, u64) >> 52) & 0x7FF, i32)
            if ex - ey > 49:
                # third round
                t = r
                w = fn * _PIO2_3
                r = t - w
                w = fn * _PIO2_3T - ((t - r) - w)
                y0b = r - w
        return n, y0b, (r - y0b) - w

    # large arguments
    if ix64 >= 0x7FF00000:
        return 0, xf - xf, xf - xf
    # z = scalbn(|x|, -ilogb(x) + 23): scale into [2**23, 2**24)
    u64v = (bits & 0xFFFFFFFFFFFFF) | 0x4160000000000000
    z = bitcast(u64v, f64)
    tx: Array[f64, Literal[3]] = undefined()
    tx[0] = int_to_float_unchecked(float_to_int_unchecked(z, i32), f64)
    z = (z - tx[0]) * _P24
    tx[1] = int_to_float_unchecked(float_to_int_unchecked(z, i32), f64)
    z = (z - tx[1]) * _P24
    tx[2] = z
    i: i32 = 2
    while tx[i] == 0.0:
        i = i - 1
    e0 = coerce(ix64 >> 20, i32) - (0x3FF + 23)
    n, ty0, ty1 = _rem_pio2_large(tx, e0, i + 1, 1)
    if sign != 0:
        return -n, -ty0, -ty1
    return n, ty0, ty1


@func()
def _sin_switch(is_32: Comptime[bool], n: i32, y0: f64, y1: f64) -> f64:
    """``sin``'s ``switch (n & 3)``: the reduced argument ``y0 (+ y1)`` picks
    between the sine and cosine kernel and their signs (the table in musl)."""
    m = n & 3
    if m == 0:
        return _sin_kernel(is_32, y0, y1, 1)
    if m == 1:
        return _cos_kernel(is_32, y0, y1)
    if m == 2:
        return -_sin_kernel(is_32, y0, y1, 1)
    return -_cos_kernel(is_32, y0, y1)


@func()
def _cos_switch(is_32: Comptime[bool], n: i32, y0: f64, y1: f64) -> f64:
    """``cos``'s ``switch (n & 3)``: the same reduced argument, the other
    diagonal of musl's table (cosine/kernel with the opposite signs)."""
    m = n & 3
    if m == 0:
        return _cos_kernel(is_32, y0, y1)
    if m == 1:
        return -_sin_kernel(is_32, y0, y1, 1)
    if m == 2:
        return -_cos_kernel(is_32, y0, y1)
    return _sin_kernel(is_32, y0, y1, 1)


@func()
def sin[T](x: T) -> T:
    """The sine of ``x`` (``f32`` or ``f64``), the pure spy implementation: a
    port of musl's ``sinf``/``sin``."""
    comptime()
    typ = typeof(x)
    is_32: Comptime = typ == f32
    if is_32:
        bits: u32 = bitcast(x, u32)
        ix: u32 = bits & 0x7FFFFFFF
        if ix <= 0x3F490FDA:
            if ix < 0x39800000:
                return x
            return coerce(_sin_kernel(is_32, coerce(x, f64), 0.0, 0), typ)
        neg: bool = (bits >> 31) != 0
        if ix <= 0x407B53D1:
            if ix <= 0x4016CBE3:
                if neg:
                    return coerce(-_cos_kernel(is_32, coerce(x, f64) + _PIO2, 0.0), typ)
                return coerce(_cos_kernel(is_32, coerce(x, f64) - _PIO2, 0.0), typ)
            if neg:
                return coerce(_sin_kernel(is_32, -(coerce(x, f64) + _PI), 0.0, 0), typ)
            return coerce(_sin_kernel(is_32, -(coerce(x, f64) - _PI), 0.0, 0), typ)
        if ix <= 0x40E231D5:
            if ix <= 0x40AFEDDF:
                if neg:
                    return coerce(_cos_kernel(is_32, coerce(x, f64) + _THREE_PIO2, 0.0), typ)
                return coerce(-_cos_kernel(is_32, coerce(x, f64) - _THREE_PIO2, 0.0), typ)
            if neg:
                return coerce(_sin_kernel(is_32, coerce(x, f64) + _TWO_PI, 0.0, 0), typ)
            return coerce(_sin_kernel(is_32, coerce(x, f64) - _TWO_PI, 0.0, 0), typ)
        if ix >= 0x7F800000:
            xf32: f64 = coerce(x, f64)
            return coerce(xf32 - xf32, typ)
        n, y0, y1 = _rem_pio2(is_32, coerce(bits, u64), coerce(x, f64))
        return coerce(_sin_switch(is_32, n, y0, y1), typ)

    bits64 = bitcast(x, u64)
    ix64 = (bits64 >> 32) & 0x7FFFFFFF
    if ix64 <= 0x3FE921FB:
        if ix64 < 0x3E500000:
            return x
        return coerce(_sin_kernel(is_32, coerce(x, f64), 0.0, 0), typ)
    if ix64 >= 0x7FF00000:
        xf64 = coerce(x, f64)
        return coerce(xf64 - xf64, typ)
    n, y0, y1 = _rem_pio2(is_32, bits64, coerce(x, f64))
    return coerce(_sin_switch(is_32, n, y0, y1), typ)


@func()
def cos[T](x: T) -> T:
    """The cosine of ``x`` (``f32`` or ``f64``), the pure spy implementation: a
    port of musl's ``cosf``/``cos``."""
    comptime()
    typ = typeof(x)
    is_32: Comptime = typ == f32
    if is_32:
        bits = bitcast(x, u32)
        ix = bits & 0x7FFFFFFF
        if ix <= 0x3F490FDA:
            if ix < 0x39800000:
                return coerce(1.0, typ)
            return coerce(_cos_kernel(is_32, coerce(x, f64), 0.0), typ)
        neg = (bits >> 31) != 0
        if ix <= 0x407B53D1:
            if ix > 0x4016CBE3:
                if neg:
                    return coerce(-_cos_kernel(is_32, coerce(x, f64) + _PI, 0.0), typ)
                return coerce(-_cos_kernel(is_32, coerce(x, f64) - _PI, 0.0), typ)
            if neg:
                return coerce(_sin_kernel(is_32, coerce(x, f64) + _PIO2, 0.0, 0), typ)
            return coerce(_sin_kernel(is_32, _PIO2 - coerce(x, f64), 0.0, 0), typ)
        if ix <= 0x40E231D5:
            if ix > 0x40AFEDDF:
                if neg:
                    return coerce(_cos_kernel(is_32, coerce(x, f64) + _TWO_PI, 0.0), typ)
                return coerce(_cos_kernel(is_32, coerce(x, f64) - _TWO_PI, 0.0), typ)
            if neg:
                return coerce(_sin_kernel(is_32, -coerce(x, f64) - _THREE_PIO2, 0.0, 0), typ)
            return coerce(_sin_kernel(is_32, coerce(x, f64) - _THREE_PIO2, 0.0, 0), typ)
        if ix >= 0x7F800000:
            xfc32: f64 = coerce(x, f64)
            return coerce(xfc32 - xfc32, typ)
        n, y0, y1 = _rem_pio2(is_32, coerce(bits, u64), coerce(x, f64))
        return coerce(_cos_switch(is_32, n, y0, y1), typ)

    bits64 = bitcast(x, u64)
    ix64 = (bits64 >> 32) & 0x7FFFFFFF
    if ix64 <= 0x3FE921FB:
        if ix64 < 0x3E46A09E:
            return coerce(1.0, typ)
        return coerce(_cos_kernel(is_32, coerce(x, f64), 0.0), typ)
    if ix64 >= 0x7FF00000:
        xfc64 = coerce(x, f64)
        return coerce(xfc64 - xfc64, typ)
    n, y0, y1 = _rem_pio2(is_32, bits64, coerce(x, f64))
    return coerce(_cos_switch(is_32, n, y0, y1), typ)


@func()
def tan[T](x: T) -> T:
    """The tangent of ``x`` (``f32`` or ``f64``), the pure spy implementation:
    ``sin(x) / cos(x)`` of the same (software) kernels."""
    return sin(x) / cos(x)  # pyright: ignore[reportOperatorIssue]

"""The software (pure spy) ceiling (``std.math``).

A port of musl's ``ceil.c`` and ``ceilf.c``.  An ``f32`` is rounded up by
manipulating its bit pattern: the exponent selects the fraction bits, which are
either cleared (a negative argument) or cleared after adding the mask so the
carry reaches the next integer (a positive one); an ``f64`` uses the
``toint = 1/DBL_EPSILON`` trick of ``ceil.c``: adding and subtracting ``2**52``
rounds the argument to a nearby integer, whose difference from the argument
tells whether a fraction was rounded up.  Where C needs a separate ``float`` and
``double`` copy, a single function with a compile-time type branch serves both:
only the branch of the requested float type is ever typed and lowered.

The public entry point is :func:`ceil`, generic over ``f32``/``f64``.  It is
what ``std.math`` calls when the compile-time option ``ceil_mode`` is
``software`` (the default).

The one deviation from musl is its ``FORCE_EVAL`` (forcing the ``inexact``
flag): spy has no floating point environment, and the expression does not change
the returned number.  The result is exact: an integer, ``NaN`` and ``+-inf``
come out unchanged, ``ceil(+-0)`` is ``+-0``, and a negative argument in
``(-1, 0)`` yields ``-0``.
"""

from ...compiler import f32, f64, typeof, u32, u64
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..core import Numeric, bitcast, coerce, compile_error

# ``ceil.c``: ``toint = 1/DBL_EPSILON``.  Adding and subtracting it rounds an
# ``f64`` to the nearest integer (under the default round-to-nearest mode).
_TOINT = 4503599627370496.0


@func()
def ceil[T: Numeric](x: T) -> T:
    """The smallest integer not less than ``x`` (``f32`` or ``f64``), the pure
    spy implementation: a port of musl's ``ceilf``/``ceil``."""
    comptime()
    typ = typeof(x)
    if typ == f32:
        u = bitcast(x, u32)
        e = (u >> 23) & 0xFF
        if e >= 0x7F + 23:
            # already an integer (and inf/NaN)
            return x
        if e >= 0x7F:
            m = coerce(0x007FFFFF, u32) >> (e - 0x7F)
            if (u & m) == 0:
                return x
            if (u >> 31) == 0:
                u = u + m
            return coerce(bitcast(u & (0xFFFFFFFF - m), f32), typ)
        # |x| < 1: 1 for a positive non-zero x, -0 for a negative one
        if (u >> 31) != 0:
            return coerce(-0.0, typ)
        if (u << 1) != 0:
            return coerce(1.0, typ)
        return x
    elif typ == f64:
        u = bitcast(x, u64)
        e = (u >> 52) & 0x7FF
        if e >= 0x3FF + 52:
            # already an integer (and inf/NaN)
            return x
        if (u << 1) == 0:
            # +-0
            return x
        negative = (u >> 63) != 0
        # y = int(x) - x, where int(x) is an integer neighbor of x
        y = x + _TOINT - _TOINT - x
        if negative:
            y = x - _TOINT + _TOINT - x
        if e <= 0x3FF - 1:
            # |x| < 1
            return coerce(-0.0 if negative else 1.0, typ)
        if y < 0:
            return coerce(x + y + 1.0, typ)
        return coerce(x + y, typ)
    compile_error("only f32 or f64")

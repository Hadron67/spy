from typing import cast

from spy.compiler import comptime, f32, f64, func, typeof, u32, u64

from ..core import Numeric, bitcast, compile_error
from ..reflect import IntType, type_info


@func(inline=True)
def abs[T: Numeric](x: T) -> T:
    """The absolute value of ``x``: ``fabs(x)`` for an ``f32``/``f64`` (the sign
    bit cleared - spy has no ``fenv``), and ``x``/``-x`` for an integer."""
    comptime()
    t = typeof(x)
    if isinstance(t2 := type_info(t), IntType):
        if t2.signed:
            return -x if x < cast(T, 0) else x
        return x
    if t == f32:
        return cast(T, bitcast(bitcast(x, u32) & 0x7FFFFFFF, f32))
    elif t == f64:
        return cast(T, bitcast(bitcast(x, u64) & 0x7FFFFFFFFFFFFFFF, f64))
    compile_error("only f32 or f64 or integer")

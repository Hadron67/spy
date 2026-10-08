"""Integer casts (``std.int``).

``truncate`` is a builtin (implemented by ``interp._builtin_truncate``); the
``(try_)int_cast`` functions are ordinary spy functions built on top of
compile-time reflection (``std.reflect``) plus ``bitcast``/``coerce``.  They
live here rather than in ``std.core`` so that they can import ``std.reflect``
(``reflect`` imports ``core``, so a ``core``-side import would be circular).
"""

from ..compiler import builtin_func, func, struct, typeof
from ..compiler.syntax import Comptime, comptime
from .core import bitcast, coerce, panic, sstr
from .reflect import IntType, reify, type_info


@builtin_func
def truncate[T](value, typ: type[T]) -> T: ...


@struct()
class IntCastError(Exception):
    """``try_int_cast`` raises this when the value does not fit the target
    integer type (see ``int_cast``)."""


def valid_range(bits: int, signed: bool) -> tuple[int, int]:
    """The inclusive range ``(min, max)`` of the ``bits``-wide integer type of
    the given signedness: ``[0, 2**bits - 1]`` when unsigned, or
    ``[-2**(bits-1), 2**(bits-1) - 1]`` when signed.  ``bits`` and ``signed``
    are compile-time, so the range is computed at compile time."""
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


@func(exceptions=IntCastError)
def try_int_cast[T](value, typ: type[T]) -> T:
    """Cast the integer ``value`` to the integer type ``typ``.

    A cast that needs no check (the source range is contained in the target's)
    returns the value directly.  Otherwise the value is checked against the
    target's range and, when it fits, narrowed with ``truncate`` (and
    ``bitcast`` when the signedness changes); an out-of-range value raises
    ``IntCastError``.  A non-integer operand is returned as is - the ``-> T``
    return annotation makes the compiler reject a conversion that does not
    type check.
    """
    comptime()
    src = type_info(typeof(value))
    comptime()
    dst = type_info(typ)
    if isinstance(s := src, IntType) and isinstance(d := dst, IntType):
        comptime()
        sb = s.bits
        comptime()
        rs = s.signed
        comptime()
        db = d.bits
        comptime()
        rd = d.signed

        # the range each type holds, and whether the source's is contained
        # in the target's: then no value can fail, and the value is returned
        # as is (the compiler widens it)
        comptime()
        src_min, src_max = valid_range(sb, rs)
        comptime()
        dst_min, dst_max = valid_range(db, rd)
        contained: Comptime = src_min >= dst_min and src_max <= dst_max
        if contained:
            return value

        # otherwise the value may not fit.  It is checked against the target
        # range clamped to the source one, so that both bounds are
        # representable in the source type (comparing against the raw target
        # bounds could form a constant that overflows it).
        comptime()
        lo = dst_min
        if src_min > dst_min:
            lo = src_min
        comptime()
        hi = dst_max
        if src_max < dst_max:
            hi = src_max
        if value >= lo and value <= hi:
            if rs == rd:
                return truncate(value, typ)
            if sb == db:
                return bitcast(value, typ)
            comptime()
            mid = IntType(db, rs)
            if sb > db:
                return bitcast(truncate(value, reify(mid)), typ)
            return bitcast(coerce(value, reify(mid)), typ)
        raise IntCastError()
    return value


def int_cast[T](value, typ: type[T]) -> T:
    """``try_int_cast`` that panics instead of raising ``IntCastError``."""
    try:
        return try_int_cast(value, typ)
    except IntCastError:
        panic(sstr(b'integer cast out of range'))

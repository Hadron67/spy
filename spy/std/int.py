"""Integer casts (``std.int``).

``truncate``, ``int_to_float_unchecked`` and ``float_to_int_unchecked`` are
builtins; the ``(try_)int_cast`` and ``(try_)int_to_float``/``(try_)float_to_int``
functions are ordinary spy functions built on top of compile-time reflection
(``std.reflect``) plus those builtins.  They live here rather than in
``std.core`` so that they can import ``std.reflect`` (``reflect`` imports
``core``, so a ``core``-side import would be circular).
"""

from ..compiler import builtin_func, f32, f64, func, struct, typeof
from ..compiler.syntax import Comptime, comptime
from .core import (
    bitcast,
    coerce,
    float_to_int_unchecked,
    int_to_float_unchecked,
    panic,
    sstr,
)
from .reflect import IntType, reify, type_info


@builtin_func
def truncate[T](value, typ: type[T]) -> T: ...


@struct()
class IntCastError(Exception):
    """``try_int_cast`` raises this when the value does not fit the target
    integer type (see ``int_cast``)."""


@func(inline=True)
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


@func(inline=True)
def int_cast[T](value, typ: type[T]) -> T:
    """``try_int_cast`` that panics instead of raising ``IntCastError``."""
    try:
        return try_int_cast(value, typ)
    except IntCastError:
        panic(sstr(b'integer cast out of range'))


@struct()
class FloatCastError(Exception):
    """``try_int_to_float``/``try_float_to_int`` raise this when the value
    cannot be converted without loss (see ``int_to_float``/``float_to_int``)."""


@func(exceptions=FloatCastError)
def try_int_to_float[T: float](value, typ: type[T]) -> T:
    """Cast the integer ``value`` to the float type ``typ`` exactly.

    A conversion that cannot lose precision - the whole source range stays
    within the integers a float of ``typ`` represents exactly, ``[-2**p, 2**p]``
    for the precision ``p`` (24 for ``f32``, 53 for ``f64``) - needs no check.
    Otherwise the value is checked against that range clamped to the source
    range (so that both bounds are representable in the source type) and the
    conversion is done by ``int_to_float_unchecked``; an integer that cannot be
    represented exactly raises ``FloatCastError``.  A value whose type is not an
    integer, or a target that is not ``f32``/``f64``, is handed to the unchecked
    conversion, which reports the mismatch.
    """
    comptime()
    src = type_info(typeof(value))
    if isinstance(s := src, IntType):
        comptime()
        sb = s.bits
        comptime()
        rs = s.signed
        # the precision of the target float type; a target that is neither
        # ``f32`` nor ``f64`` is not a float type, and the unchecked conversion
        # below reports it
        comptime()
        p = 0
        if typ == f32:
            p = 24
        elif typ == f64:
            p = 53
        else:
            return int_to_float_unchecked(value, typ)

        # the exact-integer range of the float, clamped to the source range:
        # every integer in ``[lo, hi]`` converts without loss
        comptime()
        src_min, src_max = valid_range(sb, rs)
        comptime()
        exact_min = -(1 << p)
        comptime()
        exact_max = 1 << p
        contained: Comptime = src_min >= exact_min and src_max <= exact_max
        if contained:
            return int_to_float_unchecked(value, typ)
        comptime()
        lo = exact_min
        if src_min > exact_min:
            lo = src_min
        comptime()
        hi = exact_max
        if src_max < exact_max:
            hi = src_max
        if value >= lo and value <= hi:
            return int_to_float_unchecked(value, typ)
        raise FloatCastError()
    return int_to_float_unchecked(value, typ)


@func(inline=True)
def int_to_float[T: float](value, typ: type[T]) -> T:
    """``try_int_to_float`` that panics instead of raising ``FloatCastError``."""
    try:
        return try_int_to_float(value, typ)
    except FloatCastError:
        panic(sstr(b'integer is not exactly representable as a float'))


@func(exceptions=FloatCastError)
def try_float_to_int[T: int](value, typ: type[T]) -> T:
    """Cast the float ``value`` to the integer type ``typ``, truncating towards
    zero.

    The truncated value has to fit ``typ``: a signed target holds
    ``[-2**(b-1), 2**(b-1) - 1]`` and an unsigned one ``[0, 2**b - 1]``, so the
    check is ``-2**(b-1) <= value < 2**(b-1)`` or ``0 <= value < 2**b``.  Both
    bounds are powers of two, exactly representable in any float, so the
    comparison is exact - and false for a NaN or an infinity.  A value that does
    not fit raises ``FloatCastError``; the conversion itself is
    ``float_to_int_unchecked``.
    """
    comptime()
    dst = type_info(typ)
    if isinstance(d := dst, IntType):
        comptime()
        db = d.bits
        comptime()
        rd = d.signed
        comptime()
        lo = 0
        if rd:
            lo = -(1 << (db - 1))
        comptime()
        hi = 1 << db
        if rd:
            hi = 1 << (db - 1)
        # the bounds are powers of two, exactly representable in ``value``'s
        # float type: materialize them as floats so the comparison needs no
        # int/float peer
        comptime()
        ftype = typeof(value)
        comptime()
        flo = coerce(lo, ftype)
        comptime()
        fhi = coerce(hi, ftype)
        if value >= flo and value < fhi:
            return float_to_int_unchecked(value, typ)
        raise FloatCastError()
    return float_to_int_unchecked(value, typ)


@func(inline=True)
def float_to_int[T: int](value, typ: type[T]) -> T:
    """``try_float_to_int`` that panics instead of raising ``FloatCastError``."""
    try:
        return try_float_to_int(value, typ)
    except FloatCastError:
        panic(sstr(b'float is out of range for the integer type'))

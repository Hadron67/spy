from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i8,
    i32,
    i64,
    u8,
    u32,
    u64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Comptime,
    ConstComptime,
    Opaque,
    Option,
    Ptr,
    comptime,
    ref,
)
from ..std.core import coerce, const_array, typed_undefined

# ---------------------------------------------------------------------------
# ``std.core.coerce(value, T)``: materialize ``value`` as the spy type ``T`` -
# the same conversion a store into a location of ``T`` performs.  A
# compile-time value is converted in Python and stays compile-time; a runtime
# value gets the numeric conversion ``T`` needs (widening or narrowing)
# ---------------------------------------------------------------------------


@func()
def coerce_widen(x: u32) -> u64:
    return coerce(x, u64)


@func()
def coerce_narrow(x: i32) -> i8:
    return coerce(x, i8)


@func()
def coerce_int_to_float(x: i32) -> f64:
    return coerce(x, f64)


@func()
def coerce_comptime_fold() -> u64:
    # a compile-time value folds in Python, so the comparison is a compile-time
    # bool: the ``else`` branch (whose coerce does not type check) is dead and
    # is never typed
    comptime()
    x: f64 = 1.0
    if coerce(x, f64) == 1.0:
        return 1
    return coerce(None, u64)


@func()
def coerce_option_present(x: i32) -> Option[i32]:
    return coerce(x, Option[i32])  # pyright: ignore[reportArgumentType]


@func()
def coerce_option_absent() -> Option[i32]:
    return coerce(None, Option[i32])  # pyright: ignore[reportArgumentType]


@func()
def coerce_pointer_to_int(x: i32) -> u8:
    # an address has no numeric conversion to an integer
    p = ref(x)
    return coerce(p, u8)


@func()
def typed_undefined_of_a_pointer_type() -> i32:
    # the type argument is a *value* (a type-constructor subscript ``Ptr[...]``)
    # rather than a bare type name: the inline ``type[T]`` parameter's slot is
    # declared as ``type`` (see ``interp._call_function_entry``), not as the
    # pointer type its argument solves ``T`` to
    p: Ptr[Opaque] = typed_undefined(Ptr[Opaque])
    spy_typeof(p)
    return 1


@func()
def typed_undefined_const_array_element(x: i32) -> i32:
    # the element type comes from the inline ``type[T]`` call: a runtime pointer,
    # with the array length inferred (see ``std.core.const_array``)
    a = const_array(typed_undefined(Ptr[Opaque]), 3)
    return len(a)


# ---------------------------------------------------------------------------
# compile-time values other than the scalars: a compile-time pointer (a place),
# an aggregate value, a union value and ``Any`` (see ``sval.coerce_const``)
# ---------------------------------------------------------------------------


@struct()
class CoercePair:
    a: i32
    b: i32


@struct()
class CoerceOtherPair:
    a: i32
    b: i32


@struct()
class CoerceUA:
    x: i32


@struct()
class CoerceUB:
    y: i64


@func()
def coerce_comptime_pointer() -> i32:
    # a compile-time pointer converts to a pointer type of the same pointee - a
    # const place included (its writability is decided by its box, not by the
    # view): ``ref`` of a ``ConstComptime`` gives a ``*const i32``
    n: ConstComptime[i32] = 5
    p = coerce(ref(n), Ptr[i32])
    return p[...]


@func()
def coerce_comptime_pointer_of_another_pointee() -> u8:
    # a pointer does not convert between pointee types
    n: Comptime[i32] = 5
    return coerce(ref(n), Ptr[u8])[...]


@func()
def coerce_any_takes_a_pointer() -> i32:
    # ``Any`` names no fixed type: a compile-time pointer is a value of it (it
    # used to be rejected - only a scalar converted to ``Any``)
    n: Comptime[i32] = 7
    p = coerce(ref(n), Any)  # pyright: ignore[reportArgumentType]
    return p[...]


@func()
def coerce_comptime_aggregate() -> i32:
    # a compile-time aggregate value converts to its own type
    comptime()
    p = CoercePair(2, 3)
    q = coerce(p, CoercePair)
    return q.a * 10 + q.b


@func()
def coerce_comptime_aggregate_of_another_type() -> i32:
    # an aggregate is ever only its own type, whatever its fields are
    comptime()
    p = CoercePair(2, 3)
    q = coerce(p, CoerceOtherPair)
    return q.a


@func()
def coerce_comptime_union_widens() -> i32:
    # a compile-time union value converts to a union that also holds its variants
    comptime()
    u: CoerceUA | CoerceUB = CoerceUA(3)
    v = coerce(u, CoerceUA | CoerceUB | u8)  # pyright: ignore[reportArgumentType]
    if isinstance(a := v, CoerceUA):
        return a.x
    return -1


@func()
def coerce_comptime_option_place() -> i32:
    # the compile-time *place* of an option is read as the value of the option
    ch: ConstComptime[Option[i32]] = 5
    o: Option[i32] = coerce(ref(ch), Option[i32])  # pyright: ignore[reportArgumentType]
    if (v := o) is not None:
        return v
    return -1


@func()
def coerce_comptime_union_place() -> i32:
    # likewise for the place of a tagged union
    u: ConstComptime[CoerceUA | CoerceUB] = CoerceUA(4)
    v: CoerceUA | CoerceUB = coerce(ref(u), CoerceUA | CoerceUB)  # pyright: ignore[reportArgumentType]
    if isinstance(a := v, CoerceUA):
        return a.x
    return -1


class SpyCoerceTest(TestCase):
    """``std.core.coerce(value, T)`` materializes a value as the spy type ``T``,
    folding a compile-time value in Python and emitting the numeric conversion
    a runtime one needs."""

    def test_widen(self) -> None:
        self.assertEqual(coerce_widen(0xFFFFFFFF), 0xFFFFFFFF)

    def test_narrow(self) -> None:
        self.assertEqual(coerce_narrow(100), 100)
        # the high bits are dropped: 128 truncates to the i8 value -128
        self.assertEqual(coerce_narrow(128), -128)

    def test_int_to_float(self) -> None:
        self.assertEqual(coerce_int_to_float(3), 3.0)

    def test_comptime_fold(self) -> None:
        self.assertEqual(coerce_comptime_fold(), 1)

    def test_option(self) -> None:
        self.assertEqual(coerce_option_present(7), 7)
        self.assertIsNone(coerce_option_absent())

    def test_incompatible_value(self) -> None:
        with self.assertRaises(CompileError):
            coerce_pointer_to_int(1)

    def test_typed_undefined_of_a_pointer_type(self) -> None:
        # the inline ``type[T]`` parameter takes the pointer type as a value
        self.assertEqual(typed_undefined_of_a_pointer_type(), 1)

    def test_typed_undefined_as_a_const_array_element(self) -> None:
        self.assertEqual(typed_undefined_const_array_element(0), 3)

    def test_a_compile_time_pointer(self) -> None:
        self.assertEqual(coerce_comptime_pointer(), 5)

    def test_a_pointer_of_another_pointee(self) -> None:
        with self.assertRaises(CompileError):
            coerce_comptime_pointer_of_another_pointee()

    def test_any_takes_a_pointer(self) -> None:
        self.assertEqual(coerce_any_takes_a_pointer(), 7)

    def test_a_compile_time_aggregate(self) -> None:
        self.assertEqual(coerce_comptime_aggregate(), 23)

    def test_an_aggregate_of_another_type(self) -> None:
        with self.assertRaises(CompileError):
            coerce_comptime_aggregate_of_another_type()

    def test_a_compile_time_union_value(self) -> None:
        self.assertEqual(coerce_comptime_union_widens(), 3)

    def test_a_compile_time_option_place(self) -> None:
        self.assertEqual(coerce_comptime_option_place(), 5)

    def test_a_compile_time_union_place(self) -> None:
        self.assertEqual(coerce_comptime_union_place(), 4)


all_tests = [
    SpyCoerceTest,
]

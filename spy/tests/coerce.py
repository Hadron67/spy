from typing import Any, Literal
from unittest import TestCase

from ..compiler import (
    CompileError,
    f32,
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
    Array,
    Comptime,
    ConstComptime,
    ConstMultiPtr,
    InferComptime,
    MultiPtr,
    Opaque,
    Option,
    Ptr,
    array,
    comptime,
    ref,
)
from ..std.core import coerce, const_array, typed_undefined
from ..std.float import float_truncate
from ..std.int import int_to_float, truncate

# ---------------------------------------------------------------------------
# ``std.core.coerce(value, T)``: materialize ``value`` as the spy type ``T`` -
# the same conversion a store into a location of ``T`` performs.  A
# compile-time value is converted in Python and stays compile-time; a runtime
# value only widens (``src.is_subtype_of(dst)``) - a narrowing or an
# integer -> float conversion has to be spelled out (``std.int.truncate``,
# ``std.float.float_truncate``, ``std.int.int_to_float``)
# ---------------------------------------------------------------------------


@func()
def coerce_widen(x: u32) -> u64:
    return coerce(x, u64)


@func()
def coerce_narrow(x: i32) -> i8:
    # a runtime narrowing is no longer an implicit conversion: ``coerce`` only
    # widens, so this is a compile error (see ``truncate`` below)
    return coerce(x, i8)


@func()
def coerce_int_to_float(x: i32) -> f64:
    # likewise a runtime integer no longer converts to a float implicitly
    return coerce(x, f64)


@func()
def truncate_narrow(x: i32) -> i8:
    # the explicit narrowing: ``truncate`` to the same-signedness type
    return truncate(x, i8)


@func()
def int_to_float_convert(x: i32) -> f64:
    return int_to_float(x, f64)


@func()
def float_truncate_narrow(x: f64) -> f32:
    return float_truncate(x, f32)


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


@struct()
class ZeroSized:
    pass


@func()
def typed_undefined_of_a_zero_sized_tuple() -> i32:
    # a zero-sized tuple has a unit value, exactly like a zero-sized struct: an
    # inferred (runtime) local of the type therefore holds no storage instead of
    # failing for want of a runtime representation
    t: InferComptime = (ZeroSized(), ZeroSized())
    u = typed_undefined(spy_typeof(t))
    spy_typeof(u)
    return 1


@func()
def zero_sized_tuple_local() -> i32:
    # the same for a plain inferred local: the tuple is a zero-sized value
    t = (ZeroSized(), ZeroSized())
    spy_typeof(t)
    return 1


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
def coerce_array_pointer_to_multi_pointer() -> i32:
    # a pointer to an array converts to the multi pointer of its elements: the
    # same address, now viewed one element at a time
    a: ConstComptime[Array[i32, Literal[3]]] = const_array(7, 3)
    m: Comptime = coerce(ref(a), MultiPtr[i32])
    return m[0] * 100 + m[1] * 10 + m[2]


@func()
def coerce_multi_pointer_to_pointer() -> i32:
    # and a multi pointer converts on to the single pointer of its element
    a: ConstComptime[Array[i32, Literal[3]]] = const_array(4, 3)
    m: Comptime = coerce(ref(a), MultiPtr[i32])
    p: Comptime = coerce(m, Ptr[i32])
    return p[...]


@func()
def coerce_array_pointer_of_a_runtime_array(x: i32) -> i32:
    # the same conversion for a runtime array: the address is the same value
    a = const_array(x, 3)
    m = coerce(ref(a), MultiPtr[i32])
    return m[2]


@func()
def coerce_materialized_pointer_to_an_array() -> i32:
    # here the multi pointer is *materialized* (it has no ``Comptime``
    # annotation): the storage behind it holds the whole array, not just the
    # element the view names, so indexing past the first element stays in bounds
    a: ConstComptime[Array[i32, Literal[3]]] = const_array(7, 3)
    m = coerce(ref(a), MultiPtr[i32])
    return m[0] * 100 + m[1] * 10 + m[2]


@func()
def coerce_materialized_const_pointer_to_an_array() -> i32:
    # likewise for a const view: the global constant it lowers to holds the array
    a: ConstComptime[Array[i32, Literal[3]]] = const_array(7, 3)
    m: ConstMultiPtr[i32] = coerce(ref(a), ConstMultiPtr[i32])
    return m[0] * 100 + m[1] * 10 + m[2]


@func()
def coerce_materialized_element_at_an_offset() -> i32:
    # the storage starts at the element the pointer names - the array's tail from
    # there on, so a later element reads its own value
    a: ConstComptime[Array[i32, Literal[3]]] = array(7, 8, 9, length=3)
    m: Comptime = coerce(ref(a), MultiPtr[i32])
    r = ref(m[1])
    return r[...]


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

class SpyCoerceTest(TestCase):
    """``std.core.coerce(value, T)`` materializes a value as the spy type ``T``,
    folding a compile-time value in Python and emitting the numeric conversion
    a runtime one needs."""

    def test_widen(self) -> None:
        self.assertEqual(coerce_widen(0xFFFFFFFF), 0xFFFFFFFF)

    def test_narrow(self) -> None:
        # a runtime narrowing is a compile error now; the explicit form is
        # ``std.int.truncate``, which drops the high bits (128 -> -128)
        with self.assertRaises(CompileError):
            coerce_narrow(100)
        self.assertEqual(truncate_narrow(100), 100)
        self.assertEqual(truncate_narrow(128), -128)

    def test_int_to_float(self) -> None:
        # a runtime integer no longer converts to a float implicitly; the
        # explicit form is ``std.int.int_to_float``
        with self.assertRaises(CompileError):
            coerce_int_to_float(3)
        self.assertEqual(int_to_float_convert(3), 3.0)

    def test_float_narrow(self) -> None:
        # ``std.float.float_truncate`` is the explicit ``f64`` -> ``f32``
        # narrowing
        self.assertAlmostEqual(float_truncate_narrow(1.5), 1.5)

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

    def test_typed_undefined_of_a_zero_sized_tuple(self) -> None:
        # a zero-sized tuple carries a unit value, so a runtime local of it is
        # legal even though it holds no storage (see ``sval.TupleType``)
        self.assertEqual(typed_undefined_of_a_zero_sized_tuple(), 1)

    def test_a_zero_sized_tuple_local(self) -> None:
        self.assertEqual(zero_sized_tuple_local(), 1)

    def test_a_compile_time_pointer(self) -> None:
        self.assertEqual(coerce_comptime_pointer(), 5)

    def test_a_pointer_of_another_pointee(self) -> None:
        with self.assertRaises(CompileError):
            coerce_comptime_pointer_of_another_pointee()

    def test_a_pointer_to_an_array(self) -> None:
        self.assertEqual(coerce_array_pointer_to_multi_pointer(), 100 * 7 + 10 * 7 + 7)

    def test_a_multi_pointer(self) -> None:
        self.assertEqual(coerce_multi_pointer_to_pointer(), 4)

    def test_a_pointer_to_a_runtime_array(self) -> None:
        self.assertEqual(coerce_array_pointer_of_a_runtime_array(9), 9)

    def test_a_materialized_pointer_to_an_array(self) -> None:
        self.assertEqual(coerce_materialized_pointer_to_an_array(), 100 * 7 + 10 * 7 + 7)

    def test_a_materialized_const_pointer_to_an_array(self) -> None:
        self.assertEqual(coerce_materialized_const_pointer_to_an_array(), 100 * 7 + 10 * 7 + 7)

    def test_a_materialized_element_at_an_offset(self) -> None:
        self.assertEqual(coerce_materialized_element_at_an_offset(), 8)

    def test_any_takes_a_pointer(self) -> None:
        self.assertEqual(coerce_any_takes_a_pointer(), 7)

    def test_a_compile_time_aggregate(self) -> None:
        self.assertEqual(coerce_comptime_aggregate(), 23)

    def test_an_aggregate_of_another_type(self) -> None:
        with self.assertRaises(CompileError):
            coerce_comptime_aggregate_of_another_type()

    def test_a_compile_time_union_value(self) -> None:
        self.assertEqual(coerce_comptime_union_widens(), 3)

all_tests = [
    SpyCoerceTest,
]

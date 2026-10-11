from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i8,
    i32,
    u8,
    u32,
    u64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func
from ..compiler.syntax import (
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


all_tests = [
    SpyCoerceTest,
]

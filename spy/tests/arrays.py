from typing import Literal
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    i64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Array,
    Comptime,
    array,
    ref,
)
from .comptime_structs import runtime_struct_unroll
from .structs import Blank, Small

# ---------------------------------------------------------------------------
# arrays: ``syntax.array(a1, a2, ...)`` builds an array of the elements it is
# given - in place, like a struct construction: the length of the array is the
# number of elements and its element type either the one the place it is built
# in declares or the common type of the elements - and ``a[i]`` is the *place*
# of the i-th element, read and written through like a field.  The spy type of
# an array is ``syntax.Array[T, N]``: an annotation writes the length out
# (``array`` cannot, see ``syntax``), the compiler takes it from the arguments.
# ---------------------------------------------------------------------------


@struct()
class Vec2:
    """A struct that holds an array: ``data`` is a field of the struct, and
    the elements of the array are elements of that field."""

    data: Array[i32, Literal[2]]

    def total(self) -> i32:
        return self.data[0] + self.data[1]


@func()
def read_element(x: i32) -> i32:
    a = array(x, 2)
    return a[0]


@func()
def sum_elements(x: i32) -> i32:
    a = array(x, x + 1)
    return a[0] + a[1]


@func()
def write_element(x: i32, v: i32) -> i32:
    a = array(x, x)
    a[1] = v
    return a[1]


@func()
def add_to_element(x: i32) -> i32:
    a = array(x, x + 1)
    a[0] += 10
    return a[0] * 100 + a[1]


@func()
def array_of_structs(x: i32) -> i32:
    a = array(Small(x, 1), Small(x, 2))
    return a[0].a + a[1].b


@func()
def write_struct_element(x: i32) -> i32:
    a = array(Small(x, 1), Small(x, 2))
    a[1].b = 7
    return a[1].b


@func()
def nested_array(x: i32) -> i32:
    # an array of arrays: each inner construction fills an element of the
    # outer array in place, and an element of an element is a place like any
    # other
    a = array(array(x, x + 1), array(x + 2, x + 3))
    return a[0][1] + a[1][0]


@func()
def array_through_ptr(x: i32) -> i32:
    # the address of an array is a pointer: ``p[...][i]`` is the i-th element
    # of the array ``p`` points at
    a = array(x, x + 1)
    p = ref(a)
    return p[...][0] + p[...][1]


@func()
def copy_array(x: i32) -> i32:
    a = array(x, x + 1)
    b = a
    return b[0] + b[1]


@func()
def pass_array(a: Array[i32, Literal[2]], i: i32) -> i32:
    return a[i]


@func()
def call_pass_array(x: i32) -> i32:
    # ``length`` is what the Python type checker needs to see the length of the
    # array (``array`` cannot tell it from the elements); the compiler takes it
    # from the number of elements either way
    return pass_array(array(x, x + 1, length=2), 1)


@func()
def call_pass_array_runtime_index(x: i32, i: i32) -> i32:
    # a runtime index is a runtime address (the array's own type is what the
    # address arithmetic uses)
    return pass_array(array(x, x + 1, length=2), i)


@func()
def return_array(x: i32) -> Array[i32, Literal[2]]:
    return array(x, 3, length=2)


@func()
def use_returned_array(x: i32) -> i32:
    b = return_array(x)
    return b[0] + b[1]


@func()
def wide_array(x: i64) -> i64:
    # 32 bytes: an array past the by-value limit, so one is returned (and
    # passed) through a result pointer
    a = array(x, x + 1, x + 2, x + 3)
    return a[0] + a[3]


@func()
def array_in_struct(x: i32) -> i32:
    v = Vec2(array(x, x + 1, length=2))
    return v.total() + v.data[0]


@func()
def write_struct_array(x: i32) -> i32:
    v = Vec2(array(x, x + 1, length=2))
    v.data[1] = x
    return v.total()


@func()
def length_keyword(x: i32) -> i32:
    # ``length`` is what the *Python* type checker needs (``array`` cannot tell
    # the length of the array from its elements); the compiler takes it from
    # the number of elements and ignores the keyword
    a = array(x, x + 1, length=2)
    return a[0] + a[1]


@func()
def discard_array(x: i32) -> i32:
    # an expression statement: the array is built into a temporary nobody reads
    array(x, x + 1)
    return x


@func()
def zero_sized_array(x: i32) -> i32:
    # an array of zero-sized elements holds no storage: it is written like any
    # other (the writes go nowhere), and every element equals the unit value of
    # the element type
    a = array(Blank(None), Blank(None))
    a[1] = Blank(None)
    return x if spy_typeof(a[0]) == Blank else x + 1


@func()
def empty_array_result() -> Array[i32, Literal[0]]:
    # a zero-length array is zero-sized whatever its element type is
    return array()


@func()
def pass_empty_array(a: Array[i32, Literal[0]]) -> i32:
    return 4


@func()
def use_empty_array() -> i32:
    return pass_empty_array(empty_array_result())


@func()
def element_out_of_bounds(x: i32) -> i32:
    a = array(x, x)
    return a[2]


@func()
def wrong_element_count(x: i32) -> Array[i32, Literal[2]]:
    # two elements are declared, one is given: the compiler counts the
    # arguments, not the keyword
    return array(x, length=2)


@func()
def untyped_elements(x: i32) -> i32:
    a = array(1, 2)
    return a[0] + x


@func()
def unknown_array_keyword(x: i32) -> i32:
    a = array(x, nope=1)  # pyright: ignore
    return a[0] + x


# ---------------------------------------------------------------------------
# compile-time arrays: an array built in an inline slot - an expression
# temporary or a ``Comptime`` variable - is an aggregate (like a struct) whose
# elements are their own compile-time places (see ``interp.ComptimeAggregatePtr``)
# ---------------------------------------------------------------------------


@func()
def comptime_array_element() -> i32:
    a: Comptime = array(1, 2)
    return a[1]


@func()
def comptime_array_copy() -> i32:
    a: Comptime = array(1, 2)
    b: Comptime = a
    return b[1]


@func()
def comptime_array_of_structures() -> i32:
    a: Comptime = array(Small(1, 2), Small(3, 4))
    b: Comptime = a
    return b[1].a * 10 + b[0].b


@func()
def comptime_nested_array() -> i32:
    a: Comptime = array(array(1, 2), array(3, 4))
    return a[1][0]


@func()
def comptime_array_with_a_runtime_element(x: i32) -> i32:
    # a runtime element is a runtime place of its own: the other element is
    # still compile-time
    a: Comptime = array(x, 2)
    return a[0] + a[1]


@func()
def comptime_array_from_a_runtime_value(x: i32) -> i32:
    # a whole runtime *array* value assigned to a compile-time variable: its
    # elements are read out of the value itself (``mir.ExtractValue``)
    a = array(x, x + 1)
    c: Comptime = a
    return c[1]


class SpyComptimeArrayTest(TestCase):
    """Compile-time arrays: an array built in an inline slot - an expression
    temporary or a ``Comptime`` variable - is an aggregate whose elements are
    their own compile-time places, like a struct's fields."""

    def test_element_read(self) -> None:
        self.assertEqual(comptime_array_element(), 2)

    def test_copy_is_a_value(self) -> None:
        self.assertEqual(comptime_array_copy(), 2)

    def test_array_of_structures(self) -> None:
        self.assertEqual(comptime_array_of_structures(), 32)

    def test_nested_array(self) -> None:
        self.assertEqual(comptime_nested_array(), 3)

    def test_runtime_element(self) -> None:
        self.assertEqual(comptime_array_with_a_runtime_element(5), 7)

    def test_a_whole_runtime_value(self) -> None:
        # a whole runtime array value into a compile-time variable: the elements
        # are read out of the value itself
        self.assertEqual(comptime_array_from_a_runtime_value(3), 4)

    def test_a_runtime_struct_cannot_unroll(self) -> None:
        with self.assertRaises(CompileError):
            runtime_struct_unroll()


class SpyArrayTest(TestCase):
    """Arrays: ``array(a1, a2, ...)`` builds an array in place - the length is
    the number of elements, the element type the one the place it is built in
    declares or the common type of the elements - and ``a[i]`` is the place of
    the i-th element, read and written through like a field."""

    def test_read_and_sum_elements(self) -> None:
        self.assertEqual(read_element(5), 5)
        self.assertEqual(sum_elements(5), 11)

    def test_write_element(self) -> None:
        self.assertEqual(write_element(5, 9), 9)
        self.assertEqual(add_to_element(5), 1506)

    def test_elements_are_structs(self) -> None:
        self.assertEqual(array_of_structs(5), 7)
        self.assertEqual(write_struct_element(5), 7)

    def test_nested_array(self) -> None:
        self.assertEqual(nested_array(5), 13)

    def test_array_through_pointer(self) -> None:
        self.assertEqual(array_through_ptr(5), 11)

    def test_array_value_is_copied(self) -> None:
        self.assertEqual(copy_array(5), 11)

    def test_array_argument_and_result(self) -> None:
        self.assertEqual(call_pass_array(5), 6)
        self.assertEqual(call_pass_array_runtime_index(5, 0), 5)
        self.assertEqual(call_pass_array_runtime_index(5, 1), 6)
        self.assertEqual(use_returned_array(5), 8)

    def test_wide_array(self) -> None:
        # 32 bytes: the array is a by-reference aggregate at the call boundary
        self.assertEqual(wide_array(5), 13)

    def test_array_field(self) -> None:
        self.assertEqual(array_in_struct(5), 16)
        self.assertEqual(write_struct_array(5), 10)

    def test_length_keyword_is_ignored(self) -> None:
        self.assertEqual(length_keyword(5), 11)

    def test_discarded_array(self) -> None:
        self.assertEqual(discard_array(5), 5)

    def test_zero_sized_array(self) -> None:
        self.assertEqual(zero_sized_array(5), 5)

    def test_empty_array(self) -> None:
        self.assertEqual(use_empty_array(), 4)

    def test_element_out_of_bounds(self) -> None:
        with self.assertRaises(CompileError):
            element_out_of_bounds(5)

    def test_wrong_element_count(self) -> None:
        with self.assertRaises(CompileError):
            wrong_element_count(5)

    def test_untyped_elements_are_rejected(self) -> None:
        # an untyped integer literal has no runtime type of its own, so an
        # array of them has no element type either
        with self.assertRaises(CompileError):
            untyped_elements(5)

    def test_unknown_keyword(self) -> None:
        with self.assertRaises(CompileError):
            unknown_array_keyword(5)


all_tests = [
    SpyComptimeArrayTest,
    SpyArrayTest,
]

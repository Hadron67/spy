"""Tuples as runtime values.  A tuple now has a real runtime representation - it
is a runtime aggregate of its elements, laid out like a struct - so a tuple
value can be stored in a plain variable, assigned and read element by element.
``t[i]`` (``i`` a compile-time integer) is the *lvalue* of its i-th element, and
``len(t)`` is the arity (a compile-time integer).

A tuple's arity is fixed once it is built: the old ``t += u`` - the in-place,
zero-copy extension of a tuple place that grew the arity by aliasing ``u``'s
element places onto it - is *gone* (the tuple it relied on, "a tuple is its
tuple of element places", is not the model any more; see
``doc/comptime-refactor.task.md`` 7.5).  A tuple is built element by element
instead, and ``t = t + u`` was never supported (there is no tuple ``+``).
"""

from unittest import TestCase

from ..compiler import CompileError, i32, syntax
from ..compiler.dsl import func, struct
from ..compiler.syntax import Comptime

# ---------------------------------------------------------------------------
# building a tuple value element by element
# ---------------------------------------------------------------------------


@func()
def sum_accumulated() -> i32:
    # the ``reduce`` shape: a compile-time loop filling one element per
    # iteration.  The tuple is built first (its arity is fixed), then filled
    t: Comptime = (0, 0, 0)
    syntax.unroll()
    for i in range(3):
        t[i] = i * 10  # pyright: ignore
    return t[0] + t[1] + t[2]


@func()
def concat_two(x: i32, y: i32) -> i32:
    # concatenating two tuples: with a fixed arity the result is built from the
    # elements of both
    a = (x, x + 1)
    b = (y, y + 1)
    t = (a[0], b[0])
    return t[0] + t[1]


@func()
def append_literal() -> i32:
    # a tuple literal is the value itself
    t: Comptime = (5, 6)
    return t[0] + t[1]


# ---------------------------------------------------------------------------
# a non-copyable element is constructed in the tuple's storage, never copied
# ---------------------------------------------------------------------------


@struct(copyable=False)
class Handle:
    id: i32


@func()
def sum_handles_via_tuple(x: i32, y: i32) -> i32:
    # the non-copyable elements are *constructed* directly in the tuple's
    # storage; they are never copied (the whole value is never read - only its
    # ``id`` field, through the element's own place)
    t = (Handle(x), Handle(y))
    return t[0].id + t[1].id


@struct(copyable=False)
class Holder:
    id: i32


@struct(copyable=False)
class HolderBox:
    tag: i32
    h: Holder


@func()
def copy_noncopyable_field_into_a_tuple(x: i32) -> i32:
    # a tuple element is storage: filling it *copies* the value in, so a
    # non-copyable source is rejected (see ``sval.Type.is_copyable``)
    w = HolderBox(0, Holder(x))
    t = (w.h,)
    return t[0].id


# ---------------------------------------------------------------------------
# an element holds the written value: a stable source is copied, not aliased
# ---------------------------------------------------------------------------


@struct(copyable=True)
class Two:
    x: i32
    y: i32


@func()
def element_written_in_a_branch(n: i32) -> i32:
    # a runtime write into a tuple element inside a branch: the element is the
    # tuple's own storage, so reading it after the join is dominated
    t = (n, n)
    if n > 0:
        t[0] = n + 1  # pyright: ignore
    return t[0]


@func()
def element_copies_the_value(x: i32) -> i32:
    # writing a variable into an element copies it: a later write to the
    # variable is not seen through the tuple (there is no aliasing)
    v: i32 = x
    t = (v,)
    v = 100
    return t[0]


@func()
def field_element_copies_the_value(x: i32) -> i32:
    # likewise for a field: the element holds the value the field had
    val = Two(x, x + 1)
    t = (val.x,)
    val.x = 100
    return t[0]


# ---------------------------------------------------------------------------
# only ``+=`` used to extend a tuple; there is no tuple ``+``/``+=`` now
# ---------------------------------------------------------------------------


@func()
def plain_tuple_plus(a: tuple[i32, i32], b: tuple[i32, i32]) -> i32:
    # ``a + b`` is not implemented: there is no tuple ``+``
    return (a + b)[0]


@func()
def tuple_inplace_add(x: i32) -> i32:
    # the augmented assignment knows no ``+=`` for a tuple any more
    t = (x, x)
    t += (1, 2)  # pyright: ignore
    return t[0]


@func()
def tuple_inplace_sub(x: i32) -> i32:
    # neither does it know ``-=`` (it reports the one tuple form it knows)
    t = (x, x)
    t -= (1, 2)  # pyright: ignore
    return t[0]


# ---------------------------------------------------------------------------
# a runtime control flow write into a tuple element
# ---------------------------------------------------------------------------


@func()
def pair2(x: i32) -> tuple[i32, i32]:
    return x, x + 1


@func()
def tuple_literal_loop(x: i32) -> i32:
    # a tuple built from runtime values holds its elements in memory, so a
    # runtime loop may write them
    t = (x, x + 1)
    y: i32 = 0
    while y < 3:
        t[0] = t[0] + 1  # pyright: ignore
        y = y + 1
    return t[0]


@func()
def multi_value_result_loop(x: i32) -> i32:
    # likewise for a tuple a multi-value call packed into a plain variable
    t = pair2(x)
    y: i32 = 0
    while y < 3:
        t[0] = t[0] + 1  # pyright: ignore
        y = y + 1
    return t[0]


class SpyTupleBuildTest(TestCase):
    """Building a tuple value element by element (the old in-place extension
    ``t += u`` is gone)."""

    def test_a_compile_time_loop_filling_elements(self) -> None:
        self.assertEqual(sum_accumulated(), 30)

    def test_building_from_two_tuple_places(self) -> None:
        self.assertEqual(concat_two(3, 4), 7)

    def test_a_literal(self) -> None:
        self.assertEqual(append_literal(), 11)

    def test_a_non_copyable_element_is_constructed_in_place(self) -> None:
        # a successful compile is the assertion: a copy would be rejected
        self.assertEqual(sum_handles_via_tuple(9, 5), 14)

    def test_an_element_written_in_a_runtime_branch(self) -> None:
        # the write happens in a branch and is read after the join
        self.assertEqual(element_written_in_a_branch(3), 4)

    def test_an_element_copies_the_written_value(self) -> None:
        self.assertEqual(element_copies_the_value(7), 7)

    def test_a_field_element_copies_the_written_value(self) -> None:
        self.assertEqual(field_element_copies_the_value(7), 7)


class SpyTupleInplaceRejectedTest(TestCase):
    """There is no tuple ``+``/``+=``: the old in-place extension is gone."""

    def test_plain_tuple_plus_is_not_supported(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            plain_tuple_plus((1, 2), (3, 4))
        self.assertIn("unsupported operator '+'", str(ctx.exception))

    def test_tuple_inplace_add_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tuple_inplace_add(1)
        self.assertIn('a tuple does not support ``+=``', str(ctx.exception))

    def test_another_augmented_operator_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tuple_inplace_sub(1)
        self.assertIn('a tuple does not support ``+=``', str(ctx.exception))

    def test_copying_a_non_copyable_field_into_a_tuple_is_rejected(self) -> None:
        # a tuple element is storage: the non-copyable source cannot be copied in
        with self.assertRaises(CompileError) as ctx:
            copy_noncopyable_field_into_a_tuple(9)
        self.assertIn('non-copyable type', str(ctx.exception))


class SpyTupleRuntimeLoopTest(TestCase):
    """A tuple's elements hold runtime values in memory, so a runtime loop can
    write them."""

    def test_a_tuple_literal(self) -> None:
        self.assertEqual(tuple_literal_loop(10), 13)

    def test_a_multi_value_result(self) -> None:
        self.assertEqual(multi_value_result_loop(10), 13)


all_tests = [
    SpyTupleBuildTest,
    SpyTupleInplaceRejectedTest,
    SpyTupleRuntimeLoopTest,
]

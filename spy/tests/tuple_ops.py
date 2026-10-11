"""``t += u`` on a tuple: a tuple has no storage of its own, so a tuple variable
*is* its tuple of element places (an ``interp.ComptimeTuplePtr``).  ``+=`` grows
it *in place* by aliasing ``u``'s elements onto it - nothing is copied, not even
for an element of a non-copyable type - while ``t = t + u`` is not supported
(there is no tuple ``+``): only the augmented assignment extends a tuple.

The accumulation below is the ``reduce`` shape this exists for: a compile-time
loop appending one (runtime) element per iteration, which a plain assignment
could not do because a committed tuple variable has a fixed arity.
"""

from unittest import TestCase

from ..compiler import CompileError, i32, syntax
from ..compiler.dsl import func, struct
from ..compiler.syntax import Comptime, array

# ---------------------------------------------------------------------------
# extending a tuple place in place
# ---------------------------------------------------------------------------


@func()
def sum_accumulated() -> i32:
    # the ``reduce`` shape: a compile-time loop appending one runtime element per
    # iteration; each iteration's ``v`` is a fresh place, aliased into ``t``
    t: Comptime = ()
    syntax.unroll()
    for i in range(3):
        v: i32 = i * 10
        t += (v,)
        i = i + 1
    return t[0] + t[1] + t[2]


@func()
def concat_two(x: i32, y: i32) -> i32:
    # ``t += u`` where ``u`` is another tuple place: its elements are aliased
    a: Comptime = (x,)
    b: Comptime = (y,)
    a += b
    return a[0] + a[1]


@func()
def append_literal() -> i32:
    # a bare (compile-time) element is materialized in a fresh place of its own
    t: Comptime = ()
    t += (5, 6)
    return t[0] + t[1]


# ---------------------------------------------------------------------------
# a non-copyable element is never copied out of its place
# ---------------------------------------------------------------------------


@struct(copyable=False)
class Handle:
    id: i32


@func()
def sum_handles_via_tuple(x: i32, y: i32) -> i32:
    # a non-copyable element is aliased into the tuple, never copied: were ``+=``
    # to copy, this would fail to compile (loading a non-copyable is rejected,
    # see ``sval.Type.is_copyable``), and the whole value is never read - only
    # its ``id`` field, through the element's own place
    h1 = Handle(x)
    h2 = Handle(y)
    t: Comptime = ()
    t += (h1,)
    t += (h2,)
    return t[0].id + t[1].id


# ---------------------------------------------------------------------------
# a block-local element place is copied, a stable one is aliased
# ---------------------------------------------------------------------------


@struct(copyable=True)
class Two:
    x: i32
    y: i32


@func()
def append_runtime_indexed_element(n: i32) -> i32:
    # a tuple element whose address is *block-local* - the subscript's ``Gep``
    # with a runtime index, inside the branch - is copied into a place of its
    # own, so reading the tuple after the join is dominated (the bug of a raw
    # aliased register; see ``_tuple_extend_element``)
    a = array(n, n + 1, n + 2, n + 3)
    t: Comptime = ()
    if n > 0:
        t += (a[n - 1],)
    return t[0]


@func()
def appended_variable_stays_aliased(x: i32) -> i32:
    # a whole variable's address is stable (an ``Alloca``), so it is aliased: a
    # later write to the variable is seen through the tuple
    v: i32 = x
    t: Comptime = ()
    t += (v,)
    v = 100
    return t[0]


@func()
def appended_field_is_copied(x: i32) -> i32:
    # a field's address is block-local (a ``Gep``), so the element is copied: a
    # later write to the field is *not* seen through the tuple
    val = Two(x, x + 1)
    t: Comptime = ()
    t += (val.x,)
    val.x = 100
    return t[0]


# ---------------------------------------------------------------------------
# a non-copyable block-local element place can be neither copied nor aliased
# ---------------------------------------------------------------------------


@struct(copyable=False)
class Holder:
    id: i32


@struct(copyable=False)
class HolderBox:
    tag: i32
    h: Holder


@func()
def append_noncopyable_block_local_place(x: i32) -> i32:
    # a non-copyable field's address is block-local, so it has no copy to be
    # made and is rejected (see ``_tuple_extend_element``)
    w = HolderBox(0, Holder(x))
    t: Comptime = ()
    t += (w.h,)
    return t[0].id


# ---------------------------------------------------------------------------
# only ``+=`` extends a tuple
# ---------------------------------------------------------------------------


@func()
def plain_tuple_plus(a: tuple[i32, i32], b: tuple[i32, i32]) -> i32:
    # ``a + b`` is not implemented: there is no tuple ``+``, only ``+=``
    return (a + b)[0]


@func()
def tuple_inplace_sub(a: tuple[i32, i32]) -> i32:
    # the augmented assignment only knows ``+=`` for a tuple
    a -= (1, 2)  # pyright: ignore
    return a[0]


# ---------------------------------------------------------------------------
# a runtime control flow write into a tuple element place
# ---------------------------------------------------------------------------


@func()
def pair2(x: i32) -> tuple[i32, i32]:
    return x, x + 1


@func()
def tuple_literal_loop(x: i32) -> i32:
    # a tuple built from runtime values holds its elements in memory, so a
    # runtime loop may write them (see pending-problems.md #3)
    t: Comptime = (x, x + 1)
    y: i32 = 0
    while y < 3:
        t[0] = t[0] + 1
        y = y + 1
    return t[0]


@func()
def multi_value_result_loop(x: i32) -> i32:
    # likewise for a tuple a multi-value call packed into the variable
    t: Comptime = pair2(x)
    y: i32 = 0
    while y < 3:
        t[0] = t[0] + 1
        y = y + 1
    return t[0]


class SpyTupleRuntimeLoopTest(TestCase):
    """A tuple's element places hold runtime values in memory, so a runtime
    loop can write them (the bug of pending-problems.md #3)."""

    def test_a_tuple_literal(self) -> None:
        self.assertEqual(tuple_literal_loop(10), 13)

    def test_a_multi_value_result(self) -> None:
        self.assertEqual(multi_value_result_loop(10), 13)


class SpyTupleExtendTest(TestCase):
    """``t += u``: the in-place extension of a tuple place."""

    def test_accumulating_a_runtime_element_per_iteration(self) -> None:
        self.assertEqual(sum_accumulated(), 30)

    def test_extending_by_another_tuple_place(self) -> None:
        self.assertEqual(concat_two(3, 4), 7)

    def test_appending_a_bare_value(self) -> None:
        self.assertEqual(append_literal(), 11)

    def test_a_non_copyable_element_is_not_copied(self) -> None:
        # a successful compile is the assertion: a copy would be rejected
        self.assertEqual(sum_handles_via_tuple(9, 5), 14)

    def test_a_block_local_place_is_copied(self) -> None:
        # a runtime subscript's address is block-local, so the element is copied
        # (the register-dominance fix): the tuple reads fine after the join
        self.assertEqual(append_runtime_indexed_element(3), 5)

    def test_a_whole_variable_element_stays_aliased(self) -> None:
        # a stable address is aliased, not copied
        self.assertEqual(appended_variable_stays_aliased(7), 100)

    def test_a_field_element_is_copied(self) -> None:
        # a field's address is block-local, so the element is copied
        self.assertEqual(appended_field_is_copied(7), 7)


class SpyTupleExtendRejectedTest(TestCase):
    """Only the augmented ``+=`` extends a tuple; everything else is rejected."""

    def test_plain_tuple_plus_is_not_supported(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            plain_tuple_plus((1, 2), (3, 4))
        self.assertIn("unsupported operator '+'", str(ctx.exception))

    def test_another_augmented_operator_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tuple_inplace_sub((1, 2))
        self.assertIn("only supports in-place extension with '+='", str(ctx.exception))

    def test_a_noncopyable_block_local_place_is_rejected(self) -> None:
        # a block-local place of a non-copyable type cannot be appended: it can
        # be neither aliased (its address does not dominate later reads) nor
        # copied
        with self.assertRaises(CompileError) as ctx:
            append_noncopyable_block_local_place(9)
        self.assertIn('non-copyable type', str(ctx.exception))


all_tests = [
    SpyTupleExtendTest,
    SpyTupleExtendRejectedTest,
    SpyTupleRuntimeLoopTest,
]

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
from ..compiler.syntax import Comptime

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


all_tests = [
    SpyTupleExtendTest,
    SpyTupleExtendRejectedTest,
]

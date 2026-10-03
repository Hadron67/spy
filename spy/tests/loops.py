from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
)
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Ptr,
    ref,
)
from .basics import inc
from .errors import ErrorA, raise_a

# ---------------------------------------------------------------------------
# loops: ``while`` compiles into a dead ``loop`` block whose head re-evaluates
# the condition at every iteration; ``break`` leaves the loop, ``continue``
# jumps back to the head and the ``else`` clause runs only on a natural exit.
# A loop-carried variable is an ordinary block-local slot (memory), so no phi
# is needed across the back edge.
# ---------------------------------------------------------------------------


@func()
def sum_to(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


@func()
def count_up(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        i = i + 1
    return i


@func()
def sum_calls(n: i32) -> i32:
    # a spy call inside the loop body
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + inc(i)
        i = i + 1
    return total


@func()
def find_divisor(n: i32) -> i32:
    # leaves the loop at the first divisor; when there is none the condition
    # fails and the loop exits normally (with ``i == n``)
    i: i32 = 2
    while i < n:
        if n % i == 0:
            break
        i = i + 1
    return i


@func()
def sum_odd(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
        total = total + i
    return total


@func()
def skip_and_stop(n: i32) -> i32:
    # both ``continue`` and ``break`` in the body, each under a runtime ``if``
    i: i32 = 0
    total: i32 = 0
    while i < n:
        i = i + 1
        if i == 2:
            continue
        if i == 5:
            break
        total = total + i
    return total


@func()
def while_else_natural(n: i32) -> i32:
    # the else clause runs when the condition turns false
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        result = 100
    return result + i


@func()
def while_else_break(n: i32) -> i32:
    # ... and is skipped by a ``break``
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
        if i == 2:
            break
    else:
        result = 100
    return result + i


@func()
def while_else_continue(n: i32) -> i32:
    # a ``continue`` does not skip the else clause for good: it runs when the
    # condition finally turns false
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        result = 100
    return result + i


@func()
def nested_loop_sum(n: i32, m: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        j: i32 = 0
        while j < m:
            total = total + i * j
            j = j + 1
        i = i + 1
    return total


@func()
def loop_return(n: i32) -> i32:
    # ``while True`` whose only exit is a ``return``: the code after the loop is
    # dead and the loop never needs an exit block
    steps: i32 = 0
    while True:
        if n == 0:
            return steps
        n = n - 1
        steps = steps + 1


@func()
def loop_void(n: i32) -> None:
    i: i32 = 0
    while i < n:
        i = i + 1


@func()
def assign_outside_from_loop(n: i32) -> i32:
    # a variable declared outside the loop is written in the body: it lives in
    # memory, so the write survives the back edge
    acc: i32 = 0
    i: i32 = 0
    while i < n:
        acc += i
        i = i + 1
    return acc


@func()
def comptime_false_loop(n: i32) -> i32:
    # a compile-time false condition never runs the body and runs the else
    # clause once (only the chosen branch of the head ``if`` is emitted)
    total: i32 = 0
    while False:
        total = total + n
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 1
    return total


@func()
def break_in_try(n: i32) -> i32:
    # the try body may raise (``raise_a`` may), so the handler is live; a
    # ``break`` in the body still leaves the loop directly, without going
    # through the clause
    i: i32 = 0
    while i < 100:
        i = i + 1
        try:
            raise_a(1)
            if i == n:
                break
        except ErrorA:
            return -1
    return i


@func()
def continue_in_try(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < 10:
        i = i + 1
        try:
            raise_a(1)
            if i == n:
                continue
            total = total + i
        except ErrorA:
            return -1
    return total


@func()
def bounded_loop(n: i32) -> i32:
    # ``while True`` whose only exit is an explicit ``break``: the exit block
    # comes from that break alone (the implicit one of the lowering is
    # compile-time dead), and the code after the loop is reachable through it
    i: i32 = 0
    while True:
        if i >= n:
            break
        i = i + 1
    return i + 100


def sum_inline(n: i32) -> i32:
    # an undecorated plain function: its body - loop included - is inlined at
    # the call site, in a frame of its own
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


@func()
def call_sum_inline(n: i32) -> i32:
    return sum_inline(n)


# ---------------------------------------------------------------------------
# for loops: ``for x in iter`` desugars into an explicit iterator loop (see
# ``astgen._gen_for``); ``range`` (the builtin name) names the ``std.range``
# struct and the loop ends when ``__next__`` raises ``std.StopIteration``.
# ---------------------------------------------------------------------------


@func()
def for_sum(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + i
    return total


@func()
def for_else(n: i32) -> i32:
    # the else clause runs when the sequence is exhausted
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + i
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total


@func()
def for_break(n: i32) -> i32:
    # ... and is skipped by a ``break``
    total: i32 = 0
    for i in range(n, 0, 1):
        if i == 3:
            break
        total = total + i
    else:
        total = total + 100
    return total


@func()
def for_continue(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        if i == 2:
            continue
        total = total + i
    return total


@func()
def for_step(n: i32) -> i32:
    # start and step are given explicitly
    total: i32 = 0
    for i in range(n, 1, 2):
        total = total + i
    return total


@func()
def for_defaults(n: i32) -> i32:
    # ``start`` and ``step`` are left out: the field defaults fill them
    total: i32 = 0
    for i in range(n):
        total = total + i
    return total


@func()
def for_nested(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        for j in range(i, 0, 1):
            total = total + j
    return total


@func()
def for_call(n: i32) -> i32:
    # a spy call in the body of a for loop
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + inc(i)
    return total


@func()
def for_leaks(n: i32) -> i32:
    # the loop variable is only visible inside the loop
    for i in range(n, 0, 1):
        pass
    return i  # pyright: ignore


@func()
def for_iter_is_a_copy(n: i32) -> i32:
    # ``__iter__`` returns the range by value (a copy), so writing through the
    # returned iterator does not touch the original range
    r = range(n, 0, 1)
    it = r.__iter__()
    it.start = 42  # pyright: ignore[reportAttributeAccessIssue]
    return r.start  # pyright: ignore[reportAttributeAccessIssue]


@struct()
class Addressable:
    x: i32

    def addr(self) -> Ptr[Addressable]:
        # ``self`` is bound directly to the incoming pointer, so ``ref(self)``
        # is a ``Ptr[Self]``
        return ref(self)


@func()
def write_through_self_ref(x: i32) -> i32:
    p = Addressable(x)
    q = p.addr()
    q[...].x = 99
    return p.x


class SpyWhileTest(TestCase):
    """``while`` loops: the condition is re-evaluated at the head of every
    iteration, ``break`` leaves the loop, ``continue`` starts the next
    iteration and the ``else`` clause runs only on a natural exit."""

    def test_counting_loop(self) -> None:
        self.assertEqual(sum_to(0), 0)
        self.assertEqual(sum_to(1), 0)
        self.assertEqual(sum_to(5), 10)
        self.assertEqual(count_up(4), 4)

    def test_call_inside_the_body(self) -> None:
        self.assertEqual(sum_calls(4), 10)
        self.assertEqual(sum_calls(0), 0)

    def test_break(self) -> None:
        self.assertEqual(find_divisor(15), 3)
        self.assertEqual(find_divisor(9), 3)
        # a prime has no divisor below it: the loop exits normally with i == n
        self.assertEqual(find_divisor(7), 7)
        self.assertEqual(skip_and_stop(10), 8)
        self.assertEqual(skip_and_stop(1), 1)

    def test_continue(self) -> None:
        self.assertEqual(sum_odd(5), 9)
        self.assertEqual(sum_odd(0), 0)

    def test_while_else(self) -> None:
        # the else clause runs on a natural exit ...
        self.assertEqual(while_else_natural(3), 103)
        # ... and is skipped by a ``break``
        self.assertEqual(while_else_break(5), 2)
        self.assertEqual(while_else_break(1), 101)
        # ... while a ``continue`` only delays it
        self.assertEqual(while_else_continue(4), 104)

    def test_nested_loops(self) -> None:
        self.assertEqual(nested_loop_sum(3, 4), 18)
        self.assertEqual(nested_loop_sum(0, 4), 0)

    def test_comptime_condition(self) -> None:
        self.assertEqual(comptime_false_loop(5), 1)

    def test_loop_whose_only_exit_is_return(self) -> None:
        self.assertEqual(loop_return(4), 4)
        self.assertEqual(loop_return(0), 0)

    def test_loop_in_a_void_function(self) -> None:
        self.assertIsNone(loop_void(3))

    def test_assignment_across_iterations(self) -> None:
        self.assertEqual(assign_outside_from_loop(5), 10)

    def test_break_and_continue_in_try(self) -> None:
        # a ``break``/``continue`` in a try body leaves the loop directly; the
        # clause is only reachable when the body raises first
        self.assertEqual(break_in_try(3), 3)
        self.assertEqual(continue_in_try(3), 52)

    def test_break_out_of_while_true(self) -> None:
        self.assertEqual(bounded_loop(3), 103)
        self.assertEqual(bounded_loop(0), 100)

    def test_loop_in_an_inlined_body(self) -> None:
        self.assertEqual(call_sum_inline(4), 6)


class SpyForTest(TestCase):
    """``for`` loops: the iterable is iterated with ``__iter__``/``__next__``
    (``range`` names ``std.range``), a ``break`` leaves the loop, a
    ``continue`` starts the next iteration and the ``else`` clause runs only
    when the sequence is exhausted."""

    def test_sum(self) -> None:
        self.assertEqual(for_sum(0), 0)
        self.assertEqual(for_sum(5), 10)

    def test_else_runs_when_exhausted(self) -> None:
        self.assertEqual(for_else(3), 3 + 100)

    def test_break_skips_the_else(self) -> None:
        self.assertEqual(for_break(6), 1 + 2)
        # a break never taken runs the whole loop and the else
        self.assertEqual(for_break(2), 1 + 100)

    def test_continue(self) -> None:
        self.assertEqual(for_continue(5), 0 + 1 + 3 + 4)

    def test_start_and_step(self) -> None:
        self.assertEqual(for_step(7), 1 + 3 + 5)

    def test_defaulted_start_and_step(self) -> None:
        # ``range(n)`` leaves ``start`` and ``step`` to their defaults
        self.assertEqual(for_defaults(4), 0 + 1 + 2 + 3)
        self.assertEqual(for_defaults(0), 0)

    def test_nested(self) -> None:
        # the inner loop runs ``range(i, 0, 1)`` = ``0 .. i - 1``
        self.assertEqual(for_nested(4), 0 + 0 + (0 + 1) + (0 + 1 + 2))

    def test_call_in_the_body(self) -> None:
        self.assertEqual(for_call(4), (0 + 1) + (1 + 1) + (2 + 1) + (3 + 1))

    def test_loop_variable_is_not_visible_after_the_loop(self) -> None:
        with self.assertRaises(CompileError):
            for_leaks(3)

    def test_iter_returns_a_copy(self) -> None:
        # ``__iter__`` returns the range by value, so the loop iterates a copy
        self.assertEqual(for_iter_is_a_copy(3), 0)


class SpyMethodSelfTest(TestCase):
    """A method's ``self`` is bound directly to the incoming pointer (its
    ``FunctionIR.arg_is_ref``), so reading ``self`` implicitly loads the
    receiver and ``ref(self)`` is a ``Ptr[Self]``."""

    def test_ref_of_self_is_a_pointer_to_the_receiver(self) -> None:
        self.assertEqual(write_through_self_ref(1), 99)


all_tests = [
    SpyWhileTest,
    SpyForTest,
    SpyMethodSelfTest,
]

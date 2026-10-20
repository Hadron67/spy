from unittest import TestCase

from ..compiler import (
    CompileError,
    Comptime,
    Ptr,
    defer,
    errdefer,
    func,
    i32,
    ref,
    unroll,
)
from .errors import ErrorA

# ---------------------------------------------------------------------------
# compile-time loops: a loop preceded by the statement ``unroll()``
# unrolls its body once per compile-time iteration (the condition is a
# compile-time value that the body advances); ``break`` leaves the whole
# unrolled sequence and ``continue`` jumps to the next unrolled body.
# ---------------------------------------------------------------------------


@func()
def inline_sum() -> i32:
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 4:
        total = total + i
        i = i + 1
    return total


@func()
def inline_false() -> i32:
    # a compile-time false condition unrolls no body and runs the else clause
    total: i32 = 0
    unroll()
    while False:
        total = total + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 2
    return total


@func()
def inline_break(n: i32) -> i32:
    # a runtime-conditional break leaves all the remaining unrolled bodies
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 10:
        i = i + 1
        if i == n:
            break
        total = total + i
    return total


@func()
def inline_continue(n: i32) -> i32:
    # a runtime-conditional continue jumps to the next unrolled body
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 6:
        i = i + 1
        if i == n:
            continue
        total = total + i
    return total


@func()
def inline_else() -> i32:
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 3:
        total = total + i
        i = i + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total + i


@func()
def inline_nested() -> i32:
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 3:
        j: Comptime = 0
        unroll()
        while j < 2:
            total = total + 1
            j = j + 1
        i = i + 1
    return total


@func()
def inline_continue_always() -> i32:
    # an unconditional continue: the body has no falling end, so the next body
    # is unrolled from the block the continue jumped to
    i: Comptime = 0
    unroll()
    while i < 4:
        i = i + 1
        continue
    return i


@func()
def runtime_loop_with_inline(n: i32) -> i32:
    # an inline loop inside a runtime loop
    k: i32 = 0
    total: i32 = 0
    while k < n:
        i: Comptime = 0
        unroll()
        while i < 3:
            total = total + 1
            i = i + 1
        k = k + 1
    return total


@func()
def inline_continue_or_return(n: i32) -> i32:
    # a ``continue`` whose sibling region of the enclosing if returns: the next
    # body is unrolled from the continue's jump even though nothing falls
    # through
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 5:
        i = i + 1
        if i == n:
            return total
        if i == 3:
            continue
        total = total + i
    return total + 100


@func()
def inline_bad_cond(n: i32) -> i32:
    # a runtime condition cannot be unrolled: the cap is what reports it
    unroll()
    while n > 0:
        n = n - 1
    return n


@func()
def inline_loop_misuse(n: i32) -> i32:
    # the marker must be followed by a loop
    unroll()
    return n


# ---------------------------------------------------------------------------
# compile-time ``for`` loops: the iterable (and so the loop variable) of a loop
# marked with ``unroll()`` is a compile-time value, so the desugared
# iterator loop unrolls at compile time - ``__next__`` raises ``StopIteration``
# while the interpreter runs, and the ``break`` its ``except`` clause runs ends
# the unrolled sequence (see ``astgen._gen_for``)
# ---------------------------------------------------------------------------


@func()
def unroll_for_sum() -> i32:
    # the literals have no runtime type of their own: the whole loop is
    # compile-time
    total: i32 = 0
    unroll()
    for i in range(4, 0, 1):
        total = total + i
    return total


@func()
def unroll_for_defaults() -> i32:
    # ``start``/``step`` are left out: the field defaults fill them
    total: i32 = 0
    unroll()
    for i in range(4):
        total = total + i
    return total


@func()
def unroll_for_else() -> i32:
    total: i32 = 0
    unroll()
    for i in range(3, 0, 1):
        total = total + i
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total


@func()
def unroll_for_break(n: i32) -> i32:
    # a runtime-conditional break leaves every remaining unrolled body (and the
    # else clause, like Python)
    total: i32 = 0
    unroll()
    for i in range(10, 0, 1):
        if i == n:
            break
        total = total + i
    else:
        total = total + 100
    return total


@func()
def unroll_for_continue() -> i32:
    total: i32 = 0
    unroll()
    for i in range(5, 0, 1):
        if i == 2:
            continue
        total = total + i
    return total


@func()
def unroll_for_nested() -> i32:
    total: i32 = 0
    unroll()
    for i in range(3, 0, 1):
        unroll()
        for j in range(2, 0, 1):
            total = total + i * 10 + j
    return total


@func()
def unroll_for_typed_var() -> i32:
    # a compile-time iterator *variable*, of an explicit element type with a
    # runtime representation (``range`` names ``std.range``, so it is
    # subscriptable in spy but not in the Python type system)
    r: Comptime[range[i32]] = range(3, 0, 1)  # pyright: ignore
    total: i32 = 0
    unroll()
    for i in r:
        total = total + i
    return total


@func()
def unroll_for_runtime_iter(n: i32) -> i32:
    # a runtime iterator cannot be unrolled: the unroll cap reports it
    total: i32 = 0
    unroll()
    for i in range(n, 0, 1):
        total = total + i
    return total


# ---------------------------------------------------------------------------
# deferred bodies inside compile-time loops: a ``with defer():`` in the
# body of an unrolled loop is registered in the region that wraps that body
# (the compile-time ``if`` a ``while`` lowers to, the ``try`` a ``for`` lowers
# to), so it runs at the end of every unrolled iteration - not once per loop.
# ---------------------------------------------------------------------------


@func()
def unroll_defer_body(p: Ptr[i32]) -> i32:
    # the defer runs once per unrolled iteration
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 3:
        with defer():
            p[...] = p[...] + 1
        total = total + i
        i = i + 1
    return total


@func()
def call_unroll_defer_body(x: i32) -> i32:
    v = x
    r: i32 = unroll_defer_body(ref(v))
    return v * 100 + r


@func()
def unroll_defer_for(p: Ptr[i32]) -> i32:
    # the desugared ``for`` wraps its body in a ``try``: the defer runs when
    # each iteration's body falls through to the ``StopIteration`` clause
    total: i32 = 0
    unroll()
    for i in range(3, 0, 1):
        with defer():
            p[...] = p[...] + 1
        total = total + i
    return total


@func()
def call_unroll_defer_for(x: i32) -> i32:
    v = x
    r: i32 = unroll_defer_for(ref(v))
    return v * 100 + r


@func()
def unroll_defer_break(p: Ptr[i32], n: i32) -> i32:
    # a runtime-conditional break runs the defer of the iteration it leaves
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 10:
        with defer():
            p[...] = p[...] + 1
        i = i + 1
        if i == n:
            break
        total = total + i
    return total


@func()
def call_unroll_defer_break(x: i32, n: i32) -> i32:
    v = x
    r: i32 = unroll_defer_break(ref(v), n)
    return v * 100 + r


@func()
def unroll_defer_continue(p: Ptr[i32], n: i32) -> i32:
    # a ``continue`` leaves the iteration through the same region end, so the
    # defer runs on it too
    i: Comptime = 0
    total: i32 = 0
    unroll()
    while i < 5:
        with defer():
            p[...] = p[...] + 1
        i = i + 1
        if i == n:
            continue
        total = total + i
    return total


@func()
def call_unroll_defer_continue(x: i32, n: i32) -> i32:
    v = x
    r: i32 = unroll_defer_continue(ref(v), n)
    return v * 100 + r


@func()
def unroll_defer_else(p: Ptr[i32]) -> i32:
    # a defer in the ``else`` clause runs once, when the loop ends normally
    i: Comptime = 0
    unroll()
    while i < 2:
        i = i + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        with defer():
            p[...] = p[...] + 1
    return i


@func()
def call_unroll_defer_else(x: i32) -> i32:
    v = x
    r: i32 = unroll_defer_else(ref(v))
    return v * 100 + r


@func()
def unroll_defer_order(p: Ptr[i32]) -> i32:
    # two defers of one iteration run in reverse declaration order
    i: Comptime = 0
    unroll()
    while i < 2:
        with defer():
            p[...] = p[...] * 10 + 1
        with defer():
            p[...] = p[...] * 10 + 2
        i = i + 1
    return i


@func()
def call_unroll_defer_order(x: i32) -> i32:
    v = x
    r: i32 = unroll_defer_order(ref(v))
    return v * 100 + r


@func(exceptions=(ErrorA,))
def unroll_errdefer(p: Ptr[i32], n: i32) -> i32:
    # an ``errdefer`` runs only on the raise, which ends the whole loop
    i: Comptime = 0
    unroll()
    while i < 5:
        with errdefer():
            p[...] = p[...] + 1
        if i == n:
            raise ErrorA(1)
        i = i + 1
    return i


@func()
def call_unroll_errdefer(x: i32, n: i32) -> i32:
    v = x
    r: i32 = 0
    try:
        r = unroll_errdefer(ref(v), n)
    except ErrorA:
        r = -1
    return v * 100 + r


class SpyInlineLoopTest(TestCase):
    """Compile-time loops: a loop preceded by ``unroll()`` unrolls its
    body once per compile-time iteration, ``break`` leaves the whole unrolled
    sequence and ``continue`` jumps to the next unrolled body."""

    def test_unrolled(self) -> None:
        # the body is emitted once per iteration, not looped over at runtime
        self.assertEqual(inline_sum(), 6)

    def test_comptime_false(self) -> None:
        # no body is unrolled; the else clause runs once
        self.assertEqual(inline_false(), 2)

    def test_break_skips_the_remaining_bodies(self) -> None:
        # i == n under a runtime condition: the break leaves every remaining
        # unrolled body, so only 1 + ... + (n - 1) is added
        self.assertEqual(inline_break(3), 3)
        # a break never taken unwinds the whole loop: 1 + ... + 10
        self.assertEqual(inline_break(0), 55)

    def test_continue_jumps_to_the_next_body(self) -> None:
        # 1 + ... + 6, minus the skipped n
        self.assertEqual(inline_continue(3), 18)
        self.assertEqual(inline_continue(0), 21)

    def test_while_else(self) -> None:
        self.assertEqual(inline_else(), 106)

    def test_nested_unrolled_loops(self) -> None:
        self.assertEqual(inline_nested(), 6)

    def test_unconditional_continue(self) -> None:
        # the unroll runs through the continue's jump, with no falling end
        self.assertEqual(inline_continue_always(), 4)

    def test_continue_with_a_return_sibling(self) -> None:
        # the continue's if has a returning sibling, so the whole iteration ends
        # without falling through: the unroll still runs from the continue
        self.assertEqual(inline_continue_or_return(2), 1)
        self.assertEqual(inline_continue_or_return(0), 112)

    def test_inline_loop_inside_a_runtime_loop(self) -> None:
        self.assertEqual(runtime_loop_with_inline(2), 6)
        self.assertEqual(runtime_loop_with_inline(0), 0)

    def test_runtime_condition_is_reported(self) -> None:
        # the condition is not a compile-time value, so the loop cannot be
        # unrolled: the unroll cap reports it instead of looping forever
        with self.assertRaises(CompileError):
            inline_bad_cond(3)

    def test_marker_outside_a_while_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            inline_loop_misuse(3)


class SpyUnrollForTest(TestCase):
    """Compile-time ``for`` loops: the iterable of a loop marked with
    ``unroll()`` is a compile-time value (the loop variable may have no
    runtime representation at all), so the desugared iterator loop unrolls at
    compile time; ``break`` leaves the whole unrolled sequence and ``continue``
    jumps to the next unrolled body."""

    def test_unrolled(self) -> None:
        self.assertEqual(unroll_for_sum(), 0 + 1 + 2 + 3)

    def test_defaulted_fields(self) -> None:
        self.assertEqual(unroll_for_defaults(), 0 + 1 + 2 + 3)

    def test_else(self) -> None:
        self.assertEqual(unroll_for_else(), 0 + 1 + 2 + 100)

    def test_break_skips_the_remaining_bodies_and_the_else(self) -> None:
        self.assertEqual(unroll_for_break(2), 0 + 1)
        # a break never taken runs the whole loop and the else
        self.assertEqual(unroll_for_break(100), 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 100)

    def test_continue_jumps_to_the_next_body(self) -> None:
        self.assertEqual(unroll_for_continue(), 0 + 1 + 3 + 4)

    def test_nested_unrolled_loops(self) -> None:
        self.assertEqual(unroll_for_nested(), 1 + 21 + 41)

    def test_a_compile_time_iterator_variable(self) -> None:
        self.assertEqual(unroll_for_typed_var(), 0 + 1 + 2)

    def test_a_runtime_iterator_is_reported(self) -> None:
        # the loop only ends when a ``StopIteration`` reaches the except clause
        # at compile time, which a runtime iterator never does: the unroll cap
        # reports it instead of unrolling forever
        with self.assertRaises(CompileError):
            unroll_for_runtime_iter(3)


class SpyUnrollDeferTest(TestCase):
    """Deferred bodies in compile-time loops: a defer in the body of an
    unrolled loop is registered in the region wrapping that body (the
    compile-time ``if`` of a ``while``, the ``try`` of a ``for``), so it runs
    at the end of every unrolled iteration - not once for the whole loop."""

    def test_a_defer_runs_every_iteration(self) -> None:
        # three iterations, so the counter is bumped three times
        self.assertEqual(call_unroll_defer_body(0), 3 * 100 + 3)

    def test_a_defer_in_a_for_body_runs_every_iteration(self) -> None:
        self.assertEqual(call_unroll_defer_for(0), 3 * 100 + 3)

    def test_a_break_runs_the_defer_of_its_iteration(self) -> None:
        # the break happens on the third iteration (i == 3), after its defer
        self.assertEqual(call_unroll_defer_break(0, 3), 3 * 100 + 3)
        # a break never taken runs all ten iterations
        self.assertEqual(call_unroll_defer_break(0, 0), 10 * 100 + 55)

    def test_a_continue_runs_the_defer_of_its_iteration(self) -> None:
        # the defer runs on every iteration, the continue just skips the total
        self.assertEqual(call_unroll_defer_continue(0, 3), 5 * 100 + 12)

    def test_a_defer_in_the_else_clause_runs_once(self) -> None:
        self.assertEqual(call_unroll_defer_else(0), 1 * 100 + 2)

    def test_iteration_defers_run_in_reverse_declaration_order(self) -> None:
        self.assertEqual(call_unroll_defer_order(0), 2121 * 100 + 2)

    def test_an_errdefer_runs_only_on_the_raise(self) -> None:
        # the raise ends the whole loop, so only the raise iteration's errdefer
        # runs; the exception is caught by the caller
        self.assertEqual(call_unroll_errdefer(0, 2), 1 * 100 - 1)
        # no raise: the errdefer never runs
        self.assertEqual(call_unroll_errdefer(0, 10), 0 * 100 + 5)


all_tests = [
    SpyInlineLoopTest,
    SpyUnrollForTest,
    SpyUnrollDeferTest,
]

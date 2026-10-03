from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    syntax,
)
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
)

# ---------------------------------------------------------------------------
# compile-time loops: a loop preceded by the statement ``syntax.unroll()``
# unrolls its body once per compile-time iteration (the condition is a
# compile-time value that the body advances); ``break`` leaves the whole
# unrolled sequence and ``continue`` jumps to the next unrolled body.
# ---------------------------------------------------------------------------


@func()
def inline_sum() -> i32:
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 4:
        total = total + i
        i = i + 1
    return total


@func()
def inline_false() -> i32:
    # a compile-time false condition unrolls no body and runs the else clause
    total: i32 = 0
    syntax.unroll()
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
    syntax.unroll()
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
    syntax.unroll()
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
    syntax.unroll()
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
    syntax.unroll()
    while i < 3:
        j: Comptime = 0
        syntax.unroll()
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
    syntax.unroll()
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
        syntax.unroll()
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
    syntax.unroll()
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
    syntax.unroll()
    while n > 0:
        n = n - 1
    return n


@func()
def inline_loop_misuse(n: i32) -> i32:
    # the marker must be followed by a loop
    syntax.unroll()
    return n


# ---------------------------------------------------------------------------
# compile-time ``for`` loops: the iterable (and so the loop variable) of a loop
# marked with ``syntax.unroll()`` is a compile-time value, so the desugared
# iterator loop unrolls at compile time - ``__next__`` raises ``StopIteration``
# while the interpreter runs, and the ``break`` its ``except`` clause runs ends
# the unrolled sequence (see ``astgen._gen_for``)
# ---------------------------------------------------------------------------


@func()
def unroll_for_sum() -> i32:
    # the literals have no runtime type of their own: the whole loop is
    # compile-time
    total: i32 = 0
    syntax.unroll()
    for i in range(4, 0, 1):
        total = total + i
    return total


@func()
def unroll_for_defaults() -> i32:
    # ``start``/``step`` are left out: the field defaults fill them
    total: i32 = 0
    syntax.unroll()
    for i in range(4):
        total = total + i
    return total


@func()
def unroll_for_else() -> i32:
    total: i32 = 0
    syntax.unroll()
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
    syntax.unroll()
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
    syntax.unroll()
    for i in range(5, 0, 1):
        if i == 2:
            continue
        total = total + i
    return total


@func()
def unroll_for_nested() -> i32:
    total: i32 = 0
    syntax.unroll()
    for i in range(3, 0, 1):
        syntax.unroll()
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
    syntax.unroll()
    for i in r:
        total = total + i
    return total


@func()
def unroll_for_runtime_iter(n: i32) -> i32:
    # a runtime iterator cannot be unrolled: the unroll cap reports it
    total: i32 = 0
    syntax.unroll()
    for i in range(n, 0, 1):
        total = total + i
    return total


class SpyInlineLoopTest(TestCase):
    """Compile-time loops: a loop preceded by ``syntax.unroll()`` unrolls its
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
    ``syntax.unroll()`` is a compile-time value (the loop variable may have no
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


all_tests = [
    SpyInlineLoopTest,
    SpyUnrollForTest,
]

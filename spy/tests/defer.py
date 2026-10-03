from typing import Never
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func
from ..compiler.syntax import (
    Ptr,
    defer,
    errdefer,
    okdefer,
    ref,
)
from .errors import ErrorA

# ---------------------------------------------------------------------------
# deferred bodies (``with syntax.defer():``)
#
# A deferred body runs when the region it is declared in is left: ``defer`` on
# any exit, ``okdefer`` on a normal one and ``errdefer`` on an error one, in
# reverse declaration order.  The test functions observe the effect through a
# pointer to a local, whose address a wrapper passes with ``ref`` (a spy
# function never takes a ``Ptr`` from Python directly).
# ---------------------------------------------------------------------------


@func()
def defer_on_return(p: Ptr[i32]) -> i32:
    with defer():
        p[...] = p[...] + 1
    return 5


@func()
def call_defer_on_return(x: i32) -> i32:
    v = x
    r: i32 = defer_on_return(ref(v))
    return v * 100 + r


@func()
def defer_order(p: Ptr[i32]) -> i32:
    # the bodies run in reverse declaration order
    with defer():
        p[...] = p[...] * 10 + 1
    with defer():
        p[...] = p[...] * 10 + 2
    return 0


@func()
def call_defer_order(x: i32) -> i32:
    v = x
    r: i32 = defer_order(ref(v))
    return v * 100 + r


@func()
def defer_fallthrough(p: Ptr[i32]) -> None:
    with defer():
        p[...] = p[...] + 1


@func()
def call_defer_fallthrough(x: i32) -> i32:
    v = x
    defer_fallthrough(ref(v))
    return v


@func()
def defer_in_branch(p: Ptr[i32], c: i32) -> i32:
    # a ``return`` inside a branch runs the defer too
    with defer():
        p[...] = p[...] + 1
    if c > 0:
        return 1
    return 2


@func()
def call_defer_in_branch(x: i32, c: i32) -> i32:
    v = x
    r: i32 = defer_in_branch(ref(v), c)
    return v * 100 + r


@func()
def defer_on_break(p: Ptr[i32], n: i32) -> i32:
    i: i32 = 0
    while i < n:
        with defer():
            p[...] = p[...] + 1
        if i == 1:
            break
        i = i + 1
    return i


@func()
def call_defer_on_break(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_on_break(ref(v), n)
    return v * 100 + r


@func()
def defer_on_continue(p: Ptr[i32], n: i32) -> i32:
    i: i32 = 0
    while i < n:
        with defer():
            p[...] = p[...] + 1
        i = i + 1
        continue
    return i


@func()
def call_defer_on_continue(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_on_continue(ref(v), n)
    return v * 100 + r


@func()
def defer_ok_only(p: Ptr[i32]) -> i32:
    with okdefer():
        p[...] = p[...] + 1
    return 1


@func()
def call_defer_ok_only(x: i32) -> i32:
    v = x
    r: i32 = defer_ok_only(ref(v))
    return v * 100 + r


@func(exceptions=(ErrorA,))
def defer_err(p: Ptr[i32], n: i32) -> i32:
    # ``okdefer`` runs on the normal return, ``errdefer`` on the raise
    with okdefer():
        p[...] = p[...] + 1
    with errdefer():
        p[...] = p[...] + 100
    if n < 0:
        raise ErrorA(7)
    return n + 1


@func()
def call_defer_err(x: i32, n: i32) -> i32:
    v = x
    r: i32 = 0
    try:
        r = defer_err(ref(v), n)
    except ErrorA:
        r = -1
    return v * 100 + r


@func()
def defer_nested(p: Ptr[i32]) -> i32:
    # a defer body may declare a defer of its own: it runs when the body ends
    with defer():
        p[...] = p[...] * 10 + 1
        with defer():
            p[...] = p[...] * 10 + 2
    return 0


@func()
def call_defer_nested(x: i32) -> i32:
    v = x
    r: i32 = defer_nested(ref(v))
    return v * 100 + r


def inline_defer(p: Ptr[i32]) -> i32:
    # a plain (unregistered) function inlined at its call site: its defer runs
    # when the inlined body is left
    with defer():
        p[...] = p[...] + 1
    return 3


@func()
def call_inline_defer(x: i32) -> i32:
    v = x
    r: i32 = inline_defer(ref(v))
    return v * 100 + r


@func()
def defer_in_and_else(p: Ptr[i32], a: i32, b: i32) -> i32:
    # the ``and`` chain's condition opens a block; when the then body falls off
    # its end, the implicit ``break_if`` out of both blocks runs the defer
    r: i32 = 0
    if a > 0 and b > 0:
        with defer():
            p[...] = p[...] + 1
        r = 1
    else:
        r = 2
    return r


@func()
def call_defer_in_and_else(x: i32, a: i32, b: i32) -> i32:
    v = x
    r: i32 = defer_in_and_else(ref(v), a, b)
    return v * 100 + r


@func()
def defer_err_in_try_body(p: Ptr[i32], n: i32) -> i32:
    # the raise leaves the try body, so a defer declared in it runs
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise ErrorA(7)
        return 1
    except ErrorA:
        return 2


@func()
def call_defer_err_in_try_body(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_err_in_try_body(ref(v), n)
    return v * 100 + r


@func()
def defer_shared_raise(p: Ptr[i32], n: i32) -> i32:
    # two ``raise``s leave the same ``try`` body and reach the same typed clause:
    # they share one copy of the defer, and the payload phi of the clause merges
    # the two raises at the chain entry (each path still catches its own payload)
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise ErrorA(1)
        raise ErrorA(2)
    except ErrorA as e:
        return e.code


@func()
def call_defer_shared_raise(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_shared_raise(ref(v), n)
    return v * 100 + r


@func(exceptions=ErrorA)
def raise_a_never(n: i32) -> Never:
    # a call that always raises: it has no normal continuation
    raise ErrorA(n)


@func()
def defer_shared_call(p: Ptr[i32], n: i32) -> i32:
    # the same through the error dispatch of two calls: both share the chain and
    # deliver their payload to the clause's merged phi
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise_a_never(1)
        raise_a_never(2)
    except ErrorA as e:
        return e.code


@func()
def call_defer_shared_call(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_shared_call(ref(v), n)
    return v * 100 + r


@func()
def defer_err_caught_outside(p: Ptr[i32], n: i32) -> i32:
    # the raise is caught inside the same function, so the function body is not
    # left and its errdefer does not run
    with errdefer():
        p[...] = p[...] + 100
    try:
        if n < 0:
            raise ErrorA(7)
        return 1
    except ErrorA:
        return 2


@func()
def call_defer_err_caught_outside(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_err_caught_outside(ref(v), n)
    return v * 100 + r


@func()
def defer_in_for(p: Ptr[i32], n: i32) -> i32:
    # the loop body sits inside the ``for`` desugaring's ``try``; its falling
    # end runs the defer each iteration
    total: i32 = 0
    for i in range(n):
        with defer():
            p[...] = p[...] + 1
        total = total + i
    return total


@func()
def call_defer_in_for(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_in_for(ref(v), n)
    return v * 100 + r


@func()
def defer_with_if(p: Ptr[i32], c: i32) -> i32:
    # the body of a defer may branch: its copy carries the branch along
    with defer():
        if c > 0:
            p[...] = p[...] + 1
        else:
            p[...] = p[...] + 2
    return 0


@func()
def call_defer_with_if(x: i32, c: i32) -> i32:
    v = x
    r: i32 = defer_with_if(ref(v), c)
    return v * 100 + r


@func()
def defer_comptime_if(p: Ptr[i32]) -> i32:
    # the chosen branch of a compile-time ``if`` fell through: its defer runs
    if spy_typeof(p) == Ptr[i32]:
        with defer():
            p[...] = p[...] + 1
    return 0


@func()
def call_defer_comptime_if(x: i32) -> i32:
    v = x
    r: i32 = defer_comptime_if(ref(v))
    return v * 100 + r


@func()
def defer_bad_return() -> i32:
    with defer():
        return 1
    return 0


@func()
def defer_bad_break_outer() -> i32:
    i: i32 = 0
    while i < 3:
        with defer():
            break
        i = i + 1
    return i


class SpyDeferTest(TestCase):
    def test_a_defer_runs_on_a_return(self) -> None:
        self.assertEqual(call_defer_on_return(0), 105)

    def test_bodies_run_in_reverse_order(self) -> None:
        self.assertEqual(call_defer_order(0), 2100)

    def test_a_defer_runs_when_the_body_falls_through(self) -> None:
        self.assertEqual(call_defer_fallthrough(0), 1)

    def test_a_return_in_a_branch_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_in_branch(0, 1), 101)
        self.assertEqual(call_defer_in_branch(0, 0), 102)

    def test_a_break_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_on_break(0, 0), 0)
        self.assertEqual(call_defer_on_break(0, 1), 101)
        self.assertEqual(call_defer_on_break(0, 2), 201)

    def test_a_continue_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_on_continue(0, 0), 0)
        self.assertEqual(call_defer_on_continue(0, 3), 303)

    def test_okdefer_runs_on_a_normal_exit(self) -> None:
        self.assertEqual(call_defer_ok_only(0), 101)

    def test_errdefer_runs_only_on_an_error(self) -> None:
        self.assertEqual(call_defer_err(0, 5), 106)
        self.assertEqual(call_defer_err(0, -1), 9999)

    def test_a_defer_inside_a_defer_body(self) -> None:
        self.assertEqual(call_defer_nested(0), 1200)

    def test_an_inlined_body_runs_its_defer(self) -> None:
        self.assertEqual(call_inline_defer(0), 103)

    def test_a_defer_in_an_and_chain_body(self) -> None:
        self.assertEqual(call_defer_in_and_else(0, 1, 1), 101)
        self.assertEqual(call_defer_in_and_else(0, 1, 0), 2)
        self.assertEqual(call_defer_in_and_else(0, 0, 1), 2)

    def test_a_defer_in_a_try_body_runs_on_a_caught_raise(self) -> None:
        self.assertEqual(call_defer_err_in_try_body(0, 5), 1)
        self.assertEqual(call_defer_err_in_try_body(0, -1), 10002)

    def test_two_raises_share_a_defer_and_keep_their_payloads(self) -> None:
        self.assertEqual(call_defer_shared_raise(0, -1), 10001)
        self.assertEqual(call_defer_shared_raise(0, 5), 10002)

    def test_two_calls_share_a_defer_and_keep_their_payloads(self) -> None:
        self.assertEqual(call_defer_shared_call(0, -1), 10001)
        self.assertEqual(call_defer_shared_call(0, 5), 10002)

    def test_an_errdefer_is_not_run_when_the_error_is_caught_inside(self) -> None:
        self.assertEqual(call_defer_err_caught_outside(0, 5), 1)
        self.assertEqual(call_defer_err_caught_outside(0, -1), 2)

    def test_a_defer_inside_a_for_body(self) -> None:
        self.assertEqual(call_defer_in_for(0, 0), 0)
        self.assertEqual(call_defer_in_for(0, 3), 303)

    def test_a_defer_body_with_a_branch(self) -> None:
        self.assertEqual(call_defer_with_if(0, 1), 100)
        self.assertEqual(call_defer_with_if(0, 0), 200)

    def test_a_defer_in_a_compile_time_branch(self) -> None:
        self.assertEqual(call_defer_comptime_if(0), 100)

    def test_a_return_inside_a_defer_body_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            defer_bad_return()

    def test_a_break_out_of_a_defer_body_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            defer_bad_break_outer()


all_tests = [
    SpyDeferTest,
]

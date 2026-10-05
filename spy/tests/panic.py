from unittest import TestCase

from ..compiler import i32
from ..compiler.dsl import func
from ..compiler.syntax import (
    OK,
    UNWIND,
    Ptr,
    closure,
    defer,
    errdefer,
    okdefer,
    ref,
)
from ..std.core import (
    UnwindException,
    catch_unwind,
    deinit_panic_data,
    panic,
)
from .errors import ErrorA

# ---------------------------------------------------------------------------
# panic / catch_unwind
#
# ``std.core.panic(data)`` throws a C++ exception carrying ``data``; the
# ``std.core.catch_unwind(fn)`` builtin calls the (non-inline) closure ``fn`` and
# catches a panic it raises, delivering it as an ``UnwindException`` through the
# ordinary spy error path.  A deferred body whose flags hold ``UNWIND`` runs as
# the panic unwinds through its region.
# ---------------------------------------------------------------------------


@func()
def panic_i32(n: i32) -> i32:
    # never returns: the panic carries ``n``
    panic(n)
    return 0


@func()
def catch_panic(n: i32) -> i32:
    @closure(inline=False)
    def body() -> i32:
        return panic_i32(n)

    try:
        return catch_unwind(body)
    except UnwindException as e:
        with defer(): deinit_panic_data(e.data)
        if isinstance(v := e.data, i32):
            return 1000 + v
        return -1


@func()
def catch_no_panic(n: i32) -> i32:
    @closure(inline=False)
    def body() -> i32:
        return n + 1

    try:
        return catch_unwind(body)
    except UnwindException:
        return -1


@func()
def panic_with_defer(p: Ptr[i32], n: i32) -> None:
    # a plain ``defer`` (ALL) runs on the panic too
    with defer():
        p[...] = p[...] + 1
    panic(n)


@func()
def call_panic_with_defer(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_with_defer(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def panic_with_okdefer(p: Ptr[i32], n: i32) -> None:
    # ``okdefer`` (OK) does not run on a panic
    with okdefer():
        p[...] = p[...] + 1
    panic(n)


@func()
def call_panic_with_okdefer(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_with_okdefer(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def panic_with_errdefer(p: Ptr[i32], n: i32) -> None:
    # ``errdefer`` (ERR = RAISE|UNWIND) runs on a panic
    with errdefer():
        p[...] = p[...] + 1
    panic(n)


@func()
def call_panic_with_errdefer(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_with_errdefer(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def call_defer_flags(p: Ptr[i32], n: i32) -> i32:
    # ``defer(syntax.OK)`` behaves like ``okdefer``: no run on a panic
    with defer(OK):
        p[...] = p[...] + 1
    panic(n)


@func()
def call_defer_flags_body(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        call_defer_flags(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def panic_nested(p: Ptr[i32], n: i32) -> None:
    # a defer declared in an inner region runs before the outer one
    with defer():
        p[...] = p[...] * 10 + 1
        with defer():
            p[...] = p[...] * 10 + 2
    panic(n)


@func()
def call_panic_nested(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_nested(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def catch_nested() -> i32:
    # the inner closure's panic is caught by the inner catch_unwind, so the
    # outer one never sees a panic (the closures capture nothing runtime)
    @closure(inline=False)
    def inner() -> i32:
        panic(7)

    @closure(inline=False)
    def outer() -> i32:
        try:
            return catch_unwind(inner)
        except UnwindException as e:
            if isinstance(v := e.data, i32):
                return v + 1
            return -1

    try:
        return catch_unwind(outer)
    except UnwindException:
        return -2


@func(exceptions=ErrorA)
def raise_error(n: i32) -> i32:
    raise ErrorA(n)


@func()
def catch_spy_error(n: i32) -> i32:
    # the closure raises a spy exception, not a panic: it is delivered as such
    @closure(inline=False, exceptions=ErrorA)
    def body() -> i32:
        return raise_error(n)

    try:
        return catch_unwind(body)
    except UnwindException:
        return -1
    except ErrorA as e:
        return 2000 + e.code


@func()
def panic_inner(p: Ptr[i32], n: i32) -> None:
    with defer():
        p[...] = p[...] * 10 + 2
    panic(n)


@func()
def panic_outer(p: Ptr[i32], n: i32) -> None:
    # its defer runs after the inner frame's, as the panic unwinds outwards
    with defer():
        p[...] = p[...] * 10 + 1
    panic_inner(p, n)


@func()
def call_panic_frames(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_outer(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def panic_unwind_only(p: Ptr[i32], n: i32) -> None:
    # ``defer(UNWIND)`` runs on a panic (but not on a normal exit)
    with defer(UNWIND):
        p[...] = p[...] + 1
    panic(n)


@func()
def call_panic_unwind_only(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        panic_unwind_only(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


@func()
def defer_combined(p: Ptr[i32], n: i32) -> i32:
    # ``defer(UNWIND | OK)`` runs both on a normal exit and on a panic
    with defer(UNWIND | OK):
        p[...] = p[...] + 1
    if n < 0:
        panic(n)
    return n


@func()
def call_defer_combined_normal(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_combined(ref(v), n)
    return v * 100 + r


@func()
def call_defer_combined_panic(x: i32, n: i32) -> i32:
    v = x

    @closure(inline=False)
    def body() -> None:
        defer_combined(ref(v), n)

    try:
        catch_unwind(body)
    except UnwindException:
        return v * 100 + 1
    return v * 100


class SpyPanicTest(TestCase):
    def test_a_panic_is_caught_and_its_data_read_back(self) -> None:
        self.assertEqual(catch_panic(7), 1007)

    def test_a_call_that_does_not_panic_returns_its_value(self) -> None:
        self.assertEqual(catch_no_panic(5), 6)

    def test_a_defer_runs_on_a_panic(self) -> None:
        self.assertEqual(call_panic_with_defer(0, 3), 101)

    def test_an_okdefer_does_not_run_on_a_panic(self) -> None:
        self.assertEqual(call_panic_with_okdefer(0, 3), 1)

    def test_an_errdefer_runs_on_a_panic(self) -> None:
        self.assertEqual(call_panic_with_errdefer(0, 3), 101)

    def test_defer_flags_of_ok_do_not_run_on_a_panic(self) -> None:
        self.assertEqual(call_defer_flags_body(0, 3), 1)

    def test_nested_defers_run_in_order_on_a_panic(self) -> None:
        # the outer defer runs on the panic; its own (inner) defer runs when the
        # outer body completes: 0 -> 1 -> 12
        self.assertEqual(call_panic_nested(0, 3), 1201)

    def test_a_nested_catch_unwind_catches_the_panic(self) -> None:
        self.assertEqual(catch_nested(), 8)

    def test_a_spy_error_of_the_closure_is_not_turned_into_an_unwind(self) -> None:
        self.assertEqual(catch_spy_error(3), 2003)

    def test_defers_of_every_frame_run_as_the_panic_unwinds(self) -> None:
        # the inner frame's defer runs first (0 -> 2), then the outer's (-> 21)
        self.assertEqual(call_panic_frames(0, 3), 2101)

    def test_defer_flags_of_unwind_run_on_a_panic(self) -> None:
        self.assertEqual(call_panic_unwind_only(0, 3), 101)

    def test_a_combined_flags_expression_runs_on_both_exits(self) -> None:
        # ``defer(UNWIND | OK)``: the defer runs on the normal return and on the
        # panic (0 -> 1 in both cases)
        self.assertEqual(call_defer_combined_normal(0, 5), 105)
        self.assertEqual(call_defer_combined_panic(0, -1), 101)


all_tests = [
    SpyPanicTest,
]

from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    syntax,
    u64,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Comptime,
    Ptr,
    ref,
)
from ..std.core import compile_error
from .basics import inc
from .errors import ErrorA

# ---------------------------------------------------------------------------
# ``syntax.comptime()``: the statement marker that declares the variable of the
# declaration immediately after it as a compile-time one, the way
# ``syntax.unroll()`` marks the loop that follows it - ``syntax.comptime()``
# followed by ``a: T = e`` is ``a: Comptime[T] = e``, and followed by ``a = e``
# (no annotation) it is ``a: Comptime = e`` (see ``astgen._gen_stmt``)
# ---------------------------------------------------------------------------


@struct()
class MarkerState:
    on: spy_bool
    n: i32


@func()
def comptime_marker_typed() -> i32:
    # the marker plus a typed declaration is ``i: Comptime[i32] = 0``: the loop
    # condition is a compile-time value, so the loop unrolls
    syntax.comptime()
    i: i32 = 0
    total: i32 = 0
    syntax.unroll()
    while i < 4:
        total = total + i
        i = i + 1
    return total


@func()
def comptime_marker_untyped() -> i32:
    # no annotation: the value determines the type (a bare ``Comptime``), and an
    # untyped literal has no runtime representation at all
    syntax.comptime()
    n = 4
    total: i32 = 0
    syntax.unroll()
    while n > 0:
        total = total + n
        n = n - 1
    return total


@func()
def comptime_marker_holds_a_type(x: i32) -> spy_bool:
    # a compile-time variable is what holds a type (a plain local cannot)
    syntax.comptime()
    t = spy_typeof(x)
    return t == i32


@func()
def comptime_marker_struct(x: i32) -> i32:
    # a compile-time struct declared with the marker: its field conditions a
    # compile-time loop
    syntax.comptime()
    s: MarkerState = MarkerState(True, 0)
    total: i32 = 0
    syntax.unroll()
    while s.on:
        s.on = False
        total = total + x
    return total


@func()
def comptime_marker_holds_a_runtime_value(x: i32) -> i32:
    # a compile-time variable of a runtime type holds the runtime value
    syntax.comptime()
    v: i32 = x
    return v + 1


@func()
def comptime_marker_without_a_value() -> i32:
    # the marker also marks a declaration that writes no value
    syntax.comptime()
    n: i32
    n = 5
    return n + 1


@func()
def comptime_marker_destructuring() -> i32:
    # the marker marks every name a destructuring declaration declares
    syntax.comptime()
    a, b = 1, 2
    total: i32 = 0
    syntax.unroll()
    while a > 0:
        total = total + a * 10 + b
        a = a - 1
    return total


@func()
def comptime_marker_in_a_branch(c: spy_bool) -> i32:
    # the marker marks a declaration of the block it sits in
    total: i32 = 0
    if c:
        syntax.comptime()
        k = 3
        total = total + k
    return total


@func()
def comptime_declared_after_runtime_control_flow(n: i32) -> i32:
    # a compile-time local written after runtime control flow: its slot is
    # declared where it is written, so the ``Alloca`` and the value's store stay
    # in one runtime block (see pending-problems.md #5)
    x: i32 = 0
    if n > 0:
        x = n
    c: Comptime = 7
    return c + x


@func()
def comptime_marker_before_a_non_declaration() -> i32:
    syntax.comptime()
    return 1


@func()
def comptime_marker_at_the_end():
    syntax.comptime()


@func()
def comptime_marker_redundant() -> i32:
    # the annotation already declares the variable compile-time
    syntax.comptime()
    a: Comptime = 1
    return a


@func()
def comptime_marker_on_an_assignment() -> i32:
    # the marker marks a *declaration*: this statement assigns to a variable
    # that is bound already
    a: i32 = 1
    syntax.comptime()
    a = 2
    return a


@func()
def comptime_marker_as_a_value() -> i32:
    return syntax.comptime()  # pyright: ignore[reportReturnType]


@func()
def comptime_marker_before_a_loop() -> i32:
    # the marker must come immediately before the declaration it marks
    syntax.comptime()
    syntax.unroll()
    while False:
        pass
    return 1


@func()
def tuple_declare(x: i32) -> i32:
    a, b = x, x + 1
    return a * 10 + b


@func()
def tuple_swap(a: i32, b: i32) -> i32:
    # known design flaw: a destructuring is written into its targets in order,
    # so ``a, b = b, a`` reads ``a`` after it has been overwritten (see
    # ``_gen_assign``): both variables end up holding ``b``
    a, b = b, a
    return a * 10 + b


@func()
def tuple_nested(x: i32) -> i32:
    a, (b, c) = x, (x + 1, x + 2)
    return a * 100 + b * 10 + c


@func()
def tuple_of_calls(x: i32) -> i32:
    # the right-hand side of a destructuring is generated straight into the
    # targets: each call writes its result into the target it is bound to,
    # with no intermediate tuple value (see ``_gen_assign``)
    a, b = inc(x), inc(x + 1)
    return a * 10 + b


@func()
def bad_comptime_tuple_branch(cond: spy_bool) -> i32:
    # one branch of the if-expression is a tuple and the other a single value:
    # the slot is stored with incompatible types (see ``init_tuple``)
    t: Comptime = (1, 2) if cond else inc(3)
    p, q = t
    return p * 10 + q


@struct()
class Slice[T]:
    ptr: T
    len: u64


class SpyComptimeMarkerTest(TestCase):
    """``syntax.comptime()``: the statement marker that declares the variable
    of the declaration right after it as a compile-time one, the way
    ``syntax.unroll()`` marks the loop that follows it.  It stands for the
    ``Comptime`` annotation: ``syntax.comptime()`` + ``a: T = e`` is
    ``a: Comptime[T] = e``, and with no annotation ``a: Comptime = e``."""

    def test_a_typed_declaration(self) -> None:
        # ``syntax.comptime()`` + ``i: i32 = 0`` is ``i: Comptime[i32] = 0``
        self.assertEqual(comptime_marker_typed(), 0 + 1 + 2 + 3)

    def test_an_unannotated_declaration(self) -> None:
        # ... and with no annotation it is ``n: Comptime = 4``
        self.assertEqual(comptime_marker_untyped(), 4 + 3 + 2 + 1)

    def test_it_holds_a_type(self) -> None:
        self.assertTrue(comptime_marker_holds_a_type(1))

    def test_a_compile_time_struct(self) -> None:
        self.assertEqual(comptime_marker_struct(5), 5)

    def test_it_may_hold_a_runtime_value(self) -> None:
        self.assertEqual(comptime_marker_holds_a_runtime_value(5), 6)

    def test_a_declaration_without_a_value(self) -> None:
        self.assertEqual(comptime_marker_without_a_value(), 6)

    def test_a_destructuring_declaration(self) -> None:
        self.assertEqual(comptime_marker_destructuring(), 1 * 10 + 2)

    def test_a_declaration_in_a_branch(self) -> None:
        self.assertEqual(comptime_marker_in_a_branch(True), 3)
        self.assertEqual(comptime_marker_in_a_branch(False), 0)

    def test_a_declaration_after_runtime_control_flow(self) -> None:
        # the declaration sits after a runtime ``if``: its slot is declared right
        # there, not at the block's head
        self.assertEqual(comptime_declared_after_runtime_control_flow(0), 7)
        self.assertEqual(comptime_declared_after_runtime_control_flow(4), 11)

    def test_it_must_be_followed_by_a_declaration(self) -> None:
        with self.assertRaises(CompileError):
            comptime_marker_before_a_non_declaration()
        with self.assertRaises(CompileError):
            comptime_marker_at_the_end()

    def test_a_redundant_annotation_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_redundant()
        self.assertIn('already declared compile-time', str(ctx.exception))

    def test_it_marks_a_declaration_not_an_assignment(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_on_an_assignment()
        self.assertIn('declares a fresh variable', str(ctx.exception))

    def test_it_is_not_a_value(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_as_a_value()
        self.assertIn('statement marker', str(ctx.exception))

    def test_it_must_come_immediately_before_the_declaration(self) -> None:
        with self.assertRaises(CompileError):
            comptime_marker_before_a_loop()


class SpyTupleTest(TestCase):
    """Compile-time tuples: destructuring assignment unpacks a tuple of
    values into a tuple of target addresses (which may nest)."""

    def test_declaring_destructuring(self) -> None:
        self.assertEqual(tuple_declare(4), 45)

    def test_swap_destructuring_is_a_known_flaw(self) -> None:
        # the right-hand side is written into its targets in order, so the
        # second element reads its source after the first has overwritten it:
        # ``a, b = b, a`` gives ``b, b`` (a known design flaw)
        self.assertEqual(tuple_swap(1, 2), 22)

    def test_nested_destructuring(self) -> None:
        self.assertEqual(tuple_nested(4), 456)

    def test_destructuring_from_calls(self) -> None:
        self.assertEqual(tuple_of_calls(3), 45)

    def test_tuple_into_a_comptime_variable_conflict(self) -> None:
        # a slot holding a tuple cannot also be stored a single value, and the
        # conflict is reported when the slot is committed
        with self.assertRaises(CompileError):
            bad_comptime_tuple_branch(True)


# ---------------------------------------------------------------------------
# ``syntax.typeof``: the *type probe*.  Its argument is only *typed* - its
# instructions run into a detached block that is never lowered, which compiles
# whatever spy functions the expression names - so ``typeof(f(x))`` yields the
# type of the call without emitting (or ever running) it.
# ---------------------------------------------------------------------------


@func()
def probe_inner(x: i32) -> i32:
    return x * 2


@func()
def probe_outer(x: i32) -> i32:
    return x + 1


@func()
def typeof_a_nested_call(x: i32) -> i32:
    # ``typeof`` accepts any expression: both calls are typed (and compiled)
    # while nothing is emitted
    t: Comptime = spy_typeof(probe_outer(probe_inner(x)))
    if t == i32:
        return 1
    return 0


@func()
def probe_bump(p: Ptr[i32]) -> i32:
    p[...] = p[...] + 1
    return p[...]


@func()
def typeof_has_no_runtime_effect(p: Ptr[i32]) -> i32:
    # the argument is only typed: ``probe_bump`` is compiled but never emitted,
    # so the pointee keeps its value
    t: Comptime = spy_typeof(probe_bump(p))
    if t == i32:
        return p[...]
    return -1


@func()
def call_typeof_has_no_runtime_effect(x: i32) -> i32:
    v = x
    return typeof_has_no_runtime_effect(ref(v))


@func(exceptions=(ErrorA,))
def probe_raises() -> i32:
    raise ErrorA(7)


@func()
def typeof_a_raising_call() -> i32:
    # the error path of a raising call would have to leave the probe, which is
    # rejected
    t: Comptime = spy_typeof(probe_raises())
    if t == i32:
        return 0
    return 1


class SpyTypeOfTest(TestCase):
    """``syntax.typeof`` is a type probe: it types its argument (compiling the
    functions it names) without emitting any code, so the argument has no
    runtime effect."""

    def test_a_nested_call(self) -> None:
        self.assertEqual(typeof_a_nested_call(1), 1)

    def test_the_argument_has_no_runtime_effect(self) -> None:
        # the pointee is untouched: ``probe_bump`` was typed, not called
        self.assertEqual(call_typeof_has_no_runtime_effect(5), 5)

    def test_a_raising_call_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            typeof_a_raising_call()
        self.assertIn('typeof', str(ctx.exception))


# ---------------------------------------------------------------------------
# ``std.core.compile_error``: abort the compilation with a compile-time byte
# string (a string literal is encoded to bytes at parse time), so a spy body can
# report an error while it is being compiled
# ---------------------------------------------------------------------------


@func()
def compile_error_literal() -> i32:
    compile_error('boom')
    return 0


@func()
def compile_error_bytes() -> i32:
    compile_error(b'boom')
    return 0


@func()
def compile_error_number() -> i32:
    # the message must be a compile-time byte string
    compile_error(1)  # pyright: ignore[reportArgumentType]
    return 0


@func()
def compile_error_runtime(x: i32) -> i32:
    # ... and it may not be a runtime value
    compile_error(x)  # pyright: ignore[reportArgumentType]
    return 0


@func()
def compile_error_in_a_dead_branch() -> i32:
    # a compile-time condition makes one branch dead: its ``compile_error``
    # never runs
    syntax.comptime()
    x: i32 = 1
    if x == 1:
        return 1
    else:
        compile_error('dead')
        return 0


class SpyCompileErrorTest(TestCase):
    """``std.core.compile_error`` aborts the compilation with the message of its
    (compile-time, byte-string) argument."""

    def test_a_string_literal(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            compile_error_literal()
        self.assertEqual(str(ctx.exception), 'boom')

    def test_a_byte_string(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            compile_error_bytes()
        self.assertEqual(str(ctx.exception), 'boom')

    def test_a_non_byte_string_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            compile_error_number()
        self.assertIn('byte string', str(ctx.exception))

    def test_a_runtime_message_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            compile_error_runtime(1)
        self.assertIn('byte string', str(ctx.exception))

    def test_a_dead_branch_does_not_run_it(self) -> None:
        self.assertEqual(compile_error_in_a_dead_branch(), 1)


all_tests = [
    SpyComptimeMarkerTest,
    SpyTupleTest,
    SpyTypeOfTest,
    SpyCompileErrorTest,
]

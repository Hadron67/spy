import io
from contextlib import redirect_stdout
from unittest import TestCase

from ..compiler import (
    CompileError,
    compile_log,
    f32,
    f64,
    i32,
    u64,
)
from ..compiler import as_ as spy_as
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func
from ..std import (
    Numeric,
)

# ---------------------------------------------------------------------------
# functions under test
# ---------------------------------------------------------------------------


@func()
def smoke_test(a: i32, b: i32) -> i32:
    return a + b


@func()
def add[T: Numeric](a: T, b: T) -> T:
    return a + b


@func()
def add_u64(a: u64, b: u64) -> u64:
    return a + b


@func()
def sub(a: i32, b: i32) -> i32:
    return a - b


@func()
def mul(a: i32, b: i32) -> i32:
    return a * b


@func()
def negative_literals(n: i32) -> i32:
    # a negative literal is a ``-`` over an untyped literal, not a bare constant
    # like ``3``: it is evaluated into an expression temporary, which is a
    # compile-time box of the literal's own (untyped) type (see
    # ``sval.coerce_const``)
    if n < -10:
        return -1
    return n * -2 + 3


@func()
def mod(a: i32, b: i32) -> i32:
    return a % b


@func()
def scale(a: f64, b: f64) -> f64:
    return a * b + 1.0


@func()
def add_default[T: Numeric](a: T, b: T = 0) -> T:
    return a + b


def add_inline[T: Numeric](a: T, b: T) -> T:
    compile_log("add_inline was compiled")
    return a + b


def abs_inline(n: i32) -> i32:
    if n < 0:
        return -n
    else:
        return n


def clamp_inline(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def call_abs(n: i32) -> i32:
    return abs_inline(n)


@func()
def call_clamp(n: i32) -> i32:
    return clamp_inline(n)


@func()
def call_abs_plus_one(n: i32) -> i32:
    return abs_inline(n) + 1


@func()
def call_add[T: Numeric](a: T, b: T) -> T:
    return add(a, b)


@func()
def call_inline[T: Numeric](a: T, b: T) -> T:
    return add_inline(a, b)


@func()
def call_inline_log(a: i32, b: i32) -> i32:
    return add_inline(a, b)


@func()
def use_default[T: Numeric](a: T) -> T:
    return add_default(a) # pyright: ignore[reportReturnType]


@func()
def accumulate(a: i32, b: i32) -> i32:
    a += b
    return a


@func()
def sign(n: i32) -> i32:
    if n > 0:
        return 1
    else:
        return -1


@func()
def clamped(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def assign_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    return x


@func()
def assign_in_both_branches(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    else:
        x = b + 1
    return x


@func()
def assign_parameter_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # a parameter is bound in the function body's scope: a branch writes it
    if c:
        a = b
    return a


@func()
def assign_tuple_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # ``x`` is bound outside (written), ``y`` is bound nowhere (declared here)
    x = a
    if c:
        x, y = b, a
        return x + y
    return x


@func()
def assign_from_choose(c: spy_bool, d: spy_bool, a: i32, b: i32) -> i32:
    # the branches of an if-expression write the variable the branch sees
    x = a
    if d:
        x = a if c else b
    return x


@func()
def branch_declaration(c: spy_bool, a: i32) -> i32:
    # a name bound nowhere is declared in the branch it is assigned in
    if c:
        y = a
        y = y + 1
        return y
    return a


@func()
def use_branch_declaration(c: spy_bool, a: i32) -> i32:
    if c:
        y = a
    # ``y`` was declared inside the branch, so it is not bound here: a spy
    # compile error (pyright cannot tell, hence the ignore)
    return y  # pyright: ignore


@func()
def le(a: i32, b: i32) -> spy_bool:
    return a <= b


@func()
def eq(a: i32, b: i32) -> spy_bool:
    return a == b


@func()
def is_i32(a) -> spy_bool:
    return spy_typeof(a) == i32


@func()
def is_u64(a) -> spy_bool:
    return spy_typeof(a) == u64


@func()
def nothing(_: i32) -> None:
    pass


@func()
def fact(n: i32) -> i32:
    if n <= 1:
        return 1
    return n * fact(n - 1)


@func()
def is_even(n: i32) -> spy_bool:
    if n == 0:
        return True
    return is_odd(n - 1)


@func()
def is_odd(n: i32) -> spy_bool:
    if n == 0:
        return False
    return is_even(n - 1)


@func()
def inc(a: i32) -> i32:
    return a + 1


@func()
def twice_inc(a: i32) -> i32:
    return inc(inc(a))


@func()
def id_f32(a: f32) -> f32:
    return a


@func()
def untyped_local(n: i32) -> i32:
    x = 1
    x = 2
    return x + n


@func()
def untyped_return():
    return 1


@func()
def untyped_param(x) -> i32:
    return x


@func()
def call_untyped_param() -> i32:
    return untyped_param(1)


class SpyFunctionCallTest(TestCase):
    """Basic function calls: arithmetic, generics, calls between spy
    functions, inlining of plain functions, compile-time dispatch and
    recursion."""

    def test_smoke(self) -> None:
        self.assertEqual(smoke_test(2, 3), 5)
        self.assertEqual(smoke_test(-4, 1), -3)

    def test_arithmetic(self) -> None:
        self.assertEqual(sub(7, 12), -5)
        self.assertEqual(mul(6, 7), 42)
        self.assertEqual(mod(17, 5), 2)

    def test_negative_literals(self) -> None:
        # a negative literal is a ``-`` applied to an untyped literal, not a
        # bare constant like ``3``: it is evaluated into an expression
        # temporary, which is a compile-time box holding the literal's own
        # (untyped) type - a value position, a comparison and an operand alike
        self.assertEqual(negative_literals(0), 3)
        self.assertEqual(negative_literals(5), -7)
        self.assertEqual(negative_literals(-20), -1)

    def test_generic_int(self) -> None:
        # a plain Python int marshals to the default signed 64-bit type
        # (see ``dsl._INT_LITERAL_BITS``)
        self.assertEqual(add(2, 3), 5)

    def test_generic_float(self) -> None:
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)

    def test_float_promotion(self) -> None:
        self.assertAlmostEqual(scale(3.0, 0.5), 2.5)

    def test_non_default_integer_type(self) -> None:
        # ``spy.as_`` binds an argument to an explicit spy type
        self.assertEqual(add_u64(spy_as(2**63 - 1, u64), spy_as(2, u64)), 2**63 + 1)

    def test_default_argument(self) -> None:
        self.assertEqual(add_default(41), 41)
        self.assertEqual(add_default(40, 2), 42)

    def test_call_spy_function(self) -> None:
        self.assertEqual(call_add(20, 22), 42)

    def test_inline_plain_function(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline(20, 22), 42)

    def test_inline_with_runtime_branches(self) -> None:
        # the inlined body has a runtime ``if`` whose branches both return
        # (the ``Block``/``Break`` shape of an inlined return)
        self.assertEqual(call_abs(-5), 5)
        self.assertEqual(call_abs(5), 5)
        # a branch returns, the other falls through to the trailing return
        self.assertEqual(call_clamp(42), 42)
        self.assertEqual(call_clamp(101), 100)
        # the inlined result is consumed by the caller's continuation
        self.assertEqual(call_abs_plus_one(-5), 6)

    def test_call_with_default(self) -> None:
        self.assertEqual(use_default(7), 7)

    def test_augmented_assignment(self) -> None:
        self.assertEqual(accumulate(5, 6), 11)

    def test_runtime_if_both_return(self) -> None:
        self.assertEqual(sign(5), 1)
        self.assertEqual(sign(-5), -1)

    def test_runtime_if_fallthrough(self) -> None:
        self.assertEqual(clamped(42), 42)
        self.assertEqual(clamped(101), 100)

    def test_runtime_if_assignment(self) -> None:
        # an assignment in a branch writes the variable the branch sees
        self.assertEqual(assign_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_in_both_branches(True, 1, 2), 2)
        self.assertEqual(assign_in_both_branches(False, 1, 2), 3)
        # ... a parameter included
        self.assertEqual(assign_parameter_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_parameter_in_branch(False, 1, 2), 1)
        # a destructuring target written outside, one declared in the branch
        self.assertEqual(assign_tuple_in_branch(True, 1, 2), 3)
        self.assertEqual(assign_tuple_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_from_choose(False, True, 1, 2), 2)
        self.assertEqual(assign_from_choose(True, True, 1, 2), 1)
        self.assertEqual(assign_from_choose(True, False, 1, 2), 1)

    def test_branch_declaration(self) -> None:
        # a name bound nowhere is declared in the branch it is assigned in,
        # and is invisible after it
        self.assertEqual(branch_declaration(True, 5), 6)
        self.assertEqual(branch_declaration(False, 5), 5)
        with self.assertRaises(CompileError):
            use_branch_declaration(True, 5)

    def test_comparisons(self) -> None:
        self.assertTrue(le(1, 2))
        self.assertFalse(le(2, 1))
        self.assertTrue(eq(3, 3))
        self.assertFalse(eq(3, 4))

    def test_comptime_typeof(self) -> None:
        # a plain Python int marshals to i64 (see ``dsl._INT_LITERAL_BITS``)
        self.assertFalse(is_i32(1))
        self.assertTrue(is_i32(spy_as(1, i32)))
        self.assertFalse(is_i32(1.0))
        self.assertTrue(is_u64(spy_as(1, u64)))

    def test_void_function(self) -> None:
        self.assertIsNone(nothing(3))

    def test_recursion(self) -> None:
        self.assertEqual(fact(5), 120)
        self.assertEqual(fact(0), 1)
        self.assertEqual(fact(10), 3628800)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(is_even(10))
        self.assertFalse(is_even(7))
        self.assertTrue(is_odd(7))

    def test_cross_module_call(self) -> None:
        # ``inc`` is compiled on its own first, so the compilation of
        # ``twice_inc`` references it as an external symbol of an earlier
        # module (the modular-compilation path)
        self.assertEqual(inc(10), 11)
        self.assertEqual(twice_inc(10), 12)

    def test_f32(self) -> None:
        self.assertAlmostEqual(id_f32(spy_as(0.5, f32)), 0.5)

    def test_specialization_is_cached(self) -> None:
        # repeated calls reuse the compiled specialization
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)

    def test_compile_error_on_unsupported_expression(self) -> None:
        with self.assertRaises(CompileError):
            div(1, 2)

    def test_untyped_integer_slot_is_rejected(self) -> None:
        # an untyped integer literal has no runtime type of its own: a
        # slot that has to live in memory must declare its type
        with self.assertRaises(CompileError):
            untyped_local(5)
        with self.assertRaises(CompileError):
            untyped_return()
        # the same holds for the slot of an unannotated parameter, which
        # is typed by the call (here: by an untyped literal argument)
        with self.assertRaises(CompileError):
            call_untyped_param()


@func()
def div(a: i32, b: i32) -> i32:
    return a / b # pyright: ignore[reportReturnType]


class SpyCompileLogTest(TestCase):
    def test_compile_log_prints_at_compile_time(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline_log(1, 2), 3)
        self.assertIn('add_inline was compiled', out.getvalue())


all_tests = [
    SpyFunctionCallTest,
    SpyCompileLogTest,
]

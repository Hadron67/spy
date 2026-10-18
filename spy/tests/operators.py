from unittest import TestCase

from ..compiler import (
    CompileError,
    f32,
    f64,
    i32,
    i64,
    u64,
)
from ..compiler import as_ as spy_as
from ..compiler import bool as spy_bool
from ..compiler.dsl import func, struct
from ..std.int import int_to_float

# ---------------------------------------------------------------------------
# operators: the complete integer/float operator set (true division, floor
# division, exponentiation and the bitwise/shift operators), the augmented
# form of every one of them, and struct operator overloading
# ---------------------------------------------------------------------------


@func()
def int_true_div(a: i32, b: i32) -> f64:
    # a runtime ``int / int`` divides in ``CompileVars.int_div_type`` (``f64`` by
    # default), and the operands convert to it implicitly: an ``i32`` fits the
    # exact integer range of ``f64``
    return a / b


@func()
def int_floor_div(a: i32, b: i32) -> i32:
    return a // b


@func()
def bool_comparisons(a: bool, b: bool) -> i32:
    # a ``bool`` is an ``i1``: every comparison operator is defined on it, and
    # the ordering is the ``i1`` one (``False < True``)
    n: i32 = 0
    if a == b:
        n = n + 1
    if a != b:
        n = n + 2
    if a < b:
        n = n + 4
    if a <= b:
        n = n + 8
    if a > b:
        n = n + 16
    if a >= b:
        n = n + 32
    return n


@func()
def bool_compared_with_a_literal(x: bool) -> i32:
    # a literal is a compile-time ``bool``: it converts to the runtime operand
    n: i32 = 0
    if x == True:
        n = n + 1
    if False != x:
        n = n + 2
    if x != True:
        n = n + 4
    return n


@func()
def float_floor_div(a: f64, b: f64) -> f64:
    return a // b


@func()
def mixed_float_add(a: f32, b: f64) -> f64:
    # the narrower float converts to the wider one, whichever operand it is (the
    # peer type of the two is what the operation is computed in)
    return a + b


@func()
def mixed_float_add_reflected(a: f64, b: f32) -> f64:
    return a + b


@func()
def mixed_int_float_add(a: i32, b: f64) -> f64:
    # an integer that converts to the float losslessly takes part in the
    # arithmetic: the peer type of the two is that float
    return a + b


@func()
def mixed_int_float_mul_literal(a: f64) -> f64:
    # an untyped integer literal converts to the float as well
    return a * 2


@func()
def mixed_int_float_add_lossy(a: i64, b: f64) -> f64:
    # ``i64`` does not fit the exact integer range of ``f64``: it does not join
    return a + b


@func()
def int_pow_const(a: i32) -> i32:
    return a ** 3


@func()
def int_pow_neg_const(a: i32) -> f64:
    return int_to_float(a, f64) ** -2


@func()
def int_pow_runtime(a: i32, e: i32) -> f64:
    return int_to_float(a, f64) ** int_to_float(e, f64)


@func()
def uint_pow_runtime(a: u64, e: u64) -> f64:
    return int_to_float(a, f64) ** int_to_float(e, f64)


@func()
def float_pow(a: f64, b: f64) -> f64:
    return a ** b


@func()
def float_pow_over_the_unroll_cap(a: f64) -> f64:
    # a compile-time exponent above ``CompileVars.max_exp_unroll`` becomes a
    # runtime loop instead of an unfolded sequence
    return a ** 5000


@func()
def comptime_pow() -> i32:
    return 2 ** 10


@func()
def bitwise_ops(a: i32, b: i32) -> i32:
    return (a | b) + (a & b) + (a ^ b) + (a << 1) + (a >> 1) + ~a


@func()
def bitwise_assign(a: i32, b: i32) -> i32:
    a |= b
    a &= b
    a ^= b
    a <<= 1
    a >>= 1
    return a


@func()
def augmented_ops(a: i32, b: i32) -> f64:
    a += b
    a -= b
    a *= b
    a //= b
    return int_to_float(a, f64) / int_to_float(b, f64)


@struct()
class OpVec:
    """A struct that overloads the arithmetic, comparison, unary and bitwise
    operators through magic methods (undecorated, so each is inlined)."""

    v: i32

    def __add__(self, other: OpVec) -> OpVec:
        return OpVec(self.v + other.v)

    def __radd__(self, k: i32) -> OpVec:
        return OpVec(self.v + k)

    def __iadd__(self, other: OpVec) -> OpVec:  # pyright: ignore  # noqa: PYI034
        self.v = self.v + other.v
        return self

    def __eq__(self, other: OpVec) -> spy_bool:  # pyright: ignore
        return self.v == other.v

    def __lt__(self, other: OpVec) -> spy_bool:
        return self.v < other.v

    def __neg__(self) -> OpVec:
        return OpVec(-self.v)

    def __bool__(self) -> spy_bool:
        return self.v != 0

    def __len__(self) -> i32:
        return self.v * 2

    def __and__(self, other: OpVec) -> OpVec:
        return OpVec(self.v & other.v)


@func()
def overload_add(a: i32, b: i32) -> i32:
    return (OpVec(a) + OpVec(b)).v


@func()
def overload_reflected_add(a: i32) -> i32:
    return (a + OpVec(5)).v


@func()
def overload_inplace_add(a: i32, b: i32) -> i32:
    x: OpVec = OpVec(a)
    x += OpVec(b)
    return x.v


@func()
def overload_less_than(a: i32, b: i32) -> spy_bool:
    return OpVec(a) < OpVec(b)


@func()
def overload_eq(a: i32, b: i32) -> spy_bool:
    return OpVec(a) == OpVec(b)


@func()
def overload_negate(a: i32) -> i32:
    return (-OpVec(a)).v


@func()
def overload_bool(a: i32) -> spy_bool:
    # ``__bool__`` answers the ``AsBool`` an ``if`` condition (or a ``not``)
    # runs, so this exercises it directly
    if OpVec(a):  # noqa: SIM103
        return True
    return False


@func()
def overload_len(a: i32) -> i32:
    # ``len`` of a struct answers through its own ``__len__`` (undecorated, so
    # the method is inlined)
    return len(OpVec(a))


@func()
def overload_bitand(a: i32, b: i32) -> i32:
    return (OpVec(a) & OpVec(b)).v


class SpyOperatorTest(TestCase):
    """The complete integer/float operator set, the augmented form of every
    operator, and the struct magic methods."""

    def test_integer_division_is_true_division(self) -> None:
        self.assertAlmostEqual(int_true_div(7, 2), 3.5)
        self.assertAlmostEqual(int_true_div(-7, 2), -3.5)

    def test_integer_floor_division(self) -> None:
        self.assertEqual(int_floor_div(7, 2), 3)
        self.assertEqual(int_floor_div(-7, 2), -4)
        self.assertEqual(int_floor_div(7, -2), -4)
        self.assertEqual(int_floor_div(-7, -2), 3)

    def test_float_floor_division(self) -> None:
        self.assertAlmostEqual(float_floor_div(7.5, 2.0), 3.0)
        self.assertAlmostEqual(float_floor_div(-7.5, 2.0), -4.0)

    def test_mixed_float_widths_widen_in_either_order(self) -> None:
        self.assertAlmostEqual(mixed_float_add(1.5, 2.25), 3.75)
        self.assertAlmostEqual(mixed_float_add_reflected(1.5, 2.25), 3.75)

    def test_an_integer_joins_float_arithmetic_when_it_is_lossless(self) -> None:
        self.assertAlmostEqual(mixed_int_float_add(3, 0.5), 3.5)
        self.assertAlmostEqual(mixed_int_float_mul_literal(2.5), 5.0)

    def test_an_inexact_integer_does_not_join_float_arithmetic(self) -> None:
        with self.assertRaises(CompileError):
            mixed_int_float_add_lossy(1, 0.5)

    def test_bool_comparisons(self) -> None:
        # ``False`` is the smaller one, and each operator contributes its bit
        self.assertEqual(bool_comparisons(False, False), 41)
        self.assertEqual(bool_comparisons(False, True), 14)
        self.assertEqual(bool_comparisons(True, False), 50)
        self.assertEqual(bool_comparisons(True, True), 41)

    def test_a_bool_compares_with_a_literal(self) -> None:
        self.assertEqual(bool_compared_with_a_literal(True), 3)
        self.assertEqual(bool_compared_with_a_literal(False), 4)

    def test_compile_time_exponent(self) -> None:
        self.assertEqual(int_pow_const(3), 27)
        self.assertEqual(comptime_pow(), 1024)

    def test_negative_compile_time_exponent(self) -> None:
        self.assertAlmostEqual(int_pow_neg_const(2), 0.25)

    def test_runtime_signed_exponent(self) -> None:
        self.assertAlmostEqual(int_pow_runtime(2, 10), 1024.0)
        self.assertAlmostEqual(int_pow_runtime(2, -3), 0.125)
        self.assertAlmostEqual(int_pow_runtime(5, 0), 1.0)

    def test_runtime_unsigned_exponent(self) -> None:
        self.assertAlmostEqual(uint_pow_runtime(spy_as(2, u64), spy_as(5, u64)), 32.0)

    def test_float_exponent(self) -> None:
        self.assertAlmostEqual(float_pow(4.0, 0.5), 2.0)
        self.assertAlmostEqual(float_pow(2.0, 10.0), 1024.0)

    def test_an_exponent_over_the_unroll_cap_is_a_runtime_loop(self) -> None:
        self.assertAlmostEqual(float_pow_over_the_unroll_cap(1.0001), 1.6486800559, places=6)

    def test_bitwise_operators(self) -> None:
        # 12|10=14, 12&10=8, 12^10=6, 12<<1=24, 12>>1=6, ~12=-13
        self.assertEqual(bitwise_ops(0b1100, 0b1010), 45)

    def test_bitwise_augmented_assignment(self) -> None:
        self.assertEqual(bitwise_assign(0b1100, 0b1010), 0)

    def test_every_operator_has_an_augmented_form(self) -> None:
        self.assertAlmostEqual(augmented_ops(12, 4), 3.0)

    def test_struct_add(self) -> None:
        self.assertEqual(overload_add(2, 3), 5)

    def test_struct_reflected_add(self) -> None:
        self.assertEqual(overload_reflected_add(10), 15)

    def test_struct_inplace_add(self) -> None:
        self.assertEqual(overload_inplace_add(2, 3), 5)

    def test_struct_comparisons(self) -> None:
        self.assertTrue(overload_less_than(2, 3))
        self.assertFalse(overload_less_than(3, 2))
        self.assertTrue(overload_eq(3, 3))
        self.assertFalse(overload_eq(3, 4))

    def test_struct_negate(self) -> None:
        self.assertEqual(overload_negate(7), -7)

    def test_struct_bool(self) -> None:
        self.assertFalse(overload_bool(0))
        self.assertTrue(overload_bool(1))

    def test_struct_len(self) -> None:
        self.assertEqual(overload_len(3), 6)

    def test_struct_bitand(self) -> None:
        self.assertEqual(overload_bitand(0b1100, 0b1010), 8)


all_tests = [
    SpyOperatorTest,
]

from unittest import TestCase

from ..compiler import as_ as spy_as
from ..compiler import bool as spy_bool
from ..compiler import (
    f64,
    i32,
    u64,
)
from ..compiler.dsl import func, struct

# ---------------------------------------------------------------------------
# operators: the complete integer/float operator set (true division, floor
# division, exponentiation and the bitwise/shift operators), the augmented
# form of every one of them, and struct operator overloading
# ---------------------------------------------------------------------------


@func()
def int_true_div(a: i32, b: i32) -> f64:
    return a / b


@func()
def int_floor_div(a: i32, b: i32) -> i32:
    return a // b


@func()
def float_floor_div(a: f64, b: f64) -> f64:
    return a // b


@func()
def int_pow_const(a: i32) -> i32:
    return a ** 3


@func()
def int_pow_neg_const(a: i32) -> f64:
    return a ** -2


@func()
def int_pow_runtime(a: i32, e: i32) -> f64:
    return a ** e


@func()
def uint_pow_runtime(a: u64, e: u64) -> f64:
    return a ** e


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
    return a / b


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

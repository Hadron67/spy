from unittest import TestCase

from ..compiler import (
    bool as spy_bool,
)
from ..compiler import (
    c64,
    c128,
    f32,
    f64,
    i32,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func
from ..compiler.syntax import Comptime
from ..std.reflect import ComplexType, type_info

# ---------------------------------------------------------------------------
# complex numbers: the ``Complex[T]`` type, the implicit ``T -> Complex[T]``
# conversion, the four arithmetic operations (computed part by part, inline)
# and the compile-time reflection of the type
# ---------------------------------------------------------------------------


@func()
def complex_add(a: c128, b: c128) -> c128:
    return a + b


@func()
def complex_sub(a: c128, b: c128) -> c128:
    return a - b


@func()
def complex_mul(a: c128, b: c128) -> c128:
    return a * b


@func()
def complex_div(a: c128, b: c128) -> c128:
    return a / b


@func()
def complex_add_real(z: c128, r: f64) -> c128:
    return z + r


@func()
def complex_mul_real(z: c128, r: f64) -> c128:
    return z * r


@func()
def complex_add_int(z: c128, r: i32) -> c128:
    return z + r


@func()
def complex_literal() -> c128:
    return 1 + 2j


@func()
def complex_literal_mul() -> c128:
    return (1 + 2j) * (3 + 4j)


@func()
def complex_literal_add_real() -> c128:
    return 2j + 3


@func()
def complex_real(z: c128) -> f64:
    return z.real


@func()
def complex_imag(z: c128) -> f64:
    return z.imag


@func()
def complex_local(a: c128, b: c128) -> c128:
    z: c128 = a + b
    return z * z


@func()
def complex64_add(a: c64, b: c64) -> c64:
    return a + b


@func()
def complex64_widen(z: c64) -> c128:
    return z + 0j


@func()
def complex_typeof_elem(z: c128) -> spy_bool:
    info: Comptime = type_info(spy_typeof(z))
    if isinstance(v := info, ComplexType):
        return v.elem == f64
    return False


@func()
def complex_reflect_c64_elem() -> spy_bool:
    info: Comptime = type_info(c64)
    if isinstance(v := info, ComplexType):
        return v.elem == f32
    return False


class SpyComplexTest(TestCase):
    """Complex numbers: construction, arithmetic, field access and reflection."""

    def test_two_complex_arguments_add(self) -> None:
        self.assertEqual(complex_add(1 + 2j, 3 + 4j), 4 + 6j)

    def test_two_complex_arguments_subtract(self) -> None:
        self.assertEqual(complex_sub(1 + 2j, 3 + 4j), -2 - 2j)

    def test_two_complex_arguments_multiply(self) -> None:
        # (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8 i^2 = -5 + 10i
        self.assertEqual(complex_mul(1 + 2j, 3 + 4j), -5 + 10j)

    def test_two_complex_arguments_divide(self) -> None:
        self.assertAlmostEqual(complex_div(1 + 2j, 3 + 4j), (1 + 2j) / (3 + 4j))

    def test_a_real_operand_is_converted_implicitly(self) -> None:
        self.assertEqual(complex_add_real(1 + 2j, 3.0), 4 + 2j)
        self.assertEqual(complex_mul_real(1 + 2j, 2.0), 2 + 4j)

    def test_an_integer_operand_is_converted_implicitly(self) -> None:
        self.assertEqual(complex_add_int(1 + 2j, 2), 3 + 2j)

    def test_compile_time_literals_fold(self) -> None:
        self.assertEqual(complex_literal(), 1 + 2j)
        self.assertEqual(complex_literal_mul(), -5 + 10j)
        self.assertEqual(complex_literal_add_real(), 3 + 2j)

    def test_real_and_imaginary_parts_are_fields(self) -> None:
        self.assertEqual(complex_real(3 + 4j), 3.0)
        self.assertEqual(complex_imag(3 + 4j), 4.0)

    def test_a_local_complex_variable(self) -> None:
        # (1 + 2i + 3 + 4i)^2 = (4 + 6i)^2 = 16 + 48i - 36 = -20 + 48i
        self.assertEqual(complex_local(1 + 2j, 3 + 4j), -20 + 48j)

    def test_a_complex64_value(self) -> None:
        self.assertEqual(complex64_add(1 + 2j, 3 + 4j), 4 + 6j)

    def test_a_complex64_value_widens_to_complex128(self) -> None:
        self.assertEqual(complex64_widen(1 + 2j), 1 + 2j)

    def test_the_type_of_a_complex_value_reflects_its_element(self) -> None:
        self.assertTrue(complex_typeof_elem(0j))

    def test_a_complex64_type_reflects_its_element(self) -> None:
        self.assertTrue(complex_reflect_c64_elem())


all_tests = [
    SpyComplexTest,
]

from unittest import TestCase

from ..compiler import (
    Comptime,
    c64,
    c128,
    f32,
    f64,
    func,
    i32,
)
from ..compiler import (
    bool as spy_bool,
)
from ..compiler import typeof as spy_typeof
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
    # an integer that converts to a float losslessly takes the ``T ->
    # Complex[T]`` conversion too: its real part is the integer, its imaginary
    # one zero
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
def complex_from_real_local(r: f64) -> c128:
    # the implicit ``T -> Complex[T]`` conversion applies to a store: a real
    # value assigned to a ``c128`` place becomes ``r + 0j``
    z: c128 = r
    return z


@func()
def complex_from_int_local(i: i32) -> c128:
    z: c128 = i
    return z


@func()
def complex_from_real_comptime() -> c128:
    # the same conversion into a *compile-time* aggregate destination
    z: Comptime[c128] = 1.5
    return z


@func()
def complex_from_real_return(r: f64) -> c128:
    # ... and on the way out: a real returned as a complex
    return r


@func()
def complex64_from_real_local(r: f32) -> c64:
    z: c64 = r
    return z


@func()
def complex_widen_local(z: c64) -> c128:
    # a complex of a narrower element type widens on a store
    w: c128 = z
    return w


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

    def test_a_real_is_converted_into_a_complex_local(self) -> None:
        self.assertEqual(complex_from_real_local(2.5), 2.5 + 0j)

    def test_an_integer_is_converted_into_a_complex_local(self) -> None:
        self.assertEqual(complex_from_int_local(3), 3 + 0j)

    def test_a_real_is_converted_into_a_compile_time_complex(self) -> None:
        # the conversion also applies where the destination is a compile-time
        # aggregate (the real part is the value, the imaginary part is zero)
        self.assertEqual(complex_from_real_comptime(), 1.5 + 0j)

    def test_a_real_is_converted_when_returned_as_complex(self) -> None:
        self.assertEqual(complex_from_real_return(4.5), 4.5 + 0j)

    def test_a_real_is_converted_into_a_complex64_local(self) -> None:
        self.assertEqual(complex64_from_real_local(1.5), 1.5 + 0j)

    def test_a_complex64_widens_into_a_complex128_local(self) -> None:
        self.assertEqual(complex_widen_local(1 + 2j), 1 + 2j)

    def test_the_type_of_a_complex_value_reflects_its_element(self) -> None:
        self.assertTrue(complex_typeof_elem(0j))

    def test_a_complex64_type_reflects_its_element(self) -> None:
        self.assertTrue(complex_reflect_c64_elem())


all_tests = [
    SpyComplexTest,
]

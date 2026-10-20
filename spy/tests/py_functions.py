"""Plain Python functions used at compile time (``sval.PyFunction``).

A function that is *not* decorated with ``@func`` is a compile-time helper: a
call to it is evaluated while the HIR runs, on plain Python objects - an ``int``
(also ``bool``), ``float``, ``complex``, ``bytes`` or ``str`` (spy's compile-time
text string).  Its result is one of those plain objects too, and takes the type of
the call's result location (an inferred one keeping the value's own).  Any other
argument or result - a runtime value, a spy type, an aggregate, the null value,
... - is a compile error, and a ``PyFunction`` never reaches runtime code.
"""

from unittest import TestCase

from ..compiler import (
    CompileError,
    Comptime,
    Option,
    c128,
    f64,
    func,
    i32,
)
from ..compiler import bool as spy_bool


def double(n: int) -> int:
    return n * 2


def difference(a: float, b: float) -> float:
    return a - b


def conjugate(z: complex) -> complex:
    return z.conjugate()


def join(a: bytes, b: bytes) -> bytes:
    return a + b


def upper(s: bytes) -> str:
    """Returns a Python ``str``: it crosses as spy's compile-time ``str``."""
    return s.decode("utf-8").upper()


def shout(s: str) -> str:
    """Takes and returns a Python ``str`` (spy's compile-time ``str``)."""
    return s.upper()


def is_positive(n: int) -> bool:
    return n > 0


def is_none(v) -> bool:
    """A ``None`` in, a ``None`` out: an option, a tagged union and the null/void
    values all cross as the Python ``None``/their unwrapped payload."""
    return v is None


def plus(n: int) -> int:
    """An option's payload and a tagged union's variant cross as the value they
    carry."""
    return n + 100


def maybe_double(n: int):
    """``None`` comes back as the null value (the absent value of an option)."""
    return None if n < 0 else n * 2


def a_list(n: int) -> list[int]:
    """A non-scalar result: no ``PyFunction`` may return one."""
    return [n]


@func()
def py_call_with_a_literal() -> i32:
    return double(4)


@func()
def py_call_with_a_comptime_local() -> i32:
    n: Comptime = 21
    return double(n)


@func()
def py_call_through_a_stored_value() -> i32:
    # the function value is a compile-time value like any other: it is stored in
    # a ``Comptime`` variable and read back (through a pointer, like any callee)
    f: Comptime = double
    return f(6)


@func()
def py_apply(f: Comptime, n: Comptime) -> i32:
    # a ``PyFunction`` crosses as a compile-time argument
    return f(n)


@func()
def py_call_a_passed_function() -> i32:
    return py_apply(double, 5)


@func()
def py_call_float() -> f64:
    return difference(3.5, 1.25)


@func()
def py_call_complex() -> c128:
    return conjugate(1 + 2j)


@func()
def py_call_bool():
    return is_positive(3)


@func()
def py_call_bytes():
    return join(b"ab", b"cd")


@func()
def py_call_str():
    return upper(b"hey")


@func()
def py_call_str_in():
    return shout("hey")


@func()
def py_call_with_none() -> spy_bool:
    # the ``None`` literal is the untyped null value
    return is_none(None)


@func()
def py_call_with_an_absent_option() -> spy_bool:
    # an option's absent value is the *typed* null; the variable must hold the
    # option at compile time (a runtime option cannot be handed to Python)
    o: Comptime[Option[i32]] = None
    return is_none(o)


@func()
def py_call_with_a_present_option() -> i32:
    # an option's present value hands its payload over
    o: Comptime[Option[i32]] = 7
    return plus(o)


@func()
def py_call_with_a_tagged_union() -> i32:
    # a tagged union hands the value of its variant over
    u: Comptime[i32 | spy_bool] = 7
    return plus(u)


@func()
def py_void() -> None:
    pass


@func()
def py_call_with_a_void() -> spy_bool:
    # the void unit value is the Python ``None`` too
    return is_none(py_void())


@func()
def py_call_returning_none() -> Option[i32]:
    # ``None`` comes back as the null value: the absent option
    n: Comptime = -1
    return maybe_double(n)


@func()
def py_call_returning_a_value() -> Option[i32]:
    n: Comptime = 5
    return maybe_double(n)


# ---------------------------------------------------------------------------
# errors: only plain Python scalars may cross
# ---------------------------------------------------------------------------


@func()
def py_call_with_a_runtime_argument(x: i32) -> i32:
    # a runtime value cannot be handed to a Python function: the call happens at
    # compile time
    return double(x)


@func()
def py_call_with_a_non_scalar_argument() -> i32:
    # a spy type is a compile-time value, but not a plain Python scalar
    t: Comptime = i32
    return double(t)


@func()
def py_call_a_bad_return():
    return a_list(1)


class SpyPyFunctionTest(TestCase):
    """Calling a plain Python function at compile time."""

    def test_a_plain_python_function_is_still_callable_from_python(self) -> None:
        self.assertEqual(double(3), 6)

    def test_a_literal_argument(self) -> None:
        self.assertEqual(py_call_with_a_literal(), 8)

    def test_a_compile_time_local_argument(self) -> None:
        self.assertEqual(py_call_with_a_comptime_local(), 42)

    def test_a_stored_function_value(self) -> None:
        self.assertEqual(py_call_through_a_stored_value(), 12)

    def test_a_passed_function_value(self) -> None:
        self.assertEqual(py_call_a_passed_function(), 10)

    def test_a_float_result(self) -> None:
        self.assertAlmostEqual(py_call_float(), 2.25)

    def test_a_complex_result(self) -> None:
        self.assertEqual(py_call_complex(), 1 - 2j)

    def test_a_bool_result(self) -> None:
        self.assertEqual(py_call_bool(), True)

    def test_a_bytes_result(self) -> None:
        self.assertEqual(py_call_bytes(), b"abcd")

    def test_a_str_result_is_a_text_string(self) -> None:
        self.assertEqual(py_call_str(), "HEY")

    def test_a_str_argument_and_result_cross_as_str(self) -> None:
        self.assertEqual(py_call_str_in(), "HEY")

    def test_the_none_literal_is_passed_as_none(self) -> None:
        self.assertEqual(py_call_with_none(), True)

    def test_an_absent_option_is_passed_as_none(self) -> None:
        self.assertEqual(py_call_with_an_absent_option(), True)

    def test_a_present_option_hands_its_payload_over(self) -> None:
        self.assertEqual(py_call_with_a_present_option(), 107)

    def test_a_tagged_union_hands_its_variant_over(self) -> None:
        self.assertEqual(py_call_with_a_tagged_union(), 107)

    def test_a_void_value_is_passed_as_none(self) -> None:
        self.assertEqual(py_call_with_a_void(), True)

    def test_a_none_result_is_the_absent_option(self) -> None:
        self.assertIsNone(py_call_returning_none())

    def test_a_present_result_is_the_option_payload(self) -> None:
        self.assertEqual(py_call_returning_a_value(), 10)

    def test_a_runtime_argument_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            py_call_with_a_runtime_argument(1)

    def test_a_non_scalar_argument_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            py_call_with_a_non_scalar_argument()

    def test_a_non_scalar_result_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            py_call_a_bad_return()


all_tests = [
    SpyPyFunctionTest,
]

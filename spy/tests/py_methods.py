"""Undecorated methods are plain Python functions (``sval.PyFunction``).

A method that is not decorated with ``@func`` has no special method semantics:
like an undecorated module-level function, its body runs in Python at compile
time and its arguments must be plain Python objects.  A call on a runtime value
therefore cannot reach it - an instance method would pass a runtime receiver -
so a runtime method has to be spelled ``@func(inline=True)`` (inlined) or
``@func()`` (compiled into a native call).  A class-name call passes every
argument explicitly, so it can still reach an undecorated static method (or a
method whose receiver is itself a plain Python value, such as an enumeration
member) at compile time.
"""

from enum import Enum
from unittest import TestCase

from ..compiler import (
    CompileError,
    func,
    i32,
    struct,
)
from ..compiler import (
    bool as spy_bool,
)


@struct()
class PlainMethod:
    x: i32

    def get(self) -> i32:
        return self.x


@func()
def call_plain_method(x: i32) -> i32:
    # ``get`` is a ``PyFunction``: an instance call would hand it a runtime
    # pointer as ``self``
    return PlainMethod(x).get()


@struct()
class InlineMethod:
    x: i32

    @func(inline=True)
    def get(self) -> i32:
        return self.x


@func()
def call_inline_method(x: i32) -> i32:
    return InlineMethod(x).get()


@struct()
class PyStatic:
    marker: i32 = 0

    @staticmethod
    def split_len(data: bytes) -> i32:
        # a Python-only body: the call is evaluated by Python, not compiled
        return len(data.split(b'b'))


@func()
def call_py_static() -> i32:
    return PyStatic.split_len(b'abcd')


@struct()
class GenericPyStatic[T]:
    @staticmethod
    def py_id(value: T) -> T:
        return value


@func()
def call_generic_py_static() -> i32:
    # a ``PyFunction`` needs no struct type-argument substitution, so the
    # specialization's type arguments are not bound into it
    return GenericPyStatic[i32].py_id(5)


@struct()
class PyStaticArgs:
    @staticmethod
    def py_add(a: i32, b: i32) -> i32:
        return a + b


@func()
def call_py_static_with_literals() -> i32:
    return PyStaticArgs.py_add(2, 3)


@func()
def call_py_static_with_runtime_args(x: i32) -> i32:
    # ``x`` is not a compile-time value, so it cannot cross into Python
    return PyStaticArgs.py_add(x, 1)


class PyEnum(Enum):
    A = 1
    B = 2

    def py_is_a(self) -> spy_bool:
        return self is PyEnum.A


@func()
def call_py_enum_class_method() -> spy_bool:
    return PyEnum.py_is_a(PyEnum.A)


@func()
def call_py_enum_instance(a: PyEnum) -> spy_bool:
    # an instance call passes a runtime enumeration value as ``self``
    return a.py_is_a()


class SpyPyMethodTest(TestCase):
    """Undecorated methods as compile-time Python functions."""

    def test_an_instance_call_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as caught:
            call_plain_method(3)
        message = str(caught.exception)
        self.assertIn('is not decorated with @func', message)
        self.assertIn('PyFunction', message)
        self.assertIn('@func(inline=True)', message)

    def test_an_inline_method_still_takes_a_runtime_receiver(self) -> None:
        self.assertEqual(call_inline_method(7), 7)

    def test_a_static_method_is_evaluated_by_python(self) -> None:
        self.assertEqual(call_py_static(), 2)

    def test_a_generic_static_method_is_not_bound(self) -> None:
        self.assertEqual(call_generic_py_static(), 5)

    def test_a_class_name_call_with_comptime_arguments(self) -> None:
        self.assertEqual(call_py_static_with_literals(), 5)

    def test_a_class_name_call_with_a_runtime_argument_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            call_py_static_with_runtime_args(4)

    def test_an_enum_class_name_call_passes_a_python_member(self) -> None:
        self.assertTrue(call_py_enum_class_method())

    def test_an_enum_instance_call_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            call_py_enum_instance(PyEnum.A)

    def test_python_side_calls_the_plain_function(self) -> None:
        # on the Python side the method is the ordinary Python function of the
        # class body, with the instance wrapper as ``self``
        p = PlainMethod(9)
        self.assertEqual(p.get(), 9)
        self.assertEqual(PlainMethod.get(p), 9)

    def test_python_side_static_method_is_unbound(self) -> None:
        self.assertEqual(PyStatic.split_len(b'abcd'), 2)
        self.assertEqual(PyStatic(0).split_len(b'abcd'), 2)


all_tests = [
    SpyPyMethodTest,
]

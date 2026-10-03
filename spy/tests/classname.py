from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
)
from ..compiler.dsl import func, struct
from ..std.mem import DynamicAllocator
from .structs import struct_type

# ---------------------------------------------------------------------------
# calling a struct's function through the class name, and inherited methods
# ---------------------------------------------------------------------------


class _ClassNamePlain:
    # a plain class (no ``@struct()``) used as a namespace of functions
    def bar() -> i32:  # pyright: ignore
        return 7

    def baz(y: i32) -> i32:  # pyright: ignore
        return y + 1


class _ClassNameBase:
    # a plain base class: a struct that derives from it inherits its methods
    def total(self, x: i32) -> i32:
        return self.a + x  # pyright: ignore


@struct()
class ClassNameLib:
    x: i32

    @staticmethod
    def foo() -> i32:
        return 5

    def add(self, y: i32) -> i32:
        return self.x + y


@struct()
class ClassNameChild(_ClassNameBase):
    a: i32


@struct()
class ClassNameGeneric[T]:
    v: T

    def get(self) -> T:
        return self.v


@func()
def class_name_static() -> i32:
    return ClassNameLib.foo()


@func()
def class_name_static_on_value(a: i32) -> i32:
    v = ClassNameLib(a)
    return v.foo()


@func()
def class_name_method_class_name(a: i32) -> i32:
    v = ClassNameLib(a)
    return ClassNameLib.add(v, 10)


@func()
def class_name_method_value(a: i32) -> i32:
    v = ClassNameLib(a)
    return v.add(10)


@func()
def class_name_plain() -> i32:
    return _ClassNamePlain.bar()  # pyright: ignore


@func()
def class_name_plain_method(a: i32) -> i32:
    return _ClassNamePlain.baz(a)  # pyright: ignore


@func()
def class_name_inherited_class_name(a: i32) -> i32:
    c = ClassNameChild(a)
    return ClassNameChild.total(c, 5)


@func()
def class_name_inherited_value(a: i32) -> i32:
    c = ClassNameChild(a)
    return c.total(5)


@func()
def class_name_missing() -> i32:
    return ClassNameLib.missing()  # pyright: ignore


@func()
def class_name_template(a: i32) -> i32:
    return ClassNameGeneric.get(a)  # pyright: ignore


class SpyClassNameCallTest(TestCase):
    """Calling a struct's function through the class name (``Foo.m(...)``,
    ``Foo[i32].m(x)``): the call passes every argument explicitly, with no
    implicit ``self``; a plain class (no ``@struct()``) works as a namespace
    of functions, and a struct inherits the methods of a plain base."""

    def test_a_static_method_through_the_class_name(self) -> None:
        self.assertEqual(class_name_static(), 5)

    def test_a_static_method_on_a_value_takes_no_self(self) -> None:
        self.assertEqual(class_name_static_on_value(9), 5)

    def test_a_method_through_the_class_name_passes_self_explicitly(self) -> None:
        self.assertEqual(class_name_method_class_name(1), 11)

    def test_a_method_on_a_value_still_passes_the_receiver(self) -> None:
        self.assertEqual(class_name_method_value(1), 11)

    def test_a_plain_class_is_a_namespace(self) -> None:
        self.assertEqual(class_name_plain(), 7)
        self.assertEqual(class_name_plain_method(41), 42)

    def test_a_struct_inherits_a_base_method_through_the_class_name(self) -> None:
        self.assertEqual(class_name_inherited_class_name(100), 105)

    def test_a_struct_inherits_a_base_method_on_a_value(self) -> None:
        self.assertEqual(class_name_inherited_value(100), 105)

    def test_a_missing_method_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            class_name_missing()

    def test_a_struct_template_has_to_be_specialized(self) -> None:
        with self.assertRaises(CompileError):
            class_name_template(1)

    def test_std_mem_allocator_methods_are_inherited(self) -> None:
        dyn = struct_type(DynamicAllocator)
        for name in ('alloc', 'resize', 'new', 'deinit', 'new_array', 'resize_array'):
            self.assertIsNotNone(dyn.get_method(name))


all_tests = [
    SpyClassNameCallTest,
]

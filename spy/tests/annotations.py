from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    Comptime,
    func,
    i32,
    i64,
    void,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from .generics import Pair
from .structs import Small

# ---------------------------------------------------------------------------
# annotated local variables: ``name: T = expr`` declares the type of a fresh
# variable (its slot is materialized right away, and the value is coerced to
# the declared type), and ``name: Comptime``/``name: Comptime[T]`` mark it
# compile-time - a box rather than memory, which is what lets it hold a
# compile-time-only value such as a type (and, in the new model, *only*
# compile-time values)
# ---------------------------------------------------------------------------


@func()
def annotated_local(x: i32) -> i32:
    y: i32 = x + 1
    return y * 2


@func()
def annotated_local_coerced(x: i32) -> i64:
    # the declared type governs: the value written into the slot is widened
    y: i64 = x
    return y + 1


@func()
def annotated_struct_local(x: i32) -> i32:
    p: Pair[i32] = Pair(x, 2)
    return p.total()


@func()
def annotated_plain_struct_local(x: i32) -> i32:
    s: Small = Small(x, 2)
    return s.total()


@func()
def annotated_generic_local[T](x: T) -> T:
    # the declared type is a type parameter: the call solves it
    y: T = x
    return y


@func()
def call_annotated_generic_local(x: i32) -> i32:
    return annotated_generic_local(x)


@func()
def comptime_local_holds_a_type(x: i32) -> spy_bool:
    # a compile-time variable may hold a type, a compile-time-only value
    t: Comptime = spy_typeof(x)
    return t == i32


@func()
def comptime_typed_local(x: i32) -> i32:
    # a declared compile-time type: the box only ever holds a compile-time
    # value (the runtime ``x`` is added afterwards)
    c: Comptime[i32] = 5
    return c + x


@func()
def annotated_comptime_param(x: Comptime[void]) -> spy_bool:
    # a zero-sized parameter can be marked compile-time (a runtime parameter
    # cannot be passed at compile time yet)
    return spy_typeof(x) == void


@func()
def local_holds_a_runtime_tuple(x: i32) -> i32:
    # a tuple now has a runtime representation, so a tuple of runtime values is
    # an ordinary runtime variable, unpacked through the runtime tuple place
    a = x
    b = x + 1
    t = a, b
    p, q = t
    return p * 100 + q


@func()
def passes_comptime_value(x: i32) -> spy_bool:
    # the null value is a compile-time value
    return annotated_comptime_param(None)


@func()
def passes_runtime_value_to_comptime(x: i32) -> spy_bool:
    # a runtime value would be dropped by the callee, which reads its
    # compile-time parameter as a compile-time value whatever it is given
    return annotated_comptime_param(x)  # pyright: ignore[reportArgumentType]


@func()
def bad_redeclare(x: i32) -> i32:
    y: i32 = x  # pyright: ignore[reportRedeclaration]
    y: i32 = x + 1
    return y


# ---------------------------------------------------------------------------
# ``@func(ignore_signature=True)``: the parameter and return annotations are
# read by the Python type checker only - spy infers every parameter's type from
# the call's argument and the return type from the body.  A ``Comptime`` marker
# is not a type and is kept.
# ---------------------------------------------------------------------------


@func(ignore_signature=True)
def ignored_signature_inferred(x: Any) -> Any:
    # ``Any`` is not what spy sees: ``x`` is typed by the argument (``i32``
    # here) and the body's ``+ 1`` only type-checks against that inferred type
    return x + 1


@func()
def use_ignored_signature_inferred(x: i32) -> i32:
    return ignored_signature_inferred(x)


@func(ignore_signature=True)
def ignored_signature_keeps_comptime(x: Comptime[i32], y: Any) -> Any:
    # the ``Comptime`` marker is kept even though the rest of the signature is
    # ignored, so ``x`` is still a compile-time parameter
    return x + y


@func()
def use_ignored_signature_keeps_comptime(v: i32) -> i32:
    return ignored_signature_keeps_comptime(2, v)  # ``2`` is compile-time


@func()
def pass_runtime_to_ignored_comptime(v: i32) -> i32:
    return ignored_signature_keeps_comptime(v, v)  # a runtime value: rejected


class SpyAnnotationTest(TestCase):
    """Annotated local variables: ``name: T`` declares the type of a fresh
    variable and its slot is materialized right away, while ``name:
    Comptime``/``name: Comptime[T]`` declare it compile-time - a box rather
    than memory, which is what lets it hold a compile-time-only value such as
    a type (and only compile-time values)."""

    def test_typed_local(self) -> None:
        self.assertEqual(annotated_local(3), 8)

    def test_declared_type_coerces_the_value(self) -> None:
        # the slot has the declared type, so the value written into it is
        # converted to that type
        self.assertEqual(annotated_local_coerced(3), 4)

    def test_typed_struct_locals(self) -> None:
        self.assertEqual(annotated_struct_local(3), 5)
        self.assertEqual(annotated_plain_struct_local(3), 5)

    def test_type_parameter_locals(self) -> None:
        # a local may be typed by a type parameter of the function: the
        # call solves it
        self.assertEqual(call_annotated_generic_local(7), 7)

    def test_comptime_local_holds_a_type(self) -> None:
        self.assertTrue(comptime_local_holds_a_type(1))

    def test_comptime_typed_local(self) -> None:
        self.assertEqual(comptime_typed_local(3), 8)

    def test_a_local_holds_a_runtime_tuple(self) -> None:
        self.assertEqual(local_holds_a_runtime_tuple(3), 304)

    def test_compile_time_parameter(self) -> None:
        self.assertTrue(annotated_comptime_param(None))

    def test_compile_time_parameter_from_spy_code(self) -> None:
        self.assertTrue(passes_comptime_value(3))

    def test_runtime_argument_to_a_compile_time_parameter_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            passes_runtime_value_to_comptime(3)

    def test_redeclaration_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_redeclare(3)

    def test_an_ignored_signature_is_inferred_from_the_call(self) -> None:
        # the parameter carries no spy type (``Any`` is only the checker's), so
        # its type is the argument's and the return type the body's
        self.assertEqual(use_ignored_signature_inferred(4), 5)

    def test_an_ignored_signature_keeps_a_comptime_parameter(self) -> None:
        self.assertEqual(use_ignored_signature_keeps_comptime(5), 7)

    def test_a_runtime_argument_to_an_ignored_comptime_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            pass_runtime_to_ignored_comptime(5)


all_tests = [
    SpyAnnotationTest,
]

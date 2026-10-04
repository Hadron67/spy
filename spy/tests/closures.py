from typing import Protocol
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    syntax,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, func, func_type, struct
from ..compiler.syntax import as_func_ptr

# ---------------------------------------------------------------------------
# closures: a nested ``def`` (or ``lambda``) inside a spy function captures
# the variables of the enclosing function.  A closure only exists at compile
# time - it can be called inside the function (inlined, or compiled into a
# runtime function whose captures are hidden pointer parameters) and passed to
# an inline function, but never handed to a runtime function.
# ---------------------------------------------------------------------------


@func()
def closure_capture_param(n: i32) -> i32:
    # the default closure is inlined; ``n`` is captured by reference
    def add(x: i32) -> i32:
        return x + n
    return add(3)


def apply_inline(f, x: i32) -> i32:
    # a plain Python function: inlined at its call site, so it may take a
    # closure as a parameter
    return f(x)


@func()
def closure_passed_to_inline(a: i32) -> i32:
    def add(x: i32) -> i32:
        return x + a
    return apply_inline(add, 10)


@func()
def closure_lambda_arg(a: i32) -> i32:
    # a lambda is a forced-inline closure and may be written directly as an
    # argument
    return apply_inline(lambda x: x + a, 10)


@func()
def closure_forward_declaration() -> i32:
    # the closure is written before the variable it captures is assigned: the
    # body's pre-scan declares ``x`` up front
    def f() -> i32:
        return x
    x: i32 = 7
    return f()


@func()
def closure_nested(a: i32) -> i32:
    def outer() -> i32:
        def inner() -> i32:
            return a + 1
        return inner() + 1
    return outer()


@struct()
class Box:
    v: i32


@func()
def closure_mutates_captured_field(a: i32) -> i32:
    # a captured struct is written back through the pointer the closure holds
    b = Box(a)
    def bump() -> None:
        b.v += 1
    bump()
    bump()
    return b.v


@func()
def closure_compiled(n: i32) -> i32:
    # a non-inlined closure is compiled into a runtime function that takes the
    # address of ``n`` as a hidden parameter
    @syntax.closure(inline=False)
    def add(x: i32) -> i32:
        return x + n
    return add(20)


@func()
def closure_compiled_calls(n: i32) -> i32:
    @syntax.closure(inline=False)
    def double(x: i32) -> i32:
        return twice(x)
    return double(n)


@func()
def twice(x: i32) -> i32:
    return x * 2


@func()
def closure_comptime() -> i32:
    # an inlined closure captures a compile-time variable directly
    n: syntax.Comptime = 5
    def f() -> i32:
        return n + 1
    return f()


@func()
def closure_compiled_comptime() -> i32:
    # a compile-time capture is used when the closure is compiled and does not
    # reach the runtime signature
    n: syntax.Comptime = 5
    @syntax.closure(inline=False)
    def f() -> i32:
        return n + 1
    return f()


@struct()
class ClosureError(Exception):
    code: i32


@func(exceptions=ClosureError)
def raise_closure_error(n: i32) -> i32:
    if n < 0:
        raise ClosureError(1)
    return n


@func()
def closure_compiled_raises(n: i32) -> i32:
    # a compiled closure declares the exceptions it may raise; the error it
    # propagates is dispatched to the caller's ``try``
    @syntax.closure(inline=False, exceptions=ClosureError)
    def f(x: i32) -> i32:
        return raise_closure_error(x)
    try:
        return f(n)
    except ClosureError:
        return -1


@func()
def plus_one(x: i32) -> i32:
    return x + 1

@func_type()
class PlusOneFn(Protocol):
    def __call__(self, x: i32) -> i32: ...

@func()
def closure_as_func_ptr() -> i32:
    # a capture-free, non-inlined closure is an ordinary runtime function
    @syntax.closure(inline=False)
    def inc(x: i32) -> i32:
        return x + 1
    p = as_func_ptr(PlusOneFn, inc)
    return p[...](5)


@func()
def closure_with_generic_params[T](x: T) -> T:
    # a closure may declare its own type parameters
    def identity(y: T) -> T:
        return y
    return identity(x)


@func()
def closure_in_a_loop(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        captured = i
        def add(x: i32) -> i32:
            return x + captured  # noqa: B023
        total += add(1)
    return total


@func()
def closure_naming(n: i32) -> i32:
    # a compiled closure's native name is prefixed with the function it is
    # created in (the enclosing specialization, its signature included)
    @syntax.closure(inline=False)
    def mid(x: i32) -> i32:
        @syntax.closure(inline=False)
        def inner(y: i32) -> i32:
            return y + 1
        return inner(x) + n
    return mid(2)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

@func()
def closure_capturing_as_func_ptr(n: i32) -> i32:
    @syntax.closure(inline=False)
    def f(x: i32) -> i32:
        return x + n
    p = as_func_ptr(PlusOneFn, f)
    return p[...](1)


@func()
def closure_inlined_as_func_ptr() -> i32:
    def inc(x: i32) -> i32:
        return x + 1
    p = as_func_ptr(PlusOneFn, inc)
    return p[...](1)


@func()
def closure_to_runtime_fn(n: i32) -> i32:
    # a closure may not be handed to a non-inline (runtime) function
    def add(x: i32) -> i32:
        return x + n
    return twice(add)  # pyright: ignore


@func()
def closure_enclosing_type_param[T](x: T) -> T:
    def f() -> i32:
        if spy_typeof(x) == T:
            return 1
        return 0
    return x


@func()
def closure_rejected_decorator() -> i32:
    @func()
    def f() -> i32:
        return 1
    return f()


class SpyClosureTest(TestCase):
    """Nested ``def``/``lambda`` closures: capture by reference, inline and
    compiled forms, capture-free function pointers, and the rejected cases."""

    def test_capture_a_parameter(self) -> None:
        self.assertEqual(closure_capture_param(10), 13)

    def test_pass_a_closure_to_an_inline_function(self) -> None:
        self.assertEqual(closure_passed_to_inline(5), 15)

    def test_lambda_as_an_argument(self) -> None:
        self.assertEqual(closure_lambda_arg(5), 15)

    def test_forward_declaration(self) -> None:
        self.assertEqual(closure_forward_declaration(), 7)

    def test_nested_closures(self) -> None:
        self.assertEqual(closure_nested(10), 12)

    def test_mutate_a_captured_field(self) -> None:
        self.assertEqual(closure_mutates_captured_field(1), 3)

    def test_compiled_closure(self) -> None:
        self.assertEqual(closure_compiled(10), 30)
        # the specialization is shared; each creation passes its own captures
        self.assertEqual(closure_compiled(1), 21)

    def test_compiled_closure_calls_a_function(self) -> None:
        self.assertEqual(closure_compiled_calls(21), 42)

    def test_compiled_closure_with_a_comptime_capture(self) -> None:
        self.assertEqual(closure_compiled_comptime(), 6)

    def test_inlined_closure_with_a_comptime_capture(self) -> None:
        self.assertEqual(closure_comptime(), 6)

    def test_compiled_closure_declares_its_exceptions(self) -> None:
        self.assertEqual(closure_compiled_raises(5), 5)
        self.assertEqual(closure_compiled_raises(-5), -1)

    def test_capture_free_closure_as_a_function_pointer(self) -> None:
        self.assertEqual(closure_as_func_ptr(), 6)

    def test_a_closure_with_its_own_generic_parameters(self) -> None:
        self.assertEqual(closure_with_generic_params(4), 4)

    def test_a_closure_in_a_loop(self) -> None:
        self.assertEqual(closure_in_a_loop(3), 6)

    def test_a_compiled_closures_name_is_qualified(self) -> None:
        self.assertEqual(closure_naming(10), 13)
        names = list(_GLOBAL_CONTEXT._symbol_table._symbols.keys())
        nested = [n for n in names if 'mid' in n and 'inner' in n]
        self.assertTrue(
            any('closure_naming' in n and n.index('mid') < n.index('inner') for n in nested),
            names,
        )

    def test_capturing_closure_cannot_become_a_function_pointer(self) -> None:
        with self.assertRaises(CompileError):
            closure_capturing_as_func_ptr(1)

    def test_an_inlined_closure_cannot_become_a_function_pointer(self) -> None:
        with self.assertRaises(CompileError):
            closure_inlined_as_func_ptr()

    def test_an_enclosing_type_parameter_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            closure_enclosing_type_param(1)

    def test_a_closure_cannot_be_passed_to_a_runtime_function(self) -> None:
        with self.assertRaises(CompileError):
            closure_to_runtime_fn(1)

    def test_a_non_closure_decorator_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            closure_rejected_decorator()


all_tests = [
    SpyClosureTest,
]

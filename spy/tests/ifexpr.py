from unittest import TestCase
from ..compiler import (
    CompileError,
    SpyError,
    TypeMismatchError,
    compile_log,
    f32,
    f64,
    i8,
    i32,
    i64,
    mir,
    sval,
    syntax,
    u0,
    u64,
    usize,
    void,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, _Context, decl_func, func, func_type, struct
from ..compiler.syntax import (
    Array,
    Comptime,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
    Opaque,
    Option,
    Ptr,
    array,
    as_func_ptr,
    defer,
    errdefer,
    okdefer,
    ptr_cast,
    ref,
)

from .basics import inc, le
from .structs import Large, Small


# ---------------------------------------------------------------------------
# if-expressions: ``a if c else b`` writes the branch it takes into the
# result location of the expression (a runtime branch of the MIR, or nothing
# at all when the condition is a compile-time value)
# ---------------------------------------------------------------------------


@func()
def choose_value(c: spy_bool, a: i32, b: i32) -> i32:
    return a if c else b


@func()
def choose_local(c: spy_bool, a: i32, b: i32) -> i32:
    x = a if c else b
    return x


@func()
def choose_nested(c1: spy_bool, c2: spy_bool, a: i32, b: i32, d: i32) -> i32:
    return a if c1 else (b if c2 else d)


@func()
def choose_comparison(a: i32, b: i32) -> i32:
    # a comparison as the condition
    return a if a > 0 else b


@func()
def choose_argument(c: spy_bool, a: i32, b: i32) -> i32:
    return inc(a if c else b)


@func()
def choose_arithmetic(c: spy_bool, a: i32, b: i32) -> i32:
    return (a if c else b) + 10


@func()
def choose_call_test(a: i32, b: i32, x: i32) -> i32:
    # the condition is a call of another spy function
    return a if le(x, 0) else b


@func()
def choose_comptime(a: i32, b: i32, c: spy_bool) -> i32:
    # a compile-time condition folds the expression: the dead else branch (a
    # nested if-expression) is never typed
    return a if spy_typeof(a) == i32 else (b if c else a)


@func()
def choose_comptime_else(a: i32, b: i32, c: spy_bool) -> i32:
    # ... and the branch it chooses may be the else one
    return a if spy_typeof(a) == i64 else (b if c else a)


def pick_inline(c: bool, a: i32, b: i32) -> i32:
    # an undecorated plain function: its body (with the if-expression) is
    # inlined at its call sites
    return a if c else b


@func()
def call_pick_inline(c: spy_bool, a: i32, b: i32) -> i32:
    return pick_inline(c, a, b)


@func()
def choose_struct(c: spy_bool, a: i32, b: i32) -> i32:
    # both branches construct a struct in place, in the same slot
    s = Small(a, b) if c else Small(b, b)
    return s.total()


@func()
def choose_large(c: spy_bool, x: i64) -> Large:
    # a large struct is returned through a result pointer: the branches write
    # into the caller's result location
    return Large(x, 1, 2, 3) if c else Large(x, 4, 5, 6)


@func()
def use_choose_large(c: spy_bool, x: i64) -> i64:
    return choose_large(c, x).a + choose_large(c, x).d


@func()
def choose_pointer(c: spy_bool, a: i32, b: i32) -> i32:
    p = ref(a) if c else ref(b)
    return p[...]


@func()
def choose_deref_write(c: spy_bool, a: i32, b: i32) -> i32:
    v = a
    p = ref(v)
    p[...] = b if c else a
    return v


@func()
def choose_non_bool_condition(a: i32, b: i32) -> i32:
    # spy has no truthiness: the condition has to be a bool
    return a if a else b


class SpyIfExprTest(TestCase):
    """If-expressions: ``a if c else b`` evaluates one of its branches into
    the result location of the expression - a runtime branch in the MIR, or
    just the chosen branch when the condition is a compile-time value."""

    def test_value(self) -> None:
        self.assertEqual(choose_value(True, 3, 4), 3)
        self.assertEqual(choose_value(False, 3, 4), 4)

    def test_into_a_fresh_local(self) -> None:
        # the expression is evaluated into the slot of the local it declares
        self.assertEqual(choose_local(True, 3, 4), 3)
        self.assertEqual(choose_local(False, 3, 4), 4)

    def test_nested(self) -> None:
        self.assertEqual(choose_nested(False, False, 1, 2, 3), 3)
        self.assertEqual(choose_nested(False, True, 1, 2, 3), 2)
        self.assertEqual(choose_nested(True, False, 1, 2, 3), 1)

    def test_condition(self) -> None:
        # a comparison, a call of another spy function, ...
        self.assertEqual(choose_comparison(3, 4), 3)
        self.assertEqual(choose_comparison(-3, 4), 4)
        self.assertEqual(choose_call_test(3, 4, -1), 3)
        self.assertEqual(choose_call_test(3, 4, 1), 4)

    def test_operand_and_argument(self) -> None:
        # the value of the expression feeds an operator and a call argument
        self.assertEqual(choose_arithmetic(True, 3, 4), 13)
        self.assertEqual(choose_arithmetic(False, 3, 4), 14)
        self.assertEqual(choose_argument(True, 3, 4), 4)
        self.assertEqual(choose_argument(False, 3, 4), 5)

    def test_comptime_condition(self) -> None:
        # only the chosen branch is typed, so the dead one may contain another
        # if-expression (and disagree with the result type)
        self.assertEqual(choose_comptime(3, 4, True), 3)
        self.assertEqual(choose_comptime(3, 4, False), 3)
        self.assertEqual(choose_comptime_else(3, 4, True), 4)
        self.assertEqual(choose_comptime_else(3, 4, False), 3)

    def test_inlined_body(self) -> None:
        self.assertEqual(call_pick_inline(True, 3, 4), 3)
        self.assertEqual(call_pick_inline(False, 3, 4), 4)

    def test_struct_branches(self) -> None:
        # each branch constructs its struct in place, in the same slot
        self.assertEqual(choose_struct(True, 1, 2), 3)
        self.assertEqual(choose_struct(False, 1, 2), 4)

    def test_result_pointer_branches(self) -> None:
        # a large struct is delivered through the result pointer of the
        # function: the branches write into the caller's location
        self.assertEqual(use_choose_large(True, 5), 8)
        self.assertEqual(use_choose_large(False, 5), 11)

    def test_with_pointers(self) -> None:
        self.assertEqual(choose_pointer(True, 3, 4), 3)
        self.assertEqual(choose_pointer(False, 3, 4), 4)
        self.assertEqual(choose_deref_write(True, 3, 4), 4)
        self.assertEqual(choose_deref_write(False, 3, 4), 3)

    def test_non_bool_condition(self) -> None:
        with self.assertRaises(CompileError):
            choose_non_bool_condition(1, 2)


all_tests = [
    SpyIfExprTest,
]

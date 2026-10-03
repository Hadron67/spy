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
from ..std import ConstSlicePtr, Numeric, SlicePtr, arr_slice, const_arr_slice, undefined

from .reflect import ReflectA, ReflectB


# ---------------------------------------------------------------------------
# the undefined literal: ``std.core.undefined`` is a value of any type, leaving
# its destination undefined (see ``sval.UndefinedType``/``sval.UntypedUndefined``)
# ---------------------------------------------------------------------------


def identity_i32(x: i32) -> i32:
    return x


@func()
def undef_scalar(n: i32) -> i32:
    x: i32 = undefined()
    spy_typeof(x)
    return n


@func()
def undef_pointer(n: i32) -> i32:
    p: Ptr[i32] = undefined()
    spy_typeof(p)
    return n


@func()
def undef_struct(n: i32) -> i32:
    v: ReflectA = undefined()
    spy_typeof(v)
    return n


@func()
def undef_option(n: i32) -> i32:
    o: Option[i32] = undefined()
    spy_typeof(o)
    return n


@func()
def undef_union(n: i32) -> i32:
    u: ReflectA | ReflectB = undefined()
    spy_typeof(u)
    return n


@func()
def undef_argument(n: i32) -> i32:
    # the literal converts to the parameter's type
    identity_i32(undefined())
    return n


@func()
def undef_inferred(n: i32, c: i32) -> i32:
    # the slot takes the peer type of the value and the literal
    x = n if c > 0 else undefined()
    if c == 12345:
        return x
    return n


@func()
def undef_comptime() -> i32:
    x: Comptime = undefined()
    spy_typeof(x)
    return 1


@func()
def undef_comptime_option() -> i32:
    o: Comptime[Option[i32]] = undefined()
    spy_typeof(o)
    return 2


@func()
def undef_comptime_union() -> i32:
    u: Comptime[ReflectA | ReflectB] = undefined()
    spy_typeof(u)
    return 3


@func()
def undef_return(n: i32) -> i32:
    # the literal converts at the function's result location
    if n > 0:
        return undefined()
    return n


@func()
def call_undef_return(n: i32) -> i32:
    undef_return(n)
    return n


@func()
def undef_if() -> i32:
    if undefined():
        return 1
    return 0


@func()
def undef_comptime_if() -> i32:
    x: Comptime[spy_bool] = undefined()
    if x:
        return 1
    return 0


@func()
def undef_while() -> i32:
    while undefined():
        return 1
    return 0


@func()
def undef_and_condition(a: i32) -> i32:
    if a > 0 and undefined():
        return 1
    return 0


class SpyUndefinedTest(TestCase):
    """The ``std.core.undefined`` literal: a value of any type."""

    def test_the_literal_converts_to_a_scalar(self) -> None:
        self.assertEqual(undef_scalar(7), 7)

    def test_the_literal_converts_to_a_pointer(self) -> None:
        self.assertEqual(undef_pointer(7), 7)

    def test_the_literal_converts_to_a_struct(self) -> None:
        self.assertEqual(undef_struct(7), 7)

    def test_the_literal_converts_to_an_option(self) -> None:
        self.assertEqual(undef_option(7), 7)

    def test_the_literal_converts_to_a_tagged_union(self) -> None:
        self.assertEqual(undef_union(7), 7)

    def test_the_literal_converts_to_an_argument_type(self) -> None:
        self.assertEqual(undef_argument(7), 7)

    def test_the_literal_peers_with_a_value(self) -> None:
        self.assertEqual(undef_inferred(7, 1), 7)
        self.assertEqual(undef_inferred(7, -1), 7)

    def test_the_literal_is_a_compile_time_value(self) -> None:
        self.assertEqual(undef_comptime(), 1)

    def test_a_compile_time_option_is_left_undefined(self) -> None:
        self.assertEqual(undef_comptime_option(), 2)

    def test_a_compile_time_union_is_left_undefined(self) -> None:
        self.assertEqual(undef_comptime_union(), 3)

    def test_the_literal_converts_at_a_return_location(self) -> None:
        self.assertEqual(call_undef_return(5), 5)
        self.assertEqual(call_undef_return(-5), -5)

    def test_an_undefined_if_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_if()

    def test_a_compile_time_undefined_bool_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_comptime_if()

    def test_an_undefined_while_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_while()

    def test_an_undefined_and_operand_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_and_condition(1)


all_tests = [
    SpyUndefinedTest,
]

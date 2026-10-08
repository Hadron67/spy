from typing import Literal, Protocol
from unittest import TestCase

from ..compiler import (
    CompileError,
    ConstMultiPtr,
    f32,
    i8,
    i32,
    mir,
    sval,
    u8,
)
from ..compiler import as_ as spy_as
from ..compiler.dsl import _GLOBAL_CONTEXT, decl_func, func, func_type
from ..compiler.syntax import closure, ref
from ..std import Array
from ..std.c import snprintf
from ..std.core import const_arr_ptr, gstr, undefined
from .structs import MIR_CACHE

# ---------------------------------------------------------------------------
# C-style variadic extern functions: a ``@decl_func`` (and a
# ``@func_type(callconv='c')``) may declare a trailing, unannotated ``*args``,
# which stands for the C ``...``.  A call passes any number of extra arguments
# after the declared ones, each by value with its own type - promoted by the C
# *default argument promotions* (see ``fn._c_default_promotion``), the type the
# callee reads it back as with ``va_arg``.  ``snprintf`` (declared in
# ``std.c``) is the canonical one: its extra arguments are heterogeneous.
# ---------------------------------------------------------------------------


@decl_func()
def c_variadic(fmt: ConstMultiPtr[u8], *args) -> i32:
    # a C-variadic declaration: the trailing ``*args`` is the ``...``
    ...


@decl_func()
def c_variadic_annotated(*args: i32) -> i32:
    # the ``*args`` of a C declaration may not be annotated: every argument the
    # call passes keeps its own type
    ...


@decl_func()
def c_variadic_kwargs(**kwargs: i32) -> i32:
    # C has no keyword arguments
    ...


@func_type()
class SpyVariadic(Protocol):
    # a *default*-callconv function type may not declare ``*args``
    def __call__(self, fmt: ConstMultiPtr[u8], *args) -> i32:
        ...


@func(callconv='c')
def c_variadic_definition(x: i32, *args: i32) -> i32:
    # defining a C-variadic function (reading the extra arguments with va_arg)
    # is not implemented yet: only the declaration is
    return x


@func()
def c_variadic_closure(x: i32) -> i32:
    # a C-callconv *closure* is a definition too: it may not be variadic yet
    @closure(callconv='c')
    def inner(y: i32, *args: i32) -> i32:
        return y
    return inner(x)


# --- calling a real C variadic function (``snprintf``) ---------------------


@func()
def snprintf_len(n: i32) -> i32:
    # ``snprintf`` writes into a stack-allocated buffer and returns the number
    # of characters the format produced (they fit in 10, so it is what is
    # written); ``const_arr_ptr`` turns ``&buf`` into the ``MultiPtr[u8]`` the C
    # signature asks for, and ``len(buf)`` is the array's compile-time length
    buf: Array[u8, Literal[10]] = undefined()
    r = snprintf(const_arr_ptr(ref(buf)), len(buf), gstr(b'%d'), n)
    return r


@func()
def snprintf_first_digit(n: i32) -> i32:
    # the formatted text really lands in the buffer: the first byte of ``n``
    buf: Array[u8, Literal[10]] = undefined()
    snprintf(const_arr_ptr(ref(buf)), len(buf), gstr(b'%d'), n)
    return buf[0] - 48


@func()
def snprintf_no_vararg() -> i32:
    # a call that passes no extra argument: the ``...`` takes zero
    buf: Array[u8, Literal[10]] = undefined()
    r = snprintf(const_arr_ptr(ref(buf)), len(buf), gstr(b'hi'))
    return r


@func()
def snprintf_narrow(n: i8) -> i32:
    # an ``i8`` is a byte: C promotes it to ``int`` (the callee reads an int)
    buf: Array[u8, Literal[10]] = undefined()
    r = snprintf(const_arr_ptr(ref(buf)), len(buf), gstr(b'%d'), n)
    return r


@func()
def snprintf_float(x: f32) -> i32:
    # an ``f32`` is promoted to ``double`` (the callee reads a double)
    buf: Array[u8, Literal[10]] = undefined()
    r = snprintf(const_arr_ptr(ref(buf)), len(buf), gstr(b'%.1f'), x)
    return r


class SpyCVariadicTypeTest(TestCase):
    """A C-variadic extern function is declared by a trailing, unannotated
    ``*args``: the resulting ``sval.FunctionType`` (and its MIR mirror) carry
    the variadic flag, and annotating it, ``**kwargs``, a default callconv or a
    C-variadic *definition* are all rejected."""

    def test_a_c_declaration_is_variadic(self) -> None:
        decl = _GLOBAL_CONTEXT.resolve_global(c_variadic)
        assert isinstance(decl, sval.DeclareFunction)
        self.assertTrue(decl.type.varargs)
        self.assertEqual(decl.type.callconv, 'c')
        self.assertIn('...', str(decl.type))
        mirror = decl.type.to_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.FunctionType)
        self.assertTrue(mirror.varargs)
        self.assertEqual(len(mirror.args), 1)

    def test_an_annotated_varargs_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(c_variadic_annotated)
        self.assertIn('may not be annotated', str(ctx.exception))

    def test_kwargs_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(c_variadic_kwargs)
        self.assertIn('**kwargs', str(ctx.exception))

    def test_a_default_callconv_varargs_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(SpyVariadic)
        self.assertIn('may not declare *args', str(ctx.exception))

    def test_defining_a_c_variadic_function_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_variadic_definition(1, 2)
        self.assertIn('may not declare *args', str(ctx.exception))

    def test_defining_a_c_variadic_closure_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_variadic_closure(1)
        self.assertIn('may not declare *args', str(ctx.exception))


class SpyCVariadicCallTest(TestCase):
    """A call of a declared C-variadic function passes the excess arguments by
    value, with the C default argument promotions applied."""

    def test_a_variadic_call(self) -> None:
        self.assertEqual(snprintf_len(0), 1)
        self.assertEqual(snprintf_len(42), 2)
        self.assertEqual(snprintf_len(-1234), 5)

    def test_the_formatted_text_lands_in_the_buffer(self) -> None:
        self.assertEqual(snprintf_first_digit(7), 7)
        self.assertEqual(snprintf_first_digit(42), 4)

    def test_a_narrow_integer_is_promoted_to_int(self) -> None:
        self.assertEqual(snprintf_narrow(spy_as(-5, i8)), 2)
        self.assertEqual(snprintf_narrow(spy_as(100, i8)), 3)

    def test_a_float_is_promoted_to_double(self) -> None:
        self.assertEqual(snprintf_float(spy_as(1.5, f32)), 3)

    def test_a_call_with_no_extra_argument(self) -> None:
        self.assertEqual(snprintf_no_vararg(), 2)


all_tests = [
    SpyCVariadicTypeTest,
    SpyCVariadicCallTest,
]

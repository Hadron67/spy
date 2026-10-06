from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    syntax,
)
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
)

# ---------------------------------------------------------------------------
# ``*args``/``**kwargs``: a spy function may declare them, and a call's excess
# positional/keyword arguments are bound to them.  The body sees ``args`` as a
# tuple of its element places and ``kwargs`` as a dictionary of its value
# places (``interp.ComptimeTuplePtr``/``ComptimeDictPtr``), read with ``t[i]``
# (a compile-time integer index) and ``d[key]`` (a byte-string key).
# ---------------------------------------------------------------------------


@func()
def sum_args(*args: i32) -> i32:
    # a compile-time loop over the varargs: ``len(args)`` is a compile-time
    # integer and ``args[i]`` the i-th element place
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < len(args):
        total = total + args[i]
        i = i + 1
    return total


@func()
def first_arg(*args) -> i32:
    # an unannotated ``*args``: the element type is inferred from the call
    return args[0]


@func()
def take_kwargs(**kwargs: i32) -> i32:
    # the keyword arguments are read by name, in any order
    return kwargs['a'] * 100 + kwargs['b']


@func()
def args_and_kwargs(*args: i32, **kwargs: i32) -> i32:
    return args[0] + kwargs['x']


def sum_inline(*args: i32) -> i32:
    # an undecorated function is inlined at its call site; it declares
    # ``*args`` like any other
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < len(args):
        total = total + args[i]
        i = i + 1
    return total


@func()
def call_inline(a: i32) -> i32:
    return sum_inline(a, a + 1, a + 2)


@func()
def call_closure(a: i32) -> i32:
    def add(*args: i32) -> i32:
        i: Comptime = 0
        total: i32 = 0
        syntax.unroll()
        while i < len(args):
            total = total + args[i]
            i = i + 1
        return total

    return add(a, a + 1, a + 2)


@func()
def bad_vararg_type(*args: type[i32]) -> i32:  # pyright: ignore
    return 0


class SpyVarargsTest(TestCase):
    """``*args``/``**kwargs`` parameters: a call's excess arguments bind to
    them, and the body reads them as a tuple/dictionary of places."""

    def test_varargs(self) -> None:
        self.assertEqual(sum_args(1, 2, 3), 6)
        self.assertEqual(sum_args(7), 7)
        self.assertEqual(sum_args(), 0)

    def test_unannotated_varargs(self) -> None:
        self.assertEqual(first_arg(7), 7)

    def test_kwargs(self) -> None:
        self.assertEqual(take_kwargs(a=1, b=2), 102)
        self.assertEqual(take_kwargs(b=5, a=3), 305)

    def test_args_and_kwargs(self) -> None:
        self.assertEqual(args_and_kwargs(1, x=2), 3)

    def test_an_inlined_function(self) -> None:
        self.assertEqual(call_inline(3), 3 + 4 + 5)

    def test_a_closure(self) -> None:
        self.assertEqual(call_closure(3), 3 + 4 + 5)

    def test_a_type_valued_varargs_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_vararg_type()


all_tests = [
    SpyVarargsTest,
]

from unittest import TestCase

from ..compiler import (
    f64,
    i32,
    syntax,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
)
from ..std.int import int_to_float

# ---------------------------------------------------------------------------
# every combination of an ordinary positional parameter (``p``), a
# default-valued positional parameter (``d``), ``*args`` (``v``) and
# ``**kwargs`` (``k``).
#
# The varargs/kwargs are deliberately unannotated, so their element/value types
# are inferred from the call, and the calls pass *mixed* ``i32``/``f64``
# arguments.  Spy has no implicit ``i32``/``f64`` promotion in arithmetic, and a
# store no longer converts an integer to a float either, so a mixed element is
# summed through a ``f64`` typed temporary explicitly: an annotated ``i32``
# goes through ``int_to_float``, an inferred one through ``to_f64`` (which picks
# ``int_to_float`` or a no-op by the compile-time ``typeof``).  Every function
# returns the ``f64`` sum of every argument it was handed, which is what the
# calls below assert.
# ---------------------------------------------------------------------------


@func()
def to_f64[T](x: T) -> f64:  # pyright: ignore[reportInvalidTypeVarUse]
    """Convert an inferred numeric argument to ``f64``: ``int_to_float`` for an
    integer, the value itself for a float (the branch is resolved at compile
    time from ``typeof(x)``)."""
    if spy_typeof(x) == f64:
        return x  # pyright: ignore[reportReturnType]
    return int_to_float(x, f64)


@func()
def c_none() -> f64:
    total: f64 = 0.0
    return total


@func()
def c_p(p: i32) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    total = total + tp
    return total


@func()
def c_d(d: i32 = 7) -> f64:
    total: f64 = 0.0
    td: f64 = int_to_float(d, f64)
    total = total + td
    return total


@func()
def c_v(*args) -> f64:
    total: f64 = 0.0
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    return total


@func()
def c_k(**kwargs) -> f64:
    total: f64 = 0.0
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_pd(p: i32, d: i32 = 7) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    td: f64 = int_to_float(d, f64)
    total = total + tp
    total = total + td
    return total


@func()
def c_pv(p: i32, *args) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    total = total + tp
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    return total


@func()
def c_pk(p: i32, **kwargs) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    total = total + tp
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_dv(d: i32 = 7, *args) -> f64:
    total: f64 = 0.0
    td: f64 = int_to_float(d, f64)
    total = total + td
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    return total


@func()
def c_dk(d: i32 = 7, **kwargs) -> f64:
    total: f64 = 0.0
    td: f64 = int_to_float(d, f64)
    total = total + td
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_vk(*args, **kwargs) -> f64:
    total: f64 = 0.0
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_pdv(p: i32, d: i32 = 7, *args) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    td: f64 = int_to_float(d, f64)
    total = total + tp
    total = total + td
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    return total


@func()
def c_pdk(p: i32, d: i32 = 7, **kwargs) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    td: f64 = int_to_float(d, f64)
    total = total + tp
    total = total + td
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_pvk(p: i32, *args, **kwargs) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    total = total + tp
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_dvk(d: i32 = 7, *args, **kwargs) -> f64:
    total: f64 = 0.0
    td: f64 = int_to_float(d, f64)
    total = total + td
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


@func()
def c_pdvk(p: i32, d: i32 = 7, *args, **kwargs) -> f64:
    total: f64 = 0.0
    tp: f64 = int_to_float(p, f64)
    td: f64 = int_to_float(d, f64)
    total = total + tp
    total = total + td
    i: Comptime = 0
    syntax.unroll()
    while i < len(args):
        elem: f64 = to_f64(args[i])
        total = total + elem
        i = i + 1
    tx: f64 = to_f64(kwargs['x'])
    ty: f64 = to_f64(kwargs['y'])
    total = total + tx
    total = total + ty
    return total


class SpyVarargsCombinationTest(TestCase):
    """Every combination of an ordinary positional parameter, a default-valued
    one, ``*args`` and ``**kwargs``.  The varargs/kwargs are unannotated, so the
    calls pass mixed ``i32``/``f64`` values and the function sums whatever it
    was handed (converting each element to ``f64`` explicitly), which pins down
    exactly which arguments bound to which parameter."""

    def test_none(self) -> None:
        self.assertEqual(c_none(), 0.0)

    def test_positional(self) -> None:
        self.assertEqual(c_p(1), 1.0)

    def test_default(self) -> None:
        self.assertEqual(c_d(), 7.0)
        self.assertEqual(c_d(4), 4.0)

    def test_varargs(self) -> None:
        self.assertEqual(c_v(2.5, 3), 5.5)
        self.assertEqual(c_v(3, 2.5), 5.5)
        self.assertEqual(c_v(), 0.0)

    def test_kwargs(self) -> None:
        self.assertEqual(c_k(x=5, y=6.5), 11.5)
        self.assertEqual(c_k(x=6.5, y=5), 11.5)

    def test_positional_and_default(self) -> None:
        self.assertEqual(c_pd(1), 8.0)
        self.assertEqual(c_pd(1, 4), 5.0)

    def test_positional_and_varargs(self) -> None:
        self.assertEqual(c_pv(1, 2.5, 3), 6.5)
        self.assertEqual(c_pv(1), 1.0)

    def test_positional_and_kwargs(self) -> None:
        self.assertEqual(c_pk(1, x=5, y=6.5), 12.5)

    def test_default_and_varargs(self) -> None:
        self.assertEqual(c_dv(4, 2.5, 3), 9.5)
        self.assertEqual(c_dv(), 7.0)

    def test_default_and_kwargs(self) -> None:
        self.assertEqual(c_dk(x=5, y=6.5), 18.5)
        self.assertEqual(c_dk(4, x=5, y=6.5), 15.5)

    def test_varargs_and_kwargs(self) -> None:
        self.assertEqual(c_vk(2.5, 3, x=5, y=6.5), 17.0)

    def test_positional_default_and_varargs(self) -> None:
        self.assertEqual(c_pdv(1, 4, 2.5, 3), 10.5)
        self.assertEqual(c_pdv(1), 8.0)

    def test_positional_default_and_kwargs(self) -> None:
        self.assertEqual(c_pdk(1, x=5, y=6.5), 19.5)
        self.assertEqual(c_pdk(1, 4, x=5, y=6.5), 16.5)

    def test_positional_varargs_and_kwargs(self) -> None:
        self.assertEqual(c_pvk(1, 2.5, 3, x=5, y=6.5), 18.0)
        self.assertEqual(c_pvk(1, x=5, y=6.5), 12.5)

    def test_default_varargs_and_kwargs(self) -> None:
        self.assertEqual(c_dvk(4, 2.5, 3, x=5, y=6.5), 21.0)
        self.assertEqual(c_dvk(x=5, y=6.5), 18.5)

    def test_all(self) -> None:
        self.assertEqual(c_pdvk(1, 4, 2.5, 3, x=5, y=6.5), 22.0)
        self.assertEqual(c_pdvk(1, x=5, y=6.5), 19.5)


all_tests = [
    SpyVarargsCombinationTest,
]

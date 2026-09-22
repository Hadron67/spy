"""Integration tests for the spy JIT (``spy``).

The functions under test are defined at module level and registered with
the ordinary ``@spy.func()`` decorator; a function body may call the
other registered functions by name (they are module globals, resolved by
the compile-time interpreter) exactly like a user would.  The
undecorated ``add_inline`` is deliberately left unregistered: it stays a
plain Python function and is inlined at its call sites.

Function calls are exercised by ``SpyFunctionCallTest`` below; the struct
features (declaration, layout, construction and methods) by
``SpyStructTest``/``SpyStructMirrorTest``, and generic structs by
``SpyGenericStructTest``.
The global host context caches specializations, so the tests share the
compiled functions; a test that needs a fresh compilation calls a
function no earlier test has compiled.
"""

import io
from contextlib import redirect_stdout
from typing import Any, Literal, Protocol, Self
from unittest import TestCase

from spy.dsl import func, struct

from . import (
    CompileError,
    compile_log,
    f32,
    f64,
    i8,
    i32,
    i64,
    mir,
    sval,
    syntax,
    u64,
    void,
)
from . import as_ as spy_as
from . import bool as spy_bool
from . import typeof as spy_typeof
from .syntax import Ptr, ref

# ---------------------------------------------------------------------------
# functions under test
# ---------------------------------------------------------------------------

class Numeric(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mod__(self, other: Self, /) -> Self: ...
    def __lt__(self, other: Self, /) -> spy_bool: ...
    def __gt__(self, other: Self, /) -> spy_bool: ...
    def __le__(self, other: Self, /) -> spy_bool: ...
    def __ge__(self, other: Self, /) -> spy_bool: ...


@func()
def smoke_test(a: i32, b: i32) -> i32:
    return a + b


@func()
def add[T: Numeric](a: T, b: T) -> T:
    return a + b


@func()
def add_u64(a: u64, b: u64) -> u64:
    return a + b


@func()
def sub(a: i32, b: i32) -> i32:
    return a - b


@func()
def mul(a: i32, b: i32) -> i32:
    return a * b


@func()
def mod(a: i32, b: i32) -> i32:
    return a % b


@func()
def scale(a: f64, b: f64) -> f64:
    return a * b + 1.0


@func()
def add_default[T: Numeric](a: T, b: T = 0) -> T:
    return a + b


def add_inline[T: Numeric](a: T, b: T) -> T:
    compile_log("add_inline was compiled")
    return a + b


def abs_inline(n: i32) -> i32:
    if n < 0:
        return -n
    else:
        return n


def clamp_inline(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def call_abs(n: i32) -> i32:
    return abs_inline(n)


@func()
def call_clamp(n: i32) -> i32:
    return clamp_inline(n)


@func()
def call_abs_plus_one(n: i32) -> i32:
    return abs_inline(n) + 1


@func()
def call_add[T: Numeric](a: T, b: T) -> T:
    return add(a, b)


@func()
def call_inline[T: Numeric](a: T, b: T) -> T:
    return add_inline(a, b)


@func()
def call_inline_log(a: i32, b: i32) -> i32:
    return add_inline(a, b)


@func()
def use_default[T: Numeric](a: T) -> T:
    return add_default(a) # pyright: ignore[reportReturnType]


@func()
def accumulate(a: i32, b: i32) -> i32:
    a += b
    return a


@func()
def sign(n: i32) -> i32:
    if n > 0:
        return 1
    else:
        return -1


@func()
def clamped(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def assign_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    return x


@func()
def assign_in_both_branches(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    else:
        x = b + 1
    return x


@func()
def assign_parameter_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # a parameter is bound in the function body's scope: a branch writes it
    if c:
        a = b
    return a


@func()
def assign_tuple_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # ``x`` is bound outside (written), ``y`` is bound nowhere (declared here)
    x = a
    if c:
        x, y = b, a
        return x + y
    return x


@func()
def assign_from_choose(c: spy_bool, d: spy_bool, a: i32, b: i32) -> i32:
    # the branches of an if-expression write the variable the branch sees
    x = a
    if d:
        x = a if c else b
    return x


@func()
def branch_declaration(c: spy_bool, a: i32) -> i32:
    # a name bound nowhere is declared in the branch it is assigned in
    if c:
        y = a
        y = y + 1
        return y
    return a


@func()
def use_branch_declaration(c: spy_bool, a: i32) -> i32:
    if c:
        y = a
    # ``y`` was declared inside the branch, so it is not bound here: a spy
    # compile error (pyright cannot tell, hence the ignore)
    return y  # pyright: ignore


@func()
def le(a: i32, b: i32) -> spy_bool:
    return a <= b


@func()
def eq(a: i32, b: i32) -> spy_bool:
    return a == b


@func()
def is_i32(a) -> spy_bool:
    return spy_typeof(a) == i32


@func()
def is_u64(a) -> spy_bool:
    return spy_typeof(a) == u64


@func()
def nothing(_: i32) -> None:
    pass


@func()
def fact(n: i32) -> i32:
    if n <= 1:
        return 1
    return n * fact(n - 1)


@func()
def is_even(n: i32) -> spy_bool:
    if n == 0:
        return True
    return is_odd(n - 1)


@func()
def is_odd(n: i32) -> spy_bool:
    if n == 0:
        return False
    return is_even(n - 1)


@func()
def inc(a: i32) -> i32:
    return a + 1


@func()
def twice_inc(a: i32) -> i32:
    return inc(inc(a))


@func()
def id_f32(a: f32) -> f32:
    return a


# ---------------------------------------------------------------------------
# struct values: a struct is declared by decorating a class with ``@struct()``
# - its annotated class attributes are the fields, in declaration order, and
# the functions of its body are its methods - and is laid out by the mirror
# rules of ``sval.StructType._calculate_mir``.  A construction fills the
# fields of the location it is built in (a pending default constructor, see
# ``interp``) and a small struct (up to the by-value limit) is returned by
# value, a larger one through a result pointer (``sval.returns_via_result_ptr``).
#
# The structs of this module are named in the spy source like any other global:
# as an annotation (``s: Small``), as a constructor (``Small(...)``) and as the
# type a method is called on.  The tests read the struct *type* off the handle
# with ``struct_type``; there is no Python-level constructor yet.
# ---------------------------------------------------------------------------


def struct_type(handle: Any) -> sval.StructType:
    """The struct type a ``@struct()`` class declares."""
    type = handle.as_spy_value()
    assert isinstance(type, sval.StructType)
    return type


@struct()
class Small:
    a: i32
    b: i32

    def total(self) -> i32:
        return self.a + self.b


@struct()
class Large:
    a: i64
    b: i64
    c: i64
    d: i64


@struct()
class Counter:
    """A struct with both kinds of method (a registered ``bump``, compiled
    into a native call, and a plain ``double``, inlined)."""

    n: i32

    @func()
    def bump(self, k: i32) -> i32:
        self.n = self.n + k
        return self.n

    def double(self) -> i32:
        return self.n * 2


# a struct of one field mirrors to that field's own type, and the fields of
# a struct of several fields are ordered by alignment (the least-aligned
# first) - unless the struct is ``extern_c``, which keeps the C layout
@struct()
class One:
    a: i32

    def get(self) -> i32:
        return self.a


@struct()
class Nested:
    inner: One


@struct()
class Mixed:
    wide: i64
    narrow: i8


@struct(extern_c=True)
class ExternOne:
    a: i32


@struct(extern_c=True)
class ExternMixed:
    wide: i64
    narrow: i8


# a zero-sized field occupies no storage: it has no mirror position and its
# value is the unit value of its type
@struct()
class Holder:
    v: void
    n: i32


# a declaration built by hand: a pointer field has no annotation spelling yet
# (a pointer is word-aligned, like an integer of that width)
WithPointer: Any = sval.StructTypeHead('WithPointer')
WithPointer.add_field('p', sval.PointerType(i32))  # pyright: ignore[reportArgumentType]
WithPointer.add_field('n', i8)


@func()
def struct_local(x: i32) -> i32:
    s = Small(x, 1)
    return s.a + s.b


@func()
def struct_all_comptime() -> i32:
    s = Small(1, 2)
    return s.a + s.b


@func()
def make_small(x: i32) -> Small:
    return Small(x, 2)


@func()
def use_small(x: i32) -> i32:
    s = make_small(x)
    return s.a + s.b


@func()
def sum_small(s: Small) -> i32:
    return s.a + s.b


@func()
def use_sum_small(x: i32) -> i32:
    return sum_small(Small(x, 3))


@func()
def total_small(x: i32) -> i32:
    s = Small(x, 1)
    return s.total()


@func()
def make_large(x: i64, c: spy_bool) -> Large:
    if c:
        return Large(x, 1, 2, 3)
    return Large(x, 4, 5, 6)


@func()
def use_large(x: i64, c: spy_bool) -> i64:
    return make_large(x, c).a + make_large(x, c).d


@func()
def bump_counter(n: i32, k: i32) -> i32:
    c = Counter(n)
    c.bump(k)
    return c.double()


@func()
def one_field_local(x: i32) -> i32:
    s = One(x)
    return s.a


@func()
def mixed_fields_local(w: i64, n: i8) -> i64:
    m = Mixed(w, n)
    return m.wide + m.narrow


@func()
def extern_one_field_local(x: i32) -> i32:
    s = ExternOne(x)
    return s.a


@func()
def zst_field_local(n: i32) -> i32:
    h = Holder(None, n)
    return h.n


@func()
def nested_field_local(x: i32) -> i32:
    n = Nested(One(x))
    return n.inner.a


@func()
def nested_method_local(x: i32) -> i32:
    n = Nested(One(x))
    return n.inner.get()


@func()
def untyped_local(n: i32) -> i32:
    x = 1
    x = 2
    return x + n


@func()
def untyped_return():
    return 1


@func()
def untyped_param(x) -> i32:
    return x


@func()
def call_untyped_param() -> i32:
    return untyped_param(1)


@func()
def tuple_declare(x: i32) -> i32:
    a, b = x, x + 1
    return a * 10 + b


@func()
def tuple_swap(a: i32, b: i32) -> i32:
    a, b = b, a
    return a * 10 + b


@func()
def tuple_nested(x: i32) -> i32:
    a, (b, c) = x, (x + 1, x + 2)
    return a * 100 + b * 10 + c

@struct()
class Slice[T]:
    ptr: T
    len: u64


# ---------------------------------------------------------------------------
# generic structs: ``class Foo[T]`` declares a struct *template*, and
# ``Foo[i32]`` names one specialization of it.  Every method resolves through
# the specialization it is called on - ``a.m()`` behaves like
# ``typeof(a).m(a)`` - so its ``self`` is the struct template and a call
# substitutes the specialization's type arguments into the method's
# signature.  A type parameter is also usable as a value inside a body.
# ---------------------------------------------------------------------------


@struct()
class Pair[T]:
    a: T
    b: T

    @func()
    def total(self) -> T:
        return self.a + self.b  # pyright: ignore[reportOperatorIssue]

    def doubled_a(self) -> T:
        return self.a + self.a  # pyright: ignore[reportOperatorIssue]


@struct()
class Box[T]:
    """A generic struct with a registered method that returns its type
    parameter."""

    v: T

    @func()
    def get(self) -> T:
        return self.v


@struct()
class PlainBox[T]:
    """A generic struct whose method is an inlined plain method."""

    v: T

    def get(self) -> T:
        return self.v


@struct()
class Shadow[T]:
    """A generic struct with a method that declares a type parameter of its
    own, shadowing the struct's parameter of the same name."""

    a: T

    @func()
    def pick[T](self, b: T) -> T:  # pyright: ignore[reportGeneralTypeIssues]
        return b

    @func()
    def own(self) -> T:
        return self.a


@struct()
class NestedGeneric[T]:
    """A generic struct with a generic field: the field's type is
    substituted recursively."""

    inner: Pair[T]
    n: T


@struct()
class Maker[T]:
    """A generic struct whose method builds and returns a specialization of
    another generic struct from its own type parameter."""

    a: T

    @func()
    def pair(self) -> Pair[T]:
        return Pair[T](self.a, self.a)

    @func()
    def is_own_type(self) -> spy_bool:
        return spy_typeof(self.a) == T

    def plain_is_own_type(self) -> spy_bool:
        return spy_typeof(self.a) == T


@struct()
class StructHolder:
    """A non-generic struct with a generic field type."""

    p: Pair[i32]
    extra: i32


@func()
def make_pair[T](a: T, b: T) -> Pair[T]:
    return Pair[T](a, b)


@func()
def first[T](p: Pair[T]) -> T:
    return p.a


@func()
def generic_typeof[T](a: T) -> spy_bool:  # pyright: ignore[reportInvalidTypeVarUse]
    return spy_typeof(a) == T


@func()
def generic_pair_total(x: i32) -> i32:
    return Pair[i32](x, 3).total()


@func()
def generic_pair_plain_method(x: i32) -> i32:
    return Pair[i32](x, 4).doubled_a()


@func()
def generic_box_get(x: i32) -> i32:
    return Box[i32](x).get()


@func()
def generic_plain_box_get(x: i32) -> i32:
    return PlainBox[i32](x).get()


@func()
def generic_method_with_own_type_param(x: i32) -> i32:
    return Shadow[i64](0).pick(x)


@func()
def generic_method_uses_the_struct_type(x: i32) -> i64:
    return Shadow[i64](x).own()


@func()
def generic_struct_returned(x: i32) -> i32:
    return make_pair(x, 4).total()


@func()
def generic_struct_argument(x: i32) -> i32:
    return first(Pair[i32](x, x))


@func()
def generic_nested_field(x: i32) -> i32:
    n = NestedGeneric[i32](Pair[i32](x, 1), 2)
    return n.inner.total() + n.n


@func()
def generic_method_returns_a_struct(x: i32) -> i32:
    return Maker[i32](x).pair().total()


@func()
def generic_struct_typeof_dispatch(x: i32) -> spy_bool:
    return Maker[i32](x).is_own_type()


@func()
def generic_struct_typeof_inline(x: i32) -> spy_bool:
    return Maker[i32](x).plain_is_own_type()


@func()
def is_pair_i32(p: Pair[i32]) -> spy_bool:
    return spy_typeof(p) == Pair[i32]


@func()
def generic_struct_type_compared(x: i32) -> spy_bool:
    return is_pair_i32(Pair[i32](x, x))


@func()
def generic_field_of_nongeneric(x: i32) -> i32:
    h = StructHolder(Pair[i32](x, 1), 5)
    return h.p.total() + h.extra


# a generic struct whose type parameter is used by no field, built into a
# fresh local slot whose type is not known: a construction of the bare
# template cannot tell which specialization to build
@struct()
class Phantom[T]:
    v: i32

    def get(self) -> i32:
        return self.v


# the generic arguments of a construction that names the bare template are
# taken from the type of the location it is built into - here the result
# location of ``make_pair_inferred``, whose declared return type is known
@func()
def make_pair_inferred[T](a: T, b: T) -> Pair[T]:
    return Pair(a, b)


@func()
def inferred_pair_total(x: i32) -> i32:
    return make_pair_inferred(x, 3).total()


@func()
def inferred_pair_total_f64(x: f64) -> f64:
    return make_pair_inferred(x, 2.5).total()


@func()
def explicit_generic_construction(x: i32) -> i32:
    return Pair[i32](x, 3).total()


@func()
def keyword_construction(x: i32) -> i32:
    return Pair[i32](b=x, a=3).total()


@func()
def mixed_construction(x: i32) -> i32:
    return Pair[i32](x, b=x).total()


@func()
def uninferable_construction(x: i32) -> i32:
    return Phantom(x).get()


@func()
def wrong_generic_arguments(x: i32) -> i32:
    return Pair[i32, i64](x, 3).total()  # pyright: ignore


# a struct of one struct field whose own mirror is a struct type (two fields of
# the same width): the field still sits at the address of the value itself
@struct()
class TwoI64:
    a: i64
    b: i64


@struct()
class OuterTwo:
    inner: TwoI64


@func()
def nested_struct_field(x: i64) -> i64:
    o = OuterTwo(TwoI64(x, 1))
    return o.inner.a


# ---------------------------------------------------------------------------
# pointers: ``syntax.Ptr`` is C's pointer type - ``ref(a)`` takes the address
# of ``a`` (C's ``&a``) and ``p[...]`` denotes the place the pointer value
# ``p`` points at (C's ``*p``)
# ---------------------------------------------------------------------------


@func()
def deref_local(x: i32) -> i32:
    p = ref(x)
    return p[...]


@func()
def write_through_ptr(x: i32, v: i32) -> i32:
    p = ref(x)
    p[...] = v
    return x


@func()
def add_through_ptr(x: i32) -> i32:
    p = ref(x)
    p[...] += 1
    return x


@func()
def deref_module_qualified(x: i32) -> i32:
    # ``syntax.ref`` names the same function as a plain import of ``ref``
    p = syntax.ref(x)
    return p[...]


@func()
def incr_ptr(p: Ptr[i32]) -> i32:
    p[...] = p[...] + 1
    return p[...]


@func()
def call_incr_ptr(x: i32) -> i32:
    v = x
    r = incr_ptr(ref(v))
    return v * 100 + r


@func()
def deref_generic[T](p: Ptr[T]) -> T:
    # the pointee type is solved from the pointer argument
    return p[...]


@func()
def call_deref_generic(x: i32) -> i32:
    return deref_generic(ref(x))


@func()
def ptr_identity[T, C: bool](p: Ptr[T, C]) -> Ptr[T, C]:
    # the constness is a type parameter too: a call solves it to the
    # constness of the pointer it is given
    return p


@func()
def call_ptr_identity(x: i32) -> i32:
    p = ref(x)
    q = ptr_identity(p)
    return q[...]


@func()
def read_const_ptr(p: Ptr[i32, Literal[True]]) -> i32:
    return p[...]


@func()
def call_read_const_ptr(x: i32) -> i32:
    # a mutable pointer is accepted where a const one is expected (the
    # conversion is a spy rule; ``Ptr``'s type parameter does not express it)
    return read_const_ptr(ref(x))  # pyright: ignore[reportArgumentType]


@struct()
class PtrHolder:
    p: Ptr[i32]


@func()
def field_read(x: i32) -> i32:
    h = PtrHolder(ref(x))
    return h.p[...]


@func()
def field_write(x: i32, v: i32) -> i32:
    h = PtrHolder(ref(x))
    h.p[...] = v
    return x


@func()
def field_through_ptr(a: i32, b: i32) -> i32:
    # a pointer to a struct: a field of the pointee is written and read
    # through the pointer
    s = Small(a, b)
    p = ref(s)
    p[...].a = p[...].a + 1
    return p[...].total()


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

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SpyFunctionCallTest(TestCase):
    """Basic function calls: arithmetic, generics, calls between spy
    functions, inlining of plain functions, compile-time dispatch and
    recursion."""

    def test_smoke(self) -> None:
        self.assertEqual(smoke_test(2, 3), 5)
        self.assertEqual(smoke_test(-4, 1), -3)

    def test_arithmetic(self) -> None:
        self.assertEqual(sub(7, 12), -5)
        self.assertEqual(mul(6, 7), 42)
        self.assertEqual(mod(17, 5), 2)

    def test_generic_int(self) -> None:
        # a plain Python int marshals to the default signed 64-bit type
        # (see ``dsl._INT_LITERAL_BITS``)
        self.assertEqual(add(2, 3), 5)

    def test_generic_float(self) -> None:
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)

    def test_float_promotion(self) -> None:
        self.assertAlmostEqual(scale(3.0, 0.5), 2.5)

    def test_non_default_integer_type(self) -> None:
        # ``spy.as_`` binds an argument to an explicit spy type
        self.assertEqual(add_u64(spy_as(2**63 - 1, u64), spy_as(2, u64)), 2**63 + 1)

    def test_default_argument(self) -> None:
        self.assertEqual(add_default(41), 41)
        self.assertEqual(add_default(40, 2), 42)

    def test_call_spy_function(self) -> None:
        self.assertEqual(call_add(20, 22), 42)

    def test_inline_plain_function(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline(20, 22), 42)

    def test_inline_with_runtime_branches(self) -> None:
        # the inlined body has a runtime ``if`` whose branches both return
        # (the ``Block``/``Break`` shape of an inlined return)
        self.assertEqual(call_abs(-5), 5)
        self.assertEqual(call_abs(5), 5)
        # a branch returns, the other falls through to the trailing return
        self.assertEqual(call_clamp(42), 42)
        self.assertEqual(call_clamp(101), 100)
        # the inlined result is consumed by the caller's continuation
        self.assertEqual(call_abs_plus_one(-5), 6)

    def test_call_with_default(self) -> None:
        self.assertEqual(use_default(7), 7)

    def test_augmented_assignment(self) -> None:
        self.assertEqual(accumulate(5, 6), 11)

    def test_runtime_if_both_return(self) -> None:
        self.assertEqual(sign(5), 1)
        self.assertEqual(sign(-5), -1)

    def test_runtime_if_fallthrough(self) -> None:
        self.assertEqual(clamped(42), 42)
        self.assertEqual(clamped(101), 100)

    def test_runtime_if_assignment(self) -> None:
        # an assignment in a branch writes the variable the branch sees
        self.assertEqual(assign_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_in_both_branches(True, 1, 2), 2)
        self.assertEqual(assign_in_both_branches(False, 1, 2), 3)
        # ... a parameter included
        self.assertEqual(assign_parameter_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_parameter_in_branch(False, 1, 2), 1)
        # a destructuring target written outside, one declared in the branch
        self.assertEqual(assign_tuple_in_branch(True, 1, 2), 3)
        self.assertEqual(assign_tuple_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_from_choose(False, True, 1, 2), 2)
        self.assertEqual(assign_from_choose(True, True, 1, 2), 1)
        self.assertEqual(assign_from_choose(True, False, 1, 2), 1)

    def test_branch_declaration(self) -> None:
        # a name bound nowhere is declared in the branch it is assigned in,
        # and is invisible after it
        self.assertEqual(branch_declaration(True, 5), 6)
        self.assertEqual(branch_declaration(False, 5), 5)
        with self.assertRaises(CompileError):
            use_branch_declaration(True, 5)

    def test_comparisons(self) -> None:
        self.assertTrue(le(1, 2))
        self.assertFalse(le(2, 1))
        self.assertTrue(eq(3, 3))
        self.assertFalse(eq(3, 4))

    def test_comptime_typeof(self) -> None:
        # a plain Python int marshals to i64 (see ``dsl._INT_LITERAL_BITS``)
        self.assertFalse(is_i32(1))
        self.assertTrue(is_i32(spy_as(1, i32)))
        self.assertFalse(is_i32(1.0))
        self.assertTrue(is_u64(spy_as(1, u64)))

    def test_void_function(self) -> None:
        self.assertIsNone(nothing(3))

    def test_recursion(self) -> None:
        self.assertEqual(fact(5), 120)
        self.assertEqual(fact(0), 1)
        self.assertEqual(fact(10), 3628800)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(is_even(10))
        self.assertFalse(is_even(7))
        self.assertTrue(is_odd(7))

    def test_cross_module_call(self) -> None:
        # ``inc`` is compiled on its own first, so the compilation of
        # ``twice_inc`` references it as an external symbol of an earlier
        # module (the modular-compilation path)
        self.assertEqual(inc(10), 11)
        self.assertEqual(twice_inc(10), 12)

    def test_f32(self) -> None:
        self.assertAlmostEqual(id_f32(spy_as(0.5, f32)), 0.5)

    def test_specialization_is_cached(self) -> None:
        # repeated calls reuse the compiled specialization
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)

    def test_compile_error_on_unsupported_expression(self) -> None:
        with self.assertRaises(CompileError):
            div(1, 2)

    def test_untyped_integer_slot_is_rejected(self) -> None:
        # an untyped integer literal has no runtime type of its own: a
        # slot that has to live in memory must declare its type
        with self.assertRaises(CompileError):
            untyped_local(5)
        with self.assertRaises(CompileError):
            untyped_return()
        # the same holds for the slot of an unannotated parameter, which
        # is typed by the call (here: by an untyped literal argument)
        with self.assertRaises(CompileError):
            call_untyped_param()


@func()
def div(a: i32, b: i32) -> i32:
    return a / b # pyright: ignore[reportReturnType]


class SpyStructTest(TestCase):
    """Struct values: a construction fills the fields in place - the
    arguments bind the fields by declaration order and by name - the
    annotations of a spy function name a struct like any other type, and its
    methods are called on the object."""

    def test_fields_of_a_local(self) -> None:
        self.assertEqual(struct_local(5), 6)

    def test_all_fields_comptime(self) -> None:
        self.assertEqual(struct_all_comptime(), 3)

    def test_by_value_return(self) -> None:
        self.assertEqual(use_small(5), 7)

    def test_struct_annotation(self) -> None:
        self.assertEqual(use_sum_small(5), 8)

    def test_result_pointer_return(self) -> None:
        self.assertEqual(use_large(5, True), 8)
        self.assertEqual(use_large(5, False), 11)

    def test_plain_method(self) -> None:
        self.assertEqual(total_small(5), 6)

    def test_construction_and_methods(self) -> None:
        # ``Counter(1)`` fills the field ``n`` and ``bump`` mutates the
        # object through ``self``: 1 + 2 = 3, doubled is 6
        self.assertEqual(bump_counter(1, 2), 6)

    def test_one_field_mirror(self) -> None:
        # the field of such a struct is the struct itself: the slot of the
        # local is the storage of the field, with no address arithmetic
        self.assertEqual(one_field_local(5), 5)

    def test_reordered_fields(self) -> None:
        self.assertEqual(mixed_fields_local(5, 3), 8)

    def test_extern_c_field(self) -> None:
        self.assertEqual(extern_one_field_local(5), 5)

    def test_zero_sized_field(self) -> None:
        self.assertEqual(zst_field_local(5), 5)

    def test_nested_field(self) -> None:
        self.assertEqual(nested_field_local(5), 5)

    def test_method_on_a_field(self) -> None:
        self.assertEqual(nested_method_local(5), 5)

    def test_nested_struct_field(self) -> None:
        # a struct of one struct field whose mirror is itself a struct type:
        # the field is still at the address of the value itself
        self.assertEqual(nested_struct_field(5), 5)


class SpyGenericStructTest(TestCase):
    """Generic structs: ``class Foo[T]`` declares a template, ``Foo[i32]``
    names a specialization, and a method call carries that specialization's
    type arguments into the method (``a.m()`` is ``typeof(a).m(a)``)."""

    def test_construction_and_method(self) -> None:
        self.assertEqual(generic_pair_total(2), 5)

    def test_inlined_method(self) -> None:
        self.assertEqual(generic_pair_plain_method(2), 4)

    def test_registered_method(self) -> None:
        self.assertEqual(generic_box_get(2), 2)

    def test_plain_method(self) -> None:
        self.assertEqual(generic_plain_box_get(2), 2)

    def test_method_with_its_own_type_param(self) -> None:
        # the method's own ``T`` shadows the struct's ``T``
        self.assertEqual(generic_method_with_own_type_param(5), 5)

    def test_method_uses_the_struct_type_param(self) -> None:
        # ``own`` returns the struct's ``T``: the result is an i64
        self.assertEqual(generic_method_uses_the_struct_type(5), 5)

    def test_generic_struct_is_returned(self) -> None:
        self.assertEqual(generic_struct_returned(2), 6)

    def test_generic_struct_argument(self) -> None:
        # the type parameter is solved from the generic struct argument
        self.assertEqual(generic_struct_argument(7), 7)

    def test_nested_generic_field(self) -> None:
        self.assertEqual(generic_nested_field(2), 5)

    def test_method_returns_a_generic_struct(self) -> None:
        self.assertEqual(generic_method_returns_a_struct(3), 6)

    def test_type_param_is_a_value(self) -> None:
        # ``T`` used as a value in a body is the type the call solved it to,
        # in a registered method, an inlined one and a generic function
        self.assertTrue(generic_struct_typeof_dispatch(3))
        self.assertTrue(generic_struct_typeof_inline(3))
        self.assertTrue(generic_typeof(3))

    def test_struct_type_is_compared(self) -> None:
        # a specialization is also nameable as a value inside a body
        self.assertTrue(generic_struct_type_compared(3))

    def test_generic_field_of_a_non_generic_struct(self) -> None:
        self.assertEqual(generic_field_of_nongeneric(2), 8)

    def test_construction_with_explicit_arguments(self) -> None:
        # a construction of a generic struct at a site whose specialization
        # is not known names it explicitly, positionally or by keyword
        self.assertEqual(explicit_generic_construction(2), 5)
        self.assertEqual(keyword_construction(2), 5)
        self.assertEqual(mixed_construction(2), 4)

    def test_result_location_inference(self) -> None:
        # ``make_pair_inferred`` returns ``Pair(a, b)`` without naming the
        # specialization: the return type of the function is declared, so the
        # result location's type decides it
        self.assertEqual(inferred_pair_total(2), 5)
        self.assertAlmostEqual(inferred_pair_total_f64(1.5), 4.0)

    def test_uninferable_construction(self) -> None:
        # a construction that names the bare template and whose
        # construction site has no type cannot pick a specialization
        with self.assertRaises(CompileError):
            uninferable_construction(1)

    def test_wrong_generic_arguments(self) -> None:
        with self.assertRaises(CompileError):
            wrong_generic_arguments(1)


class SpyPointerTest(TestCase):
    """Pointers: ``syntax.Ptr`` annotates a pointer type, ``ref`` takes the
    address of a value and ``p[...]`` dereferences a pointer value."""

    def test_deref_a_local(self) -> None:
        self.assertEqual(deref_local(5), 5)

    def test_write_through_a_pointer(self) -> None:
        self.assertEqual(write_through_ptr(1, 9), 9)
        self.assertEqual(add_through_ptr(1), 2)

    def test_module_qualified_ref(self) -> None:
        self.assertEqual(deref_module_qualified(6), 6)

    def test_pointer_parameter(self) -> None:
        # ``incr_ptr(ref(v))`` mutates the caller's local through the pointer:
        # both the local and the returned pointee value are 6
        self.assertEqual(call_incr_ptr(5), 606)

    def test_pointee_type_is_solved(self) -> None:
        self.assertEqual(call_deref_generic(3), 3)

    def test_constness_is_solved(self) -> None:
        # ``C`` of ``Ptr[T, C]`` is solved from the pointer argument, and a
        # mutable pointer converts to a const one
        self.assertEqual(call_ptr_identity(4), 4)
        self.assertEqual(call_read_const_ptr(8), 8)

    def test_pointer_field(self) -> None:
        self.assertEqual(field_read(3), 3)
        self.assertEqual(field_write(3, 7), 7)

    def test_field_through_a_pointer(self) -> None:
        self.assertEqual(field_through_ptr(1, 2), 4)


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


class SpyStructMirrorTest(TestCase):
    """How a struct lowers to MIR (``sval.StructType._calculate_mir``): a spy
    struct is laid out by the compiler, an ``extern_c`` one for the C ABI."""

    def test_single_field_mirrors_to_the_field(self) -> None:
        one = struct_type(One)
        self.assertEqual(one.get_mir_type(), mir.IntType(32, True))
        self.assertEqual(one.get_field_mir_indices(), (0,))
        # ... and so does a struct of one struct field, whose own mirror is
        # the mirror of the field it holds
        self.assertEqual(struct_type(Nested).get_mir_type(), mir.IntType(32, True))

    def test_single_struct_field_mirror(self) -> None:
        # the mirror of a struct of one struct field is the mirror of the
        # field it holds - which may itself be a struct type, and the field
        # still sits at the address of the value itself
        outer = struct_type(OuterTwo)
        self.assertIsInstance(outer.get_mir_type(), mir.StructType)
        self.assertTrue(outer.mirror_is_a_field())
        self.assertEqual(outer.get_field_mir_indices(), (0,))

    def test_fields_are_ordered_by_alignment(self) -> None:
        mixed = struct_type(Mixed)
        mirror = mixed.get_mir_type()
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['narrow', 'wide'])
        self.assertEqual(mixed.get_field_mir_indices(), (1, 0))

        # a pointer is word-aligned, like an integer of that width
        with_pointer = WithPointer.specialize(())
        mirror = with_pointer.get_mir_type()
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['n', 'p'])

    def test_extern_c_keeps_the_declaration_order(self) -> None:
        extern_mixed = struct_type(ExternMixed)
        mirror = extern_mixed.get_mir_type()
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['wide', 'narrow'])
        self.assertEqual(extern_mixed.get_field_mir_indices(), (0, 1))
        # an ``extern_c`` struct of one field keeps its wrapper struct, so
        # that the C layout is the one the declaration asks for
        self.assertIsInstance(struct_type(ExternOne).get_mir_type(), mir.StructType)


class SpyTupleTest(TestCase):
    """Compile-time tuples: destructuring assignment unpacks a tuple of
    values into a tuple of target addresses (which may nest)."""

    def test_declaring_destructuring(self) -> None:
        self.assertEqual(tuple_declare(4), 45)

    def test_swap_destructuring(self) -> None:
        # the right-hand side is read before any target is stored, so a
        # swap does not clobber its own operands
        self.assertEqual(tuple_swap(1, 2), 21)

    def test_nested_destructuring(self) -> None:
        self.assertEqual(tuple_nested(4), 456)


class SpyCompileLogTest(TestCase):
    def test_compile_log_prints_at_compile_time(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline_log(1, 2), 3)
        self.assertIn('add_inline was compiled', out.getvalue())


all_tests = [
    SpyFunctionCallTest,
    SpyIfExprTest,
    SpyStructTest,
    SpyStructMirrorTest,
    SpyGenericStructTest,
    SpyPointerTest,
    SpyTupleTest,
    SpyCompileLogTest,
]

from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i32,
    i64,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct

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
def inferred_from_field_values(x: i32) -> i32:
    # ``Pair(...)`` built in a fresh local slot whose type is not known: the
    # generic argument is inferred from the values written into the fields
    p = Pair(x, x)
    return p.total()


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

    def test_inference_from_field_values(self) -> None:
        # ``Pair(x, x)`` built in a fresh local slot whose type is not known:
        # the generic argument comes from the values written into the fields
        self.assertEqual(inferred_from_field_values(3), 6)

    def test_uninferable_construction(self) -> None:
        # a construction that names the bare template and whose
        # construction site has no type cannot pick a specialization
        with self.assertRaises(CompileError):
            uninferable_construction(1)

    def test_wrong_generic_arguments(self) -> None:
        with self.assertRaises(CompileError):
            wrong_generic_arguments(1)


all_tests = [
    SpyGenericStructTest,
]

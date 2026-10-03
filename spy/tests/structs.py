from typing import TYPE_CHECKING, Any, Literal, Never, Protocol, cast
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

from .basics import nothing
from .generics import OuterTwo, nested_struct_field


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


def spy_type(annotation: Any) -> sval.Type:
    """The spy type a Python annotation evaluates to (see ``sval.as_value``)."""
    type = sval.as_value(annotation, _GLOBAL_CONTEXT)
    assert isinstance(type, sval.Type)
    return type


# the MIR-mirror interning table of the global host context: the MIR types the
# tests compare against are the ones it created (see ``sval.MirLowerCache``)
MIR_CACHE = _GLOBAL_CONTEXT.mir_lower_cache


def struct_mirror(handle: Any) -> mir.Type:
    """The MIR mirror of the struct type a handle declares (see
    ``sval.StructType.get_mir_type``); the structs a test takes the mirror of
    have storage, so it has one."""
    mirror = struct_type(handle).get_mir_type(MIR_CACHE)
    assert mirror is not None
    return mirror


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


# a struct whose fields are all zero-sized is zero-sized itself: it holds no
# storage at all, and returning one is a call that returns nothing
@struct()
class Nothing:
    v: void
    z: u0


# a second zero-sized struct, a different type than ``Nothing``
@struct()
class Blank:
    v: void


# a struct whose fields declare defaults: a construction may leave them out, and
# the field takes the value the class body assigned to it (see
# ``sval.StructField``/``interp.finish_struct``)
@struct()
class Defaulted:
    x: i32
    y: i32 = 7
    z: i32 = 9


# a struct whose zero-sized field declares a default too: leaving it out stores
# nothing, since every value of a zero-sized type is its unit value
@struct()
class DefaultedZst:
    v: void = None
    n: i32 = 3


# a generic struct with a defaulted field: the default is coerced to the type
# the field got from the specialization (``cast`` keeps the Python type checker
# happy about a type parameter's default, like ``std.range``'s fields)
@struct()
class GenericDefault[T]:
    value: T
    other: T = cast(T, 0)


# a struct whose void methods are called for their effect alone, one
# registered (``reset``, compiled into a native call) and one plain
# (``clear``, inlined)
@struct()
class Sink:
    n: i32

    @func()
    def reset(self) -> None:
        self.n = 0

    def clear(self) -> None:
        self.n = 0


# a declaration built by hand: a pointer field has no annotation spelling yet
# (a pointer is word-aligned, like an integer of that width)
WithPointer: Any = sval.StructTypeHead('WithPointer')


WithPointer.add_field('p', sval.PointerType(i32))  # pyright: ignore[reportArgumentType]


WithPointer.add_field('n', i8)


@func()
def struct_defaults(x: i32) -> i32:
    # every field left out is filled from its default
    p = Defaulted(x)
    return p.x + p.y + p.z


@func()
def struct_defaults_partial(x: i32) -> i32:
    p = Defaulted(x, 100)
    return p.x + p.y + p.z


@func()
def struct_defaults_comptime(x: i32) -> i32:
    # the defaults fill an inline (compile-time) aggregate too
    p: Comptime = Defaulted(x, 1)
    return p.x + p.y + p.z


@func()
def struct_defaults_zst() -> i32:
    p = DefaultedZst()
    return p.n


@func()
def struct_defaults_generic(x: i32) -> i32:
    g = GenericDefault[i32](x)
    return g.value + g.other


@func()
def struct_missing_a_value() -> i32:
    # a field with no default may not be left out
    p = Defaulted()  # pyright: ignore[reportCallIssue]
    return p.x


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


# ``nothing`` returns no value: as an expression statement the call is dropped
# (its result location is a temporary of no further use), and bound to a
# variable it types that variable as ``void``
@func()
def discard_void_call(x: i32) -> i32:
    nothing(x)
    return x


@func()
def void_call_type(x: i32) -> spy_bool:
    y = nothing(x)
    return spy_typeof(y) == void


@func()
def zero_bits(_: i32) -> u0:
    return 0


@func()
def discard_zero_bits_call(x: i32) -> i32:
    zero_bits(x)
    return x


@func()
def zero_bits_call_type(x: i32) -> spy_bool:
    y = zero_bits(x)
    return spy_typeof(y) == u0


@func()
def call_registered_void_method(x: i32) -> i32:
    s = Sink(x)
    s.reset()
    return s.n


@func()
def call_inline_void_method(x: i32) -> i32:
    s = Sink(x)
    s.clear()
    return s.n


@func()
def make_nothing(x: i32) -> Nothing:
    return Nothing(None, 0)


# a call that returns a zero-sized struct delivers no value, but its result
# location takes the struct type: the variable bound to the call is a
# compile-time box of the type's unit value, and it is typed as the struct
@func()
def use_nothing(x: i32) -> i32:
    n = make_nothing(x)
    return x if spy_typeof(n) == Nothing else x + 1


# the same struct built where it is declared: the construction writes nothing,
# so the slot takes its type from the construction itself
@func()
def construct_nothing(x: i32) -> i32:
    n = Nothing(None, 0)
    return x if spy_typeof(n) == Nothing else x + 1


# a construction takes its type from the construction itself: ``Blank``'s only
# field is zero-sized, and its value is given all the same (a field may only be
# left out when it has a default, which is not implemented yet)
@func()
def construct_blank(x: i32) -> i32:
    b = Blank(None)
    return x if spy_typeof(b) == Blank else x + 1


# a construction delivers its value like any other store point, so the slot's
# type is still the peer type of everything stored into it: two *different*
# zero-sized structs have no peer type (rather than the first one silently
# pinning the slot)
@func()
def choose_two_zst_structs(c: spy_bool) -> spy_bool:
    s = Blank(None) if c else Nothing(None, 0)
    return spy_typeof(s) == Blank


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


class SpyStructDefaultsTest(TestCase):
    """Struct field defaults: a construction may leave out a field the class
    body gave a value, and the field takes it (coerced to its type), exactly
    like a provided argument.  A field with no default still has to be given a
    value."""

    def test_a_left_out_field_takes_its_default(self) -> None:
        self.assertEqual(struct_defaults(1), 1 + 7 + 9)

    def test_a_provided_field_overrides_its_default(self) -> None:
        self.assertEqual(struct_defaults_partial(1), 1 + 100 + 9)

    def test_defaults_in_a_compile_time_aggregate(self) -> None:
        self.assertEqual(struct_defaults_comptime(1), 1 + 1 + 9)

    def test_a_zero_sized_default_stores_nothing(self) -> None:
        self.assertEqual(struct_defaults_zst(), 3)

    def test_a_default_is_coerced_to_the_field_type(self) -> None:
        # the default is the untyped literal ``0``, written into an ``i32`` field
        self.assertEqual(struct_defaults_generic(5), 5)

    def test_a_field_without_a_default_may_not_be_left_out(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            struct_missing_a_value()
        self.assertIn("missing a value for field 'x'", str(ctx.exception))


class SpyStructMirrorTest(TestCase):
    """How a struct lowers to MIR (``sval.StructType._calculate_mir``): a spy
    struct is laid out by the compiler, an ``extern_c`` one for the C ABI."""

    def test_single_field_mirrors_to_the_field(self) -> None:
        one = struct_type(One)
        self.assertEqual(one.get_mir_type(MIR_CACHE), mir.IntType(32, True))
        self.assertEqual(one.get_field_mir_indices(MIR_CACHE), (0,))
        # ... and so does a struct of one struct field, whose own mirror is
        # the mirror of the field it holds
        self.assertEqual(struct_type(Nested).get_mir_type(MIR_CACHE), mir.IntType(32, True))

    def test_single_struct_field_mirror(self) -> None:
        # the mirror of a struct of one struct field is the mirror of the
        # field it holds - which may itself be a struct type, and the field
        # still sits at the address of the value itself
        outer = struct_type(OuterTwo)
        self.assertIsInstance(outer.get_mir_type(MIR_CACHE), mir.StructType)
        self.assertTrue(outer.mirror_is_a_field(MIR_CACHE))
        self.assertEqual(outer.get_field_mir_indices(MIR_CACHE), (0,))

    def test_fields_are_ordered_by_alignment(self) -> None:
        mixed = struct_type(Mixed)
        mirror = mixed.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['narrow', 'wide'])
        self.assertEqual(mixed.get_field_mir_indices(MIR_CACHE), (1, 0))

        # a pointer is word-aligned, like an integer of that width
        with_pointer = WithPointer.specialize(())
        mirror = with_pointer.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['n', 'p'])

    def test_extern_c_keeps_the_declaration_order(self) -> None:
        extern_mixed = struct_type(ExternMixed)
        mirror = extern_mixed.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['wide', 'narrow'])
        self.assertEqual(extern_mixed.get_field_mir_indices(MIR_CACHE), (0, 1))
        # an ``extern_c`` struct of one field keeps its wrapper struct, so
        # that the C layout is the one the declaration asks for
        self.assertIsInstance(struct_type(ExternOne).get_mir_type(MIR_CACHE), mir.StructType)


class SpyZeroSizedResultTest(TestCase):
    """A call whose result is a zero-sized type.  The callee returns no value
    (a zero-sized type has no runtime representation, so the call is a MIR call
    of the void type), but the call still writes the result type's *unit value*
    into its result location: an expression statement then commits that value
    into the temporary it drops - the location is left untyped otherwise, and
    the type is what makes its slot a compile-time box - and a variable bound
    to the call is a box of the unit value, typed as the result type."""

    def test_void_call_as_a_statement(self) -> None:
        self.assertEqual(discard_void_call(7), 7)

    def test_void_call_bound_to_a_variable(self) -> None:
        self.assertTrue(void_call_type(7))

    def test_zero_bits_call_as_a_statement(self) -> None:
        self.assertEqual(discard_zero_bits_call(7), 7)

    def test_zero_bits_call_bound_to_a_variable(self) -> None:
        self.assertTrue(zero_bits_call_type(7))

    def test_registered_void_method_as_a_statement(self) -> None:
        self.assertEqual(call_registered_void_method(7), 0)

    def test_inline_void_method_as_a_statement(self) -> None:
        self.assertEqual(call_inline_void_method(7), 0)

    def test_zero_sized_struct_result(self) -> None:
        # every field of ``Nothing`` is zero-sized, so the struct itself is: it
        # has no layout (the size and alignment estimates of a layoutless type,
        # ``mir.estimated_size_of``, decide how one is returned) and a call
        # returning one delivers no value, only the type's unit value
        self.assertEqual(use_nothing(7), 7)

    def test_zero_sized_struct_local(self) -> None:
        self.assertEqual(construct_nothing(7), 7)
        self.assertEqual(construct_blank(7), 7)

    def test_zero_sized_construction_participates_in_peer_resolution(self) -> None:
        # the construction of a zero-sized struct delivers its value as an
        # ordinary store point, so the slot's type is the peer type of all its
        # stores - and two unrelated structs have none (the first construction
        # must not pin the slot's type by itself)
        with self.assertRaises(CompileError):
            choose_two_zst_structs(True)


all_tests = [
    SpyStructTest,
    SpyStructDefaultsTest,
    SpyStructMirrorTest,
    SpyZeroSizedResultTest,
]

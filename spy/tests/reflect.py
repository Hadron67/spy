from unittest import TestCase

from ..compiler import bool as spy_bool
from ..compiler import (
    i32,
    i64,
    u32,
    u64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Array,
    Comptime,
    Option,
    Ptr,
)
from ..std.core import coerce
from ..std.reflect import (
    ArrayType,
    IntType,
    OptionType,
    PointerType,
    StructType,
    TaggedUnionType,
    reify,
    type_info,
)

# ---------------------------------------------------------------------------
# compile-time reflection: ``std.reflect.type_info`` describes a compile-time
# type (``type``) as a ``TypeInfo`` value the interpreter builds while running
# the HIR - the ``type`` a body names, or the one ``spy.typeof`` yields
# ---------------------------------------------------------------------------


@struct()
class ReflectPoint:
    ab: i32
    c: i32


@struct()
class ReflectA:
    x: i32


@struct()
class ReflectB:
    y: i32


@struct()
class ReflectBox[T]:
    value: T


@func()
def reflect_int_bits() -> i32:
    info: Comptime = type_info(i32)
    if isinstance(v := info, IntType):
        return v.bits
    return -1


@func()
def reflect_int_signed() -> spy_bool:
    info: Comptime = type_info(i32)
    if isinstance(v := info, IntType):
        return v.signed
    return False


@func()
def reflect_typeof_bits(x: i64) -> i32:
    info: Comptime = type_info(spy_typeof(x))
    if isinstance(v := info, IntType):
        return v.bits
    return -1


@func()
def reflect_ptr_not_const() -> spy_bool:
    info: Comptime = type_info(Ptr[i32])
    if isinstance(v := info, PointerType):
        return v.is_const
    return True


@func()
def reflect_ptr_child() -> spy_bool:
    info: Comptime = type_info(Ptr[i32])
    if isinstance(v := info, PointerType):
        return v.child == i32
    return False


@func()
def reflect_array_child() -> spy_bool:
    info: Comptime = type_info(Array[i32, 3])
    if isinstance(v := info, ArrayType):
        return v.child == i32
    return False


@func()
def reflect_option_child() -> spy_bool:
    info: Comptime = type_info(Option[i64])
    if isinstance(v := info, OptionType):
        return v.child == i64
    return False


@func()
def reflect_field_count() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        return v.fields.length
    return -1


@func()
def reflect_field_name_byte() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[0]
        name: Comptime = field.name
        return ord(name[0])
    return -1


@func()
def reflect_field_name_arith() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[0]
        name: Comptime = field.name
        return ord(name[1])
    return -1


@func()
def reflect_tag_count() -> i32:
    info: Comptime = type_info(ReflectA | ReflectB)
    if isinstance(v := info, TaggedUnionType):
        return v.types.length
    return -1


@func()
def reflect_plain_has_head() -> spy_bool:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        return v.head is not None
    return False


@func()
def reflect_generic_has_head() -> spy_bool:
    info: Comptime = type_info(ReflectBox[i32])
    if isinstance(v := info, StructType):
        return v.head is not None
    return False


# ---------------------------------------------------------------------------
# ``std.reflect.reify``: the inverse of ``type_info`` - the compile-time type a
# ``TypeInfo`` value (or one of its variants) describes.  A reified *struct* is
# built fresh, so it never shares the identity of the struct it came from
# ---------------------------------------------------------------------------


@func()
def reify_int() -> spy_bool:
    return reify(type_info(i32)) == i32


@func()
def reify_ptr() -> spy_bool:
    return reify(type_info(Ptr[i32])) == Ptr[i32]


@func()
def reify_option() -> spy_bool:
    return reify(type_info(Option[i64])) == Option[i64]


@func()
def reify_tagged_union() -> spy_bool:
    return reify(type_info(ReflectA | ReflectB)) == (ReflectA | ReflectB)


@func()
def reify_array_child() -> spy_bool:
    t: Comptime = reify(type_info(Array[i32, 3]))
    info: Comptime = type_info(t)
    if isinstance(v := info, ArrayType):
        return v.child == i32
    return False


@func()
def reify_array_is_the_type() -> spy_bool:
    return reify(type_info(Array[i32, 3])) == Array[i32, 3]


@func()
def reify_a_variant() -> spy_bool:
    # a variant value is accepted too, not just a whole ``TypeInfo``: the
    # payload an ``isinstance`` unwrap binds is a bare variant
    info: Comptime = type_info(i64)
    if isinstance(v := info, IntType):
        return reify(v) == i64
    return False


@func()
def reify_a_struct_is_a_fresh_type() -> spy_bool:
    return reify(type_info(ReflectPoint)) != ReflectPoint


@func()
def reify_a_struct_is_fresh_each_time() -> spy_bool:
    a: Comptime = reify(type_info(ReflectPoint))
    b: Comptime = reify(type_info(ReflectPoint))
    return a != b


@func()
def reify_a_struct_keeps_its_fields() -> i32:
    t: Comptime = reify(type_info(ReflectPoint))
    info: Comptime = type_info(t)
    if isinstance(v := info, StructType):
        return v.fields.length
    return -1


@func()
def reify_a_struct_keeps_its_field_types() -> spy_bool:
    t: Comptime = reify(type_info(ReflectPoint))
    info: Comptime = type_info(t)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[1]
        return field.name == b'c' and field.type == i32
    return False


@func()
def reify_feeds_a_coerce(x: u32) -> u64:
    # the reified type is an ordinary compile-time type value: it can be the
    # type argument of another builtin
    return coerce(reify(type_info(u64)), x)


class SpyReflectTest(TestCase):
    """``std.reflect.type_info`` at compile time."""

    def test_an_integer_type_reflects_its_width_and_signedness(self) -> None:
        self.assertEqual(reflect_int_bits(), 32)
        self.assertTrue(reflect_int_signed())

    def test_the_type_of_a_value_reflects_like_the_type(self) -> None:
        self.assertEqual(reflect_typeof_bits(0), 64)

    def test_a_pointer_type_reflects_its_child_and_constness(self) -> None:
        self.assertFalse(reflect_ptr_not_const())
        self.assertTrue(reflect_ptr_child())

    def test_an_array_type_reflects_its_element_type(self) -> None:
        self.assertTrue(reflect_array_child())

    def test_an_option_type_reflects_its_child_type(self) -> None:
        self.assertTrue(reflect_option_child())

    def test_a_struct_type_reflects_its_fields(self) -> None:
        self.assertEqual(reflect_field_count(), 2)
        # the first field is named ``ab``: its bytes are readable as the
        # compile-time ``bytes`` value the name is stored as
        self.assertEqual(reflect_field_name_byte(), ord('a'))
        self.assertEqual(reflect_field_name_arith(), ord('b'))

    def test_a_tagged_union_type_reflects_its_variants(self) -> None:
        self.assertEqual(reflect_tag_count(), 2)

    def test_a_generic_struct_reflects_its_head(self) -> None:
        # a non-generic struct has no template head; a specialization of a
        # generic one names the head it was specialized from
        self.assertFalse(reflect_plain_has_head())
        self.assertTrue(reflect_generic_has_head())


class SpyReifyTest(TestCase):
    """``std.reflect.reify`` turns a ``TypeInfo`` back into the compile-time type
    it describes."""

    def test_scalar_like_types_reify_to_themselves(self) -> None:
        self.assertTrue(reify_int())
        self.assertTrue(reify_ptr())
        self.assertTrue(reify_option())
        self.assertTrue(reify_tagged_union())

    def test_an_array_type(self) -> None:
        self.assertTrue(reify_array_child())
        self.assertTrue(reify_array_is_the_type())

    def test_a_variant_is_accepted(self) -> None:
        self.assertTrue(reify_a_variant())

    def test_a_struct_is_a_fresh_type(self) -> None:
        # a struct is rebuilt from its fields, so it is neither the original ...
        self.assertTrue(reify_a_struct_is_a_fresh_type())
        # ... nor equal to another reify of the same struct
        self.assertTrue(reify_a_struct_is_fresh_each_time())
        # ... but it still describes the same fields
        self.assertEqual(reify_a_struct_keeps_its_fields(), 2)
        self.assertTrue(reify_a_struct_keeps_its_field_types())

    def test_the_reified_type_can_be_used(self) -> None:
        self.assertEqual(reify_feeds_a_coerce(0xFFFFFFFF), 0xFFFFFFFF)


all_tests = [
    SpyReflectTest,
    SpyReifyTest,
]

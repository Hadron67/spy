from unittest import TestCase

from ..compiler import bool as spy_bool
from ..compiler import (
    i32,
    i64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Array,
    Comptime,
    Option,
    Ptr,
)
from ..std.reflect import (
    ArrayType,
    IntType,
    OptionType,
    PointerType,
    StructType,
    TaggedUnionType,
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
        return name.ptr[0]
    return -1


@func()
def reflect_field_name_arith() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[0]
        name: Comptime = field.name
        p: Comptime = name.ptr + 1
        return p[...]
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
        # the first field is named ``ab``: its bytes are readable through the
        # ``ConstSlicePtr[u8]`` the name is stored as
        self.assertEqual(reflect_field_name_byte(), ord('a'))
        self.assertEqual(reflect_field_name_arith(), ord('b'))

    def test_a_tagged_union_type_reflects_its_variants(self) -> None:
        self.assertEqual(reflect_tag_count(), 2)

    def test_a_generic_struct_reflects_its_head(self) -> None:
        # a non-generic struct has no template head; a specialization of a
        # generic one names the head it was specialized from
        self.assertFalse(reflect_plain_has_head())
        self.assertTrue(reflect_generic_has_head())


all_tests = [
    SpyReflectTest,
]

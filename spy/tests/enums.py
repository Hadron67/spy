from enum import Enum
from unittest import TestCase

from ..compiler import (
    GLOBAL_CONTEXT,
    CompileError,
    SpyError,
    c_int,
    enum,
    f32,
    func,
    i0,
    i32,
    i64,
    mir,
    struct,
    sval,
    u0,
    u8,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof

# ---------------------------------------------------------------------------
# integer enumerations: a class inheriting ``enum.Enum`` whose members have
# integer values is a spy type of its own (``sval.IntEnumType``).  It is *not*
# an integer: it never takes part in arithmetic, implicit conversion or
# comparison with an ``int``, and only its own members compare equal.  It is
# represented at runtime as the smallest integer type that holds every member
# value (unsigned, or signed when a value is negative), unless ``@enum(...)``
# names the representation explicitly.  A zero-width representation (``i0``/
# ``u0``) makes the enumeration a ZST.
# ---------------------------------------------------------------------------


class TriState(Enum):
    UNKNOWN = 0
    TRUE = 1
    FALSE = 2

    @func()
    def is_true(self) -> spy_bool:
        return self == TriState.TRUE

    def is_false(self) -> spy_bool:
        return self == TriState.FALSE

    @func()
    def and_with(self, other: TriState) -> TriState:
        return tri_state_and(self, other)

    @staticmethod
    @func()
    def unknown() -> TriState:
        return TriState.UNKNOWN

    @func(self_arg="ptr")
    def is_true_by_ref(self) -> spy_bool:
        return self == TriState.TRUE

    @func(self_arg="const ptr")
    def is_true_by_const_ref(self) -> spy_bool:
        return self == TriState.TRUE


class Sign(Enum):
    NEG = -1
    ZERO = 0
    POS = 1


@enum(c_int)
class CIntState(Enum):
    ZERO = 0
    ONE = 1


@enum(u8)
class ByteValue(Enum):
    ZERO = 0
    MAX = 255


@enum(i64)
class WideSigned(Enum):
    NEG = -1
    ZERO = 0


@enum(u0)
class ZeroSized(Enum):
    ZERO = 0


# the declarations the type builder rejects: a value that does not fit the
# named representation, and an enumeration with no members at all


@enum(u8)
class OutOfRange(Enum):
    TOO_BIG = 256


@enum(i0)
class NonZeroZeroWidth(Enum):
    ONE = 1


class Empty(Enum):
    pass


@func()
def tri_state_and(a: TriState, b: TriState) -> TriState:
    if a == TriState.FALSE or b == TriState.FALSE:
        return TriState.FALSE
    if a == TriState.TRUE and b == TriState.TRUE:
        return TriState.TRUE
    return TriState.UNKNOWN


@func()
def tri_state_not(a: TriState) -> TriState:
    if a == TriState.TRUE:
        return TriState.FALSE
    if a == TriState.FALSE:
        return TriState.TRUE
    return TriState.UNKNOWN


@func()
def tri_state_is_true(a: TriState) -> spy_bool:
    return a == TriState.TRUE


@func()
def tri_state_ne(a: TriState, b: TriState) -> spy_bool:
    return a != b


@func()
def tri_state_typeof_matches(a: TriState) -> spy_bool:
    return spy_typeof(a) == TriState


@func()
def tri_state_pick(which: i32) -> TriState:
    if which == 0:
        return TriState.UNKNOWN
    if which == 1:
        return TriState.TRUE
    return TriState.FALSE


@func()
def tri_state_comptime_eq() -> spy_bool:
    return TriState.TRUE == TriState.TRUE


@func()
def tri_state_comptime_ne() -> spy_bool:
    return TriState.TRUE == TriState.FALSE


@func()
def sign_negate(s: Sign) -> Sign:
    if s == Sign.NEG:
        return Sign.POS
    return Sign.ZERO


@func()
def sign_is_neg(s: Sign) -> spy_bool:
    return s == Sign.NEG


@func()
def wide_pick(which: i32) -> WideSigned:
    if which == 0:
        return WideSigned.NEG
    return WideSigned.ZERO


@struct()
class WideMachine:
    state: WideSigned
    count: i32


@func()
def wide_machine_make(state: WideSigned, count: i32) -> WideMachine:
    return WideMachine(state, count)


@func()
def zero_sized_roundtrip(x: ZeroSized) -> ZeroSized:
    return x


@func()
def tri_state_method_true(a: TriState) -> spy_bool:
    return a.is_true()


@func()
def tri_state_method_false(a: TriState) -> spy_bool:
    return a.is_false()


@func()
def tri_state_and_with(a: TriState, b: TriState) -> TriState:
    return a.and_with(b)


@func()
def tri_state_class_call(a: TriState) -> spy_bool:
    return TriState.is_true(a)


@func()
def tri_state_on_member() -> TriState:
    return TriState.TRUE.and_with(TriState.FALSE)


@func()
def tri_state_static() -> TriState:
    return TriState.unknown()


@func()
def tri_state_self_arg_ref(a: TriState) -> spy_bool:
    return a.is_true_by_ref()


@func()
def tri_state_self_arg_const_ref(a: TriState) -> spy_bool:
    return a.is_true_by_const_ref()


@struct()
class Machine:
    state: TriState
    count: i32

    def is_on(self) -> spy_bool:
        return self.state == TriState.TRUE


@func()
def machine_make(state: TriState, count: i32) -> Machine:
    return Machine(state, count)


@func()
def machine_on() -> Machine:
    return Machine(TriState.TRUE, 1)


# the operations an enumeration does not support: comparing it to an integer or
# to another enumeration, arithmetic, and converting an integer to it


@func()
def enum_bad_int_compare(a: TriState) -> spy_bool:
    return a == 1


@func()
def enum_bad_cross_compare(a: TriState, b: Sign) -> spy_bool:
    return a == b


@func()
def enum_bad_order(a: TriState, b: TriState) -> spy_bool:
    return a < b  # type: ignore[operator]


@func()
def enum_bad_arith(a: TriState, b: TriState) -> TriState:
    return a + b  # type: ignore[operator]


@func()
def enum_bad_from_int(n: i32) -> TriState:
    return n  # type: ignore[return-value]


class SpyEnumTest(TestCase):
    """Integer enumerations: definition, comparison, marshaling and layout."""

    def enum_type(self, cls: type) -> sval.IntEnumType:
        type = sval.as_value(cls, GLOBAL_CONTEXT)
        assert isinstance(type, sval.IntEnumType)
        return type

    def test_the_and_of_two_states(self) -> None:
        self.assertIs(tri_state_and(TriState.FALSE, TriState.TRUE), TriState.FALSE)
        self.assertIs(tri_state_and(TriState.TRUE, TriState.TRUE), TriState.TRUE)
        self.assertIs(tri_state_and(TriState.TRUE, TriState.UNKNOWN), TriState.UNKNOWN)

    def test_a_member_is_returned_as_its_python_member(self) -> None:
        self.assertIs(tri_state_not(TriState.TRUE), TriState.FALSE)
        self.assertIs(tri_state_not(TriState.FALSE), TriState.TRUE)
        self.assertIs(tri_state_not(TriState.UNKNOWN), TriState.UNKNOWN)

    def test_equality_and_inequality(self) -> None:
        self.assertTrue(tri_state_is_true(TriState.TRUE))
        self.assertFalse(tri_state_is_true(TriState.FALSE))
        self.assertTrue(tri_state_ne(TriState.TRUE, TriState.FALSE))
        self.assertFalse(tri_state_ne(TriState.TRUE, TriState.TRUE))

    def test_the_type_of_a_value_is_its_enumeration(self) -> None:
        self.assertTrue(tri_state_typeof_matches(TriState.TRUE))

    def test_compile_time_member_comparison_folds(self) -> None:
        self.assertTrue(tri_state_comptime_eq())
        self.assertFalse(tri_state_comptime_ne())

    def test_a_runtime_value_selects_a_member(self) -> None:
        self.assertIs(tri_state_pick(0), TriState.UNKNOWN)
        self.assertIs(tri_state_pick(1), TriState.TRUE)
        self.assertIs(tri_state_pick(2), TriState.FALSE)

    def test_negative_members_use_a_signed_representation(self) -> None:
        self.assertIs(sign_negate(Sign.NEG), Sign.POS)
        self.assertIs(sign_negate(Sign.POS), Sign.ZERO)
        self.assertTrue(sign_is_neg(Sign.NEG))
        self.assertFalse(sign_is_neg(Sign.POS))

    def test_an_enum_field_of_a_struct_crosses_the_boundary(self) -> None:
        machine = machine_on()
        self.assertIs(machine.state, TriState.TRUE)
        self.assertEqual(machine.count, 1)
        self.assertTrue(machine.is_on())

    def test_python_side_construction_and_method_call(self) -> None:
        machine = Machine(TriState.FALSE, 7)
        self.assertIs(machine.state, TriState.FALSE)
        self.assertEqual(machine.count, 7)
        self.assertFalse(machine.is_on())
        made = machine_make(TriState.UNKNOWN, 3)
        self.assertIs(made.state, TriState.UNKNOWN)
        self.assertEqual(made.count, 3)

    def test_a_decorated_method_from_spy_code(self) -> None:
        self.assertTrue(tri_state_method_true(TriState.TRUE))
        self.assertFalse(tri_state_method_true(TriState.FALSE))

    def test_an_undecorated_method_is_inlined_from_spy_code(self) -> None:
        self.assertTrue(tri_state_method_false(TriState.FALSE))
        self.assertFalse(tri_state_method_false(TriState.TRUE))

    def test_a_method_taking_another_value_of_the_enum(self) -> None:
        self.assertIs(tri_state_and_with(TriState.FALSE, TriState.TRUE), TriState.FALSE)
        self.assertIs(tri_state_and_with(TriState.TRUE, TriState.TRUE), TriState.TRUE)

    def test_a_member_method_call_from_spy_code(self) -> None:
        self.assertIs(tri_state_on_member(), TriState.FALSE)

    def test_a_method_called_through_the_class_name(self) -> None:
        self.assertTrue(tri_state_class_call(TriState.TRUE))
        self.assertFalse(tri_state_class_call(TriState.FALSE))

    def test_a_static_method(self) -> None:
        self.assertIs(tri_state_static(), TriState.UNKNOWN)
        self.assertIs(TriState.unknown(), TriState.UNKNOWN)
        self.assertIs(TriState.TRUE.unknown(), TriState.UNKNOWN)

    def test_python_side_method_calls(self) -> None:
        # a decorated method goes through the handle (compiled), an undecorated
        # one is the plain Python function of the class body
        self.assertTrue(TriState.TRUE.is_true())
        self.assertFalse(TriState.TRUE.is_false())
        self.assertIs(TriState.TRUE.and_with(TriState.FALSE), TriState.FALSE)
        self.assertTrue(TriState.is_true(TriState.TRUE))

    def test_self_is_by_value_by_default(self) -> None:
        enum_type = self.enum_type(TriState)
        handle = enum_type.get_method('is_true')
        assert handle is not None
        first = handle.get_entry().hir.signature.positional.by_id[0]
        self.assertIs(first.type, enum_type)

    def test_an_explicit_self_arg_ptr_passes_self_by_reference(self) -> None:
        enum_type = self.enum_type(TriState)
        handle = enum_type.get_method('is_true_by_ref')
        assert handle is not None
        first = handle.get_entry().hir.signature.positional.by_id[0]
        self.assertIsInstance(first.type, sval.PointerType)
        self.assertFalse(first.type.is_const)
        self.assertIs(first.type.elem, enum_type)
        self.assertTrue(tri_state_self_arg_ref(TriState.TRUE))
        self.assertFalse(tri_state_self_arg_ref(TriState.FALSE))

    def test_an_explicit_self_arg_const_ptr_passes_self_by_const_reference(self) -> None:
        enum_type = self.enum_type(TriState)
        handle = enum_type.get_method('is_true_by_const_ref')
        assert handle is not None
        first = handle.get_entry().hir.signature.positional.by_id[0]
        self.assertIsInstance(first.type, sval.PointerType)
        self.assertTrue(first.type.is_const)
        self.assertIs(first.type.elem, enum_type)
        self.assertTrue(tri_state_self_arg_const_ref(TriState.TRUE))
        self.assertFalse(tri_state_self_arg_const_ref(TriState.FALSE))

    def test_the_representation_is_the_smallest_integer_type(self) -> None:
        tri_state_mir = self.enum_type(TriState).to_mir_type(GLOBAL_CONTEXT.mir_lower_cache)
        self.assertEqual(tri_state_mir, mir.IntType(2, False))
        sign_mir = self.enum_type(Sign).to_mir_type(GLOBAL_CONTEXT.mir_lower_cache)
        self.assertEqual(sign_mir, mir.IntType(2, True))

    def test_a_manual_c_int_representation(self) -> None:
        target = GLOBAL_CONTEXT.target_info()
        enum_type = self.enum_type(CIntState)
        self.assertEqual(enum_type.int_type, sval.IntType(target.c_int_bits, True))
        self.assertEqual(
            enum_type.to_mir_type(GLOBAL_CONTEXT.mir_lower_cache),
            mir.IntType(target.c_int_bits, True),
        )

    def test_a_manual_unsigned_representation(self) -> None:
        enum_type = self.enum_type(ByteValue)
        self.assertEqual(enum_type.int_type, sval.IntType(8, False))
        self.assertEqual(
            enum_type.to_mir_type(GLOBAL_CONTEXT.mir_lower_cache), mir.IntType(8, False),
        )

    def test_a_representation_wider_than_needed(self) -> None:
        enum_type = self.enum_type(WideSigned)
        self.assertEqual(enum_type.int_type, sval.IntType(64, True))
        self.assertEqual(
            enum_type.to_mir_type(GLOBAL_CONTEXT.mir_lower_cache), mir.IntType(64, True),
        )

    def test_a_zero_width_representation_is_a_zst(self) -> None:
        enum_type = self.enum_type(ZeroSized)
        self.assertIsNone(enum_type.to_mir_type(GLOBAL_CONTEXT.mir_lower_cache))
        self.assertEqual(enum_type.classify(), sval.SpecialTypeKind.ZST)
        unit = enum_type.get_unit_value()
        self.assertIsInstance(unit, sval.IntEnumValue)
        assert isinstance(unit, sval.IntEnumValue)
        self.assertIs(unit.type, enum_type)
        self.assertEqual(enum_type.value_of(unit.index), 0)

    def test_zero_width_int_range_is_zero(self) -> None:
        i0 = sval.IntType(0, True)
        u0 = sval.IntType(0, False)
        self.assertEqual(sval.int_range(i0), (0, 0))
        self.assertEqual(sval.int_range(u0), (0, 0))
        self.assertEqual(sval.coerce_const(0, i0), sval.Int(0, i0))
        self.assertRaises(CompileError, sval.coerce_const, 1, i0)

    def test_a_manual_representation_crosses_the_boundary(self) -> None:
        self.assertIs(wide_pick(0), WideSigned.NEG)
        self.assertIs(wide_pick(1), WideSigned.ZERO)
        machine = wide_machine_make(WideSigned.NEG, 3)
        self.assertIs(machine.state, WideSigned.NEG)
        self.assertEqual(machine.count, 3)

    def test_a_zero_width_enumeration_crosses_the_boundary(self) -> None:
        self.assertIsNone(zero_sized_roundtrip(ZeroSized.ZERO))

    def test_a_member_out_of_range_is_rejected(self) -> None:
        self.assertRaises(CompileError, sval.as_value, OutOfRange, GLOBAL_CONTEXT)

    def test_a_nonzero_member_of_a_zero_width_enum_is_rejected(self) -> None:
        self.assertRaises(CompileError, sval.as_value, NonZeroZeroWidth, GLOBAL_CONTEXT)

    def test_an_empty_enum_is_rejected(self) -> None:
        self.assertRaises(CompileError, sval.as_value, Empty, GLOBAL_CONTEXT)

    def test_a_non_integer_representation_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            @enum(f32)
            class Bad(Enum):
                ZERO = 0

    def test_the_bare_decorator_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            @enum  # type: ignore[arg-type]
            class Bad(Enum):
                ZERO = 0

    def test_decorating_a_non_enum_class_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            @enum(c_int)
            class Bad:
                pass

    def test_comparing_an_enumeration_to_an_integer_is_rejected(self) -> None:
        self.assertRaises(CompileError, enum_bad_int_compare, TriState.TRUE)

    def test_comparing_two_different_enumerations_is_rejected(self) -> None:
        self.assertRaises(CompileError, enum_bad_cross_compare, TriState.TRUE, Sign.POS)

    def test_ordering_an_enumeration_is_rejected(self) -> None:
        self.assertRaises(CompileError, enum_bad_order, TriState.TRUE, TriState.FALSE)

    def test_arithmetic_on_an_enumeration_is_rejected(self) -> None:
        self.assertRaises(CompileError, enum_bad_arith, TriState.TRUE, TriState.TRUE)

    def test_converting_an_integer_to_an_enumeration_is_rejected(self) -> None:
        self.assertRaises(CompileError, enum_bad_from_int, 1)

    def test_passing_a_plain_integer_is_rejected(self) -> None:
        self.assertRaises(SpyError, tri_state_and, 1, 2)


all_tests = [
    SpyEnumTest,
]

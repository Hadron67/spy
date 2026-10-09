from enum import Enum
from unittest import TestCase

from ..compiler import (
    CompileError,
    SpyError,
    i32,
    mir,
    sval,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, func, struct

# ---------------------------------------------------------------------------
# integer enumerations: a class inheriting ``enum.Enum`` whose members have
# integer values is a spy type of its own (``sval.IntEnumType``).  It is *not*
# an integer: it never takes part in arithmetic, implicit conversion or
# comparison with an ``int``, and only its own members compare equal.  It is
# represented at runtime as the smallest integer type that holds every member
# value (unsigned, or signed when a value is negative).
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

    @func(sfv=False)
    def is_true_by_ref(self) -> spy_bool:
        return self == TriState.TRUE


class Sign(Enum):
    NEG = -1
    ZERO = 0
    POS = 1


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
def tri_state_sfv_ref(a: TriState) -> spy_bool:
    return a.is_true_by_ref()


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
        type = sval.as_value(cls, _GLOBAL_CONTEXT)
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

    def test_an_explicit_sfv_false_passes_self_by_reference(self) -> None:
        enum_type = self.enum_type(TriState)
        handle = enum_type.get_method('is_true_by_ref')
        assert handle is not None
        first = handle.get_entry().hir.signature.positional.by_id[0]
        self.assertIsInstance(first.type, sval.PointerType)
        self.assertIs(first.type.elem, enum_type)
        self.assertTrue(tri_state_sfv_ref(TriState.TRUE))
        self.assertFalse(tri_state_sfv_ref(TriState.FALSE))

    def test_the_representation_is_the_smallest_integer_type(self) -> None:
        tri_state_mir = self.enum_type(TriState).to_mir_type(_GLOBAL_CONTEXT.mir_lower_cache)
        self.assertEqual(tri_state_mir, mir.IntType(2, False))
        sign_mir = self.enum_type(Sign).to_mir_type(_GLOBAL_CONTEXT.mir_lower_cache)
        self.assertEqual(sign_mir, mir.IntType(2, True))

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

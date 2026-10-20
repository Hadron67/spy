from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    Comptime,
    Option,
    Ptr,
    func,
    i32,
    i64,
    ref,
    struct,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from .structs import Large

# ---------------------------------------------------------------------------
# tagged unions: ``A | B``, the tag test ``isinstance(u, A)`` / the unwrap
# ``isinstance(e := u, A)``, and ``match u: case A(): ...`` (which tests the
# tag) / ``match u: case A() as e: ...`` (which binds the payload)
# ---------------------------------------------------------------------------


@struct()
class TU_A:
    x: i32


@struct()
class TU_B:
    y: i32


@struct()
class TU_C:
    z: i32


@struct()
class TU_Z1:
    pass


@struct()
class TU_Z2:
    pass


# the pointer variant's type, behind a name (``isinstance`` of a subscripted
# marker is spy syntax the Python type checker does not model)
TU_PTR_I32: Any = Ptr[i32]


@func()
def tu_make(sel: i32) -> TU_A | TU_B:
    if sel == 0:
        return TU_A(3)
    return TU_B(4)


@func()
def tu_tag_of(u: TU_A | TU_B) -> i32:
    # ``isinstance`` without an unwrap is an ordinary boolean test
    if isinstance(u, TU_A):
        return 1
    return 2


@func()
def tu_tag_of_make(sel: i32) -> i32:
    return tu_tag_of(tu_make(sel))


@func()
def tu_unwrap(u: TU_A | TU_B) -> i32:
    if isinstance(a := u, TU_A):
        return a.x
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_unwrap_make(sel: i32) -> i32:
    return tu_unwrap(tu_make(sel))


@func()
def tu_negated(u: TU_A | TU_B) -> i32:
    if not isinstance(u, TU_A):
        return 1
    return 0


@func()
def tu_negated_make(sel: i32) -> i32:
    return tu_negated(tu_make(sel))


@func()
def tu_widen(u: TU_A | TU_B) -> i32:
    # a value of a subset union converts to the wider one (the tag is remapped)
    v: TU_A | TU_B | TU_C = u
    if isinstance(c := v, TU_C):
        return c.z
    if isinstance(a := v, TU_A):
        return a.x
    if isinstance(b := v, TU_B):
        return b.y
    return -1


@func()
def tu_widen_make(sel: i32) -> i32:
    return tu_widen(tu_make(sel))


@func()
def tu_match(u: TU_A | TU_B) -> i32:
    match u:
        case TU_A():
            return 1
        case TU_B():
            return 2
    return -1


@func()
def tu_match_make(sel: i32) -> i32:
    return tu_match(tu_make(sel))


@func()
def tu_match_wildcard(u: TU_A | TU_B) -> i32:
    match u:
        case TU_A():
            return 1
        case _:
            return 2


@func()
def tu_match_wildcard_make(sel: i32) -> i32:
    return tu_match_wildcard(tu_make(sel))


@func()
def tu_match_bind(u: TU_A | TU_B) -> i32:
    # ``case T() as name`` binds the payload of the matched variant
    match u:
        case TU_A() as a:
            return a.x
        case TU_B() as b:
            return b.y
    return -1


@func()
def tu_match_bind_make(sel: i32) -> i32:
    return tu_match_bind(tu_make(sel))


@func()
def tu_match_bind_wildcard(u: TU_A | TU_B) -> i32:
    # the wildcard binds nothing: only the case that names a variant binds
    match u:
        case TU_A() as a:
            return a.x
        case _:
            return -1


@func()
def tu_match_bind_wildcard_make(sel: i32) -> i32:
    return tu_match_bind_wildcard(tu_make(sel))


@func()
def tu_match_tag(u: TU_A | TU_B) -> i32:
    # a plain-name subject: the tag is tested, no payload is bound
    match u:
        case TU_A():
            return 1
        case TU_B():
            return 2
    return -1


@func()
def tu_match_tag_make(sel: i32) -> i32:
    return tu_match_tag(tu_make(sel))


@func()
def tu_match_no_shadow(u: TU_A | TU_B) -> i32:
    # no payload is bound, so the subject name still names the union in the
    # case body, where it can be unwrapped
    match u:
        case TU_A():
            if isinstance(a := u, TU_A):
                return a.x
            return -1
        case _:
            return -2


@func()
def tu_match_no_shadow_make(sel: i32) -> i32:
    return tu_match_no_shadow(tu_make(sel))


@func()
def tu_scalar(sel: i32) -> i32:
    u: i32 | i64
    if sel == 0:
        u = 5
    else:
        u = 100
    if isinstance(i := u, i32):
        return i
    return -1


@func()
def tu_zst(sel: i32) -> i32:
    # every variant is zero-sized: only the tag is stored
    u: TU_Z1 | TU_Z2
    if sel == 0:
        u = TU_Z1()
    else:
        u = TU_Z2()
    if isinstance(_ := u, TU_Z1):
        return 1
    return 2


@func()
def tu_single() -> i32:
    # ``A | A`` de-duplicates to a single variant, which has no tag of its own
    u: TU_A | TU_A  # noqa: PYI016
    u = TU_A(9)
    if isinstance(a := u, TU_A):
        return a.x
    return -1


@func()
def tu_ptr(p: Ptr[i32]) -> i32:
    u: Ptr[i32] | i32
    u = p
    if isinstance(pp := u, TU_PTR_I32):
        return pp[...]
    return 0


@func()
def tu_ptr_driver() -> i32:
    x: i32 = 42
    return tu_ptr(ref(x))


@func()
def tu_comptime() -> i32:
    # a declared ``Comptime`` tagged union: a fully compile-time union value,
    # held in the compile-time box (a ``sval.TaggedUnionValue``)
    u: Comptime[TU_A | TU_B] = TU_A(2)
    if isinstance(a := u, TU_A):
        return a.x
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_comptime_reassigned(x: i32) -> i32:
    # switching a compile-time union to a variant with a *runtime* payload is
    # rejected: a runtime value cannot be stored into a compile-time variable
    u: Comptime[TU_A | TU_B]
    u = TU_A(2)
    u = TU_B(x)
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_comptime_reassigned_comptime() -> i32:
    # ... and a compile-time payload stays a box
    u: Comptime[TU_A | TU_B] = TU_A(2)
    u = TU_B(7)
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_typeof(u: TU_A | TU_B) -> i32:
    if spy_typeof(u) == (TU_A | TU_B):
        return 1
    return 0


@func()
def tu_typeof_make(sel: i32) -> i32:
    return tu_typeof(tu_make(sel))


@func()
def tu_large_make(sel: i32) -> Large | TU_A:
    # a union large enough to be returned through a result pointer
    if sel == 0:
        return Large(1, 2, 3, 4)
    return TU_A(5)


@func()
def tu_large_use(sel: i32) -> i32:
    u: Large | TU_A = tu_large_make(sel)
    if isinstance(a := u, TU_A):
        return a.x
    return 0


@struct()
class TU_Holder:
    u: TU_A | TU_B
    n: i32


@func()
def tu_field(sel: i32) -> i32:
    h: TU_Holder
    if sel == 0:
        h = TU_Holder(TU_A(6), 1)
    else:
        h = TU_Holder(TU_B(7), 2)
    if isinstance(a := h.u, TU_A):
        return a.x
    return 0


@func()
def tu_field_write(sel: i32) -> i32:
    h: TU_Holder = TU_Holder(TU_A(1), 0)
    if sel == 0:
        h.u = TU_A(8)
    else:
        h.u = TU_B(9)
    if isinstance(a := h.u, TU_A):
        return a.x
    return -1


@func()
def tu_bad_variant() -> i32:
    u: TU_A | TU_B
    u = TU_A(1)
    if isinstance(u, TU_C):
        return 1
    return 0


@func()
def tu_bad_pattern() -> i32:
    u: TU_A | TU_B
    u = TU_A(1)
    match u:
        case TU_A(x):
            return x
        case _:
            return 0


@func()
def tu_capture_pattern() -> i32:
    u: TU_A | TU_B
    u = TU_A(1)
    match u:
        case e:  # noqa: F841 (the pattern is what is under test)
            return 0


# ---------------------------------------------------------------------------
# a construction delivered into the payload of an ``Option`` that holds a
# tagged union (or a variant's own type): the option layers are stepped through
# before the union is tagged (see ``HirRunner._peel_option_layers``)
# ---------------------------------------------------------------------------


@struct()
class TU_OptHolder:
    u: Option[TU_A | TU_B]


@func()
def tu_opt_union_declared(c: spy_bool) -> i32:
    # a declared ``Option`` storage: the construction is deferred, and delivered
    # into the option's payload once the slot is committed
    u: Option[TU_A | TU_B] = TU_A(2) if c else None
    if (v := u) is not None and isinstance(a := v, TU_A):
        return a.x
    return -1


@func()
def tu_opt_union_comptime() -> i32:
    # a compile-time ``Option`` storage (a box): the same delivery into cells
    u: Comptime[Option[TU_A | TU_B]] = TU_A(3)
    if (v := u) is not None and isinstance(a := v, TU_A):
        return a.x
    return -1


@func()
def tu_opt_union_field() -> i32:
    # an ``Option`` union field of a struct being constructed
    h = TU_OptHolder(TU_A(6))
    if (v := h.u) is not None and isinstance(a := v, TU_A):
        return a.x
    return -1


# ---------------------------------------------------------------------------
# a union slot both branches of an ``if`` expression construct: every
# construction delivers the variant it is, at its own position (see
# ``HirRunner._is_union_delivery``)
# ---------------------------------------------------------------------------


@func()
def tu_pair(c: spy_bool) -> i32:
    u: TU_A | TU_B = TU_A(4) if c else TU_B(5)
    if isinstance(a := u, TU_A):
        return a.x
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_pair_option(c: spy_bool) -> i32:
    # the same, delivered into the payload of an ``Option`` storage
    u: Option[TU_A | TU_B] = TU_A(6) if c else TU_B(7)
    if (v := u) is not None:
        if isinstance(a := v, TU_A):
            return a.x
        if isinstance(b := v, TU_B):
            return b.y
    return -1


@func()
def tu_pair_comptime(c: spy_bool) -> i32:
    # a compile-time ``Box`` holds one value only, so two variants under a
    # *runtime* condition are still rejected: only a compile-time condition
    # folds the expression to a single construction (see ``tu_comptime``)
    u: Comptime[TU_A | TU_B] = TU_A(8) if c else TU_B(9)
    if isinstance(a := u, TU_A):
        return a.x
    return -1


class SpyTaggedUnionTest(TestCase):
    def test_tag_test_without_unwrap(self) -> None:
        self.assertEqual(tu_tag_of_make(0), 1)
        self.assertEqual(tu_tag_of_make(1), 2)

    def test_unwrap_binds_the_payload(self) -> None:
        self.assertEqual(tu_unwrap_make(0), 3)
        self.assertEqual(tu_unwrap_make(1), 4)

    def test_negated_test(self) -> None:
        self.assertEqual(tu_negated_make(0), 0)
        self.assertEqual(tu_negated_make(1), 1)

    def test_a_subset_union_converts(self) -> None:
        # the tag is remapped when the variants' order differs
        self.assertEqual(tu_widen_make(0), 3)
        self.assertEqual(tu_widen_make(1), 4)

    def test_match(self) -> None:
        self.assertEqual(tu_match_make(0), 1)
        self.assertEqual(tu_match_make(1), 2)

    def test_match_wildcard(self) -> None:
        self.assertEqual(tu_match_wildcard_make(0), 1)
        self.assertEqual(tu_match_wildcard_make(1), 2)

    def test_match_without_binding(self) -> None:
        # ``match u:`` (a plain-name subject) tests the tag only
        self.assertEqual(tu_match_tag_make(0), 1)
        self.assertEqual(tu_match_tag_make(1), 2)

    def test_match_binds_the_payload(self) -> None:
        # ``case T() as name`` binds the payload of the matched variant
        self.assertEqual(tu_match_bind_make(0), 3)
        self.assertEqual(tu_match_bind_make(1), 4)

    def test_match_binds_the_payload_before_a_wildcard(self) -> None:
        self.assertEqual(tu_match_bind_wildcard_make(0), 3)
        self.assertEqual(tu_match_bind_wildcard_make(1), -1)

    def test_match_subject_keeps_naming_the_union(self) -> None:
        # nothing is bound, so the subject name is still the union in the body
        self.assertEqual(tu_match_no_shadow_make(0), 3)
        self.assertEqual(tu_match_no_shadow_make(1), -2)

    def test_scalar_variants(self) -> None:
        # an untyped integer literal takes the first variant it fits
        self.assertEqual(tu_scalar(0), 5)
        self.assertEqual(tu_scalar(1), 100)

    def test_zero_sized_variants(self) -> None:
        self.assertEqual(tu_zst(0), 1)
        self.assertEqual(tu_zst(1), 2)

    def test_single_variant(self) -> None:
        self.assertEqual(tu_single(), 9)

    def test_pointer_variant(self) -> None:
        self.assertEqual(tu_ptr_driver(), 42)

    def test_compile_time_storage(self) -> None:
        self.assertEqual(tu_comptime(), 2)

    def test_compile_time_storage_reassigned(self) -> None:
        # switching to a runtime payload is rejected; a compile-time payload may
        # still switch the variant
        with self.assertRaises(CompileError):
            tu_comptime_reassigned(7)
        self.assertEqual(tu_comptime_reassigned_comptime(), 7)

    def test_typeof(self) -> None:
        self.assertEqual(tu_typeof_make(0), 1)

    def test_a_large_union_through_a_result_pointer(self) -> None:
        self.assertEqual(tu_large_use(0), 0)
        self.assertEqual(tu_large_use(1), 5)

    def test_a_tagged_union_field(self) -> None:
        self.assertEqual(tu_field(0), 6)
        self.assertEqual(tu_field(1), 0)

    def test_writing_a_tagged_union_field(self) -> None:
        self.assertEqual(tu_field_write(0), 8)
        self.assertEqual(tu_field_write(1), -1)

    def test_an_option_of_a_tagged_union(self) -> None:
        # the construction of a variant into an ``Option`` storage is delivered
        # into the option's payload: the option layer is stepped through before
        # the union is tagged
        self.assertEqual(tu_opt_union_declared(True), 2)
        self.assertEqual(tu_opt_union_declared(False), -1)
        self.assertEqual(tu_opt_union_comptime(), 3)
        self.assertEqual(tu_opt_union_field(), 6)

    def test_both_branches_construct_a_variant(self) -> None:
        self.assertEqual(tu_pair(True), 4)
        self.assertEqual(tu_pair(False), 5)
        self.assertEqual(tu_pair_option(True), 6)
        self.assertEqual(tu_pair_option(False), 7)

    def test_a_compile_time_union_holds_one_variant(self) -> None:
        # a box holds a single value: two variants under a runtime condition are
        # still the ``incompatible types`` error rather than a delivery of its own
        with self.assertRaises(CompileError):
            tu_pair_comptime(True)

    def test_a_type_that_is_not_a_variant_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tu_bad_variant()
        self.assertIn('not a variant', str(ctx.exception))

    def test_an_unsupported_match_pattern_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            tu_bad_pattern()

    def test_a_capture_pattern_is_rejected(self) -> None:
        # ``case e:`` binds the whole subject - only ``case T() as e`` is allowed
        with self.assertRaises(CompileError):
            tu_capture_pattern()


class SpyPythonSideUnionTest(TestCase):
    """Tagged unions from the Python side: a union value unwraps to the value of
    the variant it holds (it has no Python-side representation of its own)."""

    def test_a_union_unwraps_to_its_variant(self) -> None:
        a = tu_make(0)
        self.assertEqual(a.x, 3) # pyright: ignore
        b = tu_make(1)
        self.assertEqual(b.y, 4) # pyright: ignore

    def test_a_variant_instance_crosses_as_a_union(self) -> None:
        self.assertEqual(tu_unwrap(tu_make(0)), 3)  # pyright: ignore
        self.assertEqual(tu_unwrap(tu_make(1)), 4)  # pyright: ignore

    def test_a_constructed_variant_crosses_as_a_union(self) -> None:
        self.assertEqual(tu_unwrap(TU_A(9)), 9)  # pyright: ignore

    def test_a_scalar_union_unwraps_to_its_variant(self) -> None:
        self.assertEqual(tu_scalar(0), 5)
        self.assertEqual(tu_scalar(1), 100)


all_tests = [
    SpyTaggedUnionTest,
    SpyPythonSideUnionTest,
]

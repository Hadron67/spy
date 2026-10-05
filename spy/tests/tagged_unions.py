from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    i64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Comptime,
    Ptr,
    ref,
)
from .structs import Large

# ---------------------------------------------------------------------------
# tagged unions: ``A | B``, the tag test ``isinstance(u, A)`` / the unwrap
# ``isinstance(e := u, A)``, and ``match (e := u): case A(): ...`` (which binds
# the payload) / ``match u: case A(): ...`` (which tests only)
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
    match (_ := u):
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
    match (_ := u):
        case TU_A():
            return 1
        case _:
            return 2


@func()
def tu_match_wildcard_make(sel: i32) -> i32:
    return tu_match_wildcard(tu_make(sel))


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
    # a declared ``Comptime`` tagged union: its own place form (see
    # ``ComptimeTaggedUnionPtr``)
    u: Comptime[TU_A | TU_B] = TU_A(2)
    if isinstance(a := u, TU_A):
        return a.x
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
    match (_ := u):
        case TU_A(x):
            return x
        case _:
            return 0


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

    def test_a_type_that_is_not_a_variant_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tu_bad_variant()
        self.assertIn('not a variant', str(ctx.exception))

    def test_an_unsupported_match_pattern_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            tu_bad_pattern()


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

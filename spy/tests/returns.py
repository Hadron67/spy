from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    Comptime,
    Option,
    c128,
    func,
    i32,
    i64,
    mir,
    void,
)
from ..std.int import truncate
from .structs import Large, Small, struct_mirror

# ---------------------------------------------------------------------------
# multiple return values: a function annotated ``-> tuple[T1, T2, ...]``
# returns one value per element of the tuple.  The lowered function returns
# one of them by value and delivers every other one through a hidden result
# pointer (see ``sval.make_ret_spec``); the caller either destructures the
# call (``a, b = f()``) or keeps the packed results in a plain tuple variable
# (a tuple has a runtime representation now, so packing is a plain assignment;
# only when some result leaf is compile-time does the target have to be a
# ``Comptime`` variable, see doc/comptime-refactor.task.md 7.3).
# ---------------------------------------------------------------------------


def mir_signature(handle: Any) -> tuple[tuple[mir.Type, ...], mir.MayBeVoidType]:
    """The lowered MIR signature - the argument types and the return type - of
    the one specialization of a registered function.  The function must have
    been compiled already (by calling one of its wrappers)."""
    entry = handle.get_entry()
    assert len(entry.specs) == 1, 'the function was compiled more than once'
    instance = next(iter(entry.specs.values()))
    return tuple(instance.mir.args), instance.mir.ret_type


@func()
def min_max(a: i32, b: i32) -> tuple[i32, i32]:
    if a < b:
        return a, b
    return b, a


@func()
def use_min_max(a: i32, b: i32) -> i32:
    lo, hi = min_max(a, b)
    return lo * 100 + hi


@func()
def forward_min_max(a: i32, b: i32) -> tuple[i32, i32]:
    return min_max(a, b)


@func()
def use_forward_min_max(a: i32, b: i32) -> i32:
    lo, hi = forward_min_max(a, b)
    return lo * 100 + hi


@func()
def packed_multi(a: i32) -> i32:
    # every result leaf is a runtime value, so the packed result is a plain
    # tuple value: a plain variable can hold it
    packed = min_max(a, a + 1)
    lo, hi = packed
    return lo * 100 + hi


@func()
def mixed_multi(x: i32) -> tuple[i32, Large, Small]:
    return x, Large(x, x, x, x), Small(x, x + 1)


@func()
def use_mixed_multi(x: i32) -> i32:
    n, large, small = mixed_multi(x)
    return n * 10000 + small.b * 1000 + truncate(large.d, i32) * 100 + small.a


@func()
def big_first(x: i32) -> tuple[Large, i32]:
    return Large(x, x, x, x), x + 100


@func()
def use_big_first(x: i32) -> i32:
    large, n = big_first(x)
    return n * 1000 + truncate(large.d, i32)


@func()
def two_larges(x: i32) -> tuple[Large, Large]:
    return Large(x, x, x, x), Large(x + 1, x + 1, x + 1, x + 1)


@func()
def use_two_larges(x: i32) -> i32:
    p, q = two_larges(x)
    return truncate(p.d, i32) * 100 + truncate(q.d, i32)


@func()
def void_and_value(x: i32) -> tuple[void, i32]:
    return None, x + 1


@func()
def use_void_and_value(x: i32) -> i32:
    _, n = void_and_value(x)  # pyright: ignore[reportAssignmentType]
    return n


@func()
def python_pair(a: i64, b: i64) -> tuple[i64, i64]:
    # only ever called from Python, so the i32 wrappers above keep a single
    # specialization each
    return b, a


@func()
def nested_multi(x: i32) -> tuple[i32, tuple[i32, Large], Small]:
    return x, (x + 1, Large(x, x, x, x)), Small(x, x + 2)


@func()
def use_nested_multi(x: i32) -> i32:
    n, (m, large), small = nested_multi(x)
    return n * 100000 + m * 10000 + truncate(large.d, i32) * 100 + small.b


@func()
def forward_nested_multi(x: i32) -> tuple[i32, tuple[i32, Large], Small]:
    return nested_multi(x)


@func()
def use_forward_nested_multi(x: i32) -> i32:
    n, (m, large), small = forward_nested_multi(x)
    return n * 100000 + m * 10000 + truncate(large.d, i32) * 100 + small.b


@func()
def packed_nested_multi(x: i32) -> i32:
    # a packed multi-value result is a tuple of the result *elements*, a nested
    # result being a tuple value of its own: indexed nested (see
    # ``test_nested_packed_into_a_plain_variable``)
    packed = nested_multi(x)
    return packed[0] * 100000 + packed[1][0] * 10000 + truncate(packed[1][1].d, i32) * 100 + packed[2].b


@func()
def nested_group_first(x: i32) -> tuple[tuple[Large, i32], i32]:
    return (Large(x, x, x, x), x + 1), x + 2


@func()
def use_nested_group_first(x: i32) -> i32:
    (large, n), m = nested_group_first(x)
    return truncate(large.d, i32) * 1000 + n * 100 + m


@func()
def python_nested_pair(a: i64, b: i64) -> tuple[i64, tuple[i64, i64], i64]:
    return a, (b, a + b), b


@func()
def group_target_held_whole(x: i32) -> i32:
    # a group the caller does not unpack further: the inner ``tuple[...]`` is a
    # runtime value of its own now, so it can be held in a plain variable (it
    # used to have to be unpacked into a tuple target)
    _n, inner, _small = nested_multi(x)
    return inner[0] + truncate(inner[1].d, i32)


@func()
def bad_nested_arity(x: i32) -> i32:
    _n, (_m, _large), _small, _extra = nested_multi(x)  # pyright: ignore
    return x


@func()
def bad_nested_ellipsis(x: i32) -> tuple[i32, tuple[i32, ...]]:
    return x  # pyright: ignore[reportReturnType]


@func()
def packed_multi_assign(x: i32) -> i32:
    # a packed multi-value result into a plain variable: allowed now that the
    # tuple has a runtime representation (only a compile-time result leaf would
    # force a ``Comptime`` variable)
    packed = min_max(x, x + 1)
    return packed[0] * 100 + packed[1]


@func()
def bad_multi_arity(x: i32) -> i32:
    _a, _b, _c = min_max(x, x + 1)  # pyright: ignore
    return x


@func()
def bad_ellipsis_return(x: i32) -> tuple[i32, ...]:
    return x  # pyright: ignore[reportReturnType]


# ---------------------------------------------------------------------------
# the same multi-value returns without a declared annotation: the arity and the
# types are inferred from the body's result location, exactly like a
# single-value return (see ``interp``).  A tuple literal, a forwarded
# multi-value call and a packed tuple value all become the element places
# of the result location, which then takes the same convention an annotation
# would have fixed.
# ---------------------------------------------------------------------------


@func()
def infer_min_max(a: i32, b: i32):
    # a tuple literal, written on two paths
    if a < b:
        return a, b
    return b, a


@func()
def use_infer_min_max(a: i32, b: i32) -> i32:
    lo, hi = infer_min_max(a, b)
    return lo * 100 + hi


@func()
def infer_forward(a: i32, b: i32):
    # forwarding another function's multi-value result
    return min_max(a, b)


@func()
def use_infer_forward(a: i32, b: i32) -> i32:
    lo, hi = infer_forward(a, b)
    return lo * 100 + hi


@func()
def infer_packed(a: i32, b: i32):
    # a packed tuple value returned from a function with no return annotation
    packed = min_max(a, b)
    return packed


@func()
def use_infer_packed(a: i32, b: i32) -> i32:
    lo, hi = infer_packed(a, b)
    return lo * 100 + hi


@func()
def infer_mixed(x: i32):
    return x, Large(x, x, x, x), Small(x, x + 1)


@func()
def use_infer_mixed(x: i32) -> i32:
    n, large, small = infer_mixed(x)
    return n * 10000 + small.b * 1000 + truncate(large.d, i32) * 100 + small.a


@func()
def infer_nested(x: i32):
    return x, (x + 1, Large(x, x, x, x)), Small(x, x + 2)


@func()
def use_infer_nested(x: i32) -> i32:
    n, (m, large), small = infer_nested(x)
    return n * 100000 + m * 10000 + truncate(large.d, i32) * 100 + small.b


@func()
def infer_group_first(x: i32):
    return (Large(x, x, x, x), x + 1), x + 2


@func()
def use_infer_group_first(x: i32) -> i32:
    (large, n), m = infer_group_first(x)
    return truncate(large.d, i32) * 1000 + n * 100 + m


@func()
def infer_void_and_value(x: i32):
    return None, x + 1


@func()
def use_infer_void_and_value(x: i32) -> i32:
    _, n = infer_void_and_value(x)  # pyright: ignore[reportAssignmentType]
    return n


@func()
def bad_infer_conflict(a: i32) -> i32:
    # one path returns a tuple, another a scalar: the result location has no
    # single type (pyright cannot tell, hence the ignore)
    if a < 0:
        return a, a  # pyright: ignore
    return a


@func()
def bad_infer_falloff(a: i32):
    # the body falls off its end after a tuple return (pyright cannot tell)
    if a < 0:
        return a, a


# ---------------------------------------------------------------------------
# a packed tuple is a runtime aggregate: its element count is known with
# ``len`` (a compile-time integer), and its i-th element is a *place* read and
# written with ``t[i]`` (a compile-time integer index, bounds-checked).  The
# packed tuple mirrors the *result elements*: one element per result element, a
# nested result being a tuple value of its own (so ``packed[1][0]`` picks into
# the middle group of ``nested_multi``), and its length is the top-level arity.
# ---------------------------------------------------------------------------


@func()
def packed_len(x: i32) -> i32:
    packed = min_max(x, x + 1)
    return len(packed)


@func()
def nested_packed_len(x: i32) -> i32:
    # one element per result element: the nested result is a single one of the
    # three
    packed = nested_multi(x)
    return len(packed)


@func()
def packed_index(x: i32) -> i32:
    packed = min_max(x, x + 100)
    return packed[0] * 1000 + packed[1]


@func()
def packed_nested_index(x: i32) -> i32:
    # a nested packed tuple: element 0 is the leading scalar, element 1 the inner
    # tuple (indexed again)
    packed = nested_multi(x)
    return packed[0] * 100 + packed[1][0]


@func()
def nested_tail(x: i32) -> tuple[i32, i32, tuple[i32, i32]]:
    return x, x + 1, (x + 2, x + 3)


@func()
def use_packed_nested_tail(x: i32) -> i32:
    # the packed result keeps the nesting: ``ret[2]`` is the inner tuple
    ret = nested_tail(x)
    return ret[0] * 1000 + ret[1] * 100 + ret[2][0] * 10 + ret[2][1]


@func()
def packed_nested_tail_len(x: i32) -> i32:
    ret = nested_tail(x)
    return len(ret)


@func()
def repacked_nested(x: i32) -> i32:
    # the first assignment commits the slot (its ``CommitSlot`` runs at the end of
    # the statement), so the packed result of the second call is delivered into
    # an already materialized (nested) tuple place
    ret = nested_tail(x)
    ret = nested_tail(x + 10)
    return ret[0] * 100 + ret[2][0]


@func()
def write_packed_nested(x: i32) -> i32:
    ret = nested_tail(x)
    ret[2][1] = 99  # pyright: ignore
    return ret[2][0] * 1000 + ret[2][1]


@func()
def nested_struct(x: i32) -> tuple[i32, i32, tuple[Large, Small]]:
    # the nested group is a tuple of two structs: the packed result keeps it
    # whole, so both are reached through ``ret[2]``
    return x, x + 1, (Large(x, x, x, x), Small(x + 2, x + 3))


@func()
def use_packed_nested_struct(x: i32) -> i32:
    ret = nested_struct(x)
    return ret[0] * 10000 + ret[1] * 1000 + truncate(ret[2][0].d, i32) * 10 + ret[2][1].b


@func()
def nested_struct_len(x: i32) -> i32:
    ret = nested_struct(x)
    return len(ret)


@func()
def nested_groups(x: i32) -> tuple[tuple[Large, i32], i32, tuple[Small, i32]]:
    # two nested groups, each holding a struct, and one of them leading
    return (Large(x, x, x, x), x + 1), x + 2, (Small(x + 3, x + 4), x + 5)


@func()
def use_packed_nested_groups(x: i32) -> i32:
    ret = nested_groups(x)
    return truncate(ret[0][0].d, i32) * 100000 + ret[0][1] * 10000 + ret[1] * 100 + ret[2][0].b


@func()
def use_destructured_nested_groups(x: i32) -> i32:
    # the same shape destructured: each struct group is bound whole
    (large, n), m, (small, k) = nested_groups(x)
    return truncate(large.d, i32) * 1000 + n * 100 + m * 10 + small.b + k


@func()
def nested_deep_struct(x: i32) -> tuple[i32, tuple[Small, tuple[Large, i32]]]:
    # two levels of nesting, the innermost group ending in a struct
    return x, (Small(x + 1, x + 2), (Large(x, x, x, x), x + 3))


@func()
def use_packed_nested_deep_struct(x: i32) -> i32:
    ret = nested_deep_struct(x)
    return (
        ret[0] * 100000
        + ret[1][0].b * 10000
        + truncate(ret[1][1][0].d, i32) * 1000
        + ret[1][1][1] * 100
    )


@func()
def packed_index_write(x: i32) -> i32:
    packed = min_max(x, x + 1)
    packed[0] = 100  # pyright: ignore
    return packed[0] + packed[1]


@func()
def bad_packed_index_oob(x: i32) -> i32:
    packed = min_max(x, x + 1)
    return packed[5]  # pyright: ignore


@func()
def bad_packed_index_runtime(x: i32) -> i32:
    packed = min_max(x, x + 1)
    return packed[x]


@func()
def bad_packed_index_negative(x: i32) -> i32:
    packed = min_max(x, x + 1)
    return packed[-1]


class SpyMultiReturnTest(TestCase):
    """A function annotated ``-> tuple[T1, T2, ...]`` returns several values:
    the lowered function returns one of them by value and delivers every other
    one through a hidden result pointer (see ``sval.make_ret_spec``), and the
    caller destructures the call or keeps the packed results in a plain tuple
    variable."""

    def test_destructuring_two_scalars(self) -> None:
        self.assertEqual(use_min_max(2, 7), 207)
        self.assertEqual(use_min_max(9, 3), 309)

    def test_lowered_signature_of_two_scalars(self) -> None:
        # the first result that fits in registers is returned by value, the
        # other one through a result pointer: ``fn(i32, i32, *i32) -> i32``
        self.assertEqual(use_min_max(2, 7), 207)
        args, ret = mir_signature(min_max)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(args, (i32_mir, i32_mir, mir.PointerType(i32_mir, False)))
        self.assertEqual(ret, i32_mir)

    def test_forwarding_a_multi_value_call(self) -> None:
        # ``return f()`` where both functions return two values writes each
        # result into the enclosing function's own result location
        self.assertEqual(use_forward_min_max(2, 7), 207)

    def test_packed_into_a_plain_variable(self) -> None:
        self.assertEqual(packed_multi(2), 203)

    def test_python_side_call(self) -> None:
        # the Python side allocates the storage of the result pointers and
        # returns the values as a tuple
        self.assertEqual(python_pair(2, 7), (7, 2))

    def test_nested_destructuring(self) -> None:
        # the middle result is a ``tuple[...]`` of its own: the leaves are
        # delivered separately and the caller regroups them
        self.assertEqual(use_nested_multi(3), 340305)

    def test_lowered_signature_of_a_nested_return(self) -> None:
        # the leaves are delivered depth first: the first scalar returns by
        # value, the other scalar and both aggregates through a pointer each
        self.assertEqual(use_nested_multi(3), 340305)
        args, ret = mir_signature(nested_multi)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(args[0], i32_mir)
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(i32_mir, False),
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(struct_mirror(Small), False),
            ),
        )

    def test_a_leading_group_still_returns_a_scalar_by_value(self) -> None:
        self.assertEqual(use_nested_group_first(3), 3405)
        args, ret = mir_signature(nested_group_first)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(i32_mir, False),
            ),
        )

    def test_forwarding_a_nested_multi_value_call(self) -> None:
        self.assertEqual(use_forward_nested_multi(3), 340305)

    def test_nested_packed_into_a_plain_variable(self) -> None:
        self.assertEqual(packed_nested_multi(3), 340305)

    def test_python_side_call_of_a_nested_return(self) -> None:
        self.assertEqual(python_nested_pair(2, 7), (2, (7, 9), 7))

    def test_a_group_held_in_a_plain_variable(self) -> None:
        # a nested tuple result kept whole in a plain variable (a tuple is a
        # runtime value now); it used to have to be unpacked into a tuple target
        self.assertEqual(group_target_held_whole(3), 7)

    def test_nested_arity_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_nested_arity(3)

    def test_a_nested_varying_number_of_results_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_nested_ellipsis(3)

    def test_aggregates_go_through_result_pointers(self) -> None:
        # a scalar returns by value, a large struct and a small one through a
        # result pointer each: ``fn(i32, *Large, *Small) -> i32``
        self.assertEqual(use_mixed_multi(3), 34303)
        args, ret = mir_signature(mixed_multi)
        self.assertEqual(ret, mir.IntType(32, True))
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(struct_mirror(Small), False),
            ),
        )

    def test_a_large_leading_result_still_returns_the_scalar(self) -> None:
        # the by-value result need not be the first one
        self.assertEqual(use_big_first(3), 103003)
        args, ret = mir_signature(big_first)
        self.assertEqual(ret, mir.IntType(32, True))
        self.assertEqual(
            args[1:],
            (mir.PointerType(struct_mirror(Large), False),),
        )

    def test_every_result_through_a_pointer_returns_void(self) -> None:
        self.assertEqual(use_two_larges(3), 304)
        args, ret = mir_signature(two_larges)
        self.assertIs(ret, mir.VOID)
        self.assertEqual(len(args), 3)

    def test_a_zero_sized_result_is_the_unit_value(self) -> None:
        # a zero-sized result has no place of its own: it is delivered as its
        # unit value, and the other result still returns by value
        self.assertEqual(use_void_and_value(3), 4)
        args, ret = mir_signature(void_and_value)
        self.assertEqual(args, (mir.IntType(32, True),))
        self.assertEqual(ret, mir.IntType(32, True))

    def test_packing_into_a_plain_variable(self) -> None:
        # every result is a runtime value, so the packed tuple goes into a plain
        # variable (the old "only a ``Comptime`` variable" rule is gone)
        self.assertEqual(packed_multi_assign(3), 304)

    def test_arity_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_multi_arity(3)

    def test_a_varying_number_of_results_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_ellipsis_return(3)

    def test_len_of_a_packed_tuple(self) -> None:
        self.assertEqual(packed_len(3), 2)

    def test_len_of_a_nested_packed_tuple(self) -> None:
        # the length is the top-level arity, not the number of leaves
        self.assertEqual(nested_packed_len(3), 3)

    def test_indexing_a_packed_tuple(self) -> None:
        self.assertEqual(packed_index(3), 3103)

    def test_indexing_a_nested_packed_tuple(self) -> None:
        self.assertEqual(packed_nested_index(3), 304)

    def test_a_packed_nested_tail_keeps_its_nesting(self) -> None:
        # ``ret[2]`` is the inner ``tuple[i32, i32]`` of the packed result
        self.assertEqual(use_packed_nested_tail(1), 1234)

    def test_len_of_a_packed_nested_tail(self) -> None:
        self.assertEqual(packed_nested_tail_len(1), 3)

    def test_repacking_a_committed_nested_place(self) -> None:
        # the target slot is committed by the first assignment, so the second
        # packed call delivers into a materialized place
        self.assertEqual(repacked_nested(1), 1113)

    def test_writing_a_nested_packed_tuple_element(self) -> None:
        self.assertEqual(write_packed_nested(1), 3099)

    def test_a_packed_nested_tuple_with_structs(self) -> None:
        # the nested group holds the two structs whole: ``ret[2][0]``/``ret[2][1]``
        self.assertEqual(use_packed_nested_struct(3), 34036)
        self.assertEqual(nested_struct_len(3), 3)

    def test_packing_nested_groups_with_structs(self) -> None:
        self.assertEqual(use_packed_nested_groups(1), 120305)

    def test_destructuring_nested_groups_with_structs(self) -> None:
        self.assertEqual(use_destructured_nested_groups(1), 1241)

    def test_packing_a_deeply_nested_struct_tuple(self) -> None:
        self.assertEqual(use_packed_nested_deep_struct(2), 242500)

    def test_lowered_signature_of_a_nested_struct_return(self) -> None:
        # the leaves are delivered depth first across the nesting: the first
        # scalar returns by value, the large struct and every other leaf through
        # a result pointer each
        self.assertEqual(use_packed_nested_groups(1), 120305)
        args, ret = mir_signature(nested_groups)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(args[0], i32_mir)
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(i32_mir, False),
                mir.PointerType(struct_mirror(Small), False),
                mir.PointerType(i32_mir, False),
            ),
        )

    def test_writing_a_packed_tuple_element(self) -> None:
        self.assertEqual(packed_index_write(3), 104)

    def test_a_packed_tuple_index_out_of_bounds_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_packed_index_oob(3)

    def test_a_packed_tuple_index_must_be_a_compile_time_integer(self) -> None:
        with self.assertRaises(CompileError):
            bad_packed_index_runtime(3)

    def test_a_negative_packed_tuple_index_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_packed_index_negative(3)


class SpyInferredMultiReturnTest(TestCase):
    """A function with no return annotation whose body returns several values:
    the arity and the types are inferred from the body's result location, and
    the lowered form follows the same convention as an annotated one."""

    def test_a_tuple_literal(self) -> None:
        self.assertEqual(use_infer_min_max(2, 7), 207)
        self.assertEqual(use_infer_min_max(9, 3), 309)

    def test_the_lowered_signature_matches_the_declared_one(self) -> None:
        # an inferred multi-value return lowers exactly like the annotated one:
        # one scalar by value, the other through a result pointer, so
        # ``fn(i32, i32, *i32) -> i32``
        self.assertEqual(use_infer_min_max(2, 7), 207)
        self.assertEqual(use_infer_forward(2, 7), 207)
        i32_mir = mir.IntType(32, True)
        for fn in (infer_min_max, infer_forward):
            args, ret = mir_signature(fn)
            self.assertEqual(args, (i32_mir, i32_mir, mir.PointerType(i32_mir, False)))
            self.assertEqual(ret, i32_mir)

    def test_forwarding_a_multi_value_call(self) -> None:
        self.assertEqual(use_infer_forward(2, 7), 207)

    def test_packed_into_a_plain_variable(self) -> None:
        self.assertEqual(use_infer_packed(2, 7), 207)

    def test_mixed_aggregate_results(self) -> None:
        self.assertEqual(use_infer_mixed(3), 34303)

    def test_nested_return(self) -> None:
        self.assertEqual(use_infer_nested(3), 340305)

    def test_a_leading_group(self) -> None:
        self.assertEqual(use_infer_group_first(3), 3405)

    def test_a_zero_sized_result(self) -> None:
        self.assertEqual(use_infer_void_and_value(3), 4)

    def test_a_conflicting_result_shape_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_infer_conflict(3)

    def test_falling_off_after_a_multi_value_return_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_infer_falloff(3)


# ---------------------------------------------------------------------------
# a function with no return annotation whose body returns a *compile-time*
# value of a type that has a runtime representation (a complex, a struct, an
# option): the result location keeps the value in compile-time storage (a
# ``Box``), yet the inferred convention delivers it by value or through a
# result pointer - so the value is materialized at the return site (see
# ``interp.HirRunner._finish_function``).  A scalar is unaffected (its lowering
# is a constant); an aggregate needs the ``InsertValue`` chain building it.
# ---------------------------------------------------------------------------


@func()
def infer_comptime_complex():
    z: Comptime = 1 + 2j
    return z


@func()
def annotate_comptime_complex() -> c128:
    z: Comptime = 1 + 2j
    return z


@func()
def use_infer_comptime_complex() -> c128:
    # the compile-time aggregate crosses an internal call site too
    return infer_comptime_complex()


@func()
def infer_comptime_small():
    s: Comptime = Small(1, 2)
    return s


@func()
def use_infer_comptime_small() -> i32:
    s = infer_comptime_small()
    return s.a * 10 + s.b


@func()
def infer_comptime_large():
    # outgrows the by-value limit: delivered through a hidden result pointer
    l: Comptime = Large(1, 2, 3, 4)
    return l


@func()
def use_infer_comptime_large() -> i64:
    l = infer_comptime_large()
    return l.a * 1000 + l.d


@func()
def infer_comptime_option():
    o: Comptime[Option[i32]] = 7
    return o


class SpyInferredComptimeReturnTest(TestCase):
    """A function with no return annotation whose body returns a compile-time
    value whose type has a runtime representation: the inferred convention
    delivers it as a runtime value (materialized from its compile-time storage).
    """

    def test_a_complex_value(self) -> None:
        self.assertEqual(infer_comptime_complex(), 1 + 2j)
        self.assertEqual(use_infer_comptime_complex(), 1 + 2j)

    def test_the_lowered_signature_matches_the_annotated_one(self) -> None:
        # an inferred compile-time aggregate return lowers exactly like the
        # declared one (the complex is returned by value)
        infer_comptime_complex()
        annotate_comptime_complex()
        self.assertEqual(
            mir_signature(infer_comptime_complex),
            mir_signature(annotate_comptime_complex),
        )

    def test_a_struct_value(self) -> None:
        self.assertEqual(use_infer_comptime_small(), 12)
        s = infer_comptime_small()
        self.assertEqual((s.a, s.b), (1, 2))

    def test_a_large_struct_value(self) -> None:
        self.assertEqual(use_infer_comptime_large(), 1004)
        l = infer_comptime_large()
        self.assertEqual((l.a, l.b, l.c, l.d), (1, 2, 3, 4))

    def test_a_present_option_value(self) -> None:
        self.assertEqual(infer_comptime_option(), 7)


all_tests = [
    SpyMultiReturnTest,
    SpyInferredMultiReturnTest,
    SpyInferredComptimeReturnTest,
]

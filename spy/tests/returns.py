from typing import Any
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    i64,
    mir,
    void,
)
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
)
from .structs import Large, Small, struct_mirror

# ---------------------------------------------------------------------------
# multiple return values: a function annotated ``-> tuple[T1, T2, ...]``
# returns one value per element of the tuple.  The lowered function returns
# one of them by value and delivers every other one through a hidden result
# pointer (see ``sval.make_ret_spec``); the caller either destructures the
# call (``a, b = f()``) or keeps the packed results in a ``Comptime`` variable.
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
def comptime_multi(a: i32) -> i32:
    # the packed result of a call has no runtime type of its own: a ``Comptime``
    # variable is the only place that can hold it
    packed: Comptime = min_max(a, a + 1)
    lo, hi = packed
    return lo * 100 + hi


@func()
def mixed_multi(x: i32) -> tuple[i32, Large, Small]:
    return x, Large(x, x, x, x), Small(x, x + 1)


@func()
def use_mixed_multi(x: i32) -> i32:
    n, large, small = mixed_multi(x)
    return n * 10000 + small.b * 1000 + large.d * 100 + small.a


@func()
def big_first(x: i32) -> tuple[Large, i32]:
    return Large(x, x, x, x), x + 100


@func()
def use_big_first(x: i32) -> i32:
    large, n = big_first(x)
    return n * 1000 + large.d


@func()
def two_larges(x: i32) -> tuple[Large, Large]:
    return Large(x, x, x, x), Large(x + 1, x + 1, x + 1, x + 1)


@func()
def use_two_larges(x: i32) -> i32:
    p, q = two_larges(x)
    return p.d * 100 + q.d


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
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def forward_nested_multi(x: i32) -> tuple[i32, tuple[i32, Large], Small]:
    return nested_multi(x)


@func()
def use_forward_nested_multi(x: i32) -> i32:
    n, (m, large), small = forward_nested_multi(x)
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def comptime_nested_multi(x: i32) -> i32:
    packed: Comptime = nested_multi(x)
    n, (m, large), small = packed
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def nested_group_first(x: i32) -> tuple[tuple[Large, i32], i32]:
    return (Large(x, x, x, x), x + 1), x + 2


@func()
def use_nested_group_first(x: i32) -> i32:
    (large, n), m = nested_group_first(x)
    return large.d * 1000 + n * 100 + m


@func()
def python_nested_pair(a: i64, b: i64) -> tuple[i64, tuple[i64, i64], i64]:
    return a, (b, a + b), b


@func()
def bad_group_target(x: i32) -> i32:
    # a group the caller does not unpack into a tuple target: packing it into
    # a single place of its own is not supported yet (pyright cannot tell)
    _n, _inner, _small = nested_multi(x)
    return x


@func()
def bad_nested_arity(x: i32) -> i32:
    _n, (_m, _large), _small, _extra = nested_multi(x)  # pyright: ignore
    return x


@func()
def bad_nested_ellipsis(x: i32) -> tuple[i32, tuple[i32, ...]]:
    return x  # pyright: ignore[reportReturnType]


@func()
def bad_multi_assign(x: i32) -> i32:
    # a packed result into a variable that is not ``Comptime``: pyright cannot
    # tell, hence the ignore
    _packed = min_max(x, x + 1)
    return x


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
# multi-value call and a packed ``Comptime`` value all become the element places
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
    # a packed ``Comptime`` value (pyright cannot tell)
    packed: Comptime = min_max(a, b)
    return packed  # pyright: ignore


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
    return n * 10000 + small.b * 1000 + large.d * 100 + small.a


@func()
def infer_nested(x: i32):
    return x, (x + 1, Large(x, x, x, x)), Small(x, x + 2)


@func()
def use_infer_nested(x: i32) -> i32:
    n, (m, large), small = infer_nested(x)
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def infer_group_first(x: i32):
    return (Large(x, x, x, x), x + 1), x + 2


@func()
def use_infer_group_first(x: i32) -> i32:
    (large, n), m = infer_group_first(x)
    return large.d * 1000 + n * 100 + m


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
# a packed tuple is a compile-time place tree: its element count is known with
# ``len`` (a compile-time integer), and its i-th element is a *place* read and
# written with ``t[i]`` (a compile-time integer index, bounds-checked).
# ---------------------------------------------------------------------------


@func()
def packed_len(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 1)
    return len(packed)  # pyright: ignore


@func()
def nested_packed_len(x: i32) -> i32:
    packed: Comptime = nested_multi(x)
    return len(packed)  # pyright: ignore


@func()
def packed_index(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 100)
    return packed[0] * 1000 + packed[1]  # pyright: ignore


@func()
def packed_nested_index(x: i32) -> i32:
    packed: Comptime = nested_multi(x)
    return packed[0] * 100 + packed[1][0]  # pyright: ignore


@func()
def packed_index_write(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 1)
    packed[0] = 100  # pyright: ignore
    return packed[0] + packed[1]  # pyright: ignore


@func()
def bad_packed_index_oob(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 1)
    return packed[5]  # pyright: ignore


@func()
def bad_packed_index_runtime(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 1)
    return packed[x]  # pyright: ignore


@func()
def bad_packed_index_negative(x: i32) -> i32:
    packed: Comptime = min_max(x, x + 1)
    return packed[-1]  # pyright: ignore


class SpyMultiReturnTest(TestCase):
    """A function annotated ``-> tuple[T1, T2, ...]`` returns several values:
    the lowered function returns one of them by value and delivers every other
    one through a hidden result pointer (see ``sval.make_ret_spec``), and the
    caller destructures the call or keeps the packed results in a ``Comptime``
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

    def test_packed_into_a_comptime_variable(self) -> None:
        self.assertEqual(comptime_multi(2), 203)

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

    def test_nested_packed_into_a_comptime_variable(self) -> None:
        self.assertEqual(comptime_nested_multi(3), 340305)

    def test_python_side_call_of_a_nested_return(self) -> None:
        self.assertEqual(python_nested_pair(2, 7), (2, (7, 9), 7))

    def test_a_group_target_that_is_not_a_tuple_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_group_target(3)

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

    def test_packing_into_a_plain_variable_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_multi_assign(3)

    def test_arity_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_multi_arity(3)

    def test_a_varying_number_of_results_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_ellipsis_return(3)

    def test_len_of_a_packed_tuple(self) -> None:
        self.assertEqual(packed_len(3), 2)

    def test_len_of_a_nested_packed_tuple(self) -> None:
        self.assertEqual(nested_packed_len(3), 3)

    def test_indexing_a_packed_tuple(self) -> None:
        self.assertEqual(packed_index(3), 3103)

    def test_indexing_a_nested_packed_tuple(self) -> None:
        self.assertEqual(packed_nested_index(3), 304)

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

    def test_packed_into_a_comptime_variable(self) -> None:
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


all_tests = [
    SpyMultiReturnTest,
    SpyInferredMultiReturnTest,
]

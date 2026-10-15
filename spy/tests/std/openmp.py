"""Tests for :mod:`spy.std.openmp`: ``parallel_loop`` runs its kernel across the
OpenMP thread team through the ``__kmpc_*`` runtime entry points.

Every thread writes a distinct index, so the sum over the buffer is
deterministic whatever the schedule; a length of zero must run nothing.

The OpenMP runtime is loaded up front (see ``load_openmp_runtime``); when it
cannot be found the tests are skipped, since the JIT would not resolve the
``__kmpc_*`` symbols.
"""

from typing import Literal, cast
from unittest import TestCase

from ...compiler import CompileError, func, i32, i64, u8
from ...compiler.syntax import Array, ref
from ...std.core import arr_slice, coerce, undefined
from ...std.openmp import load_openmp_runtime, parallel_loop

_LOAD_ERROR: str | None = None
try:
    load_openmp_runtime()
except RuntimeError as error:
    _LOAD_ERROR = str(error)


@func()
def parallel_fill_i32(n: i32) -> i64:
    buf: Array[i32, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i32):
        out[i] = i

    parallel_loop(n, kernel)

    total: i64 = 0
    j: i32 = 0
    while j < n:
        total = total + cast(i64, out[j])
        j = j + 1
    return total


@func()
def parallel_fill_i64(n: i64) -> i64:
    buf: Array[i64, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i64):
        out[i] = i

    parallel_loop(n, kernel)

    total: i64 = 0
    j: i64 = 0
    while j < n:
        total = total + out[j]
        j = j + 1
    return total


@func()
def parallel_unsupported_index(n: u8) -> u8:
    # the index type picks the runtime entry point, and only the four integer
    # widths have one: resolving this is a compile error
    def kernel(i: u8) -> None:
        pass

    parallel_loop(n, kernel)
    return n


@func()
def parallel_sum(n: i32) -> i64:
    # a single-valued reduction: each thread folds its slice of ``i``, the
    # partials are combined, and the sum comes back as the loop's result
    def kernel(i: i32) -> i64:
        return cast(i64, i)

    def add(a: i64, b: i64) -> i64:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_sum_count(n: i32) -> i64:
    # a multi-valued reduction: the kernel returns a (sum, count) pair which is
    # reduced element-wise, and ``parallel_loop`` delivers the pair as a
    # multi-value result (the reducer's spread operands are unannotated, so they
    # take the shape the call gives them)
    def kernel(i: i32):
        return coerce(i, i64), coerce(1, i64)

    def add(a, b):
        return a[0] + b[0], a[1] + b[1]

    total, count = parallel_loop(n, kernel, add)
    return total * 1000 + count


@func()
def parallel_reduce_zero(n: i32) -> i32:
    # a zero-length reducing loop runs no iteration: nothing is reduced and the
    # accumulator keeps its initial (undefined) value.  A successful compile and
    # run is the assertion.
    def kernel(i: i32) -> i64:
        return cast(i64, i)

    def add(a: i64, b: i64) -> i64:
        return a + b

    zero: i32 = 0
    parallel_loop(zero, kernel, add)
    return n


@func()
def parallel_missing_reducer(n: i32) -> i64:
    # a value-returning kernel with no ``reduce_op`` has nothing to combine the
    # results with: resolving this is a compile error
    def kernel(i: i32) -> i64:
        return cast(i64, i)

    return parallel_loop(n, kernel)


@func()
def parallel_void_with_reducer(n: i32) -> i32:
    # a reducer on a kernel that returns nothing is ignored (the loop just runs
    # its body): this compiles
    def kernel(i: i32) -> None:
        pass

    def add(a: i32, b: i32) -> i32:
        return a + b

    parallel_loop(n, kernel, add) # pyright: ignore
    return n


class SpyOpenMPTest(TestCase):
    """``std.openmp.parallel_loop``: the static-schedule parallel loop."""

    def setUp(self) -> None:
        if _LOAD_ERROR is not None:
            self.skipTest(_LOAD_ERROR)

    def test_a_loop_body_runs_over_every_index_once(self) -> None:
        # sum(0..999)
        self.assertEqual(parallel_fill_i32(1000), 499500)

    def test_a_zero_length_loop_runs_nothing(self) -> None:
        self.assertEqual(parallel_fill_i32(0), 0)

    def test_a_single_iteration(self) -> None:
        self.assertEqual(parallel_fill_i32(1), 0)

    def test_a_64_bit_index_selects_the_8_variant(self) -> None:
        self.assertEqual(parallel_fill_i64(1000), 499500)

    def test_an_unsupported_index_type_is_a_compile_error(self) -> None:
        with self.assertRaises(CompileError):
            parallel_unsupported_index(3)

    def test_a_single_valued_reduction(self) -> None:
        # sum(0..999)
        self.assertEqual(parallel_sum(1000), 499500)

    def test_a_single_iteration_reduction(self) -> None:
        self.assertEqual(parallel_sum(1), 0)

    def test_a_multi_valued_reduction(self) -> None:
        # sum(0..999) paired with the count
        self.assertEqual(parallel_sum_count(1000), 499500 * 1000 + 1000)

    def test_a_zero_length_reducing_loop_runs_nothing(self) -> None:
        self.assertEqual(parallel_reduce_zero(7), 7)

    def test_a_value_returning_kernel_needs_a_reducer(self) -> None:
        with self.assertRaises(CompileError):
            parallel_missing_reducer(3)

    def test_a_reducer_on_a_void_kernel_is_ignored(self) -> None:
        self.assertEqual(parallel_void_with_reducer(3), 3)


all_tests = [
    SpyOpenMPTest,
]

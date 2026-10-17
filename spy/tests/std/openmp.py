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

from ...compiler import CompileError, c64, c128, f32, f64, func, i32, i64, u8, usize
from ...compiler.syntax import Array, ref
from ...std.core import arr_slice, coerce, undefined
from ...std.int import int_cast, int_to_float
from ...std.openmp import parallel_loop


@func()
def parallel_fill_i32(n: i32) -> i64:
    buf: Array[i32, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i32):
        out[int_cast(i, usize)] = i

    parallel_loop(n, kernel)

    total: i64 = 0
    j: i32 = 0
    while j < n:
        total = total + cast(i64, out[int_cast(j, usize)])
        j = j + 1
    return total


@func()
def parallel_fill_i64(n: i64) -> i64:
    buf: Array[i64, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i64):
        out[int_cast(i, usize)] = i

    parallel_loop(n, kernel)

    total: i64 = 0
    j: i64 = 0
    while j < n:
        total = total + out[int_cast(j, usize)]
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
        return coerce(i, i64)

    def add(a: i64, b: i64) -> i64:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_sum_count(n: i32) -> i64:
    # a multi-valued reduction: the kernel returns a (sum, count) pair which is
    # reduced element-wise, and ``parallel_loop`` delivers the pair as a
    # multi-value result (the reducer receives the two tuples, one per operand -
    # an unannotated parameter takes the shape the call gives it)
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
        return coerce(i, i64)

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
        return coerce(i, i64)

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


@func()
def parallel_sum_f64(n: i32) -> f64:
    # a floating-point reduction: the values and the accumulator are ``f64``
    def kernel(i: i32) -> f64:
        return int_to_float(i, f64)

    def add(a: f64, b: f64) -> f64:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_sum_f32(n: i32) -> f32:
    # a 32-bit float reduction (every partial sum is an integer below 2**24, so
    # the result is exact whatever the schedule)
    def kernel(i: i32) -> f32:
        return int_to_float(i, f32)

    def add(a: f32, b: f32) -> f32:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_sum_c128(n: i32) -> c128:
    # a complex128 reduction: each kernel result is ``i + 1j``
    def kernel(i: i32) -> c128:
        z: c128 = int_to_float(i, f64)
        return z + 1j

    def add(a: c128, b: c128) -> c128:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_sum_c64(n: i32) -> c64:
    # a complex64 reduction: each kernel result is the real value ``i``
    def kernel(i: i32) -> c64:
        z: c64 = int_to_float(i, f32)
        return z

    def add(a: c64, b: c64) -> c64:
        return a + b

    return parallel_loop(n, kernel, add)


@func()
def parallel_write_and_reduce(n: i32) -> i64:
    # a kernel with both a side effect (it writes its index into the buffer) and
    # a return value (which is reduced): both must happen for every index
    buf: Array[i32, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i32) -> i64:
        out[int_cast(i, usize)] = i
        return coerce(i, i64)

    def add(a: i64, b: i64) -> i64:
        return a + b

    total = parallel_loop(n, kernel, add)

    written: i64 = 0
    j: i32 = 0
    while j < n:
        written = written + cast(i64, out[int_cast(j, usize)])
        j = j + 1
    return total * 1000000 + written


@func()
def parallel_write_and_pair(n: i32) -> i64:
    # a kernel with a side effect and a *multi-valued* result, reduced as a pair
    buf: Array[i32, Literal[2048]] = undefined()
    out = arr_slice(ref(buf))

    def kernel(i: i32):
        out[int_cast(i, usize)] = i
        return coerce(i, i64), coerce(1, i64)

    def add(a, b):
        return a[0] + b[0], a[1] + b[1]

    total, count = parallel_loop(n, kernel, add)

    written: i64 = 0
    j: i32 = 0
    while j < n:
        written = written + cast(i64, out[int_cast(j, usize)])
        j = j + 1
    return total * 1000000 + count * 1000 + written


class SpyOpenMPTest(TestCase):
    """``std.openmp.parallel_loop``: the static-schedule parallel loop."""

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

    def test_a_64_bit_float_reduction(self) -> None:
        self.assertEqual(parallel_sum_f64(1000), 499500.0)

    def test_a_32_bit_float_reduction(self) -> None:
        self.assertEqual(parallel_sum_f32(1000), 499500.0)

    def test_a_complex128_reduction(self) -> None:
        # sum(0..999) as the real part, 1000 as the imaginary one
        self.assertEqual(parallel_sum_c128(1000), 499500 + 1000j)

    def test_a_complex64_reduction(self) -> None:
        self.assertEqual(parallel_sum_c64(1000), 499500 + 0j)

    def test_a_kernel_that_writes_and_returns(self) -> None:
        # both the writes and the reduced result cover every index
        self.assertEqual(parallel_write_and_reduce(1000), 499500 * 1000000 + 499500)

    def test_a_kernel_that_writes_and_returns_a_pair(self) -> None:
        self.assertEqual(
            parallel_write_and_pair(1000), 499500 * 1000000 + 1000 * 1000 + 499500,
        )


all_tests = [
    SpyOpenMPTest,
]

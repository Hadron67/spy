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
from ...std.core import arr_slice, undefined
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


all_tests = [
    SpyOpenMPTest,
]

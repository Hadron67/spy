"""``std.ndarray``: a multi-dimensional array view, its subscript and slicing,
and the Python-side ``from_ndarray`` conversion."""

import unittest
from typing import Any, Literal
from unittest import TestCase

from ...compiler import CompileError, f64, func, i32, usize
from ...std.core import slice as SpySlice
from ...std.ndarray import NDArray, from_ndarray, span

np: Any
try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is an optional test dependency
    np = None


# ---------------------------------------------------------------------------
# subscript: ``a[i, j]`` gives the place of one element, and every dimension
# has to be named
# ---------------------------------------------------------------------------


@func()
def nd_get_1d(a: NDArray[i32, Literal[1]], i: usize) -> i32:
    return a[i]


@func()
def nd_first_1d(a: NDArray[i32, Literal[1]]) -> i32:
    return a[0]


@func()
def nd_get_f64(a: NDArray[f64, Literal[1]], i: usize) -> f64:
    return a[i]


@func()
def nd_get_2d(a: NDArray[i32, Literal[2]], i: usize, j: usize) -> i32:
    return a[i, j]


@func()
def nd_set_2d(a: NDArray[i32, Literal[2]], i: usize, j: usize, v: i32) -> i32:
    a[i, j] = v
    return a[i, j]


@func()
def nd_sum_3d(a: NDArray[i32, Literal[3]]) -> i32:
    return a[0, 1, 2] + a[1, 0, 1]


@func()
def nd_missing_a_dimension(a: NDArray[i32, Literal[2]], i: usize) -> i32:
    return a[i]


# ---------------------------------------------------------------------------
# slice: one argument per dimension, a ``usize`` selecting one index and a
# ``std.slice`` object naming a range
# ---------------------------------------------------------------------------


@func()
def nd_slice_1d(a: NDArray[i32, Literal[1]]) -> i32:
    b = a.slice(span(1, 4))
    return b[0] + b[2]


@func()
def nd_slice_range(a: NDArray[i32, Literal[2]]) -> i32:
    b = a.slice(span(0, 2), span(1, 3))
    return b[0, 0] + b[1, 1]


@func()
def nd_slice_index(a: NDArray[i32, Literal[2]], j: usize) -> i32:
    b = a.slice(span(None, None), j)
    return b[0, 0] + b[1, 0]


@func()
def nd_slice_dynamic(a: NDArray[i32, Literal[2]], begin: usize, end: usize) -> i32:
    b = a.slice(span(begin, end), span(None, None))
    return b[0, 0] + b[1, 0]


@func()
def nd_slice_literal_index(a: NDArray[i32, Literal[2]]) -> i32:
    b = a.slice(span(None, None), 1)
    return b[0, 0] + b[1, 0]


@func()
def nd_slice_write(a: NDArray[i32, Literal[2]]) -> i32:
    # the view shares the elements: a write through it reaches the original
    b = a.slice(span(1, 3), span(1, 2))
    b[0, 0] = 99
    return a[1, 1]


@func()
def nd_slice_bad_arity(a: NDArray[i32, Literal[2]]) -> i32:
    b = a.slice(span(0, 2))
    return b[0, 0]


@func()
def nd_slice_out_of_bounds(a: NDArray[i32, Literal[2]]) -> i32:
    b = a.slice(span(0, 100), span(None, None))
    return b[0, 0]


@func()
def nd_slice_with_a_step(a: NDArray[i32, Literal[2]]) -> i32:
    b = a.slice(SpySlice[usize](0, 2, 2), span(None, None))
    return b[0, 0]


@unittest.skipUnless(np is not None, 'numpy is not installed')
class SpyNDArrayTest(TestCase):
    """``NDArray``: a subscript names one element (every dimension given), and
    ``slice`` builds a view."""

    def test_read_2d(self) -> None:
        arr = np.arange(6, dtype=np.int32).reshape(2, 3)
        self.assertEqual(nd_get_2d(from_ndarray(arr), 0, 0), 0)
        self.assertEqual(nd_get_2d(from_ndarray(arr), 1, 2), 5)

    def test_read_1d(self) -> None:
        arr = np.arange(5, dtype=np.int32)
        self.assertEqual(nd_get_1d(from_ndarray(arr), 3), 3)
        self.assertEqual(nd_first_1d(from_ndarray(arr)), 0)

    def test_read_a_float_array(self) -> None:
        arr = np.arange(3, dtype=np.float64) + 0.5
        self.assertEqual(nd_get_f64(from_ndarray(arr), 2), 2.5)

    def test_write_2d(self) -> None:
        arr = np.arange(6, dtype=np.int32).reshape(2, 3)
        self.assertEqual(nd_set_2d(from_ndarray(arr), 1, 1, 42), 42)
        self.assertEqual(arr[1, 1], 42)

    def test_read_3d(self) -> None:
        arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        self.assertEqual(nd_sum_3d(from_ndarray(arr)), 19)

    def test_a_missing_dimension_is_rejected(self) -> None:
        arr = np.zeros((2, 3), dtype=np.int32)
        with self.assertRaises(CompileError):
            nd_missing_a_dimension(from_ndarray(arr), 0)

    def test_an_index_out_of_bounds_panics(self) -> None:
        arr = np.zeros((2, 3), dtype=np.int32)
        with self.assertRaises(Exception):  # noqa: B017
            nd_get_2d(from_ndarray(arr), 2, 0)

    def test_slice_with_ranges(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        self.assertEqual(nd_slice_range(from_ndarray(arr)), 1 + 6)

    def test_slice_1d(self) -> None:
        arr = np.arange(5, dtype=np.int32)
        self.assertEqual(nd_slice_1d(from_ndarray(arr)), 1 + 3)

    def test_slice_with_an_index_dimension(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        self.assertEqual(nd_slice_index(from_ndarray(arr), 2), 2 + 6)

    def test_slice_with_runtime_bounds(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        self.assertEqual(nd_slice_dynamic(from_ndarray(arr), 1, 3), 4 + 8)

    def test_slice_with_a_literal_index(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        self.assertEqual(nd_slice_literal_index(from_ndarray(arr)), 1 + 5)

    def test_slice_out_of_bounds_panics(self) -> None:
        arr = np.zeros((3, 4), dtype=np.int32)
        with self.assertRaises(Exception):  # noqa: B017
            nd_slice_out_of_bounds(from_ndarray(arr))

    def test_slice_with_a_step_is_rejected(self) -> None:
        arr = np.zeros((3, 4), dtype=np.int32)
        with self.assertRaises(Exception):  # noqa: B017
            nd_slice_with_a_step(from_ndarray(arr))

    def test_slice_is_a_view(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        self.assertEqual(nd_slice_write(from_ndarray(arr)), 99)
        self.assertEqual(arr[1, 1], 99)

    def test_slice_needs_one_argument_per_dimension(self) -> None:
        arr = np.zeros((2, 3), dtype=np.int32)
        with self.assertRaises(CompileError):
            nd_slice_bad_arity(from_ndarray(arr))


@unittest.skipUnless(np is not None, 'numpy is not installed')
class SpyFromNDArrayTest(TestCase):
    """``from_ndarray``: a zero-copy Python-side conversion."""

    def test_shapes_and_strides(self) -> None:
        arr = np.arange(6, dtype=np.int32).reshape(2, 3)
        nd = from_ndarray(arr)
        self.assertEqual(list(nd.size), [2, 3])
        self.assertEqual(list(nd.stride), [3, 1])

    def test_a_fortran_order_array(self) -> None:
        arr = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
        nd = from_ndarray(arr)
        self.assertEqual(list(nd.size), [2, 3])
        self.assertEqual(list(nd.stride), [1, 2])

    def test_a_non_contiguous_array(self) -> None:
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]
        nd = from_ndarray(arr)
        self.assertEqual(list(nd.size), [3, 2])
        self.assertEqual(list(nd.stride), [4, 2])
        self.assertEqual(nd_get_2d(nd, 2, 1), arr[2, 1])

    def test_an_unsupported_dtype_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            from_ndarray(np.zeros((2,), dtype='S4'))

    def test_a_negative_stride_is_rejected(self) -> None:
        arr = np.arange(4, dtype=np.int32)[::-1]
        with self.assertRaises(ValueError):
            from_ndarray(arr)


all_tests = [
    SpyNDArrayTest,
    SpyFromNDArrayTest,
]

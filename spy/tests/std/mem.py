"""Tests for :mod:`spy.std.mem`: the ``Layout`` reflection helpers and the
allocators.

A spy function that may raise cannot be called from Python (see
``dsl._RegisteredFn.__call__``), so every allocator call is wrapped in a
``try``/``except AllocError`` inside a non-raising ``@func()`` and reports a
sentinel when an allocation fails.  The tests observe memory through the
``SlicePtr``/``Ptr`` the allocator hands out, since a struct or a slice cannot
cross the Python boundary.
"""

from unittest import TestCase

from ...compiler import (
    func,
    u8,
    u32,
    u64,
    usize,
)
from ...compiler.syntax import defer, ref
from ...std.mem import (
    AllocError,
    CAllocator,
    DynamicAllocator,
    Layout,
    align_of,
    size_of,
)

# ---------------------------------------------------------------------------
# ``Layout`` reflection: ``layout_of``/``size_of``/``align_of``
# ---------------------------------------------------------------------------


@func()
def layout_size_u32() -> usize:
    return size_of(u32)


@func()
def layout_align_u64() -> usize:
    return align_of(u64)


@func()
def layout_fields() -> usize:
    # a ``Layout`` is a (size, align) pair; ``repeat`` multiplies the size
    layout = Layout(2, 3)
    return layout.size * 100 + layout.align * 10 + layout.repeat(4).size


# ---------------------------------------------------------------------------
# ``CAllocator``: ``alloc``/``resize`` and the ``Allocator`` helpers built on
# them (``new``/``free``/``new_array``/``resize_array``/``free_array``)
# ---------------------------------------------------------------------------


@func()
def c_alloc_write_read(x: u8) -> u8:
    a = CAllocator()
    try:
        s = a.alloc(Layout(1, 1))
        with defer(): a.free_array(s)
        s[0] = x
        result = s[0]
        return result
    except AllocError:
        return 255


@func()
def c_alloc_zero_is_empty() -> usize:
    # a zero-sized allocation makes no request to the C allocator: the slice is
    # empty
    a = CAllocator()
    try:
        return a.alloc(Layout(0, 1)).length
    except AllocError:
        return 255


@func()
def c_new_write_read(x: u8) -> u8:
    a = CAllocator()
    try:
        p = a.new(u8)
        with defer(): a.free(p)
        p[...] = x
        result = p[...]
        return result
    except AllocError:
        return 255


@func()
def c_new_array_sum(x: u8, n: usize) -> u8:
    a = CAllocator()
    try:
        s = a.new_array(u8, n)
        with defer(): a.free_array(s)
        s[0] = x
        s[1] = x + 1
        result = s[0] + s[1]
        return result
    except AllocError:
        return 255


@func()
def c_resize_array_grows_and_keeps(x: u8) -> u8:
    # growing reallocs, and the elements already written survive
    a = CAllocator()
    try:
        s = a.new_array(u8, 2)
        s[0] = x
        s[1] = x + 1
        grown = a.resize_array(s, 8)
        result = grown[0] + grown[1]
        a.free_array(grown)
        return result
    except AllocError:
        return 255


@func()
def c_shrink_array_keeps_prefix(x: u8) -> u8:
    # shrinking in place keeps the prefix
    a = CAllocator()
    try:
        s = a.new_array(u8, 4)
        s[0] = x
        s[1] = x + 1
        shrunk = a.resize_array(s, 2)
        result = shrunk[0] + shrunk[1]
        a.free_array(shrunk)
        return result
    except AllocError:
        return 255


@func()
def c_free_then_alloc_again(x: u8) -> u8:
    a = CAllocator()
    try:
        s = a.new_array(u8, 3)
        s[0] = x
        a.free_array(s)
        t = a.new_array(u8, 1)
        t[0] = x + 2
        result = t[0]
        a.free_array(t)
        return result
    except AllocError:
        return 255


# ---------------------------------------------------------------------------
# ``DynamicAllocator``: the ``CAllocator`` behind a vtable of function pointers
# ---------------------------------------------------------------------------


@func()
def dynamic_new_array_sum(x: u8, n: usize) -> u8:
    a = CAllocator()
    d = DynamicAllocator.create(ref(a))
    try:
        s = d.new_array(u8, n)
        with defer(): d.free_array(s)
        s[0] = x
        s[1] = x + 1
        result = s[0] + s[1]
        return result
    except AllocError:
        return 255


@func()
def dynamic_new_write_read(x: u8) -> u8:
    a = CAllocator()
    d = DynamicAllocator.create(ref(a))
    try:
        p = d.new(u8)
        with defer(): d.free(p)
        p[...] = x
        result = p[...]
        return result
    except AllocError:
        return 255


class SpyMemLayoutTest(TestCase):
    """``std.mem`` layout reflection: ``layout_of``/``size_of``/``align_of``."""

    def test_size_of(self) -> None:
        self.assertEqual(layout_size_u32(), 4)

    def test_align_of(self) -> None:
        self.assertEqual(layout_align_u64(), 8)

    def test_layout_fields_and_repeat(self) -> None:
        # (size, align) = (2, 3): 2 * 100 + 3 * 10 + repeat(4).size (8)
        self.assertEqual(layout_fields(), 238)


class SpyCAllocatorTest(TestCase):
    """``std.mem.CAllocator`` and the ``Allocator`` helpers it inherits: the
    memory it hands out is real (malloc/realloc/free) and round-trips values."""

    def test_alloc_write_read(self) -> None:
        self.assertEqual(c_alloc_write_read(65), 65)

    def test_a_zero_sized_allocation_is_empty(self) -> None:
        self.assertEqual(c_alloc_zero_is_empty(), 0)

    def test_new_write_read(self) -> None:
        self.assertEqual(c_new_write_read(42), 42)

    def test_new_array_write_read(self) -> None:
        self.assertEqual(c_new_array_sum(10, 4), 10 + 11)

    def test_resize_array_grows_and_keeps(self) -> None:
        self.assertEqual(c_resize_array_grows_and_keeps(7), 7 + 8)

    def test_shrink_array_keeps_prefix(self) -> None:
        self.assertEqual(c_shrink_array_keeps_prefix(20), 20 + 21)

    def test_free_then_alloc_again(self) -> None:
        self.assertEqual(c_free_then_alloc_again(1), 3)


class SpyDynamicAllocatorTest(TestCase):
    """``std.mem.DynamicAllocator``: a concrete allocator behind a runtime
    vtable of function pointers (``create`` wraps it, ``as_func_ptr`` builds
    the entries)."""

    def test_new_array_write_read(self) -> None:
        self.assertEqual(dynamic_new_array_sum(10, 4), 10 + 11)

    def test_new_write_read(self) -> None:
        self.assertEqual(dynamic_new_write_read(99), 99)


all_tests = [
    SpyMemLayoutTest,
    SpyCAllocatorTest,
    SpyDynamicAllocatorTest,
]

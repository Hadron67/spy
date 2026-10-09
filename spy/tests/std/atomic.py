"""Tests for :mod:`spy.std.atomic`: the atomic builtins that lower to the LLVM
atomic instructions (``atomic_load``/``atomic_store``/``atomic_rmw``/
``atomic_cmpxchg``/``atomic_fence``).

An atomic operation needs a real memory location, so every test names a local
variable's address with ``ref`` (a register the compiler kept inline would have
no address to operate on); the address forces the local into memory.
"""

from unittest import TestCase

from ...compiler import CompileError, i32
from ...compiler import bool as spy_bool
from ...compiler.dsl import func, struct
from ...compiler.syntax import ref
from ...std.atomic import (
    MemoryOrder,
    RmwOp,
    atomic_cmpxchg,
    atomic_fence,
    atomic_fetch_add,
    atomic_load,
    atomic_rmw,
    atomic_store,
    atomic_swap,
)


@func()
def store_then_load(x: i32) -> i32:
    p = ref(x)
    atomic_store(p, 42)
    return atomic_load(p)


@func()
def load_of_a_const_ptr(x: i32) -> i32:
    # a ``Ptr`` is accepted where a ``ConstPtr`` is expected
    return atomic_load(ref(x))


@func()
def swap_returns_old(x: i32) -> i32:
    p = ref(x)
    old = atomic_rmw(p, RmwOp.XCHG, 7)
    return old * 100 + x


@func()
def fetch_add_returns_old(x: i32) -> i32:
    p = ref(x)
    old = atomic_fetch_add(p, 5)
    return old * 100 + x


@func()
def rmw_and_or_xor(x: i32) -> i32:
    p = ref(x)
    atomic_rmw(p, RmwOp.OR, 8)
    atomic_rmw(p, RmwOp.XOR, 1)
    old = atomic_rmw(p, RmwOp.AND, 14)
    return old * 1000 + x


@func()
def cmpxchg_success(x: i32) -> i32:
    p = ref(x)
    old, ok = atomic_cmpxchg(p, 3, 9)
    result = old * 100 + x
    if ok:
        result += 10000
    return result


@func()
def cmpxchg_failure(x: i32) -> i32:
    p = ref(x)
    old, ok = atomic_cmpxchg(p, 4, 9)
    result = old * 100 + x
    if ok:
        result += 10000
    return result


@func()
def store_and_load_relaxed(x: i32) -> i32:
    p = ref(x)
    atomic_store(p, 5, MemoryOrder.MONOTONIC)
    return atomic_load(p, MemoryOrder.MONOTONIC)


@func()
def volatile_roundtrip(x: i32) -> i32:
    p = ref(x)
    atomic_store(p, 11, MemoryOrder.SEQ_CST, True)
    return atomic_load(p, MemoryOrder.SEQ_CST, True)


@func()
def fence_is_a_noop(x: i32) -> i32:
    atomic_fence()
    atomic_fence(MemoryOrder.ACQUIRE)
    atomic_fence(MemoryOrder.SEQ_CST)
    return x


@func()
def swap_helper(x: i32) -> i32:
    return atomic_swap(ref(x), 3)


@func()
def cmpxchg_release_success(x: i32) -> i32:
    p = ref(x)
    old, ok = atomic_cmpxchg(p, 3, 9, MemoryOrder.ACQ_REL, MemoryOrder.ACQUIRE)
    result = old * 100 + x
    if ok:
        result += 10000
    return result


# the rejected shapes: a failure ordering that is too strong, a memory ordering
# an operation does not accept, and a pointee that is not a scalar LLVM can
# access atomically


@func()
def cmpxchg_bad_failure_ordering(x: i32) -> i32:
    return atomic_cmpxchg(ref(x), 0, 1, MemoryOrder.SEQ_CST, MemoryOrder.RELEASE)[0]


@func()
def fence_monotonic_rejected(x: i32) -> i32:
    atomic_fence(MemoryOrder.MONOTONIC)
    return x


@func()
def load_release_rejected(x: i32) -> i32:
    return atomic_load(ref(x), MemoryOrder.RELEASE)


@func()
def store_acquire_rejected(x: i32) -> i32:
    atomic_store(ref(x), 1, MemoryOrder.ACQUIRE)
    return x


@struct()
class Pair:
    a: i32
    b: i32


@func()
def load_of_a_struct_pointer(a: i32, b: i32) -> i32:
    pair = Pair(a, b)
    return atomic_load(ref(pair)).a


@func()
def load_of_a_bool_pointer(x: spy_bool) -> spy_bool:
    return atomic_load(ref(x))


class SpyAtomicTest(TestCase):
    """The ``std.atomic`` builtins: atomic load/store/rmw/cmpxchg/fence."""

    def test_store_then_load(self) -> None:
        self.assertEqual(store_then_load(0), 42)

    def test_load_of_a_const_ptr(self) -> None:
        self.assertEqual(load_of_a_const_ptr(7), 7)

    def test_swap_returns_the_old_value(self) -> None:
        # the pointee becomes 7, and the old value 3 comes back
        self.assertEqual(swap_returns_old(3), 3 * 100 + 7)

    def test_fetch_add_returns_the_old_value(self) -> None:
        self.assertEqual(fetch_add_returns_old(10), 10 * 100 + 15)

    def test_the_bitwise_operations(self) -> None:
        # x = 2; or 8 -> 10; xor 1 -> 11; and 14 -> 10, old value 11
        self.assertEqual(rmw_and_or_xor(2), 11 * 1000 + 10)

    def test_cmpxchg_swaps_on_a_match(self) -> None:
        # x == 3, so the pointee becomes 9 and ``ok`` is true
        self.assertEqual(cmpxchg_success(3), 10000 + 3 * 100 + 9)

    def test_cmpxchg_leaves_the_pointee_on_a_mismatch(self) -> None:
        # x == 5, not 4: nothing is stored and ``ok`` is false
        self.assertEqual(cmpxchg_failure(5), 5 * 100 + 5)

    def test_a_relaxed_roundtrip(self) -> None:
        self.assertEqual(store_and_load_relaxed(0), 5)

    def test_a_volatile_roundtrip(self) -> None:
        self.assertEqual(volatile_roundtrip(0), 11)

    def test_a_fence_runs(self) -> None:
        self.assertEqual(fence_is_a_noop(9), 9)

    def test_the_swap_helper(self) -> None:
        # ``atomic_swap`` is a plain function delegating to ``atomic_rmw``
        self.assertEqual(swap_helper(8), 8)

    def test_a_stronger_success_ordering(self) -> None:
        self.assertEqual(cmpxchg_release_success(3), 10000 + 3 * 100 + 9)

    def test_a_failure_ordering_stronger_than_success_is_rejected(self) -> None:
        self.assertRaises(CompileError, cmpxchg_bad_failure_ordering, 1)

    def test_a_monotonic_fence_is_rejected(self) -> None:
        self.assertRaises(CompileError, fence_monotonic_rejected, 1)

    def test_a_release_load_is_rejected(self) -> None:
        self.assertRaises(CompileError, load_release_rejected, 1)

    def test_an_acquire_store_is_rejected(self) -> None:
        self.assertRaises(CompileError, store_acquire_rejected, 1)

    def test_a_non_scalar_pointee_is_rejected(self) -> None:
        self.assertRaises(CompileError, load_of_a_struct_pointer, 1, 2)

    def test_a_bool_pointee_is_rejected(self) -> None:
        # a bool is an ``i1``: LLVM will not access it atomically
        self.assertRaises(CompileError, load_of_a_bool_pointer, True)


all_tests = [
    SpyAtomicTest,
]

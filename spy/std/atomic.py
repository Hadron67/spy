"""Atomic memory operations (``std.atomic``).

``atomic_load``/``atomic_store``/``atomic_rmw``/``atomic_cmpxchg`` lower to the
LLVM atomic instructions (``load atomic`` / ``store atomic`` / ``atomicrmw`` /
``cmpxchg``), and ``atomic_fence`` to a ``fence``; the builtins are implemented
by the compile-time interpreter (see ``interp.HirRunner._builtin_atomic_*``).

The pointee of the pointer an operation takes has to be an integer or a pointer
- the byte-sized scalar types LLVM can access atomically (a ``bool`` is an
``i1``, which LLVM will not access atomically, and an integer's width must be a
power of two bytes).  Every operation carries a :class:`MemoryOrder`, the memory
ordering it is performed with.
"""

from enum import Enum

from ..compiler import ConstPtr, Ptr, builtin_func


class MemoryOrder(Enum):
    """The memory ordering of an atomic operation, from the weakest
    (``MONOTONIC``) to the strongest (``SEQ_CST``); the names are the ones
    LLVM's IR uses (``relaxed`` is spelled ``MONOTONIC`` there)."""

    MONOTONIC = 0
    ACQUIRE = 1
    RELEASE = 2
    ACQ_REL = 3
    SEQ_CST = 4


class RmwOp(Enum):
    """The operation of an :func:`atomic_rmw`: store ``XCHG`` (a swap), or
    apply ``ADD``/``SUB``/``AND``/``OR``/``XOR`` to the value already stored."""

    XCHG = 0
    ADD = 1
    SUB = 2
    AND = 3
    OR = 4
    XOR = 5


@builtin_func
def atomic_load[T](ptr: ConstPtr[T], ordering: MemoryOrder = MemoryOrder.SEQ_CST,
                   volatile: bool = False) -> T: ...


@builtin_func
def atomic_store[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST,
                    volatile: bool = False): ...


@builtin_func
def atomic_rmw[T](ptr: Ptr[T], op: RmwOp, value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST,
                  volatile: bool = False) -> T: ...


@builtin_func
def atomic_cmpxchg[T](ptr: Ptr[T], expected: T, desired: T,
                      success: MemoryOrder = MemoryOrder.SEQ_CST,
                      failure: MemoryOrder = MemoryOrder.SEQ_CST,
                      volatile: bool = False) -> tuple[T, bool]: ...


@builtin_func
def atomic_fence(ordering: MemoryOrder = MemoryOrder.SEQ_CST): ...


def atomic_fetch_add[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.ADD, value, ordering)``: add ``value`` to the
    pointee of ``ptr`` and return the value it held before."""
    return atomic_rmw(ptr, RmwOp.ADD, value, ordering)


def atomic_fetch_sub[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.SUB, value, ordering)``."""
    return atomic_rmw(ptr, RmwOp.SUB, value, ordering)


def atomic_fetch_and[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.AND, value, ordering)``."""
    return atomic_rmw(ptr, RmwOp.AND, value, ordering)


def atomic_fetch_or[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.OR, value, ordering)``."""
    return atomic_rmw(ptr, RmwOp.OR, value, ordering)


def atomic_fetch_xor[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.XOR, value, ordering)``."""
    return atomic_rmw(ptr, RmwOp.XOR, value, ordering)


def atomic_swap[T](ptr: Ptr[T], value: T, ordering: MemoryOrder = MemoryOrder.SEQ_CST) -> T:
    """``atomic_rmw(ptr, RmwOp.XCHG, value, ordering)``: store ``value`` and
    return the pointee's old value."""
    return atomic_rmw(ptr, RmwOp.XCHG, value, ordering)

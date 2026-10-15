"""OpenMP: run a loop body across the OpenMP thread team.

``parallel_loop`` implements what ``#pragma omp parallel for`` lowers to by
calling the runtime entry points itself - ``__kmpc_fork_call`` to form the
parallel region, and the ``__kmpc_for_static_init_*``/``__kmpc_for_static_fini``
pair to hand each thread its slice of the static schedule.

The host must load the OpenMP runtime (``libomp``) before anything using
``parallel_loop`` is compiled: the JIT resolves the ``__kmpc_*`` symbols from
the process, and they become visible only once the library is loaded with
``RTLD_GLOBAL``.  Call :func:`load_openmp_runtime` first (it is idempotent):

    load_openmp_runtime()
    parallel_loop(1000, kernel)
"""

import ctypes
import ctypes.util
import os
from collections.abc import Callable
from typing import Any, Protocol

from spy.std.int import int_cast

from ..compiler import (
    ConstMultiPtr,
    ConstPtr,
    Ptr,
    bool,
    decl_func,
    func,
    func_type,
    i32,
    i64,
    struct,
    typeof,
    u8,
    u32,
    u64,
    void,
)
from ..compiler.syntax import (
    ConstComptime,
    Opaque,
    as_func_ptr,
    closure,
    comptime,
    ptr_cast,
    ref,
)
from .core import (
    coerce,
    compile_error,
    typed_undefined,
)

# the ``kmp_sch_static`` schedule kind the runtime entry points take (see
# ``kmp.h``); the plain static schedule needs no chunk size
_SCHED_STATIC = 34

_LOADED = False


def load_openmp_runtime() -> None:
    """Load the OpenMP runtime (``libomp``) into the process with ``RTLD_GLOBAL``
    so the JIT can resolve the ``__kmpc_*`` entry points :func:`parallel_loop`
    calls.  Idempotent.  The library is looked up at ``SPY_LIBOMP`` first, then at
    the paths it is usually installed at; a ``RuntimeError`` names every
    candidate that failed."""
    global _LOADED
    if _LOADED:
        return
    candidates: list[str] = []
    env = os.environ.get('SPY_LIBOMP')
    if env:
        candidates.append(env)
    found = ctypes.util.find_library('omp')
    if found:
        candidates.append(found)
    candidates.extend([
        '/opt/homebrew/opt/libomp/lib/libomp.dylib',
        '/usr/local/opt/libomp/lib/libomp.dylib',
        'libomp.so',
        'libomp.dylib',
    ])
    errors: list[str] = []
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            errors.append(f'  {candidate}: {error}')
            continue
        _LOADED = True
        return
    raise RuntimeError(
        'could not load the OpenMP runtime; set SPY_LIBOMP to the path of '
        'libomp:\n' + '\n'.join(errors)
    )


# -- the runtime ABI -------------------------------------------------------
#
# ``ident_t``: the source-location descriptor every entry point takes.  The
# initial value mirrors the one clang emits (``{0, 514, 0, 22, <source>}``).
#
# ``kmpc_micro``: the outlined function the runtime calls once per thread, with
# the two thread ids followed by whatever the ``__kmpc_fork_call`` call passed
# after the function pointer.


@struct(extern_c=True)
class _Ident:
    reserved_1: i32
    flags: i32
    reserved_2: i32
    reserved_3: i32
    psource: ConstMultiPtr[u8]


def _ident() -> ConstPtr[_Ident]:
    # the ``comptime()`` marker keeps the construction compile-time, so a
    # read-only compile-time variable holds it and its pointer lowers into a
    # global automatically (see the behaviour doc §4.3, replacing
    # ``as_static_ptr``); the source is a byte string lowered the same way
    comptime()
    ident = _Ident(0, 514, 0, 22, ptr_cast(b'spy', ConstMultiPtr[u8]))
    const_value: ConstComptime[_Ident] = ident
    return ref(const_value)


@func_type(callconv='c')
class _Microtask(Protocol):
    # the parameter names must match ``parallel_loop.outlined`` exactly (a
    # function type compares its arguments by name, see ``sval.FormalArg``)
    def __call__(self, gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque]) -> None: ...


@func_type(callconv='c')
class _MicrotaskReduce(Protocol):
    # the reducing variant of ``_Microtask``: the extra pointers carry the
    # reduction's reducer (a ``reduce_op`` closure), its shared accumulator, the
    # flag that says the accumulator has a value and the critical-section name;
    # the names must match ``parallel_loop.outlined_reduce`` exactly
    def __call__(self, gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque],
                 reduce_op: Ptr[Opaque], acc: Ptr[Opaque], has: Ptr[Opaque],
                 crit: Ptr[Opaque]) -> None: ...


@decl_func()
def __kmpc_fork_call(loc: ConstPtr[_Ident], argc: i32,
                     microtask: ConstPtr[_Microtask], *args) -> None: ...

@decl_func()
def __kmpc_for_static_init_4(loc: ConstPtr[_Ident], gtid: i32, schedtype: i32,
                             plastiter: Ptr[i32], plower: Ptr[i32], pupper: Ptr[i32],
                             pstride: Ptr[i32], incr: i32, chunk: i32) -> None: ...

@decl_func()
def __kmpc_for_static_init_4u(loc: ConstPtr[_Ident], gtid: i32, schedtype: i32,
                              plastiter: Ptr[i32], plower: Ptr[u32], pupper: Ptr[u32],
                              pstride: Ptr[i32], incr: i32, chunk: i32) -> None: ...

@decl_func()
def __kmpc_for_static_init_8(loc: ConstPtr[_Ident], gtid: i32, schedtype: i32,
                             plastiter: Ptr[i32], plower: Ptr[i64], pupper: Ptr[i64],
                             pstride: Ptr[i64], incr: i64, chunk: i64) -> None: ...

@decl_func()
def __kmpc_for_static_init_8u(loc: ConstPtr[_Ident], gtid: i32, schedtype: i32,
                              plastiter: Ptr[i32], plower: Ptr[u64], pupper: Ptr[u64],
                              pstride: Ptr[i64], incr: i64, chunk: i64) -> None: ...

@decl_func()
def __kmpc_for_static_fini(loc: ConstPtr[_Ident], gtid: i32) -> None: ...

# the named critical section a thread enters while it combines its partial into
# the shared accumulator; ``crit`` points at an 8-byte, 8-aligned name the
# runtime uses as the lock's identity (``kmp_critical_name``)
@decl_func()
def __kmpc_critical(loc: ConstPtr[_Ident], gtid: i32, crit: Ptr[Opaque]) -> None: ...

@decl_func()
def __kmpc_end_critical(loc: ConstPtr[_Ident], gtid: i32, crit: Ptr[Opaque]) -> None: ...

# the index type is ``typeof(length)``: the parameters carry no spy type - the
# signature is ignored (``ignore_signature``), so ``length``'s type is inferred
# from the call's argument, the kernel's return type is ``typeof(kernel(0))``,
# and the result is inferred from the body.  The annotations serve the Python
# type checker only, and the function is inlined at its call sites.
@func(inline=True, ignore_signature=True)
def parallel_loop[R](length: int, kernel: Callable[[int], R], reduce_op: Callable[[R, R], R] | None = None) -> R:
    """
    #pragma omp parallel loop
    for (typeof(length) i = 0; i < length; i++) {
         kernel(i);
    }

    The ``kernel`` is a closure; it is turned into an ordinary runtime value and
    handed to each thread, which calls it with its slice of ``[0, length)``.  The
    schedule is ``static``, and the index is ``typeof(length)`` (one of
    ``i32``/``u32``/``i64``/``u64``), which selects the matching runtime entry
    point.  The exit of the last ``__kmpc_fork_call`` joins the team, so no
    barrier is needed after the loop.

    When ``kernel`` returns a value (a single one, or several for a
    ``tuple[...]`` annotation) and ``reduce_op`` is given, the results are
    reduced and the reduced value is returned: each thread folds its own slice
    with ``reduce_op``, and the partial results are combined under a critical
    section, so ``reduce_op`` must be associative (and commutative, since the
    order the partials are combined in is not specified).  ``reduce_op`` is
    called ``reduce_op(acc, v)`` - each operand is a kernel result: a single
    value, or the whole tuple of them for a several-valued kernel (it is *not*
    spread into one argument per element).  With a ``length`` of
    zero no iteration runs, so there is nothing to reduce and the accumulator
    is left as its initial (undefined) value.
    """
    rtc = kernel
    comptime()
    K = typeof(rtc)
    # ``LT``/``ST`` are *type values* kept in compile-time variables; ``Any`` is
    # their Python-level type, so the checker does not try to track the generic.
    comptime()
    LT: Any = typeof(length)
    comptime()
    RET = typeof(kernel(0))

    # ``LT`` selects the runtime entry point (``init_fn``) and the type of the
    # schedule bookkeeping (the ``stride`` the runtime writes and the
    # ``incr``/``chunk`` it is given): ``i32`` for the 32-bit variants, ``i64``
    # for the 64-bit ones.  Both are compile-time choices.
    comptime()
    init_fn: Any = __kmpc_for_static_init_4
    comptime()
    ST: Any = i32
    if LT == i32:
        init_fn = __kmpc_for_static_init_4
        ST = i32
    elif LT == u32:
        init_fn = __kmpc_for_static_init_4u
        ST = i32
    elif LT == i64:
        init_fn = __kmpc_for_static_init_8
        ST = i64
    elif LT == u64:
        init_fn = __kmpc_for_static_init_8u
        ST = i64
    else:
        compile_error('parallel_loop: the index type must be i32/u32/i64/u64')

    # a kernel that returns nothing just runs its body; a kernel that returns a
    # value is reduced (the two paths differ in their result, their microtask
    # and the arguments ``__kmpc_fork_call`` is given, so the choice is
    # compile-time - the condition is a compile-time constant, so only the
    # selected branch is kept)
    if RET == void:

        @closure(inline=False, callconv='c')
        def outlined(gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque]) -> None:
            # a zero length makes no iterations (the guard clang also emits, and what
            # keeps the unsigned variants from wrapping ``n - 1``)
            if n > 0:
                lastiter: i32 = 0
                # the schedule bookkeeping and the index are runtime storage: the
                # runtime writes ``lower``/``upper``/``stride`` through the pointers
                # passed to ``init_fn``, and ``i`` advances in a runtime loop (an
                # unannotated local whose stores are all compile-time would be a
                # compile-time box, whose address cannot cross to a runtime call - see
                # the behaviour doc §3.3/§4.2)
                lower = coerce(0, LT)
                upper = int_cast(n - 1, LT)
                stride = coerce(0, ST)
                init_fn(_ident(), gtid[...], _SCHED_STATIC, ref(lastiter), ref(lower), ref(upper), ref(stride), 1, 1)
                i = lower
                while i <= upper:
                    ptr_cast(kernel, Ptr[K])[...](i)
                    i += 1
                __kmpc_for_static_fini(_ident(), gtid[...])

        __kmpc_fork_call(_ident(), 2, as_func_ptr(_Microtask, outlined), int_cast(length, i64), ptr_cast(ref(rtc), Ptr[Opaque]))
        return # pyright: ignore

    # the reducing path: ``acc`` is the shared accumulator the microtask fills
    # (``has`` says whether any thread has written it yet), and ``crit`` is the
    # critical section's name - a per-call, zero-initialized 8-byte slot
    rro = reduce_op
    comptime()
    RSO = typeof(rro)

    acc = typed_undefined(RET)
    has: bool = False
    crit: u64 = 0

    @closure(inline=False, callconv='c')
    def outlined_reduce(gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque],
                        reduce_op: Ptr[Opaque], acc: Ptr[Opaque], has: Ptr[Opaque],
                        crit: Ptr[Opaque]) -> None:
        if n > 0:
            lastiter: i32 = 0
            lower = coerce(0, LT)
            upper = int_cast(n - 1, LT)
            stride = coerce(0, ST)
            init_fn(_ident(), gtid[...], _SCHED_STATIC, ref(lastiter), ref(lower), ref(upper), ref(stride), 1, 1)
            # a thread the schedule gave no iteration has nothing to fold
            if lower <= upper:
                # each runtime value is a pointer cast once and reused
                kn = ptr_cast(kernel, Ptr[K])
                rs = ptr_cast(reduce_op, Ptr[RSO])
                acc_ptr = ptr_cast(acc, Ptr[RET])
                has_ptr = ptr_cast(has, Ptr[bool])

                # fold the thread's own slice: the first result seeds the partial
                part = kn[...](lower)
                i = lower + 1
                while i <= upper:
                    part = rs[...](part, kn[...](i))  # pyright: ignore
                    i += 1

                # combine the partial into the shared accumulator: the first
                # thread to get here seeds it, the rest fold onto it
                __kmpc_critical(_ident(), gtid[...], crit)
                if has_ptr[...]:
                    acc_ptr[...] = rs[...](acc_ptr[...], part)  # pyright: ignore
                else:
                    acc_ptr[...] = part
                    has_ptr[...] = True
                __kmpc_end_critical(_ident(), gtid[...], crit)
            __kmpc_for_static_fini(_ident(), gtid[...])

    __kmpc_fork_call(
        _ident(), 6, as_func_ptr(_MicrotaskReduce, outlined_reduce),  # pyright: ignore
        int_cast(length, i64),
        ptr_cast(ref(rtc), Ptr[Opaque]),
        ptr_cast(ref(rro), Ptr[Opaque]),
        ptr_cast(ref(acc), Ptr[Opaque]),
        ptr_cast(ref(has), Ptr[Opaque]),
        ptr_cast(ref(crit), Ptr[Opaque]),
    )
    return acc # pyright: ignore

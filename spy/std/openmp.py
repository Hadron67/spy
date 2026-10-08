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
from typing import Any, Protocol

from spy.std.int import int_cast

from ..compiler import (
    ConstMultiPtr,
    ConstPtr,
    Ptr,
    decl_func,
    func_type,
    i32,
    i64,
    struct,
    typeof,
    u8,
    u32,
    u64,
)
from ..compiler.syntax import Opaque, as_func_ptr, closure, comptime, ptr_cast, ref
from .core import (
    Numeric,
    as_runtime_closure,
    as_static_ptr,
    coerce,
    compile_error,
    gstr,
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
    # the ``comptime()`` marker keeps the construction a compile-time value, so
    # ``as_static_ptr`` can lower it into a global (like ``DynamicAllocator``'s
    # vtable)
    comptime()
    ident = _Ident(0, 514, 0, 22, gstr(b'spy'))
    return as_static_ptr(ident)


@func_type(callconv='c')
class _Microtask(Protocol):
    # the parameter names must match ``parallel_loop.outlined`` exactly (a
    # function type compares its arguments by name, see ``sval.FormalArg``)
    def __call__(self, gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque]) -> None: ...


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

# the index type parameter appears once in the signature (the kernel it types is
# an unannotated closure), but it is what ``typeof(length)`` reflects on to pick
# the runtime entry point, so it stays generic
def parallel_loop[T: Numeric](length: T, kernel) -> None:  # pyright: ignore
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
    """
    rtc = as_runtime_closure(kernel)
    comptime()
    K = typeof(rtc)
    # ``LT``/``ST`` are *type values* kept in compile-time variables; ``Any`` is
    # their Python-level type, so the checker does not try to track the generic.
    comptime()
    LT: Any = typeof(length)

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

    @closure(inline=False, callconv='c')
    def outlined(gtid: Ptr[i32], bound_tid: Ptr[i32], n: i64, kernel: Ptr[Opaque]) -> None:
        # a zero length makes no iterations (the guard clang also emits, and what
        # keeps the unsigned variants from wrapping ``n - 1``)
        if n > 0:
            lastiter: i32 = 0
            lower = coerce(LT, 0)
            upper = int_cast(n - 1, LT) # TODO: replace with int_cast
            stride = coerce(ST, 0)
            init_fn(_ident(), gtid[...], _SCHED_STATIC, ref(lastiter), ref(lower), ref(upper), ref(stride), 1, 1)
            i = lower
            while i <= upper:
                ptr_cast(kernel, Ptr[K])[...](i)
                i += 1
            __kmpc_for_static_fini(_ident(), gtid[...])

    __kmpc_fork_call(_ident(), 2, as_func_ptr(_Microtask, outlined), int_cast(length, i64), ptr_cast(ref(rtc), Ptr[Opaque]))

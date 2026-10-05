# This file is imported by the compiler, make sure to avoid circular imports.

from typing import TYPE_CHECKING, Any, Never, Protocol, Self, cast

from ..compiler import builtin_func, i32, i64, u8, u32, u64, usize
from ..compiler.dsl import Callable, func, func_type, struct
from ..compiler.syntax import (
    Array,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
    Opaque,
    Option,
    Ptr,
    comptime,
    ptr_cast,
)


class Numeric(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mod__(self, other: Self, /) -> Self: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...

@struct()
class StopIteration(Exception):
    """The exception ``range.__next__`` raises when the sequence is exhausted:
    the ``for`` desugaring catches it to end the loop (see ``astgen._gen_for``)."""

@struct()
class slice[T: Numeric]:
    """The object a slice subscript (``p[a:b:c]``) builds: the bounds are all
    optional, so a bound the source left out is ``None`` (the interpreter turns
    a missing ``start`` of a pointer slice into 0, see ``interp``)."""

    start: Option[T]
    end: Option[T]
    step: Option[T] = None

@struct()
class range[T: Numeric]:
    end: T
    start: T = cast(T, 0)
    step: T = cast(T, 1)

    def __iter__(self) -> range[T]:
        return self

    def __next__(self) -> T:
        if self.start >= self.end:
            raise StopIteration()
        # ``result`` holds a value that may have no runtime representation (the
        # element of a compile-time iterator, e.g. an untyped integer literal),
        # so it is kept inline: a compile-time ``for`` over a compile-time range
        # iterates at compile time (see ``astgen._gen_for``)
        comptime()
        result = self.start
        self.start += self.step
        return result

@struct()
class ConstSlicePtr[T]:
    """The slice of a *const* multi pointer - the value ``const_arr_slice``
    builds (see ``syntax._ConstSlicePtr`` for the type-checking shape)."""

    ptr: ConstMultiPtr[T]
    length: usize

    def __iter__(self) -> _SliceValueIterator[T]:
        return _SliceValueIterator(self.ptr, self.length)

    def refs(self) -> _ConstSliceRefIterator[T]:
        return _ConstSliceRefIterator(self.ptr, self.length)

    def _out_of_bounds(self):
        panic(sstr(b'index out of bounds'))

    def __spy_getitemptr__(self, index: usize) -> ConstPtr[T]:
        if index >= self.length:
            self._out_of_bounds()
        return self.ptr + index

    @func()
    def slice(self, begin: Option[usize], end: Option[usize]) -> ConstSlicePtr[T]:
        start: usize = 0
        if (b := begin) is not None:
            start = b
        stop: usize = self.length
        if (e := end) is not None:
            stop = e
        if start > stop or stop > self.length:
            self._out_of_bounds()
        return ConstSlicePtr(self.ptr + start, stop - start)

    if TYPE_CHECKING:
        # only so that the Python type checker accepts the ``s[i]`` spellings:
        # the subscript compiles to ``hir.Subscript`` and goes through
        # ``__spy_getitemptr__`` (see ``interp.HirRunner.subscript``), never
        # through these
        def __getitem__(self, index: int) -> T: ...

@struct()
class SlicePtr[T]:
    """The slice of a mutable multi pointer - the value ``arr_slice`` builds,
    and the value a slice subscript (``p[a:b]``) produces (see
    ``syntax._SlicePtr``)."""

    ptr: MultiPtr[T]
    length: usize

    def as_const(self) -> ConstSlicePtr[T]:
        return ConstSlicePtr(self.ptr, self.length)

    def __iter__(self) -> _SliceValueIterator[T]:
        return _SliceValueIterator(self.ptr, self.length)

    def refs(self) -> _SliceRefIterator[T]:
        return _SliceRefIterator(self.ptr, self.length)

    def _out_of_bounds(self):
        panic(sstr(b'index out of bounds'))

    def __spy_getitemptr__(self, index: usize) -> Ptr[T]:
        if index >= self.length:
            self._out_of_bounds()
        return self.ptr + index

    @func()
    def slice(self, begin: Option[usize], end: Option[usize] = None) -> SlicePtr[T]:
        start: usize = unwrap_or(begin, 0)
        stop: usize = unwrap_or(end, self.length)
        if start > stop or stop > self.length:
            self._out_of_bounds()
        return SlicePtr(self.ptr + start, stop - start)

    @func()
    def copy_from(self, source: ConstSlicePtr[T]):
        for i in range(self.length):
            self[i] = source[i]

    if TYPE_CHECKING:
        def __getitem__(self, index: int) -> T: ...
        def __setitem__(self, index: int, value: T) -> None: ...

@struct()
class _SliceValueIterator[T]:
    cursor: ConstMultiPtr[T]
    remaining: usize

    def __iter__(self) -> _SliceValueIterator[T]:
        return self

    def __next__(self) -> T:
        if self.remaining == 0:
            raise StopIteration()
        self.remaining = self.remaining - 1
        result = self.cursor[...]
        self.cursor += 1
        return result

@struct()
class _ConstSliceRefIterator[T]:
    cursor: ConstMultiPtr[T]
    remaining: usize

    def __iter__(self) -> _ConstSliceRefIterator[T]:
        return self

    def __next__(self) -> ConstPtr[T]:
        if self.remaining == 0:
            raise StopIteration()
        self.remaining = self.remaining - 1
        result = self.cursor
        self.cursor += 1
        return result

@struct()
class _SliceRefIterator[T]:
    cursor: MultiPtr[T]
    remaining: usize

    def __iter__(self) -> _SliceRefIterator[T]:
        return self

    def __next__(self) -> Ptr[T]:
        if self.remaining == 0:
            raise StopIteration()
        self.remaining = self.remaining - 1
        result = self.cursor
        self.cursor += 1
        return result

def arr_slice[T, N: int](arr: Ptr[Array[T, N]]) -> SlicePtr[T]:
    """The slice of the whole array the pointer ``arr`` names: the
    ``*[N]T -> SlicePtr[T]`` conversion, written out with ``ptr_cast`` (a
    pointer to an array already carries the address of its first element)."""
    return cast(SlicePtr[T], SlicePtr(ptr_cast(arr, MultiPtr[T]), cast(int, N)))

def const_arr_slice[T, N: int](arr: ConstPtr[Array[T, N]]) -> ConstSlicePtr[T]:
    """Likewise for a const pointer: the slice of it is a ``ConstSlicePtr``."""
    return cast(ConstSlicePtr[T], ConstSlicePtr(ptr_cast(arr, ConstMultiPtr[T]), cast(int, N)))

@struct()
class NullException(Exception):
    pass

def unwrap[T](val: Option[T]) -> T:
    if (ret := val) is not None:
        return ret
    raise NullException()

def unwrap_or[T](val: Option[T], default: T) -> T:
    if (ret := val) is not None:
        return ret
    return default

@builtin_func
def undefined() -> Any: ...

# ``gstr(s)``: the compile-time byte string ``s`` as a global static constant,
# the result a ``ConstMultiPtr[u8]`` to its bytes (see ``interp``).
@builtin_func
def gstr(s: bytes) -> ConstMultiPtr[u8]: ...

# ``sstr(s)``: the compile-time byte string ``s`` as a ``ConstSlicePtr[u8]`` -
# the pointer ``gstr(s)`` returns and the number of bytes.
@builtin_func
def sstr(s: bytes) -> ConstSlicePtr[u8]: ...

@builtin_func
def as_static_ptr[T](value: T) -> ConstPtr[T]: ...

@func_type()
class DestructorFn:
    def __call__(self, ptr: Ptr[Opaque]): ...

@struct()
class OpaqueWithDestructor:
    ptr: Ptr[Opaque]
    destructor: ConstPtr[DestructorFn]

    def deinit(self):
        self.destructor[...](self.ptr)

type PanicData = i32 | u32 | i64 | u64 | ConstSlicePtr[u8] | Ptr[Opaque] | OpaqueWithDestructor

def deinit_panic_data(data: PanicData) -> None:
    if isinstance(v := data, OpaqueWithDestructor):
        v.deinit()

@struct()
class UnwindException(Exception):
    data: PanicData

@builtin_func
def panic(data: PanicData) -> Never: ...

@builtin_func
def catch_unwind[T](fn: Callable[[], T]) -> T: ...

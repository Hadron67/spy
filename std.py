from typing import Protocol, Self, cast

from spy import syntax

from . import usize
from .dsl import struct
from .syntax import (
    Array,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
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
    return cast(SlicePtr[T], SlicePtr(ptr_cast(arr, syntax.MultiPtr[T]), cast(int, N)))

def const_arr_slice[T, N: int](arr: ConstPtr[Array[T, N]]) -> ConstSlicePtr[T]:
    """Likewise for a const pointer: the slice of it is a ``ConstSlicePtr``."""
    return cast(ConstSlicePtr[T], ConstSlicePtr(ptr_cast(arr, syntax.ConstMultiPtr[T]), cast(int, N)))

from types import EllipsisType
from typing import Any, Self, overload


class ConstPtr[T]:
    def __getitem__(self, value: EllipsisType) -> T:
        ...

class Ptr[T](ConstPtr[T]):
    def __setitem__(self, value: EllipsisType, val: T) -> None:
        ...

class _ConstSlicePtr[T]:
    """The slice of a *const* multi pointer: the pointer itself and the number
    of elements it carries.  Only used for type checking - the real slice types
    live in ``std`` (``ConstSlicePtr``/``SlicePtr``); ``ptr`` is a read-only
    property so that the mutable slice may narrow it covariantly."""

    @property
    def ptr(self) -> ConstMultiPtr[T]:
        raise NotImplementedError

    length: int

class _SlicePtr[T](_ConstSlicePtr[T]):
    @property
    def ptr(self) -> MultiPtr[T]:
        raise NotImplementedError


class ConstMultiPtr[T](ConstPtr[T]):
    @overload
    def __getitem__(self, value: int | EllipsisType) -> T:
        ...

    @overload
    def __getitem__(self, value: slice) -> _ConstSlicePtr[T]:
        ...

    def __getitem__(self, value: int | EllipsisType | slice) -> T | _ConstSlicePtr[T]:
        ...

    def __add__(self, amount: int) -> Self:
        ...

    def __iadd__(self, amount: int) -> Self:
        ...

class MultiPtr[T](ConstMultiPtr[T], Ptr[T]):
    @overload
    def __getitem__(self, value: int | EllipsisType) -> T:
        ...

    @overload
    def __getitem__(self, value: slice) -> _SlicePtr[T]:
        ...

    def __getitem__(self, value: int | EllipsisType | slice) -> T | _SlicePtr[T]:
        ...

    def __setitem__(self, value: int | EllipsisType, val: T) -> None:
        ...

class Array[T, L: int]:
    def __getitem__(self, value: int) -> T:
        ...

    def __setitem__(self, value: int, val: T) -> None:
        ...

type Comptime[T = Any] = T
type Option[T] = T | None

def ref[T](val: T) -> Ptr[T]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def array[T, Len: int](*elems: T, length: Len = 0) -> Array[T, Len]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def comptime():
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def unroll():
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def ptr_cast[T](ptr: Any, target: type[T]) -> T:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

class USize:
    pass

class ISize:
    pass

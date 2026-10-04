from contextlib import contextmanager
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

class Array[T, L: int | None]:
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

class Opaque:
    """An opaque type: a dynamically-sized type of unknown layout.  It is
    converted to ``sval.OpaqueType`` and, as a bare name, is only usable behind
    a pointer - a ``Ptr[Opaque]``/``ConstPtr[Opaque]`` lowers to a void pointer.
    As the last field of a ``@struct()`` it is an opaque tail, the flexible
    member of a C struct."""

@contextmanager
def defer():
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

@contextmanager
def okdefer():
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

@contextmanager
def errdefer():
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def as_func_ptr[T](type: type[T], obj: T) -> ConstPtr[T]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def closure(*, inline: bool = True, exceptions=None, callconv: str = 'default', may_panic: bool = False):
    """The decorator of a nested ``def`` inside a spy function (a *closure*).
    It only exists at parse time - the nested function is never executed by
    Python - so ``astgen`` recognizes it by identity and reads its arguments
    as the closure's declaration, standing in for the ``@func`` decorator a
    closure cannot carry (see ``dsl.func``): ``exceptions`` is the exceptions
    it may raise (``None``, the default, means it raises nothing, ``"infer"``
    that they are inferred from the body), ``inline`` whether it is forced to
    be inlined (the default) or compiled into a runtime function, and
    ``callconv``/``may_panic`` as in ``@func``."""
    def wrapper[T](func: T) -> T:
        raise RuntimeError("Cannot call directly: this is a spy closure decorator")
    return wrapper

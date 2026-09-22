from typing import Any, Literal


class Ptr[T, C: bool = Literal[False]]:
    def __getitem__(self, value) -> T:
        ...

class Array[T, L: int]:
    ...

def ref[T](val: T) -> Ptr[T]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def deref[T, C: bool](val: Ptr[T, C]) -> T:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def array[T](*elems: T) -> Array[T, len(elems)]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

def aggr(*elems, **kwargs) -> Any:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

from types import EllipsisType
from typing import Literal


class Ptr[T, C: bool = Literal[False]]:
    def __getitem__(self, value: EllipsisType) -> T:
        ...

    def __setitem__(self, value: EllipsisType, val: T) -> None:
        ...

class Array[T, L: int]:
    ...

def ref[T](val: T) -> Ptr[T]:
    raise RuntimeError("Cannot call directly: this function can only be used in spy functions")

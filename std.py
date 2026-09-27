from typing import Protocol, Self, cast

from .dsl import struct


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
class range[T: Numeric]:
    end: T
    start: T = cast(T, 0)
    step: T = cast(T, 0)

    def __iter__(self) -> range[T]:
        return self

    def __next__(self) -> T:
        if self.start >= self.end:
            raise StopIteration()
        result = self.start
        self.start += self.step
        return result

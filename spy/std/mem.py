"""Memory utilities (work in progress)."""

from typing import Protocol

from ..compiler import ConstPtr, func_type, struct, u8, usize
from .core import SlicePtr


@struct()
class AllocError(Exception):
    pass

@func_type(exceptions=AllocError)
class AllocFn(Protocol):
    def __call__(self, size: usize, align: usize) -> SlicePtr[u8]: ...

@func_type(exceptions=AllocError)
class ResizeFn(Protocol):
    def __call__(self, ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]: ...


class Allocator(Protocol):
    def alloc(self, size: usize, align: usize) -> SlicePtr[u8]: ...
    def resize(self, ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]: ...

@struct()
class AllocatorVtable:
    alloc: ConstPtr[AllocFn]
    resize: ConstPtr[ResizeFn]

@struct()
class DynamicAllocator:
    vtable: ConstPtr[AllocatorVtable]

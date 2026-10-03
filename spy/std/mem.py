"""Memory utilities (work in progress)."""

from typing import Protocol

from ..compiler import (
    ConstPtr,
    MultiPtr,
    Ptr,
    decl_func,
    func,
    func_type,
    struct,
    u8,
    usize,
)
from ..compiler.syntax import Opaque, ptr_cast
from .c import free, malloc, realloc
from .core import SlicePtr


@struct()
class AllocError(Exception):
    pass

@func_type(exceptions=AllocError)
class AllocFn(Protocol):
    def __call__(self, data: Ptr[Opaque], size: usize, align: usize) -> SlicePtr[u8]: ...

@func_type(exceptions=AllocError)
class ResizeFn(Protocol):
    def __call__(self, data: Ptr[Opaque], ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]: ...


class Allocator(Protocol):
    def alloc(self, size: usize, align: usize) -> SlicePtr[u8]: ...
    def resize(self, ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]: ...

@struct()
class AllocatorVtable:
    alloc: ConstPtr[AllocFn]
    resize: ConstPtr[ResizeFn]

@struct()
class DynamicAllocator:
    data: Ptr[Opaque]
    vtable: ConstPtr[AllocatorVtable]

    def alloc(self, size: usize, align: usize) -> SlicePtr[u8]:
        return self.vtable[...].alloc[...](self.data, size, align)

    def resize(self, ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]:
        return self.vtable[...].resize[...](self.data, ptr, size, align)

@struct()
class CAllocator:
    @func(exceptions=AllocError)
    def alloc(self, size: usize, align: usize) -> SlicePtr[u8]:
        if (ptr := malloc(size)) is not None:
            return SlicePtr(ptr, size)
        raise AllocError()

    @func(exceptions=AllocError)
    def resize(self, ptr: SlicePtr[u8], size: usize, align: usize) -> SlicePtr[u8]:
        if size == 0:
            free(ptr.ptr)
            # TODO
        if (ret := realloc(ptr.ptr, size)) is not None:
            return SlicePtr(ret, size)
        raise AllocError()

"""Memory utilities (work in progress)."""

from typing import Any, Protocol, override

from ..compiler import (
    ConstPtr,
    MultiPtr,
    Ptr,
    builtin_func,
    func,
    func_type,
    struct,
    u8,
    usize,
)
from ..compiler.syntax import Opaque, ptr_cast
from .c import free, malloc, realloc
from .core import SlicePtr, undefined


@struct()
class AllocError(Exception):
    pass

@struct()
class Layout:
    size: usize
    align: usize

    def repeat(self, count: usize) -> Layout:
        return Layout(self.size * count, self.align)

@builtin_func
def layout_of(type: Any) -> Layout: ...

def size_of(type: Any) -> usize:
    return layout_of(type).size

def align_of(type: Any) -> usize:
    return layout_of(type).align

@func_type(exceptions=AllocError)
class AllocFn(Protocol):
    def __call__(self, data: Ptr[Opaque], layout: Layout) -> SlicePtr[u8]: ...

@func_type(exceptions=AllocError)
class ResizeFn(Protocol):
    def __call__(self, data: Ptr[Opaque], ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]: ...


class Allocator(Protocol):
    def alloc(self, layout: Layout) -> SlicePtr[u8]: ...
    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]: ...

    def new[T](self, type: type[T]) -> Ptr[T]:
        layout = layout_of(type)
        return ptr_cast(self.alloc(layout).ptr, Ptr[T])

    def deinit[T](self, ptr: Ptr[T]):
        layout = layout_of(T)
        self.resize(ptr_cast(self.alloc(layout).ptr, MultiPtr[u8]), layout, layout_of(T).repeat(0))

    def new_array[T](self, type: type[T], count: usize) -> SlicePtr[T]:
        layout = layout_of(type).repeat(count)
        return SlicePtr(ptr_cast(self.alloc(layout).ptr, MultiPtr[T]), count)

    def resize_array[T](self, ptr: SlicePtr[T], count: usize) -> SlicePtr[T]:
        layout = layout_of(T).repeat(count)
        return SlicePtr(ptr_cast(self.resize(ptr_cast(ptr.ptr, MultiPtr[u8]), layout_of(T).repeat(ptr.length), layout), MultiPtr[T]), count)

    def deinit_array[T](self, ptr: SlicePtr[T]):
        self.resize_array(ptr, 0)

@struct()
class AllocatorVtable:
    alloc: ConstPtr[AllocFn]
    resize: ConstPtr[ResizeFn]

@struct()
class DynamicAllocator(Allocator):
    data: Ptr[Opaque]
    vtable: ConstPtr[AllocatorVtable]

    def alloc(self, layout: Layout) -> SlicePtr[u8]:
        return self.vtable[...].alloc[...](self.data, layout)

    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]:
        return self.vtable[...].resize[...](self.data, ptr, layout, new_layout)

@struct()
class CAllocator(Allocator):
    @func(exceptions=AllocError)
    @override
    def alloc(self, layout: Layout) -> SlicePtr[u8]:
        if (ptr := malloc(layout.size)) is not None:
            return SlicePtr(ptr, layout.size)
        raise AllocError()

    @func(exceptions=AllocError)
    @override
    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]:
        if new_layout.size == 0:
            free(ptr)
            return SlicePtr(undefined(), 0)
        if new_layout.size <= layout.size and new_layout.align <= layout.align:
            return SlicePtr(ptr, new_layout.size)
        if (ret := realloc(ptr, new_layout.size)) is not None:
            return SlicePtr(ret, new_layout.size)
        raise AllocError()

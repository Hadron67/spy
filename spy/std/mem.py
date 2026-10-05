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
    typeof,
    u8,
    usize,
)
from ..compiler.syntax import Comptime, Opaque, as_func_ptr, closure, comptime, ptr_cast
from .c import free, malloc, realloc
from .core import SlicePtr, as_static_ptr, panic, sstr, undefined


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

class Allocator(Protocol):
    def alloc(self, layout: Layout) -> SlicePtr[u8]: ...
    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]: ...

    def _free_failed(self):
        panic(sstr(b'failed to free memory'))

    def new[T](self, typ: type[T]) -> Ptr[T]:
        layout = layout_of(typ)
        return ptr_cast(self.alloc(layout).ptr, Ptr[T])

    def free[T](self, ptr: Ptr[T]):
        layout = layout_of(T)
        try:
            self.resize(ptr_cast(ptr, MultiPtr[u8]), layout, layout_of(T).repeat(0))
        except AllocError:
            self._free_failed()

    def new_array[T](self, typ: type[T], count: usize) -> SlicePtr[T]:
        layout = layout_of(typ).repeat(count)
        return SlicePtr(ptr_cast(self.alloc(layout).ptr, MultiPtr[T]), count)

    def resize_array[T](self, ptr: SlicePtr[T], count: usize) -> SlicePtr[T]:
        layout = layout_of(T).repeat(count)
        return SlicePtr(ptr_cast(self.resize(ptr_cast(ptr.ptr, MultiPtr[u8]), layout_of(T).repeat(ptr.length), layout).ptr, MultiPtr[T]), count)

    def free_array[T](self, ptr: SlicePtr[T]):
        try:
            self.resize_array(ptr, 0)
        except AllocError:
            self._free_failed()

@func_type(exceptions=AllocError)
class AllocFn(Protocol):
    def __call__(self, data: Ptr[Opaque], layout: Layout) -> SlicePtr[u8]: ...

@func_type(exceptions=AllocError)
class ResizeFn(Protocol):
    def __call__(self, data: Ptr[Opaque], ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]: ...

@struct()
class AllocatorVtable:
    alloc: ConstPtr[AllocFn]
    resize: ConstPtr[ResizeFn]

@struct()
class DynamicAllocator(Allocator):
    data: Ptr[Opaque]
    vtable: ConstPtr[AllocatorVtable]

    @staticmethod
    @func()
    def create(allocator) -> DynamicAllocator:
        t: Comptime = typeof(allocator[...])

        @closure(inline=False, exceptions=AllocError)
        def alloc(data: Ptr[Opaque], layout: Layout) -> SlicePtr[u8]:
            return ptr_cast(data, Ptr[t])[...].alloc(layout)

        @closure(inline=False, exceptions=AllocError)
        def resize(data: Ptr[Opaque], ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]:
            return ptr_cast(data, Ptr[t])[...].resize(ptr, layout, new_layout)

        comptime()
        vtable = AllocatorVtable(
            as_func_ptr(AllocFn, alloc),
            as_func_ptr(ResizeFn, resize),
        )
        return DynamicAllocator(ptr_cast(allocator, Ptr[Opaque]), as_static_ptr(vtable))

    def alloc(self, layout: Layout) -> SlicePtr[u8]:
        return self.vtable[...].alloc[...](self.data, layout)

    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]:
        return self.vtable[...].resize[...](self.data, ptr, layout, new_layout)

@struct()
class CAllocator(Allocator):
    @func(exceptions=AllocError)
    @override
    def alloc(self, layout: Layout) -> SlicePtr[u8]:
        if layout.size == 0:
            return SlicePtr(undefined(), 0)
        if (ptr := malloc(layout.size)) is not None:
            return SlicePtr(ptr, layout.size)
        raise AllocError()

    @func(exceptions=AllocError)
    @override
    def resize(self, ptr: MultiPtr[u8], layout: Layout, new_layout: Layout) -> SlicePtr[u8]:
        if new_layout.size == 0:
            free(ptr)
            return SlicePtr[u8](undefined(), 0)
        if new_layout.size <= layout.size and new_layout.align <= layout.align:
            return SlicePtr(ptr, new_layout.size)
        if (ret := realloc(ptr, new_layout.size)) is not None:
            return SlicePtr(ret, new_layout.size)
        raise AllocError()

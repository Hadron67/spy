from ..compiler import struct, usize
from ..compiler.dsl import func
from .core import SlicePtr
from .mem import Allocator, AllocError


@struct()
class Vec[T, A: Allocator]:
    ptr: SlicePtr[T]
    size: usize
    allocator: A

    def deinit(self):
        self.allocator.free_array(self.ptr)

    @func(exceptions=AllocError)
    def ensure_capacity(self, cap: usize):
        if cap > self.ptr.length:
            new_length = cap * 2
            ptr = self.allocator.resize_array(self.ptr, new_length)
            ptr.copy_from(self.ptr.as_const())
            self.ptr = ptr

    @func(exceptions=AllocError)
    def reserve(self, count: usize) -> SlicePtr[T]:
        self.ensure_capacity(self.size + count)
        ret = self.ptr.slice(self.size)
        self.size += count
        return ret

    def push(self, value: T):
        self.reserve(1)[0] = value

    def pop(self):
        self.size -= 1

    @func(exceptions=AllocError)
    def shrink_to_fit(self):
        if self.size < self.ptr.length:
            self.ptr = self.allocator.resize_array(self.ptr, self.size)

    def __iter__(self):
        return iter(self.ptr)

    def __bool__(self):
        return self.size > 0

    def refs(self):
        return self.ptr.refs()

from typing import TYPE_CHECKING, cast

from ..compiler import Option, Ptr, struct, usize
from ..compiler.dsl import func
from ..compiler.syntax import ref
from .core import SlicePtr, panic, sstr, undefined
from .mem import Allocator, AllocError, CAllocator


@struct()
class Vec[T, A: Allocator = CAllocator]:
    ptr: SlicePtr[T]
    size: usize = 0
    allocator: A = cast(A, CAllocator())

    @staticmethod
    def new(allocator: A) -> Vec[T, A]:
        return Vec(SlicePtr(undefined(), 0), 0, allocator)

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
        if self.size == 0:
            panic(sstr(b'pop from empty vector'))
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

    def __spy_getitemptr__(self, index: usize) -> Ptr[T]:
        if index >= self.size:
            panic(sstr(b'index out of bounds'))
        return ref(self.ptr[index])

    def slice(self, begin: Option[usize], end: Option[usize] = None) -> SlicePtr[T]:
        return self.ptr.slice(begin, end)

    if TYPE_CHECKING:
        def __getitem__(self, index: int) -> T: ...
        def __setitem__(self, index: int, value: T) -> None: ...

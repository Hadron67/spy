from ..compiler import (
    ConstMultiPtr,
    MultiPtr,
    Option,
    Ptr,
    decl_func,
    i32,
    struct,
    u8,
    usize,
)
from ..compiler.syntax import Opaque, ref
from .core import ConstSlicePtr, SlicePtr


@decl_func()
def malloc(size: usize) -> Option[MultiPtr[u8]]: ...

@decl_func()
def free(ptr: MultiPtr[u8]) -> None: ...

@decl_func()
def realloc(ptr: MultiPtr[u8], size: usize) -> Option[MultiPtr[u8]]: ...

@decl_func()
def puts(ptr: MultiPtr[u8]) -> None: ...

@struct()
class CError(Exception):
    code: i32

@struct()
class FILE:
    inner: Opaque

    @staticmethod
    def open(path: MultiPtr[u8], mode: MultiPtr[u8]) -> Ptr[FILE]:
        ptr = fopen(path, mode)
        if ptr is None:
            raise CError(0)
        return ptr

    def close(self):
        fclose(ref(self))

    def read(self, buf: ConstSlicePtr[u8]) -> usize:
        return fread(buf.ptr, 1, buf.length, ref(self))

    def write(self, buf: SlicePtr[u8]) -> usize:
        return fwrite(buf.ptr, 1, buf.length, ref(self))

@decl_func()
def fopen(path: MultiPtr[u8], mode: MultiPtr[u8]) -> Option[Ptr[FILE]]: ...

@decl_func()
def fclose(ptr: Ptr[FILE]) -> None: ...

@decl_func()
def fread(ptr: ConstMultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

@decl_func()
def fwrite(ptr: MultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

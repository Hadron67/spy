from typing import override

from spy.std.io import Readable, Writable

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

@decl_func()
def snprintf(buf: MultiPtr[u8], size: usize, fmt: ConstMultiPtr[u8], *args) -> i32: ...

@struct()
class CError(Exception):
    code: i32

@struct()
class FILE(Readable, Writable):
    inner: Opaque

    @staticmethod
    def open(path: MultiPtr[u8], mode: MultiPtr[u8]) -> Ptr[FILE]:
        ptr = fopen(path, mode)
        if ptr is None:
            raise CError(0)
        return ptr

    def close(self):
        fclose(ref(self))

    @override
    def read(self, data: SlicePtr[u8]) -> usize:
        return fread(data.ptr, 1, data.length, ref(self))

    @override
    def write(self, data: ConstSlicePtr[u8]) -> usize:
        return fwrite(data.ptr, 1, data.length, ref(self))

@decl_func()
def fopen(path: MultiPtr[u8], mode: MultiPtr[u8]) -> Option[Ptr[FILE]]: ...

@decl_func()
def fclose(ptr: Ptr[FILE]) -> None: ...

@decl_func()
def fread(ptr: MultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

@decl_func()
def fwrite(ptr: ConstMultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

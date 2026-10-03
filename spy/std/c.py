from ..compiler import (
    ConstMultiPtr,
    MultiPtr,
    Option,
    Ptr,
    decl_func,
    struct,
    u8,
    usize,
)
from ..compiler.syntax import Opaque, ref
from .core import ConstSlicePtr, SlicePtr


@decl_func("malloc")
def malloc(size: usize) -> Option[MultiPtr[u8]]: ...

@decl_func("free")
def free(ptr: MultiPtr[u8]) -> None: ...

@decl_func("realloc")
def realloc(ptr: MultiPtr[u8], size: usize) -> Option[MultiPtr[u8]]: ...

@decl_func("puts")
def puts(ptr: MultiPtr[u8]) -> None: ...

@struct()
class FILE:
    inner: Opaque

    def close(self):
        fclose(ref(self))

    def read(self, buf: ConstSlicePtr[u8]) -> usize:
        return fread(buf.ptr, 1, buf.length, ref(self))

    def write(self, buf: SlicePtr[u8]) -> usize:
        return fwrite(buf.ptr, 1, buf.length, ref(self))

@decl_func("fopen")
def fopen(path: MultiPtr[u8], mode: MultiPtr[u8]) -> Option[Ptr[FILE]]: ...

@decl_func("fclose")
def fclose(ptr: Ptr[FILE]) -> None: ...

@decl_func("fread")
def fread(ptr: ConstMultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

@decl_func("fwrite")
def fwrite(ptr: MultiPtr[u8], size: usize, count: usize, file: Ptr[FILE]) -> usize: ...

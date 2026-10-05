from typing import Protocol

from spy.compiler.syntax import ref
from spy.std import MultiPtr, ptr_cast

from ..compiler import func, struct, u8, usize
from .core import ConstSlicePtr, SlicePtr


@struct()
class IOError(Exception):
    pass

@struct()
class PartialWriteError(IOError):
    pass

@struct()
class PartialReadError(IOError):
    pass

class Writable(Protocol):
    def write(self, data: ConstSlicePtr[u8]) -> usize: ...

    @func(exceptions="infer")
    def write_all(self, data: ConstSlicePtr[u8]):
        while data:
            written = self.write(data)
            if written == 0:
                raise PartialWriteError()
            data = data.slice(written)

    def write_byte(self, value: u8):
        self.write(ConstSlicePtr(ptr_cast(ref(value), MultiPtr[u8]), 1))

class Readable(Protocol):
    def read(self, data: SlicePtr[u8]) -> usize: ...

    @func(exceptions="infer")
    def read_all(self, data: SlicePtr[u8]):
        while data:
            read = self.read(data)
            if read == 0:
                raise PartialReadError()
            data = data.slice(read)

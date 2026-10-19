from typing import Protocol

from ..compiler import Array, MultiPtr, comptime, func, ref, struct, typeof, u8, usize
from .core import (
    ConstSlicePtr,
    SlicePtr,
    const_arr_slice,
    panic,
    ptr_cast,
    sstr,
    typed_undefined,
)
from .int import int_cast
from .reflect import IntType, reify, type_info


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

    @func(inline=True)
    def write_byte(self, value: u8):
        self.write(ConstSlicePtr(ptr_cast(ref(value), MultiPtr[u8]), 1))

    @func(exceptions="infer")
    def format_int(self, value):
        if isinstance(int_info := type_info(typeof(value)), IntType):
            comptime()
            bits = int_info.bits
            # log_(10) 2^b
            buf = typed_undefined(Array[u8, int(bits * 0.31) + 1])
            cursor: usize = len(buf)
            signed: bool = False

            if int_info.signed and value < 0:
                value = -value
                signed = True
            value0 = int_cast(value, reify(IntType(bits, False)))
            while value0 > 0:
                buf[cursor] = value0 % 10
                value0 //= 10
                if cursor == 0:
                    panic(sstr("buffer too small"))
                cursor -= 1
            if signed:
                buf[cursor] = ord('-')
            self.write_all(const_arr_slice(ref(buf)))


class Readable(Protocol):
    def read(self, data: SlicePtr[u8]) -> usize: ...

    @func(exceptions="infer")
    def read_all(self, data: SlicePtr[u8]) -> usize:
        ret: usize = 0
        while data:
            read = self.read(data)
            if read == 0:
                break
            data = data.slice(read)
            ret += read
        return ret

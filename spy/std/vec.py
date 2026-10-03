from ..compiler import MultiPtr, struct, usize
from .mem import Allocator


@struct()
class Vec[T, A: Allocator]:
    ptr: MultiPtr[T]
    cap: usize
    allocator: A

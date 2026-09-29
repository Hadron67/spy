"""The parameters of the compile target.

The layout of a type and the calling convention of a function depend on the
target the code is compiled for - today only through the size of a pointer:
the width of an address decides the size and alignment of every pointer type
and, through the alignment of a struct's fields, the layout of every
aggregate (the order of the fields, the variant a union holds).  An
:class:`TargetInfo` holds those parameters; one belongs to each host context
(``dsl._Context``) and is threaded into the lowering that needs it.
"""

import ctypes
from dataclasses import dataclass

HOST_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)
"""The pointer size of the host: the compiled code works with host addresses
(see ``lower``), so this is the default of a compile target."""


@dataclass(frozen=True)
class TargetInfo:
    """The parameters of the compile target.  Only the pointer size for now:
    the width of an address, in bytes (and, at the same time, its
    alignment)."""

    pointer_size: int = HOST_POINTER_SIZE
    usize_bits: int = HOST_POINTER_SIZE * 8

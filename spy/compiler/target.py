"""The parameters of the compile target.

The layout of a type and the calling convention of a function depend on the
target the code is compiled for: the size of a pointer (the width of an address
decides the size and alignment of every pointer type and, through the alignment
of a struct's fields, the layout of every aggregate - the order of the fields,
the variant a union holds), and the widths of the C integer types (see
``syntax.c_char`` and friends).  An :class:`TargetInfo` holds those parameters;
one belongs to each host context (``dsl._Context``) and is threaded into the
lowering and the value conversion that need it.
"""

import ctypes
from dataclasses import dataclass

HOST_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)
"""The pointer size of the host: the compiled code works with host addresses
(see ``lower``), so this is the default of a compile target."""

HOST_C_CHAR_BITS = ctypes.sizeof(ctypes.c_char) * 8
HOST_C_SHORT_BITS = ctypes.sizeof(ctypes.c_short) * 8
HOST_C_INT_BITS = ctypes.sizeof(ctypes.c_int) * 8
HOST_C_LONG_BITS = ctypes.sizeof(ctypes.c_long) * 8
HOST_C_LONG_LONG_BITS = ctypes.sizeof(ctypes.c_longlong) * 8
"""The widths, in bits, of the host's C integer types (``c_long`` is 32 bits
on Windows and 64 on the usual 64-bit Unix targets, so it is read off the
host rather than hardcoded).  The unsigned spellings share the width."""


@dataclass(frozen=True)
class TargetInfo:
    """The parameters of the compile target: the pointer size (the width of an
    address, in bytes, which decides the size and alignment of every pointer
    type) and the widths, in bits, of the C integer types (see
    ``syntax.c_char`` and friends)."""

    pointer_size: int = HOST_POINTER_SIZE
    usize_bits: int = HOST_POINTER_SIZE * 8
    c_char_bits: int = HOST_C_CHAR_BITS
    c_short_bits: int = HOST_C_SHORT_BITS
    c_int_bits: int = HOST_C_INT_BITS
    c_long_bits: int = HOST_C_LONG_BITS
    c_longlong_bits: int = HOST_C_LONG_LONG_BITS

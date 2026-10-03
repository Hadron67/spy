"""The spy standard library.

``std`` re-exports the core types defined in :mod:`spy.std.core` and the
``syntax`` markers a spy function body may use, so that ``from std import
T`` names any of them.
"""

from ..compiler import syntax, usize
from ..compiler.dsl import struct
from ..compiler.syntax import (
    Array,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
    Option,
    Ptr,
    comptime,
    ptr_cast,
)
from .core import (
    ConstSlicePtr,
    Numeric,
    SlicePtr,
    StopIteration,
    arr_slice,
    const_arr_slice,
    range,
    slice,
    undefined,
)

__all__ = [
    'Array',
    'ConstMultiPtr',
    'ConstPtr',
    'ConstSlicePtr',
    'MultiPtr',
    'Numeric',
    'Option',
    'Ptr',
    'SlicePtr',
    'StopIteration',
    'arr_slice',
    'comptime',
    'const_arr_slice',
    'ptr_cast',
    'range',
    'slice',
    'struct',
    'syntax',
    'undefined',
    'usize',
]

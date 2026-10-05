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
    as_static_ptr,
    const_arr_slice,
    gstr,
    range,
    slice,
    sstr,
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
    'as_static_ptr',
    'comptime',
    'const_arr_slice',
    'gstr',
    'ptr_cast',
    'range',
    'slice',
    'sstr',
    'struct',
    'syntax',
    'undefined',
    'usize',
]

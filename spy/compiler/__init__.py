"""spy - System Python: JIT-compile Python functions into machine code.

Example::

    import spy

    @spy.func()
    def add[T](a: T, b: T) -> T:
        return a + b

    print(add(1, 2))        # compiles add(i64, i64) on first call
    print(add(1.0, 2.0))    # compiles add(f64, f64)

Pipeline: the Python source of a function is lowered by ``astgen`` into
an untyped HIR, which is *run* at compile time by ``interp`` with the
concrete argument types (comptime semantics: ``spy.typeof``, compile-time
``if``, inlining of plain Python functions) into a typed MIR, which
``lower`` turns into native code via LLVM.
"""

import builtins as pybuiltins
from types import NoneType
from typing import TYPE_CHECKING

from . import builtins as _builtins
from . import syntax
from .dsl import builtin_func, decl_func, func, func_type, struct
from .errors import CompileError, SpyError, TypeMismatchError
from .sval import BoolType, ComplexType, FloatType, IntType, VoidType
from .syntax import ConstMultiPtr, ConstPtr, ISize, MultiPtr, Option, Ptr, USize, typeof

# ``spy.typeof`` is a ``syntax`` marker: ``astgen`` recognizes it by identity and
# types its argument without emitting code (see ``syntax.typeof``).
compile_log = _builtins.spy_compile_log
# ``as`` is a keyword, so the public spelling is ``spy.as_``; the attribute
# ``spy.as`` stays reachable through ``getattr`` for parity with the docs.
as_ = _builtins.spy_as
globals()['as'] = _builtins.spy_as

if TYPE_CHECKING:
    u0 = int
    u1 = int
    u8 = int
    u16 = int
    u32 = int
    u64 = int
    i1 = int
    i8 = int
    i16 = int
    i32 = int
    i64 = int
    f32 = float
    f64 = float
    c64 = complex
    c128 = complex
    usize = int
    isize = int
    c_char = int
    c_uchar = int
    c_short = int
    c_ushort = int
    c_int = int
    c_uint = int
    c_long = int
    c_ulong = int
    c_longlong = int
    c_ulonglong = int
    bool = pybuiltins.bool
    void = NoneType
else:
    u0 = IntType(0, False)
    u1 = IntType(1, False)
    u8 = IntType(8, False)
    u16 = IntType(16, False)
    u32 = IntType(32, False)
    u64 = IntType(64, False)
    i1 = IntType(1, True)
    i8 = IntType(8, True)
    i16 = IntType(16, True)
    i32 = IntType(32, True)
    i64 = IntType(64, True)
    usize = USize
    isize = ISize
    c_char = syntax.c_char
    c_uchar = syntax.c_uchar
    c_short = syntax.c_short
    c_ushort = syntax.c_ushort
    c_int = syntax.c_int
    c_uint = syntax.c_uint
    c_long = syntax.c_long
    c_ulong = syntax.c_ulong
    c_longlong = syntax.c_longlong
    c_ulonglong = syntax.c_ulonglong
    f32 = FloatType(32)
    f64 = FloatType(64)
    c64 = ComplexType(FloatType(32))
    c128 = ComplexType(FloatType(64))
    bool = BoolType()
    void = VoidType()

__all__ = [
    'CompileError',
    'ConstMultiPtr',
    'ConstPtr',
    'MultiPtr',
    'Option',
    'Ptr',
    'SpyError',
    'TypeMismatchError',
    'as_',
    'bool',
    'builtin_func',
    'c64',
    'c128',
    'c_char',
    'c_int',
    'c_long',
    'c_longlong',
    'c_short',
    'c_uchar',
    'c_uint',
    'c_ulong',
    'c_ulonglong',
    'c_ushort',
    'compile_log',
    'decl_func',
    'f32',
    'f64',
    'func',
    'func_type',
    'i1',
    'i8',
    'i16',
    'i32',
    'i64',
    'isize',
    'struct',
    'typeof',
    'u1',
    'u8',
    'u16',
    'u32',
    'u64',
    'usize',
]


import builtins as pybuiltins
from types import NoneType
from typing import TYPE_CHECKING

from . import syntax
from .sval import BoolType, ComplexType, FloatType, IntType, VoidType
from .syntax import ISize, USize

_INT_BITS = range(65)

if TYPE_CHECKING:
    # The full fixed-width integer family: ``u0``..``u64`` and ``i0``..``i64``.
    # They are listed explicitly (instead of generated) so type checkers and
    # editors can resolve every name.
    u0 = u1 = u2 = u3 = u4 = u5 = u6 = u7 = u8 = u9 = int
    u10 = u11 = u12 = u13 = u14 = u15 = u16 = u17 = u18 = u19 = int
    u20 = u21 = u22 = u23 = u24 = u25 = u26 = u27 = u28 = u29 = int
    u30 = u31 = u32 = u33 = u34 = u35 = u36 = u37 = u38 = u39 = int
    u40 = u41 = u42 = u43 = u44 = u45 = u46 = u47 = u48 = u49 = int
    u50 = u51 = u52 = u53 = u54 = u55 = u56 = u57 = u58 = u59 = int
    u60 = u61 = u62 = u63 = u64 = int
    i0 = i1 = i2 = i3 = i4 = i5 = i6 = i7 = i8 = i9 = int
    i10 = i11 = i12 = i13 = i14 = i15 = i16 = i17 = i18 = i19 = int
    i20 = i21 = i22 = i23 = i24 = i25 = i26 = i27 = i28 = i29 = int
    i30 = i31 = i32 = i33 = i34 = i35 = i36 = i37 = i38 = i39 = int
    i40 = i41 = i42 = i43 = i44 = i45 = i46 = i47 = i48 = i49 = int
    i50 = i51 = i52 = i53 = i54 = i55 = i56 = i57 = i58 = i59 = int
    i60 = i61 = i62 = i63 = i64 = int
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
    # ``IntType`` is a frozen value object, so one instance per width is enough.
    for _bits in _INT_BITS:
        globals()[f'u{_bits}'] = IntType(_bits, False)
        globals()[f'i{_bits}'] = IntType(_bits, True)
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

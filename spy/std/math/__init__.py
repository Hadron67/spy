"""Math functions (``std.math``).

:func:`sin`, :func:`cos`, :func:`tan` and :func:`sqrt` each dispatch to one of
three implementations at compile time, chosen by the host's compile-time option
``sin_mode``/``cos_mode``/``tan_mode``/``sqrt_mode``:

* ``llvm`` - the LLVM intrinsic (``llvm.sin.*``/``llvm.cos.*``/``llvm.tan.*``/
  ``llvm.sqrt.*``),
* ``libc`` - the C library ``sinf``/``sin``/``cosf``/``cos``/``tanf``/``tan``/
  ``sqrtf``/``sqrt`` (see ``std.c``),
* ``software`` (the default, also when the option is unset) - the pure spy port
  in ``std.math.trig``/``std.math.sqrt``.

The mode is read while the function is compiled (``compile_option``) and only the
selected implementation is typed and lowered, so the choice costs nothing at run
time.
"""

from ...compiler import f32, f64, typeof
from ...compiler.dsl import func
from ...compiler.syntax import Comptime, comptime
from ..c import cos as c_cos
from ..c import cosf as c_cosf
from ..c import sin as c_sin
from ..c import sinf as c_sinf
from ..c import sqrt as c_sqrt
from ..c import sqrtf as c_sqrtf
from ..c import tan as c_tan
from ..c import tanf as c_tanf
from ..core import coerce, compile_error, compile_option
from ..llvm import cos_f32 as llvm_cos_f32
from ..llvm import cos_f64 as llvm_cos_f64
from ..llvm import sin_f32 as llvm_sin_f32
from ..llvm import sin_f64 as llvm_sin_f64
from ..llvm import sqrt_f32 as llvm_sqrt_f32
from ..llvm import sqrt_f64 as llvm_sqrt_f64
from ..llvm import tan_f32 as llvm_tan_f32
from ..llvm import tan_f64 as llvm_tan_f64
from .sqrt import sqrt as software_sqrt
from .trig import cos as software_cos
from .trig import sin as software_sin
from .trig import tan as software_tan

_MODE_LLVM = b'llvm'
_MODE_LIBC = b'libc'
_MODE_SOFTWARE = b'software'


@func()
def sin[T](x: T) -> T:
    """The sine of ``x`` (``f32`` or ``f64``): ``llvm.sin.*``, the libc
    ``sinf``/``sin``, or the pure spy implementation, as the compile-time option
    ``sin_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    comptime()
    typ = typeof(x)
    mode: Comptime = compile_option('sin_mode')
    use_llvm: Comptime = mode == _MODE_LLVM
    use_libc: Comptime = mode == _MODE_LIBC
    use_software: Comptime = mode is None or mode == _MODE_SOFTWARE
    if not use_llvm and not use_libc and not use_software:
        compile_error('sin_mode must be "llvm", "libc" or "software"')
    is_32: Comptime = typ == f32
    if is_32:
        x32: f32 = coerce(x, f32)
        if use_libc:
            return coerce(c_sinf(x32), typ)
        if use_software:
            return coerce(software_sin(x32), typ)
        return coerce(llvm_sin_f32(x32), typ)
    x64: f64 = coerce(x, f64)
    if use_libc:
        return coerce(c_sin(x64), typ)
    if use_software:
        return coerce(software_sin(x64), typ)
    return coerce(llvm_sin_f64(x64), typ)


@func()
def cos[T](x: T) -> T:
    """The cosine of ``x`` (``f32`` or ``f64``): ``llvm.cos.*``, the libc
    ``cosf``/``cos``, or the pure spy implementation, as the compile-time option
    ``cos_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    comptime()
    typ = typeof(x)
    mode: Comptime = compile_option('cos_mode')
    use_llvm: Comptime = mode == _MODE_LLVM
    use_libc: Comptime = mode == _MODE_LIBC
    use_software: Comptime = mode is None or mode == _MODE_SOFTWARE
    if not use_llvm and not use_libc and not use_software:
        compile_error('cos_mode must be "llvm", "libc" or "software"')
    is_32: Comptime = typ == f32
    if is_32:
        x32: f32 = coerce(x, f32)
        if use_libc:
            return coerce(c_cosf(x32), typ)
        if use_software:
            return coerce(software_cos(x32), typ)
        return coerce(llvm_cos_f32(x32), typ)
    x64: f64 = coerce(x, f64)
    if use_libc:
        return coerce(c_cos(x64), typ)
    if use_software:
        return coerce(software_cos(x64), typ)
    return coerce(llvm_cos_f64(x64), typ)


@func()
def tan[T](x: T) -> T:
    """The tangent of ``x`` (``f32`` or ``f64``): ``llvm.tan.*``, the libc
    ``tanf``/``tan``, or the pure spy implementation (``sin(x) / cos(x)``), as the
    compile-time option ``tan_mode`` (``llvm``/``libc``/``software``) selects.  An
    unknown mode is a compile error."""
    comptime()
    typ = typeof(x)
    mode: Comptime = compile_option('tan_mode')
    use_llvm: Comptime = mode == _MODE_LLVM
    use_libc: Comptime = mode == _MODE_LIBC
    use_software: Comptime = mode is None or mode == _MODE_SOFTWARE
    if not use_llvm and not use_libc and not use_software:
        compile_error('tan_mode must be "llvm", "libc" or "software"')
    is_32: Comptime = typ == f32
    if is_32:
        x32 = coerce(x, f32)
        if use_libc:
            return coerce(c_tanf(x32), typ)
        if use_software:
            return coerce(software_tan(x32), typ)
        return coerce(llvm_tan_f32(x32), typ)
    x64: f64 = coerce(x, f64)
    if use_libc:
        return coerce(c_tan(x64), typ)
    if use_software:
        return coerce(software_tan(x64), typ)
    return coerce(llvm_tan_f64(x64), typ)


@func()
def sqrt[T](x: T) -> T:
    """The square root of ``x`` (``f32`` or ``f64``): ``llvm.sqrt.*``, the libc
    ``sqrtf``/``sqrt``, or the pure spy implementation, as the compile-time option
    ``sqrt_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    comptime()
    typ = typeof(x)
    mode: Comptime = compile_option('sqrt_mode')
    use_llvm: Comptime = mode == _MODE_LLVM
    use_libc: Comptime = mode == _MODE_LIBC
    use_software: Comptime = mode is None or mode == _MODE_SOFTWARE
    if not use_llvm and not use_libc and not use_software:
        compile_error('sqrt_mode must be "llvm", "libc" or "software"')
    is_32: Comptime = typ == f32
    if is_32:
        x32 = coerce(x, f32)
        if use_libc:
            return coerce(c_sqrtf(x32), typ)
        if use_software:
            return coerce(software_sqrt(x32), typ)
        return coerce(llvm_sqrt_f32(x32), typ)
    x64 = coerce(x, f64)
    if use_libc:
        return coerce(c_sqrt(x64), typ)
    if use_software:
        return coerce(software_sqrt(x64), typ)
    return coerce(llvm_sqrt_f64(x64), typ)

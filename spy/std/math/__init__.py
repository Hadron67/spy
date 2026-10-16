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
from ...compiler.syntax import comptime
from ..c import cos as c_cos
from ..c import cosf as c_cosf
from ..c import sin as c_sin
from ..c import sinf as c_sinf
from ..c import sqrt as c_sqrt
from ..c import sqrtf as c_sqrtf
from ..c import tan as c_tan
from ..c import tanf as c_tanf
from ..core import compile_error, compile_option
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

@func(inline=True)
def _dispatch_float(x, f32_fn, f64_fn):
    comptime()
    typ = typeof(x)
    if typ == f32:
        return f32_fn(x)
    elif typ == f64:
        return f64_fn(x)
    else:
        compile_error("must use f32 or f64")

@func(inline=True)
def _dispatch(x, opt_name, llvm_f32_fn, llvm_f64_fn, libc_f32_fn, libc_f64_fn, spy_fn):
    comptime()
    mode = compile_option(opt_name)
    if mode == b'llvm':
        return _dispatch_float(x, llvm_f32_fn, llvm_f64_fn)
    elif mode == b'libc':
        return _dispatch_float(x, libc_f32_fn, libc_f64_fn)
    elif mode == b'software' or mode is None:
        return spy_fn(x)
    else:
        compile_error('mode must be "llvm", "libc" or "software"')

@func(inline=True)
def sin[T](x: T) -> T:
    """The sine of ``x`` (``f32`` or ``f64``): ``llvm.sin.*``, the libc
    ``sinf``/``sin``, or the pure spy implementation, as the compile-time option
    ``sin_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    return _dispatch(x, "sin_mode", llvm_sin_f32, llvm_sin_f64, c_sinf, c_sin, software_sin)


@func(inline=True)
def cos[T](x: T) -> T:
    """The cosine of ``x`` (``f32`` or ``f64``): ``llvm.cos.*``, the libc
    ``cosf``/``cos``, or the pure spy implementation, as the compile-time option
    ``cos_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    return _dispatch(x, "cos_mode", llvm_cos_f32, llvm_cos_f64, c_cosf, c_cos, software_cos)


@func(inline=True)
def tan[T](x: T) -> T:
    """The tangent of ``x`` (``f32`` or ``f64``): ``llvm.tan.*``, the libc
    ``tanf``/``tan``, or the pure spy implementation (``sin(x) / cos(x)``), as the
    compile-time option ``tan_mode`` (``llvm``/``libc``/``software``) selects.  An
    unknown mode is a compile error."""
    return _dispatch(x, "tan_mode", llvm_tan_f32, llvm_tan_f64, c_tanf, c_tan, software_tan)

@func(inline=True)
def sqrt[T](x: T) -> T:
    """The square root of ``x`` (``f32`` or ``f64``): ``llvm.sqrt.*``, the libc
    ``sqrtf``/``sqrt``, or the pure spy implementation, as the compile-time option
    ``sqrt_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    return _dispatch(x, "sqrt_mode", llvm_sqrt_f32, llvm_sqrt_f64, c_sqrtf, c_sqrt, software_sqrt)

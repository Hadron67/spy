"""Math functions (``std.math``).

:func:`sin`, :func:`cos`, :func:`tan`, :func:`asin`, :func:`acos`,
:func:`sqrt`, :func:`cbrt`, :func:`pow`, :func:`exp` and :func:`log` each
dispatch to one of three implementations at compile time, chosen by the host's
compile-time option ``sin_mode``/``cos_mode``/``tan_mode``/``asin_mode``/
``acos_mode``/``sqrt_mode``/``cbrt_mode``/``pow_mode``/``exp_mode``/
``log_mode``:

* ``llvm`` - the LLVM intrinsic (``llvm.sin.*``/``llvm.cos.*``/``llvm.tan.*``/
  ``llvm.asin.*``/``llvm.acos.*``/``llvm.sqrt.*``/``llvm.cbrt.*``/
  ``llvm.pow.*``/``llvm.exp.*``/``llvm.log.*``),
* ``libc`` - the C library ``sinf``/``sin``/``cosf``/``cos``/``tanf``/``tan``/
  ``asinf``/``asin``/``acosf``/``acos``/``sqrtf``/``sqrt``/``cbrtf``/``cbrt``/
  ``powf``/``pow``/``expf``/``exp``/``logf``/``log`` (see ``std.c``),
* ``software`` (the default, also when the option is unset) - the pure spy port
  in ``std.math.trig``/``std.math.invtrig``/``std.math.sqrt``/``std.math.cbrt``/
  ``std.math.pow``/``std.math.exp``/``std.math.log``.

The mode is read while the function is compiled (``compile_option``) and only the
selected implementation is typed and lowered, so the choice costs nothing at run
time.
"""

from ...compiler import f32, f64, typeof
from ...compiler.dsl import func
from ...compiler.syntax import comptime
from ..c import acos as c_acos
from ..c import acosf as c_acosf
from ..c import asin as c_asin
from ..c import asinf as c_asinf
from ..c import cbrt as c_cbrt
from ..c import cbrtf as c_cbrtf
from ..c import cos as c_cos
from ..c import cosf as c_cosf
from ..c import exp as c_exp
from ..c import expf as c_expf
from ..c import log as c_log
from ..c import logf as c_logf
from ..c import pow as c_pow
from ..c import powf as c_powf
from ..c import sin as c_sin
from ..c import sinf as c_sinf
from ..c import sqrt as c_sqrt
from ..c import sqrtf as c_sqrtf
from ..c import tan as c_tan
from ..c import tanf as c_tanf
from ..core import compile_error, compile_option
from ..llvm import acos_f32 as llvm_acos_f32
from ..llvm import acos_f64 as llvm_acos_f64
from ..llvm import asin_f32 as llvm_asin_f32
from ..llvm import asin_f64 as llvm_asin_f64
from ..llvm import cbrt_f32 as llvm_cbrt_f32
from ..llvm import cbrt_f64 as llvm_cbrt_f64
from ..llvm import cos_f32 as llvm_cos_f32
from ..llvm import cos_f64 as llvm_cos_f64
from ..llvm import exp_f32 as llvm_exp_f32
from ..llvm import exp_f64 as llvm_exp_f64
from ..llvm import log_f32 as llvm_log_f32
from ..llvm import log_f64 as llvm_log_f64
from ..llvm import pow_f32 as llvm_pow_f32
from ..llvm import pow_f64 as llvm_pow_f64
from ..llvm import sin_f32 as llvm_sin_f32
from ..llvm import sin_f64 as llvm_sin_f64
from ..llvm import sqrt_f32 as llvm_sqrt_f32
from ..llvm import sqrt_f64 as llvm_sqrt_f64
from ..llvm import tan_f32 as llvm_tan_f32
from ..llvm import tan_f64 as llvm_tan_f64
from .cbrt import cbrt as software_cbrt
from .exp import exp as software_exp
from .invtrig import acos as software_acos
from .invtrig import asin as software_asin
from .log import log as software_log
from .pow import pow as software_pow
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
def _dispatch_float2(x, y, f32_fn, f64_fn):
    comptime()
    typ = typeof(x)
    if typ == f32:
        return f32_fn(x, y)
    elif typ == f64:
        return f64_fn(x, y)
    else:
        compile_error("must use f32 or f64")

@func(inline=True)
def _dispatch2(x, y, opt_name, llvm_f32_fn, llvm_f64_fn, libc_f32_fn, libc_f64_fn, spy_fn):
    comptime()
    mode = compile_option(opt_name)
    if mode == b'llvm':
        return _dispatch_float2(x, y, llvm_f32_fn, llvm_f64_fn)
    elif mode == b'libc':
        return _dispatch_float2(x, y, libc_f32_fn, libc_f64_fn)
    elif mode == b'software' or mode is None:
        return spy_fn(x, y)
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
def asin[T](x: T) -> T:
    """The arc sine of ``x`` in ``[-pi/2, pi/2]`` (``f32`` or ``f64``):
    ``llvm.asin.*``, the libc ``asinf``/``asin``, or the pure spy implementation,
    as the compile-time option ``asin_mode`` (``llvm``/``libc``/``software``)
    selects.  An unknown mode is a compile error."""
    return _dispatch(x, "asin_mode", llvm_asin_f32, llvm_asin_f64, c_asinf, c_asin, software_asin)

@func(inline=True)
def acos[T](x: T) -> T:
    """The arc cosine of ``x`` in ``[0, pi]`` (``f32`` or ``f64``):
    ``llvm.acos.*``, the libc ``acosf``/``acos``, or the pure spy implementation,
    as the compile-time option ``acos_mode`` (``llvm``/``libc``/``software``)
    selects.  An unknown mode is a compile error."""
    return _dispatch(x, "acos_mode", llvm_acos_f32, llvm_acos_f64, c_acosf, c_acos, software_acos)

@func(inline=True)
def sqrt[T](x: T) -> T:
    """The square root of ``x`` (``f32`` or ``f64``): ``llvm.sqrt.*``, the libc
    ``sqrtf``/``sqrt``, or the pure spy implementation, as the compile-time option
    ``sqrt_mode`` (``llvm``/``libc``/``software``) selects.  An unknown mode is a
    compile error."""
    return _dispatch(x, "sqrt_mode", llvm_sqrt_f32, llvm_sqrt_f64, c_sqrtf, c_sqrt, software_sqrt)

@func(inline=True)
def cbrt[T](x: T) -> T:
    """The cube root of ``x`` (``f32`` or ``f64``): ``llvm.cbrt.*``, the libc
    ``cbrtf``/``cbrt``, or the pure spy implementation, as the compile-time
    option ``cbrt_mode`` (``llvm``/``libc``/``software``) selects.  An unknown
    mode is a compile error."""
    return _dispatch(x, "cbrt_mode", llvm_cbrt_f32, llvm_cbrt_f64, c_cbrtf, c_cbrt, software_cbrt)

@func(inline=True)
def pow[T](x: T, y: T) -> T:
    """The power ``x ** y`` (``f32`` or ``f64``, the result taking their common
    type): ``llvm.pow.*``, the libc ``powf``/``pow``, or the pure spy
    implementation, as the compile-time option ``pow_mode``
    (``llvm``/``libc``/``software``) selects.  An unknown mode is a compile
    error."""
    return _dispatch2(x, y, "pow_mode", llvm_pow_f32, llvm_pow_f64, c_powf, c_pow, software_pow)

@func(inline=True)
def exp[T](x: T) -> T:
    """The exponential ``e ** x`` (``f32`` or ``f64``): ``llvm.exp.*``, the
    libc ``expf``/``exp``, or the pure spy implementation, as the compile-time
    option ``exp_mode`` (``llvm``/``libc``/``software``) selects.  An unknown
    mode is a compile error."""
    return _dispatch(x, "exp_mode", llvm_exp_f32, llvm_exp_f64, c_expf, c_exp, software_exp)

@func(inline=True)
def log[T](x: T) -> T:
    """The natural logarithm of ``x`` (``f32`` or ``f64``): ``llvm.log.*``,
    the libc ``logf``/``log``, or the pure spy implementation, as the
    compile-time option ``log_mode`` (``llvm``/``libc``/``software``) selects.
    An unknown mode is a compile error."""
    return _dispatch(x, "log_mode", llvm_log_f32, llvm_log_f64, c_logf, c_log, software_log)

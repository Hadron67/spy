"""Math functions (``std.math``).

:func:`sin` dispatches to one of three implementations at compile time, chosen
by the host's compile-time option ``sin_mode``:

* ``llvm`` - the LLVM intrinsic (``llvm.sin.f32``/``llvm.sin.f64``),
* ``libc`` - the C library ``sinf``/``sin`` (see ``std.c``),
* ``software`` (the default, also when the option is unset) - the pure spy port
  in ``std.math.sin``.

The mode is read while the function is compiled (``compile_option``) and only the
selected implementation is typed and lowered, so the choice costs nothing at run
time.
"""

from ...compiler import f32, f64, typeof
from ...compiler.dsl import func
from ...compiler.syntax import Comptime, comptime
from ..c import sin as c_sin
from ..c import sinf as c_sinf
from ..core import coerce, compile_error, compile_option
from ..llvm import sin_f32 as llvm_sin_f32
from ..llvm import sin_f64 as llvm_sin_f64
from .sin import sin as software_sin

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

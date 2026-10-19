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
``if``, calls to plain Python functions evaluated at compile time) into a
typed MIR, which ``lower`` turns into native code via LLVM.
"""

from . import std as std
from .compiler import *

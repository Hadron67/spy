"""spy - System Python: JIT-compile Python functions into machine code.

Example::

    import symlat.spy as spy

    cache = spy.JitContext()

    @cache.jit()
    def add[T](a: T, b: T) -> T:
        return a + b

    @cache.aot()
    def add_u64(a: spy.u64, b: spy.u64) -> spy.u64:
        return a + b

    print(add(1, 2))        # compiles add(i32, i32) on first call
    print(add(1.0, 2.0))    # compiles add(f64, f64)

Pipeline: the Python source of a function is lowered by ``astgen`` into
an untyped HIR, which is *run* at compile time by ``interp`` with the
concrete argument types (comptime semantics: ``spy.type``, compile-time
``if``, inlining of plain Python functions) into a typed MIR, which
``lower`` turns into native code via LLVM.
"""

from . import builtins as _builtins
from .dsl import JitContext
from .errors import CompileError, SpyError, TypeMismatchError
from .type import BoolType, FloatType, IntType

type = _builtins.spy_type  # ``spy.type`` is evaluated at compile time
compile_log = _builtins.spy_compile_log
# ``as`` is a keyword, so the public spelling is ``spy.as_``; the attribute
# ``spy.as`` stays reachable through ``getattr`` for parity with the docs.
as_ = _builtins.spy_as
globals()['as'] = _builtins.spy_as

u8 = IntType(8, False)
u16 = IntType(16, False)
u32 = IntType(32, False)
u64 = IntType(64, False)
i8 = IntType(8, True)
i16 = IntType(16, True)
i32 = IntType(32, True)
i64 = IntType(64, True)
f32 = FloatType(32)
f64 = FloatType(64)
bool = BoolType()

__all__ = [
    'CompileError',
    'JitContext',
    'SpyError',
    'TypeMismatchError',
    'as_',
    'bool',
    'compile_log',
    'f32',
    'f64',
    'i8',
    'i16',
    'i32',
    'i64',
    'type',
    'u8',
    'u16',
    'u32',
    'u64',
]

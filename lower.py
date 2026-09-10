"""Lowering of the typed MIR to native code.

The MIR is mapped instruction by instruction onto the textual LLVM IR
builder of ``symlat.jit.llvm`` (the same representation the rest of
``symlat`` uses); the generated module text is then JIT-compiled with
``llvmlite.binding`` (MCJIT), following the pattern of
``symlat.jit.compile.CompiledBackendFunction``.

Compilation is module-at-a-time: :func:`compile_module` lowers a whole
group of MIR functions into *one* LLVM module, so calls between them
become in-module ``define`` references.  Calls to functions compiled in
earlier modules become ``declare``d symbols that are mapped, at link
time, to the absolute addresses of the already compiled callees (every
module keeps its engine alive, so those addresses stay valid).
"""

import ctypes
from collections.abc import Sequence

from llvmlite import binding as llvm

from . import llvm as sllvm
from . import mir, sval
from .errors import CompileError, SpyError
from .fn import Backend, NativeFn
from .mir import (
    BoolType,
    BoolValue,
    FloatType,
    IntType,
    PointerType,
    StructType,
    Type,
)

class _Lowerer:
    def __init__(self, mod: mir.Module) -> None:
        self.mod = mod

    def convert_function(self, fn: mir.Function) -> sllvm.Function:
        # TODO
        raise NotImplementedError

class LLVMBackend(Backend):
    def __init__(self) -> None:
        super().__init__()


    def compile(self, mir: mir.Module) -> dict[mir.GlobalValue, NativeFn]:
        raise NotImplementedError

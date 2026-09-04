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
from . import mir
from .errors import CompileError
from .fn import NativeFn
from .mir import (
    BoolType,
    BoolValue,
    FloatType,
    FloatValue,
    IntType,
    IntValue,
    PointerType,
    StructType,
    Type,
    VoidType,
    type_str,
)

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

_ICMP_OPS = {
    'eq': sllvm.IcmpOp.EQ,
    'ne': sllvm.IcmpOp.NE,
    'lt': sllvm.IcmpOp.LT,
    'le': sllvm.IcmpOp.LE,
    'gt': sllvm.IcmpOp.GT,
    'ge': sllvm.IcmpOp.GE,
}

_CTYPE_INT = {
    (8, True): ctypes.c_int8,
    (8, False): ctypes.c_uint8,
    (16, True): ctypes.c_int16,
    (16, False): ctypes.c_uint16,
    (32, True): ctypes.c_int32,
    (32, False): ctypes.c_uint32,
    (64, True): ctypes.c_int64,
    (64, False): ctypes.c_uint64,
}


class _ModuleTypes:
    """The LLVM types of one module: the struct types are created once
    per module (their identity is what LLVM type compatibility checks
    use), mirroring the MIR struct types of the functions being
    lowered."""

    def __init__(self) -> None:
        self._structs: dict[StructType, sllvm.StructType] = {}

    def to_llvm(self, type: Type) -> sllvm.Type:
        match type:
            case BoolType():
                return sllvm.IntType(1)
            case IntType():
                return sllvm.IntType(type.bits)
            case FloatType():
                return sllvm.FloatType(type.bits)
            case VoidType():
                return sllvm.VoidType()
            case PointerType(elem):
                return sllvm.PointerType(self.to_llvm(elem))
            case StructType():
                ret = self._structs.get(type)
                if ret is None:
                    ret = sllvm.StructType(
                        *(self.to_llvm(f.type) for f in type.fields)
                    )
                    self._structs[type] = ret
                return ret
            case _:
                raise CompileError(f"type {type_str(type)} cannot be lowered to LLVM yet")

    def struct_types(self) -> list[sllvm.Type]:
        return list(self._structs.values())


def to_ctype(type: Type) -> type[ctypes._CDataType] | None:
    """The ctypes type of a *result* of the native boundary (``None``
    for a void result); arguments use ``py_entry_arg_ctype``."""
    match type:
        case BoolType():
            return ctypes.c_bool
        case IntType():
            ct = _CTYPE_INT.get((type.bits, type.signed))
            if ct is None:
                raise CompileError(f"integer type {type_str(type)} has no ctypes mapping")
            return ct
        case FloatType():
            return ctypes.c_float if type.bits == 32 else ctypes.c_double
        case PointerType():
            return ctypes.c_void_p
        case VoidType():
            # a void result: ctypes restype ``None``
            return None
        case StructType():
            return type.ctype
        case _:
            raise CompileError(f"type {type_str(type)} has no ctypes mapping")


class _Lowerer:
    """Lowers the region trees of the MIR functions of one module onto
    their shared ``sllvm.Function`` definitions (every region becomes a
    chain of LLVM basic blocks).  A call whose callee is a
    :class:`mir.Function` - of the module itself, or a function that is
    being compiled into it (recursion) - references the in-module
    ``define``; a call to a :class:`mir.Symbol` of an earlier module
    references a (cached) extern declaration."""

    def __init__(
        self, llvm_fns: dict[str, sllvm.Function], types: _ModuleTypes
    ) -> None:
        self._llvm_fns = llvm_fns
        self._types = types
        self._lowered: dict[object, sllvm.Value] = {}
        self._declarations: dict[str, sllvm.DeclareFunction] = {}

    def _to_llvm(self, type: Type) -> sllvm.Type:
        return self._types.to_llvm(type)

    def lower(self, fn: mir.Function) -> None:
        """Lower the region tree of ``fn`` into its pre-created
        ``sllvm.Function``."""
        llvm_fn = self._llvm_fns[fn.name]
        arg_values = llvm_fn.get_args()
        self._lower_region(llvm_fn, llvm_fn.entry, fn.insts, arg_values, None)

    def _lower_region(
        self,
        llvm_fn: sllvm.Function,
        block: sllvm.BasicBlock,
        insts: tuple[mir.Inst, ...] | list[mir.Inst],
        arg_values: tuple[sllvm.Value, ...],
        cont: sllvm.BasicBlock | None,
    ) -> None:
        """Lower one region into LLVM blocks.  ``cont`` is the block
        that a region falling off its end jumps to (None only at the end
        of the function, where falling off is a compile error)."""
        if len(insts) == 0:
            if not block._finished:
                assert cont is not None, 'function must not fall off its end'
                block.jmp(cont)
            return
        inst = insts[0]
        if isinstance(inst, mir.Ret):
            if inst.value is None:
                # a void return (``ret void``)
                block.ret(None)
            else:
                value = self._value(inst.value, arg_values)
                block.ret(value)
            return
        if isinstance(inst, mir.If):
            cond = self._value(inst.cond, arg_values)
            rest = insts[1:]
            cont_block = sllvm.BasicBlock() if rest else cont
            then_block = sllvm.BasicBlock()
            if len(inst.else_body) == 0:
                assert cont_block is not None
                block.br(cond, then_block, cont_block)
                self._lower_region(llvm_fn, then_block, inst.then_body, arg_values, cont_block)
            else:
                else_block = sllvm.BasicBlock()
                block.br(cond, then_block, else_block)
                self._lower_region(llvm_fn, then_block, inst.then_body, arg_values, cont_block)
                self._lower_region(llvm_fn, else_block, inst.else_body, arg_values, cont_block)
            if rest:
                assert cont_block is not None
                self._lower_region(llvm_fn, cont_block, rest, arg_values, cont)
            return
        # a plain instruction
        self._lower_inst(block, inst, arg_values)
        self._lower_region(llvm_fn, block, insts[1:], arg_values, cont)

    def _value(self, value: mir.Value, arg_values: tuple[sllvm.Value, ...]) -> sllvm.Value:
        if isinstance(value, mir.Param):
            return arg_values[value.index]
        if isinstance(value, mir.Inst):
            ret = self._lowered.get(id(value))
            if ret is None:
                raise CompileError('internal error: instruction not lowered yet')
            return ret
        if isinstance(value, BoolValue):
            return sllvm.IntValue(1 if value.value else 0, sllvm.IntType(1))
        if isinstance(value, IntValue):
            return sllvm.IntValue(value.value, sllvm.IntType(value.bits))
        if isinstance(value, FloatValue):
            return sllvm.FloatType(value.bits).from_float(value.value)
        if isinstance(value, mir.Function):
            # a function value of this very module: its definition
            llvm_fn = self._llvm_fns.get(value.name)
            if llvm_fn is None:
                raise CompileError(
                    f'internal error: function {value.name} is not part of the module'
                )
            return llvm_fn
        if isinstance(value, mir.Symbol):
            # a symbol is a callee compiled in an earlier module: it is
            # always an external here (its address is resolved at link
            # time)
            return self._declare(value)
        raise CompileError(f'cannot lower value {value!r}')

    def _declare(self, symbol: mir.Symbol) -> sllvm.DeclareFunction:
        """The declared external symbol a function value refers to.  All
        calls to the same symbol share one declaration (per module), so
        the linker resolves a single extern per callee."""
        decl = self._declarations.get(symbol.name)
        if decl is None:
            fn_type = sllvm.fn_type(
                self._to_llvm(symbol.fn_type.return_type),
                *(self._to_llvm(t) for t in symbol.fn_type.args),
            )
            decl = sllvm.DeclareFunction(symbol.name, fn_type)
            self._declarations[symbol.name] = decl
        return decl

    def _lower_inst(
        self,
        block: sllvm.BasicBlock,
        inst: mir.Inst,
        arg_values: tuple[sllvm.Value, ...],
    ) -> None:
        result: sllvm.Value | None = None
        match inst:
            case mir.Alloca(t):
                result = block.alloca(self._to_llvm(t.elem))
            case mir.Store():
                ptr = self._value(inst.ptr, arg_values)
                value = self._value(inst.value, arg_values)
                block.store(ptr, value)
            case mir.Load():
                ptr = self._value(inst.ptr, arg_values)
                result = block.load(ptr)
            case mir.Gep():
                ptr = self._value(inst.ptr, arg_values)
                result = block.get_element_ptr(ptr, 0, inst.index)
            case mir.Arith():
                lhs = self._value(inst.lhs, arg_values)
                rhs = self._value(inst.rhs, arg_values)
                match inst.op:
                    case 'add':
                        result = block.add(lhs, rhs)
                    case 'sub':
                        result = block.sub(lhs, rhs)
                    case 'mul':
                        result = block.mul(lhs, rhs)
                    case 'div':
                        result = block.div(lhs, rhs, inst.signed)
                    case 'rem':
                        result = block.rem(lhs, rhs, inst.signed)
                    case 'xor':
                        result = block.xor(lhs, rhs)
                    case _:
                        raise CompileError(f"unsupported MIR operation '{inst.op}'")
            case mir.Convert():
                value = self._value(inst.value, arg_values)
                to = self._to_llvm(inst.type)
                match inst.kind:
                    case 'sitofp':
                        result = block.int_to_float(value, to)  # type: ignore[arg-type]
                    case 'uitofp':
                        result = block.uint_to_float(value, to)  # type: ignore[arg-type]
                    case 'fpext':
                        result = block.float_ext(value, to)  # type: ignore[arg-type]
                    case 'fptrunc':
                        result = block.float_trunc(value, to)  # type: ignore[arg-type]
                    case 'sext':
                        result = block.sext(value, to)  # type: ignore[arg-type]
                    case 'zext':
                        result = block.zext(value, to)  # type: ignore[arg-type]
                    case 'trunc':
                        result = block.emit(sllvm.IntTrunc(value, to))  # type: ignore[arg-type]
                    case _:
                        raise CompileError(f"unsupported conversion '{inst.kind}'")
            case mir.Cmp():
                lhs = self._value(inst.lhs, arg_values)
                rhs = self._value(inst.rhs, arg_values)
                op = _ICMP_OPS[inst.op]
                if inst.kind == 'int':
                    result = block.icmp(op, inst.signed, lhs, rhs)
                else:
                    result = block.fcmp(op, lhs, rhs)
            case mir.Call():
                # the callee is a function value: lowering it yields the
                # in-module definition or a (cached) extern declaration
                callee = self._value(inst.callee, arg_values)
                result = block.call(callee, *(self._value(a, arg_values) for a in inst.args))
            case _:
                raise CompileError(f'unsupported MIR instruction {type(inst).__name__}')
        if result is not None:
            self._lowered[id(inst)] = result


def py_entry_arg_ctype(type: Type) -> type[ctypes._CDataType]:
    """The ctypes type of one *argument of the Python-facing entry* of
    a native function: a by-value struct formal is passed as a pointer
    there (the entry is a generated thunk - see :func:`compile_module` -
    that loads the struct and calls the by-value function)."""
    if isinstance(type, StructType):
        return ctypes.c_void_p
    return to_ctype(type)  # type: ignore[return-value]


def _ctypes_thunk(
    types: _ModuleTypes, value_fn: sllvm.Function, fn: mir.Function
) -> sllvm.Function:
    """The Python-facing entry of one MIR function that takes a
    by-value struct parameter: a pointer-form wrapper that loads each
    struct argument and calls the by-value (value-form) function.

    ctypes and the LLVM ABI disagree on the machine convention of
    aggregate by-value arguments (LLVM classifies IR-level aggregates
    itself instead of following the platform C ABI), so a struct cannot
    be handed to a JIT-compiled function by value through ctypes; the
    Python boundary passes the *address* of the instance and the wrapper
    performs the copy.  Spy-to-spy calls never go through it: they call
    the value-form function directly."""
    assert any(isinstance(a.type, StructType) for a in fn.args)
    assert fn.ret_type is not None, f'function {fn.name} is not fully typed'
    thunk = sllvm.Function(f'{fn.name}.py')
    arg_types = tuple(
        sllvm.PointerType(types.to_llvm(a.type))
        if isinstance(a.type, StructType)
        else types.to_llvm(a.type)
        for a in fn.args
    )
    args = thunk.add_args(*arg_types)
    thunk.set_return_type(types.to_llvm(fn.ret_type))
    block = thunk.entry
    call_args = tuple(
        block.load(arg) if isinstance(a.type, StructType) else arg
        for a, arg in zip(fn.args, args)
    )
    if isinstance(fn.ret_type, VoidType):
        block.call(value_fn, *call_args)
        block.ret(None)
    else:
        block.ret(block.call(value_fn, *call_args))
    return thunk


def compile_module(
    fns: Sequence[mir.Function],
    extern: dict[str, NativeFn],
) -> list[NativeFn]:
    """JIT-compile a group of MIR functions together into one LLVM
    module, returning one :class:`NativeFn` per function.

    The functions must have distinct names.  ``extern`` maps the symbol
    names of spy functions compiled by *earlier* modules to their
    addresses; every declaration the generated module refers to must be
    resolvable from it.
    """
    assert len({fn.name for fn in fns}) == len(fns), 'duplicate function names in one module'
    types = _ModuleTypes()

    # create one sllvm.Function per MIR function up front: bodies may
    # call any function of the module, and the call sites need the
    # callee's definition (its signature) to type the call
    llvm_fns: dict[str, sllvm.Function] = {}
    for fn in fns:
        assert fn.ret_type is not None, f'function {fn.name} is not fully typed'
        if isinstance(fn.ret_type, StructType):
            raise CompileError(
                f"functions that return a {type_str(fn.ret_type)} value are not "
                f'supported yet (function {fn.name}): return its fields instead'
            )
        llvm_fn = sllvm.Function(fn.name)
        llvm_fn.add_args(*(types.to_llvm(a.type) for a in fn.args))
        llvm_fn.set_return_type(types.to_llvm(fn.ret_type))
        llvm_fns[fn.name] = llvm_fn

    lowerer = _Lowerer(llvm_fns, types)
    for fn in fns:
        lowerer.lower(fn)

    # the Python-facing entry of every function with a by-value struct
    # parameter is a pointer-form thunk (see ``_ctypes_thunk``)
    thunk_by_fn: dict[str, sllvm.Function] = {}
    for fn in fns:
        if any(isinstance(a.type, StructType) for a in fn.args):
            thunk = _ctypes_thunk(types, llvm_fns[fn.name], fn)
            thunk_by_fn[fn.name] = thunk

    mod = sllvm.Module()
    module_values: list[sllvm.Value] = [llvm_fns[fn.name] for fn in fns]
    module_values.extend(thunk_by_fn.values())
    mod.add_recursively(values=module_values)
    # register every struct type the module refers to (its writer needs
    # the ``%struct = type {...}`` definitions)
    mod.add_recursively(types=types.struct_types())
    lines = mod.write()

    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine()
    llvm_mod = llvm.parse_assembly('\n'.join(lines))
    llvm_mod.verify()

    backing_mod = llvm.parse_assembly('')
    engine = llvm.create_mcjit_compiler(backing_mod, tm)
    engine.add_module(llvm_mod)
    module_names = ', '.join(sorted(fn.name for fn in fns))
    for f in llvm_mod.functions:
        if not f.is_declaration:
            continue
        native_fn = extern.get(f.name)
        if native_fn is None:
            raise CompileError(
                f"cannot resolve the external function {f.name} referenced by {module_names}"
            )
        engine.add_global_mapping(f, native_fn.addr)
    engine.finalize_object()
    engine.run_static_constructors()

    rets: list[NativeFn] = []
    for fn in fns:
        assert fn.ret_type is not None, f'function {fn.name} is not fully typed'
        # spy-to-spy calls (in-module, recursive or across modules)
        # link to the value-form function, whose arguments are by value;
        # the ctypes entry bound below is the pointer-form thunk, when
        # one exists (see ``_ctypes_thunk``)
        value_addr = engine.get_function_address(fn.name)
        thunk = thunk_by_fn.get(fn.name)
        entry_addr = (
            engine.get_function_address(thunk.name)
            if thunk is not None
            else value_addr
        )
        arg_ctypes = tuple(py_entry_arg_ctype(t) for t in (a.type for a in fn.args))
        restype = to_ctype(fn.ret_type)
        proto = ctypes.CFUNCTYPE(restype, *arg_ctypes)  # type: ignore[arg-type]
        entry = ctypes.cast(entry_addr, proto)
        ret = NativeFn(fn.name, tuple(a.type for a in fn.args), fn.ret_type, lines)
        ret._engine = engine
        ret._addr = value_addr
        ret._entry = entry
        rets.append(ret)
    return rets

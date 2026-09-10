"""Lowering of the typed MIR to native code.

The MIR is mapped instruction by instruction onto the textual LLVM IR
builder of ``symlat.jit.llvm`` (the same representation the rest of
``symlat`` uses); the generated module text is then JIT-compiled with
``llvmlite.binding`` (MCJIT), following the pattern of
``symlat.jit.compile.CompiledBackendFunction``.

Compilation is module-at-a-time: :class:`LLVMBackend` lowers a whole group
of MIR functions into *one* LLVM module, so calls between them become
in-module ``define`` references.  Calls to functions compiled in earlier
modules become ``declare``d symbols whose addresses are mapped, at link
time, to the absolute addresses of the already compiled callees (every
module keeps its engine alive, so those addresses stay valid); the
backend remembers every symbol it has emitted, which is how a later
module resolves the symbols it imports.

The names come from the MIR symbol table: ``mir.Module.finish`` has
already given every symbol a unique name, which is sanitized into an
LLVM identifier and handed to the LLVM builder; the builder's own symbol
table handles any (in practice impossible) remaining collision.

The Python-facing entry of a native function is the value-form function
itself, except when the machine ABI cannot carry its arguments or result
through ctypes: a *by-value struct* formal is passed as a pointer there
(the entry is a generated thunk that loads the struct and calls the
value-form function) and a *by-value struct* result is written into a
caller-provided out buffer.  Spy-to-spy calls never go through the thunk:
they call the value-form function directly.
"""

import ctypes
from typing import Any

from llvmlite import binding as llvm

from . import llvm as sllvm
from . import mir
from .errors import CompileError
from .fn import Backend, NativeFn
from .util import StrBiMap, sanitize_name

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
        self._structs: dict[mir.StructType, sllvm.StructType] = {}

    def to_llvm(self, type: mir.MayBeVoidType) -> sllvm.Type:
        match type:
            case mir.VoidType():
                return sllvm.VoidType()
            case mir.BoolType():
                return sllvm.IntType(1)
            case mir.IntType():
                return sllvm.IntType(type.bits)
            case mir.FloatType():
                return sllvm.FloatType(type.bits)
            case mir.PointerType():
                return sllvm.PointerType(self.to_llvm(type.elem))
            case mir.ArrayType():
                return sllvm.ArrayType(self.to_llvm(type.elem), type.length)
            case mir.FunctionType():
                return sllvm.fn_type(
                    self.to_llvm(type.return_type),
                    *(self.to_llvm(a) for a in type.args),
                )
            case mir.StructType():
                ret = self._structs.get(type)
                if ret is None:
                    ret = sllvm.StructType(
                        sanitize_name(type.name_base or 'anon'),
                        *(self.to_llvm(f.type) for f in type.fields),
                    )
                    self._structs[type] = ret
                return ret
            case _:
                raise CompileError(f'type {type!r} cannot be lowered to LLVM')

    def struct_types(self) -> list[sllvm.Type]:
        return list(self._structs.values())


def _ctypes_of_int(bits: int, signed: bool) -> Any:
    for width in (8, 16, 32, 64):
        if bits <= width:
            return _CTYPE_INT[(width, signed)]
    return ctypes.c_int64 if signed else ctypes.c_uint64


def to_ctype(type: mir.MayBeVoidType) -> Any:
    """The ctypes type of a MIR type crossing the native boundary
    (``None`` for a void result)."""
    match type:
        case mir.VoidType():
            return None
        case mir.BoolType():
            return ctypes.c_bool
        case mir.IntType():
            return _ctypes_of_int(type.bits, type.signed)
        case mir.FloatType():
            return ctypes.c_float if type.bits == 32 else ctypes.c_double
        case mir.PointerType():
            return ctypes.c_void_p
        case mir.StructType():
            return struct_ctype(type)
        case _:
            raise CompileError(f'type {type!r} has no ctypes mapping')


def struct_ctype(struct: mir.StructType) -> type[ctypes.Structure]:
    """The ctypes ``Structure`` subclass mirroring the memory layout of
    the MIR struct ``struct`` - the Python-side memory *view* of a struct
    value crossing the native boundary (the out buffer of a function
    returning the struct by value).  The class is built once and cached
    on the MIR type."""
    cls = struct.ctype
    if cls is None:
        fields: list[tuple[str, Any]] = []
        for i, field in enumerate(struct.fields):
            ctype = to_ctype(field.type)
            assert ctype is not None
            fields.append((field.name or f'field{i}', ctype))
        cls = type(
            sanitize_name(struct.name_base or 'anon'),
            (ctypes.Structure,),
            {'_fields_': fields, '__module__': __name__},
        )
        struct.ctype = cls
    return cls


def py_entry_arg_ctype(type: mir.Type) -> Any:
    """The ctypes type of one *argument of the Python-facing entry*: a
    by-value struct formal is passed as a pointer there (the entry is a
    generated thunk - see :func:`_py_entry_thunk` - that loads the struct
    and calls the by-value function)."""
    if isinstance(type, mir.StructType):
        return ctypes.c_void_p
    ctype = to_ctype(type)
    assert ctype is not None
    return ctype


class _Lowerer:
    """Lowers the flat instruction lists of the MIR functions of one
    module onto their shared ``sllvm.Function`` definitions (each list
    becomes a chain of LLVM basic blocks; an ``If`` branches into its
    two marker-delimited regions, whose falling branches join the code
    after the matching ``End``).  A call whose callee is a
    :class:`mir.Function` - of the module itself, or a function that is
    being compiled into it (recursion) - references the in-module
    ``define``; a call to a :class:`mir.ExternAnonSymbol` of an earlier
    module references a (cached) extern declaration."""

    def __init__(
        self,
        llvm_fns: dict[mir.Function, sllvm.Function],
        types: _ModuleTypes,
        symbols: StrBiMap[mir.StructType | mir.GlobalValue],
    ) -> None:
        self._llvm_fns = llvm_fns
        self._types = types
        self._symbols = symbols
        self._lowered: dict[object, sllvm.Value] = {}
        self._declarations: dict[mir.GlobalValue, sllvm.DeclareFunction] = {}

    def _to_llvm(self, type: mir.MayBeVoidType) -> sllvm.Type:
        return self._types.to_llvm(type)

    @property
    def declarations(self) -> dict[mir.GlobalValue, sllvm.DeclareFunction]:
        """The extern declarations this lowering created, by the MIR
        symbol of the earlier module they refer to."""
        return self._declarations

    def _link_name(self, value: mir.GlobalValue) -> str:
        """The link name of a MIR global in this module: the name its
        symbol-table entry carries, sanitized into an LLVM identifier."""
        return sanitize_name(self._symbols.get_key(value))

    def lower(self, fn: mir.Function) -> None:
        """Lower the flat instruction list of ``fn`` into its
        pre-created ``sllvm.Function``.

        Every ``mir.Alloca`` is lowered into the function's entry block
        first, whatever position the interpreter emitted it at: a slot
        may first be stored inside a runtime branch (an inlined body
        whose result is delivered per path, see ``interp``), and its
        address must then be defined on every path that stores to it or
        reads it later."""
        llvm_fn = self._llvm_fns[fn]
        arg_values = llvm_fn.get_args()
        for inst in fn.insts:
            if isinstance(inst, mir.Alloca):
                self._lower_inst(llvm_fn.entry, inst, arg_values)
        self._lower_region(
            llvm_fn, llvm_fn.entry, fn.insts, 0, len(fn.insts), arg_values, None, ()
        )

    def _scan_block(
        self, insts: list[mir.Inst], i: int, has_else: bool
    ) -> tuple[int | None, int]:
        """The positions of the ``Else`` (only when ``has_else``, i.e.
        the opener at ``i`` is an ``mir.If``; ``None`` when the block has
        no else branch) and ``End`` markers that close the block opened
        at ``i``, found by a balanced scan forward (nested blocks - both
        ``If`` and ``Block`` - close their own markers first)."""
        depth = 0
        p_else: int | None = None
        for j in range(i + 1, len(insts)):
            inst = insts[j]
            if isinstance(inst, (mir.If, mir.Block)):
                depth += 1
            elif isinstance(inst, mir.End):
                if depth == 0:
                    return p_else, j
                depth -= 1
            elif isinstance(inst, mir.Else) and depth == 0:
                if not has_else:
                    raise CompileError('internal error: an Else marker inside a Block')
                p_else = j
        raise CompileError('internal error: unclosed block in the MIR')

    def _lower_region(
        self,
        llvm_fn: sllvm.Function,
        block: sllvm.BasicBlock,
        insts: list[mir.Inst],
        start: int,
        end: int,
        arg_values: tuple[sllvm.Value, ...],
        cont: sllvm.BasicBlock | None,
        exits: tuple[sllvm.BasicBlock | None, ...],
    ) -> None:
        """Lower the instructions ``insts[start:end]`` - one branch body
        of the flat stream, delimited by its enclosing markers - into LLVM
        blocks.  ``cont`` is the block the code jumps to when it runs off
        the end of its region (None only at the very end of the function,
        which never falls off).  ``exits`` holds the jump target of a
        ``mir.Break`` per enclosing block, innermost last:
        ``exits[-level]`` is the block just after the ``End`` of the
        ``level``-th enclosing block (a ``mir.Block`` or ``mir.If``)."""
        i = start
        while i < end:
            inst = insts[i]
            if isinstance(inst, mir.Ret):
                if inst.value is None:
                    block.ret(None)
                else:
                    block.ret(self._value(inst.value, arg_values))
                return
            if isinstance(inst, mir.Break):
                level = inst.level
                if level < 1 or level > len(exits):
                    raise CompileError(f'internal error: break level {level} out of range')
                target = exits[-level]
                assert target is not None
                block.jmp(target)
                return
            if isinstance(inst, (mir.If, mir.Block)):
                has_else = isinstance(inst, mir.If)
                p_else, p_end = self._scan_block(insts, i, has_else)
                after = p_end + 1
                cont_block = sllvm.BasicBlock() if after < end else cont
                inner_exits = exits + (cont_block,)
                if has_else:
                    cond = self._value(inst.cond, arg_values)
                    then_block = sllvm.BasicBlock()
                    if p_else is None:
                        # an empty else branch: a false condition goes to
                        # the code after the ``If``
                        assert cont_block is not None
                        block.br(cond, then_block, cont_block)
                        self._lower_region(
                            llvm_fn, then_block, insts, i + 1, p_end,
                            arg_values, cont_block, inner_exits,
                        )
                    else:
                        else_block = sllvm.BasicBlock()
                        block.br(cond, then_block, else_block)
                        self._lower_region(
                            llvm_fn, then_block, insts, i + 1, p_else,
                            arg_values, cont_block, inner_exits,
                        )
                        self._lower_region(
                            llvm_fn, else_block, insts, p_else + 1, p_end,
                            arg_values, cont_block, inner_exits,
                        )
                else:
                    # a ``Block`` is entered unconditionally
                    self._lower_region(
                        llvm_fn, block, insts, i + 1, p_end,
                        arg_values, cont_block, inner_exits,
                    )
                if after < end:
                    assert cont_block is not None
                    self._lower_region(
                        llvm_fn, cont_block, insts, after, end,
                        arg_values, cont, exits,
                    )
                return
            if isinstance(inst, mir.Alloca):
                # already lowered into the entry block (see ``lower``)
                i += 1
                continue
            self._lower_inst(block, inst, arg_values)
            i += 1
        if not block._finished:
            if cont is None:
                # the body of the function proper fell off its end: an
                # implicit ``ret void`` (only a void function may do so)
                block.ret(None)
            else:
                block.jmp(cont)

    def _value(self, value: mir.Value, arg_values: tuple[sllvm.Value, ...]) -> sllvm.Value:
        if isinstance(value, mir.Param):
            return arg_values[value.index]
        if isinstance(value, mir.Inst):
            ret = self._lowered.get(id(value))
            if ret is None:
                raise CompileError('internal error: instruction not lowered yet')
            return ret
        if isinstance(value, mir.BoolValue):
            return sllvm.IntValue(1 if value.value else 0, sllvm.IntType(1))
        if isinstance(value, mir.Int):
            return sllvm.IntValue(value.value, sllvm.IntType(value.type.bits))
        if isinstance(value, mir.Float):
            return sllvm.FloatType(value.type.bits).from_float(value.value)
        if isinstance(value, (mir.Function, mir.ExternAnonSymbol, mir.ExternSymbol)):
            return self._global(value)
        raise CompileError(f'cannot lower value {value!r}')

    def _global(self, value: mir.GlobalValue) -> sllvm.Value:
        """The lowered form of a function value: the in-module
        ``define`` of a :class:`mir.Function`, or a (cached) extern
        declaration of a symbol compiled in an earlier module."""
        if isinstance(value, mir.Function):
            llvm_fn = self._llvm_fns.get(value)
            if llvm_fn is None:
                raise CompileError(
                    f'internal error: function {value.name_base} is not part of the module'
                )
            return llvm_fn
        decl = self._declarations.get(value)
        if decl is None:
            type = value.get_type()
            if isinstance(type, mir.PointerType):
                type = type.elem
            if not isinstance(type, mir.FunctionType):
                raise CompileError(f'cannot lower the symbol {value!r}')
            fn_type = sllvm.fn_type(
                self._to_llvm(type.return_type),
                *(self._to_llvm(a) for a in type.args),
            )
            decl = sllvm.DeclareFunction(self._link_name(value), fn_type)
            self._declarations[value] = decl
        return decl

    def _lower_inst(
        self,
        block: sllvm.BasicBlock,
        inst: mir.Inst,
        arg_values: tuple[sllvm.Value, ...],
    ) -> None:
        result: sllvm.Value | None = None
        match inst:
            case mir.Nop():
                return
            case mir.Alloca():
                result = block.alloca(self._to_llvm(inst.type))
            case mir.Store():
                block.store(
                    self._value(inst.ptr, arg_values), self._value(inst.value, arg_values)
                )
            case mir.Load():
                result = block.load(self._value(inst.ptr, arg_values))
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
                    case 'fptosi':
                        result = block.emit(sllvm.FloatToInt(value, to))  # type: ignore[arg-type]
                    case 'fptoui':
                        result = block.emit(sllvm.FloatToUInt(value, to))  # type: ignore[arg-type]
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
                callee = self._value(inst.callee, arg_values)
                result = block.call(
                    callee, *(self._value(a, arg_values) for a in inst.args)
                )
            case _:
                raise CompileError(f'unsupported MIR instruction {type(inst).__name__}')
        if result is not None:
            self._lowered[id(inst)] = result


def _py_entry_thunk(
    types: _ModuleTypes,
    value_fn: sllvm.Function,
    fn: mir.Function,
    out_struct_ret: bool,
    link_name: str,
) -> sllvm.Function:
    """The Python-facing entry of one MIR function whose value form
    ctypes cannot call directly: a pointer-form wrapper that loads every
    by-value struct argument, calls the by-value function and - for a
    by-value struct result - stores the returned struct into a trailing
    out buffer.  Spy-to-spy calls never go through it."""
    thunk = sllvm.Function(f'{link_name}.py')
    arg_types = tuple(
        sllvm.PointerType(types.to_llvm(a)) if isinstance(a, mir.StructType) else types.to_llvm(a)
        for a in fn.args
    )
    args = thunk.add_args(*arg_types)
    out: sllvm.Value | None = None
    if out_struct_ret:
        assert isinstance(fn.ret_type, mir.StructType)
        out = thunk.get_arg(thunk.add_arg(sllvm.PointerType(types.to_llvm(fn.ret_type))))
        thunk.set_return_type(sllvm.VoidType())
    else:
        thunk.set_return_type(types.to_llvm(fn.ret_type))
    block = thunk.entry
    call_args = tuple(
        block.load(arg) if isinstance(a, mir.StructType) else arg
        for a, arg in zip(fn.args, args)
    )
    if out_struct_ret:
        assert out is not None
        block.store(out, block.call(value_fn, *call_args))
        block.ret(None)
    elif isinstance(fn.ret_type, mir.VoidType):
        block.call(value_fn, *call_args)
        block.ret(None)
    else:
        block.ret(block.call(value_fn, *call_args))
    return thunk


class _NativeFn(NativeFn):
    """The concrete compiled artifact of one specialization: the ctypes
    entry bound to its Python-facing ABI and the address of its value
    form (what other native modules link against)."""

    def __init__(
        self,
        name: str,
        arg_types: tuple[mir.Type, ...],
        ret_type: mir.MayBeVoidType,
        lines: list[str],
        addr: int,
        entry: object,
        arg_ctypes: list[Any],
        out_struct_type: type[ctypes.Structure] | None,
    ) -> None:
        self.name = name
        self.arg_types = arg_types
        self.ret_type = ret_type
        self._lines = lines
        self._addr = addr
        self._entry = entry
        self._arg_ctypes = arg_ctypes
        self._out_struct_type = out_struct_type

    @property
    def addr(self) -> int:
        return self._addr

    def call(self, *values: ctypes._CDataType) -> ctypes._CDataType | None:
        converted: list[object] = []
        for ctype, value in zip(self._arg_ctypes, values):
            if ctype is ctypes.c_void_p and isinstance(value, ctypes.Structure):
                converted.append(ctypes.addressof(value))
            else:
                converted.append(value)
        assert self._entry is not None
        if self._out_struct_type is not None:
            out = self._out_struct_type()
            self._entry(*converted, ctypes.addressof(out))  # type: ignore[operator]
            return out
        return self._entry(*converted)  # type: ignore[operator]

    def print_all(self) -> list[str]:
        return self._lines


class LLVMBackend(Backend):
    """The MCJIT backend of the spy compiler: compiles one MIR module at
    a time, remembers the LLVM global value of every MIR global it has
    lowered and the address of every symbol it has emitted, so that a
    later module can resolve the symbols it imports."""

    def __init__(self) -> None:
        super().__init__()
        # every MIR global value that has been lowered, mapped to its
        # LLVM counterpart in the module it was lowered into
        self._globals: dict[mir.GlobalValue, sllvm.GlobalValue] = {}
        # the address of every emitted symbol, by its link name
        self._exported: dict[str, int] = {}
        # the engines are kept alive so the addresses stay valid
        self._engines: list[object] = []

    def compile(self, module: mir.Module) -> dict[mir.GlobalValue, NativeFn]:
        symbols = module.finish()
        fns = [v for v in symbols.values() if isinstance(v, mir.Function)]
        types = _ModuleTypes()

        # the MIR symbol table has already assigned every symbol its
        # unique name; sanitize it into an LLVM identifier and hand it to
        # the LLVM builder, whose own symbol table resolves any remaining
        # collision
        def link_name(value: mir.GlobalValue) -> str:
            return sanitize_name(symbols.get_key(value))

        # create one sllvm.Function per MIR function up front: bodies may
        # call any function of the module, and the call sites need the
        # callee's definition (its signature) to type the call
        llvm_fns: dict[mir.Function, sllvm.Function] = {}
        for fn in fns:
            llvm_fn = sllvm.Function(link_name(fn))
            llvm_fn.add_args(*(types.to_llvm(a) for a in fn.args))
            llvm_fn.set_return_type(types.to_llvm(fn.ret_type))
            llvm_fns[fn] = llvm_fn
            self._globals[fn] = llvm_fn

        lowerer = _Lowerer(llvm_fns, types, symbols)
        for fn in fns:
            lowerer.lower(fn)
        for sym, decl in lowerer.declarations.items():
            self._globals[sym] = decl

        # the Python-facing entry of every function whose value form
        # ctypes cannot call directly (see ``_py_entry_thunk``)
        entry_by_fn: dict[mir.Function, sllvm.Function] = {}
        for fn in fns:
            out_struct_ret = isinstance(fn.ret_type, mir.StructType)
            has_struct_arg = any(isinstance(a, mir.StructType) for a in fn.args)
            if out_struct_ret or has_struct_arg:
                entry_by_fn[fn] = _py_entry_thunk(
                    types, llvm_fns[fn], fn, out_struct_ret, link_name(fn)
                )

        lmod = sllvm.Module()
        module_values: list[sllvm.Value] = [llvm_fns[fn] for fn in fns]
        module_values.extend(entry_by_fn.values())
        lmod.add_recursively(values=module_values)
        lmod.add_recursively(types=types.struct_types())
        lmod.finish()
        lines = lmod.write()

        target = llvm.Target.from_default_triple()
        tm = target.create_target_machine()
        llvm_mod = llvm.parse_assembly('\n'.join(lines))
        llvm_mod.verify()

        backing_mod = llvm.parse_assembly('')
        engine = llvm.create_mcjit_compiler(backing_mod, tm)
        engine.add_module(llvm_mod)
        module_names = ', '.join(sorted(fn.name_base for fn in fns))
        for f in llvm_mod.functions:
            if not f.is_declaration:
                continue
            addr = self._exported.get(f.name)
            if addr is None:
                raise CompileError(
                    f'cannot resolve the external symbol {f.name} referenced '
                    f'by the module ({module_names})'
                )
            engine.add_global_mapping(f, addr)
        engine.finalize_object()
        engine.run_static_constructors()
        self._engines.append(engine)

        rets: dict[mir.GlobalValue, NativeFn] = {}
        for fn in fns:
            # the link name the LLVM module actually emitted (it resolves
            # a collision itself, so read it back rather than assume it)
            name = lmod.get_global_name(llvm_fns[fn])
            value_addr = engine.get_function_address(name)
            entry_fn = entry_by_fn.get(fn)
            if entry_fn is not None:
                entry_name = lmod.get_global_name(entry_fn)
                entry_addr = engine.get_function_address(entry_name)
            else:
                entry_addr = value_addr
            arg_ctypes: list[Any] = [py_entry_arg_ctype(a) for a in fn.args]
            out_struct_type: type[ctypes.Structure] | None = None
            if isinstance(fn.ret_type, mir.StructType):
                arg_ctypes.append(ctypes.c_void_p)
                out_struct_type = struct_ctype(fn.ret_type)
            restype = to_ctype(fn.ret_type)
            proto = ctypes.CFUNCTYPE(restype, *arg_ctypes)  # type: ignore[arg-type]
            entry = ctypes.cast(entry_addr, proto)
            native = _NativeFn(
                name,
                tuple(fn.args),
                fn.ret_type,
                lines,
                value_addr,
                entry,
                arg_ctypes,
                out_struct_type,
            )
            rets[fn] = native
            self._exported[name] = value_addr
        return rets

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Self, override

from .binop import BinaryOp, CompareOp
from .errors import CompileError

# ---------------------------------------------------------------------------
# static types
# ---------------------------------------------------------------------------


class Type:
    @abstractmethod
    def get_children(self) -> tuple[Any, ...]:
        ...


class VoidType:
    """The void type, representing no value."""

    def __repr__(self) -> str:
        return 'void'

    def get_children(self) -> tuple[Any, ...]:
        return ()

VOID = VoidType()

type MayBeVoidType = Type | VoidType

@dataclass(frozen=True)
class BoolType(Type):
    """Booleans; they are ``i1`` at the LLVM level."""

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type


class StructType(Type):
    """The static type of a struct value (and of the elements of struct
    storage).  The type is an identity object mirroring one spy struct
    type (``sval.StructType``); two structs are equal only when they are
    the same object, which is what keeps the types of one struct apart
    from an accidentally identical one.

    The fields are the fields of the *mirror*: ``fields[i]`` is the
    :class:`FormalArg` (name and type) of the i-th field of the mirror.
    A spy struct orders its fields by alignment and drops the zero-sized
    ones, while an ``extern_c`` struct keeps the declaration order (see
    ``sval.StructType._calculate_mir``).

    ``ctype`` is the ctypes class mirroring the LLVM layout of the
    struct (a plain ``ctypes.Structure`` subclass built from the MIR
    fields, zero-sized spy fields excluded - they occupy no storage);
    it is materialized on demand by ``lower`` when a value of the type
    crosses the Python boundary, and cached here once built.
    """

    def __init__(
        self,
        name_base: str | None,
        fields: tuple[FormalArg, ...] | None,
    ) -> None:
        self.name_base = name_base
        self.fields: list[FormalArg] = []
        if fields is not None:
            self.fields.extend(fields)
        # the ctypes class mirroring the LLVM layout of this struct (what
        # a struct value crossing the Python boundary is viewed as);
        # materialized on demand and cached by ``lower``
        self.ctype: Any = None

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    def get_children(self) -> tuple[Any, ...]:
        return tuple(f.type for f in self.fields)


@dataclass(frozen=True)
class PointerType(Type):
    elem: MayBeVoidType
    is_const: bool = False

    def get_children(self) -> tuple[Any, ...]:
        return (self.elem,)

@dataclass(frozen=True)
class ArrayType(Type):
    elem: Type
    length: int

    def get_children(self) -> tuple[Any, ...]:
        return (self.elem,)

@dataclass(frozen=True)
class FunctionType(Type):
    """The signature of a function value (the element type of its
    pointer)."""

    args: tuple[Type, ...]
    return_type: MayBeVoidType

    def get_children(self) -> tuple[Any, ...]:
        return (*self.args, self.return_type)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


class Value:
    @abstractmethod
    def get_type(self) -> MayBeVoidType:
        ...

    @abstractmethod
    def get_children(self) -> tuple[Any, ...]:
        ...

@dataclass(frozen=True)
class BoolValue(Value):
    value: bool

    @override
    def get_type(self) -> MayBeVoidType:
        return BoolType()

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class Int(Value):
    value: int
    type: IntType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class Float(Value):
    value: float
    type: FloatType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return ()


@dataclass(frozen=True)
class NullValue(Value):
    """The null pointer constant: the value an absent ``Option[T]`` whose
    representation is a pointer (or holds one as its tag) has."""

    type: PointerType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)


class GlobalValue(Value):
    def __hash__(self) -> int:
        return object.__hash__(self)

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def get_children(self) -> tuple[Any, ...]:
        return ()

    @abstractmethod
    def get_name(self) -> tuple[str, bool]:
        """Returns (name, can_be_renamed)"""
        ...

@dataclass(frozen=True)
class ExternSymbol(GlobalValue):
    """External symbol, used to import/export symbol from/into the symbol table. Not currently used, but will be used in the future."""
    name: str
    type: Type

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)

    @override
    def get_name(self) -> tuple[str, bool]:
        return self.name, False


class ExternAnonSymbol(GlobalValue):
    """An anonymous external symbol, used to import previously compiled spy function."""
    def __init__(self, name_base: str, type: Type) -> None:
        self.name_base = name_base
        self.type = type

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)

    @override
    def get_name(self) -> tuple[str, bool]:
        return self.name_base, True

@dataclass(eq=False)
class Param(Value):
    """The index-th formal argument of the enclosing function's lowered
    signature - a by-value parameter, a by-reference one, or the hidden
    result pointer."""

    index: int
    type: Type
    name: str = ''

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)


class Inst(Value):
    """A MIR instruction; the object itself acts as its result register
    (instructions have identity, mirroring ``llvm``)."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> MayBeVoidType:
        return VOID

    def get_children(self) -> tuple[Any, ...]:
        return ()

    @abstractmethod
    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ...

@dataclass(eq=False)
class Alloca(Inst):
    """Allocate a slot for one value; produces a pointer to ``type``.  An
    ``Alloca`` may sit in any block; the lowerer hoists every one of them
    into the function's entry block, so its position carries no meaning."""

    type: Type

    @override
    def get_type(self) -> MayBeVoidType:
        return PointerType(self.type)

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


@dataclass(eq=False)
class Load(Inst):
    ptr: Value

    @override
    def get_type(self) -> MayBeVoidType:
        ptr_type = self.ptr.get_type()
        assert isinstance(ptr_type, PointerType) and ptr_type.elem is not None
        return ptr_type.elem

    def get_children(self) -> tuple[Any, ...]:
        return (self.ptr,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        return self if ptr is self.ptr else replace(self, ptr=ptr)


@dataclass(eq=False)
class Store(Inst):
    ptr: Value
    value: Value

    def get_children(self) -> tuple[Any, ...]:
        return (self.ptr, self.value)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        value = f(self.value)
        if ptr is self.ptr and value is self.value:
            return self
        return replace(self, ptr=ptr, value=value)


@dataclass(eq=False)
class Gep(Inst):
    """The address of a struct field or of an array element: ``ptr`` must
    point at a struct or an array value and ``index`` names what the address is
    taken of - the position of a field in the *mirror* of the struct (the
    declaration index mapped by ``sval.StructType.get_field_mir_indices``,
    always a constant), or the position of an element of an array (a constant,
    or a value when the index is only known at runtime).  The result is a
    pointer to the field/the element; its type is computed here from the static
    type of ``ptr`` (mirroring LLVM's ``getelementptr``)."""

    ptr: Value
    index: int | Value

    def __init__(self, ptr: Value, index: int | Value) -> None:
        self.ptr = ptr
        self.index = index
        ptype = ptr.get_type()
        if not isinstance(ptype, PointerType):
            raise CompileError(f'cannot take an element of a {ptype} value')
        elem = ptype.elem
        if isinstance(elem, StructType):
            if not isinstance(index, int):
                raise CompileError(f'a field of {elem} is taken by a constant index')
            if index < 0 or index >= len(elem.fields):
                raise CompileError(f'field index {index} is out of bounds for {elem}')
            self.type: Type = PointerType(elem.fields[index].type)
            return
        if isinstance(elem, ArrayType):
            if isinstance(index, int) and not 0 <= index < elem.length:
                raise CompileError(f'element index {index} is out of bounds for {elem}')
            self.type = PointerType(elem.elem)
            return
        raise CompileError(
            f'cannot take an element of a {ptype} value '
            '(element access requires a struct or array value)'
        )

    @override
    def get_type(self) -> Type:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        if isinstance(self.index, int):
            return (self.ptr,)
        return (self.ptr, self.index)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        index = self.index if isinstance(self.index, int) else f(self.index)
        if ptr is self.ptr and index is self.index:
            return self
        return replace(self, ptr=ptr, index=index)


@dataclass(eq=False)
class Arith(Inst):
    """Integer/float arithmetic.

    ``op`` is one of ``'+'``, ``'-'``, ``'*'`` (integer or float,
    chosen by ``type``), ``'/'`` (float) and ``'%'`` (integer).  Integer
    division/remainder honor ``signed``.
    """

    op: BinaryOp
    signed: bool
    lhs: Value
    rhs: Value
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.lhs, self.rhs)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        lhs = f(self.lhs)
        rhs = f(self.rhs)
        if lhs is self.lhs and rhs is self.rhs:
            return self
        return replace(self, lhs=lhs, rhs=rhs)


@dataclass(eq=False)
class Convert(Inst):
    """A value conversion: 'sitofp', 'uitofp', 'fpext', 'fptrunc',
    'zext', 'sext', 'trunc', 'fptosi' or 'fptoui'."""

    kind: str
    value: Value
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False)
class Cmp(Inst):
    """A comparison producing a bool; ``op`` is one of '==', '!=', '<',
    '<=', '>', '>='."""

    op: CompareOp
    signed: bool
    kind: str  # 'int' or 'float'
    lhs: Value
    rhs: Value

    @override
    def get_type(self) -> MayBeVoidType:
        return BoolType()

    def get_children(self) -> tuple[Any, ...]:
        return (self.lhs, self.rhs)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        lhs = f(self.lhs)
        rhs = f(self.rhs)
        if lhs is self.lhs and rhs is self.rhs:
            return self
        return replace(self, lhs=lhs, rhs=rhs)


@dataclass(eq=False)
class Call(Inst):
    """A call of a function value returning a value of type ``type``
    (``mir.VOID`` for a call of a void function, which produces no
    result).  The callee is either a :class:`Function` (compiled in the
    same LLVM module) or a :class:`GlobalValue` of an earlier module
    (:class:`ExternAnonSymbol`, or an :class:`ExternSymbol` resolved from
    the process)."""

    callee: Value
    args: tuple[Value, ...]
    type: MayBeVoidType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.callee, *self.args)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        callee = f(self.callee)
        args = tuple(f(arg) for arg in self.args)
        if callee is self.callee and all(a is b for a, b in zip(args, self.args)):
            return self
        return replace(self, callee=callee, args=args)


# ---------------------------------------------------------------------------
# basic blocks
# ---------------------------------------------------------------------------

class BasicBlock:
    """A basic block of the MIR: a straight-line run of instructions that
    ends with a :class:`Terminator` (a ``jmp``, a ``br`` or a ``ret``).

    Like ``llvm.BasicBlock`` it is *not* a value and is *not* registered
    anywhere: the blocks of a function are reached by walking the outgoing
    edges from its entry block (see :meth:`collect_blocks`), and a block
    that no path reaches is simply never lowered.
    """

    def __init__(self) -> None:
        self.insts: list[Inst] = []
        # set once the terminator of the block has been emitted; a block
        # whose path ends in a pending return insertion is finished too
        self.is_finished = False

    def emit(self, inst: Inst) -> Inst:
        """Append one instruction to this block.  A terminator ends the
        block, so nothing may follow it - the assertion catches an
        instruction that would end up on a dead path."""
        assert not self.is_finished, 'cannot emit into a finished basic block'
        self.insts.append(inst)
        if isinstance(inst, Terminator):
            self.is_finished = True
        return inst

    def get_outgoing_blocks(self) -> tuple[BasicBlock, ...]:
        """The successors of this block: the targets of its terminator.
        A block that is still being built - or whose path ends in a
        pending insertion - has none."""
        if len(self.insts) == 0:
            return ()
        last = self.insts[-1]
        if isinstance(last, Terminator):
            return last.get_targets()
        return ()

    def collect_blocks(self) -> list[BasicBlock]:
        """Every block reachable from this one - this block first, then
        its successors in the order a depth-first walk reaches them."""
        ret: list[BasicBlock] = []
        seen: set[BasicBlock] = set()
        todo: list[BasicBlock] = [self]
        while len(todo) > 0:
            block = todo.pop()
            if block in seen:
                continue
            seen.add(block)
            ret.append(block)
            children = list(block.get_outgoing_blocks())
            children.reverse()
            todo.extend(children)
        return ret


class Terminator(Inst):
    """The transfer instruction that ends a basic block: a :class:`Jmp`, a
    :class:`Br` or a :class:`Ret`.  It produces no value, and its targets
    are blocks rather than values, so ``get_children`` never returns
    them."""

    @abstractmethod
    def get_targets(self) -> tuple[BasicBlock, ...]:
        """The blocks this terminator transfers control to."""
        ...


@dataclass(eq=False)
class Jmp(Terminator):
    """An unconditional jump to ``target``."""

    target: BasicBlock

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.target,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


@dataclass(eq=False)
class Br(Terminator):
    """A two-way conditional branch: control goes to ``if_true`` when
    ``cond`` holds and to ``if_false`` otherwise."""

    cond: Value
    if_true: BasicBlock
    if_false: BasicBlock

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.if_true, self.if_false)

    def get_children(self) -> tuple[Any, ...]:
        return (self.cond,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        cond = f(self.cond)
        return self if cond is self.cond else replace(self, cond=cond)


@dataclass(eq=False)
class Ret(Terminator):
    """Return from the enclosing function; ends the path of its block.
    ``value`` is None for a void return (a ``ret void``)."""

    value: Value | None

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return ()

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,) if self.value is not None else ()

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        if self.value is None:
            return self
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False)
class Insertion(Inst):
    """A placeholder standing for the instructions that deliver a slot's
    pending action (or its storage), and, when ``value`` is set, for the
    value those instructions define.  It is emitted at a position of the
    body before its type is known and is *spliced* there - into the block
    it sits in - once it has been filled in; :func:`normalize` does the
    flattening and substitutes every reference to it."""

    insts: list[Inst]
    value: Value | None
    type: MayBeVoidType = VOID

    @override
    def get_type(self) -> MayBeVoidType:
        return self.value.get_type() if self.value is not None else self.type

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        if self.value is None:
            return self
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


def _flatten(insts: list[Inst]) -> list[Inst]:
    """Replace every :class:`Insertion` of ``insts`` by its instructions
    (recursively), preserving the order - iteratively, so that a deep
    nesting cannot exhaust the Python stack."""
    out: list[Inst] = []
    pending: list[tuple[list[Inst], int]] = [(insts, 0)]
    while pending:
        cur, index = pending.pop()
        while index < len(cur):
            inst = cur[index]
            index += 1
            if isinstance(inst, Insertion):
                if index < len(cur):
                    pending.append((cur, index))
                cur = inst.insts
                index = 0
                continue
            out.append(inst)
    return out


def normalize(fn: Function) -> None:
    """Eliminate the :class:`Insertion` placeholders of every block of
    ``fn`` by flattening them and substituting their values, then check
    that every block ends with a terminator."""
    blocks = fn.entry.collect_blocks()

    # every insertion that stands for a value maps to that value, so that
    # the operands referring to the insertion are rewritten to it
    repl: dict[Value, Value] = {}
    todo: list[list[Inst]] = [block.insts for block in blocks]
    while todo:
        insts = todo.pop()
        for inst in insts:
            if isinstance(inst, Insertion):
                if inst.value is not None:
                    repl[inst] = inst.value
                todo.append(inst.insts)

    def resolve(value: Value) -> Value:
        # a replacement chain (an insertion resolved to another insertion, an
        # instruction to its mapped copy) is followed iteratively
        seen: set[Value] = set()
        while value in repl:
            if value in seen:
                raise CompileError('cycle in the MIR substitution')
            seen.add(value)
            value = repl[value]
        if isinstance(value, Insertion):
            raise CompileError('a MIR insertion is referenced before it is resolved')
        return value

    for block in blocks:
        block.insts = _flatten(block.insts)

    # an operand that resolves to another instruction replaces that
    # instruction everywhere it is used, so the rewrite is iterated to a
    # fixpoint (a value is always defined before the instructions it feeds)
    changed = True
    while changed:
        changed = False
        for block in blocks:
            for index, inst in enumerate(block.insts):
                mapped = inst.map_values(resolve)
                if mapped is not inst:
                    block.insts[index] = mapped
                    repl[inst] = mapped
                    changed = True

    for block in blocks:
        if len(block.insts) == 0 or not isinstance(block.insts[-1], Terminator):
            raise CompileError(
                'every basic block must end with a jump, a branch or a return'
            )
        block.is_finished = True


@dataclass(eq=False)
class Function(GlobalValue):
    """One compiled MIR function.  As a value it is the in-module
    function value of a call target: a call whose callee is this object
    is lowered to a call of the ``define``d function (functions of one
    module are compiled together).

    ``entry`` is the entry basic block; the rest of the body is reached
    from it (``entry.collect_blocks()``).  Every block ends with a
    terminator, so the CFG is complete."""

    name_base: str
    args: list[Type]
    arg_names: list[str | None]
    ret_type: MayBeVoidType
    entry: BasicBlock = field(default_factory=BasicBlock)
    is_complete: bool = False
    impose_linkname: bool = False

    @override
    def get_type(self) -> Type:
        """The pointer type of the function's lowered MIR signature (the
        hidden result pointer included).  The callee side of a call in
        the MIR is the function object itself, so this view exists for
        the declaration an imported symbol needs."""
        assert self.is_complete
        return PointerType(FunctionType(tuple(self.args), self.ret_type), True)

    @override
    def get_name(self) -> tuple[str, bool]:
        return self.name_base, not self.impose_linkname

    def get_children(self) -> tuple[Any, ...]:
        """Everything this function references: the argument and return
        types of its lowered signature, and the operands of its
        instructions.  Instruction operands (registers) are flattened
        away - the instructions that define them are part of the body and
        are traversed on their own - and so are the blocks a terminator
        branches to, so the result holds only the values and types the
        module must also declare."""
        ret: list[Type] = list(self.args)
        if not isinstance(self.ret_type, VoidType):
            ret.append(self.ret_type)
        for block in self.entry.collect_blocks():
            for inst in block.insts:
                for child in inst.get_children():
                    if not isinstance(child, Inst):
                        ret.append(child)
        return tuple(ret)

def collect_symbols(entry: list[GlobalValue]) -> set[GlobalValue | StructType]:
    symbols: set[GlobalValue | StructType] = set()
    todo: list[Value | StructType] = [a for a in entry]
    while todo:
        value = todo.pop()
        if value in symbols:
            continue
        if isinstance(value, (StructType, GlobalValue)):
            symbols.add(value)
        todo.extend(reversed([a for a in value.get_children() if not isinstance(a, Inst)]))
    return symbols

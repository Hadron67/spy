"""The typed MIR.

The interpreter ("runs" the untyped HIR) emits a :class:`Function` per
specialization as one *flat* list of typed instructions: the body of
one specialization, with every runtime branch inlined into it and
delimited by the :class:`If`/``Else``/``End`` markers (WASM-style),
mirroring the HIR.  Control flow is structured (no basic blocks, no
phi): a branch that ends in a :class:`Ret` returns on that path, a
branch that does not return falls through to the code after its
``End`` marker in the enclosing list - exactly the shape of control
flow that recursion needs.  An inlined function body is delimited by
:class:`Block`/``End`` markers (a ``Block`` is entered unconditionally,
like a WASM ``block``), and a path may leave it early with a
:class:`Break` (WASM ``br``) - the way an inlined ``return`` leaves the
inlined body; ``If`` and ``Block`` both count as blocks for a
``Break``'s ``level``.

The MIR owns its static type system (:class:`Type`): a closed,
LLVM-shaped universe of the types a runtime register can have.  The
interpreter computes its types in the ``spy`` type system of
``type.py`` (which also has to represent compile-time values - type
descriptors, functions, ... - that never cross into runtime code) and
mirrors them into these types when it emits an instruction.  ``lower``
then maps the flat list onto LLVM basic blocks.

The representation is deliberately close to LLVM so that ``lower`` is a
mechanical mapping; every MIR value exposes a ``.type`` (a MIR type).
A compiled function is a :class:`Function` value: the host creates and
registers it - with an empty body - *before* the body is run, so a call
the body makes to it (recursion) resolves to the very :class:`Function`
being typed; calls within one LLVM module reference the callee's
:class:`Function` (lowered to a ``define``).  A callee compiled in an
*earlier* module is referenced by a :class:`Symbol` (lowered to an
external declaration).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

from typing_extensions import override

from .errors import CompileError

# ---------------------------------------------------------------------------
# static types
# ---------------------------------------------------------------------------


class Type:
    pass


class VoidType:
    """The void type, representing no value."""

    def __repr__(self) -> str:
        return 'void'

VOID = VoidType()

MayBeVoidType: TypeAlias = Type | VoidType

@dataclass(frozen=True)
class BoolType(Type):
    """Booleans; they are ``i1`` at the LLVM level."""


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool


@dataclass(frozen=True)
class FloatType(Type):
    bits: int


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type


class StructType(Type):
    """The static type of a struct value (and of the elements of struct
    storage).  The type is an identity object mirroring one spy struct
    type (``type.StructType``); two structs are equal only when they are
    the same object, which is what keeps the types of one struct apart
    from an accidentally identical one.

    Fields are positional: ``fields[i]`` is the type of the i-th field,
    in declaration order (the LLVM layout of the mirrored ``sllvm``
    struct follows the same order).  ``spy_type`` is a back reference to
    the spy-side descriptor, which carries the field names, the method
    table and the Python-side ctypes class.

    ``ctype`` is the ctypes class mirroring the LLVM layout of the
    struct (a plain ``ctypes.Structure`` subclass built from the MIR
    fields, zero-sized spy fields excluded - they occupy no storage);
    it is materialized on demand by ``lower`` when a value of the type
    crosses the Python boundary, and cached here once built.
    """

    def __init__(
        self,
        spy_type: Any,
        fields: tuple[FormalArg, ...],
    ) -> None:
        self.spy_type = spy_type
        self.fields = fields
        self.ctype: Any = None  # lazily built from the MIR fields by ``lower``

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(frozen=True)
class PointerType(Type):
    elem: MayBeVoidType
    is_const: bool = False

@dataclass(frozen=True)
class FunctionType(Type):
    """The signature of a function value (the element type of its
    pointer)."""

    args: tuple[Type, ...]
    return_type: MayBeVoidType


def type_str(type: Type) -> str:
    """A short printable name of a type (used in error messages).  It
    mirrors the strings of the corresponding ``spy`` types; the mangled
    names of compiled specializations are still built from the ``spy``
    side."""
    match type:
        case BoolType():
            return 'bool'
        case IntType():
            return ('i' if type.signed else 'u') + str(type.bits)
        case FloatType():
            return 'f' + str(type.bits)
        case PointerType(elem):
            return '*' + str(elem)
        case StructType():
            return type.spy_type.name
        case FunctionType(args, ret):
            return f'fn({", ".join(type_str(a) for a in args)}) -> {ret}'
        case _:
            return str(type)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


class Value:
    @abstractmethod
    def get_type(self) -> MayBeVoidType:
        raise NotImplementedError


@dataclass(frozen=True)
class BoolValue(Value):
    value: bool

    @override
    def get_type(self) -> MayBeVoidType:
        return BoolType()


@dataclass(frozen=True)
class Int(Value):
    value: int
    type: IntType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type


@dataclass(frozen=True)
class Float(Value):
    value: float
    type: FloatType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type


@dataclass(frozen=True)
class Symbol(Value):
    """A function value bound to a module symbol of an *earlier* module:
    the target of a native call whose definition is linked in at compile
    time.  Lowered to a ``declare``d external symbol whose address is
    resolved at link time."""

    name: str
    fn_type: FunctionType

    @override
    def get_type(self) -> MayBeVoidType:
        return PointerType(self.fn_type)


@dataclass
class Param(Value):
    """The index-th by-value argument of the enclosing function."""

    index: int
    type: Type
    name: str = ''

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type


class Inst(Value):
    """A MIR instruction; the object itself acts as its result register
    (instructions have identity, mirroring ``symlat.jit.llvm``)."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> MayBeVoidType:
        return VOID

@dataclass(eq=False)
class Alloca(Inst):
    """Allocate a slot for one value; produces a pointer to ``type``."""

    type: Type

    @override
    def get_type(self) -> MayBeVoidType:
        return PointerType(self.type)


@dataclass(eq=False)
class Load(Inst):
    ptr: Value

    @override
    def get_type(self) -> MayBeVoidType:
        ptr_type = self.ptr.get_type()
        assert isinstance(ptr_type, PointerType) and ptr_type.elem is not None
        return ptr_type.elem


@dataclass(eq=False)
class Store(Inst):
    ptr: Value
    value: Value


@dataclass(eq=False)
class Gep(Inst):
    """The address of a struct field: ``ptr`` must point at a struct
    value and ``index`` names the field (by declaration index).  The
    result is a pointer to the field; its type is computed here from the
    static type of ``ptr`` (mirroring LLVM's ``getelementptr``)."""

    ptr: Value
    index: int

    def __init__(self, ptr: Value, index: int) -> None:
        self.ptr = ptr
        self.index = index
        ptype = ptr.type  # type: ignore[attr-defined]
        if not isinstance(ptype, PointerType) or not isinstance(ptype.elem, StructType):
            raise CompileError(
                f'cannot take a field of a {type_str(ptype)} value '
                '(field access requires a struct value)'
            )
        self.type: Type = PointerType(ptype.elem.fields[index].type)


@dataclass(eq=False)
class Arith(Inst):
    """Integer/float arithmetic.

    ``op`` is one of ``'add'``, ``'sub'``, ``'mul'`` (integer or float,
    chosen by ``type``), ``'div'`` and ``'rem'`` (float division is
    ``'div'`` with a float result type).  Integer division/remainder
    honor ``signed``.
    """

    op: str
    signed: bool
    lhs: Value
    rhs: Value
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type


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


@dataclass(eq=False)
class Cmp(Inst):
    """A comparison producing a bool; ``op`` is one of 'eq', 'ne', 'lt',
    'le', 'gt', 'ge'."""

    op: str
    signed: bool
    kind: str  # 'int' or 'float'
    lhs: Value
    rhs: Value

    @override
    def get_type(self) -> MayBeVoidType:
        return BoolType()


@dataclass(eq=False)
class Call(Inst):
    """A call of a function value returning a value of type ``type``
    (``None`` for a call of a void function, which produces no
    result).  The callee is either a :class:`Function` (compiled in the
    same LLVM module) or a :class:`Symbol` (compiled in an earlier
    module)."""

    callee: Value
    args: tuple[Value, ...]
    type: MayBeVoidType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type


@dataclass(eq=False)
class Ret(Inst):
    """Return from the enclosing function; ends its path (no code of
    the enclosing block after a return is emitted).  ``value`` is None
    for a void return (a ``ret void``)."""

    value: Value | None


@dataclass(eq=False)
class Nop(Inst):
    """A no-op instruction; used to reserve space in the MIR body for
    a future instruction to be emitted at a known position."""



@dataclass(eq=False)
class If(Inst):
    """A runtime branch typed by the interpreter (WASM-style marker):
    the instructions of the two branches follow it in the same list,
    delimited by the matching :class:`Else` (when the ``if`` has an
    else branch) and :class:`End` markers.  A branch that ends in a
    :class:`Ret` (or a :class:`Break` that leaves it) returns on that
    path; a branch that does not return falls through to the code after
    the matching ``End`` (the interpreter only emits code after an
    ``If`` that is reachable)."""

    cond: Value


@dataclass(eq=False)
class Block(Inst):
    """An anonymous code block (WASM-style marker), opened by the
    interpreter around the body of every inlined function: the body
    follows it in the same list, closed by the matching :class:`End`
    marker.  Unlike an :class:`If`, a ``Block`` is entered
    unconditionally - it produces no branch of its own - but a
    :class:`Break` may leave it early (the way an inlined ``return``
    leaves the inlined body before it ends)."""


@dataclass(eq=False)
class Else(Inst):
    """The marker that starts the else branch of an :class:`If` (absent
    when the ``if`` has no else branch).  It produces no value; it only
    delimits the flat instruction list."""


@dataclass(eq=False)
class End(Inst):
    """The marker that closes a block opened by an :class:`If` or a
    :class:`Block`.  It produces no value; it only delimits the flat
    instruction list."""


@dataclass(eq=False)
class Break(Inst):
    """Leave ``level`` enclosing blocks (a :class:`Block` or an
    :class:`If` each count as one block, the innermost enclosing block
    being ``level`` 1 - the WASM ``br level - 1``) and continue with
    the code just after the ``End`` of the last block left, skipping
    any code of the exited blocks in between.  Like a :class:`Ret`, a
    ``Break`` ends the path of the region it sits in: the interpreter
    emits no code of the same region after it (the regions of the
    exited blocks - sibling branches, code after their ``End`` - are
    emitted after the ``Break`` and are still lowered)."""

    level: int


@dataclass(eq=False)
class Function(Value):
    """One compiled MIR function.  As a value it is the in-module
    function value of a call target: a call whose callee is this object
    is lowered to a call of the ``define``d function (functions of one
    module are compiled together)."""

    name: str
    args: tuple[Type, ...]
    arg_names: tuple[str | None, ...]
    ret_type: MayBeVoidType
    insts: list[Inst]

    @override
    def get_type(self) -> Type:
        """The type of the function value: a pointer to the function's
        (logical) signature - the callee side of calls in the MIR is
        always the *lowered* form, so this logical view is only used by
        the host."""
        return PointerType(FunctionType(self.args, self.ret_type), True)

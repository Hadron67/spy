"""The typed MIR.

The interpreter ("runs" the untyped HIR) emits a :class:`Function` per
specialization: a *tree* of straight-line regions.  A region is a list
of typed instructions; a :class:`Ret` returns on that path, and a
runtime :class:`If` carries two sub-regions.  Control flow is
structured (no basic blocks, no phi): a branch that does not return
falls through to the code after the ``If`` in its enclosing region,
which is exactly the shape of control flow that recursion needs.

The MIR owns its static type system (:class:`Type`): a closed,
LLVM-shaped universe of the types a runtime register can have.  The
interpreter computes its types in the ``spy`` type system of
``type.py`` (which also has to represent compile-time values - type
descriptors, functions, ... - that never cross into runtime code) and
mirrors them into these types when it emits an instruction.  ``lower``
then maps the region tree onto LLVM basic blocks.

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

from dataclasses import dataclass
from typing import Any

from .errors import CompileError

# ---------------------------------------------------------------------------
# static types
# ---------------------------------------------------------------------------


class Type:
    pass


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
class VoidType(Type):
    """The return type of a void function (its body returns no value).
    No runtime value ever has this type; a call of a void function
    produces no result."""


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
    """

    def __init__(
        self,
        spy_type: Any,
        fields: tuple[FormalArg, ...],
    ) -> None:
        self.spy_type = spy_type
        self.fields = fields
        # the ctypes Structure subclass mirroring the struct layout
        # (the Python class of the instances, ``spy_type._py_cls``)
        self.ctype: Any = None

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: bool = False

@dataclass(frozen=True)
class FunctionType(Type):
    """The signature of a function value (the element type of its
    pointer)."""

    args: tuple[Type, ...]
    return_type: Type


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
        case VoidType():
            return 'void'
        case PointerType(elem):
            return '*' + type_str(elem)
        case StructType():
            return type.spy_type.name
        case FunctionType(args, ret):
            return f'fn({', '.join(type_str(a) for a in args)}) -> {type_str(ret)}'
        case _:
            return str(type)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


class Value:
    pass


@dataclass(frozen=True)
class BoolValue(Value):
    value: bool

    @property
    def type(self) -> Type:
        return BoolType()


@dataclass(frozen=True)
class IntValue(Value):
    value: int
    bits: int
    signed: bool

    @property
    def type(self) -> Type:
        return IntType(self.bits, self.signed)


@dataclass(frozen=True)
class FloatValue(Value):
    value: float
    bits: int

    @property
    def type(self) -> Type:
        return FloatType(self.bits)


@dataclass(frozen=True)
class Symbol(Value):
    """A function value bound to a module symbol of an *earlier* module:
    the target of a native call whose definition is linked in at compile
    time.  Lowered to a ``declare``d external symbol whose address is
    resolved at link time."""

    name: str
    fn_type: FunctionType

    @property
    def type(self) -> Type:
        return PointerType(self.fn_type)


@dataclass
class Param(Value):
    """The index-th by-value argument of the enclosing function."""

    index: int
    type: Type
    name: str = ''


class Inst(Value):
    """A MIR instruction; the object itself acts as its result register
    (instructions have identity, mirroring ``symlat.jit.llvm``)."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(eq=False)
class Alloca(Inst):
    """Allocate a slot for one value; produces a pointer to ``type``."""

    type: PointerType


@dataclass(eq=False)
class Load(Inst):
    ptr: Value
    type: Type


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


@dataclass(eq=False)
class Convert(Inst):
    """A value conversion: 'sitofp', 'uitofp', 'fpext', 'fptrunc',
    'zext', 'sext', 'trunc', 'fptosi' or 'fptoui'."""

    kind: str
    value: Value
    type: Type


@dataclass(eq=False)
class Cmp(Inst):
    """A comparison producing a bool; ``op`` is one of 'eq', 'ne', 'lt',
    'le', 'gt', 'ge'."""

    op: str
    signed: bool
    kind: str  # 'int' or 'float'
    lhs: Value
    rhs: Value

    @property
    def type(self) -> Type:
        return BoolType()


@dataclass(eq=False)
class Call(Inst):
    """A call of a function value returning a value of type ``type``.
    The callee is either a :class:`Function` (compiled in the same LLVM
    module) or a :class:`Symbol` (compiled in an earlier module)."""

    callee: Value
    args: tuple[Value, ...]
    type: Type


@dataclass(eq=False)
class Ret(Inst):
    """Return from the enclosing function; ends its region (no code of
    the region is executed after a return).  ``value`` is None for a
    void return (a ``ret void``)."""

    value: Value | None


@dataclass(eq=False)
class If(Inst):
    """A runtime branch typed by the interpreter.  ``then_body`` and
    ``else_body`` are regions (lists of instructions) that may contain
    further control flow.  A region that ends in a :class:`Ret` returns
    on that path; a region that does not return falls through to the
    code after this ``If`` in the enclosing region."""

    cond: Value
    then_body: tuple[Inst, ...]
    else_body: tuple[Inst, ...]


@dataclass(eq=False)
class Function(Value):
    """One compiled MIR function.  As a value it is the in-module
    function value of a call target: a call whose callee is this object
    is lowered to a call of the ``define``d function (functions of one
    module are compiled together).

    The host creates the function (with an empty body) and registers it
    before the body is typed, so that recursive calls made by the body
    resolve to the very object being filled in.  ``ret_type`` is fixed
    at creation when the function declares one; otherwise it stays
    ``None`` until the body has been typed (a function still being
    typed and without a declared return type can only be observed by a
    recursive call, which then is a compile error)."""

    name: str
    args: tuple[FormalArg, ...]
    ret_type: Type | None
    insts: list[Inst]

    @property
    def type(self) -> Type:
        """The type of the function value: a pointer to the function's
        signature."""
        assert self.ret_type is not None, 'the function is still being typed'
        return PointerType(FunctionType(tuple(a.type for a in self.args), self.ret_type))

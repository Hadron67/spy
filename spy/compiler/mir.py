from abc import abstractmethod
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Self, override

from .binop import BinaryOp, CompareOp
from .errors import CompileError, SpyError

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


class NoReturn:
    """The return type of a function that never returns: its lowered form
    returns void and is marked ``noreturn``, and a call of it does not come back
    (the block it sits in ends with it)."""

    def __repr__(self) -> str:
        return 'noreturn'

    def get_children(self) -> tuple[Any, ...]:
        return ()

NORETURN = NoReturn()

type ReturnType = MayBeVoidType | NoReturn

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

    ``fam_type`` is the element type of the struct's *flexible array member*
    (a C FAM), or ``VOID`` for an opaque tail: the struct's first
    dynamically-sized field, which is not one of ``fields`` (it has no size of
    its own) but is placed *after* them all (see ``lower._ModuleTypes``, which
    appends a ``[0 x T]`` for it).  A struct with a FAM has no value of its
    own - only a pointer to it is one.
    """

    def __init__(
        self,
        name_base: str | None,
        fields: tuple[FormalArg, ...] | None,
        fam_type: MayBeVoidType | None = None,
    ) -> None:
        self.name_base = name_base
        self.fields: list[FormalArg] = []
        if fields is not None:
            self.fields.extend(fields)
        self.fam_type = fam_type
        # the ctypes class mirroring the LLVM layout of this struct (what
        # a struct value crossing the Python boundary is viewed as);
        # materialized on demand and cached by ``lower``
        self.ctype: Any = None

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    def get_children(self) -> tuple[Any, ...]:
        children: list[Any] = [f.type for f in self.fields]
        if isinstance(self.fam_type, Type):
            # the FAM's own type is used by the lowered struct: it has to be
            # collected like any other nested type (see ``collect_symbols``)
            children.append(self.fam_type)
        return tuple(children)


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


class UnionType(Type):
    """The static type of an untagged union value - the payload of an error
    union: exactly one variant is stored, and the largest one occupies the
    storage.  The union carries no tag of its own (the tag lives next to it,
    as the error code), so a variant value is written and read through a
    reinterpretation (:class:`BitCast`) of the address of a union value.

    The type is an identity object mirroring one ``sval.UnionType``: two
    unions are equal only when they are the same object."""

    def __init__(self, name_base: str | None, payload: Type) -> None:
        self.name_base = name_base
        self.payload = payload

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __repr__(self) -> str:
        return f'<union {self.payload!r}>'

    def get_children(self) -> tuple[Any, ...]:
        return (self.payload,)

@dataclass(frozen=True)
class FunctionType(Type):
    """The signature of a function value (the element type of its
    pointer)."""

    args: tuple[Type, ...]
    return_type: ReturnType
    # the calling convention: ``'default'`` is the spy one; any other value
    # names a C one (see ``sval.FunctionType.callconv``)
    callconv: str = 'default'
    # whether the function may panic: a call of one may have an unwind edge
    # (see ``CallMayPanic``); a function may panic by default
    may_panic: bool = True
    # whether the function is C-variadic (a trailing ``...``): a call of it may
    # carry any number of extra arguments after ``args`` (see
    # ``sval.FunctionType.varargs``)
    varargs: bool = False

    def get_children(self) -> tuple[Any, ...]:
        return (*self.args, self.return_type)


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------


def estimated_size_of(type: Type, pointer_size: int) -> int:
    """The estimated size of the MIR type ``type`` in bytes, laid out with
    pointers of ``pointer_size`` bytes.  The MIR type *is* the layout: it
    holds a struct's fields in the order the layout puts them in (see
    ``sval.StructType._calculate_mir``) and a union's storage variant already
    selected, so this is the size of the value the lowered code works with.
    A type that has no layout at all - a function type, which is dynamically
    sized - raises :class:`SpyError`."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return (type.bits + 7) // 8
        case FloatType():
            return (type.bits + 7) // 8
        case PointerType():
            return pointer_size
        case ArrayType():
            return type.length * estimated_size_of(type.elem, pointer_size)
        case StructType():
            offset = 0
            for field in type.fields:
                align = estimated_alignment_of(field.type, pointer_size)
                offset = (offset + align - 1) // align * align
                offset += estimated_size_of(field.type, pointer_size)
            align = estimated_alignment_of(type, pointer_size)
            return (offset + align - 1) // align * align
        case UnionType():
            # a union holds its storage variant's storage (no tag of its own)
            return estimated_size_of(type.payload, pointer_size)
        case _:
            raise SpyError(f"type {type} has no layout")


def estimated_alignment_of(type: Type, pointer_size: int) -> int:
    """The estimated alignment of the MIR type ``type`` in bytes, for pointers
    of ``pointer_size`` bytes (see :func:`estimated_size_of`).  An array aligns
    to its element type, so ``estimated_alignment_of(T[0])`` is that of ``T``.
    A type that has no layout at all (a function type) raises :class:`SpyError`."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return max(1, type.bits // 8)
        case FloatType():
            return type.bits // 8
        case PointerType():
            return pointer_size
        case ArrayType():
            return estimated_alignment_of(type.elem, pointer_size)
        case StructType():
            align = max(
                (estimated_alignment_of(field.type, pointer_size) for field in type.fields),
                default=1,
            )
            if type.fam_type is not None and not isinstance(type.fam_type, VoidType):
                # the flexible member is a ``[0 x T]``: it adds no size but its
                # element type's alignment
                align = max(align, estimated_alignment_of(type.fam_type, pointer_size))
            return align
        case UnionType():
            return estimated_alignment_of(type.payload, pointer_size)
        case _:
            raise SpyError(f"type {type} has no layout")


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


@dataclass(frozen=True, slots=True)
class Dangling(Value):
    """A non-null, *dangling* pointer constant: its address is the pointee's
    alignment (rust's ``NonNull::dangling``).  It is what
    ``sval.Undefined(Ptr[T])`` lowers to, because null is not a legal value of
    a non-null pointer, so an undefined pointer may not be null either
    (see ``lower``)."""

    type: PointerType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)


@dataclass(frozen=True)
class UndefValue(Value):
    """The undefined value of the aggregate type ``type``: the base an
    ``InsertValue`` chain starts from (LLVM's ``undef``).  It carries no
    information a consumer may rely on - it only exists to give the chain a
    value to insert into."""

    type: Type

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)


@dataclass(frozen=True, slots=True)
class AggregateConstant:
    """A constant aggregate value: the initializer of a
    :class:`GlobalConstant`.  It is deliberately *not* a ``Value`` - an
    aggregate constant cannot be an operand of an instruction (an LLVM
    aggregate constant only appears as an initializer) - so it is only ever the
    ``value`` of a global or nested inside another aggregate constant.  The
    members are in *layout* (mirror) order; a member is a plain ``Value`` (a
    scalar, a global reference), a nested ``AggregateConstant``, or a
    :class:`UnionConstant`."""

    type: StructType | ArrayType
    values: tuple[AggregateConstant | UnionConstant | Value, ...]

    def get_children(self) -> tuple[Any, ...]:
        return (self.type, *self.values)


@dataclass(frozen=True, slots=True)
class UnionConstant:
    """A constant union value: the variant ``variant`` of the union ``union``,
    with its constant ``value``.  Like :class:`AggregateConstant` it is not a
    ``Value``: a union constant is only ever an initializer, and one that holds
    a variant other than the union's storage variant is laid out the way clang
    lays out such a global - as a literal ``{<variant>, [pad x i8]}`` (see
    ``lower``)."""

    union: UnionType
    variant: Type
    value: AggregateConstant | UnionConstant | Value

    def get_children(self) -> tuple[Any, ...]:
        return (self.union, self.variant, self.value)


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


@dataclass(eq=False, slots=True)
class GlobalConstant(GlobalValue):
    """A global static constant: the pointer to a static location holding the
    constant ``value``.  ``type`` is the pointer type of the expression that
    names the location (a ``ConstPtr[T]``); the lowerer emits the location (see
    ``lower``).  The value is a plain ``Value`` for a scalar (or a global
    reference), or an ``AggregateConstant`` / ``UnionConstant`` for an
    aggregate (which cannot be an instruction operand)."""

    value: AggregateConstant | UnionConstant | Value
    type: PointerType

    @override
    def get_type(self) -> MayBeVoidType:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.type, self.value)

    @override
    def get_name(self) -> tuple[str, bool]:
        return 'const', True


@dataclass(eq=False, slots=True)
class GlobalStringValue(GlobalValue):
    """A global static constant holding a byte string: the pointer to a static
    read-only location the bytes are emitted into (see ``lower``).  ``data``
    already carries the trailing NUL the ``std.core.gstr``/``sstr`` builtins
    append; the pointer type is always ``*u8`` - the constness/variant are the
    expression's own, carried by the sval type the interpreter types a value
    with (LLVM lowers every pointer to the same opaque ``ptr``).  Two byte
    strings are deduplicated by their content (see ``interp``), so the same
    bytes share one global."""

    data: bytes

    @override
    def get_type(self) -> MayBeVoidType:
        return PointerType(IntType(8, False))

    @override
    def get_name(self) -> tuple[str, bool]:
        return 'str', True


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


@dataclass(eq=False, slots=True)
class Sizeof(Inst):
    """The size, in bytes, of the MIR type ``type`` (``std.mem.layout_of``).
    A type is not a value, so it is carried here as an operand and measured by
    the lowerer against the target's layout - a spy type has no compile-time
    size.  ``bits`` is the width of the integer result (the target ``usize``)."""

    type: Type
    bits: int

    @override
    def get_type(self) -> Type:
        return IntType(self.bits, False)

    def get_children(self) -> tuple[Any, ...]:
        return (self.type,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


@dataclass(eq=False, slots=True)
class Alignof(Inst):
    """The alignment, in bytes, of the MIR type ``type`` (see :class:`Sizeof`
    for the shared shape)."""

    type: Type
    bits: int

    @override
    def get_type(self) -> Type:
        return IntType(self.bits, False)

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


class AtomicOrdering(Enum):
    """The memory ordering of an atomic operation (``std.atomic.MemoryOrder``);
    the values are the spellings LLVM's textual IR uses (see
    ``llvm.Ordering``)."""

    MONOTONIC = 'monotonic'
    ACQUIRE = 'acquire'
    RELEASE = 'release'
    ACQ_REL = 'acq_rel'
    SEQ_CST = 'seq_cst'


type AtomicRmwOp = Literal['xchg', 'add', 'sub', 'and', 'or', 'xor']


@dataclass(eq=False)
class AtomicLoad(Inst):
    """An atomic load (LLVM's ``load atomic``): read the pointee of ``ptr``
    with the given memory ``ordering`` (and ``volatile`` when asked).  The
    result has the pointee's type."""

    ptr: Value
    ordering: AtomicOrdering
    volatile: bool = False

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
class AtomicStore(Inst):
    """An atomic store (LLVM's ``store atomic``): write ``value`` through
    ``ptr`` with the given memory ``ordering`` (and ``volatile`` when asked)."""

    ptr: Value
    value: Value
    ordering: AtomicOrdering
    volatile: bool = False

    def get_children(self) -> tuple[Any, ...]:
        return (self.ptr, self.value)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        value = f(self.value)
        if ptr is self.ptr and value is self.value:
            return self
        return replace(self, ptr=ptr, value=value)


@dataclass(eq=False)
class AtomicRmw(Inst):
    """A read-modify-write (LLVM's ``atomicrmw``): apply ``op`` to the pointee
    of ``ptr`` and ``value`` with the given memory ``ordering``, and produce
    the *old* pointee value.  ``op`` is one of ``'xchg'``, ``'add'``,
    ``'sub'``, ``'and'``, ``'or'`` and ``'xor'`` (the integer operations)."""

    op: AtomicRmwOp
    ptr: Value
    value: Value
    ordering: AtomicOrdering
    volatile: bool = False

    @override
    def get_type(self) -> MayBeVoidType:
        ptr_type = self.ptr.get_type()
        assert isinstance(ptr_type, PointerType) and ptr_type.elem is not None
        return ptr_type.elem

    def get_children(self) -> tuple[Any, ...]:
        return (self.ptr, self.value)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        value = f(self.value)
        if ptr is self.ptr and value is self.value:
            return self
        return replace(self, ptr=ptr, value=value)


@dataclass(eq=False)
class AtomicCmpxchg(Inst):
    """A compare-and-exchange (LLVM's ``cmpxchg``): if the pointee of ``ptr``
    equals ``expected``, store ``desired`` into it.  The result is a struct
    ``{value: T, ok: bool}`` holding the *old* pointee value and whether the
    exchange happened (LLVM's ``{T, i1}``); the two are read out with
    ``ExtractValue``.  ``success``/``failure`` are the memory orderings of the
    two outcomes (the failure one must be no stronger than ``success``, and
    never ``release``/``acq_rel``)."""

    ptr: Value
    expected: Value
    desired: Value
    success: AtomicOrdering
    failure: AtomicOrdering
    volatile: bool = False

    def __init__(
        self,
        ptr: Value,
        expected: Value,
        desired: Value,
        success: AtomicOrdering,
        failure: AtomicOrdering,
        volatile: bool = False,
    ) -> None:
        self.ptr = ptr
        self.expected = expected
        self.desired = desired
        self.success = success
        self.failure = failure
        self.volatile = volatile
        ptr_type = ptr.get_type()
        assert isinstance(ptr_type, PointerType) and isinstance(ptr_type.elem, Type)
        # the ``{value, ok}`` result LLVM's ``cmpxchg`` yields
        self.type = StructType(
            None,
            (FormalArg('value', ptr_type.elem), FormalArg('ok', BoolType())),
        )

    @override
    def get_type(self) -> Type:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.ptr, self.expected, self.desired)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        ptr = f(self.ptr)
        expected = f(self.expected)
        desired = f(self.desired)
        if ptr is self.ptr and expected is self.expected and desired is self.desired:
            return self
        return replace(self, ptr=ptr, expected=expected, desired=desired)


@dataclass(eq=False)
class Fence(Inst):
    """A memory ordering fence (LLVM's ``fence``): prevents the compiler and
    the hardware from reordering memory operations across it."""

    ordering: AtomicOrdering

    def get_children(self) -> tuple[Any, ...]:
        return ()

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


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
            if elem.fam_type is not None and index == len(elem.fields):
                # the struct's flexible member sits past every field of the
                # mirror: its address is a pointer to its element type (or a
                # void pointer for an opaque tail)
                self.type = PointerType(elem.fam_type)
                return
            if index < 0 or index >= len(elem.fields):
                raise CompileError(f'field index {index} is out of bounds for {elem}')
            self.type: Type = PointerType(elem.fields[index].type)
            return
        if isinstance(elem, ArrayType):
            if isinstance(index, int) and not 0 <= index < elem.length:
                raise CompileError(f'element index {index} is out of bounds for {elem}')
            self.type = PointerType(elem.elem)
            return
        # a pointer to anything else is *offset* by the index: the address of
        # that many pointees after it, an element of an array of them (what a
        # ``MultiPtr`` and ``mptr + n`` are)
        self.type = PointerType(elem)

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
class ExtractValue(Inst):
    """The field ``index`` of the struct (or the element ``index`` of the
    array) *value* ``value``: ``Gep``'s counterpart in the value domain - it
    reads a field out of an aggregate value instead of taking the address of
    one (an aggregate passed by value has no address of its own).  ``index`` is
    the position in the *mirror* of the struct (the declaration index mapped by
    ``sval.StructType.get_field_mir_indices``), or an array element position.
    The result type is the field's/element's own type."""

    value: Value
    index: int

    def __init__(self, value: Value, index: int) -> None:
        self.value = value
        self.index = index
        vtype = value.get_type()
        if isinstance(vtype, StructType):
            if index < 0 or index >= len(vtype.fields):
                raise CompileError(f'field index {index} is out of bounds for {vtype}')
            self.type: Type = vtype.fields[index].type
            return
        if isinstance(vtype, ArrayType):
            if not 0 <= index < vtype.length:
                raise CompileError(f'element index {index} is out of bounds for {vtype}')
            self.type = vtype.elem
            return
        raise CompileError(f'cannot extract a field of a {vtype} value')

    @override
    def get_type(self) -> Type:
        return self.type

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False)
class InsertValue(Inst):
    """The aggregate value ``value`` with its field ``index`` (or element
    ``index`` of an array) replaced by ``elem``: ``ExtractValue``'s counterpart
    in the value domain, which builds an aggregate value from its fields
    instead of reading one out of it.  ``index`` is the position in the
    *mirror* of the struct (the declaration index mapped by
    ``sval.StructType.get_field_mir_indices``), or an array element position.
    The result has the type of ``value``."""

    value: Value
    elem: Value
    index: int

    def __init__(self, value: Value, elem: Value, index: int) -> None:
        self.value = value
        self.elem = elem
        self.index = index
        vtype = value.get_type()
        if isinstance(vtype, StructType):
            if index < 0 or index >= len(vtype.fields):
                raise CompileError(f'field index {index} is out of bounds for {vtype}')
            self.type: Type = vtype
            return
        if isinstance(vtype, ArrayType):
            if not 0 <= index < vtype.length:
                raise CompileError(f'element index {index} is out of bounds for {vtype}')
            self.type = vtype
            return
        raise CompileError(f'cannot insert a field of a {vtype} value')

    @override
    def get_type(self) -> MayBeVoidType:
        return self.value.get_type()

    def get_children(self) -> tuple[Any, ...]:
        return (self.value, self.elem)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        value = f(self.value)
        elem = f(self.elem)
        if value is self.value and elem is self.elem:
            return self
        return replace(self, value=value, elem=elem)


@dataclass(eq=False)
class Select(Inst):
    """The value ``if_true`` or ``if_false``, chosen by the boolean ``cond``
    (LLVM's ``select``): a value-level conditional that branches nowhere, so
    both operands are always computed.  The two values have to agree on their
    type, which is also the type of the result."""

    cond: Value
    if_true: Value
    if_false: Value

    def __init__(self, cond: Value, if_true: Value, if_false: Value) -> None:
        self.cond = cond
        self.if_true = if_true
        self.if_false = if_false

    @override
    def get_type(self) -> MayBeVoidType:
        return self.if_true.get_type()

    def get_children(self) -> tuple[Any, ...]:
        return (self.cond, self.if_true, self.if_false)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        cond = f(self.cond)
        if_true = f(self.if_true)
        if_false = f(self.if_false)
        if cond is self.cond and if_true is self.if_true and if_false is self.if_false:
            return self
        return replace(self, cond=cond, if_true=if_true, if_false=if_false)


@dataclass(eq=False)
class Arith(Inst):
    """Integer/float arithmetic and integer bitwise/shift operations.

    ``op`` is one of ``'+'``, ``'-'``, ``'*'`` (integer or float, chosen by
    ``type``), ``'/'`` (truncating integer division, or float division),
    ``'%'`` (integer remainder), the bitwise ``'|'``, ``'&'`` and ``'^'``, and
    the shifts ``'<<'`` and ``'>>'``.  Division, remainder and the right shift
    honor the signedness of the integer ``type`` (a float ``type`` ignores
    it)."""

    op: BinaryOp
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
class Floor(Inst):
    """Round a float value toward negative infinity (LLVM's ``llvm.floor``)."""

    value: Value

    @override
    def get_type(self) -> MayBeVoidType:
        return self.value.get_type()

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False)
class Pow(Inst):
    """Raise the float value ``lhs`` to the float power ``rhs`` (LLVM's
    ``llvm.pow``).  ``type`` is the (float) type of the result."""

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
class BitCast(Inst):
    """Reinterpret a value of one type as a value of another type of the same
    size.  It is how a union variant is written and read through the address of
    the union's storage: the address of the union is reinterpreted as the
    address of the variant.  A pointer-to-pointer cast has no representation of
    its own: the LLVM IR pointers are untyped (``ptr``), so it lowers to the
    address it is given (see ``lower``)."""

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
class AsUnion(Inst):
    """A value of one of a union's variants reinterpreted as a value of the
    union itself (the union's storage holds the variant): a non-union value to a
    union value.  ``type`` is the target ``UnionType``; the source value's type
    is the variant."""

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
class ExtractUnion(Inst):
    """A variant value read out of a union value: a union value to a non-union
    value.  ``type`` is the variant's type."""

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
class UnionCast(Inst):
    """A union value reinterpreted as another union value (whose variants are a
    subset or a superset of its own): a union value to a union value.  ``type``
    is the target ``UnionType``."""

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
    '<=', '>', '>='.  The domain the operands live in - integers, floats or
    pointers - is read off their type: numbers compare with ``icmp``/``fcmp``
    (the signedness of an integer ``icmp`` follows the integer type), and
    pointers compare against each other with ``icmp`` (only equality is
    defined on them).  Both operands share one type."""

    op: CompareOp
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
class Phi(Inst):
    """A value that is one of several incoming values, chosen by the
    predecessor block control arrived from: how one merge point - the entry of
    an ``except`` block - takes the error payload pointer that several error
    dispatches deliver to it.  ``incomings`` pairs every incoming value with
    the block it flows from; its result type is the (common) type of the
    values."""

    incomings: list[tuple[Value, BasicBlock]]

    def add_incoming(self, value: Value, block: BasicBlock) -> None:
        self.incomings.append((value, block))

    @override
    def get_type(self) -> MayBeVoidType:
        return self.incomings[0][0].get_type()

    def get_children(self) -> tuple[Any, ...]:
        return tuple(value for value, _ in self.incomings)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        changed = False
        incomings: list[tuple[Value, BasicBlock]] = []
        for value, block in self.incomings:
            mapped = f(value)
            if mapped is not value:
                changed = True
            incomings.append((mapped, block))
        return self if not changed else replace(self, incomings=incomings)


@dataclass(eq=False)
class Call(Inst):
    """A call of a function value returning a value of type ``type``
    (``mir.VOID`` for a call of a void function, which produces no
    result, and ``mir.NoReturn`` for a call of a function that never
    returns, which ends the block it sits in - see :func:`ends_block`).
    The callee is either a :class:`Function` (compiled in the same LLVM
    module) or a :class:`GlobalValue` of an earlier module
    (:class:`ExternAnonSymbol`, or an :class:`ExternSymbol` resolved from
    the process)."""

    callee: Value
    args: tuple[Value, ...]
    type: ReturnType

    def is_noreturn(self) -> bool:
        """Whether this call never comes back: what it calls is a function
        that never returns."""
        return isinstance(self.type, NoReturn)

    @override
    def get_type(self) -> MayBeVoidType:
        return VOID if isinstance(self.type, NoReturn) else self.type

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
        block, and so does a call of a function that never returns (see
        ``ends_block``), so nothing may follow it - the assertion catches an
        instruction that would end up on a dead path."""
        assert not self.is_finished, 'cannot emit into a finished basic block'
        self.insts.append(inst)
        if ends_block(inst):
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


@dataclass(eq=False)
class Terminator(Inst):
    """The transfer instruction that ends a basic block: a :class:`Jmp`, a
    :class:`Br` or a :class:`Ret`.  It produces no value, and its targets
    are blocks rather than values, so ``get_children`` never returns
    them."""

    @abstractmethod
    def get_targets(self) -> tuple[BasicBlock, ...]:
        """The blocks this terminator transfers control to."""
        ...

    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        """The deferred bodies this transfer triggers on its way out, in the
        order they run (empty for a transfer that leaves no ``defer`` region, or
        one whose regions have already been instantiated - see
        ``instantiate_defers``)."""
        return ()


def ends_block(inst: Inst) -> bool:
    """Whether ``inst`` ends its basic block: a terminator, or a call of a
    function that never returns (a :class:`NoReturn` call), after which no
    instruction can run."""
    return isinstance(inst, Terminator) or (
        isinstance(inst, Call) and inst.is_noreturn()
    )


@dataclass(eq=False)
class Jmp(Terminator):
    """An unconditional jump to ``target``.  ``defer_blocks`` are the deferred
    bodies the jump runs first, in order (see ``Terminator.get_defer_blocks``)."""

    target: BasicBlock
    defer_blocks: tuple[BasicBlock, ...] = ()

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.target,)

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return self.defer_blocks

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


@dataclass(eq=False)
class Br(Terminator):
    """A two-way conditional branch: control goes to ``if_true`` when
    ``cond`` holds and to ``if_false`` otherwise.  Each edge has its own
    deferred bodies to run first (``if_true_defer_blocks``/:
    ``if_false_defer_blocks``), because the two edges can leave different
    regions (a conditional ``break`` leaves on the true edge only, a plain
    runtime ``if`` leaves none)."""

    cond: Value
    if_true: BasicBlock
    if_false: BasicBlock
    if_true_defer_blocks: tuple[BasicBlock, ...] = ()
    if_false_defer_blocks: tuple[BasicBlock, ...] = ()

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.if_true, self.if_false)

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return (*self.if_true_defer_blocks, *self.if_false_defer_blocks)

    def get_children(self) -> tuple[Any, ...]:
        return (self.cond,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        cond = f(self.cond)
        return self if cond is self.cond else replace(self, cond=cond)


@dataclass(eq=False)
class Switch(Terminator):
    """A multi-way branch on an integer value (the error code of a call):
    control goes to the target of the case whose constant matches ``value``, or
    to ``default`` when no case matches."""

    value: Value
    default: BasicBlock
    cases: tuple[tuple[int, BasicBlock], ...]

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.default, *(block for _, block in self.cases))

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False)
class Ret(Terminator):
    """Return from the enclosing function; ends the path of its block.
    ``value`` is None for a void return (a ``ret void``).  ``defer_blocks`` are
    the deferred bodies the return runs first, in order."""

    value: Value | None
    defer_blocks: tuple[BasicBlock, ...] = ()

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return ()

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return self.defer_blocks

    def get_children(self) -> tuple[Any, ...]:
        return (self.value,) if self.value is not None else ()

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        if self.value is None:
            return self
        value = f(self.value)
        return self if value is self.value else replace(self, value=value)


@dataclass(eq=False, slots=True)
class EndDefer(Terminator):
    """The end of a deferred body (see ``hir.Defer``): the terminator of the
    shared block tree the interpreter emitted a ``with syntax.defer():`` body
    into.  ``defer_blocks`` are the deferred bodies the *normal* completion of
    the body triggers - the ``defer``/``okdefer`` blocks declared inside it, in
    reverse declaration order.

    The tree is shared: every transfer that triggers the defer refers to its
    entry, so a block's single terminator cannot name one continuation.
    ``instantiate_defers`` therefore expands one fresh copy of the tree per
    distinct use (transfers that run the same bodies and continue in the same
    block share a copy) and replaces every ``EndDefer`` with the ``Jmp`` that
    continues the copy (running the nested triggers first).  After instantiation
    no ``EndDefer`` is reachable any more."""

    defer_blocks: tuple[BasicBlock, ...] = ()

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return ()

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return self.defer_blocks

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


@dataclass(eq=False)
class Panic(Terminator):
    """End the path by throwing a panic carrying the ``PanicData`` in ``data``
    (a pointer to the value, see ``interp``): it never returns, so it ends its
    block.  ``lower`` emits the throw inline (allocate the C++ exception, copy
    the data in, ``__cxa_throw``, ``unreachable``), so no runtime helper is
    needed (see ``interp._call_builtin``).

    Like ``CallMayPanic``, when the panic would run deferred bodies on its way
    out of the enclosing regions the throw is an ``invoke`` whose unwind edge
    runs ``unwind_defers`` and reaches ``unwind_path`` (the function-wide resume
    block).  Both are unset when no deferred body would run: the throw is then a
    plain ``call`` and unwinding passes straight through the frame."""

    data: Value
    unwind_path: BasicBlock | None = None
    unwind_defers: tuple[BasicBlock, ...] = ()

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return () if self.unwind_path is None else (self.unwind_path,)

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return self.unwind_defers

    def get_children(self) -> tuple[Any, ...]:
        return (self.data,)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        data = f(self.data)
        return self if data is self.data else replace(self, data=data)


@dataclass(eq=False)
class CallMayPanic(Terminator):
    """A call that may panic while the enclosing regions still have deferred
    bodies a panic would run (see ``interp``): a terminator with two
    successors, which also produces the value of the call (like ``Call``).

    ``normal_path`` is where a normal return continues (the value of ``type``
    is produced there).  The unwind edge first runs ``unwind_defers`` - the
    deferred bodies the panic triggers on its way out, the shared templates of
    ``Terminator.get_defer_blocks`` - and then reaches ``unwind_path``, the
    single block per function holding the ``Resume`` that carries the unwinding
    on.  A call whose panic would run no deferred body is a plain ``Call``
    instead: unwinding through the frame then needs no landing pad."""

    callee: Value
    args: tuple[Value, ...]
    type: ReturnType
    normal_path: BasicBlock
    unwind_path: BasicBlock
    unwind_defers: tuple[BasicBlock, ...] = ()

    def is_noreturn(self) -> bool:
        return isinstance(self.type, NoReturn)

    @override
    def get_type(self) -> MayBeVoidType:
        return VOID if isinstance(self.type, NoReturn) else self.type

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.normal_path, self.unwind_path)

    @override
    def get_defer_blocks(self) -> tuple[BasicBlock, ...]:
        return self.unwind_defers

    def get_children(self) -> tuple[Any, ...]:
        return (self.callee, *self.args)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        callee = f(self.callee)
        args = tuple(f(arg) for arg in self.args)
        if callee is self.callee and all(a is b for a, b in zip(args, self.args)):
            return self
        return replace(self, callee=callee, args=args)


@dataclass(eq=False)
class CatchUnwind(Terminator):
    """A call of a closure that catches a panic the callee may throw (the
    ``std.core.catch_unwind`` builtin, see ``interp``): like ``CallMayPanic`` it
    produces the value of the call on ``normal_path``, but its unwind edge goes
    straight to ``catch_path`` - a block the interpreter filled with the
    construction of the ``UnwindException`` and its dispatch (a ``raise``) - and
    not through any deferred body: the panic is caught here, so the enclosing
    regions are not left.  The caught ``PanicData`` pointer is written into the
    place ``catch_data`` points at by the landing pad ``lower`` synthesizes."""

    callee: Value
    args: tuple[Value, ...]
    type: ReturnType
    normal_path: BasicBlock
    catch_path: BasicBlock
    catch_data: Value

    def is_noreturn(self) -> bool:
        return isinstance(self.type, NoReturn)

    @override
    def get_type(self) -> MayBeVoidType:
        return VOID if isinstance(self.type, NoReturn) else self.type

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return (self.normal_path, self.catch_path)

    def get_children(self) -> tuple[Any, ...]:
        return (self.callee, *self.args, self.catch_data)

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        callee = f(self.callee)
        args = tuple(f(arg) for arg in self.args)
        catch_data = f(self.catch_data)
        if (
            callee is self.callee
            and all(a is b for a, b in zip(args, self.args))
            and catch_data is self.catch_data
        ):
            return self
        return replace(self, callee=callee, args=args, catch_data=catch_data)


@dataclass(eq=False)
class Resume(Terminator):
    """Resume unwinding after the deferred bodies of a panic have run: the tail
    of every ``CallMayPanic``'s unwind chain reaches the single block holding one
    of these (the function-wide ``unwind_path``).  ``lower`` emits the ``resume``
    with the exception the landing pad stored (see ``interp``)."""

    @override
    def get_targets(self) -> tuple[BasicBlock, ...]:
        return ()

    def map_values(self, f: Callable[[Value], Value]) -> Self:
        return self


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


def _reachable_with_defers(entry: BasicBlock) -> list[BasicBlock]:
    """Every block reachable from ``entry``, following both the targets and the
    deferred bodies of its terminators.  The deferred bodies of a transfer are
    only reachable through ``get_defer_blocks`` (they are templates, not real
    CFG edges), so :meth:`BasicBlock.collect_blocks` misses them;
    :func:`normalize` needs them all the same - their instructions hold
    insertions to flatten."""
    ret: list[BasicBlock] = []
    seen: set[BasicBlock] = set()
    todo: list[BasicBlock] = [entry]
    while len(todo) > 0:
        block = todo.pop()
        if block in seen:
            continue
        seen.add(block)
        ret.append(block)
        children: list[BasicBlock] = []
        if len(block.insts) > 0:
            last = block.insts[-1]
            if isinstance(last, Terminator):
                children.extend(last.get_targets())
                children.extend(last.get_defer_blocks())
        children.reverse()
        todo.extend(children)
    return ret


def normalize(fn: Function) -> None:
    """Eliminate the :class:`Insertion` placeholders of every block of
    ``fn`` by flattening them and substituting their values, then check
    that every block ends with a terminator."""
    blocks = _reachable_with_defers(fn.entry)

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
        if len(block.insts) == 0 or not ends_block(block.insts[-1]):
            raise CompileError(
                'every basic block must end with a jump, a branch, a return or a'
                ' call of a function that never returns'
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
    ret_type: ReturnType
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
        if isinstance(self.ret_type, Type):
            ret.append(self.ret_type)
        for block in self.entry.collect_blocks():
            for inst in block.insts:
                for child in inst.get_children():
                    if not isinstance(child, Inst):
                        ret.append(child)
        return tuple(ret)

def collect_symbols(entry: list[GlobalValue]) -> set[GlobalValue | StructType]:
    symbols: set[GlobalValue | StructType] = set()
    todo: list[Any] = [a for a in entry]
    while todo:
        value = todo.pop()
        if value in symbols:
            continue
        if isinstance(value, (StructType, GlobalValue)):
            symbols.add(value)
        todo.extend(reversed([a for a in value.get_children() if not isinstance(a, Inst)]))
    return symbols


# ---------------------------------------------------------------------------
# deferred bodies
# ---------------------------------------------------------------------------


def _repoint_phi_incoming(target: BasicBlock, old: BasicBlock, new: BasicBlock) -> None:
    """Replace the predecessor ``old`` of every ``Phi`` of ``target`` by
    ``new``: a transfer that runs deferred bodies first reaches the target from
    the tail of the instantiated chain, not from the block that emitted it."""
    for inst in target.insts:
        if isinstance(inst, Phi):
            for index, (value, pred) in enumerate(inst.incomings):
                if pred is old:
                    inst.incomings[index] = (value, new)


def _instantiate_chain(entries: tuple[BasicBlock, ...], cont: BasicBlock) -> tuple[BasicBlock, BasicBlock]:
    """The entry and the tail of an explicit chain running the deferred bodies
    ``entries`` (in order) and then continuing in ``cont``: a fresh copy of every
    entry is made, each wired to the next (see :func:`_clone_template`), and the
    returned tail is the block that transfers to ``cont`` (the last copy's own
    tail).  With no entries the chain is empty and both are ``cont``.  The
    recursion follows the source nesting of the defers - one level per body
    declared inside another - so it is shallow."""
    if len(entries) == 0:
        return cont, cont
    entry = cont
    tail = cont
    first = True
    for deferred in reversed(entries):
        entry, copied_tail = _clone_template(deferred, entry)
        if first:
            # the first copy made is the innermost body: its tail is the one
            # that reaches ``cont``
            tail = copied_tail
            first = False
    return entry, tail


def _clone_template(root: BasicBlock, cont: BasicBlock) -> tuple[BasicBlock, BasicBlock]:
    """A fresh copy of the deferred body tree rooted at ``root``, continuing in
    ``cont`` when the body has finished: every block and instruction of the copy
    is new (the values the body reads from enclosing scopes are shared), the
    operands and the blocks the terminators refer to are remapped onto the copy,
    the transfers inside the body with deferred bodies of their own are
    instantiated in turn, and the copy's :class:`EndDefer` is replaced by the
    ``Jmp`` that runs the body's own triggers and continues in ``cont``.  Returns
    the entry and the tail of the copy (the block that transfers to ``cont``)."""
    # the blocks of the tree, following the real targets only: an ``EndDefer``
    # is a leaf, and a body declared inside another is reached through a
    # transfer's deferred blocks (expanded below), not through a real edge
    tblocks: list[BasicBlock] = []
    seen: set[BasicBlock] = set()
    todo: list[BasicBlock] = [root]
    while len(todo) > 0:
        block = todo.pop()
        if block in seen:
            continue
        seen.add(block)
        tblocks.append(block)
        if len(block.insts) > 0:
            last = block.insts[-1]
            if isinstance(last, Terminator):
                todo.extend(last.get_targets())

    # clone every instruction; the map original -> copy also drives the operand
    # substitution (a value from an enclosing scope is not in the map, so it is
    # shared by the copy)
    repl: dict[Value, Value] = {}
    for block in tblocks:
        for inst in block.insts:
            repl[inst] = copy(inst)

    def resolve(value: Value) -> Value:
        seen: set[Value] = set()
        while value in repl:
            if value in seen:
                raise CompileError('cycle in a deferred body copy')
            seen.add(value)
            value = repl[value]
        return value

    def resolve_inst(inst: Inst) -> Inst:
        resolved = resolve(inst)
        assert isinstance(resolved, Inst)
        return resolved

    changed = True
    while changed:
        changed = False
        for block in tblocks:
            for inst in block.insts:
                clone = resolve_inst(inst)
                mapped = clone.map_values(resolve)
                if mapped is not clone:
                    repl[clone] = mapped
                    changed = True

    bmap: dict[BasicBlock, BasicBlock] = {block: BasicBlock() for block in tblocks}
    for block in tblocks:
        new = bmap[block]
        new.insts = [resolve_inst(inst) for inst in block.insts]
    # first remap every block reference onto the copies
    for block in tblocks:
        new = bmap[block]
        for index, inst in enumerate(new.insts):
            new.insts[index] = _remap_targets(inst, bmap)
    # then expand the deferred bodies the transfers of the copy have of their own
    # and give the copy's ``EndDefer`` its continuation
    tail: BasicBlock | None = None
    for block in tblocks:
        new = bmap[block]
        term = new.insts[-1]
        if isinstance(term, Jmp) and len(term.defer_blocks) > 0:
            entry, chain_tail = _instantiate_chain(term.defer_blocks, term.target)
            _repoint_phi_incoming(term.target, new, chain_tail)
            term.target = entry
            term.defer_blocks = ()
        elif isinstance(term, Br) and (
            len(term.if_true_defer_blocks) > 0 or len(term.if_false_defer_blocks) > 0
        ):
            if len(term.if_true_defer_blocks) > 0:
                entry, chain_tail = _instantiate_chain(term.if_true_defer_blocks, term.if_true)
                _repoint_phi_incoming(term.if_true, new, chain_tail)
                term.if_true = entry
                term.if_true_defer_blocks = ()
            if len(term.if_false_defer_blocks) > 0:
                entry, chain_tail = _instantiate_chain(term.if_false_defer_blocks, term.if_false)
                _repoint_phi_incoming(term.if_false, new, chain_tail)
                term.if_false = entry
                term.if_false_defer_blocks = ()
        elif isinstance(term, EndDefer):
            if len(term.defer_blocks) > 0:
                entry, chain_tail = _instantiate_chain(term.defer_blocks, cont)
                new.insts[-1] = Jmp(entry)
                tail = chain_tail
            else:
                new.insts[-1] = Jmp(cont)
                tail = new
        elif isinstance(term, CallMayPanic) and len(term.unwind_defers) > 0:
            entry, chain_tail = _instantiate_chain(term.unwind_defers, term.unwind_path)
            _repoint_phi_incoming(term.unwind_path, new, chain_tail)
            term.unwind_path = entry
            term.unwind_defers = ()
        elif isinstance(term, Panic) and len(term.unwind_defers) > 0:
            assert term.unwind_path is not None
            entry, chain_tail = _instantiate_chain(term.unwind_defers, term.unwind_path)
            _repoint_phi_incoming(term.unwind_path, new, chain_tail)
            term.unwind_path = entry
            term.unwind_defers = ()
        elif isinstance(term, CatchUnwind):
            # the catch edge carries no deferred body: the panic is caught here
            pass
    assert tail is not None, 'a deferred body copy has an EndDefer'
    return bmap[root], tail


def _remap_targets(inst: Inst, bmap: dict[BasicBlock, BasicBlock]) -> Inst:
    """Rewrite the block references of one cloned instruction ``inst`` onto the
    copied blocks ``bmap``.  The deferred bodies a transfer has of its own are
    expanded separately, once every block reference has been remapped."""
    match inst:
        case Jmp():
            inst.target = bmap[inst.target]
            return inst
        case Br():
            inst.if_true = bmap[inst.if_true]
            inst.if_false = bmap[inst.if_false]
            return inst
        case CallMayPanic():
            inst.normal_path = bmap[inst.normal_path]
            inst.unwind_path = bmap[inst.unwind_path]
            return inst
        case CatchUnwind():
            inst.normal_path = bmap[inst.normal_path]
            inst.catch_path = bmap[inst.catch_path]
            return inst
        case Panic():
            if inst.unwind_path is not None:
                inst.unwind_path = bmap[inst.unwind_path]
            return inst
        case Switch():
            inst.default = bmap[inst.default]
            inst.cases = tuple((value, bmap[block]) for value, block in inst.cases)
            return inst
        case Phi():
            inst.incomings = [(value, bmap[pred]) for value, pred in inst.incomings]
            return inst
        case _:
            return inst


# defer instantiation ---

@dataclass(eq=False, slots=True)
class _ChainInstance:
    """One instantiated defer chain, cached by the deferred bodies it runs and
    the block it continues in (see ``instantiate_defers``).

    ``entry`` is the block every trigger of the chain transfers to and ``tail``
    the block that reaches the continuation (or, for a return chain, ends the
    shared copy of the return).  The entry is the merge point of every trigger,
    so the values the triggers deliver to the continuation cannot reach it along
    their own edge any more: ``phis`` maps each ``Phi`` of a real continuation
    to the ``Phi`` synthesized at the entry that merges this chain's triggers'
    values for it, and ``ret_value`` is that merge for a return chain (which has
    no continuation block to hold it).  ``triggers`` are the transfers that were
    redirected to the chain."""

    entry: BasicBlock
    tail: BasicBlock
    phis: dict[Phi, Phi]
    ret_value: Phi | None
    triggers: list[BasicBlock]


def _phi_incoming(phi: Phi, pred: BasicBlock) -> Value | None:
    """The value ``phi`` takes when control arrives from ``pred``, or None when
    it has no incoming from it (a phi of a continuation a chain feeds always
    does, for every trigger of the chain)."""
    for value, block in phi.incomings:
        if block is pred:
            return value
    return None


def instantiate_defers(fn: Function) -> None:
    """Expand the shared deferred bodies of ``fn`` into explicit copies,
    rewriting the blocks of ``fn`` in place.  A copy is shared by every transfer
    that runs the same deferred bodies and continues in the same block, so the
    same path is instantiated only once; every transfer is replaced by a
    transfer that runs its deferred bodies first, so that afterwards no reachable
    transfer carries deferred blocks and no reachable :class:`EndDefer` is left,
    and the shared templates become unreachable and are never lowered.  The
    expansion is driven by the deferred blocks the interpreter collected on every
    transfer, so it is this pass - not the lowering - that decides where a shared
    body continues.

    A shared copy has several predecessors, so the values a trigger delivers to
    the continuation - the incomings of its ``Phi``s, or the value a return
    returns - can no longer flow along its own edge.  The chain entry merges them
    into a fresh ``Phi`` instead and the continuation reads that one value (see
    :class:`_ChainInstance`).  The merges are recorded as the triggers are found
    and folded in once the whole function has been walked."""
    # every chain instantiated so far, keyed by the deferred bodies it runs and
    # the block it continues in (None for a return, whose chain ends in a fresh
    # copy of the return rather than a jump to a continuation block)
    chains: dict[tuple[tuple[BasicBlock, ...], BasicBlock | None], _ChainInstance] = {}

    def jump_chain(entries: tuple[BasicBlock, ...], cont: BasicBlock, trigger: BasicBlock) -> BasicBlock:
        """The entry of the chain running ``entries`` and then continuing in
        ``cont``, created on first use, with the values ``trigger`` delivers to
        the phis of ``cont`` recorded into the entry's merge phis.  Returns the
        block ``trigger`` is to transfer to."""
        chain = chains.get((entries, cont))
        if chain is None:
            entry, tail = _instantiate_chain(entries, cont)
            chain = _ChainInstance(entry, tail, {}, None, [])
            chains[(entries, cont)] = chain
        chain.triggers.append(trigger)
        for cont_phi in cont.insts:
            if not isinstance(cont_phi, Phi):
                continue
            value = _phi_incoming(cont_phi, trigger)
            if value is None:
                continue
            entry_phi = chain.phis.get(cont_phi)
            if entry_phi is None:
                entry_phi = Phi([])
                chain.phis[cont_phi] = entry_phi
                chain.entry.insts.insert(0, entry_phi)
            entry_phi.add_incoming(value, trigger)
        return chain.entry

    def return_chain(entries: tuple[BasicBlock, ...], value: Value | None, trigger: BasicBlock) -> BasicBlock:
        """The entry of the chain running ``entries`` and then returning
        ``value``, created on first use; the values the return sites return are
        merged into a phi at the entry.  Returns the block ``trigger`` is to
        transfer to."""
        chain = chains.get((entries, None))
        if chain is None:
            tail = BasicBlock()
            entry, _ = _instantiate_chain(entries, tail)
            merged: Phi | None = None
            if value is not None:
                merged = Phi([])
                entry.insts.insert(0, merged)
            tail.emit(Ret(merged))
            chain = _ChainInstance(entry, tail, {}, merged, [])
            chains[(entries, None)] = chain
        chain.triggers.append(trigger)
        if value is None:
            assert chain.ret_value is None, 'a return chain cannot mix a value and none'
        else:
            assert chain.ret_value is not None, 'a return chain cannot mix a value and none'
            chain.ret_value.add_incoming(value, trigger)
        return chain.entry

    for block in fn.entry.collect_blocks():
        if len(block.insts) == 0:
            continue
        last = block.insts[-1]
        if isinstance(last, Jmp) and len(last.defer_blocks) > 0:
            last.target = jump_chain(last.defer_blocks, last.target, block)
            last.defer_blocks = ()
        elif isinstance(last, Br) and (
            len(last.if_true_defer_blocks) > 0 or len(last.if_false_defer_blocks) > 0
        ):
            if len(last.if_true_defer_blocks) > 0:
                last.if_true = jump_chain(last.if_true_defer_blocks, last.if_true, block)
                last.if_true_defer_blocks = ()
            if len(last.if_false_defer_blocks) > 0:
                last.if_false = jump_chain(last.if_false_defer_blocks, last.if_false, block)
                last.if_false_defer_blocks = ()
        elif isinstance(last, Ret) and len(last.defer_blocks) > 0:
            # a return has no target to send the chain to: it ends in a fresh
            # copy of the return itself
            block.insts[-1] = Jmp(return_chain(last.defer_blocks, last.value, block))
        elif isinstance(last, CallMayPanic) and len(last.unwind_defers) > 0:
            # the unwind edge runs its deferred bodies first and reaches the
            # function-wide resume block (which holds no phis: the exception is
            # carried through the landing pad slot, see ``lower``)
            last.unwind_path = jump_chain(last.unwind_defers, last.unwind_path, block)
            last.unwind_defers = ()
        elif isinstance(last, Panic) and len(last.unwind_defers) > 0:
            assert last.unwind_path is not None
            last.unwind_path = jump_chain(last.unwind_defers, last.unwind_path, block)
            last.unwind_defers = ()
    # every trigger is known now: the continuations no longer receive their
    # values from the triggers themselves (the shared tail reaches them instead),
    # so replace those incoming edges by the one merged value the entry holds
    for chain in chains.values():
        triggers = set(chain.triggers)
        for cont_phi, entry_phi in chain.phis.items():
            cont_phi.incomings = [
                (value, pred) for value, pred in cont_phi.incomings if pred not in triggers
            ]
            cont_phi.incomings.append((entry_phi, chain.tail))
    # the invariant of this pass: nothing deferred is left on the reachable CFG
    for block in fn.entry.collect_blocks():
        for inst in block.insts:
            if isinstance(inst, EndDefer):
                raise CompileError('a deferred body was not instantiated')
            if isinstance(inst, (Jmp, Br, Ret, CallMayPanic, Panic)) and len(inst.get_defer_blocks()) > 0:
                raise CompileError('a deferred body was not instantiated')

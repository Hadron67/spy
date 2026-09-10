"""The spy type system of the compile-time interpreter.

Types appear in two roles:

* as the type annotation values in AOT functions (``spy.u64``,
  ``spy.f64``, ...), and
* as compile-time values inside a function body (``spy.typeof(a) ==
  spy.u64``).

The static types attached to the registers of the typed MIR are the
mirrors of these types defined by ``mir``; the interpreter converts
between the two when it emits instructions.

Types are immutable and compare structurally (two ``IntType(64, False)``
types are structurally (two ``IntType(64, False)``
are equal), which is what makes the compile-time comparisons in
``spy.typeof(a) == spy.u64`` work.
"""

from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, override

from spy.util import frozendict

from . import mir
from .errors import SpyError

if typing.TYPE_CHECKING:
    from spy.fn import RawArgList

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.typeof``) and other compile-time objects.
    Concrete values expose their spy type as ``.type``."""
    @abstractmethod
    def get_type(self) -> Type:
        ...

AnyValue = Value | int | float | str | bool


class AsValue(Value):
    """A Python value bound to an explicit spy type (``spy.as_(x, T)``).

    It appears only at the Python call boundary: the interpreter reads
    the type off it (``type_of`` returns :attr:`type`) to type the call,
    while the native call is handed the wrapped Python value.  Defining
    it here (rather than in ``builtins``) keeps the boundary marshaling
    of ``sval`` self-contained."""

    def __init__(self, value: Any, type: Type) -> None:
        self.value = value
        self.type = type

    @override
    def get_type(self) -> Type:
        return self.type

    def __repr__(self) -> str:
        return f'AsValue({self.value!r}, {self.type!r})'

class Type(Value):
    def get_unit_value(self) -> AnyValue | None:
        """Th          e canonical *unit value* of a zero-sized type (ZST): ``None``
        when the type has a runtime representation (it is not
        zer   o-sized), othervc       wise the one compile-time value every value of
        the type equals - ``Void()`` for the void type, ``Int(0, T)`` for
        a zero-bit integer, an ``AggregateValue`` for a struct whose
        fields are all ZSTs.  A ZST has no runtime representation: its
        ``mir`` mirror is ``None`` (``to_mir_type`` returns ``None``)."""
        return None

    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, self.__class__)

    def resolve_peer_type(self, other: Type) -> Type | None:
        return other if self.is_subtype_of(other) else None

    @abstractmethod
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        """The MIR mirror of this spy type: the static type the runtime
        register of a value of this type has.  The mapping is one-to-one
        over the types that can cross into runtime code.  A zero-sized
        type has no runtime representation and mirrors to ``None`` (the
        MIR's "void": a function whose return type is a ZST returns
        void); a spy struct type mirrors to one :class:`mir.StructType`
        object (created lazily and cached on the descriptor), so that all
        values of one struct share one identity.  Types with no MIR
        mirror at all (``TypeType``, ``AnyFunction``, ...) are a compile
        error."""
        ...

    def is_zst(self) -> bool:
        return isinstance(self.to_mir_type(), mir.VoidType)

@dataclass(frozen=True)
class TypeType(Type):
    level: int
    @override
    def get_type(self) -> Type:
        return TypeType(self.level + 1)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        type = other.get_type()
        assert isinstance(type, TypeType)
        return self.level <= type.level

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return f'type({self.level})'

class TypeVar(Type):
    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def __eq__(self, value: object, /) -> bool:
        return self is value

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class TupleType(Type):
    types: tuple[Type, ...]
    has_ellipsis: bool

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return f'tuple[{", ".join(str(t) for t in self.types)}{", ..." if self.has_ellipsis else ""}]'


@dataclass(frozen=True)
class StrDictType(Type):
    values: frozendict[str, Type]

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return f'{{{", ".join(f"{k}: {v}" for k, v in self.values.items())}}}'

TYPE_TYPE = TypeType(0)

@dataclass(frozen=True)
class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.BoolType:
        return mir.BoolType()

    def __str__(self) -> str:
        return 'bool'

@dataclass(frozen=True)
class EmptyType(Type):
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        return other

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return 'empty'

@dataclass(frozen=True)
class VoidType(Type):
    """The unit type: a zero-sized type (ZST) whose unique value is
    :class:`Void` (``sval.Void()``).  It is the spy type of ``None`` -
    the return type of a function that returns no value (declared as
    ``-> None``, or inferred for a body without value returns) - and it
    has no runtime representation: its ``mir`` mirror is ``None``
    (``to_mir_type`` returns ``None``) and no load/store is ever
    emitted for it."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        return Void()

    @override
    def to_mir_type(self) -> mir.VoidType:
        return mir.VOID

    def __str__(self) -> str:
        return 'void'

class Void(Value):
    """The unique *value* of the unit type :class:`VoidType` (which is a
    zero-sized type): the compile-time object that denotes "no value" -
    the result of a void call, the yield of a void inlined body, ...  It
    replaces the ``None`` sentinel the interpreter used for these."""

    @override
    def get_type(self) -> Type:
        return VoidType()

    def __str__(self) -> str:
        return 'void{{}}'

@dataclass
class ConstRef(Value):
    value: AnyValue

    @override
    def get_type(self) -> Type:
        return PointerType(type_of(self.value), is_const=True)

    def __str__(self) -> str:
        return '&' + str(self.value)

class BuiltinFn(Value):
    """A ``spy.*`` builtin that the compile-time interpreter evaluates
    while running the HIR (``spy.typeof``, ``spy.compile_log``).  The
    name identifies the builtin to the interpreter; ``spy.as_`` is not
    a compile-time builtin (it only exists at the call boundary)."""

    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def get_type(self) -> Type:
        return AnyFunction()

    def __str__(self) -> str:
        return f'spy.{self.name}'

@dataclass(frozen=True)
class AnyIntType(Type):
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return 'int'


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        if self.bits == 0:
            return Int(0, self)
        return None

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return mir.VOID if self.bits == 0 else mir.IntType(self.bits, self.signed)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        if isinstance(other, AnyIntType):
            return True
        if not isinstance(other, IntType):
            return False
        self_range = int_range(self)
        other_range = int_range(other)
        return self_range[0] >= other_range[0] and self_range[1] <= other_range[1]

    def peer_type_with_value(self, value: int):
        lower, upper = int_range(self)
        lower = min(lower, value)
        upper = max(upper, value)
        return min_int_type(lower, upper)

    def resolve_peer_type(self, other: Type) -> Type | None:
        match other:
            case IntType():
                self_range = int_range(self)
                other_range = int_range(other)
                return min_int_type(min(self_range[0], other_range[0]), max(self_range[1], other_range[1]))
            case ValueType():
                match other.value:
                    case int():
                        return self.peer_type_with_value(other.value)
                    case Int():
                        return self.peer_type_with_value(other.value.value)
        return None

    def __str__(self) -> str:
        return f'{'i' if self.signed else 'u'}{self.bits}'

@dataclass(frozen=True)
class Int(Value):
    value: int
    type: IntType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return str(self.value) + str(self.type)

@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __post_init__(self) -> None:
        assert self.bits in (32, 64), f"unsupported float bits {self.bits}"

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.FloatType:
        return mir.FloatType(self.bits)

    def is_subtype_of(self, other: Type) -> bool:
        if not isinstance(other, FloatType):
            return False
        return self.bits <= other.bits

    def __str__(self) -> str:
        return f"f{self.bits}"

@dataclass(frozen=True)
class Float(Type):
    value: float
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.value}{self.type}"

@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: bool = False

    @override
    def get_type(self) -> Type:
        child = self.elem.get_type()
        assert isinstance(child, TypeType)
        return TypeType(child.level + 1)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        child = self.elem.to_mir_type()
        if child is None:
            return None
        return mir.PointerType(child, self.is_const)

    def __str__(self) -> str:
        return f"{'ptr' if not self.is_const else 'cptr'}({self.elem})"

@dataclass(frozen=True)
class Undefined(Value):
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return "undefined"

@dataclass(frozen=True)
class ValueType(Type):
    value: AnyValue

    @staticmethod
    def create(value: AnyValue):
        type = type_of(value)
        return ValueType(value) if type.get_unit_value() is None else type

    @override
    def get_type(self) -> Type:
        return type_of(self.value).get_type()

    @override
    def get_unit_value(self) -> AnyValue | None:
        return self.value

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, ValueType):
            return self if self.value == other.value else None
        return other.resolve_peer_type(self)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return f"Literal({self.value})"


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type
    default_value: AnyValue | None


@dataclass(frozen=True)
class FunctionType(Type):
    args: tuple[FormalArg, ...]
    return_type: Type

    @property
    def via_result_ptr(self) -> bool:
        """Whether a call of a function of this signature delivers its
        result by writing into a caller-provided result location (see
        :func:`returns_via_result_ptr`)."""
        return returns_via_result_ptr(self.return_type)

    @override
    def get_type(self) -> Type:
        level = 0
        for arg in self.args:
            child = arg.type.get_type()
            assert isinstance(child, TypeType)
            level = max(level, child.level)
        child = self.return_type.get_type()
        assert isinstance(child, TypeType)
        level = max(level, child.level)
        return TypeType(level)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        args: list[mir.Type] = []
        for arg in self.args:
            mir_type = arg.type.to_mir_type()
            if mir_type is None:
                return None
            if not isinstance(mir_type, mir.VoidType):
                args.append(mir_type)
        ret_type = self.return_type.to_mir_type()
        if ret_type is None:
            return None
        return mir.FunctionType(tuple(args), ret_type)

    def __str__(self) -> str:
        return f"fn({', '.join(str(arg.type) for arg in self.args)}) -> {self.return_type}"


@dataclass(frozen=True)
class StructField:
    name: str
    type: Type

@dataclass(frozen=True)
class AggregateValue(Value):
    values: tuple[AnyValue, ...]
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type}({', '.join(str(value) for value in self.values)})"


class StructType(Type):
    """A spy struct type: the object ``@cache.struct()`` binds to the class
    name.  It doubles as the Python-side constructor of struct *values*:
    calling ``Foo(a, b)`` creates a native struct instance whose memory
    follows the LLVM layout (see ``_py_cls``).

    The identity of the object *is* the identity of the type (two structs
    are equal only if they are the same object), which is what makes
    ``spy.typeof(x) == Foo`` work.
    """

    def __init__(self, name_base: str) -> None:
        self.name_base = name_base
        self._fields: list[StructField] = []
        # the spy methods of the struct, by name.  ``@cache.struct``
        # extracts them from the Python class when the struct is created
        # (a decorated method contributes its registration handle, a
        # plain function stays a plain function and is inlined on call);
        # methods may also be added later, including methods that have no
        # counterpart in the Python class
        self.methods: dict[str, Any] = {}
        # the ``__init__`` method (a registration handle) of a
        # user-provided constructor, or None when the default constructor
        # (which writes the arguments into the fields in declaration
        # order) applies
        self.custom_init: Any | None = None
        # the Python-side callable that constructs struct instances
        # (``Foo(a, b)``); installed by the host when the struct is
        # created
        self._py_init: Any = None
        # the Python class of the struct instances (a ctypes.Structure
        # subclass mirroring the LLVM layout); installed by the host
        self._py_cls: Any = None
        # the MIR mirror of the type, created lazily when the lowering
        # of a function body needs it (see ``to_mir_type``); None until
        # then
        self._mir: mir.MayBeVoidType | None = None
        # the mirror position of every field, in declaration order (see
        # ``get_field_mir_indices``), computed together with the mirror
        self._field_mir_indices: tuple[int | None, ...] | None = None

    def add_field(self, name: str, type: Type) -> None:
        assert all(f.name != name for f in self._fields)
        self._fields.append(StructField(name, type))

    def bind_default_ctor_args[T](self, args: RawArgList[T]) -> tuple[T | None, ...]:
        """Bind the arguments of a struct construction ``Foo(...)`` to
        the fields of the default constructor (the fields in declaration
        order).  Positional arguments bind the leading fields, keyword
        arguments bind fields by name.  A zero-sized field occupies no
        storage and needs no argument: binding it yields ``None`` (the
        caller writes nothing to it).  A missing argument for a field
        with a runtime representation is a ``TypeError``."""
        ret: list[T | None] = []
        positional = args.positional
        kwargs = args.kwargs
        field_names = {f.name for f in self._fields}
        for key in kwargs:
            if key not in field_names:
                raise TypeError(f"got an unexpected keyword argument '{key}'")
        for i, field in enumerate(self._fields):
            if i < len(positional):
                if field.name in kwargs:
                    raise TypeError(f"got multiple values for field '{field.name}'")
                ret.append(positional[i])
                continue
            if field.name in kwargs:
                ret.append(kwargs[field.name])
                continue
            if field.type.get_unit_value() is not None:
                ret.append(None)
                continue
            raise TypeError(f"missing a value for field '{field.name}'")
        if len(positional) > len(self._fields):
            raise TypeError(
                f"takes {len(self._fields)} positional arguments but "
                f"{len(positional)} were given"
            )
        return tuple(ret)

    @property
    def fields(self) -> tuple[StructField, ...]:
        return tuple(self._fields)

    def field_index(self, name: str) -> int | None:
        """The declaration index of the field ``name``, or None when the
        struct has no such field."""
        for i, field in enumerate(self._fields):
            if field.name == name:
                return i
        return None

    def field_type(self, name: str) -> Type | None:
        """The spy type of the field ``name``."""
        index = self.field_index(name)
        return self._fields[index].type if index is not None else None

    def method_of(self, name: str) -> Any:
        """The spy method ``name`` of the struct: its registration handle
        (a decorated ``@aot``/``@jit`` method) or its plain function
        (inlined on call).  Raises a ``KeyError`` when the struct has no
        such method."""
        return self.methods[name]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Python-side construction of a struct value: ``Foo(a, b)``
        allocates a native instance and runs the struct's constructor
        (the ``__init__`` method, or the default field-wise constructor)
        on it."""
        if self._py_init is None:
            raise SpyError(
                f'struct {self.name_base} is not bound to a spy context; '
                'define it with @cache.struct() and construct it after the '
                'module has loaded'
            )
        return self._py_init(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<spy struct {self.name_base}>'

    def __str__(self) -> str:
        return self.name_base

    @override
    def get_type(self) -> Type:
        level = 0
        for field in self._fields:
            child = field.type.get_type()
            assert isinstance(child, TypeType)
            level = max(level, child.level)
        return TypeType(level)

    @override
    def get_unit_value(self) -> AnyValue | None:
        values: list[AnyValue] = []
        for field in self._fields:
            val = field.type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return self.get_mir_type()

    def _calculate_mir(self) -> None:
        """Build (once) the MIR mirror of the struct together with the
        position map of its fields: the fields whose spy type has no
        runtime representation (``to_mir_type`` is ``None`` - a
        zero-sized type) occupy no storage and are dropped from the
        mirror, so a field's mirror position differs from its
        declaration index, and a zero-sized field itself has no
        position at all (its map entry is ``None``)."""
        if self._mir is None:
            assert self._field_mir_indices is None
            fields: list[mir.FormalArg] = []
            field_mir_indices: list[int | None] = []
            for field in self._fields:
                field_type = field.type.to_mir_type()
                assert field_type is not None, f"field {field.name!r} has no MIR representation"
                if isinstance(field_type, mir.VoidType):
                    field_mir_indices.append(None)
                else:
                    field_mir_indices.append(len(fields))
                    fields.append(mir.FormalArg(field.name, field_type))
            self._field_mir_indices = tuple(field_mir_indices)
            if all(a is None for a in self._field_mir_indices):
                self._mir = None
            else:
                self._mir = mir.StructType(self.name_base, tuple(fields))

    def get_mir_type(self) -> mir.MayBeVoidType:
        """The (cached) MIR mirror of the struct: the one
        :class:`mir.StructType` object every value of the struct
        mirrors to (created lazily, shared by all users), with the
        zero-sized fields dropped from the LLVM layout."""
        self._calculate_mir()
        assert self._mir is not None
        return self._mir

    def get_field_mir_indices(self) -> tuple[int | None, ...]:
        """The mirror position of every field, in declaration order: the
        i-th entry is the position of the i-th field in the mirror
        returned by :meth:`get_mir_type` - zero-sized fields occupy no
        position and map to ``None``."""
        self._calculate_mir()
        assert self._field_mir_indices is not None
        return self._field_mir_indices

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)


# ---------------------------------------------------------------------------
# function values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnyFunction(Type):
    """The type of a function value whose signature is not known: a lazy
    ``@jit`` function is only typed when a call specializes it.  It has
    no MIR mirror - such a value never crosses into runtime code."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return "anyfn"


def int_range(type: IntType) -> tuple[int, int]:
    if type.signed:
        return (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
    return (0, 2 ** type.bits - 1)

def min_int_type(lower: int, upper: int) -> IntType:
    bits = (lower - 1).bit_length() + 1
    bits_upper = (upper - 1).bit_length() + 1
    return IntType(max(bits, bits_upper), lower < 0)

# ---------------------------------------------------------------------------
# the return convention of a type: whether a function returning it returns a
# value, or writes the result into a caller-provided result location
# ---------------------------------------------------------------------------

# an aggregate of at most this many bytes is returned by value by default;
# larger ones are returned through a result pointer (the limit matches the
# size that the C ABIs of the supported targets pass in registers)
_AGGREGATE_VALUE_RETURN_LIMIT = 16


def _alignment_of(type: Type) -> int:
    """The natural alignment of a type, in bytes (the layout rules of
    the ctypes instances - and of the LLVM structs they mirror - for the
    types that may occur in a struct).  Zero-sized (ZST) types occupy no
    storage and are skipped inside aggregates."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8 if type.bits != 0 else 1
        case FloatType():
            return type.bits // 8
        case StructType():
            return max(
                (
                    _alignment_of(f.type)
                    for f in type.fields
                    if f.type.get_unit_value() is None
                ),
                default=1,
            )
        case _:
            raise SpyError(f"type {type} has no layout")


def _size_of(type: Type) -> int:
    """The size of a type in bytes, rounded up to its alignment (the
    natural layout the ctypes instances - and the LLVM structs they
    mirror - use).  A ZST is zero-sized; ZST fields occupy no storage
    inside an aggregate."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8 if type.bits != 0 else 0
        case FloatType():
            return type.bits // 8
        case StructType():
            offset = 0
            for field in type.fields:
                if field.type.get_unit_value() is not None:
                    # a zero-sized field occupies no storage
                    continue
                align = _alignment_of(field.type)
                offset = (offset + align - 1) // align * align
                offset += _size_of(field.type)
            align = _alignment_of(type)
            return (offset + align - 1) // align * align
        case _:
            raise SpyError(f"type {type} has no layout")


def returns_via_result_ptr(type: Type) -> bool:
    """Whether a function returning ``type`` delivers its result by
    writing into a caller-provided result location (a hidden result
    pointer parameter) instead of returning the value directly.

    The convention is a property of the *return type*, decided here once
    and consulted everywhere a function's signature is lowered (the
    function type is the single source of the decision - never the
    registration entry).  The default policy: aggregates are returned by
    value while they are small (up to
    :data:`_AGGREGATE_VALUE_RETURN_LIMIT` bytes) and through a result
    pointer once they outgrow it; a future per-struct override or a new
    aggregate kind (arrays) only needs to extend this function.  Scalars
    are always returned by value."""
    match type:
        case StructType():
            return _size_of(type) > _AGGREGATE_VALUE_RETURN_LIMIT
        case _:
            return False


def pass_by_ref(type: Type) -> bool:
    """Whether a parameter of spy type ``type`` is passed by reference
    (as a const pointer) rather than by value: the calling convention of
    one argument, decided by the compiler.

    The policy mirrors :func:`returns_via_result_ptr`: an aggregate too
    large to be passed in registers (larger than the by-value limit) is
    passed as a pointer, everything else by value.  A parameter whose
    formal declares it as a reference (``SignatureFormalArg.by_ref``) is
    passed by reference regardless of its type."""
    match type:
        case StructType():
            return _size_of(type) > _AGGREGATE_VALUE_RETURN_LIMIT
        case _:
            return False


def concretize_type(type: Type) -> Type:
    """Replace a compile-time-only type by the concrete type a runtime
    value of it is represented by: the open integer type ``AnyIntType``
    (the type of a plain Python ``int``) becomes the default signed
    integer type (:data:`INT_DEFAULT_BITS`).  Every other type passes
    through unchanged."""
    if isinstance(type, AnyIntType):
        return IntType(INT_DEFAULT_BITS, True)
    return type


# ---------------------------------------------------------------------------
# mapping Python values to spy types
# ---------------------------------------------------------------------------


def type_of(value: AnyValue) -> Type:
    """The spy type a Python *value* is marshaled to at the call boundary.

    ``None`` is returned for values that have no spy representation (e.g.
    compile-time objects like type descriptors, which never cross the
    boundary).
    """
    if isinstance(value, Value):
        return value.get_type()
    match value:
        case bool():
            return BoolType()
        case int():
            return AnyIntType()
        case float():
            return FloatType(64)
        case str():
            # strings are compiled as arrays of u8; until arrays get their
            # own type they are represented by a const pointer to u8
            return PointerType(IntType(8, False), is_const=True)


def value_repr(value: object) -> str:
    """Human readable description of a Python value (used in errors)."""
    match value:
        case int():
            return f'integer {value}'
        case float():
            return f'float {value}'
        case str():
            return 'string'
        case bool():
            return 'bool'
        case _:
            cls = type(value)
            descriptor = getattr(cls, '__spy_struct_type__', None)
            if descriptor is not None:
                return f'a {descriptor.name} struct'
            return repr(value)

def as_value(value: Any, type_vars: dict[typing.TypeVar, Value] | None = None) -> AnyValue:
    """The spy-domain value of a Python compile-time object: Python
    scalars and ``sval.Value`` objects pass through, and ``None`` is the
    unit value of the zero-sized void type (``Void()``).  Class objects
    of the scalar types map to their default spy types."""
    if isinstance(value, (Value, int, float, str, bool)):
        return value
    if value is None:
        # ``None`` denotes the void value: the unit value of the
        # zero-sized ``VoidType``
        return Void()
    if value is int:
        return IntType(32, True)
    if value is float:
        return FloatType(64)
    if value is str:
        return PointerType(IntType(8, False), is_const=True)
    if value is bool:
        return BoolType()
    if isinstance(value, typing.TypeVar):
        if type_vars is None:
            raise TypeError(f'cannot convert {value} to a value')
        return type_vars[value]

    raise TypeError(f'cannot convert {value} to a value')

def negate(value: AnyValue) -> AnyValue | None:
    if isinstance(value, (int, float)):
        return -value
    return None

@dataclass(frozen=True)
class _Constraint:
    lhs: Value
    rhs: Value
    is_subtype: bool = False  # True when lhs is a subtype of rhs

class _SolvedTypeVar:
    """The solver state of one type parameter.

    A parameter is either *solved* - bound to a value (``_value``), with
    no subtype bounds recorded - or *bounded*: declared a subtype of
    every value in ``_subtypes`` (its upper bounds), without a solution
    yet.  Solving the equality constraints of a bounded parameter binds
    it to a value that must satisfy every recorded bound."""

    def __init__(self) -> None:
        self._value: Value | None = None  # non-None: this type var is solved to this value, in this case _subtypes is None
        self._subtypes: set[Value] | None = None  # non-None: all values in this set are subtypes of this type var, in this case _value is None

    def _add_bound(self, value: Value) -> None:
        if self._subtypes is None:
            self._subtypes = set()
        self._subtypes.add(value)

class TypeVarSolver:
    """A constraint solver over spy values, used to resolve the type
    parameters of a generic function to the concrete spy types a call
    determines.

    ``add_constraint`` collects one constraint - an *equality*
    (``lhs == rhs``, the default) or a *subtyping* (``lhs <: rhs``,
    ``is_subtype=True``) - and ``finish`` solves them, binding every
    constrained type parameter to a value.  Spy types have no
    structural subtyping (one concrete spy type is a subtype of another
    only when they are equal), so a subtype constraint is satisfiable
    exactly when an equality one is; the solver still tracks the bounds
    of a parameter separately so that ``finish`` can bind a parameter
    that only ever appears on the left of subtype constraints.  When a
    constraint cannot be satisfied, ``finish`` raises
    :class:`TypeMismatchError`.
    """

    def __init__(self) -> None:
        self._type_var_values: dict[TypeVar, _SolvedTypeVar] = {}
        self._unsatisfied: list[_Constraint] = []

    def _solved(self, tv: TypeVar) -> _SolvedTypeVar:
        stv = self._type_var_values.get(tv)
        if stv is None:
            stv = _SolvedTypeVar()
            self._type_var_values[tv] = stv
        return stv

    def _add_unsatisfied(self, lhs: Value, rhs: Value, is_subtype: bool = False) -> None:
        self._unsatisfied.append(_Constraint(lhs, rhs, is_subtype))

    def _solve_type_var_bound(self, v: TypeVar, bound: Value, is_subtype: bool) -> None:
        solved = self._solved(v)
        if is_subtype:
            assert solved._value is None
            if solved._subtypes is None:
                solved._subtypes = set()
            solved._subtypes.add(bound)
        else:
            if solved._subtypes is not None:
                for st in solved._subtypes:
                    assert isinstance(st, Type) and isinstance(bound, Type)
                    if not st.is_subtype_of(bound):
                        self._add_unsatisfied(st, bound, True)
                solved._subtypes = None
            solved._value = bound

    def add_constraint(self, lhs: Value, rhs: Value, is_subtype: bool = False):
        todo = [(lhs, rhs, is_subtype)]
        while todo:
            lhs, rhs, is_subtype = todo.pop()
            if lhs == rhs:
                continue
            if is_subtype:
                # in the case we concern, TypeVar cannot appear on the left side of a subtype constraint
                assert not isinstance(lhs, TypeVar)
                if isinstance(rhs, TypeVar):
                    self._solve_type_var_bound(rhs, lhs, True)
                if isinstance(lhs, TypeVar) and isinstance(rhs, TypeVar) and not lhs.is_subtype_of(rhs):
                    self._add_unsatisfied(lhs, rhs, is_subtype)
                self._add_unsatisfied(lhs, rhs, is_subtype)
            else:
                if isinstance(rhs, TypeVar) and not isinstance(lhs, TypeVar):
                    t = lhs
                    lhs = rhs
                    rhs = t

                if isinstance(lhs, TypeVar):
                    self._solve_type_var_bound(lhs, rhs, False)

                # only top-level spy values are unified for now: a
                # constraint between two compound types (an aggregate
                # containing a type parameter, a pointer to one, ...) is
                # recorded as unsatisfied and reused once the solver
                # learns to unify their children
                if lhs != rhs:
                    self._add_unsatisfied(lhs, rhs, is_subtype)

    def finish(self):
        for type_var, sv in self._type_var_values.items():
            if sv._value is None and sv._subtypes is not None:
                type = EmptyType()
                for st in sv._subtypes:
                    assert isinstance(st, Type)
                    pt = type.resolve_peer_type(st)
                    if pt is None:
                        self._add_unsatisfied(type_var, st, True)
                        continue
                    type = pt
                sv._value = type

    def get_solved(self) -> dict[TypeVar, Value]:
        return {
            k: v._value
            for k, v in self._type_var_values.items()
            if v._value is not None
        }

def replace_type_var(value: Value, reps: dict[TypeVar, Value]) -> Value:
    match value:
        case TypeVar():
            return reps.get(value, value)
        case _:
            return value

def replace_type_vars_type(type: Type, reps: dict[TypeVar, Value]) -> Type:
    ret = replace_type_var(type, reps)
    assert isinstance(ret, Type)
    return ret

def is_comptime_only_type(type: Type) -> bool:
    match type:
        case AnyIntType() | TypeType():
            return True
        case _:
            return False

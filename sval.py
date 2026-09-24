"""The spy type system of the compile-time interpreter.

Types appear in two roles:

* as the parameter and return annotation values of a function
  (``spy.u64``, ``spy.f64``, ...), and
* as compile-time values inside a function body (``spy.typeof(a) ==
  spy.u64``).

The static types attached to the registers of the typed MIR are the
mirrors of these types defined by ``mir``; the interpreter converts
between the two when it emits instructions.

Types are immutable and compare structurally (two ``IntType(64, False)``
instances are equal) - except the identity types ``TypeVar`` and
``StructType``, which are equal only to themselves.  That is what makes
the compile-time comparisons in ``spy.typeof(a) == spy.u64`` work.
"""

from __future__ import annotations

import ctypes
import typing
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum, auto
from types import NoneType
from typing import Any, Literal, override

from spy.util import IdentityObj, IndexedMap, frozendict

from . import mir, syntax
from .errors import CompileError, SpyError

INT_DEFAULT_BITS = 32
"""The signedness/width of the default spy integer type: the type a
plain Python ``int`` maps to (see ``as_value``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.typeof``) and other compile-time objects.
    Concrete values report their spy type through ``get_type()``."""
    @abstractmethod
    def get_type(self) -> Type:
        ...

type AnyValue = Value | int | float | str | bool

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
        """The canonical *unit value* of a zero-sized type (ZST): ``None``
        when the type has a runtime representation (it is not
        zero-sized), otherwise the one compile-time value every value of
        the type equals - ``Void()`` for the void type, ``Int(0, T)`` for
        a zero-bit integer, an ``AggregateValue`` for a struct whose
        fields are all ZSTs.  A ZST has no runtime representation: its
        ``mir`` mirror is the void type (``to_mir_type`` returns
        ``mir.VOID``)."""
        return None

    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, self.__class__)

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, NullType):
            # a type and the null value peer to the option of the type
            return OptionType(self)
        if isinstance(other, OptionType):
            return _resolve_option_peer(self, other)
        return other if self.is_subtype_of(other) else None

    @abstractmethod
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        """The MIR mirror of this spy type: the static type the runtime
        register of a value of this type has.  A zero-sized type has no
        runtime representation and mirrors to the MIR's void type
        (:data:`mir.VOID` - a function whose return type is a ZST returns
        void); a spy struct type mirrors to the one MIR type every value
        of the struct shares (created lazily and cached on the
        descriptor, see :meth:`StructType._calculate_mir`).  Types that
        cannot cross into runtime code at all (``TypeType``,
        ``AnyFunction``, ...) have no mirror and return ``None``."""
        ...

    def is_zst(self) -> bool:
        return isinstance(self.to_mir_type(), mir.VoidType)

    def is_copyable(self) -> bool:
        return True

    def get_type_children(self) -> tuple[Type, ...]:
        return ()

    def contains(self, needle: Type):
        todo: list[Type] = [self]
        while todo:
            current = todo.pop()
            if current is needle:
                return True
            todo.extend(reversed(current.get_type_children()))
        return False

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
    def get_type(self) -> Type:
        # a type variable stands for a type of its own
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class TupleType(Type):
    """A Python ``tuple[T1, T2, ...]``: the return annotation of a function
    that returns several values.  It is a compile-time type only - a tuple
    of values has no runtime representation of its own; every function that
    returns one delivers its elements separately (see ``make_ret_spec``)."""

    types: tuple[Type, ...]
    has_ellipsis: bool

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """A tuple is a subtype of a tuple of the same element types (with
        the same fixed-or-varying shape) and of nothing else: like an array,
        the element types are compared for equality rather than subtyped, as
        the base rule would take any tuple for any tuple."""
        return isinstance(other, TupleType) and self == other

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return self.types

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
    :class:`Void` (``sval.Void()``).  It is the declared return type of a
    function that returns no value (``-> None``, or one inferred for a body
    without value returns), and it has no runtime representation: its
    ``mir`` mirror is the MIR void type (``to_mir_type`` returns
    ``mir.VOID``) and no load/store is ever emitted for it."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        return Void()

    @override
    def to_mir_type(self) -> mir.VoidType:
        return mir.VOID

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        # the null value converts to the void type: the peer of the two is the
        # void type itself (rather than the option of a void value)
        if isinstance(other, NullType):
            return self
        return super().resolve_peer_type(other)

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

@dataclass(frozen=True)
class NullType(Type):
    """The type of the :class:`Null` value - what the Python literal
    ``None`` evaluates to: no value of any particular type, which is exactly
    what a value of an ``Option[T]`` may be.  It is a zero-sized type (its
    unit value is :class:`Null`) that converts to the void type - so ``None``
    still works where a void value is expected - and to ``Option[T]`` for
    every ``T``."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        return Null()

    @override
    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, (NullType, VoidType, OptionType))

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        # Null is the absent value: it peers with the void type (it converts
        # to it), with an option (it is one of its values), and with any other
        # type by making it optional
        match other:
            case NullType() | VoidType() | OptionType():
                return other
            case _:
                return OptionType(other)

    @override
    def to_mir_type(self) -> mir.VoidType:
        return mir.VOID

    def __str__(self) -> str:
        return 'null'

class Null(Value):
    """The unique value of :class:`NullType`: the compile-time object the
    Python literal ``None`` denotes.  Its only use is as a value of an
    ``Option[T]``, where it is the absent one."""

    @override
    def get_type(self) -> Type:
        return NullType()

    def __str__(self) -> str:
        return 'null'

@dataclass(frozen=True)
class OptionType(Type):
    """The optional type ``Option[T]``: a value of the type ``T``, or the
    :class:`Null` value.  Both ``T`` and :class:`NullType` convert to it (see
    :meth:`resolve_peer_type` and :func:`coerce_const`), so a ``T`` and a
    ``NullType`` unify to an ``Option[T]``.

    The representation is chosen by :meth:`to_mir_type` from the child type
    (see :func:`find_first_pointer_type_pos`): a zero-sized ``T`` - which
    holds no value - only keeps whether a value is there, a ``T`` that still
    holds a free pointer uses it as the absent tag (the option then *is* the
    ``T``: ``Option[Ptr[X]]`` has the representation of ``Ptr[X]``), and any
    other ``T`` a struct of a ``bool`` tag and the ``T`` itself.  An option is
    therefore never zero-sized."""

    child: Type

    @override
    def get_type(self) -> Type:
        child = self.child.get_type()
        assert isinstance(child, TypeType)
        return TypeType(child.level + 1)

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return (self.child,)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, OptionType) and self.child.is_subtype_of(other.child)

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        match other:
            case NullType():
                return self
            case OptionType():
                child = self.child.resolve_peer_type(other.child)
            case _:
                child = self.child.resolve_peer_type(other)
        return None if child is None else OptionType(child)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        child = self.child
        child_mir = child.to_mir_type()
        if child_mir is None:
            return None
        if isinstance(child_mir, mir.VoidType):
            # a zero-sized child carries no value: only whether there is one
            return mir.BoolType()
        if find_first_pointer_type_pos(child) is not None:
            # the child holds a pointer, which is null exactly when the option
            # is: the child itself is the representation
            return child_mir
        # no pointer to use as the tag: a struct of the tag and the value
        return _option_struct_mir(child, child_mir)

    @override
    def is_copyable(self) -> bool:
        return self.child.is_copyable()

    def __str__(self) -> str:
        return f'Option[{self.child}]'

_OPTION_STRUCT_MIRS: dict[Type, mir.StructType] = {}

def _option_struct_mir(child: Type, child_mir: mir.Type) -> mir.StructType:
    """The MIR mirror of an ``Option[T]`` whose ``T`` has no pointer to be
    tagged on: a struct of a ``bool`` tag and the value.  The mirror is
    interned per child type - a MIR struct type is an identity object, so two
    ``Option[T]`` of the same ``T`` must share one (the module declares one
    LLVM struct per MIR struct type)."""
    ret = _OPTION_STRUCT_MIRS.get(child)
    if ret is None:
        ret = mir.StructType(
            'option',
            (
                mir.FormalArg('tag', mir.BoolType()),
                mir.FormalArg('value', child_mir),
            ),
        )
        _OPTION_STRUCT_MIRS[child] = ret
    return ret

def _resolve_option_peer(type: Type, option: OptionType) -> Type | None:
    """The peer type of ``type`` and the option ``option``: the peer type of
    their children, as an option (``Option[resolve_peer_type(type, option.child)]``)."""
    child = type.resolve_peer_type(option.child)
    return None if child is None else OptionType(child)

def find_first_pointer_type_pos(type: Type, shift: int = 0) -> tuple[int, ...] | None:
    """The position of the first pointer of ``type`` that an option wrapping it
    may use as its absent tag, in the order of ``get_type_children`` (a
    struct's fields, an array's element, an option's child), or ``None`` when
    no such pointer is left.  The position of a :class:`PointerType` itself is
    the empty tuple, and the position of one inside a child is that child's
    position followed by the position inside it.

    An ``Option`` consumes one pointer for its own tag (see
    ``OptionType.to_mir_type``), so the pointers it already claims are not
    available again: with ``n`` pointers ``Option[T]`` still has ``n - 1``, and
    entering an option while searching skips one (that is what ``shift``
    starts the count with - the outer ``Option[Option[T]]`` claims ``T``'s
    second pointer, not its first).  Iterative (an explicit stack and two
    counters), so nesting costs no Python stack."""
    pointers = 0
    claimed = shift
    todo: list[tuple[Type, tuple[int, ...]]] = [(type, ())]
    while todo:
        current, pos = todo.pop()
        if isinstance(current, PointerType):
            if claimed == pointers:
                return pos
            pointers += 1
            continue
        if isinstance(current, OptionType):
            # this option already uses one pointer of its child as its tag
            claimed += 1
        children = current.get_type_children()
        for index in range(len(children) - 1, -1, -1):
            todo.append((children[index], pos + (index,)))
    return None

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

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, (IntType, AnyIntType)):
            return other
        if isinstance(other, NullType):
            return OptionType(self)
        if isinstance(other, OptionType):
            return _resolve_option_peer(self, other)
        return None

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
        if isinstance(other, NullType):
            return OptionType(self)
        if isinstance(other, OptionType):
            return _resolve_option_peer(self, other)
        match other:
            case IntType():
                self_range = int_range(self)
                other_range = int_range(other)
                return min_int_type(min(self_range[0], other_range[0]), max(self_range[1], other_range[1]))
            case AnyIntType():
                return self
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

class PointerVariant(IntEnum):
    SINGLE = auto()
    MULTI = auto()

@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: AnyValue = False # bool
    variant: PointerVariant = PointerVariant.SINGLE

    @override
    def get_type(self) -> Type:
        child = self.elem.get_type()
        assert isinstance(child, TypeType)
        return TypeType(child.level + 1)

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        # the pointee type, and the constness while it is still a type
        # parameter (a pointer type constrains both, see ``TypeVarSolver``)
        if isinstance(self.is_const, Type):
            return (self.elem, self.is_const)
        return (self.elem,)

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        if not isinstance(self.is_const, bool):
            return None
        child = self.elem.to_mir_type()
        if child is None:
            return None
        return mir.PointerType(child, self.is_const)

    def __str__(self) -> str:
        return f"{'ptr' if not self.is_const else 'cptr'}({self.elem})"

@dataclass(frozen=True, slots=True)
class ArrayType(Type):
    """A spy array type: ``length`` values of the element type ``elem``, in a
    row.  The length is a *value* (a Python ``int``, or an ``Int``): a signature
    that takes or returns an array spells it out, since the Python type system
    cannot infer it from the arguments of ``array(...)`` (see ``syntax``)."""

    elem: Type
    length: AnyValue # int

    @property
    def length_int(self) -> int | None:
        """The length as a Python ``int``, or None when it is not one - a
        length written as a type parameter that no call has solved yet."""
        match self.length:
            case int():
                return self.length
            case Int():
                return self.length.value
            case _:
                return None

    @override
    def get_type(self) -> Type:
        child = self.elem.get_type()
        assert isinstance(child, TypeType)
        return TypeType(child.level + 1)

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        # the element type, and the length while it is still a type parameter
        # (a length that is a type constrains it, see ``TypeVarSolver``)
        if isinstance(self.length, Type):
            return (self.elem, self.length)
        return (self.elem,)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """An array is a subtype of an array of the same length and the same
        element type, and of nothing else: the two have to share their layout,
        so the element type is compared for equality rather than subtyped
        (widening the elements of an array is a conversion, not a subtype -
        the element types of one construction are unified one level down).
        Note that the base rule would take any array for any array."""
        return (
            isinstance(other, ArrayType)
            and self.length_int is not None
            and self.length_int == other.length_int
            and self.elem == other.elem
        )

    @override
    def get_unit_value(self) -> AnyValue | None:
        """The unit value of a zero-sized array: the aggregate of the unit
        values of its elements - an array is zero-sized when it has no
        elements at all, or when its element type is (see
        :meth:`to_mir_type`)."""
        length = self.length_int
        if length is None:
            return None
        unit = self.elem.get_unit_value()
        if unit is None:
            # a zero-length array holds no storage whatever its element type
            return AggregateValue((), self) if length == 0 else None
        return AggregateValue((unit,) * length, self)

    @override
    def is_copyable(self) -> bool:
        """An array is copyable when its element type is."""
        return self.length == 0 or self.elem.is_copyable()

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        length = self.length_int
        if length is None:
            return None
        if length == 0 or self.elem.is_zst():
            # a zero-sized array holds no storage whatever its length: it has
            # no mirror of its own
            return mir.VOID
        elem = self.elem.to_mir_type()
        if elem is None or isinstance(elem, mir.VoidType):
            # an element with no mirror of its own (the zero-sized case is
            # already handled above) leaves the array without one either
            return None
        return mir.ArrayType(elem, length)

    def __str__(self) -> str:
        return f"{self.elem}[{self.length}]"

@dataclass(frozen=True, slots=True)
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
        return mir.VoidType()

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
    # the spy type of the values the function returns: the type of a single
    # result, or a ``tuple[...]`` when it returns several (see
    # :func:`make_ret_spec`)
    return_type: Type

    @property
    def ret_spec(self) -> RetSpec:
        """How a call of a function of this signature delivers its result
        (see :func:`make_ret_spec`)."""
        return make_ret_spec(self.return_type)

    @override
    def get_type(self) -> Type:
        level = 0
        for arg in self.args:
            child = arg.type.get_type()
            assert isinstance(child, TypeType)
            level = max(level, child.level)
        for leaf in iter_ret_leaves(self.ret_spec):
            child = leaf.type.get_type()
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
        ret_type: mir.MayBeVoidType = mir.VOID
        for leaf in iter_ret_leaves(self.ret_spec):
            mir_type = leaf.type.to_mir_type()
            if mir_type is None:
                return None
            if leaf.via_result_ptr:
                if isinstance(mir_type, mir.VoidType):
                    return None
                args.append(mir.PointerType(mir_type, False))
            else:
                ret_type = mir_type
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

@dataclass(frozen=True)
class StructModifiers:
    extern_c: bool = False
    copyable: bool | Literal["inherit"] = "inherit"

class StructTypeHead(Type, IdentityObj):
    """The declaration of a spy struct: its name, its declared generic type
    parameters, its modifiers, its fields (in declaration order) and its
    methods, by name.  A struct *type* - what annotations and values name -
    is a specialization of a head (:class:`StructType`), so a non-generic
    struct has exactly one, and the head itself is only ever an annotation
    of a generic struct (a template is not a type of any value yet).
    """

    def __init__(self, name_base: str, generic_args: tuple[TypeVar, ...] = (), modifiers: StructModifiers | None = None) -> None:
        self.name_base = name_base
        self.generic_args = generic_args
        self.modifiers = modifiers or StructModifiers()
        self.fields: IndexedMap[str, StructField] = IndexedMap()
        self.methods: dict[str, Any] = {}
        self._specs: dict[tuple[Value, ...], StructType] = {}

    def add_field(self, name: str, type: Type) -> None:
        """Declare one field, appended after the fields declared so far."""
        assert name not in self.fields.by_key, f'{self.name_base} already has a field {name!r}'
        self.fields.add(name, StructField(name, type))

    def specialize(self, generic_args: tuple[Value, ...]) -> StructType:
        """The struct type this head declares for ``generic_args``: the one
        specialization of the head for those arguments (created lazily, so
        that every reference to the same struct type names one object)."""
        if len(generic_args) != len(self.generic_args):
            raise CompileError(
                f'{self.name_base} takes {len(self.generic_args)} generic '
                f'argument(s) but {len(generic_args)} were given'
            )
        if generic_args in self._specs:
            return self._specs[generic_args]
        ret = StructType(self, generic_args)
        self._specs[generic_args] = ret
        return ret

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        raise CompileError(
            f'{self} is a struct template: a struct type is a specialization '
            'of it, and only one has a MIR mirror'
        )

    def __repr__(self) -> str:
        return f'<spy struct {self}>'

    def __str__(self) -> str:
        if len(self.generic_args) == 0:
            return self.name_base
        return f'{self.name_base}[{", ".join(str(a) for a in self.generic_args)}]'

class StructType(Type):
    """A spy struct *type*: one specialization of a :class:`StructTypeHead`,
    which is what a value, an annotation or ``spy.typeof(x) == Foo`` names.

    The identity of the object *is* the identity of the type (two structs
    are equal only if they are the same object), which is what makes
    ``spy.typeof(x) == Foo`` work; a specialization is cached on its head,
    so naming the same struct twice names the same object.
    """

    def __init__(self, head: StructTypeHead, generic_args: tuple[Value, ...]) -> None:
        self.head = head
        self.generic_args = generic_args

        self._fields: IndexedMap[str, StructField] | None = None
        self._mir: mir.MayBeVoidType | None = None
        # whether the mirror is the mirror of the struct's own single stored
        # field rather than a wrapper struct (see ``mirror_is_a_field``)
        self._mir_is_a_field = False
        # the mirror position of every field, in declaration order (see
        # ``get_field_mir_indices``), computed together with the mirror
        self._field_mir_indices: tuple[int | None, ...] | None = None

    @property
    def name_base(self) -> str:
        return self.head.name_base

    @property
    def modifiers(self) -> StructModifiers:
        return self.head.modifiers

    def get_method(self, name: str) -> Any | None:
        return self.head.methods.get(name)

    def fields(self) -> IndexedMap[str, StructField]:
        """The fields of this specialization, in declaration order: the
        declared fields with the head's generic type parameters replaced by
        this specialization's arguments.  Computed once and cached."""
        if self._fields is None:
            reps = {k: v for k, v in zip(self.head.generic_args, self.generic_args)}
            self._fields = self.head.fields.map(lambda f: StructField(f.name, replace_type_vars_type(f.type, reps)))
        return self._fields

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """Structs do not have subtypes yet: a struct type is a subtype of
        itself and of nothing else (in particular, not of another struct)."""
        return self is other

    def field_index(self, name: str) -> int | None:
        """The declaration index of the field ``name``, or None when the
        struct has no such field."""
        return self.fields().by_key.get(name)

    def field_type(self, name: str) -> Type | None:
        """The spy type of the field ``name``."""
        index = self.field_index(name)
        return None if index is None else self.fields().get_by_id(index).type

    def __repr__(self) -> str:
        return f'<spy struct {self}>'

    def __str__(self) -> str:
        return self.name_base

    @override
    def get_type(self) -> Type:
        level = 0
        for field in self.fields().values():
            child = field.type.get_type()
            assert isinstance(child, TypeType)
            level = max(level, child.level)
        return TypeType(level)

    @override
    def get_unit_value(self) -> AnyValue | None:
        values: list[AnyValue] = []
        for field in self.fields().values():
            val = field.type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

    @override
    def is_copyable(self) -> bool:
        """A struct is copyable when its ``copyable`` modifier says so;
        ``inherit`` (the default) means its fields all are."""
        match self.modifiers.copyable:
            case 'inherit':
                return all(f.type.is_copyable() for f in self.fields().values())
            case copyable:
                return copyable

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return tuple(f.type for f in self.fields().values())

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return self.get_mir_type()

    def _calculate_mir(self) -> None:
        if self._mir is not None:
            return
        assert self._field_mir_indices is None

        # the fields that occupy storage, each with the mirror of its type:
        # a zero-sized field occupies none and has no mirror position
        fields = self.fields().values()
        mirrored: list[tuple[int, StructField, mir.Type]] = []
        for index, field in enumerate(fields):
            field_mir = field.type.to_mir_type()
            if field_mir is None:
                # a field has to have a runtime representation: a struct
                # whose field has none (a compile-time-only type, such as the
                # type of an untyped literal) has no layout
                raise CompileError(
                    f"field '{field.name}' of {self} has type {field.type}, "
                    f"which has no runtime representation"
                )
            if not isinstance(field_mir, mir.VoidType):
                mirrored.append((index, field, field_mir))

        # an ``extern_c`` struct is laid out for the C ABI: the mirror holds
        # its fields in declaration order.  A spy struct is laid out by the
        # compiler, which is free to reorder: the least-aligned fields come
        # first (the mirror then packs tighter), a non-``extern_c`` struct
        # that holds exactly one field *is* that field - its mirror is the
        # field's own mirror, with no wrapper struct - and one that holds
        # none is a zero-sized type, mirroring to the void type
        if not self.modifiers.extern_c:
            mirrored.sort(key=lambda field: estimated_alignment_of(field[1].type))

        indices: list[int | None] = [None] * len(fields)
        for position, (index, _, _) in enumerate(mirrored):
            indices[index] = position
        self._field_mir_indices = tuple(indices)

        if len(mirrored) == 0:
            self._mir = mir.VoidType()
        elif len(mirrored) == 1 and not self.modifiers.extern_c:
            self._mir = mirrored[0][2]
            self._mir_is_a_field = True
        else:
            self._mir = mir.StructType(
                self.name_base,
                tuple(
                    mir.FormalArg(field.name, field_mir)
                    for _, field, field_mir in mirrored
                ),
            )

    def get_mir_type(self) -> mir.MayBeVoidType:
        """The (cached) MIR mirror of the struct: the one ``mir`` type every
        value of the struct mirrors to (created lazily, shared by all users),
        with the zero-sized fields dropped.  An ``extern_c`` struct mirrors
        to a ``mir.StructType`` of its declaration order; a spy struct orders
        the fields by alignment instead, mirrors to the type of its own field
        when it holds exactly one, and to the void type when it holds none."""
        self._calculate_mir()
        assert self._mir is not None
        return self._mir

    def get_field_mir_indices(self) -> tuple[int | None, ...]:
        """The mirror position of every field, in declaration order: the
        i-th entry is the position of the i-th field in the mirror returned
        by :meth:`get_mir_type` - a zero-sized field occupies no position
        and maps to ``None``.  A mirror that is the type of the struct's own
        field (see :meth:`_calculate_mir`) has that field at position 0, and
        the field sits at the address of the value itself."""
        self._calculate_mir()
        assert self._field_mir_indices is not None
        return self._field_mir_indices

    def mirror_is_a_field(self) -> bool:
        """Whether the MIR mirror of this struct is the mirror of the
        struct's own single stored field rather than a wrapper struct (see
        :meth:`_calculate_mir`): that field then sits at the address of the
        value itself, so taking its address needs no field indirection.
        Note that the field's mirror may itself be a ``mir.StructType`` -
        the mirror of a *wrapper* struct and the mirror that *is* the field
        cannot be told apart by that type alone."""
        self._calculate_mir()
        return self._mir_is_a_field

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)


# ---------------------------------------------------------------------------
# function values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnyFunction(Type):
    """The type of a function value whose signature is not known: a
    lazily compiled ``@func()`` function is only typed when a call
    specializes it.  It has no MIR mirror - such a value never crosses
    into runtime code."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self) -> mir.MayBeVoidType | None:
        return None

    def __str__(self) -> str:
        return "anyfn"

@dataclass(frozen=True, slots=True)
class BoundMethod(Value):
    """A method of one struct *specialization*: the function value of the
    method together with the type-argument values of the struct it was
    resolved on (the struct's generic type parameters -> the values of the
    specialization).  A method call resolves through the static type of the
    base - ``a.foo(b)`` behaves like ``typeof(a).foo(a, b)`` - so the method
    value has to carry those values: the method's signature names the
    struct's type parameters (its ``self`` is the struct template), and a
    call substitutes them into it (see ``interp``).

    The field is a ``FunctionValue`` in practice; it stays an
    :class:`AnyValue` because ``sval`` cannot depend on ``fn``."""

    fn: AnyValue
    generic_var_values: frozendict[TypeVar, Value]

    @override
    def get_type(self) -> Type:
        return AnyFunction()

    def __str__(self) -> str:
        return f'bound_method({self.fn})'


def int_range(type: IntType) -> tuple[int, int]:
    if type.signed:
        return (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
    return (0, 2 ** type.bits - 1)

def min_int_type(lower: int, upper: int) -> IntType:
    """The smallest integer type whose range contains ``[lower, upper]``."""
    if lower < 0:
        # signed: it needs ``-2**(bits-1) <= lower`` and
        # ``upper <= 2**(bits-1) - 1``
        need = max(-lower, upper + 1, 1)
        return IntType((need - 1).bit_length() + 1, True)
    return IntType(max(upper.bit_length(), 1), False)

# ---------------------------------------------------------------------------
# the return convention of a type: whether a function returning it returns a
# value, or writes the result into a caller-provided result location
# ---------------------------------------------------------------------------

# an aggregate of at most this many bytes is returned by value by default;
# larger ones are returned through a result pointer (the limit matches the
# size that the C ABIs of the supported targets pass in registers)
_AGGREGATE_VALUE_RETURN_LIMIT = 16

_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)
"""The size (and alignment) of a pointer of the host: the addresses the
compiled code works with are the host's (see ``lower``)."""

def estimated_size_of(type: Type) -> int:
    """Returns the estimated size of a type in bytes. The size is obtained
    using ctypes size rule, but is not guaranteed to be the actual size of
    the type. A zero-sized type - ``void``, a zero-bit integer, a struct
    that holds no storage - returns 0; literal types have no layout and
    raise :class:`SpyError`."""
    match type:
        case VoidType():
            return 0
        case BoolType():
            return 1
        case IntType():
            return (type.bits + 7) // 8
        case FloatType():
            return (type.bits + 7) // 8
        case PointerType():
            return _POINTER_BYTES
        case ArrayType():
            # an array is as big as its elements together, and holds no
            # storage at all when it has no element
            length = type.length_int
            if length is None:
                raise SpyError(f"array {type} has no constant length")
            if length == 0 or type.elem.is_zst():
                return 0
            return length * estimated_size_of(type.elem)
        case OptionType():
            # an option is laid out like its MIR mirror (see
            # ``OptionType.to_mir_type``): a ``bool`` when the child is
            # zero-sized, the child itself when it holds a pointer, and a
            # struct of the tag and the value otherwise
            child = type.child
            if child.is_zst():
                return 1
            if find_first_pointer_type_pos(child) is not None:
                return estimated_size_of(child)
            align = estimated_alignment_of(child)
            offset = (1 + align - 1) // align * align
            size = offset + estimated_size_of(child)
            return (size + align - 1) // align * align
        case StructType():
            offset = 0
            for field in type.fields().values():
                # a zero-sized field occupies no storage: it has no size and
                # imposes no alignment requirement on what follows it
                align = estimated_alignment_of(field.type)
                offset = (offset + align - 1) // align * align
                offset += estimated_size_of(field.type)
            align = estimated_alignment_of(type)
            return (offset + align - 1) // align * align
        case _:
            raise SpyError(f"type {type} has no layout")

def estimated_alignment_of(type: Type) -> int:
    """Estimated alignment of a type in bytes. Like :func:`estimated_size_of`,
    this is not guaranteed to be the actual alignment of the type. A
    zero-sized type - ``void``, a zero-bit integer, a struct that holds no
    storage - has alignment 1; literal types have no layout and raise
    :class:`SpyError`."""
    match type:
        case VoidType():
            return 1
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8 if type.bits != 0 else 1
        case FloatType():
            return type.bits // 8
        case PointerType():
            return _POINTER_BYTES
        case ArrayType():
            length = type.length_int
            if length is None:
                raise SpyError(f"array {type} has no constant length")
            if length == 0 or type.elem.is_zst():
                return 1
            return estimated_alignment_of(type.elem)
        case OptionType():
            # the tag of a zero-sized child, and a tag next to the value,
            # impose no alignment of their own; the rest follows the child
            if type.child.is_zst():
                return 1
            return estimated_alignment_of(type.child)
        case StructType():
            return max(
                (
                    estimated_alignment_of(f.type)
                    for f in type.fields().values()
                    if not f.type.is_zst()
                ),
                default=1,
            )
        case _:
            raise SpyError(f"type {type} has no layout")

def _mentions_type_var(type: Type) -> bool:
    """Whether ``type`` still names a type parameter somewhere inside it,
    so that its layout - and with it its calling convention - is not known
    until a call substitutes the parameter."""
    todo: list[Type] = [type]
    while todo:
        current = todo.pop()
        if isinstance(current, TypeVar):
            return True
        todo.extend(current.get_type_children())
    return False


def returns_via_result_ptr(type: Type) -> bool:
    """Whether a function returning ``type`` delivers its result by
    writing into a caller-provided result location (a hidden result
    pointer parameter) instead of returning the value directly.

    This is the default policy, a property of the *return type*: an
    aggregate is returned by value while it is small (up to
    :data:`_AGGREGATE_VALUE_RETURN_LIMIT` bytes) and through a result
    pointer once it outgrows it, and a new aggregate kind (arrays) only
    needs to extend this function.  Scalars are always returned by
    value.  A signature may override the default
    (``fn.Signature.ret_spec``)."""
    match type:
        case StructType() | ArrayType() | OptionType():
            if _mentions_type_var(type):
                # the layout is not known until the call substitutes the
                # type parameter: assumed small now, re-decided on substitution
                return False
            return estimated_size_of(type) > _AGGREGATE_VALUE_RETURN_LIMIT
        case _:
            return False


@dataclass(frozen=True, slots=True)
class RetValue:
    """One *leaf* result value of a function: the spy type of the value and
    whether it is delivered through a hidden result pointer rather than
    returned by value.  The type is always a runtime type - a ``tuple[...]``
    annotation nests a :class:`RetTuple` instead."""

    type: Type
    via_result_ptr: bool


@dataclass(frozen=True, slots=True)
class RetTuple:
    """A group of results: the ``tuple[...]`` a function returns (or one
    written as an element of that annotation).  The whole return type is one
    :class:`RetSpec` - a :class:`RetValue` for a single value, a
    :class:`RetTuple` when it is a ``tuple[...]`` - so a single result is just
    the trivial case and the nesting mirrors the annotation.

    The group is a *compile-time* regrouping of the values of its elements: a
    tuple has no runtime representation of its own, so the lowered signature
    delivers its leaves (see :func:`make_ret_spec`) and the caller regroups
    them (see ``interp``)."""

    # the ``tuple[...]`` the group was written as (its ``types`` are the
    # element types, one per entry of ``values``)
    type: TupleType
    values: tuple[RetSpec, ...]


type RetSpec = RetValue | RetTuple


def iter_ret_leaves(spec: RetSpec) -> Iterator[RetValue]:
    """The leaf values of a return spec, in declaration order (depth first)."""
    work: list[RetSpec] = [spec]
    while work:
        node = work.pop()
        match node:
            case RetValue():
                yield node
            case RetTuple(values=values):
                work.extend(reversed(values))


def make_ret_spec(type: Type) -> RetSpec:
    """The return convention of a function whose return annotation is the spy
    type ``type`` (a ``tuple[...]`` for several values, nested at whatever
    depth it is written): the annotation as one :class:`RetSpec` tree.  A
    ``tuple[...]`` - at the top level or nested - becomes a :class:`RetTuple`
    of its elements, and every other type a :class:`RetValue` leaf.

    At most one leaf is returned *by value* - the first one that fits in
    registers (``returns_via_result_ptr`` says so) - and every other leaf is
    delivered by writing through a hidden result pointer; when no leaf
    qualifies the function returns void.  A zero-sized leaf has no value to
    return: it is delivered as its unit value and never takes the by-value
    slot."""
    # the leaf types, in declaration order (depth first)
    leaves: list[Type] = []
    work: list[Type] = [type]
    while work:
        item = work.pop()
        match item:
            case TupleType():
                if item.has_ellipsis:
                    raise CompileError(
                        'a varying number of return values has no fixed shape'
                    )
                work.extend(reversed(item.types))
            case _:
                leaves.append(item)
    chosen: int | None = None
    for index, leaf_type in enumerate(leaves):
        if leaf_type.get_unit_value() is not None:
            # a zero-sized value is delivered as its unit value, not
            # through the by-value slot
            continue
        if not returns_via_result_ptr(leaf_type):
            chosen = index
            break
    via = tuple(
        leaf_type.get_unit_value() is None and index != chosen
        for index, leaf_type in enumerate(leaves)
    )
    # rebuild the tree of the annotation, marking every leaf with how it is
    # delivered (a ``None`` on the work stack closes the group it opened; the
    # whole type is itself a group when it is a ``tuple[...]``)
    index = 0
    groups: list[list[RetSpec]] = [[]]
    group_types: list[TupleType] = []
    if isinstance(type, TupleType):
        build_work: list[Type | None] = list(reversed(type.types))
    else:
        build_work = [type]
    while build_work:
        item = build_work.pop()
        if item is None:
            group_type = group_types.pop()
            values = tuple(groups.pop())
            groups[-1].append(RetTuple(group_type, values))
            continue
        match item:
            case TupleType():
                build_work.append(None)
                build_work.extend(reversed(item.types))
                groups.append([])
                group_types.append(item)
            case _:
                groups[-1].append(RetValue(item, via[index]))
                index += 1
    assert index == len(leaves) and len(groups) == 1
    if isinstance(type, TupleType):
        return RetTuple(type, tuple(groups[0]))
    assert len(groups[0]) == 1
    return groups[0][0]


def pass_by_ref(type: Type) -> bool:
    """Whether a parameter of spy type ``type`` is passed by reference
    (as a const pointer) rather than by value: the calling convention of
    one argument, decided by the compiler.

    The policy mirrors :func:`returns_via_result_ptr`: an aggregate too
    large to be passed in registers (larger than the by-value limit) is
    passed as a pointer, everything else by value.  The caller otherwise
    passes a parameter by reference when its formal declares it as one
    (``fn.SignatureFormalArg.by_ref``), whatever its type."""
    match type:
        case StructType() | ArrayType() | OptionType():
            return estimated_size_of(type) > _AGGREGATE_VALUE_RETURN_LIMIT
        case _:
            return False


# ---------------------------------------------------------------------------
# mapping Python values to spy types
# ---------------------------------------------------------------------------


def type_of(value: AnyValue, int_literal_bits: int | None = None) -> Type:
    """The spy type a Python *value* is marshaled to at the call boundary,
    or ``None`` for a plain Python object that has no marshaling (a
    tuple, a class, ...).  A compile-time object is an ``sval.Value`` and
    reports its own spy type.
    """
    if isinstance(value, Value):
        return value.get_type()
    match value:
        case bool():
            return BoolType()
        case int():
            return AnyIntType() if int_literal_bits is None else IntType(int_literal_bits, True)
        case float():
            return FloatType(64)
        case str():
            return PointerType(IntType(8, False), True)

class AsSpyValue:
    @abstractmethod
    def as_spy_value(self) -> AnyValue:
        ...

class StructDecl(AsSpyValue):
    """A Python-level object that declares a spy struct: the handle a
    ``@struct()`` class binds to (``dsl._RegisteredClass``).  Its spy value
    is the struct it declares.  The parser tells a construction from an
    ordinary call by the *type* of the callee object (see ``astgen``),
    because asking a function handle for its spy value parses the function
    body - which may reenter the parser (a recursive function)."""

@dataclass(frozen=True, slots=True)
class StructTypeApplication:
    """A struct template applied to generic arguments whose spy value is not
    known yet: ``Foo[T]`` written in an annotation evaluates to this at the
    Python level (see ``dsl._RegisteredClass.__getitem__``), because Python
    evaluates the annotation in the annotation scope of the annotated
    function or class - the arguments name the type parameters of that
    scope, which ``__getitem__`` does not see.  :func:`as_value` turns the
    application into the struct specialization once it is given that scope
    (its ``type_vars``).

    Not a :class:`Value`: it is a transient Python-level object that never
    denotes a value of the spy domain."""

    struct: StructTypeHead
    generic_vars: tuple[Any, ...]

def as_value(value: Any, type_vars: dict[typing.TypeVar, Value] | None = None, resolver: GlobalResolver | None = None) -> AnyValue:
    """The spy-domain value of a Python compile-time object: Python
    scalars and ``sval.Value`` objects pass through, and ``None`` is the
    ``Null`` value (the absent value of an option, the unit value of the
    zero-sized ``NullType``).  Class objects of the scalar types map to their
    default spy types, and any object that knows its own spy value
    (``as_spy_value``, the protocol of a struct class, see
    ``dsl._RegisteredClass``) is asked for it."""
    if isinstance(value, (Value, int, float, str, bool)):
        return value
    if value is None:
        # ``None`` denotes the null value: the absent value of an option
        return Null()
    if value is int:
        return AnyIntType()
    if value is float:
        return FloatType(64)
    if value is str:
        return PointerType(IntType(8, False), is_const=True)
    if value is bool:
        return BoolType()
    if isinstance(value, typing.TypeVar):
        if type_vars is None or value not in type_vars:
            raise TypeError(f'cannot convert {value} to a value')
        return type_vars[value]
    if isinstance(value, StructTypeApplication):
        # ``Foo[T]``: resolve its arguments in the scope it was written in,
        # then specialize the template for them
        resolved: list[Value] = []
        for arg in value.generic_vars:
            arg_value = as_value(arg, type_vars, resolver)
            if not isinstance(arg_value, Value):
                raise TypeError(
                    f'cannot use {arg!r} as a generic argument of {value.struct}'
                )
            resolved.append(arg_value)
        return value.struct.specialize(tuple(resolved))
    if isinstance(value, AsSpyValue):
        return value.as_spy_value()
    if typing.get_origin(value) is tuple:
        # ``tuple[T1, T2, ...]``: the return annotation of a function that
        # returns several values.  ``tuple[T, ...]`` is the variable-length
        # form Python allows; it names no fixed set of results, so it is
        # kept as-is (a ``TupleType`` with ``has_ellipsis``) and rejected by
        # the signatures that cannot use it
        raw = typing.get_args(value)
        has_ellipsis = len(raw) > 0 and raw[-1] is Ellipsis
        elems = raw[:-1] if has_ellipsis else raw
        types: list[Type] = []
        for arg in elems:
            type = as_value(arg, type_vars, resolver)
            if not isinstance(type, Type):
                raise TypeError(f'{arg!r} is not a type')
            types.append(type)
        return TupleType(tuple(types), has_ellipsis)
    if typing.get_origin(value) is typing.Literal:
        # ``Literal[X]`` denotes the value ``X``: the default of a type
        # parameter (the constness of a pointer, ``C: bool = Literal[False]``)
        # evaluates to one
        args = typing.get_args(value)
        if len(args) == 1 and isinstance(args[0], (bool, int, str)):
            return args[0]
        raise TypeError(f'cannot convert {value!r} to a value')
    if typing.get_origin(value) is syntax.Ptr:
        # ``Ptr[T]``/``Ptr[T, C]``: C's pointer type.  The second argument is
        # the constness of the pointer (see ``_as_constness``); Python
        # inserts the default declared by the class when it is left out, so
        # a ``Ptr[T]`` annotation arrives with a ``Literal[False]``
        args = typing.get_args(value)
        if len(args) not in (1, 2):
            raise TypeError(f'cannot convert {value!r} to a value')
        elem = as_value(args[0], type_vars, resolver)
        if not isinstance(elem, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        if len(args) == 2:
            return PointerType(elem, _as_constness(args[1], type_vars, resolver))
        return PointerType(elem, False)
    if typing.get_origin(value) is syntax.Array:
        # ``Array[T, L]``: ``L`` values of type ``T``.  The length is a *value*
        # (a Python ``int``, or the type parameter it is written as); the
        # constructor ``array(...)`` takes it from the number of elements it is
        # given, which the Python type system cannot express, so an annotation
        # that names the element type alone has to write the length out
        args = typing.get_args(value)
        if len(args) != 2:
            raise TypeError(f'cannot convert {value!r} to a value')
        elem = as_value(args[0], type_vars, resolver)
        if not isinstance(elem, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        return ArrayType(elem, as_value(args[1], type_vars, resolver))
    if typing.get_origin(value) is syntax.Option:
        # ``Option[T]``: ``T`` or the ``Null`` value.  The alias
        # ``type Option[T] = T | None`` evaluates a subscripted use to a
        # specialization of the alias, whose origin is the alias itself
        args = typing.get_args(value)
        if len(args) != 1:
            raise TypeError(f'cannot convert {value!r} to a value')
        child = as_value(args[0], type_vars, resolver)
        if not isinstance(child, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        return OptionType(child)
    if typing.get_origin(value) is typing.Union:
        # ``T | None``: the same as ``Option[T]`` (the alias is defined that
        # way), written out directly.  ``None`` in the union is the null
        # value, never a type of its own
        args = typing.get_args(value)
        rest = tuple(a for a in args if a is not NoneType)
        if len(rest) != len(args) - 1 or len(rest) != 1:
            raise TypeError(f'cannot convert {value!r} to a value')
        child = as_value(rest[0], type_vars, resolver)
        if not isinstance(child, Type):
            raise TypeError(f'{rest[0]!r} is not a type')
        return OptionType(child)

    raise TypeError(f'cannot convert {value} to a value')

def unwrap_comptime(annotation: Any) -> tuple[bool, Any]:
    """Split an *evaluated* annotation into its ``Comptime`` marker and the
    type it wraps: ``(True, None)`` for the bare ``Comptime`` (a compile-time
    variable whose type its value determines), ``(True, T)`` for
    ``Comptime[T]`` (a compile-time variable of the declared type ``T``) and
    ``(False, annotation)`` for any other annotation (an ordinary declared
    type).  ``None`` (no annotation written) splits to ``(False, None)``.

    ``syntax.Comptime`` is a PEP 695 type alias (``type Comptime[T] = T``),
    which Python keeps on the evaluated annotation: the bare alias is the
    marker itself and ``Comptime[T]`` a generic alias whose origin is it (see
    :func:`as_value`, which converts the type it wraps)."""
    if annotation is None:
        return False, None
    if annotation is syntax.Comptime:
        return True, None
    if typing.get_origin(annotation) is syntax.Comptime:
        args = typing.get_args(annotation)
        if len(args) != 1:
            raise TypeError(f'Comptime takes exactly one type argument, got {annotation!r}')
        return True, args[0]
    return False, annotation

def _as_constness(value: Any, type_vars: dict[typing.TypeVar, Value] | None, resolver: GlobalResolver | None) -> AnyValue:
    """The spy value of the constness argument of a pointer type: a Python
    ``bool``, or the type parameter it is written as, which a call then
    solves to one of the two (see ``TypeVarSolver``)."""
    result = as_value(value, type_vars, resolver)
    if isinstance(result, (bool, Value)):
        return result
    raise TypeError(f'cannot use {value!r} as the constness of a pointer')

def negate(value: AnyValue) -> AnyValue | None:
    if isinstance(value, (int, float)):
        return -value
    return None

@dataclass(frozen=True)
class _Constraint:
    lhs: AnyValue
    rhs: AnyValue
    is_subtype: bool = False  # True when lhs is a subtype of rhs

class _SolvedTypeVar:
    """The solver state of one type parameter.

    A parameter is either *solved* - bound to a value (``_value``), with
    no subtype bounds recorded - or *bounded*: every value in
    ``_subtypes`` is a subtype of it (its lower bounds), without a
    solution yet.  Solving the equality constraints of a bounded
    parameter binds it to the peer type of its recorded bounds."""

    def __init__(self) -> None:
        self._value: AnyValue | None = None  # non-None: this type var is solved to this value, in this case _subtypes is None
        self._subtypes: set[Type] | None = None  # non-None: all values in this set are subtypes of this type var, in this case _value is None

    def _add_bound(self, value: Type) -> None:
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
    constrained type parameter to a value.  Spy types have structural
    subtyping only where the type defines it - integers by range, floats
    by width, types by level; elsewhere a subtype is equal to its
    supertype.  The solver tracks the bounds of a parameter separately
    so that ``finish`` can bind a parameter that only ever appears on the
    right of subtype constraints (as the supertype of its bounds).  A
    constraint that cannot be satisfied is recorded in ``_unsatisfied``
    (and is not reported yet); a generic parameter that stays unsolved
    makes ``fn.Signature.solve_param_types`` raise
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

    def _add_unsatisfied(self, lhs: AnyValue, rhs: AnyValue, is_subtype: bool = False) -> None:
        self._unsatisfied.append(_Constraint(lhs, rhs, is_subtype))

    def _solve_type_var_bound(self, v: TypeVar, bound: AnyValue, is_subtype: bool) -> None:
        solved = self._solved(v)
        if is_subtype:
            assert solved._value is None
            assert isinstance(bound, Type)
            if solved._subtypes is None:
                solved._subtypes = set()
            solved._subtypes.add(bound)
        else:
            if solved._subtypes is not None:
                for st in solved._subtypes:
                    if isinstance(bound, Type) and not st.is_subtype_of(bound):
                        self._add_unsatisfied(st, bound, True)
                solved._subtypes = None
            solved._value = bound

    def substitute_solved(self, value: AnyValue) -> AnyValue:
        while True:
            if not isinstance(value, TypeVar):
                return value
            solved = self._solved(value)
            if solved is None or solved._value is None:
                return value
            value = solved._value

    def add_constraint(self, lhs: AnyValue, rhs: AnyValue, is_subtype: bool = False):
        todo = [(lhs, rhs, is_subtype)]
        while todo:
            lhs, rhs, is_subtype = todo.pop()
            lhs = self.substitute_solved(lhs)
            rhs = self.substitute_solved(rhs)

            if lhs == rhs:
                continue

            if not is_subtype and isinstance(rhs, TypeVar) and not isinstance(lhs, TypeVar):
                t = lhs
                lhs = rhs
                rhs = t

            if isinstance(lhs, TypeVar):
                # in the case we concern, TypeVar cannot appear on the left side of a subtype constraint
                assert not is_subtype
                self._solve_type_var_bound(lhs, rhs, is_subtype)

            if isinstance(rhs, TypeVar):
                self._solve_type_var_bound(rhs, lhs, is_subtype)

            # fall back to equal constraint
            if isinstance(lhs, StructType) and isinstance(rhs, StructType) and lhs.head is rhs.head and len(lhs.generic_args) == len(rhs.generic_args):
                todo.extend((l, r, False) for l, r in zip(reversed(lhs.generic_args), reversed(rhs.generic_args)))
            # an option constrains its child type: two options constrain their
            # children with each other, and an option against a plain type
            # constrains its child with that type (a ``T`` and a ``Null`` both
            # convert to ``Option[T]``)
            if isinstance(lhs, OptionType) and isinstance(rhs, OptionType):
                todo.append((lhs.child, rhs.child, False))
            elif isinstance(lhs, OptionType) and not isinstance(rhs, (NullType, TypeVar)):
                # the child is constrained on the right: a subtype constraint
                # may only put a type parameter on its right side
                todo.append((rhs, lhs.child, is_subtype))
            elif isinstance(rhs, OptionType) and not isinstance(lhs, (NullType, TypeVar)):
                todo.append((lhs, rhs.child, is_subtype))
            if isinstance(lhs, PointerType) and isinstance(rhs, PointerType):
                # a pointer constrains its pointee type, and - while it is
                # still a type parameter - the constness of the pointer
                todo.append((lhs.elem, rhs.elem, False))
                if isinstance(lhs.is_const, Value) or isinstance(rhs.is_const, Value):
                    todo.append((lhs.is_const, rhs.is_const, False))

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

    def get_solved(self) -> dict[TypeVar, AnyValue]:
        return {
            k: v._value
            for k, v in self._type_var_values.items()
            if v._value is not None
        }

def replace_type_var(value: AnyValue, reps: Mapping[TypeVar, AnyValue]) -> AnyValue:
    match value:
        case TypeVar():
            return reps.get(value, value)
        case PointerType():
            # the constness is substituted too: it may be a type parameter
            # (``Ptr[T, C]``), which a call solves to a ``bool``
            return PointerType(
                replace_type_vars_type(value.elem, reps),
                replace_type_var(value.is_const, reps),
            )
        case ArrayType():
            # the length is substituted too: it may be a type parameter
            # (``Array[T, N]``)
            return ArrayType(
                replace_type_vars_type(value.elem, reps),
                replace_type_var(value.length, reps),
            )
        case OptionType():
            return OptionType(replace_type_vars_type(value.child, reps))
        case TupleType():
            return TupleType(
                tuple(replace_type_vars_type(t, reps) for t in value.types),
                value.has_ellipsis,
            )
        case StructType():
            # a struct type carries its type arguments: substituting into it
            # rebuilds the specialization (and keeps the identity of the one
            # the head caches, so equal references stay equal).  The
            # arguments of a struct are always types
            args: list[Value] = []
            for arg in value.generic_args:
                substituted = replace_type_var(arg, reps)
                assert isinstance(substituted, Value)
                args.append(substituted)
            if all(new is old for new, old in zip(args, value.generic_args)):
                return value
            return value.head.specialize(tuple(args))
        case _:
            return value

def replace_type_vars_type(type: Type, reps: Mapping[TypeVar, AnyValue]) -> Type:
    ret = replace_type_var(type, reps)
    assert isinstance(ret, Type)
    return ret

def is_comptime_only_type(type: Type) -> bool:
    match type:
        case AnyIntType() | TypeType():
            return True
        case ArrayType():
            # an array of a compile-time-only type holds no value that could
            # live in memory, whatever its length
            return is_comptime_only_type(type.elem)
        case OptionType():
            # an option of a compile-time-only type has no representation
            # either: it mirrors to nothing (see ``OptionType.to_mir_type``)
            return is_comptime_only_type(type.child)
        case _:
            return False

def is_numeric_type(type: Type):
    match type:
        case AnyIntType() | IntType() | FloatType():
            return True
        case _:
            return False

def coerce_const(value: AnyValue, type: Type) -> AnyValue:
    """Turn a Python value into the typed spy value of the spy type
    ``type`` (an ``Int``/``Float``/``Void``/``Type``/``bool``); the
    interpreter builds the MIR constant from it later."""
    if isinstance(value, AsValue):
        value = value.value
    match type:
        case BoolType():
            if not isinstance(value, bool):
                raise CompileError(f"cannot use {value!r} as a bool constant")
            return value
        case IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompileError(f"cannot use {value!r} as an integer constant")
            if type.signed:
                lo, hi = (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
            else:
                lo, hi = (0, 2 ** type.bits - 1)
            if not lo <= value <= hi:
                raise CompileError(
                    f"integer constant {value} is out of range for {type}"
                )
            return Int(value, type)
        case FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CompileError(f"cannot use {value} as a float constant")
            return Float(float(value), type)
        case TypeType():
            if not isinstance(value, Type):
                raise CompileError(f"cannot use {value} as a type constant")
            if value.get_type() != type:
                raise CompileError(f"cannot use {value} as a type constant")
            return value
        case OptionType():
            # a value of an option is either the null value, or a value of the
            # child type (which the interpreter then tags as present)
            if isinstance(value, Null):
                return value
            return coerce_const(value, type.child)
        case VoidType():
            # the void type's unit value is what ``None`` used to be: a null
            # value converts to it (``NullType`` is a subtype of ``VoidType``)
            if isinstance(value, (Void, Null)):
                return Void()
            raise CompileError(f"cannot use {value!r} as a void constant")
        case _:
            raise CompileError(
                f"cannot create a constant of type {type} from {value}"
            )

class GlobalResolver:
    @abstractmethod
    def resolve_global(self, value: Any) -> AnyValue | None:
        """The spy value a global object referenced inside a function
        body resolves to.  A function registered in this host - reached
        as the raw function object or through the callable view its
        decorated name binds to - resolves to its function entry (created
        lazily when it is not parsed yet).  The host also resolves the
        ``spy.*`` builtins, the struct classes it declares and the plain
        Python functions it inlines; any other object is not a spy value
        of this host and returns ``None`` (the object stays a plain
        compile-time Python value)."""
        ...

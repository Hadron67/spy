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
from dataclasses import dataclass
from typing import Any, Literal, override

from spy.util import IdentityObj, IndexedMap, frozendict

from . import mir
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
    has no runtime representation: its ``mir`` mirror is the MIR void
    type (``to_mir_type`` returns ``mir.VOID``) and no load/store is
    ever emitted for it."""

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

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, (IntType, AnyIntType)):
            return other
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
    the type. A zero-sized type that has a layout - a zero-bit integer, a
    struct with no stored field - returns 0; ``void`` and literal types
    have no layout and raise :class:`SpyError`."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return (type.bits + 7) // 8
        case FloatType():
            return (type.bits + 7) // 8
        case PointerType():
            return _POINTER_BYTES
        case StructType():
            offset = 0
            for field in type.fields().values():
                # a zero-bit field has alignment 1 and occupies no size
                align = estimated_alignment_of(field.type)
                offset = (offset + align - 1) // align * align
                offset += estimated_size_of(field.type)
            align = estimated_alignment_of(type)
            return (offset + align - 1) // align * align
        case _:
            raise SpyError(f"type {type} has no layout")

def estimated_alignment_of(type: Type) -> int:
    """Estimated alignment of a type in bytes. Like :func:`estimated_size_of`,
    this is not guaranteed to be the actual alignment of the type. A zero-bit
    integer and a struct with no stored field have alignment 1; ``void`` and
    literal types have no layout and raise :class:`SpyError`."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8 if type.bits != 0 else 1
        case FloatType():
            return type.bits // 8
        case PointerType():
            return _POINTER_BYTES
        case StructType():
            return max(
                (
                    estimated_alignment_of(f.type)
                    for f in type.fields().values()
                    if f.type.get_unit_value() is None
                ),
                default=1,
            )
        case _:
            raise SpyError(f"type {type} has no layout")

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
    (``fn.Signature.ret_by_ref``)."""
    match type:
        case StructType():
            return estimated_size_of(type) > _AGGREGATE_VALUE_RETURN_LIMIT
        case _:
            return False


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
        case StructType():
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
    unit value of the zero-sized void type (``Void()``).  Class objects
    of the scalar types map to their default spy types, and any object
    that knows its own spy value (``as_spy_value``, the protocol of a
    struct class, see ``dsl._RegisteredClass``) is asked for it."""
    if isinstance(value, (Value, int, float, str, bool)):
        return value
    if value is None:
        # ``None`` denotes the void value: the unit value of the
        # zero-sized ``VoidType``
        return Void()
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
    no subtype bounds recorded - or *bounded*: every value in
    ``_subtypes`` is a subtype of it (its lower bounds), without a
    solution yet.  Solving the equality constraints of a bounded
    parameter binds it to the peer type of its recorded bounds."""

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

    def substitute_solved(self, value: Value):
        while True:
            if not isinstance(value, TypeVar):
                return value
            solved = self._solved(value)
            if solved is None or solved._value is None:
                return value
            value = solved._value

    def add_constraint(self, lhs: Value, rhs: Value, is_subtype: bool = False):
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
        case PointerType():
            type = replace_type_var(value.elem, reps)
            assert isinstance(type, Type)
            return PointerType(type, value.is_const)
        case StructType():
            # a struct type carries its type arguments: substituting into it
            # rebuilds the specialization (and keeps the identity of the one
            # the head caches, so equal references stay equal)
            args = tuple(replace_type_var(a, reps) for a in value.generic_args)
            if all(new is old for new, old in zip(args, value.generic_args)):
                return value
            return value.head.specialize(args)
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
        case VoidType():
            if isinstance(value, Void):
                return value
            raise CompileError(f"cannot use {value} as a void constant")
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

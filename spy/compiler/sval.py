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

import typing
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum, auto
from types import NoneType
from typing import Any, Literal, override

from . import mir, syntax
from .errors import CompileError
from .target import TargetInfo
from .util import FrozenArraySet, IdentityObj, IndexedMap, TriState, frozendict

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

type AnyValue = Value | int | float | bytes | bool

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

class SpecialTypeKind(IntEnum):
    """How a spy type maps onto runtime code (see ``Type.classify``).

    ``NONE`` - an ordinary type, with a MIR mirror of its own; ``COMPTIME`` -
    a type only compile-time values may have (it has no mirror); ``ZST`` -
    a zero-sized (unit) type, which has no storage and no mirror of its own;
    ``DST`` - a dynamically-sized type (a function type), whose mirror only a
    pointer to it may use: a value of one cannot be allocated, loaded or
    returned."""

    NONE = auto()
    COMPTIME = auto()
    ZST = auto()
    DST = auto()


class MirLowerCache:
    """The MIR mirrors a lowering host creates for the spy types that must be
    *interned*: a ``mir.StructType``/``mir.UnionType`` is an identity object
    and a module declares one LLVM struct/union per MIR type, so two equal spy
    ``Option[T]``/unions have to lower to one MIR type.  One cache belongs to
    one host context (``dsl._Context``) and, with it, to one *target*: the
    target's pointer size decides the layout a mirror is built from (the order
    of a struct's fields, the variant a union holds), so the types of one
    target are not the types of another."""

    def __init__(self, target: TargetInfo) -> None:
        self.target = target
        self._union_mirs: dict[UnionType, mir.UnionType] = {}
        self._option_struct_mirs: dict[Type, mir.StructType] = {}
        self._tagged_union_mirs: dict[TaggedUnionType, mir.StructType] = {}

    def union_mir(self, type: UnionType, payload: mir.Type) -> mir.UnionType:
        """The (interned) MIR mirror of the payload union ``type``."""
        ret = self._union_mirs.get(type)
        if ret is None:
            ret = mir.UnionType('union', payload)
            self._union_mirs[type] = ret
        return ret

    def option_struct_mir(self, child: Type, child_mir: mir.Type | None) -> mir.StructType:
        """The (interned) MIR mirror of the ``Option[T]`` whose ``T``
        (``child``) has no pointer to be tagged on: a struct of a ``bool`` tag
        and the value.  A dynamically-sized child has no mirror of its own: it
        is the option's flexible member, placed last (the option itself is then
        dynamically sized)."""
        ret = self._option_struct_mirs.get(child)
        if ret is None:
            if child.classify() == SpecialTypeKind.DST:
                ret = mir.StructType(
                    'option',
                    (mir.FormalArg('tag', mir.BoolType()),),
                    child.fam_mir_type(self),
                )
            else:
                assert child_mir is not None
                ret = mir.StructType(
                    'option',
                    (
                        mir.FormalArg('tag', mir.BoolType()),
                        mir.FormalArg('value', child_mir),
                    ),
                )
            self._option_struct_mirs[child] = ret
        return ret

    def tagged_union_mir(self, type: TaggedUnionType, tag: mir.Type, payload: mir.Type) -> mir.StructType:
        """The (interned) MIR struct of the tagged union ``type`` whose payload
        holds storage: a tag and the (untagged) payload union."""
        ret = self._tagged_union_mirs.get(type)
        if ret is None:
            ret = mir.StructType('tagged_union', (
                mir.FormalArg('tag', tag),
                mir.FormalArg('payload', payload),
            ))
            self._tagged_union_mirs[type] = ret
        return ret


class SpecialTypes:
    """The spy types the type rules need but the type system cannot build on
    its own, because they live in ``std`` (a host library): the ``slice`` struct
    a subscript of a multi pointer builds, and the ``SlicePtr``/``ConstSlicePtr``
    structs a slice of a pointer is.  One belongs to each host context (see
    :meth:`CompileContext.special_types`) and is handed to every type function
    that has to know them."""

    def __init__(
        self,
        slice_type: StructTypeHead,
        slice_ptr_type: StructTypeHead,
        const_slice_ptr_type: StructTypeHead,
    ) -> None:
        self.slice_type = slice_type
        self.slice_ptr_type = slice_ptr_type
        self.const_slice_ptr_type = const_slice_ptr_type

    def slice_ptr_of(self, elem: Type, is_const: AnyValue) -> StructType:
        """The slice of a pointer to the element type ``elem``: ``SlicePtr``
        when the pointer is mutable and ``ConstSlicePtr`` when it is const (the
        constness of the slice is its *type*, not a type argument)."""
        head = self.const_slice_ptr_type if is_const is True else self.slice_ptr_type
        return head.specialize((elem,))


class Type(Value):
    def get_unit_value(self) -> AnyValue | None:
        """The canonical *unit value* of a zero-sized type (ZST): ``None``
        when the type has a runtime representation (it is not
        zero-sized), otherwise the one compile-time value every value of
        the type equals - ``Void()`` for the void type, ``Int(0, T)`` for
        a zero-bit integer, an ``AggregateValue`` for a struct whose
        fields are all ZSTs.  A ZST has no runtime representation at all:
        it has no mirror of its own (``to_mir_type`` returns ``None``)."""
        return None

    def classify(self) -> SpecialTypeKind:
        """How a value of this type maps onto runtime code (see
        :class:`SpecialTypeKind`).  The classification is computed from the
        type alone - it never asks for the MIR mirror - so it is what a
        caller uses to decide whether a mirror can be asked for at all.

        The default is :attr:`SpecialTypeKind.NONE`; the types that have no
        mirror, no storage or no size override it."""
        return SpecialTypeKind.NONE

    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, self.__class__)

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, UndefinedType):
            # the undefined type is the bottom: a value of it is a value of any
            # type, so the other type wins
            return self
        if isinstance(other, NullType):
            # a type and the null value peer to the option of the type
            return OptionType(self)
        if isinstance(other, OptionType):
            return _resolve_option_peer(self, other)
        if isinstance(other, TaggedUnionType):
            # a tagged union on the other side knows which of its variants this
            # type is: a value of one of them peers with the union itself
            return other.resolve_peer_type(self)
        return other if self.is_subtype_of(other) else None

    @abstractmethod
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        """The MIR mirror of this spy type: the static type the runtime
        register of a value of this type has, or ``None`` when there is no
        such type - a zero-sized type (which holds no storage), a
        compile-time-only type, and a struct template all have no mirror.
        A spy struct type mirrors to the one MIR type every value of the
        struct shares (created lazily and cached on the descriptor, see
        :meth:`StructType._calculate_mir`); a function that returns a
        zero-sized type returns no value at all.

        ``cache`` interns the mirrors that have to be one object per
        lowering host (see :class:`MirLowerCache`); most types ignore it."""
        ...

    def is_zst(self) -> bool:
        return self.classify() == SpecialTypeKind.ZST

    def is_copyable(self) -> bool:
        return True

    def get_type_children(self) -> tuple[Type, ...]:
        return ()

    def fam_mir_type(self, cache: MirLowerCache) -> mir.MayBeVoidType:
        """The MIR mirror this *dynamically-sized* type contributes as the
        flexible member (the FAM) of the aggregate that holds it - the element
        type of an unsized array, ``VOID`` for an opaque type, the mirror of a
        dynamically-sized struct.  Only a type whose ``classify`` is
        :attr:`SpecialTypeKind.DST` has one."""
        raise CompileError(f'{self} is not a dynamically-sized type')

    def contains(self, needle: Type):
        todo: list[Type] = [self]
        while todo:
            current = todo.pop()
            if current is needle:
                return True
            todo.extend(reversed(current.get_type_children()))
        return False

    def __or__(self, other: Any) -> TaggedUnionApplication:
        """``A | B`` written in an annotation: a tagged-union application, which
        :func:`as_value` turns into a :class:`TaggedUnionType` once the scope it
        was written in is known."""
        return union_application(self, other)

    def __ror__(self, other: Any) -> TaggedUnionApplication:
        return union_application(other, self)

@dataclass(frozen=True)
class TypeType(Type):
    """The type of every spy type: the values of this type are the compile-time
    types themselves (``spy.typeof(a) == spy.i32``), so it is what the ``type``
    annotation of a parameter that takes a type stands for."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'type'

@dataclass(frozen=True)
class AnyType(Type):
    """The type of an arbitrary compile-time value (the ``Any`` annotation):
    it has no runtime representation and takes any value as it is, so it is what
    a reflective field that names a value of no fixed type - a struct field's
    default value - is declared with."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'any'

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
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
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
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return f'tuple[{", ".join(str(t) for t in self.types)}{", ..." if self.has_ellipsis else ""}]'


@dataclass(frozen=True)
class UnionType(Type):
    """The payload of an error union: an untagged union that holds exactly one
    of ``types`` (the exception structs), the largest one occupying the
    storage.  It has no tag of its own - the error code next to it *is* the
    tag - so a variant value is written and read through a reinterpretation of
    the payload's address (see ``mir.UnionType``)."""

    types: frozenset[Type]

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return tuple(self.types)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """A union is a subtype of a union of the same variants and of
        nothing else (the variants are compared for equality, like the
        elements of an array)."""
        return isinstance(other, UnionType) and self == other

    @override
    def get_unit_value(self) -> AnyValue | None:
        """A union holds no storage when it has no variant, or when every
        variant is zero-sized: it then equals the unit value, which carries no
        variant at all (see :class:`UnionValue`)."""
        for type in self.types:
            if type.get_unit_value() is None:
                return None
        return UnionValue(self)

    def storage_variant(self, cache: MirLowerCache) -> Type | None:
        """The variant whose storage the union uses: the largest one (and,
        among equally large ones, the most aligned), or None when every
        variant is zero-sized.  The sizes are those of the variants'
        *mirrors*: the layout of a variant is the one the lowered code uses
        (see ``mir.estimated_size_of``)."""
        pointer_size = cache.target.pointer_size
        best: Type | None = None
        best_key: tuple[int, int] | None = None
        for type in self.types:
            variant_mir = type.to_mir_type(cache)
            if variant_mir is None:
                # a variant with no storage of its own - a zero-sized one (or a
                # compile-time-only one, which has no runtime representation
                # either) - never takes the union's storage
                continue
            key = (
                mir.estimated_size_of(variant_mir, pointer_size),
                mir.estimated_alignment_of(variant_mir, pointer_size),
            )
            if best_key is None or key > best_key:
                best = type
                best_key = key
        return best

    @override
    def classify(self) -> SpecialTypeKind:
        if any(type.classify() == SpecialTypeKind.COMPTIME for type in self.types):
            return SpecialTypeKind.COMPTIME
        if any(type.classify() == SpecialTypeKind.DST for type in self.types):
            return SpecialTypeKind.DST
        if all(type.classify() == SpecialTypeKind.ZST for type in self.types):
            return SpecialTypeKind.ZST
        return SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if any(type.classify() == SpecialTypeKind.DST for type in self.types):
            # the union is laid out as its largest variant: a dynamically-sized
            # one has no size to hold, so such a union has no representation yet
            raise CompileError(
                f'a union with a dynamically-sized variant ({self}) '
                f'is not supported yet'
            )
        payload = self.storage_variant(cache)
        if payload is None:
            # every variant is zero-sized: the union holds no storage
            return None
        mir_payload = payload.to_mir_type(cache)
        if mir_payload is None:
            return None
        return cache.union_mir(self, mir_payload)

    def __str__(self) -> str:
        return f"union[{", ".join(str(t) for t in self.types)}]"


@dataclass(frozen=True)
class UnionValue(Value):
    """The value of a payload union that holds no storage (every variant is
    zero-sized, or there are none): a union value carries no variant of its own
    - the error code next to the payload *is* the tag - so the only thing such a
    value says is its union type.  A union with storage has no compile-time
    value at all."""

    type: UnionType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"union(({self.type}))"


@dataclass(frozen=True, slots=True)
class TaggedUnionType(Type):
    """A tagged union ``A | B | C``: a value of exactly one of ``types`` (the
    variants, in the order written) together with the tag that says which one.
    The tag is the variant's position, so the order of the variants is part of
    the type: two tagged unions with the same variants in a different order are
    each a subtype of the other, and a conversion between them remaps the tag
    (see ``interp``).

    The representation (see :meth:`to_mir_type`) is a struct of the tag and the
    payload - the (untagged) :class:`UnionType` of the variants - or, when the
    payload holds no storage (every variant is zero-sized), just the tag.  A
    tagged union of a *single* variant is that variant's type: it has no tag of
    its own, its representation is the variant's, and it is zero-sized when the
    variant is."""

    types: FrozenArraySet[Type]

    @property
    def tag_bits(self) -> int:
        """The width of the tag: enough bits for the variant's position (none
        for a single variant, which has no tag of its own)."""
        return (len(self.types) - 1).bit_length()

    def tag_type(self) -> IntType:
        return IntType(self.tag_bits, False)

    def payload_type(self) -> UnionType:
        return UnionType(frozenset(self.types))

    def has_tag(self) -> bool:
        return len(self.types) > 1

    def variant_index(self, type: Type) -> int | None:
        """The position of the variant ``type`` (an exact match), or None."""
        for index, variant in enumerate(self.types):
            if variant == type:
                return index
        return None

    def variant_index_for(self, type: Type) -> int | None:
        """The position of the variant ``type`` is a subtype of, or None: which
        variant a value of the spy type ``type`` belongs to."""
        for index, variant in enumerate(self.types):
            if type.is_subtype_of(variant):
                return index
        return None

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return tuple(self.types)

    @override
    def get_unit_value(self) -> AnyValue | None:
        """A tagged union holds no storage when it has a single variant and
        that variant is zero-sized: every value of it equals the variant's unit
        value under tag 0."""
        if len(self.types) != 1:
            return None
        unit = self.types[0].get_unit_value()
        if unit is None:
            return None
        return TaggedUnionValue(self, 0, unit)

    @override
    def classify(self) -> SpecialTypeKind:
        if len(self.types) == 1:
            # a single variant is that variant's type
            return self.types[0].classify()
        if any(type.classify() == SpecialTypeKind.COMPTIME for type in self.types):
            return SpecialTypeKind.COMPTIME
        if any(type.classify() == SpecialTypeKind.DST for type in self.types):
            return SpecialTypeKind.DST
        # with more than one variant the tag alone holds storage, so a tagged
        # union is never zero-sized
        return SpecialTypeKind.NONE

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """A tagged union is a subtype of a tagged union whose variants include
        all of its own (a conversion remaps the tag, see ``interp``)."""
        return (
            isinstance(other, TaggedUnionType)
            and all(variant in other.types for variant in self.types)
        )

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, UndefinedType):
            return self
        if isinstance(other, TaggedUnionType):
            # the union with the smaller variant set is the peer type: a value
            # of it is a value of the larger one as well (with a remapped tag)
            if all(variant in other.types for variant in self.types):
                return other
            if all(variant in self.types for variant in other.types):
                return self
            return None
        if self.variant_index_for(other) is not None:
            return self
        return None

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if any(type.classify() == SpecialTypeKind.DST for type in self.types):
            # the payload union holds one variant's storage: a dynamically-sized
            # variant has none, so such a union has no representation yet
            raise CompileError(
                f'a tagged union with a dynamically-sized variant ({self}) '
                f'is not supported yet'
            )
        if len(self.types) == 1:
            # a single variant: the representation is the variant's own
            return self.types[0].to_mir_type(cache)
        payload = self.payload_type().to_mir_type(cache)
        tag = self.tag_type().to_mir_type(cache)
        if payload is None:
            # every variant is zero-sized: only the tag is stored
            return tag
        assert tag is not None
        return cache.tagged_union_mir(self, tag, payload)

    def __str__(self) -> str:
        return ' | '.join(str(type) for type in self.types)


def tagged_union_of(types: tuple[Type, ...]) -> TaggedUnionType:
    """The tagged union of ``types``: the variants, in the order written, with
    the duplicates dropped (the order is part of the type - it is the tag)."""
    variants: list[Type] = []
    for type in types:
        if not any(type == variant for variant in variants):
            variants.append(type)
    assert len(variants) > 0, 'a tagged union has at least one variant'
    return TaggedUnionType(FrozenArraySet(variants))


@dataclass(frozen=True, slots=True)
class TaggedUnionValue(Value):
    """The compile-time constant of a tagged union: the value ``value`` of the
    variant ``index`` (see :func:`coerce_const`).  The interpreter's own
    compile-time value is ``interp.ComptimeTaggedUnionValue``, exactly like the
    option's is ``interp.ComptimeOption``: this one is what the type rules can
    build on (a default field value, e.g.)."""

    type: TaggedUnionType
    index: int
    value: AnyValue

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'{self.type}({self.index}: {self.value!r})'


@dataclass(frozen=True, slots=True)
class TaggedUnionApplication:
    """A ``|`` chain written in an *annotation*, whose operands are not known
    yet: Python evaluates the annotation in the annotation scope of the
    annotated function or class, so the items may name its type parameters,
    which this handle does not see.  :func:`as_value` turns it into a
    :class:`TaggedUnionType` once it is given that scope and the host to resolve
    the items in (a struct declared by another context resolves to that
    context's copy).

    Not a :class:`Value`: it is a transient Python-level object that never
    denotes a value of the spy domain."""

    items: tuple[Any, ...]

    def __or__(self, other: Any) -> TaggedUnionApplication:
        return TaggedUnionApplication((*self.items, other))

    def __ror__(self, other: Any) -> TaggedUnionApplication:
        return TaggedUnionApplication((other, *self.items))


def union_application(a: Any, b: Any) -> TaggedUnionApplication:
    """The ``a | b`` application: a :class:`TaggedUnionApplication` operand is
    flattened, so a chain ``A | B | C`` is one application of three items."""
    items: list[Any] = []
    for item in (a, b):
        if isinstance(item, TaggedUnionApplication):
            items.extend(item.items)
        else:
            items.append(item)
    return TaggedUnionApplication(tuple(items))


@dataclass(frozen=True)
class ResultType(Type):
    """The result type ``ResultType[return_type, T1, T2, ...]`` of a function:
    the type of the value it returns normally and the set of exceptions it may
    raise, in error-code order.  It is a compile-time-only type: it never has a
    value of its own at runtime, because when a function returns one it is
    *spread out* into the value, the error code and the payload union (see
    :func:`make_ret_spec`).

    An ordinary function carries the implicit "no error" tag ``0``, so its i-th
    exception is tagged ``i + 1`` and the code needs ``len(types).bit_length()``
    bits (see :attr:`tag_bits`).  A function whose return type is the *empty*
    type - a body that never delivers a value, so that no ``return`` path can
    exist - has no such tag: its i-th exception is tagged ``i``, one exception
    needs no code at all, and a function that also raises nothing never returns
    at all (``mir.NoReturn``).

    A result type with an empty exception set is the unit type of
    :class:`Success` (whatever it returns): the outcome of a path that raises no
    error."""

    return_type: Type
    types: FrozenArraySet[Type]

    @property
    def tag_base(self) -> int:
        """The error code of the first exception: ``0`` when there is no value
        to return (the codes carry no "no error" tag), ``1`` otherwise (``0``
        is then the tag of a successful return)."""
        return 0 if isinstance(self.return_type, EmptyType) else 1

    @property
    def tag_bits(self) -> int:
        """The width in bits of the error code: the smallest that can hold the
        tags the function uses (``0`` for a function that raises nothing, and
        for the one exception of a value-less function, whose tags are
        ``0..len(types)`` and ``0..len(types) - 1`` respectively)."""
        n = len(self.types)
        if n == 0:
            return 0
        return (n - 1).bit_length() if self.tag_base == 0 else n.bit_length()

    @property
    def code_type(self) -> IntType:
        """The type of the error code: an unsigned integer of :attr:`tag_bits`
        bits (``u0`` for an empty error union, and for a value-less function
        with a single exception)."""
        return IntType(self.tag_bits, False)

    @property
    def union(self) -> UnionType:
        """The payload union: the storage of the exception value the error
        code tags."""
        return UnionType(frozenset(self.types))

    def code_of(self, exception: Type) -> int:
        """The error code of ``exception`` in this result type: the position it
        holds in the exception set, offset by :attr:`tag_base`."""
        return self.tag_base + self.types.index(exception)

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return (self.return_type, *self.types)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """A result type is a subtype of a result type whose exception set
        contains every exception of this one - so the empty error union is a
        subtype of every error union."""
        return isinstance(other, ResultType) and all(t in other.types for t in self.types)

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        """The peer of two result types: the union of their exception sets, in
        first-delivery order (this one's exceptions first), with this one's
        return type.  A result type is never the type of a runtime value, so
        this is only reached for the parts of a signature."""
        if isinstance(other, UndefinedType):
            return self
        if not isinstance(other, ResultType):
            return None
        types = list(self.types)
        for type in other.types:
            if type not in types:
                types.append(type)
        return ResultType(self.return_type, FrozenArraySet(types))

    @override
    def get_unit_value(self) -> AnyValue | None:
        # an empty error union is the unit type, whose unique value is
        # ``Success``; a non-empty one always has a code, which is not
        # zero-sized
        return Success() if len(self.types) == 0 else None

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        # compile-time only: the type is spread out into the value, the code and
        # the payload union when it is lowered (see ``make_ret_spec``)
        return None

    def __str__(self) -> str:
        exceptions = ", ".join(str(t) for t in self.types)
        return f"result[{self.return_type}, {exceptions}]"


class Success(Value):
    """The unique *value* of the empty error union of :class:`ResultType`:
    the outcome of a path that raises no error.  Delivering it into an error
    location writes the "no error" tag (``0``)."""

    @override
    def get_type(self) -> Type:
        return ResultType(VoidType(), FrozenArraySet())

    def __str__(self) -> str:
        return 'success'


@dataclass(frozen=True)
class StrDictType(Type):
    values: frozendict[str, Type]

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return f'{{{", ".join(f"{k}: {v}" for k, v in self.values.items())}}}'

TYPE_TYPE = TypeType()

@dataclass(frozen=True)
class BytesType(Type):
    """A compile-time byte string: the spy type of a ``bytes`` literal/value.
    A byte string only exists at compile time - it has no MIR mirror of its own
    - and is turned into a runtime pointer/slice by the ``std.core.gstr`` /
    ``std.core.sstr`` builtins (see ``interp``).  Its compile-time value is the
    raw Python ``bytes`` object."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'bytes'

@dataclass(frozen=True)
class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return mir.BoolType()

    def __str__(self) -> str:
        return 'bool'

@dataclass(frozen=True)
class EmptyType(Type):
    """The *empty* type: no value of it exists at all, so a location of it
    (a function's result location of a body that never delivers a result) can
    never be written.  It is what marks a function that never returns a value:
    with no exceptions either it can never return at all (a ``mir.NoReturn``
    function), and one that raises delivers its errors through the codes
    ``0..len(types) - 1`` alone - it carries no "no error" code (see
    :class:`ResultType`).

    For the machinery of the compiler it behaves like a zero-sized type: it
    has no mirror of its own and the value a slot of it holds is the "no
    value" marker ``Void()``."""

    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        # no value of the empty type exists; a slot of it holds the "no value"
        # marker, exactly like a zero-sized location
        return Void()

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        return other

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.ZST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'empty'

@dataclass(frozen=True)
class VoidType(Type):
    """The unit type: a zero-sized type (ZST) whose unique value is
    :class:`Void` (``sval.Void()``).  It is the declared return type of a
    function that returns no value (``-> None``, or one inferred for a body
    without value returns), and it has no runtime representation: it has no
    mirror of its own (``to_mir_type`` returns ``None``) and no load/store is
    ever emitted for it."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        return Void()

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.ZST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

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
        if isinstance(other, UndefinedType):
            return self
        match other:
            case NullType() | VoidType() | OptionType():
                return other
            case _:
                return OptionType(other)

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.ZST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

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

@dataclass(frozen=True, slots=True)
class TypedNull(Value):
    """The absent value of an ``Option[T]``, *typed*: the compile-time form of
    a missing option value, where the untyped :class:`Null` is what the Python
    literal ``None`` evaluates to and peers with every type.  A compile-time
    option value is either this - absent - or an ``interp.ComptimeOption`` -
    present; a compile-time aggregate field of an option type holds one of the
    two (see ``std.slice``).

    The type argument is the option's *child*, so the absent value of an option
    of an option is expressible as well (its static type is
    ``Option[child]``)."""

    child: Type

    @override
    def get_type(self) -> Type:
        return OptionType(self.child)

    def __str__(self) -> str:
        return f'null[{self.child}]'

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
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return (self.child,)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        return isinstance(other, OptionType) and self.child.is_subtype_of(other.child)

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, UndefinedType):
            return self
        match other:
            case NullType():
                return self
            case OptionType():
                child = self.child.resolve_peer_type(other.child)
            case _:
                child = self.child.resolve_peer_type(other)
        return None if child is None else OptionType(child)

    @override
    def classify(self) -> SpecialTypeKind:
        match self.child.classify():
            case SpecialTypeKind.COMPTIME:
                return SpecialTypeKind.COMPTIME
            case SpecialTypeKind.DST:
                return SpecialTypeKind.DST
            case _:
                # an option is never zero-sized: it keeps whether a value is there
                return SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        child = self.child
        if child.is_zst():
            # a zero-sized child carries no value: only whether there is one
            return mir.BoolType()
        child_mir = child.to_mir_type(cache)
        if child_mir is None:
            if child.classify() == SpecialTypeKind.DST:
                # a dynamically-sized child has no mirror of its own: the option
                # stores it as its flexible member (the tag alone has storage)
                return cache.option_struct_mir(child, None)
            return None
        if find_first_pointer_type_pos(child) is not None:
            # the child holds a pointer, which is null exactly when the option
            # is: the child itself is the representation
            return child_mir
        # no pointer to use as the tag: a struct of the tag and the value
        return cache.option_struct_mir(child, child_mir)

    @override
    def is_copyable(self) -> bool:
        return self.child.is_copyable()

    def __str__(self) -> str:
        return f'Option[{self.child}]'

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


def alignment_of(type: Type, cache: MirLowerCache) -> int:
    """The alignment in bytes of the spy type ``type`` for the target of
    ``cache``.  A type that has a MIR mirror is measured through it
    (``mir.estimated_alignment_of``); a zero-sized type has no mirror, so its
    alignment is read off its *structure*: an array aligns to its element
    (``T[0]``/``T[?]``/``T[N]`` alike), a struct to the maximum of its fields
    (1 when it has none), a union to the maximum of its variants, a
    single-variant tagged union to that variant, and every other leaf ZST
    (void, ``u0``, ``Null``, ``undefined``, ...) to 1.  Iterative (an explicit
    worklist), so a deeply nested type costs no Python stack.

    Note that a struct's alignment takes the maximum over *every* declared
    field here, which is what makes an all-ZST struct such as
    ``struct { x: i64[0] }`` align to 8.  A struct that has storage drops its
    zero-sized fields from its MIR mirror (see ``StructType._calculate_mir``),
    so for one of those this structural alignment is *not* what the lowered
    layout uses; reconciling the two is left for later (see the README)."""
    pointer_size = cache.target.pointer_size
    best = 1
    todo: list[Type] = [type]
    while len(todo) > 0:
        current = todo.pop()
        if not current.is_zst():
            mir_type = current.to_mir_type(cache)
            if mir_type is None:
                raise CompileError(f'cannot take the alignment of {current}')
            best = max(best, mir.estimated_alignment_of(mir_type, pointer_size))
            continue
        if isinstance(current, ArrayType):
            todo.append(current.elem)
        elif isinstance(current, StructType):
            todo.extend(field.type for field in current.fields().values())
        elif isinstance(current, UnionType):
            todo.extend(current.types)
        elif isinstance(current, TaggedUnionType):
            # a tagged union is zero-sized only with a single variant
            todo.append(current.types[0])
        # any other leaf ZST contributes 1, which is the running maximum
    return best


def aggregate_type_length(type: Type) -> int:
    """The number of fields (or elements) of an aggregate type, in declaration
    (element) order.  An ``Option`` is *not* an aggregate (its representation
    is chosen from its child, see :meth:`OptionType.to_mir_type`), so asking it
    for one is an error."""
    if isinstance(type, StructType):
        return len(type.fields().by_id)
    if isinstance(type, ArrayType):
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot tell how many elements {type} holds')
        return length
    raise CompileError(f'{type} is not an aggregate')

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
    while running the HIR (``spy.compile_log``).  The name identifies the
    builtin to the interpreter; ``spy.as_`` is not a compile-time builtin
    (it only exists at the call boundary), and ``spy.typeof`` is a
    ``syntax`` marker lowered to a type probe rather than a builtin."""

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
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'int'

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, UndefinedType):
            return self
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
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.ZST if self.bits == 0 else SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None if self.bits == 0 else mir.IntType(self.bits, self.signed)

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
        if isinstance(other, UndefinedType):
            return self
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
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
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

def _peer_const(a: AnyValue, b: AnyValue) -> AnyValue:
    # the constness two pointers peer to: the const one (a ``*T`` converts to a
    # const ``*T``, not the other way around).  A constness that is still a type
    # parameter is kept as it is - the solver constrains it (see
    # ``TypeVarSolver.add_constraint``)
    if isinstance(a, bool) and isinstance(b, bool):
        return a or b
    return a if isinstance(a, Type) else b


def _points_at_elements_of(ptr: PointerType, other: PointerType) -> bool:
    """Whether the pointer ``ptr`` - a pointer to an array - is the *multi*
    pointer ``other`` of the array's elements: a pointer to an array points at its
    first element, and its elements follow one another, so the two carry the same
    address."""
    return (
        ptr.variant == PointerVariant.SINGLE
        and other.variant == PointerVariant.MULTI
        and isinstance(ptr.elem, ArrayType)
        and ptr.elem.elem == other.elem
    )


@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: AnyValue = False # bool
    variant: PointerVariant = PointerVariant.SINGLE

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """One pointer is a subtype of another when the address it carries may
        be used as the other's: the pointee type has to be the same (pointers do
        not convert between pointee types), the constness may only be added (a
        ``*T`` is a ``const *T``), and a *multi* pointer is a single one (a
        multi pointer may be dereferenced like a single one, but not the other
        way around: indexing needs the multi form).  A pointer to an array is an
        exception to the pointee rule: it is also a *multi* pointer to the
        array's elements (its first element is the array's address and the
        elements follow one another)."""
        if not isinstance(other, PointerType):
            return False
        if self.elem == other.elem:
            if self.variant == PointerVariant.SINGLE and other.variant == PointerVariant.MULTI:
                # a single pointer is not a multi one: indexing it is meaningless
                return False
        elif not _points_at_elements_of(self, other):
            return False
        # the constness may not be dropped: a ``const *T`` does not convert to a
        # ``*T``
        return not (self.is_const is True and other.is_const is not True)

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        """The peer of two pointers is the one both convert to: the pointee
        types have to match, the result is const when either is, and it is a
        single pointer as soon as either is (a multi pointer converts to a
        single one, not the other way around)."""
        if isinstance(other, UndefinedType):
            return self
        if isinstance(other, NullType):
            return OptionType(self)
        if isinstance(other, OptionType):
            return _resolve_option_peer(self, other)
        if isinstance(other, PointerType):
            if self.elem != other.elem:
                return None
            is_const = _peer_const(self.is_const, other.is_const)
            if self.variant == other.variant:
                variant = self.variant
            else:
                # a multi pointer and a single one: the single one is what both
                # convert to
                variant = PointerVariant.SINGLE
            return PointerType(self.elem, is_const, variant)
        return other if self.is_subtype_of(other) else None

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        # the pointee type, and the constness while it is still a type
        # parameter (a pointer type constrains both, see ``TypeVarSolver``)
        if isinstance(self.is_const, Type):
            return (self.elem, self.is_const)
        return (self.elem,)

    @override
    def classify(self) -> SpecialTypeKind:
        # a pointer always has a size, even one to a dynamically-sized type
        if not isinstance(self.is_const, bool) or self.elem.classify() == SpecialTypeKind.COMPTIME:
            return SpecialTypeKind.COMPTIME
        return SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if not isinstance(self.is_const, bool):
            return None
        if self.elem.is_zst():
            # the pointee has no value of its own: the pointer is a void pointer
            return mir.PointerType(mir.VOID, self.is_const)
        if isinstance(self.elem, OpaqueType):
            # a pointer to an opaque type is a void pointer
            return mir.PointerType(mir.VOID, self.is_const)
        if isinstance(self.elem, ArrayType) and self.elem.length is None:
            # a pointer to an unsized array points at its first element: it is a
            # plain pointer to that element type (``*[?]T -> *T``)
            elem = self.elem.elem.to_mir_type(cache)
            if elem is None:
                return None
            return mir.PointerType(elem, self.is_const)
        child = self.elem.to_mir_type(cache)
        if child is None:
            return None
        return mir.PointerType(child, self.is_const)

    def __str__(self) -> str:
        variant = 'mptr' if self.variant == PointerVariant.MULTI else 'ptr'
        if self.is_const is True:
            variant = 'c' + variant
        return f"{variant}({self.elem})"

@dataclass(frozen=True, slots=True)
class ArrayType(Type):
    """A spy array type: ``length`` values of the element type ``elem``, in a
    row.  The length is a *value* (a Python ``int``, or an ``Int``), or ``None``
    for an *unsized* array - a dynamically-sized type whose length is not part
    of the type (see ``classify``).  A signature that takes or returns a sized
    array spells the length out, since the Python type system cannot infer it
    from the arguments of ``array(...)`` (see ``syntax``)."""

    elem: Type
    length: AnyValue | None # int; None = unknown length (a DST)

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
        return TYPE_TYPE

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
        An *unsized* array (``length is None``) is a subtype of another unsized
        array of the same element type.  Note that the base rule would take any
        array for any array."""
        if not isinstance(other, ArrayType) or self.elem != other.elem:
            return False
        if self.length is None or other.length is None:
            return self.length is None and other.length is None
        return self.length_int is not None and self.length_int == other.length_int

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
    def classify(self) -> SpecialTypeKind:
        if self.length is None:
            # an array of unknown length holds no storage of its own: it is a
            # dynamically-sized type (a C flexible array member)
            return SpecialTypeKind.DST
        length = self.length_int
        if length is None:
            # the length is not known yet, so nothing is known about the layout
            return SpecialTypeKind.COMPTIME
        if length == 0:
            # a zero-length array holds no storage whatever its element type
            return SpecialTypeKind.ZST
        match self.elem.classify():
            case SpecialTypeKind.COMPTIME:
                return SpecialTypeKind.COMPTIME
            case SpecialTypeKind.DST:
                return SpecialTypeKind.DST
            case SpecialTypeKind.ZST:
                return SpecialTypeKind.ZST
            case _:
                return SpecialTypeKind.NONE

    @override
    def fam_mir_type(self, cache: MirLowerCache) -> mir.MayBeVoidType:
        assert self.length is None, 'only an unsized array is a flexible member'
        elem = self.elem.to_mir_type(cache)
        if elem is None or self.elem.is_zst() or self.elem.classify() == SpecialTypeKind.DST:
            raise CompileError(
                f'the elements of the unsized array {self} have no fixed runtime '
                f'representation'
            )
        return elem

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if self.length is None:
            # an unsized array has no mirror of its own: only a pointer to it is
            # a value (and that pointer is a plain pointer to its elements, see
            # ``PointerType.to_mir_type``)
            return None
        length = self.length_int
        if length is None:
            return None
        if length == 0 or self.elem.is_zst():
            # a zero-sized array holds no storage whatever its length: it has
            # no mirror of its own
            return None
        elem = self.elem.to_mir_type(cache)
        if elem is None:
            # an element with no mirror of its own leaves the array without
            # one either
            return None
        return mir.ArrayType(elem, length)

    def __str__(self) -> str:
        length = '?' if self.length is None else str(self.length)
        return f"{self.elem}[{length}]"

@dataclass(frozen=True, slots=True)
class OpaqueType(Type):
    """An opaque type (``syntax.Opaque``): a dynamically-sized type of unknown
    layout.  It has no value of its own - only a pointer to it is one, and that
    pointer is a ``void*`` (see ``PointerType.to_mir_type``).  As the last field
    of a struct it is an opaque tail (see ``sval.StructType._calculate_mir`` and
    ``mir.StructType.fam_type``)."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        # an opaque type has no size of its own
        return SpecialTypeKind.DST

    @override
    def fam_mir_type(self, cache: MirLowerCache) -> mir.MayBeVoidType:
        # an opaque tail is lowered as a zero-length ``i8`` array (it has no
        # type to name; the pointer to it is a void pointer)
        return mir.VOID

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        # an opaque type cannot be lowered on its own: only a pointer to it is a
        # value (see ``PointerType.to_mir_type``)
        return None

    def __str__(self) -> str:
        return 'Opaque'

@dataclass(frozen=True, slots=True)
class Undefined(Value):
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return "undefined"

class UntypedUndefined(Value):
    """The unique value of :class:`UndefinedType`: what the ``std.core.undefined``
    builtin evaluates to at compile time.  It is the *untyped* undefined - a
    value of no fixed type - which converts to a value of any type: written into
    a location of the type ``T`` it becomes the typed :class:`Undefined` of
    ``T`` (see :func:`coerce_const`)."""

    @override
    def get_type(self) -> Type:
        return UndefinedType()

    def __str__(self) -> str:
        return 'undefined'

@dataclass(frozen=True, slots=True)
class UndefinedType(Type):
    """The type of :class:`UntypedUndefined` - the undefined value
    ``std.core.undefined`` evaluates to.  It is a zero-sized type whose only
    value is :class:`UntypedUndefined`, and it converts to any other type: a
    value of it may be written where a value of any type is expected, leaving
    the location undefined (see :func:`coerce_const` and ``interp``).  It is the
    *bottom* of the peer rule as well: paired with any other type, the other type
    wins."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        return UntypedUndefined()

    @override
    def is_subtype_of(self, other: Type) -> bool:
        return True

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        # undefined takes the type of whatever it is paired with
        return other

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.ZST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'undefined'

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
    def classify(self) -> SpecialTypeKind:
        # the type of an untyped literal: a compile-time value the runtime
        # location it is written to has to declare the type of
        return SpecialTypeKind.ZST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return f"Literal({self.value})"


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type
    default_value: AnyValue | None


# the shared, immutable default of ``FunctionType.exceptions``: a function that
# raises nothing (see ``FrozenArraySet``)
_NO_EXCEPTIONS: FrozenArraySet[Type] = FrozenArraySet()


@dataclass(frozen=True)
class FunctionType(Type):
    args: tuple[FormalArg, ...]
    # the spy type of the values the function returns: the type of a single
    # result, or a ``tuple[...]`` when it returns several (see
    # :func:`make_ret_spec`)
    return_type: Type
    # the exceptions the function may raise, in error-code order (an empty
    # set for a function that raises nothing).  A function that does not use
    # the default calling convention may not raise at all (see ``callconv``)
    exceptions: FrozenArraySet[Type] = _NO_EXCEPTIONS
    # the calling convention: ``'default'`` is the spy convention (a result
    # outgrowing the by-value limit goes through a hidden result pointer, an
    # aggregate argument may be passed by reference, and the function may
    # raise); any other value names a C convention, in which every argument is
    # passed by value, the result is returned by value, and the function may
    # not raise
    callconv: str = 'default'
    # whether the function may panic (see ``interp``: a call of one may unwind
    # through the enclosing deferred bodies); a function may panic by default
    may_panic: bool = True

    def ret_spec(self, cache: MirLowerCache) -> RetSpec:
        """How a call of a function of this signature delivers its result
        (see :func:`make_ret_spec`); ``cache`` is the MIR-mirror cache of the
        host the call is compiled for."""
        if len(self.exceptions) == 0:
            base: Type = self.return_type
        else:
            base = ResultType(self.return_type, self.exceptions)
        return make_ret_spec(base, cache, force_by_value=self.callconv != 'default')

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        # a function type is dynamically sized: a value of it has no size of its
        # own, so only a pointer to one is a value (see ``SpecialTypeKind``)
        return SpecialTypeKind.DST

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        is_c = self.callconv != 'default'
        args: list[mir.Type] = []
        for arg in self.args:
            if arg.type.is_zst():
                continue
            if not is_c and pass_by_ref(arg.type, cache) is TriState.TRUE:
                # an argument the default convention passes as a const pointer
                # (a dynamically-sized one always is: its pointee type is what
                # the pointer is built from)
                arg_ptr = PointerType(arg.type, is_const=False).to_mir_type(cache)
                if arg_ptr is None:
                    return None
                args.append(arg_ptr)
            else:
                if arg.type.classify() == SpecialTypeKind.DST:
                    # the C convention forces a dynamically-sized argument by
                    # value, which has no size to pass
                    raise CompileError(
                        f'a C function may not take the dynamically-sized type {arg.type}'
                    )
                mir_type = arg.type.to_mir_type(cache)
                if mir_type is None:
                    return None
                args.append(mir_type)
        ret_type: mir.MayBeVoidType = mir.VOID
        for leaf in iter_ret_leaves(self.ret_spec(cache)):
            if leaf.type.is_zst():
                # a zero-sized result is delivered as its unit value, not
                # through the by-value slot
                continue
            if leaf.via_result_ptr:
                # delivered through a hidden result pointer (a dynamically-sized
                # result always is)
                ptr_mir = PointerType(leaf.type, is_const=False).to_mir_type(cache)
                if ptr_mir is None:
                    return None
                args.append(ptr_mir)
            else:
                if leaf.type.classify() == SpecialTypeKind.DST:
                    # only a non-default convention could force a dynamically-
                    # sized result by value, which has no size to return
                    raise CompileError(
                        f'a C function may not return the dynamically-sized type {leaf.type}'
                    )
                mir_type = leaf.type.to_mir_type(cache)
                if mir_type is None:
                    return None
                ret_type = mir_type
        return mir.FunctionType(tuple(args), ret_type, self.callconv, self.may_panic)

    def __str__(self) -> str:
        return f"fn({', '.join(str(arg.type) for arg in self.args)}) -> {self.return_type}"


@dataclass(frozen=True)
class DeclareFunction(Value):
    """An external function declared by ``@decl_func(linkname)``: its signature
    (a :class:`FunctionType`) and the link name it resolves to.  Its own type is
    a const pointer to the signature - a function pointer - and the interpreter
    lowers it to a ``mir.ExternSymbol`` (see ``interp``)."""

    type: FunctionType
    linkname: str

    @override
    def get_type(self) -> Type:
        return PointerType(self.type, is_const=True)

    def __str__(self) -> str:
        return f'decl_func({self.linkname!r}: {self.type})'


@dataclass(frozen=True)
class StructField:
    name: str
    type: Type
    # the value a construction leaves the field at when it is not given
    # (``None`` when the field has no default): the value the class body
    # assigned to the annotated attribute (see ``dsl``), as a spy value, which
    # a construction coerces to the field's type (see ``interp.finish_struct``)
    default: AnyValue | None = None

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

    def __init__(self, name_base: str, generic_args: tuple[TypeVar, ...] = (), modifiers: StructModifiers | None = None, generic_defaults: tuple[AnyValue | None, ...] = ()) -> None:
        self.name_base = name_base
        self.generic_args = generic_args
        # the declared default of every type parameter (None for one that has
        # none): a use that leaves the last arguments out takes them
        self.generic_defaults = generic_defaults
        self.modifiers = modifiers or StructModifiers()
        self.fields: IndexedMap[str, StructField] = IndexedMap()
        self.methods: dict[str, Any] = {}
        # the methods of the body declared ``@staticmethod``: they take no
        # receiver, so a call does not pass one (see ``interp.call_method``)
        self.static_methods: set[str] = set()
        self._specs: dict[tuple[AnyValue, ...], StructType] = {}

    def is_static_method(self, name: str) -> bool:
        """Whether the method ``name`` of this struct declares
        ``@staticmethod`` (it takes no ``self``)."""
        return name in self.static_methods

    @override
    def get_type(self) -> Type:
        # a head is a type-level compile-time object (the template a struct
        # specialization is an application of), so its own type is ``type``:
        # that is what lets reflection carry it in an ``Any`` field (see
        # ``std.reflect.StructType.head``)
        return TYPE_TYPE

    def add_field(self, name: str, type: Type, default: AnyValue | None = None) -> None:
        """Declare one field, appended after the fields declared so far, with
        the spy value a construction leaves it at when it is not given."""
        assert name not in self.fields.by_key, f'{self.name_base} already has a field {name!r}'
        self.fields.add(name, StructField(name, type, default))

    def specialize(self, generic_args: tuple[AnyValue, ...]) -> StructType:
        """The struct type this head declares for ``generic_args``: the one
        specialization of the head for those arguments (created lazily, so
        that every reference to the same struct type names one object)."""
        if len(generic_args) < len(self.generic_args):
            # a trailing type parameter the use left out takes its declared
            # default (see ``dsl._RegisteredClass.get_entry``)
            padded = list(generic_args)
            for index in range(len(padded), len(self.generic_args)):
                default = (
                    self.generic_defaults[index]
                    if index < len(self.generic_defaults) else None
                )
                if default is None:
                    break
                padded.append(default)
            generic_args = tuple(padded)
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
    def classify(self) -> SpecialTypeKind:
        # a template denotes no value of its own: only a specialization is a
        # type
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
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

    def __init__(self, head: StructTypeHead, generic_args: tuple[AnyValue, ...]) -> None:
        self.head = head
        self.generic_args = generic_args

        self._fields: IndexedMap[str, StructField] | None = None
        # the mirror (None while it is not computed yet, and for a zero-sized
        # struct, which has none of its own); ``_field_mir_indices`` being set
        # is what tells a computed mirror from a missing one
        self._mir: mir.Type | None = None
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

    def is_static_method(self, name: str) -> bool:
        """Whether the method ``name`` of this struct is a ``@staticmethod``
        (see :meth:`StructTypeHead.is_static_method`)."""
        return self.head.is_static_method(name)

    def fields(self) -> IndexedMap[str, StructField]:
        """The fields of this specialization, in declaration order: the
        declared fields with the head's generic type parameters replaced by
        this specialization's arguments.  Computed once and cached."""
        if self._fields is None:
            reps = {k: v for k, v in zip(self.head.generic_args, self.generic_args)}
            self._fields = self.head.fields.map(
                lambda f: StructField(
                    f.name,
                    replace_type_vars_type(f.type, reps),
                    replace_type_vars_value(f.default, reps),
                )
            )
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
        return TYPE_TYPE

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
    def classify(self) -> SpecialTypeKind:
        dst = False
        zst = True
        for field in self.fields().values():
            match field.type.classify():
                case SpecialTypeKind.COMPTIME:
                    # a compile-time-only field leaves the struct with no runtime
                    # representation at all
                    return SpecialTypeKind.COMPTIME
                case SpecialTypeKind.DST:
                    # a dynamically-sized field leaves the struct with no size of
                    # its own (like a C struct with a flexible array member)
                    dst = True
                    zst = False
                case SpecialTypeKind.ZST:
                    pass
                case _:
                    zst = False
        if dst:
            return SpecialTypeKind.DST
        return SpecialTypeKind.ZST if zst else SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return self.get_mir_type(cache)

    def _calculate_mir(self, cache: MirLowerCache) -> None:
        if self._field_mir_indices is not None:
            return

        # the fields that occupy storage, each with the mirror of its type:
        # a zero-sized field occupies none and has no mirror position.  A
        # dynamically-sized field (an unsized array, an opaque type, a
        # dynamically-sized struct) has no mirror of its own either: the first
        # one becomes the struct's flexible member (the FAM, see
        # ``mir.StructType.fam_type``), placed after every field with storage,
        # and every further one occupies no position at all (see
        # ``get_field_mir_indices``)
        fields = self.fields().values()
        mirrored: list[tuple[int, StructField, mir.Type]] = []
        fam_type: mir.MayBeVoidType | None = None
        fam_index: int | None = None
        for index, field in enumerate(fields):
            if field.type.is_zst():
                continue
            if field.type.classify() == SpecialTypeKind.DST:
                if fam_index is None:
                    fam_index = index
                    fam_type = field.type.fam_mir_type(cache)
                continue
            field_mir = field.type.to_mir_type(cache)
            if field_mir is None:
                # a field has to have a runtime representation: a struct
                # whose field has none (a compile-time-only type, such as the
                # type of an untyped literal) has no layout
                raise CompileError(
                    f"field '{field.name}' of {self} has type {field.type}, "
                    f"which has no runtime representation"
                )
            mirrored.append((index, field, field_mir))

        # an ``extern_c`` struct is laid out for the C ABI: the mirror holds
        # its fields in declaration order.  A spy struct is laid out by the
        # compiler, which is free to reorder: the least-aligned fields come
        # first (the mirror then packs tighter), a non-``extern_c`` struct
        # that holds exactly one field *is* that field - its mirror is the
        # field's own mirror, with no wrapper struct - and one that holds
        # none is a zero-sized type, with no mirror of its own
        if not self.modifiers.extern_c:
            pointer_size = cache.target.pointer_size
            mirrored.sort(
                key=lambda entry: mir.estimated_alignment_of(entry[2], pointer_size),
            )

        indices: list[int | None] = [None] * len(fields)
        for position, (index, _, _) in enumerate(mirrored):
            indices[index] = position
        if fam_index is not None:
            # the flexible member sits after every field with storage: that is
            # the index ``mir.Gep`` - and the lowered LLVM struct - puts it at
            indices[fam_index] = len(mirrored)
        self._field_mir_indices = tuple(indices)

        if len(mirrored) == 0 and fam_type is None:
            self._mir = None
        elif len(mirrored) == 1 and not self.modifiers.extern_c and fam_type is None:
            self._mir = mirrored[0][2]
            self._mir_is_a_field = True
        else:
            self._mir = mir.StructType(
                self.name_base,
                tuple(
                    mir.FormalArg(field.name, field_mir)
                    for _, field, field_mir in mirrored
                ),
                fam_type,
            )

    @override
    def fam_mir_type(self, cache: MirLowerCache) -> mir.MayBeVoidType:
        # a dynamically-sized struct contributes its own mirror as the flexible
        # member of the aggregate that holds it (the mirror already carries its
        # own FAM, if any)
        mir_type = self.get_mir_type(cache)
        if mir_type is None:
            raise CompileError(f'{self} is a dynamically-sized struct with no layout')
        return mir_type

    def get_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        """The (cached) MIR mirror of the struct: the one ``mir`` type every
        value of the struct mirrors to (created lazily, shared by all users),
        with the zero-sized fields dropped, or ``None`` when the struct is
        zero-sized (it has no storage and so no mirror of its own).  An
        ``extern_c`` struct mirrors to a ``mir.StructType`` of its declaration
        order; a spy struct orders the fields by alignment instead, and mirrors
        to the type of its own field when it holds exactly one.  A struct with
        a dynamically-sized field mirrors to a ``mir.StructType`` whose
        ``fam_type`` is that field's flexible member (see
        :meth:`_calculate_mir`)."""
        self._calculate_mir(cache)
        return self._mir

    def get_field_mir_indices(self, cache: MirLowerCache) -> tuple[int | None, ...]:
        """The mirror position of every field, in declaration order: the
        i-th entry is the position of the i-th field in the mirror returned
        by :meth:`get_mir_type` - a zero-sized field occupies no position
        and maps to ``None``, and so does every dynamically-sized field
        beyond the first (which is the struct's FAM, at the position past
        every field with storage).  A mirror that is the type of the struct's
        own field (see :meth:`_calculate_mir`) has that field at position 0,
        and the field sits at the address of the value itself."""
        self._calculate_mir(cache)
        assert self._field_mir_indices is not None
        return self._field_mir_indices

    def mirror_is_a_field(self, cache: MirLowerCache) -> bool:
        """Whether the MIR mirror of this struct is the mirror of the
        struct's own single stored field rather than a wrapper struct (see
        :meth:`_calculate_mir`): that field then sits at the address of the
        value itself, so taking its address needs no field indirection.
        Note that the field's mirror may itself be a ``mir.StructType`` -
        the mirror of a *wrapper* struct and the mirror that *is* the field
        cannot be told apart by that type alone."""
        self._calculate_mir(cache)
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
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return "anyfn"

@dataclass(frozen=True, slots=True)
class ClosureType(Type):
    """The type of a *closure* value: a nested ``def``/``lambda`` that
    captures variables of an enclosing spy function.  A closure only exists
    at compile time - only an inline function can take one as an argument -
    so it has no MIR mirror (see ``SpecialTypeKind.COMPTIME``).  ``fn_type``
    is the closure's declared signature as a function type when it is
    complete, and ``None`` otherwise."""

    fn_type: FunctionType | None = None

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    def __str__(self) -> str:
        return 'closure' if self.fn_type is None else f'closure({self.fn_type})'


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
    generic_var_values: frozendict[TypeVar, AnyValue]

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


def returns_via_result_ptr(type: Type, cache: MirLowerCache) -> bool:
    """Whether a function returning ``type`` delivers its result by
    writing into a caller-provided result location (a hidden result
    pointer parameter) instead of returning the value directly.

    This is the default policy, a property of the *return type*: an
    aggregate is returned by value while it is small (up to
    :data:`_AGGREGATE_VALUE_RETURN_LIMIT` bytes) and through a result
    pointer once it outgrows it, and a new aggregate kind (arrays) only
    needs to extend this function.  Scalars are always returned by
    value, and a dynamically-sized type - which has no size at all - is
    always delivered through a result pointer.  The size is the one of the
    type's MIR mirror - the layout the lowered code uses (see
    ``mir.estimated_size_of``) - for pointers of the target the cache
    belongs to.  A signature may override the default
    (``fn.ReturnSignature.ret_spec``)."""
    if type.classify() == SpecialTypeKind.DST:
        # a dynamically-sized type has no size to return by value: it is always
        # delivered through a hidden result pointer (see also
        # ``HirRunner._ret_leaf_ptr``)
        return True
    match type:
        case StructType() | ArrayType() | OptionType() | UnionType() | TaggedUnionType():
            if _mentions_type_var(type):
                # the layout is not known until the call substitutes the
                # type parameter: assumed small now, re-decided on substitution
                return False
            mir_type = type.to_mir_type(cache)
            if mir_type is None:
                # an aggregate with no mirror of its own is zero-sized: it is
                # delivered as its unit value, not through a result pointer
                return False
            return (
                mir.estimated_size_of(mir_type, cache.target.pointer_size)
                > _AGGREGATE_VALUE_RETURN_LIMIT
            )
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
    written as an element of that annotation), or the ``ResultType[...]`` group
    a function's logical return type carries (its value, its error code and its
    payload union).  The whole return type is one :class:`RetSpec` - a
    :class:`RetValue` for a single value, a :class:`RetTuple` when it is a
    ``tuple[...]`` or a ``ResultType[...]`` - so a single result is just the
    trivial case and the nesting mirrors the annotation.

    The group is a *compile-time* regrouping of the values of its elements: a
    tuple has no runtime representation of its own, so the lowered signature
    delivers its leaves (see :func:`make_ret_spec`) and the caller regroups
    them (see ``interp``).  An error group holds exactly two leaves - the error
    code and the payload union."""

    # the ``TupleType`` (or ``ErrorUnionType``) the group was written as (its
    # ``types`` are the element types, one per entry of ``values``)
    type: Type
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


def ret_returned_type(spec: RetSpec) -> Type | None:
    """The spy type of the value the lowered function returns directly (the
    by-value result), or ``None`` when it returns void - every value then goes
    through a result pointer, or is zero-sized."""
    for leaf in iter_ret_leaves(spec):
        if not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
            return leaf.type
    return None


def ret_by_value_index(spec: RetSpec) -> int | None:
    """The position of the one leaf returned by value (its value has storage
    and fits in registers) among all the leaves, in depth-first declaration
    order, or ``None`` when the lowered function returns void."""
    for index, leaf in enumerate(iter_ret_leaves(spec)):
        if not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
            return index
    return None


def ret_spec_value_is_empty(spec: RetSpec | None) -> bool:
    """Whether the value part of a return spec is the empty type: such a
    function has no value to deliver, so no path of it may return (see
    :class:`EmptyType`)."""
    return isinstance(spec, RetValue) and isinstance(spec.type, EmptyType)


def result_leaves(type: Type) -> list[Type]:
    """The leaf types of a return annotation ``type``, in declaration order
    (depth first): the shapes ``tuple[...]`` and ``ResultType[...]`` spread
    into their elements, every other type is a leaf of its own (see
    :func:`make_ret_spec`).  The leaves are a property of the *type* alone -
    no layout is needed, only the *delivery* of each one has one."""
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
            case ResultType():
                # the result type spreads into the value, the error code and the
                # payload union (in that order)
                work.append(item.union)
                work.append(item.code_type)
                work.append(item.return_type)
            case _:
                leaves.append(item)
    return leaves


def make_ret_spec(type: Type, cache: MirLowerCache, force_by_value: bool = False) -> RetSpec:
    """The return convention of a function whose return annotation is the spy
    type ``type`` (a ``tuple[...]`` for several values, nested at whatever
    depth it is written): the annotation as one :class:`RetSpec` tree.  A
    ``tuple[...]`` - at the top level or nested - becomes a :class:`RetTuple`
    of its elements, a ``ResultType[...]`` a :class:`RetTuple` of the value it
    returns, its error code and its payload union (in that order), and every
    other type a :class:`RetValue` leaf.

    At most one leaf is returned *by value* - the first one that fits in
    registers (``returns_via_result_ptr`` says so) - and every other leaf is
    delivered by writing through a hidden result pointer; when no leaf
    qualifies the function returns void.  A zero-sized leaf has no value to
    return: it is delivered as its unit value and never takes the by-value
    slot (a value-less function's payload union can take the slot of the code
    its exception set needs no longer).  ``force_by_value`` overrides the
    by-value policy: the first non-zero-sized leaf is returned by value even
    when it outgrows the limit (the C convention, see
    ``FunctionType.callconv``)."""
    # the leaf types, in declaration order (depth first)
    leaves = result_leaves(type)
    chosen: int | None = None
    for index, leaf_type in enumerate(leaves):
        if leaf_type.get_unit_value() is not None:
            # a zero-sized value is delivered as its unit value, not
            # through the by-value slot
            continue
        if force_by_value or not returns_via_result_ptr(leaf_type, cache):
            chosen = index
            break
    via = tuple(
        leaf_type.get_unit_value() is None and index != chosen
        for index, leaf_type in enumerate(leaves)
    )
    # rebuild the tree of the annotation, marking every leaf with how it is
    # delivered (a ``None`` on the work stack closes the group it opened; the
    # whole type is itself a group when it is a ``tuple[...]`` or a
    # ``ResultType[...]``)
    index = 0
    groups: list[list[RetSpec]] = [[]]
    group_types: list[Type] = []
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
            case ResultType():
                build_work.append(None)
                build_work.append(item.union)
                build_work.append(item.code_type)
                build_work.append(item.return_type)
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


def pass_by_ref(type: Type, cache: MirLowerCache) -> TriState:
    """Whether a parameter of spy type ``type`` is passed by reference
    (as a const pointer) rather than by value: the calling convention of
    one argument, decided by the compiler.

    The policy mirrors :func:`returns_via_result_ptr`: an aggregate too
    large to be passed in registers (larger than the by-value limit) is
    passed as a pointer, everything else by value; the size is the one of
    the type's MIR mirror, for pointers of the cache's target.  A
    dynamically-sized type (a function type) has no size to pass, so it is
    always passed as a pointer.

    A type that still names a type parameter has no layout to size, so the
    answer is ``UNKNOWN``: the convention is settled when a call substitutes
    the parameter (see ``fn.Signature.specialize``, which combines the
    answer with what the formal declares)."""
    if _mentions_type_var(type):
        # the layout is not known until a call substitutes the type parameter:
        # left to the specialization to decide
        return TriState.UNKNOWN
    match type.classify():
        case SpecialTypeKind.DST:
            return TriState.TRUE
        case SpecialTypeKind.COMPTIME:
            return TriState.TRUE
        case SpecialTypeKind.ZST:
            return TriState.FALSE
        case SpecialTypeKind.NONE:
            mir_type = type.to_mir_type(cache)
            assert mir_type is not None
            return TriState.TRUE if mir.estimated_size_of(mir_type, cache.target.pointer_size) > _AGGREGATE_VALUE_RETURN_LIMIT else TriState.FALSE

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
        case bytes():
            return BytesType()
        case str():
            raise CompileError(
                'the str type is not available in spy: use bytes (a string '
                'literal is encoded to bytes automatically)'
            )

class StructDecl:
    """A Python-level object that declares a spy struct: the handle a
    ``@struct()`` class binds to (``dsl._RegisteredClass``).  The parser tells
    a construction from an ordinary call by the *type* of the callee object
    (see ``astgen``), because asking a function handle for its spy value
    parses the function body - which may reenter the parser (a recursive
    its spy value is asked of the host that owns it (see
    ``GlobalResolver.resolve_global``), so the struct is built in the
    context that resolves the declaration."""

    def __or__(self, other: Any) -> TaggedUnionApplication:
        return union_application(self, other)

    def __ror__(self, other: Any) -> TaggedUnionApplication:
        return union_application(other, self)

@dataclass(frozen=True, slots=True)
class StructTypeApplication:
    """A struct template applied to generic arguments whose spy value is not
    known yet: ``Foo[T]`` written in an annotation evaluates to this at the
    Python level (see ``dsl._RegisteredClass.__getitem__``), because Python
    evaluates the annotation in the annotation scope of the annotated
    function or class - the arguments name the type parameters of that
    scope, which ``__getitem__`` does not see.  :func:`as_value` turns the
    application into the struct specialization once it is given that scope
    (its ``type_vars``) and the host to resolve the template in (so that a
    struct declared by another context resolves to that context's copy).

    Not a :class:`Value`: it is a transient Python-level object that never
    denotes a value of the spy domain."""

    struct: StructDecl
    generic_vars: tuple[Any, ...]

    def __or__(self, other: Any) -> TaggedUnionApplication:
        return union_application(self, other)

    def __ror__(self, other: Any) -> TaggedUnionApplication:
        return union_application(other, self)

_POINTER_TYPES = (syntax.Ptr, syntax.ConstPtr, syntax.MultiPtr, syntax.ConstMultiPtr)
"""The ``syntax`` classes that name a pointer type: ``ConstPtr``/``ConstMultiPtr``
are the const spellings, ``MultiPtr``/``ConstMultiPtr`` the multi ones (see
``PointerVariant``)."""

def as_value(value: Any, ctx: CompileContext, type_vars: dict[typing.TypeVar, Value] | None = None) -> AnyValue:
    """The spy-domain value of a Python compile-time object: Python
    scalars and ``sval.Value`` objects pass through, and ``None`` is the
    ``Null`` value (the absent value of an option, the unit value of the
    zero-sized ``NullType``).  Class objects of the scalar types map to their
    default spy types, and any other object that the host knows - a struct
    class or a registered function handle - is resolved through ``resolver``
    (see :class:`GlobalResolver`), so that the object is resolved *in the
    resolving context* and every context gets its own handle (see
    ``dsl._Context.resolve_global``).  The resolver is required: an object
    only the host knows cannot be converted without one, and converting it in
    the wrong context would break the isolation between contexts."""
    if isinstance(value, (Value, int, float, bytes, bool)):
        return value
    if isinstance(value, str) or value is str:
        # spy has no ``str`` type: a string is a byte string (``bytes``), and a
        # string literal is encoded at parse time (see ``astgen``)
        raise CompileError(
            'the str type is not available in spy: use bytes (a string literal '
            "is encoded to bytes automatically)"
        )
    if value is None:
        # ``None`` denotes the null value: the absent value of an option
        return Null()
    if value is int:
        return AnyIntType()
    if value is float:
        return FloatType(64)
    if value is bytes:
        return BytesType()
    if value is type:
        return TYPE_TYPE
    if value is typing.Any:
        return AnyType()
    if value is bool:
        return BoolType()
    if value is syntax.USize:
        return IntType(ctx.target_info().usize_bits, False)
    if value is syntax.ISize:
        return IntType(ctx.target_info().usize_bits, True)
    if value is syntax.Opaque:
        # an opaque type: a dynamically-sized type of unknown layout, of which
        # only a pointer (a void pointer) is a value
        return OpaqueType()
    if value is typing.Never or value is typing.NoReturn:
        # ``Never`` (and its deprecated alias ``NoReturn``): a function that
        # returns it never returns a value at all (see :class:`EmptyType`)
        return EmptyType()
    if isinstance(value, typing.TypeVar):
        if type_vars is None or value not in type_vars:
            raise TypeError(f'cannot convert {value} to a value')
        return type_vars[value]
    if isinstance(value, typing.TypeAliasType):
        # a PEP 695 ``type X = ...`` alias used *without* subscripting: Python
        # keeps the alias object itself on the evaluated annotation, so it is
        # unwrapped to the type it stands for (a subscripted use is a generic
        # alias whose origin is the alias, handled by the ``get_origin`` cases
        # below)
        return as_value(value.__value__, ctx, type_vars)
    if isinstance(value, StructTypeApplication):
        # ``Foo[T]``: resolve the template in the host (a struct declared by
        # another context resolves to this context's copy), then its arguments
        # in the scope it was written in, then specialize it
        head = ctx.resolve_global(value.struct)
        if not isinstance(head, StructTypeHead):
            raise TypeError(f'cannot use {value.struct} as a generic struct template')
        resolved: list[AnyValue] = []
        for arg in value.generic_vars:
            arg_value = as_value(arg, ctx, type_vars)
            # a generic argument is any compile-time value: a type, or a plain
            # value a type parameter of the struct stands for
            resolved.append(arg_value)
        return head.specialize(tuple(resolved))
    if isinstance(value, TaggedUnionApplication):
        # ``A | B | ...`` written in an annotation: the items are resolved in
        # this context (a struct of another context resolves to this one's
        # copy).  A single ``None`` among the items makes it an option (the
        # ``T | None`` spelling), like the ``typing.Union`` case below
        variants: list[Type] = []
        none_count = 0
        for item in value.items:
            if item is None:
                none_count += 1
                continue
            variant = as_value(item, ctx, type_vars)
            if not isinstance(variant, Type):
                raise TypeError(f'{item!r} is not a type')
            variants.append(variant)
        if none_count > 0:
            if none_count != 1 or len(variants) != 1:
                raise TypeError(f'cannot convert {value!r} to a value')
            return OptionType(variants[0])
        return tagged_union_of(tuple(variants))
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
            elem_type = as_value(arg, ctx, type_vars)
            if not isinstance(elem_type, Type):
                raise TypeError(f'{arg!r} is not a type')
            types.append(elem_type)
        return TupleType(tuple(types), has_ellipsis)
    if typing.get_origin(value) is typing.Literal:
        # ``Literal[X]`` denotes the value ``X``: the default of a type
        # parameter (the constness of a pointer, ``C: bool = Literal[False]``)
        # evaluates to one
        args = typing.get_args(value)
        if len(args) == 1 and isinstance(args[0], (bool, int, bytes)):
            return args[0]
        raise TypeError(f'cannot convert {value!r} to a value')
    if typing.get_origin(value) in _POINTER_TYPES:
        # ``Ptr[T]``/``ConstPtr[T]``: a pointer to ``T`` that may be
        # dereferenced (a single one); ``MultiPtr[T]``/``ConstMultiPtr[T]`` the
        # same address, which may also be indexed like an array (see
        # ``PointerVariant``).  The constness is the *class*, not a type
        # argument: ``ConstPtr``/``ConstMultiPtr`` are the const spellings
        args = typing.get_args(value)
        if len(args) != 1:
            raise TypeError(f'cannot convert {value!r} to a value')
        elem = as_value(args[0], ctx, type_vars)
        if not isinstance(elem, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        origin = typing.get_origin(value)
        return PointerType(
            elem,
            origin in (syntax.ConstPtr, syntax.ConstMultiPtr),
            PointerVariant.MULTI if origin in (syntax.MultiPtr, syntax.ConstMultiPtr) else PointerVariant.SINGLE,
        )
    if typing.get_origin(value) is syntax.Array:
        # ``Array[T, L]``: ``L`` values of type ``T``.  The length is a *value*
        # (a Python ``int``, or the type parameter it is written as); the
        # constructor ``array(...)`` takes it from the number of elements it is
        # given, which the Python type system cannot express, so an annotation
        # that names the element type alone has to write the length out
        args = typing.get_args(value)
        if len(args) != 2:
            raise TypeError(f'cannot convert {value!r} to a value')
        elem = as_value(args[0], ctx, type_vars)
        if not isinstance(elem, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        if args[1] is None or args[1] is NoneType:
            # ``Array[T, None]``: an array of unknown length (a DST; Python
            # normalizes the ``None`` of a subscripted alias to ``NoneType``)
            return ArrayType(elem, None)
        return ArrayType(elem, as_value(args[1], ctx, type_vars))
    if typing.get_origin(value) is syntax.Option:
        # ``Option[T]``: ``T`` or the ``Null`` value.  The alias
        # ``type Option[T] = T | None`` evaluates a subscripted use to a
        # specialization of the alias, whose origin is the alias itself
        args = typing.get_args(value)
        if len(args) != 1:
            raise TypeError(f'cannot convert {value!r} to a value')
        child = as_value(args[0], ctx, type_vars)
        if not isinstance(child, Type):
            raise TypeError(f'{args[0]!r} is not a type')
        return OptionType(child)
    if typing.get_origin(value) is typing.Union:
        # ``A | B | ...`` or ``T | None`` written out directly.  ``None`` in the
        # union is the null value, never a type of its own: with exactly one
        # other member the union is the option of it (the ``Option[T]`` alias is
        # written that way), any other number of members makes a tagged union
        args = typing.get_args(value)
        rest = tuple(a for a in args if a is not NoneType)
        none_count = len(args) - len(rest)
        if none_count == 1 and len(rest) == 1:
            child = as_value(rest[0], ctx, type_vars)
            if not isinstance(child, Type):
                raise TypeError(f'{rest[0]!r} is not a type')
            return OptionType(child)
        if none_count == 0 and len(rest) >= 1:
            members: list[Type] = []
            for arg in rest:
                variant = as_value(arg, ctx, type_vars)
                if not isinstance(variant, Type):
                    raise TypeError(f'{arg!r} is not a type')
                members.append(variant)
            return tagged_union_of(tuple(members))
        raise TypeError(f'cannot convert {value!r} to a value')

    # anything else is asked of the host: an object it knows (a struct class
    # or a registered function handle) resolves in *this* context, so that
    # every context sees its own handle (see ``dsl._Context.resolve_global``)
    host_value = ctx.resolve_global(value)
    if host_value is not None:
        return host_value
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
    by width; elsewhere a subtype is equal to its supertype.  The solver
    tracks the bounds of a parameter separately so that ``finish`` can bind a
    parameter that only ever appears on the
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
            if isinstance(lhs, ArrayType) and isinstance(rhs, ArrayType):
                # an array constrains its element type and its length (a length
                # that is still a type parameter is solved to the value itself);
                # an unsized array (``length is None``) constrains nothing
                todo.append((lhs.elem, rhs.elem, False))
                if lhs.length is not None and rhs.length is not None:
                    todo.append((lhs.length, rhs.length, False))

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
            # the pointee is substituted; the constness is a plain ``bool``
            # (the pointer class spells it), and is kept as it is
            return PointerType(
                replace_type_vars_type(value.elem, reps),
                value.is_const,
                value.variant,
            )
        case ArrayType():
            # the length is substituted too: it may be a type parameter
            # (``Array[T, N]``); an unsized array keeps its ``None``
            return ArrayType(
                replace_type_vars_type(value.elem, reps),
                None if value.length is None else replace_type_var(value.length, reps),
            )
        case OptionType():
            return OptionType(replace_type_vars_type(value.child, reps))
        case TupleType():
            return TupleType(
                tuple(replace_type_vars_type(t, reps) for t in value.types),
                value.has_ellipsis,
            )
        case UnionType():
            return UnionType(frozenset(replace_type_vars_type(t, reps) for t in value.types))
        case TaggedUnionType():
            return tagged_union_of(tuple(replace_type_vars_type(t, reps) for t in value.types))
        case ResultType():
            return ResultType(
                replace_type_vars_type(value.return_type, reps),
                FrozenArraySet(replace_type_vars_type(t, reps) for t in value.types),
            )
        case StructType():
            # a struct type carries its type arguments: substituting into it
            # rebuilds the specialization (and keeps the identity of the one
            # the head caches, so equal references stay equal).  An argument is
            # any compile-time value: a type, or a plain one a type parameter
            # stands for
            args: list[AnyValue] = []
            for arg in value.generic_args:
                args.append(replace_type_var(arg, reps))
            if all(new is old for new, old in zip(args, value.generic_args)):
                return value
            return value.head.specialize(tuple(args))
        case _:
            return value

def replace_type_vars_type(type: Type, reps: Mapping[TypeVar, AnyValue]) -> Type:
    ret = replace_type_var(type, reps)
    assert isinstance(ret, Type)
    return ret

def replace_type_vars_value(value: AnyValue | None, reps: Mapping[TypeVar, AnyValue]) -> AnyValue | None:
    """A compile-time value (a struct field's default, e.g.) with the type
    parameters ``reps`` names replaced by their type arguments; ``None`` - a
    field that has no default - stays ``None``."""
    if value is None:
        return None
    return replace_type_var(value, reps)

def is_numeric_type(type: Type):
    match type:
        case AnyIntType() | IntType() | FloatType():
            return True
        case _:
            return False

def coerce_const_tagged_union(value: AnyValue, type: TaggedUnionType) -> AnyValue:
    """A constant of the tagged union ``type``: the value of the variant it
    belongs to under that variant's tag.  A ``TaggedUnionValue`` of a compatible
    union (a subset/superset) is remapped to ``type``'s tag."""
    if isinstance(value, TaggedUnionValue):
        if not value.type.is_subtype_of(type):
            raise CompileError(f"cannot use {value!r} as a constant of {type}")
        index = type.variant_index(value.type.types[value.index])
        assert index is not None
        return TaggedUnionValue(type, index, coerce_const(value.value, type.types[index]))
    value_type = type_of(value)
    index = None if value_type is None else type.variant_index_for(value_type)
    if index is None:
        # an untyped literal (a plain Python int, say) takes the first variant it
        # fits, like a store through the interpreter does
        for candidate, variant in enumerate(type.types):
            try:
                coerced = coerce_const(value, variant)
            except CompileError:
                continue
            return TaggedUnionValue(type, candidate, coerced)
        raise CompileError(f"cannot use {value!r} as a constant of {type}")
    return TaggedUnionValue(type, index, coerce_const(value, type.types[index]))


def coerce_const(value: AnyValue, type: Type) -> AnyValue:
    """Turn a Python value into the typed spy value of the spy type
    ``type`` (an ``Int``/``Float``/``Void``/``Type``/``bool``, or the value
    itself for the untyped type of an integer literal); the interpreter
    builds the MIR constant from it later."""
    if isinstance(value, AsValue):
        value = value.value
    if isinstance(value, UntypedUndefined):
        # the undefined literal is the value of any type: of a zero-sized type it
        # is that type's unit value, of any other the typed undefined of it
        unit = type.get_unit_value()
        return unit if unit is not None else Undefined(type)
    if isinstance(value, Undefined):
        # an *undetermined* value - the value of a place that holds nothing yet,
        # or of one that holds no storage at all - is a value of any type, so
        # the coercion only re-tags it with the type of its new location
        return Undefined(type)
    if isinstance(value, (Int, Float)):
        # an already-typed constant (the value a compile-time location holds):
        # the target type governs, like the constant of any other location
        value = value.value
    match type:
        case BoolType():
            if not isinstance(value, bool):
                raise CompileError(f"cannot use {value!r} as a bool constant")
            return value
        case BytesType():
            if not isinstance(value, bytes):
                raise CompileError(f"cannot use {value!r} as a bytes constant")
            return value
        case AnyIntType():
            # the type of an untyped integer literal: its values are the plain
            # Python ints the source wrote, so the literal itself is what a
            # location that may hold a compile-time value (a ``Comptime``
            # variable, an expression temporary) keeps.  The type has no runtime
            # representation, so a *runtime* location of it is rejected instead
            # (see ``_no_runtime_type``).
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompileError(
                    f"cannot use {value!r} as an untyped integer constant"
                )
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
        case AnyType():
            # an ``Any`` value takes the value as it is: it names no fixed type
            return value
        case OptionType():
            # a value of an option is either the null value, or a value of the
            # child type (which the interpreter then tags as present).  A
            # *typed* null has to be the absent value of this very option
            if isinstance(value, TypedNull):
                if value.child != type.child:
                    raise CompileError(
                        f"cannot use {value!r} as a constant of {type}"
                    )
                return value
            if isinstance(value, Null):
                return value
            return coerce_const(value, type.child)
        case TaggedUnionType():
            return coerce_const_tagged_union(value, type)
        case UnionType():
            # a union value carries no variant (the error code next to the
            # payload is the tag), and a constant of it exists only for a
            # union without storage: any such value is the unit value.  A
            # union *value* is never converted to another union - the storage
            # is reinterpreted through a pointer instead (see ``interp.store``)
            if type.get_unit_value() is None:
                raise CompileError(f"cannot create a constant of the union {type}")
            if isinstance(value, UnionValue):
                return UnionValue(type)
            raise CompileError(f"cannot use {value!r} as a constant of the union {type}")
        case VoidType():
            # the void type's unit value is what ``None`` used to be: a null
            # value converts to it (``NullType`` is a subtype of ``VoidType``)
            if isinstance(value, (Void, Null)):
                return Void()
            raise CompileError(f"cannot use {value!r} as a void constant")
        case _ if type.classify() == SpecialTypeKind.DST:
            # a dynamically-sized type has no value of its own to build one from
            raise CompileError(
                f"cannot create a constant of the dynamically-sized type {type}"
            )
        case _:
            raise CompileError(
                f"cannot create a constant of type {type} from {value}"
            )

class CompileContext:
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

    def target_info(self) -> TargetInfo:
        ...

    def mir_cache(self) -> MirLowerCache:
        ...

    def special_types(self) -> SpecialTypes:
        ...

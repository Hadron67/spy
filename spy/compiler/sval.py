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

import functools
import math
import struct
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from types import NoneType
from typing import Any, Literal, TypeGuard, cast, override

from . import mir, syntax
from .errors import CoerceError, CompileError
from .target import TargetInfo
from .util import FrozenArraySet, IdentityObj, IndexedMap, TriState, frozendict

if typing.TYPE_CHECKING:
    from .fn import ArgList, FunctionValue, RawArgList

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

type AnyValue = Value | int | float | complex | bytes | str | bool

def is_sval(val: Any) -> TypeGuard[AnyValue]:
    """Whether ``val`` is a compile-time spy value - one of the forms of
    ``AnyValue`` (an ``sval.Value`` such as a scalar snapshot, a type
    descriptor or an ``sval.ComptimePtr``, or a plain Python scalar).  The
    interpreter holds such values directly; its own runtime-side values
    (``RuntimeVal``, ``PendingSlot``) are not ``sval`` values."""
    return isinstance(val, (Value, int, float, complex, bytes, str, bool))

type ScalarValue = (
    Type | Void | Null | Int | Float | IntEnumValue | UndefinedScalar
    | UntypedUndefined | DeclareFunction | BoundMethod | BuiltinFn | PyFunction
    | ComptimePtr | CastedPtr | FunctionValue
    | int | float | complex | bytes | str | bool
)
"""The value union of the scalar types: every compile-time value that may
occupy one cell of a ``Box`` (see :func:`scalar_leaves`).  The values of the
aggregate types - ``AggregateValue``, ``TaggedUnionValue``, ``UnionValue``,
``OptionValue``, ``TypedNull``, ``UndefinedAggregate`` - are not scalar."""

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
        self._complex_mirs: dict[ComplexType, mir.StructType] = {}
        # a ``TupleType``/``StrDictType`` mirrors to an equivalent synthesised
        # ``StructType`` (see ``TupleType.to_mir_type``); the synthesised struct
        # is cached here so that its layout is interned and the aggregate
        # element accessors can be reached through it
        self._tuple_structs: dict[TupleType, StructType] = {}
        self._strdict_structs: dict[StrDictType, StructType] = {}

    def tuple_struct(self, type: TupleType) -> StructType:
        """The synthesised ``StructType`` a ``TupleType`` mirrors to: positional
        fields, in element order (created once and cached)."""
        ret = self._tuple_structs.get(type)
        if ret is None:
            head = StructTypeHead('tuple')
            for index, elem in enumerate(type.types):
                head.add_field(f'_{index}', elem)
            ret = head.specialize(())
            self._tuple_structs[type] = ret
        return ret

    def strdict_struct(self, type: StrDictType) -> StructType:
        """The synthesised ``StructType`` a ``StrDictType`` mirrors to: one
        named field per entry, in entry order (created once and cached)."""
        ret = self._strdict_structs.get(type)
        if ret is None:
            head = StructTypeHead('dict')
            for name, elem in type.values.items():
                head.add_field(name, elem)
            ret = head.specialize(())
            self._strdict_structs[type] = ret
        return ret

    def union_mir(self, type: UnionType, payload: mir.Type) -> mir.UnionType:
        """The (interned) MIR mirror of the payload union ``type``."""
        ret = self._union_mirs.get(type)
        if ret is None:
            ret = mir.UnionType(f'union[{payload}]', payload)
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
                    f'option[{child}]',
                    (mir.FormalArg('tag', mir.BoolType()),),
                    child.fam_mir_type(self),
                )
            else:
                assert child_mir is not None
                ret = mir.StructType(
                    f'option[{child_mir}]',
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
            ret = mir.StructType(f'tagged_union[{payload}]', (
                mir.FormalArg('tag', tag),
                mir.FormalArg('payload', payload),
            ))
            self._tagged_union_mirs[type] = ret
        return ret

    def complex_mir(self, type: ComplexType) -> mir.StructType:
        """The (interned) MIR mirror of the complex type ``type``: a struct of
        its real and imaginary parts (both of the element float)."""
        ret = self._complex_mirs.get(type)
        if ret is None:
            elem = mir.FloatType(type.elem.bits)
            ret = mir.StructType(f'complex[{elem}]', (
                mir.FormalArg('real', elem),
                mir.FormalArg('imag', elem),
            ))
            self._complex_mirs[type] = ret
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
        """The *peer type* of this type and ``other``: the type a value of
        either one converts to, or ``None`` when the two have no common type.
        It is the rule a location of no declared type is typed by, folding left
        to right over the types of the values written into it (see
        ``interp.PendingSlot`` and ``interp._common_element_type``), and what
        ``std.core.lub_type`` reports.

        The two *bottom* types (:class:`UndefinedType`, the type of
        ``undefined``, and :class:`EmptyType`, the type of a location that is
        never written) are the identity: a value of one is a value of any type.
        ``null`` (:class:`NullType`) makes the other type optional, an
        :class:`OptionType` peers through its child, and a variant peers with
        the :class:`TaggedUnionType` that holds it.  Two ordinary types peer to
        the one that converts to the other, in whichever direction that is.

        The rule is commutative for every type except :class:`ResultType`,
        whose peer keeps its own return type and puts its own exceptions first
        (see :meth:`ResultType.resolve_peer_type`); a caller must therefore fold
        in a fixed, documented order."""
        if isinstance(other, (UndefinedType, EmptyType)):
            # a bottom: a value of it is a value of any type, so the other type
            # wins
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
        if self.is_subtype_of(other):
            return other
        if other.is_subtype_of(self):
            # the reverse direction: a value of ``other`` converts to this one
            return self
        return None

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
        """Whether a value of this type may be copied: loaded out of a place,
        passed by value, or returned by value.  A dynamically-sized type has no
        value of its own - only a pointer to it is a value - so it is never
        copyable; a container or struct override decides the rest (an aggregate
        is copyable only when its members are)."""
        return self.classify() != SpecialTypeKind.DST

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

    def __call__(self, value: Any) -> AnyValue:
        """Build the typed spy value of ``value`` at the Python calling
        boundary: the explicit-type spelling ``T(value)`` a caller uses where an
        argument's type has to be named (``u64(2)`` for a plain Python ``int``,
        which marshals to ``i64`` by default).  A scalar is concretized to the
        compile-time constant ``coerce_const`` builds; a pointer (or a
        dynamically-sized function value) is the raw address it carries (see
        ``glue.pointer_value``)."""
        if isinstance(self, PointerType) or self.classify() == SpecialTypeKind.DST:
            from . import glue
            return cast(AnyValue, glue.pointer_value(value, self))
        return coerce_const(value, self)


class ScalarType(Type):
    """Base of the *scalar* spy types: the types whose values are the atomic
    compile-time values - integers, floats, booleans, bytes, type values,
    function values, pointers, ....  A value of a scalar type occupies exactly
    one cell when a compile-time aggregate is flattened into a ``Box`` (see
    ``scalar_count``), including the zero-sized scalar types (whose value is
    the unit value)."""


class AggregateType(Type):
    """Base of the *aggregate* spy types: structs, arrays, tuples, string
    dictionaries, complex numbers, options, tagged unions and payload unions.
    A value of an aggregate type is flattened into one cell per scalar leaf
    when it is stored in a ``Box`` (see ``scalar_count``)."""


@dataclass(frozen=True)
class TypeType(ScalarType):
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
class AnyType(ScalarType):
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

class TypeVar(ScalarType):
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
class TupleType(AggregateType):
    """A Python ``tuple[T1, T2, ...]``: an aggregate of its elements, laid
    out like a struct of the same field types (a tuple mirrors to a
    synthesised struct with one positional field per element, see
    ``MirLowerCache.tuple_struct``, and maps its element indices onto the
    mirror through ``get_field_mir_indices``).  A tuple whose elements all
    have a runtime representation is therefore an ordinary runtime
    aggregate: a location can hold it and a value of it can be copied as a
    whole (when every element is copyable, see ``is_copyable``).  An element
    with no runtime representation of its own (a type, a ``bytes``, a type
    variable) - and likewise the varying-length shape ``tuple[T, ...]`` -
    leaves the whole tuple a compile-time-only value (see ``classify``).

    Two positions treat a tuple specially: a parameter annotation passes it
    flattened - one argument per element, which the callee rebinds into one
    place tree (see ``fn.CompoundArgNode``) - and a return annotation is a
    multiple return value, whose elements a function delivers separately
    (see ``make_ret_spec``)."""

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
    def get_unit_value(self) -> AnyValue | None:
        """The unit value of a zero-sized tuple: the aggregate of its elements'
        unit values (a tuple is zero-sized when they all are, like a struct).  A
        varying-length tuple has no fixed layout, so it has none."""
        if self.has_ellipsis:
            return None
        values: list[AnyValue] = []
        for type in self.types:
            val = type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

    @override
    def classify(self) -> SpecialTypeKind:
        if self.has_ellipsis:
            # a varying-length tuple has no fixed runtime layout
            return SpecialTypeKind.COMPTIME
        dst = False
        zst = True
        for t in self.types:
            match t.classify():
                case SpecialTypeKind.COMPTIME:
                    return SpecialTypeKind.COMPTIME
                case SpecialTypeKind.DST:
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
    def is_copyable(self) -> bool:
        return super().is_copyable() and all(t.is_copyable() for t in self.types)

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if self.classify() == SpecialTypeKind.COMPTIME:
            return None
        # a tuple mirrors to an equivalent synthesised struct (see
        # ``MirLowerCache.tuple_struct``), reusing every struct layout rule
        return cache.tuple_struct(self).to_mir_type(cache)

    def get_field_mir_indices(self, cache: MirLowerCache) -> tuple[int | None, ...]:
        return cache.tuple_struct(self).get_field_mir_indices(cache)

    def mirror_is_a_field(self, cache: MirLowerCache) -> bool:
        return cache.tuple_struct(self).mirror_is_a_field(cache)

    def __str__(self) -> str:
        return f'tuple[{", ".join(str(t) for t in self.types)}{", ..." if self.has_ellipsis else ""}]'


@dataclass(frozen=True)
class UnionType(AggregateType):
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
    def is_copyable(self) -> bool:
        return super().is_copyable() and all(t.is_copyable() for t in self.types)

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
class TaggedUnionType(AggregateType):
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
    def is_copyable(self) -> bool:
        return super().is_copyable() and all(t.is_copyable() for t in self.types)

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
        if isinstance(other, (UndefinedType, EmptyType)):
            return self
        if isinstance(other, TaggedUnionType):
            # the union with the smaller variant set is the peer type: a value
            # of it is a value of the larger one as well (with a remapped tag)
            if all(variant in other.types for variant in self.types):
                return other
            if all(variant in self.types for variant in other.types):
                return self
            return None
        if isinstance(other, (NullType, OptionType)):
            # ``null`` is never one of the variants, not even of a union that
            # holds ``void``: the union peers to its option instead (which is
            # what ``NullType.resolve_peer_type`` does as well)
            return super().resolve_peer_type(other)
        if self.variant_index_for(other) is not None:
            # a value of one of the variants peers with the union itself
            return self
        return super().resolve_peer_type(other)

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
    variant ``index`` (see :func:`coerce_const`).  It is what the interpreter
    holds for a fully compile-time tagged union, and what the type rules can
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
        this is only reached for the parts of a signature.

        Note that this is deliberately *not* commutative (nor associative): the
        peer keeps the *left* operand's return type and puts its exceptions
        first, which is the delivery order the caller folds in.  Every other
        type peers symmetrically, so this is the one exception to the rule
        ``a.resolve_peer_type(b) == b.resolve_peer_type(a)`` (see
        :meth:`Type.resolve_peer_type`)."""
        if isinstance(other, (UndefinedType, EmptyType)):
            return self
        if not isinstance(other, ResultType):
            return super().resolve_peer_type(other)
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
class StrDictType(AggregateType):
    values: frozendict[str, Type]

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return tuple(self.values.values())

    @override
    def get_unit_value(self) -> AnyValue | None:
        """The unit value of a zero-sized string dictionary: the aggregate of
        its entries' unit values (a ``dict[str, T]`` is zero-sized when all of
        them are, like a struct)."""
        values: list[AnyValue] = []
        for type in self.values.values():
            val = type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

    @override
    def classify(self) -> SpecialTypeKind:
        dst = False
        zst = True
        for t in self.values.values():
            match t.classify():
                case SpecialTypeKind.COMPTIME:
                    return SpecialTypeKind.COMPTIME
                case SpecialTypeKind.DST:
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
    def is_copyable(self) -> bool:
        return super().is_copyable() and all(t.is_copyable() for t in self.values.values())

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        if self.classify() == SpecialTypeKind.COMPTIME:
            return None
        return cache.strdict_struct(self).to_mir_type(cache)

    def get_field_mir_indices(self, cache: MirLowerCache) -> tuple[int | None, ...]:
        return cache.strdict_struct(self).get_field_mir_indices(cache)

    def mirror_is_a_field(self, cache: MirLowerCache) -> bool:
        return cache.strdict_struct(self).mirror_is_a_field(cache)

    def __str__(self) -> str:
        entries = ', '.join(f'{k}: {v}' for k, v in self.values.items())
        return '{' + entries + '}'

TYPE_TYPE = TypeType()

@dataclass(frozen=True)
class BytesType(ScalarType):
    """A compile-time byte string: the spy type of a ``bytes`` literal/value.
    A byte string only exists at compile time - it has no MIR mirror of its own
    - and lowers into a runtime pointer/slice automatically (``sstr``, or a
    ``ptr_cast`` to a pointer type; see ``interp``).  Its compile-time value is
    the raw Python ``bytes`` object."""

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
class StrType(ScalarType):
    """A compile-time text string: the spy type of a ``str`` literal/value.
    Like a byte string a text string only exists at compile time - it has no MIR
    mirror of its own - and lowers into a runtime pointer/slice automatically
    (``sstr``, or a ``ptr_cast`` to a pointer type; see ``interp``), encoding to
    UTF-8 on the way into memory.  Its compile-time value is the raw Python
    ``str`` object; its length is the number of characters (the encoded byte
    length is ``std.core.encode_length``)."""

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
        return 'str'

@dataclass(frozen=True)
class BoolType(ScalarType):
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
class EmptyType(ScalarType):
    """The *empty* type: no value of it exists at all, so a location of it
    (a function's result location of a body that never delivers a result) can
    never be written.  It is what marks a function that never returns a value:
    with no exceptions either it can never return at all (a ``mir.NoReturn``
    function), and one that raises delivers its errors through the codes
    ``0..len(types) - 1`` alone - it carries no "no error" code (see
    :class:`ResultType`).

    For the machinery of the compiler it behaves like a zero-sized type: it
    has no mirror of its own and the value a slot of it holds is the "no
    value" marker ``Void()``.

    It is a *bottom* of the peer rule as well, exactly like
    :class:`UndefinedType`: paired with any type it is that type (see
    :meth:`Type.resolve_peer_type`), which is what makes it the identity the
    fold in ``TypeVarSolver.finish`` starts from."""

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
class VoidType(ScalarType):
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
class NullType(ScalarType):
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
        # Null is the absent value: it peers with an option (it is one of its
        # values) and with any other type by making it optional.  It does *not*
        # peer to the void type: ``None`` is still usable where a void value is
        # expected, but that is a coercion (see ``coerce_const``), not this
        # rule - peering the two would make the peer of ``null`` and a tagged
        # union holding ``void`` ambiguous
        if isinstance(other, (UndefinedType, EmptyType)):
            return self
        match other:
            case NullType() | OptionType():
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
    option value is either this - absent - or an :class:`OptionValue` - present;
    a compile-time aggregate field of an option type holds one of the two (see
    ``std.slice``).

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
class OptionType(AggregateType):
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
        if isinstance(other, (UndefinedType, EmptyType)):
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
        return super().is_copyable() and self.child.is_copyable()

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
    counters), so nesting costs no Python stack.

    A zero-sized node holds no storage and so holds no usable pointer: it is
    not descended into.  (A zero-length array still reports its element as a
    child even though it has no elements, so descending would otherwise treat
    a pointer that has no storage as a tag.)"""
    pointers = 0
    claimed = shift
    todo: list[tuple[Type, tuple[int, ...]]] = [(type, ())]
    while todo:
        current, pos = todo.pop()
        if current.classify() != SpecialTypeKind.NONE:
            # ZST has no storage, DST may not have storage (length of a FAM might be zero), so it cannot hold a pointer and
            # an option may use as its tag: do not descend into it
            continue
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


def tagged_union_shape(
    type: TaggedUnionType, cache: MirLowerCache,
) -> Literal['single', 'tag_only', 'tag_payload']:
    """How the tagged union ``type`` is represented (see
    ``TaggedUnionType.to_mir_type``): ``'single'`` (a single variant, which is
    represented as the variant itself), ``'tag_only'`` (every variant is
    zero-sized, so only the tag is stored) or ``'tag_payload'`` (a struct of
    the tag and the payload union)."""
    if len(type.types) == 1:
        return 'single'
    if type.payload_type().to_mir_type(cache) is None:
        return 'tag_only'
    return 'tag_payload'


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
    if isinstance(type, ComplexType):
        # the real and the imaginary part
        return 2
    raise CompileError(f'{type} is not an aggregate')


def _widest_variant_by_cells(types: frozenset[Type] | tuple[Type, ...] | FrozenArraySet[Type]) -> Type | None:
    """The variant that occupies the most cells when flattened (``None`` for an
    empty variant set), i.e. the variant that widens the payload union."""
    best: Type | None = None
    best_count = -1
    for variant in types:
        count = len(scalar_leaves(variant))
        if count > best_count:
            best = variant
            best_count = count
    return best


@functools.lru_cache(maxsize=2048)
def scalar_leaves(type: Type) -> tuple[ScalarType, ...]:
    """The scalar *leaf* types of ``type``, in flattened (depth-first) order:
    the cells a value of ``type`` occupies in a :class:`Box`.  A scalar type
    occupies exactly one cell (including a zero-sized scalar type); an aggregate
    is flattened by its structure - a struct by its fields (an ``is_ref``
    field's cell is the ``*const T`` pointer it stores), an array/tuple by its
    elements, a string dictionary by its entries, a complex by ``(re, im)``, an
    option by ``[tag, payload...]`` and a union/tagged union by its widest
    variant (with the tag cell in front for a tagged union).  Iterative (an
    explicit stack), so nesting costs no Python stack."""
    out: list[ScalarType] = []
    stack: list[Type] = [type]
    while stack:
        current = stack.pop()
        if isinstance(current, ScalarType):
            out.append(current)
            continue
        if isinstance(current, StructType):
            children: list[Type] = []
            for field in current.fields().values():
                children.append(
                    PointerType(field.type, is_const=True) if field.is_ref else field.type
                )
            stack.extend(reversed(children))
            continue
        if isinstance(current, TupleType):
            if current.has_ellipsis:
                raise CompileError('a varying-length tuple has no fixed flattened layout')
            stack.extend(reversed(current.types))
            continue
        if isinstance(current, ArrayType):
            length = current.length_int
            if length is None:
                raise CompileError(f'cannot flatten the unsized array {current}')
            stack.extend([current.elem] * length)
            continue
        if isinstance(current, StrDictType):
            stack.extend(reversed(list(current.values.values())))
            continue
        if isinstance(current, ComplexType):
            stack.append(current.elem)
            stack.append(current.elem)
            continue
        if isinstance(current, OptionType):
            stack.append(current.child)
            stack.append(BoolType())
            continue
        if isinstance(current, TaggedUnionType):
            # a tagged union is always ``[tag, payload...]``, a single variant
            # included (the tag still occupies one cell, even when ``tag_bits``
            # is 0)
            stack.append(current.payload_type())
            stack.append(current.tag_type())
            continue
        if isinstance(current, UnionType):
            widest = _widest_variant_by_cells(current.types)
            if widest is None:
                continue
            # FIXME: recursion?
            leaves = scalar_leaves(widest)
            width = _union_cell_width(current.types)
            out.extend(leaves)
            out.extend([VoidType()] * (width - len(leaves)))
            continue
        raise CompileError(f'cannot flatten {current} into compile-time storage')
    return tuple(out)


def _union_cell_width(types: frozenset[Type] | tuple[Type, ...] | FrozenArraySet[Type]) -> int:
    """The cell width of a payload union: the width of its widest variant."""
    width = 0
    for variant in types:
        width = max(width, len(scalar_leaves(variant)))
    return width


def scalar_count(type: Type) -> int:
    """The number of cells a value of ``type`` occupies in a ``Box`` (see
    :func:`scalar_leaves`)."""
    return len(scalar_leaves(type))


def field_scalar_offset(type: AggregateType, index: int) -> int:
    """The offset of the ``index``-th field/element/entry of the aggregate
    ``type`` in the flattened cell sequence (see :func:`scalar_leaves`).  For an
    option a tagged union, index ``0`` is the tag and the payload starts at
    ``1``; for a payload union index ``0`` is the payload itself (the tag next
    to it is not stored in a ``Box``)."""
    if isinstance(type, StructType):
        offset = 0
        for i, field in enumerate(type.fields().values()):
            if i == index:
                return offset
            offset += 1 if field.is_ref else scalar_count(field.type)
    elif isinstance(type, TupleType):
        return sum(scalar_count(t) for t in type.types[:index])
    elif isinstance(type, ArrayType):
        return index * scalar_count(type.elem)
    elif isinstance(type, StrDictType):
        return sum(scalar_count(t) for t in list(type.values.values())[:index])
    elif isinstance(type, ComplexType):
        return index
    elif isinstance(type, (OptionType, TaggedUnionType)):
        return 0 if index == 0 else 1
    elif isinstance(type, UnionType):
        return 0
    raise CompileError(f'{type} has no field {index}')


def is_scalar_value(value: AnyValue) -> TypeGuard[ScalarValue]:
    """Whether ``value`` is one of the value forms a scalar type may hold (see
    :data:`ScalarValue`): every compile-time value except the aggregate value
    forms - an aggregate, an option, a tagged union, a payload union, the typed
    null, the undefined aggregate and the error-union success value."""
    return not isinstance(value, (
        AggregateValue, OptionValue, TaggedUnionValue,
        UnionValue, TypedNull, UndefinedAggregate, Success,
    ))


def flatten_natural(value: AnyValue, type: Type) -> list[ScalarValue]:
    """The natural (unpadded) flattened cells of the compile-time ``value`` of
    ``type`` (see :func:`flatten_value`)."""
    if isinstance(type, ScalarType):
        if not is_scalar_value(value):
            raise CompileError(f'cannot flatten {value!r} as {type}')
        return [value]
    if isinstance(value, AggregateValue):
        values = value.values
    else:
        raise CompileError(f'cannot flatten {value!r} as {type}')
    if isinstance(type, StructType):
        cells: list[ScalarValue] = []
        for index, field in enumerate(type.fields().values()):
            if field.is_ref:
                # the cell holds the compile-time pointer itself
                cell = values[index]
                assert is_scalar_value(cell)
                cells.append(cell)
            else:
                cells.extend(flatten_value(values[index], field.type))
        return cells
    if isinstance(type, TupleType):
        cells = []
        for index, elem in enumerate(type.types):
            cells.extend(flatten_value(values[index], elem))
        return cells
    if isinstance(type, StrDictType):
        cells = []
        for index, elem in enumerate(type.values.values()):
            cells.extend(flatten_value(values[index], elem))
        return cells
    if isinstance(type, ArrayType):
        cells = []
        for elem_value in values:
            cells.extend(flatten_value(elem_value, type.elem))
        return cells
    if isinstance(type, ComplexType):
        re, im = values[0], values[1]
        assert is_scalar_value(re) and is_scalar_value(im)
        return [re, im]
    raise CompileError(f'cannot flatten {value!r} as {type}')


def flatten_value(value: AnyValue, type: Type) -> list[ScalarValue]:
    """The flattened cell sequence of the compile-time ``value`` of ``type``
    (see :func:`scalar_leaves`), padded to the type's width with an
    ``UndefinedScalar`` of each remaining leaf type."""
    if isinstance(value, (UndefinedAggregate, UndefinedScalar, UntypedUndefined)):
        # an undefined value flattens to undefined cells (one per leaf)
        return [UndefinedScalar(leaf) for leaf in scalar_leaves(type)]
    if isinstance(type, OptionType):
        if isinstance(value, OptionValue):
            cells: list[ScalarValue] = [True]
            cells.extend(flatten_value(value.value, type.child))
        elif isinstance(value, (Null, TypedNull)):
            cells = [False]
        else:
            raise CompileError(f'cannot flatten the option value {value!r}')
    elif isinstance(type, TaggedUnionType):
        if not isinstance(value, TaggedUnionValue):
            raise CompileError(f'cannot flatten the tagged union value {value!r}')
        cells = [Int(value.index, type.tag_type())]
        cells.extend(flatten_value(value.value, type.types[value.index]))
    else:
        cells = flatten_natural(value, type)
    leaves = scalar_leaves(type)
    while len(cells) < len(leaves):
        cells.append(UndefinedScalar(leaves[len(cells)]))
    if len(cells) != len(leaves):
        raise CompileError(f'cannot flatten {value!r} as {type}')
    return cells


def unflatten_cells(cells: list[ScalarValue], offset: int, type: Type) -> AnyValue:
    """Reconstruct a compile-time value of ``type`` from the flattened cells at
    ``offset`` (the inverse of :func:`flatten_value`)."""
    if isinstance(type, ScalarType):
        return cells[offset]
    if isinstance(type, StructType):
        values: list[AnyValue] = []
        for index, field in enumerate(type.fields().values()):
            field_offset = offset + field_scalar_offset(type, index)
            if field.is_ref:
                values.append(cells[field_offset])
            elif field.is_comptime:
                cell = cells[field_offset]
                # an unset compile-time field reads its default
                if isinstance(cell, UndefinedScalar) and field.default is not None:
                    values.append(field.default)
                else:
                    values.append(cell)
            else:
                values.append(unflatten_cells(cells, field_offset, field.type))
        return AggregateValue(tuple(values), type)
    if isinstance(type, TupleType):
        return AggregateValue(
            tuple(
                unflatten_cells(cells, offset + field_scalar_offset(type, index), elem)
                for index, elem in enumerate(type.types)
            ),
            type,
        )
    if isinstance(type, StrDictType):
        return AggregateValue(
            tuple(
                unflatten_cells(cells, offset + field_scalar_offset(type, index), elem)
                for index, elem in enumerate(type.values.values())
            ),
            type,
        )
    if isinstance(type, ArrayType):
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot flatten the unsized array {type}')
        step = scalar_count(type.elem)
        return AggregateValue(
            tuple(
                unflatten_cells(cells, offset + index * step, type.elem)
                for index in range(length)
            ),
            type,
        )
    if isinstance(type, ComplexType):
        return AggregateValue((cells[offset], cells[offset + 1]), type)
    if isinstance(type, OptionType):
        if cells[offset] is True:
            return OptionValue(unflatten_cells(cells, offset + 1, type.child))
        return TypedNull(type.child)
    if isinstance(type, TaggedUnionType):
        tag = cells[offset]
        index = tag.value if isinstance(tag, Int) else 0
        return TaggedUnionValue(
            type, index, unflatten_cells(cells, offset + 1, type.types[index]),
        )
    if isinstance(type, UnionType):
        unit = type.get_unit_value()
        if unit is None:
            raise CompileError(f'cannot build a value of the union {type}')
        return unit
    raise CompileError(f'cannot unflatten {type}')


def fill_comptime_defaults(box: Box, type: Type, offset: int) -> None:
    """Fill the cell of every ``is_comptime`` struct field reachable from
    ``(type, offset)`` with that field's declared default: a box's cells start as
    the flattened undefined, so a compile-time field would otherwise read
    undefined instead of its default.  Iterative (an explicit stack), so type
    nesting costs no Python stack."""
    stack: list[tuple[Type, int]] = [(type, offset)]
    while stack:
        current, base = stack.pop()
        if isinstance(current, StructType):
            for index, field0 in enumerate(current.fields().values()):
                field_offset = base + field_scalar_offset(current, index)
                if field0.is_comptime:
                    if field0.default is not None:
                        decl = (
                            PointerType(field0.type, is_const=True)
                            if field0.is_ref else field0.type
                        )
                        cells = flatten_value(field0.default, decl)
                        box.cells[field_offset:field_offset + len(cells)] = cells
                elif not field0.is_ref:
                    stack.append((field0.type, field_offset))
        elif isinstance(current, TupleType):
            for index, elem in enumerate(current.types):
                stack.append((elem, base + field_scalar_offset(current, index)))
        elif isinstance(current, StrDictType):
            for index, elem in enumerate(current.values.values()):
                stack.append((elem, base + field_scalar_offset(current, index)))
        elif isinstance(current, ArrayType):
            length = current.length_int
            if length is None:
                continue
            step = scalar_count(current.elem)
            for index in range(length):
                stack.append((current.elem, base + index * step))


def load_box_cells(box: Box | bytes | str, offset: int, type: Type) -> AnyValue:
    """Read a compile-time value of ``type`` out of the compile-time storage
    ``box`` at ``offset`` (a byte-string store holds raw bytes, a text-string
    store a raw ``str``; the offset is a byte index in the former and a
    character index in the latter)."""
    if isinstance(box, bytes):
        if isinstance(type, BytesType):
            # a byte-string store holds the bytes themselves
            return box[offset:]
        if isinstance(type, IntType):
            return Int(box[offset], type)
        raise CompileError(f'cannot read {type} out of a byte string')
    if isinstance(box, str):
        if isinstance(type, StrType):
            # a text-string store holds the ``str`` itself
            return box[offset:]
        raise CompileError(f'cannot read {type} out of a string')
    return unflatten_cells(box.cells, offset, type)


@dataclass(frozen=True, slots=True)
class BuiltinSignature:
    """The argument shape of a spy builtin the compile-time interpreter
    evaluates (see :class:`BuiltinFn`).  A builtin is only ever called at
    compile time, on already-evaluated arguments, so - unlike a spy function's
    ``fn.Signature`` - it carries no types: the positional parameters in
    declaration order, each ``(name, default_value)`` (``None`` for a parameter
    with no default), keyed by name (see ``util.IndexedMap``), and whether the
    builtin takes ``*args``/``**kwargs``."""

    positional: IndexedMap[str, tuple[str, AnyValue | None]]
    varargs: bool = False
    kwargs: bool = False

    def bind[T](
        self, args: RawArgList[T], default_converter: Callable[[AnyValue], T]
    ) -> ArgList[T]:
        """Bind the arguments of one builtin call, exactly like
        ``fn.Signature.bind_arg_pos``: the positional arguments fill the
        declared parameters in order, the excess bind ``*args``, keyword
        arguments bind by name (an unknown one is an error unless the builtin
        takes a ``**kwargs``), and a parameter the call leaves out takes its
        default value, converted by ``default_converter``.  Every binding error
        is a :class:`CompileError`."""
        from .fn import ArgList

        n = len(self.positional.by_id)
        values: dict[int, T] = {}
        varargs_out: list[T] = []
        for i, value in enumerate(args.positional):
            if i < n:
                values[i] = value
            elif self.varargs:
                varargs_out.append(value)
            else:
                raise CompileError(
                    f'takes {n} positional argument(s) but {len(args.positional)} were given'
                )
        kwargs_out: dict[str, T] = {}
        for key, value in args.kwargs.items():
            idx = self.positional.by_key.get(key)
            if idx is not None:
                if idx in values:
                    raise CompileError(f"got multiple values for argument '{key}'")
                values[idx] = value
            elif self.kwargs:
                kwargs_out[key] = value
            else:
                raise CompileError(f"got an unexpected keyword argument '{key}'")
        bound: list[T] = []
        for i in range(n):
            if i in values:
                bound.append(values[i])
                continue
            name, default = self.positional.by_id[i]
            if default is None:
                raise CompileError(f"missing required argument '{name}'")
            bound.append(default_converter(default))
        return ArgList(tuple(bound), tuple(varargs_out), frozendict(kwargs_out))


class BuiltinFn(Value):
    """A spy builtin that the compile-time interpreter evaluates while
    running the HIR (``std.core.compile_log``).  The name identifies the
    builtin to the interpreter; ``signature`` (see :class:`BuiltinSignature`)
    is what the interpreter binds the call's arguments with before evaluating
    the builtin.  ``spy.typeof`` is a ``syntax`` marker lowered to a type probe
    rather than a builtin."""

    name: str
    signature: BuiltinSignature

    def __init__(self, name: str, signature: BuiltinSignature) -> None:
        self.name = name
        self.signature = signature

    @override
    def get_type(self) -> Type:
        return AnyFunction()

    def __str__(self) -> str:
        return f'spy.{self.name}'

@dataclass(frozen=True)
class AnyIntType(ScalarType):
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def classify(self) -> SpecialTypeKind:
        return SpecialTypeKind.COMPTIME

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return None

    @override
    def is_subtype_of(self, other: Type) -> bool:
        # the untyped integer converts to any integer type and to any float
        # type: it only ever exists at compile time (a runtime location of it is
        # rejected), so its *value*, which is what decides whether the
        # conversion is lossless, is checked where the value is coerced (see
        # ``coerce_const``)
        return isinstance(other, (AnyIntType, IntType, FloatType))

    def __str__(self) -> str:
        return 'int'

    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, (IntType, AnyIntType)):
            return other
        if isinstance(other, ComplexType):
            # an integer is a real number: the complex side owns the conversion
            return other.resolve_peer_type(self)
        return super().resolve_peer_type(other)

@dataclass(frozen=True)
class IntType(ScalarType):
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
        if isinstance(other, FloatType):
            # an integer converts to a float only when the float represents its
            # whole range exactly, so that no value of it rounds (a *value*
            # narrows by value, see ``coerce_const``; a runtime value can only
            # go by its type, see ``float_exact_int_range``)
            self_range = int_range(self)
            exact_range = float_exact_int_range(other)
            return self_range[0] >= exact_range[0] and self_range[1] <= exact_range[1]
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
            case ComplexType():
                # an integer is a real number: the complex side owns the
                # conversion
                return other.resolve_peer_type(self)
        return super().resolve_peer_type(other)

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


class IntEnumType(ScalarType, IdentityObj):
    """A spy integer enumeration: the type an ``enum.Enum`` class with integer
    member values declares.  It is *not* an integer - it never takes part in
    arithmetic, implicit conversion or comparison with an ``int`` - and its
    runtime representation is the smallest integer type that holds every member
    value (unsigned when all the values are non-negative, signed otherwise),
    unless ``@enum(...)`` names one explicitly.  A zero-width representation
    (``i0``/``u0``) makes the enumeration a ZST.

    Like a struct type it is compared by identity, so a Python enum class has
    exactly one ``IntEnumType`` in each host context (see
    ``CompileContext.int_enum_type``); the reverse mapping back to the class is
    ``CompileContext.enum_class_of``.  ``members`` holds the members in
    declaration order, as ``name -> (name, value)``."""

    def __init__(self, name: str, members: IndexedMap[str, tuple[str, int]], int_type: IntType | None = None) -> None:
        self.name = name
        self.members = members
        values = [value for _, value in members.by_id]
        if len(values) == 0:
            # an empty enumeration has no value at all, like ``EmptyType``
            raise CompileError(f'the enum {name} must have at least one member')
        if int_type is None:
            # the smallest integer type that holds every member value: unsigned
            # for non-negative values, signed as soon as one is negative (see
            # ``min_int_type``, which always reserves at least one bit)
            self.int_type = min_int_type(min(values), max(values))
        else:
            # a representation named explicitly (``@enum(...)``): every member
            # value has to fit in it
            lower, upper = int_range(int_type)
            for member_name, value in members.by_id:
                if not lower <= value <= upper:
                    raise CompileError(
                        f'the enum {name} member {member_name!r} has the value '
                        f'{value}, which does not fit in {int_type}'
                    )
            self.int_type = int_type
        # the methods of the enum class body, by name (see
        # ``dsl._Context._collect_enum_methods``): a struct-like set, except an
        # enum method's ``self`` is the value itself
        self.methods: dict[str, Any] = {}
        # the methods declared ``@staticmethod`` (they take no receiver)
        self.static_methods: set[str] = set()

    def get_method(self, name: str) -> Any | None:
        """The method ``name`` of this enumeration, or ``None`` for any other
        name."""
        return self.methods.get(name)

    def is_static_method(self, name: str) -> bool:
        """Whether the method ``name`` is a ``@staticmethod`` (it takes no
        receiver, so a call does not pass one)."""
        return name in self.static_methods

    def value_of(self, index: int) -> int:
        """The integer value of the member at ``index``."""
        return self.members.by_id[index][1]

    def index_of(self, name: str) -> int | None:
        """The index of the member named ``name``, or ``None`` for any other
        name."""
        return self.members.by_key.get(name)

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_unit_value(self) -> Value | None:
        if self.int_type.bits != 0:
            return None
        # a zero-width enumeration holds the single value ``0`` (the range check
        # leaves no other member value), so its unit value is that member
        for index, (_name, value) in enumerate(self.members.by_id):
            if value == 0:
                return IntEnumValue(self, index)
        raise AssertionError(f'the zero-width enum {self.name} has no zero member')

    @override
    def classify(self) -> SpecialTypeKind:
        return self.int_type.classify()

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return self.int_type.to_mir_type(cache)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        # an enumeration is a subtype of itself only: it is not a subtype of an
        # integer, and two different enumerations are unrelated
        return self is other

    def __repr__(self) -> str:
        return f'<spy enum {self}>'

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class IntEnumValue(Value):
    """One member of an :class:`IntEnumType`: the type and the *index* of the
    member in its declaration order (the integer value is read from the type,
    see ``IntEnumType.value_of``)."""

    type: IntEnumType
    index: int

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'{self.type}.{self.type.members.by_id[self.index][0]}'


_INT_ENUM_BASE = Enum
"""The Python base class a spy enumeration inherits: ``enum.Enum``.  ``str``
(and any non-integer) member values are rejected when the type is built (see
``int_enum_type``)."""


def is_int_enum_class(value: Any) -> TypeGuard[type]:
    """Whether ``value`` is the class of a spy enumeration (a subclass of
    ``enum.Enum``)."""
    return isinstance(value, type) and issubclass(value, _INT_ENUM_BASE)

def is_int_enum_member(value: Any) -> TypeGuard[Enum]:
    """Whether ``value`` is a member of a spy enumeration."""
    return isinstance(value, _INT_ENUM_BASE)


def int_enum_type(name: str, cls: type, int_type: IntType | None = None) -> IntEnumType:
    """The :class:`IntEnumType` the ``enum.Enum`` class ``cls`` declares: its
    members in declaration order, each with an integer value (aliases are
    skipped, as Python does).  ``int_type`` is the representation ``@enum(...)``
    named, or ``None`` for the smallest integer type that holds every value."""
    members: IndexedMap[str, tuple[str, int]] = IndexedMap()
    # iterating an ``Enum`` class yields its canonical members, in declaration
    # order (aliases are skipped, as Python does); the checker does not know
    # ``type`` is iterable here
    for member in cast(Any, cls):
        value = member.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise CompileError(
                f'the enum {name} member {member.name!r} must have an integer '
                f'value, got {value!r}'
            )
        members.add(member.name, (member.name, value))
    return IntEnumType(name, members, int_type)

@dataclass(frozen=True)
class FloatType(ScalarType):
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

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, ComplexType):
            # a real float is a complex one with a zero imaginary part: the
            # complex side owns the conversion
            return other.resolve_peer_type(self)
        return super().resolve_peer_type(other)

    def __str__(self) -> str:
        return f"f{self.bits}"

@dataclass(frozen=True)
class Float(Value):
    value: float
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.value}{self.type}"

@dataclass(frozen=True, slots=True)
class ComplexType(AggregateType):
    """A complex number type ``Complex[T]``: a value of a real part and an
    imaginary part, each of the float type ``elem``.  A real (float or integer)
    value converts to it implicitly with a zero imaginary part (the ``T ->
    Complex[T]`` conversion), and ``Complex[T]`` widens to ``Complex[T']``
    when ``T`` widens to ``T'``."""

    elem: FloatType

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

    @override
    def get_type_children(self) -> tuple[Type, ...]:
        return (self.elem,)

    @override
    def classify(self) -> SpecialTypeKind:
        # a complex number always has storage: its element type is a float and
        # never zero-sized, so it is an ordinary type with a mirror of its own
        return SpecialTypeKind.NONE

    @override
    def to_mir_type(self, cache: MirLowerCache) -> mir.Type | None:
        return cache.complex_mir(self)

    @override
    def is_subtype_of(self, other: Type) -> bool:
        """A complex type is a subtype of a complex one whose element type its
        own element converts to (the wider float)."""
        return isinstance(other, ComplexType) and self.elem.is_subtype_of(other.elem)

    @override
    def resolve_peer_type(self, other: Type) -> Type | None:
        if isinstance(other, (IntType, AnyIntType)):
            # a real integer operand is taken as a complex one of the element
            # float, with a zero imaginary part (like a float does)
            return self
        if isinstance(other, ComplexType):
            elem = self.elem.resolve_peer_type(other.elem)
            return ComplexType(elem) if isinstance(elem, FloatType) else None
        if isinstance(other, FloatType):
            # a real float operand widens the element type, and the value
            # converts to the complex one with a zero imaginary part
            return ComplexType(FloatType(max(self.elem.bits, other.bits)))
        return super().resolve_peer_type(other)

    def field_index(self, name: str) -> int | None:
        """The index of the component the field name ``name`` names: the real
        part (``real``) or the imaginary one (``imag``), or ``None`` for any
        other name."""
        if name == 'real':
            return 0
        if name == 'imag':
            return 1
        return None

    def get_field_mir_indices(self, cache: MirLowerCache) -> tuple[int | None, ...]:
        """A complex number mirrors to a struct of its real and imaginary parts,
        in field order, with no reordering (see ``MirLowerCache.complex_mir``)."""
        return (0, 1)

    def mirror_is_a_field(self, cache: MirLowerCache) -> bool:
        return False

    def __str__(self) -> str:
        return f'c{self.elem.bits * 2}'

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


def pointer_object_type(ptr: ComptimePtr) -> Type:
    """The object a compile-time pointer names, for materializing it into runtime
    storage: its pointee, or - when it names an element of an array it is the
    multi pointer of (see :func:`_points_at_elements_of`) - the array's *tail*
    from that element on, the whole array for its first element.  The tail is what
    the storage behind the address actually holds, and naming it keeps the
    remaining length: materializing only the element would leave a global (or a
    slot) holding a single element, so indexing past it would read out of
    bounds.

    The tail starts at the pointed element, so its length is the array's minus
    the element's index - the cell offset over the element's cell width (a value
    of an element is :func:`scalar_count` cells of the box)."""
    if isinstance(ptr.box, Box):
        box_type = ptr.box.type
        if isinstance(box_type, ArrayType) and box_type.elem == ptr.type.elem:
            length = box_type.length_int
            if length is not None:
                index = ptr.offset // scalar_count(box_type.elem)
                return ArrayType(box_type.elem, length - index)
    return ptr.type.elem


@dataclass(frozen=True)
class PointerType(ScalarType):
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
        return super().resolve_peer_type(other)

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


class Box(IdentityObj):
    """A compile-time *storage box*: the storage a comptime alloca
    (``syntax.Comptime``/``ConstComptime``) reserves.  Its identity is its
    object identity, so two :class:`ComptimePtr`\\ s into the same box compare
    equal exactly when they also share their ``offset`` and their view type.

    A box **flattens** the value it stores into a sequence of scalar cells
    (``cells``, see :func:`scalar_leaves`); the value of the whole object starts
    at cell ``0``.  ``type`` is the declared (or committed) spy type of the
    whole object, and ``is_const`` makes the box read-only once a
    ``ConstComptime`` commits.  A cell that was never written reads as an
    ``UndefinedScalar`` of its leaf type."""

    def __init__(self, name_base: str, type: Type | None = None, is_const: bool = False) -> None:
        self.name_base = name_base
        self.type = type
        self.is_const = is_const
        self.cells: list[ScalarValue] = []

    def reset(self, type: Type) -> None:
        """Declare ``type`` as the box's type and (re)initialize its flattened
        cells to the flattened undefined of that type."""
        self.type = type
        self.cells = [UndefinedScalar(leaf) for leaf in scalar_leaves(type)]

    def __repr__(self) -> str:
        return f'<box {self.name_base} {self.type}>'


@dataclass(frozen=True, slots=True)
class ComptimePtr(Value):
    """A pointer into compile-time storage: ``box`` (a :class:`Box`, or a raw
    ``bytes`` byte string / ``str`` text string - a read-only, cell-less store)
    together with the ``offset`` of the pointed object's first scalar in
    ``box.cells`` (``0`` is the object itself), viewed through the pointer type
    ``type``.  Equality and hashing are structural (``box`` is an identity
    object, ``bytes``/``str`` compare by content), so a ``ComptimePtr`` is its
    own specialization key: the same box, offset and view type."""

    box: Box | bytes | str
    offset: int
    type: PointerType

    @override
    def get_type(self) -> Type:
        return self.type

    def is_writable(self) -> bool:
        """Whether the pointee may be written through this pointer: both the
        view (``type.is_const``) and the box (``Box.is_const``) have to allow it
        (a ``bytes``/``str`` box is never writable)."""
        if self.type.is_const is not False:
            return False
        if isinstance(self.box, (bytes, str)):
            return False
        return not self.box.is_const

    def __str__(self) -> str:
        return f'&box[{self.offset}]({self.type.elem})'


@dataclass(frozen=True, slots=True)
class CastedPtr(Value):
    """A compile-time pointer reinterpreted by ``std.core.ptr_cast``: the
    underlying :class:`ComptimePtr` together with the pointer type it is now
    viewed through.  The pointee cannot be read or written through it - a cast
    back to a readable type is needed first (see ``interp``)."""

    ptr: ComptimePtr
    type: PointerType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'cast_ptr({self.ptr} as {self.type})'


@dataclass(frozen=True, slots=True)
class OptionValue(Value):
    """The *present* value of an ``Option[T]``: the child value ``value``.
    (The absent value is :class:`Null`/``TypedNull``.)  It is a value form - like
    :class:`AggregateValue` - and does not enter :data:`ScalarValue`."""

    value: AnyValue

    @override
    def get_type(self) -> Type:
        return OptionType(type_of(self.value))

    def __str__(self) -> str:
        return f'some({self.value})'


@dataclass(frozen=True, slots=True)
class ArrayType(AggregateType):
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
        """An array is copyable when it is sized and its element type is (an
        empty array has nothing to copy)."""
        return super().is_copyable() and (self.length == 0 or self.elem.is_copyable())

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
class OpaqueType(ScalarType):
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
class UndefinedScalar(Value):
    """The *undefined* value of a **scalar** type ``type``: the value a place
    of that type holds when nothing has been written to it yet.  It is a legal
    cell value of a ``Box``.  (A *typed* undefined - the coercion of
    ``UntypedUndefined`` - of an aggregate type is an
    :class:`UndefinedAggregate` instead.)"""

    type: ScalarType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return "undefined"


@dataclass(frozen=True, slots=True)
class UndefinedAggregate(Value):
    """The *undefined* value of an **aggregate** type ``type``: the value form
    of an aggregate that was never written.  It is not a cell value - it does
    not enter :data:`ScalarValue` - so storing it into a ``Box`` expands it by
    the type's structure and every scalar leaf cell becomes an
    ``UndefinedScalar`` of the leaf's type (see :func:`scalar_count`)."""

    type: AggregateType

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
class UndefinedType(ScalarType):
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
        if isinstance(other, EmptyType):
            # the two bottoms peer to the canonical one (``undefined``)
            return self
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
class ValueType(ScalarType):
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
class FunctionType(ScalarType):
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
    # whether the function is C-variadic (a trailing ``...``): the call passes
    # any number of extra arguments after the declared ones, each by value with
    # its own type (see ``Signature.specialize``).  Only a non-default calling
    # convention may declare it (see ``dsl._build_fn_type``)
    varargs: bool = False

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
                if not arg.type.is_copyable():
                    # the C convention forces the argument by value, which would
                    # copy it: a non-copyable value cannot be passed
                    raise CompileError(
                        f'a C function may not take the non-copyable type {arg.type}'
                    )
                mir_type = arg.type.to_mir_type(cache)
                if mir_type is None:
                    return None
                args.append(mir_type)
        ret_type: mir.MayBeVoidType = mir.VOID
        for leaf in iter_ret_leaves(self.ret_spec(cache)):
            if not isinstance(leaf, RetValue):
                # a compile-time leaf takes no MIR result slot
                continue
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
                if not leaf.type.is_copyable():
                    # only a non-default convention could force a non-copyable
                    # result by value, which would copy it
                    raise CompileError(
                        f'a C function may not return the non-copyable type {leaf.type}'
                    )
                mir_type = leaf.type.to_mir_type(cache)
                if mir_type is None:
                    return None
                ret_type = mir_type
        return mir.FunctionType(
            tuple(args), ret_type, self.callconv, self.may_panic, self.varargs,
        )

    def __str__(self) -> str:
        args = ', '.join(str(arg.type) for arg in self.args)
        if self.varargs:
            args = f'{args}, ...' if len(self.args) > 0 else '...'
        return f"fn({args}) -> {self.return_type}"


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
    # whether the field is compile-time known (Zig semantics): it does not enter
    # the MIR layout and reading it on a runtime struct gives its default value
    is_comptime: bool = False
    # whether the field is logically a reference: it stores a ``*const T`` (or
    # nothing when ``T`` is zero-sized), a read goes through an extra load and a
    # write through it is an error (see ``interp.field_index_addr``)
    is_ref: bool = False

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

    A struct head is neither a scalar nor an aggregate type: it denotes no
    value of its own (only a specialization does).
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

    def add_field(self, name: str, type: Type, default: AnyValue | None = None, is_comptime: bool = False, is_ref: bool = False) -> None:
        """Declare one field, appended after the fields declared so far, with
        the spy value a construction leaves it at when it is not given."""
        assert name not in self.fields.by_key, f'{self.name_base} already has a field {name!r}'
        self.fields.add(name, StructField(name, type, default, is_comptime, is_ref))

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

class StructType(AggregateType):
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
                    f.is_comptime,
                    f.is_ref,
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
            if field.is_comptime:
                if field.default is None:
                    return None
                values.append(field.default)
                continue
            val = field.type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

    @override
    def is_copyable(self) -> bool:
        """A struct is copyable when it is not dynamically sized and its
        ``copyable`` modifier says so; ``inherit`` (the default) means its
        fields all are."""
        if not super().is_copyable():
            return False
        match self.modifiers.copyable:
            case 'inherit':
                return all(
                    True if f.is_ref else f.type.is_copyable()
                    for f in self.fields().values()
                    if not f.is_comptime
                )
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
            if field.is_comptime:
                if field.default is None:
                    # a compile-time field with no default: the struct is
                    # compile-time-only (there is no value to store)
                    return SpecialTypeKind.COMPTIME
                # a compile-time field does not take part in the ZST decision
                continue
            if field.is_ref:
                # a reference field stores a ``*const T`` (nothing when ``T`` is
                # zero-sized): it takes part by the pointer, not by ``T``
                if field.type.is_zst():
                    continue
                if field.type.classify() == SpecialTypeKind.COMPTIME:
                    return SpecialTypeKind.COMPTIME
                zst = False
                continue
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
            if field.is_comptime:
                # a compile-time field does not enter the MIR layout at all
                continue
            if field.is_ref:
                if field.type.is_zst():
                    # the pointee holds no storage: neither does the field
                    continue
                field_mir = PointerType(field.type, is_const=True).to_mir_type(cache)
                if field_mir is None:
                    raise CompileError(
                        f"reference field '{field.name}' of {self} has no pointer "
                        f"representation"
                    )
                mirrored.append((index, field, field_mir))
                continue
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
class AnyFunction(ScalarType):
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


class PyFunction(Value):
    """A plain Python function - one *not* decorated with ``@func`` - used as a
    compile-time helper.  It is only ever called while the HIR runs (see
    ``interp``), so it has no runtime representation at all: its arguments and
    result are plain Python scalars (an ``int``/``bool``/``float``/``complex``/
    ``bytes``, and a ``str`` which crosses as its UTF-8 byte string).

    Its value is a *const function pointer* (see ``get_type``), like a
    registered function's (``fn.FunctionValue``), so a call reaches it through a
    pointer/place like any other callee.  The wrapped object is kept as it is:
    nothing about it (its signature, its annotations, ...) is read here - the
    interpreter binds the call's arguments by Python's own rules (see
    ``interp.HirRunner._call_py_function``).

    The value is not interned: two ``PyFunction`` values are equal only when they
    wrap the very same object (identity), so that reconstructing one is harmless."""

    fn: Callable[..., Any]

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyFunction) and self.fn is other.fn

    def __hash__(self) -> int:
        return object.__hash__(self.fn)

    @override
    def get_type(self) -> Type:
        # a const function pointer, like a registered function's value; its
        # signature is not a spy one, so the pointee is the untyped AnyFunction
        return PointerType(AnyFunction(), is_const=True)

    def __str__(self) -> str:
        return f'pyfn({getattr(self.fn, "__qualname__", self.fn)})'


def int_range(type: IntType) -> tuple[int, int]:
    if type.bits == 0:
        # a zero-width integer (a ZST) holds only ``0``; the signed formula would
        # produce ``(-0.5, -0.5)`` here
        return (0, 0)
    if type.signed:
        return (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
    return (0, 2 ** type.bits - 1)


def float_exact_int_range(type: FloatType) -> tuple[int, int]:
    """``[-(2**p), 2**p]``: the integers a float type represents exactly, ``p``
    being the precision of its significand (24 for ``f32``, 53 for ``f64``).
    Every integer in the range converts to the float without loss, and one
    outside it may round to a neighbour.  It is the range an integer has to
    stay within for the float conversion to be lossless: an integer *type*
    checks its whole range against it (see ``IntType.is_subtype_of``), and
    ``std.int.try_int_to_float`` uses it to decide whether the whole source
    range is exact.  A single compile-time value is checked for its own exact
    representability instead, which may be exact outside the range (see
    ``int_to_float_is_exact``)."""
    p = 53 if type.bits == 64 else 24
    return (-(1 << p), 1 << p)


def int_to_float_is_exact(value: int, type: FloatType) -> bool:
    """Whether the integer constant ``value`` converts to ``type`` without loss.

    Unlike :func:`float_exact_int_range`, which bounds the integers a float type
    represents exactly as a whole, this tests the value itself: a power of two
    far outside that range is still exact (``2**63`` in ``f64``, say).  The
    value is exact when its Python ``float`` (an ``f64``) round-trips to it and,
    for ``f32``, it also survives the rounding to the target precision."""
    try:
        as_float = float(value)
    except OverflowError:
        # beyond the largest finite ``f64`` value
        return False
    if not math.isfinite(as_float) or int(as_float) != value:
        return False
    if type.bits == 64:
        return True
    try:
        rounded = struct.unpack('<f', struct.pack('<f', as_float))[0]
    except OverflowError:
        # beyond the largest finite ``f32`` value
        return False
    return rounded == as_float

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
    value, and a non-copyable type - a dynamically-sized one (which has no
    size at all), or a struct whose copy is forbidden - is always delivered
    through a result pointer.  The size is the one of the
    type's MIR mirror - the layout the lowered code uses (see
    ``mir.estimated_size_of``) - for pointers of the target the cache
    belongs to.  A signature may override the default
    (``fn.ReturnSignature.ret_spec``)."""
    if not type.is_copyable():
        # a non-copyable type has no value to deliver by copy: it is always
        # delivered through a hidden result pointer (see also
        # ``HirRunner._ret_leaf_ptr``)
        return True
    match type:
        case StructType() | ArrayType() | TupleType() | StrDictType() | OptionType() | UnionType() | TaggedUnionType() | ComplexType():
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


@dataclass(frozen=True, slots=True)
class RetComptime:
    """A leaf result value that is fully compile-time known
    (``classify() == COMPTIME``): it takes neither a by-value slot nor a MIR
    result argument.  Its value is delivered at compile time - through the
    caller's result location for an inline call, or through a per-callee list
    for a non-inline one (see ``interp``).  ``index`` is its position among the
    :class:`RetComptime` leaves, in depth-first declaration order."""

    type: Type
    index: int


type RetSpec = RetValue | RetTuple | RetComptime


def iter_ret_leaves(spec: RetSpec) -> Iterator[RetValue | RetComptime]:
    """The leaf values of a return spec, in declaration order (depth first),
    including the :class:`RetComptime` leaves (so that a spec leaf aligns with
    one reserved place, see ``interp._deliver_result``)."""
    work: list[RetSpec] = [spec]
    while work:
        node = work.pop()
        match node:
            case RetValue() | RetComptime():
                yield node
            case RetTuple(values=values):
                work.extend(reversed(values))


def ret_returned_type(spec: RetSpec) -> Type | None:
    """The spy type of the value the lowered function returns directly (the
    by-value result), or ``None`` when it returns void - every value then goes
    through a result pointer, is zero-sized or is compile-time known."""
    for leaf in iter_ret_leaves(spec):
        if isinstance(leaf, RetValue) and not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
            return leaf.type
    return None


def ret_by_value_index(spec: RetSpec) -> int | None:
    """The position of the one leaf returned by value (its value has storage
    and fits in registers) among all the leaves, in depth-first declaration
    order, or ``None`` when the lowered function returns void.  The count
    includes the :class:`RetComptime` leaves (a compile-time leaf takes no
    by-value slot)."""
    for index, leaf in enumerate(iter_ret_leaves(spec)):
        if isinstance(leaf, RetValue) and not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
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
        if leaf_type.classify() == SpecialTypeKind.COMPTIME:
            # a compile-time leaf is delivered at compile time: it takes no
            # by-value slot
            continue
        if leaf_type.get_unit_value() is not None:
            # a zero-sized value is delivered as its unit value, not
            # through the by-value slot
            continue
        if force_by_value or not returns_via_result_ptr(leaf_type, cache):
            chosen = index
            break
    via = tuple(
        leaf_type.classify() != SpecialTypeKind.COMPTIME
        and leaf_type.get_unit_value() is None
        and index != chosen
        for index, leaf_type in enumerate(leaves)
    )
    # rebuild the tree of the annotation, marking every leaf with how it is
    # delivered (a ``None`` on the work stack closes the group it opened; the
    # whole type is itself a group when it is a ``tuple[...]`` or a
    # ``ResultType[...]``)
    index = 0
    comptime_index = 0
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
                if item.classify() == SpecialTypeKind.COMPTIME:
                    groups[-1].append(RetComptime(item, comptime_index))
                    comptime_index += 1
                else:
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
    dynamically-sized type (a function type) has no size to pass, and a
    non-copyable type has no copy to make, so both are always passed as a
    pointer.

    A type that still names a type parameter has no layout to size, so the
    answer is ``UNKNOWN``: the convention is settled when a call substitutes
    the parameter (see ``fn.Signature.specialize``, which combines the
    answer with what the formal declares)."""
    if _mentions_type_var(type):
        # the layout is not known until a call substitutes the type parameter:
        # left to the specialization to decide
        return TriState.UNKNOWN
    if not type.is_copyable():
        # a non-copyable value has no copy to make: it is always passed as a
        # pointer, like a dynamically-sized one
        return TriState.TRUE
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
        case complex():
            return ComplexType(FloatType(64))
        case bytes():
            return BytesType()
        case str():
            return StrType()

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
    if isinstance(value, IntEnumValue):
        return value
    if isinstance(value, _INT_ENUM_BASE) or is_int_enum_class(value):
        # a spy enumeration: one of its members (the typed member value) or the
        # class itself (the ``IntEnumType``).  The host resolves both, so that
        # the object is resolved in the resolving context (see
        # ``CompileContext.resolve_global``)
        resolved_enum = ctx.resolve_global(value)
        if resolved_enum is None:
            raise CompileError(f'cannot resolve the enum {value!r} to a spy value')
        return resolved_enum
    if isinstance(value, (Value, int, float, bytes, str, bool, complex)):
        return value
    if value is None:
        # ``None`` denotes the null value: the absent value of an option
        return Null()
    if value is int:
        return AnyIntType()
    if value is float:
        return FloatType(64)
    if value is complex:
        # the ``complex`` type maps to a complex of the default float width, like
        # a plain Python ``float`` maps to ``f64``
        return ComplexType(FloatType(64))
    if value is bytes:
        return BytesType()
    if value is str:
        return StrType()
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
    if value is syntax.c_char:
        return IntType(ctx.target_info().c_char_bits, True)
    if value is syntax.c_uchar:
        return IntType(ctx.target_info().c_char_bits, False)
    if value is syntax.c_short:
        return IntType(ctx.target_info().c_short_bits, True)
    if value is syntax.c_ushort:
        return IntType(ctx.target_info().c_short_bits, False)
    if value is syntax.c_int:
        return IntType(ctx.target_info().c_int_bits, True)
    if value is syntax.c_uint:
        return IntType(ctx.target_info().c_int_bits, False)
    if value is syntax.c_long:
        return IntType(ctx.target_info().c_long_bits, True)
    if value is syntax.c_ulong:
        return IntType(ctx.target_info().c_long_bits, False)
    if value is syntax.c_longlong:
        return IntType(ctx.target_info().c_longlong_bits, True)
    if value is syntax.c_ulonglong:
        return IntType(ctx.target_info().c_longlong_bits, False)
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


def unwrap_const_comptime(annotation: Any) -> tuple[bool, Any]:
    """Like :func:`unwrap_comptime`, for the ``syntax.ConstComptime`` marker:
    ``(True, None)`` for the bare ``ConstComptime``, ``(True, T)`` for
    ``ConstComptime[T]`` and ``(False, annotation)`` for anything else."""
    if annotation is None:
        return False, None
    if annotation is syntax.ConstComptime:
        return True, None
    if typing.get_origin(annotation) is syntax.ConstComptime:
        args = typing.get_args(annotation)
        if len(args) != 1:
            raise TypeError(f'ConstComptime takes exactly one type argument, got {annotation!r}')
        return True, args[0]
    return False, annotation

def unwrap_infer_comptime(annotation: Any) -> tuple[bool, Any]:
    """Like :func:`unwrap_comptime`, for the ``syntax.InferComptime`` marker:
    ``(True, None)`` for the bare ``InferComptime``, ``(True, T)`` for
    ``InferComptime[T]`` and ``(False, annotation)`` for anything else.  The
    marker only applies to a *local* declaration, so the parameter path rejects
    it (see ``astgen``)."""
    if annotation is None:
        return False, None
    if annotation is syntax.InferComptime:
        return True, None
    if typing.get_origin(annotation) is syntax.InferComptime:
        args = typing.get_args(annotation)
        if len(args) != 1:
            raise TypeError(f'InferComptime takes exactly one type argument, got {annotation!r}')
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
        case ComplexType():
            elem = replace_type_vars_type(value.elem, reps)
            assert isinstance(elem, FloatType)
            return ComplexType(elem)
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
            raise CoerceError(f"cannot use {value!r} as a constant of {type}")
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
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")
    return TaggedUnionValue(type, index, coerce_const(value, type.types[index]))


def _typed_undefined(type: Type) -> AnyValue:
    """The typed undefined value of ``type``: the unit value of a zero-sized
    type, an :class:`UndefinedScalar` of a scalar type, or an
    :class:`UndefinedAggregate` of an aggregate type."""
    unit = type.get_unit_value()
    if unit is not None:
        return unit
    if isinstance(type, ScalarType):
        return UndefinedScalar(type)
    if isinstance(type, AggregateType):
        return UndefinedAggregate(type)
    raise CoerceError(f'cannot create an undefined value of {type}')


def _coerce_const_option(value: AnyValue, type: OptionType) -> AnyValue:
    """A constant of the option ``type``: the absent value is the *typed* null
    of this very option - the untyped ``Null`` names the option it is absent for -
    and anything else is a present value of the child type, the ``OptionValue`` of
    the child constant (which is what the interpreter reads the payload out of,
    see ``interp._option_payload``)."""
    if isinstance(value, TypedNull):
        if value.child != type.child:
            raise CoerceError(f"cannot use {value!r} as a constant of {type}")
        return value
    if isinstance(value, Null):
        return TypedNull(type.child)
    if isinstance(value, OptionValue):
        return OptionValue(coerce_const(value.value, type.child))
    return OptionValue(coerce_const(value, type.child))


def _coerce_const_complex(value: AnyValue, type: ComplexType) -> AnyValue:
    """A constant of the complex type ``type``: a Python ``complex`` (or a real
    value, whose imaginary part is zero) as an aggregate of its two parts.  A
    complex aggregate of another element type is re-coerced part by part (the
    element widens or narrows)."""
    if isinstance(value, AggregateValue) and isinstance(value.type, ComplexType):
        return AggregateValue(
            (
                coerce_const(value.values[0], type.elem),
                coerce_const(value.values[1], type.elem),
            ),
            type,
        )
    if isinstance(value, complex):
        re, im = value.real, value.imag
    elif isinstance(value, (Int, Float)):
        # an already-typed spy numeric constant (the ``f64`` a compile-time
        # ``coerce`` folded to, say): its real part is the value, its imaginary
        # one zero
        re, im = value, 0.0
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoerceError(f"cannot use {value!r} as a complex constant")
    else:
        re, im = value, 0.0
    # the parts are coerced like any other value: an integer real part is
    # checked for losslessness (see the ``FloatType`` case of ``coerce_const``)
    return AggregateValue(
        (coerce_const(re, type.elem), coerce_const(im, type.elem)),
        type,
    )


def _is_function_value(value: AnyValue) -> bool:
    """Whether ``value`` is the value of a registered function (a
    :class:`fn.FunctionValue`).  ``sval`` cannot name that class at import time -
    ``fn`` imports ``sval`` - so it is imported here (the same local import
    ``BuiltinSignature.bind`` makes)."""
    from .fn import FunctionValue
    return isinstance(value, FunctionValue)


def coerce_const(value: AnyValue, type: Type) -> AnyValue:
    """The constant of the spy type ``type`` the compile-time value ``value``
    converts to - what an assignment, a call argument or a ``std.core.coerce``
    builds out of it (the interpreter lowers it to a MIR constant later).  Every
    compile-time value converts here, whatever its shape: a plain Python scalar,
    an already-typed ``Int``/``Float``, a type, an enumeration member, the
    null/void unit values, an aggregate or union value form, a ``ComptimePtr``
    (a pointer value) and a function value (a function pointer).

    Raises :class:`~spy.errors.CoerceError` when ``value`` has no such
    conversion."""
    # -- a value of any type ---------------------------------------------------
    if isinstance(value, UntypedUndefined):
        # the undefined literal is the value of any type: of a zero-sized type it
        # is that type's unit value, of any other the typed undefined of it
        return _typed_undefined(type)
    if isinstance(value, (UndefinedScalar, UndefinedAggregate)):
        # an *undetermined* value - the value of a place that holds nothing yet,
        # or of one that holds no storage at all - is a value of any type, so the
        # coercion re-tags it with the type of its new location.  A value of the
        # very type the target names is already the one it wants, and a value of
        # a *member* of an option or a union is the present payload/variant of one
        if value.type == type:
            return value
        if isinstance(type, OptionType):
            return _coerce_const_option(value, type)
        if isinstance(type, TaggedUnionType):
            return coerce_const_tagged_union(value, type)
        return _typed_undefined(type)
    if isinstance(type, AnyType):
        # an ``Any`` location names no fixed type, so every value is a value of
        # it: it is taken as it is
        return value

    # -- the conversions the target type decides -------------------------------
    if isinstance(type, OptionType):
        # a value of an option is either the null value, or a value of the child
        # type, which the interpreter then tags as present
        return _coerce_const_option(value, type)
    if isinstance(type, TaggedUnionType):
        return coerce_const_tagged_union(value, type)
    if isinstance(type, UnionType):
        # a union value carries no variant (the error code next to the
        # payload is the tag), and a constant of it exists only for a
        # union without storage: any such value is the unit value.  A
        # union *value* is never converted to another union - the storage
        # is reinterpreted through a pointer instead (see ``interp.store``)
        if type.get_unit_value() is None:
            raise CoerceError(f"cannot create a constant of the union {type}")
        if isinstance(value, UnionValue):
            return UnionValue(type)
        raise CoerceError(f"cannot use {value!r} as a constant of the union {type}")
    if isinstance(type, ComplexType):
        return _coerce_const_complex(value, type)
    if isinstance(type, (TupleType, StrDictType)):
        # a tuple and a ``dict[str, T]`` have no other type to convert from: a
        # value whose own type is the target is its own constant
        if type_of(value) == type:
            return value
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")

    # -- the values that *are* their own value ---------------------------------
    if isinstance(value, ComptimePtr):
        # a compile-time pointer is a pointer value.  The pointee type has to be
        # the same (a pointer does not convert between pointee types) - except a
        # pointer to an array, which is also the *multi* pointer of its elements:
        # the two carry the same address, and the view is what addressing goes by,
        # so it is retyped (like ``ptr_cast`` does).  The constness is *not*
        # checked: whether the pointee may be written is decided by the box as
        # well (see ``ComptimePtr.is_writable``), so the view is only a view, and
        # the ``*const T`` of a read-only pointer still coerces to the ``*T`` a
        # variable or a parameter declares
        if isinstance(type, PointerType):
            if value.type.elem == type.elem:
                return value
            if _points_at_elements_of(value.type, type):
                return ComptimePtr(value.box, value.offset, type)
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")
    if isinstance(value, CastedPtr):
        # a pointer reinterpreted by ``ptr_cast``: reading or writing through it
        # is not supported yet (see ``CastedPtr``)
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")
    if isinstance(value, (DeclareFunction, PyFunction)) or _is_function_value(value):
        # a registered function value, a Python function used at compile time or
        # an external declaration *is* a const function pointer (see
        # ``FunctionValue.get_type``/``PyFunction.get_type``): its concrete type
        # is checked where it is lowered (see ``interp``)
        if isinstance(type, PointerType):
            return value
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")
    if isinstance(value, AggregateValue):
        # an aggregate is ever only its own type: its value is the tree of its
        # field values, and no other type has one
        if isinstance(type, AggregateType) and value.type == type:
            return value
        raise CoerceError(f"cannot use {value!r} as a constant of {type}")
    if isinstance(value, (Int, Float)):
        # an already-typed constant (the value a compile-time location holds):
        # the target type governs, like the constant of any other location
        value = value.value
    if type.classify() == SpecialTypeKind.COMPTIME and type_of(value) == type:
        # a value of a type only compile-time values have (a closure, an
        # error-union success, an untyped function pointer, ...) is its own
        # constant: there is no other value to build it from
        return value

    # -- the scalar types ------------------------------------------------------
    match type:
        case BoolType():
            if not isinstance(value, bool):
                raise CoerceError(f"cannot use {value!r} as a bool constant")
            return value
        case BytesType():
            if not isinstance(value, bytes):
                raise CoerceError(f"cannot use {value!r} as a bytes constant")
            return value
        case StrType():
            if not isinstance(value, str):
                raise CoerceError(f"cannot use {value!r} as a str constant")
            return value
        case AnyIntType():
            # the type of an untyped integer literal: its values are the plain
            # Python ints the source wrote, so the literal itself is what a
            # location that may hold a compile-time value (a ``Comptime``
            # variable, an expression temporary) keeps.  The type has no runtime
            # representation, so a *runtime* location of it is rejected instead
            # (see ``_no_runtime_type``).
            if isinstance(value, bool) or not isinstance(value, int):
                raise CoerceError(
                    f"cannot use {value!r} as an untyped integer constant"
                )
            return value
        case IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise CoerceError(f"cannot use {value!r} as an integer constant")
            lo, hi = int_range(type)
            if not lo <= value <= hi:
                raise CoerceError(
                    f"integer constant {value} is out of range for {type}"
                )
            return Int(value, type)
        case FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CoerceError(f"cannot use {value} as a float constant")
            # an integer converts to a float only when it is exactly
            # representable.  Unlike a runtime integer, whose whole *type*
            # range is checked (see ``IntType.is_subtype_of``), a compile-time
            # value is checked by *value*: a power of two outside
            # ``float_exact_int_range`` is still exact (``2**63`` in ``f64``,
            # say).  A lossy conversion has to be spelled out (see
            # ``std.int.int_to_float``)
            if isinstance(value, int) and not int_to_float_is_exact(value, type):
                raise CoerceError(
                    f"integer constant {value} is not exactly representable "
                    f"as {type}"
                )
            return Float(float(value), type)
        case TypeType():
            if not isinstance(value, Type):
                raise CoerceError(f"cannot use {value} as a type constant")
            if value.get_type() != type:
                raise CoerceError(f"cannot use {value} as a type constant")
            return value
        case NullType():
            # the null value's own type: only the null literal is a value of it
            if isinstance(value, Null):
                return Null()
            raise CoerceError(f"cannot use {value!r} as a null constant")
        case VoidType():
            # the void type's unit value is what ``None`` used to be: a null
            # value converts to it (``NullType`` is a subtype of ``VoidType``)
            if isinstance(value, (Void, Null)):
                return Void()
            raise CoerceError(f"cannot use {value!r} as a void constant")
        case IntEnumType():
            # a value of the enumeration is one of its members: a typed member
            # value of this very enumeration, or one of its Python members (a
            # plain integer is never accepted - an enumeration is not an integer)
            if isinstance(value, IntEnumValue):
                if value.type is not type:
                    raise CoerceError(f"cannot use {value!r} as a constant of {type}")
                return value
            if isinstance(value, _INT_ENUM_BASE):
                index = type.index_of(value.name)
                if index is None or type.value_of(index) != value.value:
                    raise CoerceError(f"cannot use {value!r} as a constant of {type}")
                return IntEnumValue(type, index)
            raise CoerceError(f"cannot use {value!r} as a constant of {type}")
        case _ if type.classify() == SpecialTypeKind.DST:
            # a dynamically-sized type has no value of its own to build one from
            raise CoerceError(
                f"cannot create a constant of the dynamically-sized type {type}"
            )
        case _:
            raise CoerceError(
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

    @abstractmethod
    def enum_class_of(self, type: IntEnumType) -> type:
        """The Python ``enum.Enum`` class the host's ``IntEnumType`` ``type``
        was built from, so that a value read back at the Python boundary can be
        delivered as its member (the reverse of the resolution
        ``resolve_global`` performs for an enum class)."""
        ...

    def target_info(self) -> TargetInfo:
        ...

    def mir_cache(self) -> MirLowerCache:
        ...

    def special_types(self) -> SpecialTypes:
        ...

    @abstractmethod
    def get_compile_option(self, name: str) -> AnyValue | None:
        """The value the host stored for the compile-time option ``name``, or
        ``None`` when it is not set (the absent value of a spy value read by
        ``std.core.compile_option``)."""
        ...

    @abstractmethod
    def set_compile_option(self, name: str, value: AnyValue) -> None:
        """Store ``value`` as the compile-time option ``name`` of this host."""
        ...

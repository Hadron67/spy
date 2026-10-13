from typing import Any

from ..compiler import Option, builtin_func, struct
from .core import ConstSlicePtr

type TypeInfo = IntType | PointerType | ArrayType | StructType | UnionType | TaggedUnionType | OptionType | ComplexType | TupleType | StrDictType

@struct()
class IntType:
    bits: int
    signed: bool


@struct()
class PointerType:
    child: type
    is_const: bool
    # variant is not yet included

@struct()
class ArrayType:
    child: type
    size: Option[int]

@struct()
class StructField:
    name: bytes
    type: type
    default_value: Option[Any]
    # whether the field is compile-time known, and whether it is held by
    # reference (see ``sval.StructField``): the reported ``type`` is the logical
    # type ``T``, not the ``*const T`` an ``is_ref`` field stores
    is_comptime: bool
    is_ref: bool

@struct()
class StructType:
    fields: ConstSlicePtr[StructField]
    head: Option[Any]

@struct()
class UnionType:
    fields: ConstSlicePtr[type]

@struct()
class TaggedUnionType:
    types: ConstSlicePtr[type]

@struct()
class OptionType:
    child: type

@struct()
class ComplexType:
    elem: type


@struct()
class TupleType:
    # the type of every element, in order, and whether it is written
    # ``tuple[T, ...]`` (there is no fixed count then)
    types: ConstSlicePtr[type]
    has_ellipsis: bool


@struct()
class DictEntry:
    name: bytes
    type: type


@struct()
class StrDictType:
    # the named entries of a ``**kwargs``/``dict[str, T]`` shape, in order
    entries: ConstSlicePtr[DictEntry]


@builtin_func
def type_info(ty: Any) -> TypeInfo: ...

@builtin_func
def reify(info: TypeInfo) -> type: ...

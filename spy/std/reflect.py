from typing import Any

from ..compiler import Option, builtin_func, struct, u8
from .core import ConstSlicePtr

type TypeInfo = IntType | PointerType | ArrayType | StructType | UnionType | TaggedUnionType | OptionType

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
    name: ConstSlicePtr[u8]
    type: type
    default_value: Option[Any]

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

@builtin_func
def type_info(ty: Any) -> TypeInfo: ...

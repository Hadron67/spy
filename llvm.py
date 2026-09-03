import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import EllipsisType
from typing import final, override

from .util import ObjectCounter, StrBiMap, gen_get_children


class NameContext:
    @abstractmethod
    def get_struct_type_name(self, type: 'StructType') -> str:
        pass

    @abstractmethod
    def get_global_name(self, value: 'GlobalValue') -> str:
        pass

class Type:
    def from_float(self, value: float) -> 'Value':
        raise TypeError(f'cannot create float from {self}')

    def from_int(self, value: int) -> 'Value':
        return self.from_float(float(value))

    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return str(self)

    def get_children(self) -> 'list[Type]':
        return []

    def is_compatible(self, other: 'Type'):
        return self == other

    @abstractmethod
    def size(self, pointer_size: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def align(self, pointer_size: int) -> int:
        raise NotImplementedError

class Value:
    @abstractmethod
    def get_type(self) -> Type:
        pass

    @final
    def stringify(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        return self.get_type().stringify(name_context) + ' ' + self.stringify_value(name_context, local_counter)

    @abstractmethod
    def stringify_value(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        return str(self)

    def get_children(self) -> 'list[Value]':
        return []

# def gen_get_children(cls=None, excludes: set[str] | None = None):
#     def wrapper(cls):
#         gen_get_children0(cls, Value if issubclass(cls, Value) else Type, excludes)
#         return cls

#     if cls is not None:
#         return wrapper(cls)
#     return wrapper

@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __str__(self) -> str:
        match self.bits:
            case 16:
                return 'half'
            case 32:
                return 'float'
            case 64:
                return 'double'
            case _:
                raise ValueError(f"invalid float bits {self.bits}")

    @override
    def from_float(self, value: float) -> 'Value':
        return FloatValue(value, self)

    @override
    def size(self, pointer_size: int) -> int:
        return (self.bits + 7) // 8

    @override
    def align(self, pointer_size: int) -> int:
        return self.size(pointer_size)

F32 = FloatType(32)
F64 = FloatType(64)

@dataclass(frozen=True)
class IntType(Type):
    bits: int

    def __str__(self) -> str:
        return f'{"i"}{self.bits}'

    @final
    def get_range(self, signed: bool) -> tuple[int, int]:
        return IntType.get_range_from_data(self.bits, signed)

    @staticmethod
    def get_range_from_data(bits: int, signed: bool):
        if signed:
            return (-2**(bits - 1), 2**(bits - 1) - 1)
        else:
            return (0, 2**bits - 1)

    @override
    def from_int(self, value: int) -> 'Value':
        min, max = self.get_range(True)
        if not min <= value <= max:
            raise ValueError(f'value {value} out of range for {self}')
        return IntValue(value, self)

    @override
    def size(self, pointer_size: int) -> int:
        return (self.bits + 7) // 8

    @override
    def align(self, pointer_size: int) -> int:
        return self.size(pointer_size)

I8 = IntType(8)
I32 = IntType(32)
I64 = IntType(64)

@dataclass(frozen=True)
class LabelType(Type):
    def __str__(self) -> str:
        return 'label'

@dataclass(frozen=True)
@gen_get_children
class PointerType(Type):
    child: Type

    def __str__(self) -> str:
        return f"{self.child}*"

    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return 'ptr'

    @override
    def is_compatible(self, other: Type):
        return isinstance(other, PointerType)

    @override
    def size(self, pointer_size: int) -> int:
        return pointer_size

    @override
    def align(self, pointer_size: int) -> int:
        return pointer_size


@dataclass(frozen=True)
@gen_get_children
class FnType(Type):
    args: tuple[Type, ...]
    return_type: Type
    varargs: bool = False

    def __str__(self) -> str:
        return f'fn({", ".join(str(arg) for arg in self.args)}) -> {self.return_type}'

    @override
    def get_children(self) -> 'list[Type]':
        ret = list(self.args)
        ret.append(self.return_type)
        return ret

    @override
    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        ret = self.return_type.stringify(name_context)
        args = [i.stringify(name_context) for i in self.args]
        if self.varargs:
            args.append('...')
        return f"{ret} ({' '.join(args)})"

def fn_type(ret: Type | None, *args: Type | EllipsisType) -> FnType:
    varargs = False
    args2: list[Type] = []
    for arg in args:
        match arg:
            case Type():
                args2.append(arg)
            case EllipsisType():
                varargs = True
    return FnType(args=tuple(args2), return_type=ret if ret is not None else VoidType(), varargs=varargs)

@dataclass
class VoidType(Type):
    def __str__(self) -> str:
        return 'void'

class AggregateType(Type):
    pass

@gen_get_children
class StructType(AggregateType):
    fields: list[Type]

    @override
    def __init__(self, *fields: Type) -> None:
        self.fields = list(fields)

    def add_field(self, type: Type):
        self.fields.append(type)

    @override
    def __str__(self) -> str:
        return f"struct@{id(self)} {{{", ".join(str(i) for i in self.fields)}}}"

    def write_definition(self, name_context: 'NameContext') -> list[str]:
        return [f"%{name_context.get_struct_type_name(self)} = type {{{', '.join(i.stringify(name_context) for i in self.fields)}}}"]

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def __eq__(self, value: object, /) -> bool:
        return self is value

    @override
    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        if name_context is not None:
            return '%' + name_context.get_struct_type_name(self)
        else:
            return str(self)

@dataclass(frozen=True)
@gen_get_children
class ArrayType(AggregateType):
    child: Type
    length: int

    def __str__(self) -> str:
        return f"[{self.length} x {self.child}]"

    @override
    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return f"[{self.length} x {self.child.stringify(name_context)}]"

    @override
    def size(self, pointer_size: int) -> int:
        return self.length * self.child.size(pointer_size)

    @override
    def align(self, pointer_size: int) -> int:
        return self.child.align(pointer_size)


class LocalValue(Value):
    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    @final
    @override
    def stringify_value(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        assert local_counter is not None
        return f"%{local_counter.get_id(self)}"

class ArgValue(LocalValue):
    type: Type
    index: int

    def __init__(self, type: Type, index: int) -> None:
        self.type = type
        self.index = index

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} %{self.index}"

    def __repr__(self) -> str:
        return f"ArgValue@{id(self)}(type={self.type!r}, index={self.index})"

@dataclass
class VoidValue(Value):
    def __str__(self) -> str:
        return ''

    @override
    def get_type(self) -> Type:
        return VoidType()

@dataclass(frozen=True)
class IntValue(Value):
    value: int
    type: IntType

    def __str__(self) -> str:
        return str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

BOOL_TYPE = IntType(1)
BOOL_TRUE = IntValue(1, BOOL_TYPE)
BOOL_FALSE = IntValue(0, BOOL_TYPE)

@dataclass(frozen=True)
class FloatValue(Value):
    value: float
    type: FloatType

    @override
    def __str__(self) -> str:
        return str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

@dataclass
@gen_get_children
class NullValue(Value):
    type: PointerType

    @override
    def __str__(self) -> str:
        return "null"

    @override
    def get_type(self) -> Type:
        return self.type

_CHAR_CODES = [

]

@dataclass
class Undef(Value):
    type: Type

    @override
    def __str__(self) -> str:
        return "undef"

class Module(NameContext):
    _globals: 'StrBiMap[GlobalValue]'
    _global_constants: 'dict[Value, GlobalValue]'
    _struct_types: 'StrBiMap[StructType]'

    def __init__(self) -> None:
        self._globals = StrBiMap()
        self._struct_types = StrBiMap()
        self._global_constants = {}

    def add_recursively(self, types: list[Type] | None = None, values: 'list[Value] | None' = None):
        todo_values: list[Value] = []
        todo_types: list[Type] = []
        if types is not None:
            todo_types.extend(types)
        if values is not None:
            todo_values.extend(values)

        while len(todo_values) > 0:
            value = todo_values.pop()
            if isinstance(value, (Inst, BasicBlock)):
                continue
            if isinstance(value, GlobalValue):
                if self._globals.has_value(value):
                    continue
                name = value.get_required_name()
                assert name is None or not self._globals.has_key(name)
                self._add_global(value, name)
            todo_values.extend(value.get_children())
            todo_types.append(value.get_type())

            while len(todo_types) > 0:
                type = todo_types.pop()
                if isinstance(type, StructType):
                    if self._struct_types.has_value(type):
                        continue
                    self._add_struct_type(type)
                todo_types.extend(type.get_children())

    @override
    def get_global_name(self, value: 'GlobalValue') -> str:
        return self._globals.get_key(value)

    @override
    def get_struct_type_name(self, type: 'StructType') -> str:
        return self._struct_types.get_key(type)

    def _add_global(self, value: 'GlobalValue', name_prefix: str | None = None):
        if name_prefix is None:
            name_prefix = value.get_default_name_prefix()
        self._globals.add(self._globals.next_unique_name(name_prefix), value)

    def _add_struct_type(self, type: StructType, name_prefix: str | None = None):
        if name_prefix is None:
            name_prefix = 'struct'
        self._struct_types.add(self._struct_types.next_unique_name(name_prefix), type)

    def write(self) -> list[str]:
        ret: list[str] = []
        for type in self._struct_types.values():
            ret.extend(type.write_definition(self))
        for value in self._globals.values():
            ret.extend(value.write_definition(self))
        return ret

class GlobalValue(Value):
    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @abstractmethod
    def write_definition(self, name_context: NameContext) -> list[str]:
        pass

    @override
    @final
    def stringify_value(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue] | None = None) -> str:
        return '@' + name_context.get_global_name(self)

    def get_required_name(self) -> str | None:
        return None

    @abstractmethod
    def get_default_name_prefix(self) -> str:
        raise NotImplementedError

class GlobalValueFlags:
    IS_CONST = 1
    IS_PRIVATE = 2
    IS_UNNAMED_ADDR = 4
    COMMON = 8
    GLOBAL = 16

    DEFAULT = 7

    @staticmethod
    def str(flags: int):
        ret = ''
        if flags & GlobalValueFlags.COMMON:
            ret += 'common '
        if flags & GlobalValueFlags.GLOBAL:
            ret += 'global '
        if flags & GlobalValueFlags.IS_CONST:
            ret += 'private '
        if flags & GlobalValueFlags.IS_PRIVATE:
            ret += 'unnamed_addr '
        if flags & GlobalValueFlags.IS_UNNAMED_ADDR:
            ret += 'constant '
        return ret

@gen_get_children
class GlobalScalarValue(GlobalValue):
    value: Value
    flags: int

    @override
    def __init__(self, value: Value, flags: int = GlobalValueFlags.DEFAULT) -> None:
        self.value = value
        self.flags = flags

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        flags = GlobalValueFlags.str(self.flags)
        return [f'@{name_context.get_global_name(self)} = {flags}{self.value.stringify(name_context)}']

    @override
    def get_type(self) -> Type:
        return PointerType(self.value.get_type())

    @override
    def get_default_name_prefix(self):
        return "global"

@gen_get_children
class GlobalAggregateValue(GlobalValue):
    values: list[Value]
    flags: int
    type: Type

    @override
    def __init__(self, type: Type, *values: Value, flags: int = GlobalValueFlags.DEFAULT) -> None:
        self.values = list(values)
        self.flags = flags
        self.type = type
        match type:
            case StructType():
                assert len(values) == len(type.fields), f"length mismatch: {len(values)} != {len(type.fields)}"
                for value, result_type in zip(values, type.fields):
                    value_type = value.get_type()
                    assert result_type.is_compatible(value_type), f"incompatible types {result_type} and {value_type}"
            case ArrayType():
                assert len(values) == type.length, f"length mismatch: {len(values)} != {type.length}"
                for value in values:
                    value_type = value.get_type()
                    assert type.child.is_compatible(value_type), f"incompatible types {type.child} and {value_type}"
            case _:
                raise TypeError(f"{type} is not an aggregate type")

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        flags = GlobalValueFlags.str(self.flags)
        type = self.type.stringify(name_context)
        values = ', '.join(v.stringify(name_context) for v in self.values)
        return [f'@{name_context.get_global_name(self)} = {flags}{type} {{{values}}}']

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    @override
    def get_default_name_prefix(self):
        return "global"

@gen_get_children
class GlobalZeroAggregateValue(GlobalValue):
    type: Type
    flags: int

    def __init__(self, type: Type, flags: int = GlobalValueFlags.DEFAULT) -> None:
        self.type = type
        self.flags = flags
        assert isinstance(type, AggregateType), "type must be an aggregate type"

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        flags = GlobalValueFlags.str(self.flags)
        type = self.type.stringify(name_context)
        return [f'@{name_context.get_global_name(self)} = {flags}{type} zeroinitializer']

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    @override
    def get_default_name_prefix(self) -> str:
        return 'global'

def escape_byte(b: int) -> str:
    # 可打印 ASCII 且不特殊的字符直接输出
    if 32 <= b <= 126 and chr(b) not in ['\\', '"']:
        return chr(b)
    # 特殊字符转义
    if b == 0:
        return r"\00"
    if b == ord('"'):
        return r"\""
    if b == ord('\\'):
        return r"\\"
    # 其他非打印字符用 \xx 十六进制
    return f"\\{b:02x}"

class GlobalStringValue(GlobalValue):
    @override
    def __init__(self, value: bytes, flags: int = GlobalValueFlags.DEFAULT) -> None:
        self.value = value
        self.flags = flags

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        flags = GlobalValueFlags.str(self.flags)
        return [f'@{name_context.get_global_name(self)} = {flags}[{len(self.value)} x i8] c"{''.join(escape_byte(b) for b in self.value)}"']

    @override
    def get_type(self) -> Type:
        return PointerType(I8)

    @override
    def get_default_name_prefix(self):
        return "global"

@gen_get_children
class DeclareFunction(GlobalValue):
    type: FnType
    name: str

    def __init__(self, name: str, type: FnType, internal: bool = False):
        self.name = name
        self.type = type

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    def write_args(self, name_context: NameContext) -> list[str]:
        args: list[str] = []
        for arg in self.type.args:
            args.append(arg.stringify(name_context))
        if self.type.varargs:
            args.append('...')
        return args

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        return [f"declare {self.type.return_type.stringify(name_context)} @{self.name}({', '.join(self.write_args(name_context))})"]

    def get_required_name(self) -> str | None:
        return self.name

    @override
    def get_default_name_prefix(self):
        return 'fn'

class FunctionState(IntEnum):
    BUILDING_ARGS = 0
    BUILDING_RETURN_TYPE = 1
    BUILDING_BODY = 2
    FINISHED = 3

class IFunction:
    @abstractmethod
    def add_arg(self, type: Type) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_arg(self, index: int) -> Value:
        raise NotImplementedError

class FunctionArgs(IFunction):

    def __init__(self, parent: 'FunctionArgs | None' = None) -> None:
        self.parent = parent
        self.args: list[ArgValue] = []

    @override
    def get_arg(self, index: int) -> Value:
        return self.args[index]

    @override
    def add_arg(self, type: Type) -> int:
        ret = len(self.args)
        self.args.append(ArgValue(type, ret))
        return ret

class Function(GlobalValue, IFunction):
    def __init__(self, name: str | None = None, entry: 'BasicBlock | None' = None) -> None:
        self.name = name
        self._entry = entry if entry is not None else BasicBlock()
        self._args: list[ArgValue] = []
        self._type: FnType | None = None

    @property
    def entry(self):
        return self._entry

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        assert self._type is not None
        ret: list[str] = []
        local_counter: ObjectCounter[LocalValue] = ObjectCounter()
        ret.append(f'define {'internal ' if self.name is None else ''}{self._type.return_type.stringify(name_context)} @{name_context.get_global_name(self)}({', '.join(i.stringify(name_context, local_counter) for i in self._args)}) {{')

        blocks = self._entry.collect_blocks()

        # allocate the indices first, as required by llvm ir
        for block in blocks:
            local_counter.get_id(block)
            for inst in block.insts:
                if not isinstance(inst.get_type(), VoidType):
                    local_counter.get_id(inst)

        for block in blocks:
            ret.append(str(local_counter.get_id(block)) + ':')
            for line in block.write(name_context, local_counter):
                ret.append('    ' + line)
        ret.append('}')
        return ret

    @override
    def get_type(self) -> Type:
        assert self._type is not None
        return PointerType(self._type)

    @override
    def add_arg(self, type: Type) -> int:
        assert self._type is None or self._type.varargs
        ret = ArgValue(type, len(self._args))
        self._args.append(ret)
        return ret.index

    def add_args(self, *types: Type):
        return tuple(self.get_arg(self.add_arg(t)) for t in types)

    def get_args(self):
        return tuple(self._args)

    @override
    def get_arg(self, index: int) -> Value:
        return self._args[index]

    def set_return_type(self, type: Type, varargs: bool = False):
        assert self._type is None
        self._type = FnType(tuple(i.type for i in self._args), type, varargs)

    @override
    def get_children(self) -> list[Value]:
        assert self._type is not None
        # exclude all Inst when collecting

        ret: list[Value] = []
        for b in self._entry.collect_blocks():
            for inst in b.insts:
                ret.extend(i for i in inst.get_children() if not isinstance(i, Inst) and not isinstance(i, BasicBlock))
        return ret

    def get_required_name(self) -> str | None:
        return self.name

    @override
    def get_default_name_prefix(self) -> str:
        return 'fn'

class Inst(LocalValue):
    @abstractmethod
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        pass

    def try_evaluate(self) -> Value | None:
        return None

class Nop(Inst):
    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        raise RuntimeError("Nop in inst block")

class BasicBlock(LocalValue):
    @override
    def __init__(self) -> None:
        self.insts: list[Inst] = []
        self._finished = False

    @override
    def get_children(self) -> 'list[Value]':
        raise RuntimeError("cannot be called directly")

    def get_outgoing_blocks(self):
        ret: list[BasicBlock] = []
        seen_blocks: set[BasicBlock] = set()
        for inst in self.insts:
            for c in inst.get_children():
                if isinstance(c, BasicBlock) and c not in seen_blocks:
                    seen_blocks.add(c)
                    ret.append(c)
        return ret

    @final
    def write(self, name_context: NameContext, inst_counter: ObjectCounter[LocalValue]) -> list[str]:
        ret: list[str] = []
        for inst in self.insts:
            if isinstance(inst, Nop):
                continue
            if inst.get_type() == VoidType():
                ret.append(inst.stringify_inst(name_context, inst_counter))
            else:
                ret.append(f"%{inst_counter.get_id(inst)} = {inst.stringify_inst(name_context, inst_counter)}")
        return ret

    def collect_blocks(self):
        ret: list[BasicBlock] = []
        seen_blocks: set[BasicBlock] = set()
        todo: list[BasicBlock] = [self]
        while len(todo) > 0:
            item = todo.pop()
            if item in seen_blocks:
                continue
            seen_blocks.add(item)
            ret.append(item)
            children = item.get_outgoing_blocks()
            children.reverse()
            todo.extend(children)
        return ret

    @override
    def get_type(self) -> Type:
        return LabelType()

    # build methods

    def emit(self, inst: Inst) -> Value:
        assert not self._finished
        if isinstance(inst, Branch):
            self._finished = True
        value = inst.try_evaluate()
        if value is not None:
            return value
        self.insts.append(inst)
        return inst

    def add(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                if isinstance(rhs, IntValue) and rhs.value < 0:
                    return self.emit(Binary(Sub(), lhs, IntValue(-rhs.value, rhs.type)))
                return self.emit(Binary(Add(), lhs, rhs))
            case FloatType():
                return self.emit(Binary(FAdd(), lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def sub(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Binary(Sub(), lhs, rhs))
            case FloatType():
                return self.emit(Binary(FSub(), lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def mul(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Binary(Mul(), lhs, rhs))
            case FloatType():
                return self.emit(Binary(FMul(), lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def div(self, lhs: Value, rhs: Value, signed: bool) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Binary(SDiv() if signed else UDiv(), lhs, rhs))
            case FloatType():
                return self.emit(Binary(FDiv(), lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def fdiv(self, lhs: Value, rhs: Value):
        return self.emit(Binary(FDiv(), lhs, rhs))

    def atomicrmw(self, op: 'BinaryOp', ptr: Value, rhs: Value, ordering: 'Ordering') -> Value:
        return self.emit(AtomicRmw(op, ptr, rhs, ordering))

    def rem(self, lhs: Value, rhs: Value, signed: bool) -> Value:
        return self.emit(Binary(SRem() if signed else URem(), lhs, rhs))

    def load(self, ptr: Value) -> Value:
        return self.emit(Load(ptr))

    def store(self, ptr: Value, value: Value | int):
        if isinstance(value, Value):
            self.emit(Store(ptr, value))
        else:
            ptr_type = ptr.get_type()
            assert isinstance(ptr_type, PointerType)
            assert isinstance(ptr_type.child, IntType)
            return self.emit(Store(ptr, IntValue(value, ptr_type.child)))

    def alloca(self, type: Type):
        return self.emit(Alloca(type))

    def ret(self, value: Value | None = None):
        self.emit(Ret(value if value is not None else VoidValue()))

    def extract_value(self, value: Value, *indices: int):
        return self.emit(ExtractValue(value, indices))

    def insert_value(self, value: Value, elem_value: Value, *indices: int):
        return self.emit(InsertValue(value, elem_value, *indices))

    def fneg(self, value: Value):
        return self.emit(FNeg(value))

    def int_to_float(self, value: Value, float_type: FloatType):
        return self.emit(IntToFloat(value, float_type))

    def uint_to_float(self, value: Value, float_type: FloatType):
        return self.emit(UIntToFloat(value, float_type))

    def zext(self, value: Value, type: IntType):
        return self.emit(UIntExt(value, type))

    def sext(self, value: Value, type: IntType):
        return self.emit(IntExt(value, type))

    def float_trunc(self, value: Value, type: FloatType):
        return self.emit(FloatTrunc(value, type))

    def float_ext(self, value: Value, type: FloatType):
        return self.emit(FloatExt(value, type))

    def get_element_ptr(self, array: Value, *indices: Value | int) -> Value:
        return self.emit(GetElementPtr(array, indices))

    def ptrtoint(self, value: Value, type: IntType):
        return self.emit(PtrToInt(value, type))

    def icmp(self, op: 'IcmpOp', signed: bool, lhs: Value, rhs: Value):
        return self.emit(Icmp(op, signed, lhs, rhs))

    def fcmp(self, op: 'IcmpOp', lhs: Value, rhs: Value):
        return self.emit(Fcmp(op, lhs, rhs))

    def and_(self, lhs: Value, rhs: Value):
        return self.emit(Binary(AndOp(), lhs, rhs))

    def or_(self, lhs: Value, rhs: Value):
        return self.emit(Binary(OrOp(), lhs, rhs))

    def xor(self, lhs: Value, rhs: Value):
        return self.emit(Binary(XorOp(), lhs, rhs))

    def phi(self, *incomings: 'tuple[Value, BasicBlock]'):
        ret = self.emit(Phi(*incomings))
        assert isinstance(ret, Phi)
        return ret

    def br(self, cond: Value, if_true: 'BasicBlock', if_false: 'BasicBlock'):
        self.emit(Br(cond, if_true, if_false))

    def jmp(self, target: 'BasicBlock'):
        self.emit(BrDirect(target))

    def call(self, fn: Value, *args: Value):
        return self.emit(Call(fn, args))

    def float_func(self, value: Value, f32_fn: Value, f64_fn: Value):
        type = value.get_type()
        assert isinstance(type, FloatType)
        match type.bits:
            case 32:
                return self.call(f32_fn, value)
            case 64:
                return self.call(f64_fn, value)
            case _:
                raise TypeError(f"Cannot take {f32_fn} or {f64_fn} of {value.get_type()}")

    def float_func2(self, value: Value, exponent: Value, f32_fn: Value, f64_fn: Value):
        """Like :meth:`float_func`, but for a binary float function."""
        type = value.get_type()
        assert isinstance(type, FloatType)
        match type.bits:
            case 32:
                return self.call(f32_fn, value, exponent)
            case 64:
                return self.call(f64_fn, value, exponent)
            case _:
                raise TypeError(f"Cannot take {f32_fn} or {f64_fn} of {value.get_type()}")

    def sqrt(self, value: Value, reg_name: str | None = None) -> Value:
        return self.float_func(value, SQRT_F32, SQRT_F64)

    def pow(self, value: Value, exponent: Value) -> Value:
        return self.float_func2(value, exponent, POW_F32, POW_F64)

    def exp(self, value: Value) -> Value:
        return self.float_func(value, EXP_F32, EXP_F64)

    def sin(self, value: Value) -> Value:
        return self.float_func(value, SIN_F32, SIN_F64)

    def cos(self, value: Value) -> Value:
        return self.float_func(value, COS_F32, COS_F64)

    def ln(self, value: Value) -> Value:
        return self.float_func(value, LN_F32, LN_F64)

@gen_get_children
class Unary(Inst):
    child: Value

    def __init__(self, child: Value) -> None:
        self.child = child

    @abstractmethod
    def _check(self):
        pass

    @abstractmethod
    def head_name(self) -> str:
        pass

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"{self.head_name()} {self.child.stringify(name_context, local_counter)}"

    def get_type(self) -> Type:
        return self.child.get_type()

class FNeg(Unary):
    @override
    def head_name(self) -> str:
        return "fneg"

    @override
    def _check(self):
        type = self.child.get_type()
        assert isinstance(type, FloatType), f"float type expected, got {type}"

    def try_evaluate(self) -> Value | None:
        match self.child:
            case FloatValue(value, t):
                return FloatValue(-value, t)
        return None

class BinaryOp:
    @abstractmethod
    def get_type(self) -> Type:
        raise NotImplementedError

    @abstractmethod
    def head_name(self) -> str:
        raise NotImplementedError

    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        return None

@gen_get_children
class Binary(Inst):
    op: BinaryOp
    lhs: Value
    rhs: Value
    type: Type

    def __init__(self, op: BinaryOp, lhs: Value, rhs: Value) -> None:
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert lhs_type == rhs_type, f"incompatible types {lhs_type} and {rhs_type}"
        self.type = lhs_type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'{self.op.head_name()} {self.type} {self.lhs.stringify_value(name_context, local_counter)}, {self.rhs.stringify_value(name_context, local_counter)}'

    @override
    def get_type(self) -> Type:
        return self.lhs.get_type()

    @override
    def try_evaluate(self) -> Value | None:
        return self.op.try_evaluate(self.lhs, self.rhs)

class Add(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'add'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a + b, t)
        return None

class FAdd(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'fadd'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a + b, t)
            case FloatValue(0, t), _:
                return rhs
            case _, FloatValue(0, t):
                return lhs
        return None

class Sub(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'sub'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a - b, t)
        return None

class Mul(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'mul'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a * b, t)
        return None

class FMul(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'fmul'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a * b, t)
            case FloatValue(0.0, t), _:
                return FloatValue(0.0, t)
            case _, FloatValue(0.0, t):
                return FloatValue(0.0, t)
        return None

class FSub(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'fsub'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a - b, t)
            case _, FloatValue(0, _):
                return lhs
        return None

class SRem(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'srem'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a % b, t)
        return None

class URem(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'urem'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a % b, t)
        return None

class FRem(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'frem'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(math.fmod(a, b), t)
        return None

class SDiv(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'sdiv'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a // b, t)
        return None

class AndOp(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'and'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a & b, t)
        return None

class OrOp(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'or'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a | b, t)
        return None

class XorOp(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'xor'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a ^ b, t)
        return None

class UDiv(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'udiv'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a // b, t)
        return None

class FDiv(BinaryOp):
    @override
    def head_name(self) -> str:
        return 'fdiv'

    @override
    def try_evaluate(self, lhs: Value, rhs: Value) -> Value | None:
        match lhs, rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a / b, t)
        return None

class Ordering(Enum):
    ACQUIRE = 'acquire'
    RELEASE = 'release'
    ACQ_REL = 'acq_rel'
    SEQ_CST = 'seq_cst'
    RELAXED = 'relaxed'
    MONOTONIC = 'monotonic'

class AtomicRmw(Inst):
    op: BinaryOp
    ptr: Value
    value: Value
    ordering: Ordering

    def __init__(self, op: BinaryOp, ptr: Value, value: Value, ordering: Ordering = Ordering.SEQ_CST):
        self.op = op
        self.ptr = ptr
        self.value = value
        self.ordering = ordering

        ptr_type = ptr.get_type()
        value_type = value.get_type()
        assert isinstance(ptr_type, PointerType), f"pointer type expected, got {ptr_type}"
        assert ptr_type.child.is_compatible(value_type), f"incompatible types: {ptr_type.child} and {value_type}"

    @override
    def get_type(self) -> Type:
        return self.value.get_type()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'atomicrmw {self.op.head_name()} {self.ptr.stringify(name_context, local_counter)}, {self.value.stringify(name_context, local_counter)} {self.ordering.value}'

@gen_get_children
class Load(Inst):
    ptr: Value
    type: Type

    def __init__(self, ptr: Value):
        self.ptr = ptr
        type = ptr.get_type()
        assert isinstance(type, PointerType), f"pointer type expected, got {type}"
        self.type = type.child

    @override
    def get_type(self) -> Type:
        return self.type

    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'load {self.type.stringify(name_context)}, {self.ptr.stringify(name_context, local_counter)}'

@gen_get_children
class Alloca(Inst):
    type: Type

    @override
    def __init__(self, type: Type) -> None:
        self.type = type

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        type = self.type.stringify(name_context)
        return f"alloca {type}"

@gen_get_children
class ConversionInst[T: Type](Inst):
    value: Value
    type: T

    def __init__(self, value: Value, type: T) -> None:
        self.value = value
        self.type = type

    @abstractmethod
    def head_name(self) -> str:
        pass

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'{self.head_name()} {self.value.stringify(name_context, local_counter)} to {self.type.stringify(name_context)}'

    @override
    def get_type(self) -> Type:
        return self.type

class FloatExt(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'fpext'

class FloatTrunc(ConversionInst[FloatType]):
    def head_name(self) -> str:
        return 'fptrunc'

class FloatToInt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'fptosi'

class FloatToUInt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'fptoui'

class IntToFloat(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'sitofp'

    def try_evaluate(self) -> Value | None:
        if isinstance(self.value, IntValue):
            return FloatValue(float(self.value.value), self.type)
        return None

class UIntToFloat(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'uitofp'

    def try_evaluate(self) -> Value | None:
        if isinstance(self.value, IntValue):
            return FloatValue(float(self.value.value), self.type)
        return None

class IntTrunc(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'trunc'

class IntExt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'sext'

class UIntExt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'zext'

@gen_get_children
class GetElementPtr(Inst):
    ptr: Value
    indices: tuple[Value, ...]
    ptr_type: Type
    type: PointerType

    def __init__(self, ptr: Value, indices: tuple[Value | int, ...]):
        self.ptr = ptr
        indices0: list[Value] = []

        ptr_type = self.ptr.get_type()
        assert isinstance(ptr_type, PointerType), "expected pointer type"
        type = ptr_type.child
        i0 = indices[0]
        if isinstance(i0, int):
            i0 = IntValue(i0, I64)
        i0_type = i0.get_type()
        assert isinstance(i0_type, IntType) and i0_type.bits == 64, f"i64 is required for array subscripts, got {i0_type}"
        indices0.append(i0)
        for i in indices[1:]:
            match type:
                case StructType():
                    if isinstance(i, int):
                        i = IntValue(i, I32)
                    assert isinstance(i, IntValue) and i.type.bits == 32, f"i32 constant is required for struct index, got {i}"
                    assert i.value < len(type.fields), f"struct index {i.value} is out of bounds for type {type}"
                    indices0.append(i)
                    type = type.fields[i.value]
                case ArrayType():
                    if isinstance(i, int):
                        i = IntValue(i, I64)
                    i_type = i.get_type()
                    assert isinstance(i_type, IntType) and i_type.bits >= 32, f"i32 or i64 is required for array index, got {i_type}"
                    indices0.append(i)
                    type = type.child
                case _:
                    raise TypeError(f"cannot get element of type {type}")
        self.indices = tuple(indices0)
        self.ptr_type = ptr_type.child
        self.type = PointerType(type)

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'getelementptr inbounds {self.ptr_type.stringify(name_context)}, {self.ptr.stringify(name_context, local_counter)}, {', '.join(i.stringify(name_context, local_counter) for i in self.indices)}'

@gen_get_children
class Call(Inst):
    fn: Value
    args: tuple[Value, ...]
    fn_type: FnType

    def __init__(self, fn: Value, args: tuple[Value, ...]):
        self.fn = fn
        self.args = args
        fn_ptr_type = fn.get_type()
        assert isinstance(fn_ptr_type, PointerType), "expected pointer type"
        fn_type = fn_ptr_type.child
        assert isinstance(fn_type, FnType), "expected function type"
        if not fn_type.varargs:
            assert len(args) == len(fn_type.args), "argument mismatch"
        else:
            assert len(args) >= len(fn_type.args), "argument mismatch"
        for i in range(len(fn_type.args)):
            arg = args[i]
            expected = fn_type.args[i]
            arg_type = arg.get_type()
            assert expected.is_compatible(arg_type), f"argument type mismatch at {i}-th arg, expected {expected}, got {arg_type}"
        self.fn_type = fn_type

    @override
    def get_type(self) -> Type:
        return self.fn_type.return_type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        ret_type = self.fn_type.return_type.stringify(name_context)
        fn = self.fn.stringify_value(name_context, local_counter)
        args_type = ''
        if self.fn_type.varargs:
            args_type = f" ({', '.join(arg.stringify(name_context) for arg in self.fn_type.args)}, ...)"

        args = [arg.stringify(name_context, local_counter) for arg in self.args]
        return f'call {ret_type}{args_type} {fn}({', '.join(args)})'

@gen_get_children
class Store(Inst):
    ptr: Value
    value: Value

    def __init__(self, ptr: Value, value: Value):
        self.ptr = ptr
        self.value = value
        ptr_type = ptr.get_type()
        value_type = value.get_type()
        assert isinstance(ptr_type, PointerType), "pointer type expected"
        assert ptr_type.child.is_compatible(value_type), f"incompatible types {ptr_type} and {value_type}"

    @override
    def get_type(self) -> Type:
        return VoidType()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'store {self.value.stringify(name_context, local_counter)}, {self.ptr.stringify(name_context, local_counter)}'

@gen_get_children
class ExtractValue(Inst):
    value: Value
    indices: tuple[int, ...]
    type: Type

    def __init__(self, value: Value, indices: tuple[int, ...]) -> None:
        self.value = value
        self.indices = indices
        type = value.get_type()
        assert len(indices) > 0
        for i in indices:
            match type:
                case StructType():
                    assert i < len(type.fields), f"index {i} is out of bounds for type {type}"
                    type = type.fields[i]
                case ArrayType():
                    assert i < type.length, f"index {i} is out of bounds for type {type}"
                    type = type.child
                case _:
                    raise TypeError(f"cannot extract value from type {type}")
        self.type = type

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"extractvalue {self.value.stringify(name_context, local_counter)}, {', '.join(str(i) for i in self.indices)}"

@gen_get_children
class InsertValue(Inst):
    value: Value
    elem_value: Value
    indices: tuple[int, ...]

    @override
    def __init__(self, value: Value, elem_value: Value, *indices: int) -> None:
        type = value.get_type()
        for i in indices:
            match type:
                case StructType():
                    assert i < len(type.fields), f"index {i} is out of bounds for type {type}"
                    type = type.fields[i]
                case ArrayType():
                    assert i < type.length, f"index {i} is out of bounds of {type.length}"
                    type = type.child
                case _:
                    raise TypeError(f"{type} is not an aggregate type")
        self.value = value
        self.elem_value = elem_value
        self.indices = indices

    @override
    def get_type(self) -> Type:
        return self.value.get_type()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        value = self.value.stringify(name_context, local_counter)
        elem_value = self.elem_value.stringify(name_context, local_counter)
        return f"insertvalue {value}, {elem_value}, {', '.join(str(i) for i in self.indices)}"

class Branch(Inst):
    pass

@gen_get_children
class Br(Branch):
    condition: Value
    if_true: BasicBlock
    if_false: BasicBlock

    def __init__(self, cond: Value, if_true: BasicBlock, if_false: BasicBlock) -> None:
        self.condition = cond
        self.if_true = if_true
        self.if_false = if_false
        cond_type = cond.get_type()
        assert cond_type == BOOL_TYPE, f"bool type expected, got {cond_type}"

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        cond = self.condition.stringify(name_context, local_counter)
        if_true = self.if_true.stringify(name_context, local_counter)
        if_false = self.if_false.stringify(name_context, local_counter)
        return f"br {cond}, {if_true}, {if_false}"

    @override
    def get_type(self) -> Type:
        return VoidType()

@gen_get_children
class BrDirect(Branch):
    target: BasicBlock

    @override
    def __init__(self, target: BasicBlock) -> None:
        self.target = target

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"br {self.target.stringify(name_context, local_counter)}"

    @override
    def get_type(self) -> Type:
        return VoidType()

@gen_get_children
class Ret(Branch):
    value: Value

    def __init__(self, value: Value) -> None:
        self.value = value

    @override
    def get_type(self) -> Type:
        return VoidType()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'ret {self.value.stringify(name_context, local_counter)}'

@gen_get_children
class Phi(Inst):
    type: Type
    incomings: list[tuple[Value, BasicBlock]]

    @override
    def __init__(self, incoming: tuple[Value, BasicBlock]) -> None:
        self.type = incoming[0].get_type()
        self.incomings = [incoming]

    def add_incoming(self, value: Value, block: BasicBlock):
        type = value.get_type()
        assert self.type == type, f"types mismatch: {self.type} and {type}"
        self.incomings.append((value, block))

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        type = self.type.stringify(name_context)
        labels = (f"[{v.stringify_value(name_context, local_counter)}, {b.stringify_value(name_context, local_counter)}]" for v, b in self.incomings)
        return f"phi {type} {', '.join(labels)}"

class IcmpOp(Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"

@gen_get_children
class Icmp(Inst):
    type: IntType
    signed: bool
    op: IcmpOp
    lhs: Value
    rhs: Value

    @override
    def __init__(self, op: IcmpOp, signed: bool, lhs: Value, rhs: Value) -> None:
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert isinstance(lhs_type, IntType), f"int type expected, got {lhs_type}"
        assert lhs_type == rhs_type, f"types mismatch: {lhs_type} and {rhs_type}"
        self.type = lhs_type
        self.signed = signed
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        op: str = ''
        match self.op:
            case IcmpOp.EQ | IcmpOp.NE:
                op = self.op.value
            case _:
                op = ('s' if self.signed else 'u') + self.op.value
        type = self.type.stringify(name_context)
        lhs = self.lhs.stringify_value(name_context, local_counter)
        rhs = self.rhs.stringify_value(name_context, local_counter)
        return f"icmp {op} {type} {lhs}, {rhs}"

    @override
    def get_type(self) -> Type:
        return BOOL_TYPE

@gen_get_children
class Fcmp(Inst):
    """A floating-point comparison, producing an ``i1``.

    The predicate names follow LLVM: ``eq``/``ne`` become ``oeq``/``une`` and
    the ordering predicates take the ``o`` (ordered) prefix, so that a ``NaN``
    operand compares false everywhere, matching numpy semantics.
    """

    op: IcmpOp
    lhs: Value
    rhs: Value
    type: FloatType

    @override
    def __init__(self, op: IcmpOp, lhs: Value, rhs: Value) -> None:
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert isinstance(lhs_type, FloatType), f"float type expected, got {lhs_type}"
        assert lhs_type == rhs_type, f"types mismatch: {lhs_type} and {rhs_type}"
        self.type = lhs_type
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        match self.op:
            case IcmpOp.EQ:
                op = 'oeq'
            case IcmpOp.NE:
                op = 'une'
            case _:
                op = 'o' + self.op.value
        type = self.type.stringify(name_context)
        lhs = self.lhs.stringify_value(name_context, local_counter)
        rhs = self.rhs.stringify_value(name_context, local_counter)
        return f"fcmp {op} {type} {lhs}, {rhs}"

    @override
    def get_type(self) -> Type:
        return BOOL_TYPE

@gen_get_children
class PtrToInt(Inst):
    value: Value
    type: IntType

    def __init__(self, value: Value, type: IntType) -> None:
        value_type = value.get_type()
        assert isinstance(value_type, PointerType), f"expected pointer type, got {value_type}"
        self.value = value
        self.type = type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        value = self.value.stringify(name_context, local_counter)
        type = self.type.stringify(name_context)
        return f"ptrtoint {value} to {type}"

    @override
    def get_type(self) -> Type:
        return self.type

SQRT_F32 = DeclareFunction('llvm.sqrt.f32', FnType((F32,), F32))
SQRT_F64 = DeclareFunction('llvm.sqrt.f64', FnType((F64,), F64))
POW_F32 = DeclareFunction('llvm.pow.f32', FnType((F32, F32), F32))
POW_F64 = DeclareFunction('llvm.pow.f64', FnType((F64, F64), F64))
EXP_F32 = DeclareFunction('llvm.exp.f32', FnType((F32,), F32))
EXP_F64 = DeclareFunction('llvm.exp.f64', FnType((F64,), F64))
LN_F32 = DeclareFunction('llvm.log.f32', FnType((F32,), F32))
LN_F64 = DeclareFunction('llvm.log.f64', FnType((F64,), F64))
SIN_F32 = DeclareFunction('llvm.sin.f32', FnType((F32,), F64))
SIN_F64 = DeclareFunction('llvm.sin.f64', FnType((F64,), F64))
COS_F32 = DeclareFunction('llvm.cos.f32', FnType((F32,), F32))
COS_F64 = DeclareFunction('llvm.cos.f64', FnType((F64,), F64))

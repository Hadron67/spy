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
are equal), which is what makes the compile-time comparisons in
``spy.typeof(a) == spy.u64`` work.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, override

from spy import mir

from .errors import CompileError, SpyError

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.typeof``) and other compile-time objects.
    Concrete values expose their spy type as ``.type``."""
    @abstractmethod
    def type(self) -> 'Type':
        raise NotImplementedError

class Type(Value):
    @override
    def type(self) -> 'Type':
        return TYPE_TYPE

@dataclass(frozen=True)
class TypeType(Type):
    level: int
    @override
    def type(self) -> Type:
        return TypeType(self.level + 1)

TYPE_TYPE = TypeType(0)

@dataclass(frozen=True)
class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    @override
    def type(self) -> Type:
        return TYPE_TYPE


@dataclass(frozen=True)
class VoidType(Type):
    """The type of ``None``: the return type of a function that returns no
    value (declared as ``-> None``, or inferred for a body without value
    returns).  It is a spy type only in the signature of such functions;
    there are no runtime values of this type (a call of a void function
    produces no value)."""

    @override
    def type(self) -> Type:
        return TYPE_TYPE

@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool


@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __post_init__(self) -> None:
        assert self.bits in (32, 64), f"unsupported float bits {self.bits}"


@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: bool = False


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type


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

@dataclass(frozen=True)
class FunctionCallArgInfo:
    index: int # pass this argument at the given index

class FunctionReturnInfo:
    pass

@dataclass(frozen=True)
class FunctionValueReturnInfo(FunctionReturnInfo):
    """Return by value: No result location pointer, mir function returns `mir_type`."""
    mir_type: mir.Type

@dataclass(frozen=True)
class FunctionRetLocReturnInfo(FunctionReturnInfo):
    """Return by result location: the result pointer is the `arg_index`-th argument, mir function returns void."""
    arg_index: int

@dataclass(frozen=True)
class FunctionCallInfo:
    total_mir_args: int
    args_map: tuple[FunctionCallArgInfo | None, ...] # non-None: pass this argument using the given info; None: don't pass this argument
    return_info: FunctionReturnInfo

@dataclass(frozen=True)
class StructField:
    name: str
    type: Type


class StructType(Type):
    """A spy struct type: the object ``@cache.struct()`` binds to the class
    name.  It doubles as the Python-side constructor of struct *values*:
    calling ``Foo(a, b)`` creates a native struct instance whose memory
    follows the LLVM layout (see ``_py_cls``).

    The identity of the object *is* the identity of the type (two structs
    are equal only if they are the same object), which is what makes
    ``spy.typeof(x) == Foo`` work.
    """

    def __init__(self, name: str) -> None:
        self.name = name
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
        # the MIR mirror of the type, created lazily when a function
        # body needs it (see ``interp.to_mir_type``)
        self._mir: mir.Type | None = None

    def add_field(self, name: str, type: Type) -> None:
        assert all(f.name != name for f in self._fields)
        self._fields.append(StructField(name, type))

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
                f'struct {self.name} is not bound to a JitContext; '
                'define it with @cache.struct() and construct it after the '
                'module has loaded'
            )
        return self._py_init(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<spy struct {self.name}>'

    @override
    def type(self) -> Type:
        return TYPE_TYPE

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

def int_range(type: IntType) -> tuple[int, int]:
    if type.signed:
        return (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
    return (0, 2 ** type.bits - 1)


def type_str(type: Type) -> str:
    """A short, printable name of a type (used in error messages and in the
    mangled names of compiled specializations)."""
    match type:
        case BoolType():
            return 'bool'
        case IntType():
            return ('i' if type.signed else 'u') + str(type.bits)
        case FloatType():
            return 'f' + str(type.bits)
        case PointerType(elem, is_const):
            return '*' + ('const ' if is_const else '') + type_str(elem)
        case FunctionType(args, ret):
            return f'fn({', '.join(type_str(a.type) for a in args)}) -> {type_str(ret)}'
        case AnyFunction():
            return 'any fn'
        case VoidType():
            return 'void'
        case StructType():
            return type.name
        case _:
            return str(type)


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
    types that may occur in a struct)."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8
        case FloatType():
            return type.bits // 8
        case StructType():
            return max(_alignment_of(f.type) for f in type.fields)
        case _:
            raise SpyError(f"type {type_str(type)} has no layout")


def _size_of(type: Type) -> int:
    """The size of a type in bytes, rounded up to its alignment (the
    natural layout the ctypes instances - and the LLVM structs they
    mirror - use)."""
    match type:
        case BoolType():
            return 1
        case IntType():
            return type.bits // 8
        case FloatType():
            return type.bits // 8
        case StructType():
            offset = 0
            for field in type.fields:
                align = _alignment_of(field.type)
                offset = (offset + align - 1) // align * align
                offset += _size_of(field.type)
            align = _alignment_of(type)
            return (offset + align - 1) // align * align
        case _:
            raise SpyError(f"type {type_str(type)} has no layout")


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


def function_call_info(type: FunctionType) -> FunctionCallInfo:
    """The lowering plan of one call of a function of this signature
    into MIR: how the by-value arguments map onto the positions of the
    lowered MIR argument list, and how the result is returned.  This is
    the single place the lowered form of a call is derived from the
    *function type* (so a future call through a function pointer lowers
    identically): every argument keeps its position, and a return type
    delivered through a result location (see :func:`returns_via_result_ptr`)
    appends the hidden result pointer as the trailing MIR argument."""
    n = len(type.args)
    args_map = tuple(FunctionCallArgInfo(i) for i in range(n))
    if returns_via_result_ptr(type.return_type):
        return FunctionCallInfo(n + 1, args_map, FunctionRetLocReturnInfo(arg_index=n))
    return FunctionCallInfo(
        n, args_map, FunctionValueReturnInfo(mir_type=to_mir_type(type.return_type))
    )


# ---------------------------------------------------------------------------
# mirroring spy types into MIR types: the only place MIR types are produced
# from spy types (valid spy types always mirror to valid MIR - ``lower`` maps
# MIR onto LLVM the same one-way way)
# ---------------------------------------------------------------------------


def to_mir_type(type: Type) -> mir.Type:
    """The MIR mirror of a spy type: the static type the runtime register
    of a value of ``type`` has.  The mapping is one-to-one over the types
    that can cross into runtime code.  A spy struct type mirrors to one
    :class:`mir.StructType` object (created lazily and cached on the
    descriptor), so that all values of one struct share one identity."""
    match type:
        case BoolType():
            return mir.BoolType()
        case IntType():
            return mir.IntType(type.bits, type.signed)
        case FloatType():
            return mir.FloatType(type.bits)
        case VoidType():
            return mir.VoidType()
        case StructType():
            return struct_mir_type(type)
        case PointerType(elem, _):
            # const-ness is not tracked in the MIR
            return mir.PointerType(to_mir_type(elem))
        case FunctionType(args, ret):
            return mir.FunctionType(tuple(to_mir_type(a.type) for a in args), to_mir_type(ret))
        case _:
            raise CompileError(f"spy type {type!r} has no MIR representation")


def struct_mir_type(type: StructType) -> mir.Type:
    """The (cached) MIR mirror of one spy struct type: fields in
    declaration order, mirroring the LLVM layout of the struct."""
    ret = type._mir
    if ret is not None:
        return ret
    fields: list[mir.FormalArg] = []
    for field in type.fields:
        fields.append(mir.FormalArg(field.name, to_mir_type(field.type)))
    ret = mir.StructType(type, tuple(fields))
    ret.ctype = type._py_cls
    type._mir = ret
    return ret


# ---------------------------------------------------------------------------
# mapping Python values to spy types
# ---------------------------------------------------------------------------


def value_type(value: object) -> Type | None:
    """The spy type a Python *value* is marshaled to at the call boundary.

    ``None`` is returned for values that have no spy representation (e.g.
    compile-time objects like type descriptors, which never cross the
    boundary).
    """
    match value:
        case bool():
            return BoolType()
        case int():
            return IntType(INT_DEFAULT_BITS, True)
        case float():
            return FloatType(64)
        case str():
            # strings are compiled as arrays of u8; until arrays get their
            # own type they are represented by a const pointer to u8
            return PointerType(IntType(8, False), is_const=True)
        case _:
            # a struct instance knows its spy struct type: its Python
            # class (built by the host) carries a back reference
            cls = type(value)
            descriptor = getattr(cls, '__spy_struct_type__', None)
            if descriptor is not None:
                return descriptor
            return None


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

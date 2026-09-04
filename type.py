"""The spy type system of the compile-time interpreter.

Types appear in two roles:

* as the type annotation values in AOT functions (``spy.u64``,
  ``spy.f64``, ...), and
* as compile-time values inside a function body (``spy.type(a) ==
  spy.u64``).

The static types attached to the registers of the typed MIR are the
mirrors of these types defined by ``mir``; the interpreter converts
between the two when it emits instructions.

Types are immutable and compare structurally (two ``IntType(64, False)``
are equal), which is what makes the compile-time comparisons in
``spy.type(a) == spy.u64`` work.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, override

from .errors import SpyError

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.type``) and other compile-time objects.
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
    ``spy.type(x) == Foo`` work.
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
        self._mir: Any | None = None

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

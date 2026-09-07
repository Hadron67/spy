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

import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, override

from . import mir
from .errors import CompileError, SpyError, TypeMismatchError

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.typeof``) and other compile-time objects.
    Concrete values expose their spy type as ``.type``."""
    @abstractmethod
    def get_type(self) -> 'Type':
        raise NotImplementedError

AnyValue = Value | int | float | str | bool

class Type(Value):
    def get_unit_value(self) -> Value | None:
        """The canonical *unit value* of a zero-sized type (ZST): ``None``
        when the type has a runtime representation (it is not
        zero-sized), otherwise the one compile-time value every value of
        the type equals - ``Void()`` for the void type, ``Int(0, T)`` for
        a zero-bit integer, an ``AggregateValue`` for a struct whose
        fields are all ZSTs.  A ZST has no runtime representation: its
        ``mir`` mirror is ``None`` (``to_mir_type`` returns ``None``)."""
        return None

@dataclass(frozen=True)
class TypeType(Type):
    level: int
    @override
    def get_type(self) -> Type:
        return TypeType(self.level + 1)


class TypeVar(Type):
    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def __eq__(self, value: object, /) -> bool:
        return self is value

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)


TYPE_TYPE = TypeType(0)

@dataclass(frozen=True)
class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE


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


class Void(Value):
    """The unique *value* of the unit type :class:`VoidType` (which is a
    zero-sized type): the compile-time object that denotes "no value" -
    the result of a void call, the yield of a void inlined body, ...  It
    replaces the ``None`` sentinel the interpreter used for these."""

    @override
    def get_type(self) -> Type:
        return VoidType()

class BuiltinFn(Value):
    @override
    def get_type(self) -> 'Type':
        return AnyFunction()

@dataclass
class ComptimeIntType(Type):
    def get_type(self) -> 'Type':
        return TYPE_TYPE

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

@dataclass(frozen=True)
class Int(Value):
    value: int
    type: IntType

    @override
    def get_type(self) -> Type:
        return self.type

@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __post_init__(self) -> None:
        assert self.bits in (32, 64), f"unsupported float bits {self.bits}"

    @override
    def get_type(self) -> Type:
        return TYPE_TYPE

@dataclass(frozen=True)
class Float(Type):
    value: float
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type


@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: bool = False

    @override
    def get_type(self) -> Type:
        child = self.elem.get_type()
        assert isinstance(child, TypeType)
        return TypeType(child.level + 1)


@dataclass(frozen=True)
class Undefined(Value):
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

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

@dataclass(frozen=True)
class StructField:
    name: str
    type: Type

@dataclass(frozen=True)
class AggregateValue(Value):
    values: tuple[Value, ...]
    type: Type

    @override
    def get_type(self) -> Type:
        return self.type

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

    def bind_default_ctor_args[T](self, positional: tuple[T, ...], named: dict[str, T]) -> tuple[T, ...]:
        ret: list[T | None] = []
        # TODO
        raise NotImplementedError

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
    def get_type(self) -> Type:
        level = 0
        for field in self._fields:
            child = field.type.get_type()
            assert isinstance(child, TypeType)
            level = max(level, child.level)
        return TypeType(level)

    @override
    def get_unit_value(self) -> Value | None:
        values: list[Value] = []
        for field in self._fields:
            val = field.type.get_unit_value()
            if val is None:
                return None
            values.append(val)
        return AggregateValue(tuple(values), self)

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
                field_type = to_mir_type(field.type)
                if isinstance(field_type, mir.VoidType):
                    field_mir_indices.append(None)
                else:
                    field_mir_indices.append(len(fields))
                    fields.append(mir.FormalArg(field.name, field_type))
            self._field_mir_indices = tuple(field_mir_indices)
            if all(a is None for a in self._field_mir_indices):
                self._mir = None
            else:
                self._mir = mir.StructType(self, tuple(fields))

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
    def get_type(self) -> 'Type':
        return TYPE_TYPE

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
        case TypeVar():
            return type.name
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
            raise SpyError(f"type {type_str(type)} has no layout")


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


def function_call_info(type: FunctionType) -> tuple[FunctionCallInfo, mir.FunctionType]:
    """The lowering plan of one call of a function of this signature
    into MIR: how the by-value arguments map onto the positions of the
    lowered MIR argument list, and how the result is returned.  This is
    the single place the lowered form of a call is derived from the
    *function type* (so a future call through a function pointer lowers
    identically): every argument keeps its position, a parameter whose
    type has no runtime representation (``to_mir_type`` is ``None`` - a
    zero-sized type) occupies no position and is not passed (it has no
    runtime representation; the callee binds its own unit value), and a
    return type delivered through a result location (see
    :func:`returns_via_result_ptr`) appends the hidden result pointer as
    the trailing MIR argument.

    Besides the call plan, the *lowered* MIR signature of the function
    is returned: the :class:`mir.FunctionType` the lowered argument
    list and return type form (the form a :class:`mir.Symbol` or
    function value of the signature carries)."""
    args_map: list[FunctionCallArgInfo | None] = []
    mir_args: list[mir.Type] = []
    for arg in type.args:
        mir_type = to_mir_type(arg.type)
        if isinstance(mir_type, mir.VoidType):
            args_map.append(None)
        else:
            args_map.append(FunctionCallArgInfo(len(mir_args)))
            mir_args.append(mir_type)
    if returns_via_result_ptr(type.return_type):
        # the result is written into a caller-provided location whose
        # address is appended as the trailing MIR argument; the lowered
        # function returns void
        result_type = to_mir_type(type.return_type)
        assert not isinstance(result_type, mir.VoidType)
        return FunctionCallInfo(
            len(mir_args) + 1,
            tuple(args_map),
            FunctionRetLocReturnInfo(arg_index=len(mir_args)),
        ), mir.FunctionType(tuple(mir_args) + (mir.PointerType(result_type),), mir.VOID)
    return (
        FunctionCallInfo(
            len(mir_args),
            tuple(args_map),
            FunctionValueReturnInfo(mir_type=to_mir_type(type.return_type)),
        ),
        mir.FunctionType(tuple(mir_args), to_mir_type(type.return_type)),
    )


# ---------------------------------------------------------------------------
# mirroring spy types into MIR types: the only place MIR types are produced
# from spy types (valid spy types always mirror to valid MIR - ``lower`` maps
# MIR onto LLVM the same one-way way)
# ---------------------------------------------------------------------------


def to_mir_type(type: Type) -> mir.MayBeVoidType:
    """The MIR mirror of a spy type: the static type the runtime register
    of a value of ``type`` has.  The mapping is one-to-one over the types
    that can cross into runtime code.  A zero-sized type has no runtime
    representation and mirrors to ``None`` (the MIR's "void": a function
    whose return type is a ZST returns void); a spy struct type mirrors
    to one :class:`mir.StructType` object (created lazily and cached on
    the descriptor), so that all values of one struct share one
    identity.  Types with no MIR mirror at all (``TypeType``,
    ``AnyFunction``, ...) are a compile error."""
    match type:
        case BoolType():
            return mir.BoolType()
        case IntType():
            return mir.VOID if type.bits == 0 else mir.IntType(type.bits, type.signed)
        case FloatType():
            return mir.FloatType(type.bits)
        case VoidType():
            return mir.VOID
        case StructType():
            return type.get_mir_type()
        case PointerType():
            return mir.PointerType(to_mir_type(type.elem), type.is_const)
        case FunctionType():
            args: list[mir.Type] = []
            for arg in type.args:
                mir_type = to_mir_type(arg.type)
                if not isinstance(mir_type, mir.VoidType):
                    args.append(mir_type)
            return mir.FunctionType(tuple(args), to_mir_type(type.return_type))
        case _:
            raise CompileError(f"spy type {type!r} has no MIR representation")

# ---------------------------------------------------------------------------
# mapping Python values to spy types
# ---------------------------------------------------------------------------


def type_of(value: Any) -> Type | None:
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
            return ComptimeIntType()
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


def _subtype_conflict(lhs: Value, rhs: Value) -> TypeMismatchError:
    return TypeMismatchError(
        f'conflicting types {_type_or_repr(lhs)} and {_type_or_repr(rhs)}'
    )


def _type_or_repr(value: Value) -> str:
    if isinstance(value, Type):
        return type_str(value)
    return repr(value)


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
        self._subtypes: set[Value] | None = None  # non-None: this type var is a subtype of all values in the set, in this case _value is None

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
        self._constraints: list[_Constraint] = []

    def _solved(self, tv: TypeVar) -> _SolvedTypeVar:
        stv = self._type_var_values.get(tv)
        if stv is None:
            stv = _SolvedTypeVar()
            self._type_var_values[tv] = stv
        return stv

    def _whnf_one(self, value: Value) -> Value:
        if isinstance(value, TypeVar) and value in self._type_var_values:
            val = self._type_var_values[value]._value
            if val is not None:
                return val
        return value

    def _whnf(self, value: Value) -> Value:
        next = self._whnf_one(value)
        while next != value:
            value = next
            next = self._whnf_one(value)
        return value

    def add_constraint(self, lhs: Value, rhs: Value, is_subtype: bool = False) -> None:
        self._constraints.append(_Constraint(lhs, rhs, is_subtype))

    def finish(self) -> None:
        """Solve the pending constraints, binding the type parameters to
        their solutions.  Raises :class:`TypeMismatchError` when a
        constraint cannot be satisfied."""
        pending = self._constraints
        self._constraints = []
        # equality constraints are solved eagerly; a constraint that
        # cannot be decided in one pass (its type parameter only carries
        # a subtype bound that is not resolved yet) is retried after the
        # rest of the pass has made its progress
        while len(pending) > 0:
            leftover: list[_Constraint] = []
            progress = False
            for constraint in pending:
                if self._solve_one(constraint):
                    progress = True
                else:
                    leftover.append(constraint)
            if not progress:
                raise TypeMismatchError(
                    'cannot solve the remaining type constraints'
                )
            pending = leftover
        # bind the type parameters that only subtype constraints
        # constrain (their bounds are recorded on the parameter; a
        # bound that is itself a type parameter is followed to its own
        # solution)
        while True:
            progress = False
            for tv, stv in self._type_var_values.items():
                if stv._value is not None or stv._subtypes is None:
                    continue
                bounds = {self._whnf(b) for b in stv._subtypes}
                if len(bounds) > 1:
                    raise _subtype_conflict(*(tuple(bounds)[:2]))
                val = next(iter(bounds))
                if isinstance(val, TypeVar) and self._solved(val)._value is None:
                    # the sole bound is itself unsolved: it is resolved
                    # in a later pass of this loop
                    continue
                stv._value = val
                progress = True
            if not progress:
                # every remaining bounded parameter has an unsolved
                # bound, so no value can be assigned
                for tv, stv in self._type_var_values.items():
                    if stv._value is None and stv._subtypes is not None:
                        raise _subtype_conflict(tv, next(iter(stv._subtypes)))
                return

    def get_solved(self) -> dict[TypeVar, Value]:
        return {
            k: self._whnf(v._value)
            for k, v in self._type_var_values.items()
            if v._value is not None
        }

    def _solve_one(self, constraint: _Constraint) -> bool:
        """Solve one constraint. Returns True when the constraint is
        fully handled, False when it cannot be decided yet (its type
        parameter only carries a subtype bound that is not resolved) and
        the caller must retry it."""
        lhs = self._whnf(constraint.lhs)
        rhs = self._whnf(constraint.rhs)
        if lhs == rhs:
            return True
        if constraint.is_subtype:
            # a subtype constraint involving a type parameter records
            # the other side as a bound of the parameter; between two
            # resolved values it can only hold when they are equal
            if isinstance(lhs, TypeVar):
                self._solved(lhs)._add_bound(rhs)
                return True
            if isinstance(rhs, TypeVar):
                # ``lhs <: T``: under the trivial subtype relation this
                # constrains T to lhs, which the bound records
                self._solved(rhs)._add_bound(lhs)
                return True
            raise _subtype_conflict(lhs, rhs)
        # an equality constraint: bring the type parameter (if any) to
        # the left hand side
        if isinstance(rhs, TypeVar) and not isinstance(lhs, TypeVar):
            t = lhs
            lhs = rhs
            rhs = t
        if not isinstance(lhs, TypeVar):
            # neither side is a type parameter: two distinct resolved
            # values can never become equal
            raise _subtype_conflict(lhs, rhs)
        stv = self._solved(lhs)
        # the solution must satisfy the subtype bounds recorded on the
        # parameter
        if stv._subtypes is not None:
            for bound in stv._subtypes:
                b = self._whnf(bound)
                if isinstance(b, TypeVar) and not isinstance(rhs, TypeVar):
                    # the bound is an unsolved type parameter: whether
                    # rhs satisfies it is only known once it resolves
                    return False
                if b != rhs:
                    raise _subtype_conflict(rhs, b)
        if isinstance(rhs, TypeVar):
            # an equality between two type parameters: solving the lhs
            # to the rhs chains the two; when the rhs carries a subtype
            # bound, the equality is only decidable once it resolves
            if self._solved(rhs)._subtypes is not None:
                return False
            stv._value = rhs
            return True
        stv._value = rhs
        return True

def replace_type_var(value: Value, reps: dict[TypeVar, Value]) -> Value:
    match value:
        case TypeVar():
            return reps.get(value, value)
        case _:
            return value

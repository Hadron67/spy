"""Compile-time interpretation of the HIR ("running" the HIR).

The interpreter executes the linear HIR instruction stream of a function
against the concrete argument types, emitting the typed MIR along the
way.  Every executed HIR instruction that is used as an operand leaves a
value in a register table keyed by the instruction object itself,
mirroring how ``llvm`` registers work: operands of later instructions
are references to earlier instruction objects.  An instruction that
writes through a result location (``Binary``, ``Unary``,
``BinaryAssign``, ``CallInplace``, ``CallMethodInplace``) produces no
register of its own.

Values in the register table are either

* :class:`ComptimeVal` - a compile-time value of the ``spy`` domain
  (an ``sval.AnyValue``: a Python scalar, a spy type descriptor, a
  function to call/inline, ...).  "No value" is the unit value
  ``sval.Void()`` - the unique value of the zero-sized void type - never
  Python ``None``,
* :class:`ComptimeTuple`/:class:`ComptimeDict` - a compile-time
  aggregate whose elements are themselves interpreter values,
* :class:`RuntimeVal` - the object of an already emitted MIR
  instruction (a typed runtime value),
* :class:`PendingSlot` - an executed ``Alloca`` that is only committed
  (into real memory, or into a :class:`ComptimeBox`) when
  ``hir.CommitSlot`` runs; a slot whose content type is zero-sized never
  gets memory: it only records its unit value, or
* :class:`ComptimeBox` - the compile-time memory a committed
  compile-time slot materializes into (a pointer to a compile-time
  value).

Instructions whose operands are all compile-time values are evaluated
eagerly in Python (the comptime semantics of the DSL); instructions
with runtime operands emit typed MIR.  A compile-time value flows into
runtime code only by being converted to a typed constant of the type
the runtime operation expects.  The interpreter types everything in the
``spy`` type system of ``sval`` and *mirrors* the spy types into MIR
only when an instruction is emitted (``sval.Type.to_mir_type``): it
never reads
the MIR types of the values it produced back for a decision - a valid
spy type always lowers to a valid MIR type (open loop), exactly like
``lower`` maps MIR onto LLVM without reading LLVM types back.

Calls are dispatched at compile time:

* calls to the ``spy`` builtins (``spy.typeof``, ``spy.compile_log``) are
  evaluated eagerly,
* calls to other spy functions (jit or aot) become native ``call``
  instructions to the specialization selected by the argument types,
* calls to plain Python functions inline the callee body into the
  current stream.

An inlined body is delimited in the emitted MIR by a ``Block``/``End``
pair, and it may contain runtime ``if`` branches like the function
proper.  Every return of the body stores its value into the call's
result location on its own runtime path, whatever the path is, and a
return inside a runtime branch leaves the body with a ``Break`` out of
its ``Block`` - the result location's memory is the join of the paths
(its alloca is hoisted by ``lower`` so every path shares one address).
A body whose paths all deliver one value stores only once, right
before its ``End``; a single-path body's store/load round trip is
cleaned up afterwards by ``opt``.

Both kinds of function calls push a *frame* holding the by-value
arguments (resolved by ``hir.Arg`` leaves): an inlined plain-Python
callee pushes it on the current runner, and a called spy function gets a
runner of its own instead (the caller suspends until that runner has
typed it, see ``Analyser._request_function``).  The interpreter types an
``Alloca`` when its first store executes, so the untyped HIR needs no
type information of its own.
"""

import operator
import types as pytypes
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Self, override

from . import hir, mir, sval
from .binop import BinaryOp, BoolOp, CompareOp, UnaryOp
from .errors import CompileError
from .fn import (
    ArgEntry,
    ArgList,
    CallSignature,
    CompileBatch,
    FunctionInstance,
    FunctionValue,
    NativeFn,
    RawArgList,
    ReturnSignature,
    SpecializedComptimeArg,
    SpecializedFormalArg,
    SpecializedRuntimeArg,
)
from .sval import GlobalResolver
from .util import frozendict

_MAX_INLINE_DEPTH = 64

_PY_OPS: dict[str, Any] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '**': operator.pow,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

class InterpVal:
    pass

@dataclass
class ComptimeVal(InterpVal):
    obj: sval.AnyValue


@dataclass
class RuntimeVal(InterpVal):
    """A value of the already emitted typed MIR.  The interpreter's own
    knowledge of the static type of the value lives here in the ``spy``
    type system (``sval``) - the MIR type of the value is only ever
    *produced* from it (``sval.Type.to_mir_type``), never read back for a
    decision."""

    value: mir.Value
    type: sval.Type


@dataclass
class ComptimeBox(InterpVal):
    """A comptime-time writable box. Note that ``value`` does not have to
    be a comptime-time value: it also can be a runtime value :class:`RuntimeVal`."""

    type: sval.Type
    value: InterpVal


class _PendingActionData:
    """One action recorded by a :class:`PendingSlot`: how it is delivered
    once the slot has an address is decided by the runner (see
    ``HirRunner._exec_pending_action``), the slot itself only asks for the
    type it contributes and whether it is compile-time."""

    @abstractmethod
    def info(self) -> tuple[sval.Type, bool]:
        """Returns (type, is_comptime)"""
        ...

@dataclass
class _PendingAction:
    """A pending action together with the MIR insertion block it is
    delivered into: the block sits at the position the action was recorded
    at, so it is spliced into the body after it has been filled."""

    insertion: mir.Insertion
    data: _PendingActionData

@dataclass
class _PendingStore(_PendingActionData):
    """One store point recorded by a :class:`PendingSlot`: the spy type of
    the stored value and whether it is compile-time.  The store itself is
    delivered once the slot's final type is known (the stored value is
    coerced to it then)."""

    type: sval.Type
    is_comptime: bool
    value: InterpVal

    def info(self) -> tuple[sval.Type, bool]:
        return self.type, self.is_comptime

@dataclass
class _PendingPtrConvertion(_PendingActionData):
    type: sval.Type
    input: InterpVal
    output: mir.Insertion

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return self.type, False

@dataclass(slots=True)
class PendingSlot(InterpVal):
    """The value of an executed ``hir.Alloca`` before it is *committed*.
    In this phase a store (or an RLS call) into the slot only records a
    :class:`_PendingAction`; the slot acquires its final type (the
    pairwise ``resolve_peer_type`` of the action types) and its storage
    when ``hir.CommitSlot`` runs, which materializes it into a
    :class:`RuntimeVal` (a pointer to real memory) or a
    :class:`ComptimeBox` (see ``HirRunner._commit_pending_slot``).

    ``allow_inline`` marks the slots astgen allocates for an expression
    temporary (``Alloca(True)``): a temporary whose stores are all
    compile-time becomes a :class:`ComptimeBox` instead of memory.

    ``insertion`` is the position the slot's storage is produced at: a
    :class:`mir.Insertion` emitted where the ``Alloca`` ran, whose
    instructions the slot's commit fills with the :class:`mir.Alloca` (or,
    for an element/field place of an aggregate construction, with the
    ``Gep`` addressing it, see ``finish_array``/``finish_struct``)."""

    insertion: mir.Insertion
    allow_inline: bool
    stores: list[_PendingAction] = field(default_factory=list)
    committed: InterpVal | None = None

    def committed_type(self) -> sval.Type:
        type: sval.Type | None = None
        for store in self.stores:
            slot_type, _ = store.data.info()
            if type is None:
                type = slot_type
            else:
                peer = type.resolve_peer_type(slot_type)
                if peer is None:
                    raise CompileError(
                        f"a slot is stored with incompatible types {type} and {slot_type}"
                    )
                type = peer
        if type is None:
            type = sval.EmptyType()
        return type

    def is_comptime(self) -> bool:
        return self.allow_inline and all(store.data.info()[1] for store in self.stores)


@dataclass(frozen=True, slots=True)
class ComptimeTuple(InterpVal):
    values: tuple[ArgEntry[InterpVal], ...]

@dataclass
class ComptimeDict(InterpVal):
    values: dict[str, ArgEntry[InterpVal]]

def _is_comptime_val(val: InterpVal) -> bool:
    todo = [val]
    while todo:
        val = todo.pop()
        match val:
            case RuntimeVal():
                return False
            case PendingSlot():
                if val.committed is None:
                    return False
                todo.append(val.committed)
            case ComptimeTuple():
                todo.extend(a.value for a in val.values)
            case ComptimeDict():
                todo.extend(a.value for a in val.values.values())
    return True

class BlockFrameData:
    pass

@dataclass
class IfBlockData(BlockFrameData):
    chosen: bool | None = None
    then_returns: bool | None = None

@dataclass
class BlockFrame:
    entry: int
    data: BlockFrameData

class InlineFrame:
    def __init__(self, generic_var_values: dict[sval.TypeVar, InterpVal], arg_values: tuple[InterpVal, ...], ret_loc: InterpVal, insts: tuple[hir.Inst, ...]) -> None:
        self.generic_var_values = generic_var_values
        self.arg_values = arg_values
        self.ret_loc = ret_loc
        self.insts = insts
        self.pc: int = 0
        self.block_stack: list[BlockFrame] = []
        self.regs: dict[hir.Inst, InterpVal] = {}

    def ret_levels(self):
        open_ifs = 0
        for bf in self.block_stack:
            data = bf.data
            match data:
                case IfBlockData():
                    if data.chosen is None:
                        open_ifs += 1
                case _:
                    raise AssertionError('unexpected block frame data')
        return open_ifs + 1 if open_ifs > 0 else 0

# ---------------------------------------------------------------------------
# stateless helpers of the interpreter: pure functions over their arguments
# (argument/prototype construction, Python-literal constants, compile-time
# operators and operator error messages) - none of them uses instance state,
# so none of them is a method of :class:`HirRunner`
# ---------------------------------------------------------------------------


def _sval_to_runtime(value: sval.AnyValue) -> mir.Value:
    match value:
        case bool():
            return mir.BoolValue(value)
        case sval.Int():
            return mir.Int(value.value, mir.IntType(value.type.bits, value.type.signed))
        case sval.Float():
            return mir.Float(value.value, mir.FloatType(value.type.bits))
        case _:
            raise CompileError(f"cannot return the compile-time value {value!r}")


def _to_runtime(ev: InterpVal) -> mir.Value:
    """Materialize a value as a typed MIR value: a runtime value yields
    its MIR object, a compile-time value a constant built from it
    (``_sval_to_runtime``), and a committed slot the value it was
    materialized into; an uncommitted slot or a compile-time box is
    rejected."""
    match ev:
        case RuntimeVal():
            return ev.value
        case PendingSlot():
            if ev.committed is None:
                raise CompileError('cannot use an uncommitted slot as a runtime value')
            return _to_runtime(ev.committed)
        case ComptimeVal():
            return _sval_to_runtime(ev.obj)
        case ComptimeBox():
            raise CompileError('cannot use a compile-time box as a runtime value')
    raise CompileError('cannot return this value')

def _to_comptime(value: InterpVal) -> sval.AnyValue | None:
    match value:
        case ComptimeVal():
            return value.obj
        case _:
            return None

def _shallow_normalize(ev: InterpVal) -> InterpVal:
    """Follow a committed :class:`PendingSlot` to the pointer value it was
    materialized into (a :class:`RuntimeVal` pointer or a
    :class:`ComptimeBox`), so that pointer operations see one shape.  An
    uncommitted slot is left alone for ``load``/``store`` to handle."""
    if isinstance(ev, PendingSlot) and ev.committed is not None:
        return ev.committed
    return ev

def _type_of(ev: InterpVal, allow_value_type: bool = False) -> sval.Type | None:
    """The spy type of the value ``ev`` denotes, or None when it has
    no spy representation (an un-typable compile-time object).
    Should work on non-normalized values."""
    match ev:
        case PendingSlot():
            if ev.committed is None:
                return None
            return _type_of(ev.committed, allow_value_type)
        case ComptimeBox():
            # a compile-time writable pointer
            return sval.PointerType(ev.type, is_const=False)
        case RuntimeVal(_, type):
            if isinstance(type, sval.ValueType) and not allow_value_type:
                return sval.type_of(type.value)
            return type
        case ComptimeVal(obj):
            return sval.type_of(obj) if not allow_value_type else sval.ValueType(obj)
        case _:
            return None

def _arg_type_of(arg: ArgEntry[InterpVal]) -> sval.Type | None:
    """The spy type of the *value* an argument denotes: a reference
    argument carries the address of its value, so one pointer layer is
    stripped here.  ``None`` when the value has no spy type yet (a slot
    that is not committed, an un-typable compile-time object)."""
    type = _type_of(arg.value)
    if not arg.is_ref:
        return type
    if type is None:
        return None
    assert isinstance(type, sval.PointerType), f"pointer expected, got {type}"
    return type.elem

def _struct_generic_var_values(struct: sval.StructType) -> frozendict[sval.TypeVar, sval.Value]:
    """The type-argument values of one struct *specialization*: its generic
    type parameters -> the values this specialization binds them to (empty
    for a non-generic struct).  A method resolved through the struct carries
    them, so that a call can substitute them into the method's signature
    (see :class:`sval.BoundMethod`)."""
    return frozendict(zip(struct.head.generic_args, struct.generic_args))

# ---------------------------------------------------------------------------
# stateless helpers of field/element access and struct/array construction:
# pure functions over the interpreter values they are given (the struct/array
# structure is read off those, not off any runner state)
# ---------------------------------------------------------------------------


def _index_value(index: int) -> InterpVal:
    """A compile-time field/element index as an interpreter value."""
    return ComptimeVal(sval.Int(index, sval.IntType(64, False)))

def _comptime_index(index: InterpVal) -> int:
    """The Python integer a compile-time index denotes (a struct field is
    always named by a compile-time index)."""
    if isinstance(index, ComptimeVal) and isinstance(index.obj, sval.Int):
        return index.obj.value
    raise CompileError('a struct field index must be a compile-time integer')

def _mir_index(index: InterpVal) -> int | mir.Value:
    """The MIR index an element index denotes: a constant when it is known
    at compile time, the runtime value otherwise."""
    if isinstance(index, ComptimeVal) and isinstance(index.obj, sval.Int):
        return index.obj.value
    if isinstance(index, RuntimeVal):
        return index.value
    raise CompileError('an array element index must be an integer')

def _callee_object(callee: InterpVal) -> sval.AnyValue | None:
    """The compile-time object a call callee - or the struct operand of a
    construction - denotes, or None when it denotes none.  A callee is a
    reference to a function value, a builtin or a struct; a subscripted
    struct template (``Foo[i32]``) is materialized by ``astgen._as_ref`` into a
    compile-time box when it is a callee, so a boxed callee is unwrapped
    here."""
    todo = [callee]
    while todo:
        ev = todo.pop()
        match ev:
            case ComptimeVal(obj):
                return obj.value if isinstance(obj, sval.ConstRef) else obj
            case ComptimeBox():
                todo.append(ev.value)
            case PendingSlot() if ev.committed is not None:
                todo.append(ev.committed)
            case _:
                return None
    return None

def _place_type(place: InterpVal) -> sval.Type | None:
    """The type of the value a place holds: a slot whose type is not decided
    yet reports the peer type of the values stored into it (and None when it
    holds no value), a place that is materialized the type of the value it
    holds."""
    place = _shallow_normalize(place)
    if isinstance(place, PendingSlot):
        if place.committed is None:
            type = place.committed_type()
            return None if isinstance(type, sval.EmptyType) else type
        place = place.committed
    type = _type_of(place)
    if isinstance(type, sval.PointerType):
        return type.elem
    return None

def _array_elem_type_of(value: InterpVal) -> sval.ArrayType | None:
    """The array type ``value`` points at, or None when it points at
    something else (or at nothing: a slot whose type is not decided yet)."""
    type = _type_of(value)
    if isinstance(type, sval.PointerType) and isinstance(type.elem, sval.ArrayType):
        return type.elem
    return None

def _common_element_type(elements: tuple[InterpVal, ...]) -> sval.Type:
    """The one type all the elements of an array have: the peer type of the
    type of every element place (see ``_place_type``).  An array is
    homogeneous, so the elements have to agree; every element is coerced to
    that type when its value is written."""
    type: sval.Type | None = None
    for element in elements:
        element_type = _place_type(element)
        if element_type is None:
            continue
        peer = element_type if type is None else type.resolve_peer_type(element_type)
        if peer is None:
            raise CompileError(
                f'the elements of an array must have a common type, '
                f'got {type} and {element_type}'
            )
        type = peer
    if type is None:
        raise CompileError(
            'the elements of an array with no element have no type of their '
            'own: the place it is built in has to declare the array type'
        )
    return type

def _array_construction_type(array: InterpVal, elements: tuple[InterpVal, ...]) -> sval.ArrayType:
    """The array type a construction builds: as many elements as it was
    given (the constructor has no way to name the length, see ``syntax``),
    of the element type the storage ``array`` points at declares - a place
    holds one type, so the array built in it has to agree with it - or else
    of the common type of the elements themselves."""
    array_type = _array_elem_type_of(array)
    if array_type is None:
        return sval.ArrayType(_common_element_type(elements), len(elements))
    declared = array_type.length_int
    if declared is None:
        raise CompileError(f'cannot tell how many elements {array_type} holds')
    if declared != len(elements):
        raise CompileError(
            f'{array_type} holds {declared} element(s), but {len(elements)} were given'
        )
    return array_type

def _infer_struct_generic_args(
    head: sval.StructTypeHead, fields: dict[int, InterpVal]
) -> sval.StructType:
    """Infer the generic arguments of a struct construction that names the
    bare template (``Foo(...)``) from the values written into its fields,
    exactly like a generic call types its type parameters: every provided
    field constrains the (type-parameter-valued) declared type of the field
    to the type of the value written into it.  A field the construction
    leaves out - a field of a zero-sized type - constrains nothing."""
    declared = head.fields
    solver = sval.TypeVarSolver()
    for index, place in fields.items():
        if index < 0 or index >= len(declared.by_id):
            continue
        place_type = _place_type(place)
        if place_type is not None:
            solver.add_constraint(place_type, declared.get_by_id(index).type, True)
    solver.finish()
    solved = solver.get_solved()
    generic_args: list[sval.Value] = []
    for type_var in head.generic_args:
        if type_var not in solved:
            raise CompileError(
                f'cannot infer the generic argument {type_var.name} of struct '
                f'{head.name_base} from this construction; give it '
                f'explicitly, e.g. {head.name_base}[...](...)'
            )
        arg = solved[type_var]
        assert isinstance(arg, sval.Value), 'a struct type argument is a value'
        generic_args.append(arg)
    return head.specialize(tuple(generic_args))

def _struct_construction_type(
    struct: sval.AnyValue | None,
    dest: InterpVal,
    fields: dict[int, InterpVal],
) -> sval.StructType:
    """The struct type a construction builds: ``struct`` itself when it
    names a specialization.  A *template* (the bare name ``Foo``) names a
    :class:`sval.StructTypeHead`, which has no type of its own: the generic
    arguments are the ones the destination was already specialized with (the
    result location of a function whose return type is declared, an existing
    struct value, ...), or else the ones the provided field values determine
    (see ``_infer_struct_generic_args``)."""
    if isinstance(struct, sval.StructType):
        return struct
    if isinstance(struct, sval.StructTypeHead):
        if dest is not None:
            dest_type = _type_of(dest)
            elem = dest_type.elem if isinstance(dest_type, sval.PointerType) else None
            if isinstance(elem, sval.StructType) and elem.head is struct:
                return elem
        return _infer_struct_generic_args(struct, fields)
    raise CompileError(f'{struct!r} is not a struct')

def _no_runtime_type(type: sval.Type) -> CompileError:
    """The error for a runtime location whose type has no representation
    of its own.  A compile-time-only type (the type of an untyped integer
    literal) is called out by name: the location has to declare its type,
    since such a literal is only ever resolvable at compile time."""
    if sval.is_comptime_only_type(type):
        return CompileError(
            f'{type} is the type of an untyped literal: a runtime location '
            f'cannot hold it and must declare its type'
        )
    return CompileError(f'cannot give a value of type {type} a runtime representation')

# ---------------------------------------------------------------------------
# compile-time operators: pure functions over the spy values of the
# operands (used when every operand of an operation is a compile-time value)
# ---------------------------------------------------------------------------


def _comptime_py_op(op: str, lhs: Any, rhs: Any) -> Any:
    """Apply the Python operator ``op`` to two compile-time values."""
    fn = _PY_OPS.get(op)
    if fn is None:
        raise CompileError(f"operator '{op}' is not supported at compile time")
    try:
        return fn(lhs, rhs)
    except Exception as e:
        raise CompileError(
            f"cannot apply '{op}' to {lhs!r} and {rhs!r} at compile time: {e}"
        ) from e

def _convert_inst(
    value: mir.Value, from_type: sval.Type, to_type: sval.Type
) -> mir.Inst | None:
    """Build (but do not emit) the conversion of ``value`` from
    ``from_type`` to ``to_type``; returns ``None`` when no conversion
    instruction is needed (the types are equal, or both are pointers)."""
    if from_type == to_type:
        return None
    mir_to_type = to_type.to_mir_type()
    assert mir_to_type is not None and not isinstance(mir_to_type, mir.VoidType)
    if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.IntType):
        if from_type.bits < to_type.bits:
            kind = 'sext' if from_type.signed else 'zext'
        else:
            kind = 'trunc'
        return mir.Convert(kind, value, mir_to_type)
    if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.FloatType):
        kind = 'sitofp' if from_type.signed else 'uitofp'
        return mir.Convert(kind, value, mir_to_type)
    if isinstance(from_type, sval.FloatType) and isinstance(to_type, sval.FloatType):
        kind = 'fpext' if from_type.bits < to_type.bits else 'fptrunc'
        return mir.Convert(kind, value, mir_to_type)
    if isinstance(from_type, sval.PointerType) and isinstance(to_type, sval.PointerType):
        if from_type.is_const and not to_type.is_const:
            raise CompileError(
                f"cannot convert a {from_type} value to {to_type}"
            )
        return None
    raise CompileError(
        f"cannot convert a {from_type} value to {to_type}"
    )

# ---------------------------------------------------------------------------
# the compile-time host interface
# ---------------------------------------------------------------------------

class ResultMode(IntEnum):
    VALUE = auto()
    INPLACE = auto()


class PollResult(IntEnum):
    AGAIN = auto()
    SUSPEND = auto()
    DONE = auto()

class HirRunner:
    """Runs one function body (and everything it inlines) at compile
    time, filling the pre-created typed :class:`mir.Function` of one
    specialization.

    The compile-time host is reached through the :class:`Analyser`
    (``analyser``) as ``self._analyser._resolver``: it is typed as the
    :class:`FunctionResolver` interface it implements (``dsl._Context``
    in practice) and resolves a global object referenced inside a
    function body to its spy value (its function entry, or ``None`` for
    anything that is not a spy object).  The nested specializations a
    body calls are requested through ``Analyser._request_function``.
    """

    def __init__(self, analyser: Analyser, fn_instance: FunctionInstance) -> None:
        self._analyser = analyser
        # the frames of the function bodies under execution: the function
        # proper at the bottom, one frame per inlined plain function
        # above it (see ``_in_function_proper``; each frame carries the
        # HIR of its body, see ``InlineFrame``)
        self._frames: list[InlineFrame] = []
        # the function proper whose body is currently being typed (see
        # ``_materialize_result_ptr``)
        self._fn_instance = fn_instance
        self._mir_block_stack: list[list[mir.Inst]] = [fn_instance.mir.insts]
        # the ``hir.Ret`` positions of the function proper whose return
        # convention is not fixed yet (an unannotated return type); their
        # ``mir.Ret`` is filled in by ``_finish_function`` once the result
        # location has been materialized
        self._deferred_returns: list[mir.Insertion] = []
        self.return_sig: ReturnSignature | None = None

        self._fn_req_resumer: Callable[[Self, mir.Value, ReturnSignature]] | None = None

    # -- entry point ---------------------------------------------------------

    def run_function(
        self,
        body: tuple[hir.Inst, ...],
        sig: CallSignature,
        ret_sig: ReturnSignature | None,
        generic_var_values: dict[sval.TypeVar, InterpVal],
    ):
        # reset the per-specialization state; the result location of the
        # function proper is reserved first so its slot sits at a known
        # position in the body
        self.return_sig = None
        self.resume_info = None
        self._deferred_returns = []
        ret_loc = self.alloca(False)
        frame = InlineFrame(generic_var_values, (), ret_loc, body)
        self._frames.append(frame)
        mir_args = self._fn_instance.mir.args
        args = self._init_args_from_signature(sig, mir_args)
        for arg in mir_args:
            assert arg is not None
        frame.arg_values = args
        if ret_sig is not None:
            self._materialize_result_ptr(ret_sig.ret_type, ret_sig.ret_by_ref)

    def _init_one_arg(self, node: SpecializedFormalArg, mir_args: list[mir.Type]) -> InterpVal:
        match node:
            case SpecializedComptimeArg():
                return ComptimeVal(sval.ConstRef(node.value))
            case SpecializedRuntimeArg():
                mir_type = node.type.to_mir_type()
                if mir_type is None or isinstance(mir_type, mir.VoidType):
                    raise _no_runtime_type(node.type)
                type = mir_type
                index = len(mir_args)
                if node.is_ref:
                    type = mir.PointerType(type, True)
                    mir_args.append(type)
                    return RuntimeVal(mir.Param(index, type), sval.PointerType(node.type, True))
                else:
                    mir_args.append(type)
                    ret = self.alloca()
                    self._commit_pending_slot(ret, node.type)
                    self.store(ret, RuntimeVal(mir.Param(index, type), node.type))
                    return ret
            case _:
                raise CompileError(f'unsupported specialized argument {node!r}')

    def push_insts(self, insts: list[mir.Inst]) -> None:
        self._mir_block_stack.append(insts)

    def pop_insts(self) -> list[mir.Inst]:
        return self._mir_block_stack.pop()

    def _init_args_from_signature(
        self,
        signature: CallSignature,
        mir_args: list[mir.Type],
    ) -> tuple[InterpVal, ...]:
        arg_values: list[InterpVal] = []

        for arg in signature.positional:
            arg_values.append(self._init_one_arg(arg[1], mir_args))

        if signature.varargs:
            arg_values.append(ComptimeTuple(tuple(ArgEntry(self._init_one_arg(a, mir_args), True) for a in signature.varargs)))
        if signature.kwargs:
            arg_values.append(ComptimeDict({k: ArgEntry(self._init_one_arg(v, mir_args), True) for k, v in signature.kwargs.items()}))

        return tuple(arg_values)

    def _materialize_result_ptr(self, type: sval.Type, ret_by_ref: bool | None = None) -> None:
        """Fix the return convention of the function proper from the spy
        type its result location holds (the peer type of all its store
        points, or its declared return annotation).  A result delivered
        through a result pointer appends the hidden result pointer formal
        to the lowered signature *after* every declared argument and makes
        the function return void; the location is then the memory of that
        pointer.  A direct return fixes the MIR return type and leaves the
        location recording its value.

        The convention is a property of the return type
        (``sval.returns_via_result_ptr``) unless the signature declares it.
        """
        if self.return_sig is not None:
            if self.return_sig.ret_type != type:
                raise CompileError(
                    f"function returns values of conflicting types "
                    f"{self.return_sig.ret_type} and {type}"
                )
            return
        if ret_by_ref is None:
            ret_by_ref = sval.returns_via_result_ptr(type)
        mir_fn = self._fn_instance.mir
        location = self._current_result_loc()
        if ret_by_ref:
            mir_type = type.to_mir_type()
            if mir_type is None or isinstance(mir_type, mir.VoidType):
                raise CompileError(f'cannot return {type} through a result pointer')
            ptr_type = mir.PointerType(mir_type, False)
            index = len(mir_fn.args)
            mir_fn.args.append(ptr_type)
            mir_fn.arg_names.append('$result')
            self._fn_instance.ret_sig = ReturnSignature(True, type)
            self.return_sig = self._fn_instance.ret_sig
            assert isinstance(location, PendingSlot)
            self._commit_pending_slot(location, type, ptr=mir.Param(index, ptr_type))
            mir_fn.ret_type = mir.VOID
        else:
            mir_ret = type.to_mir_type()
            if mir_ret is None:
                raise _no_runtime_type(type)
            mir_fn.ret_type = mir_ret
            self.return_sig = ReturnSignature(False, type)
            self._commit_pending_slot(location, type)

    # -- return statements ------------------------------------------------

    def _current_result_loc(self) -> InterpVal:
        assert len(self._frames) > 0, 'no function result location'
        return self._frames[-1].ret_loc

    def _in_function_proper(self) -> bool:
        """Whether the instructions currently being executed are those
        of the function proper (whose ``return`` emits a typed return)
        rather than of an inlined plain function (whose ``return`` just
        yields a value to the caller): the frames stack holds the
        function proper at its bottom and one frame per inlined body
        above it, so the innermost body is the function proper exactly
        when it is the only frame."""
        return len(self._frames) == 1

    # -- running the flat stream -------------------------------------------

    def _run_machine(self) -> PollResult:
        """The flat execution loop: walks the instruction list of the
        executing frame (see ``_step``) until the run of the function
        proper ended (``DONE``) or a called spy function's specialization
        was just started and must be typed by a runner of its own
        (``SUSPEND``, see ``Analyser._run``).  There is no recursion: the
        walk is
        linear over one flat list per frame; the ``If``/``Else``/``End``
        markers delimit the blocks, and every control state lives in the
        block stacks of the frames."""
        while True:
            ret = self._step()
            if ret != PollResult.AGAIN:
                return ret

    def _pop_frame(self) -> None:
        if len(self._frames) > 1:
            self._emit(mir.End())
        self._frames.pop()

    def _step(self) -> PollResult:
        """Execute one step of the machine: the instruction at the pc of
        the executing frame, advancing the pc - or the end of the
        frame's instruction list (its body fell off its end)."""
        frame = self._frames[-1]
        if frame.pc >= len(frame.insts):
            # the body fell off its end: every block is closed (see the
            # marker transitions) - the run of the frame ended
            assert not frame.block_stack
            if len(self._frames) == 1:
                return PollResult.DONE
            self._pop_frame()
            return PollResult.AGAIN
        inst = frame.insts[frame.pc]
        frame.pc += 1
        return self._exec_inst(inst)

    def _scan_block(self, entry: int) -> tuple[int | None, int]:
        return hir.scan_block(self._frames[-1].insts, entry)

    def _exec_inst(self, inst: hir.Inst) -> PollResult:
        """Execute one instruction of the executing frame.  A control
        instruction changes the execution state: an ``If`` opens a block
        (the walk continues into the chosen branch of a compile-time
        ``if``, or types the branch regions of a runtime ``if``), the
        ``Else``/``End`` markers close the branch being walked, a
        ``return`` cuts the current path (see ``_cut``); every other
        instruction only advances the state of the frame (its register
        table, and the MIR emitted so far)."""

        frame = self._frames[-1]
        regs = frame.regs
        match inst:
            case hir.Ret():
                if not self._in_function_proper():
                    level = frame.ret_levels()
                    if level > 0:
                        self._emit(mir.Break(level))
                    return self._cut()
                if self.return_sig is None:
                    # the return convention is not fixed yet (an unannotated
                    # return type): the ``mir.Ret`` is filled in by
                    # ``_finish_function`` once the result location has been
                    # materialized
                    block = mir.Insertion([], None)
                    self._emit(block)
                    self._deferred_returns.append(block)
                    return self._cut()
                location = self._current_result_loc()
                if self.return_sig.ret_by_ref or self.return_sig.ret_type.is_zst():
                    self._emit(mir.Ret(None))
                else:
                    self._emit(mir.Ret(_to_runtime(self.load(location))))
                return self._cut()
            case hir.AsBool():
                return self.as_bool(self.operand_arg(inst.value), inst)
            case hir.BinaryAssign():
                return self.binary_assign(inst.op, self.operand(inst.lhs), self.operand_arg(inst.rhs))
            case hir.If():
                self._exec_if(inst)
            case hir.Else():
                self._exec_else()
            case hir.End():
                self._exec_end()
            case hir.Load():
                regs[inst] = self.load(self.operand(inst.ptr))
            case hir.Alloca():
                regs[inst] = self.alloca(inst.allow_comptime)
            case hir.Store():
                self.store(self.operand(inst.ptr), self.operand(inst.value))
            case hir.StoreVoidRetloc():
                self.store(self._current_result_loc(), ComptimeVal(sval.Void()))
            case hir.Tuple():
                regs[inst] = ComptimeTuple(tuple(self.operand_arg(v) for v in inst.values))
            case hir.Binary():
                return self._eval_binary(inst.op, self.operand_arg(inst.lhs), self.operand_arg(inst.rhs), self.operand(inst.ret))
            case hir.Compare():
                return self._eval_cmp(inst.op, self.operand_arg(inst.lhs), self.operand_arg(inst.rhs), inst)
            case hir.BoolOp():
                return self._eval_boolop(inst.op, self.operand_arg(inst.lhs), self.operand_arg(inst.rhs), inst)
            case hir.Unary():
                return self._eval_unary(inst.op, self.operand_arg(inst.operand), self.operand(inst.ret))
            case hir.CallInplace():
                return self.call(self.operand(inst.callee), self.operand_arglist(inst.args), self.operand(inst.ret))
            case hir.CallMethodInplace():
                return self.call_method(self.operand(inst.base), inst.name, self.operand_arglist(inst.args), self.operand(inst.ret))
            case hir.FieldAddr():
                regs[inst] = self.exec_field_name_addr(self.operand(inst.base), inst.name, inst.is_aggregate_init)
            case hir.FieldIndexAddr():
                regs[inst] = self.field_index_addr(
                    self.operand(inst.base), _index_value(inst.index), inst.is_aggregate_init
                )
            case hir.FinishStruct():
                self.finish_struct(
                    self.operand(inst.struct),
                    self.operand(inst.dest),
                    tuple(self.operand(v) for v in inst.indices),
                    frozendict((k, self.operand(v)) for k, v in inst.names.items()),
                )
            case hir.FinishArray():
                self.finish_array(self.operand(inst.array), tuple(self.operand(e) for e in inst.elements))
            case hir.CommitSlot():
                self._commit_pending_slot(self.operand(inst.slot))
            case hir.Subscript():
                self.subscript(self.operand(inst.base), self.operand_arg(inst.index), inst)
            case _:
                raise CompileError(f"unsupported instruction {inst}")
        return PollResult.AGAIN

    def _exec_if(self, inst: hir.If) -> None:
        """An ``if`` at the pc of the executing frame: the walk just
        passed its ``If`` marker.  A compile-time condition keeps only
        the chosen branch - the other branch is dead, its instructions
        are skipped (never typed or emitted); a runtime condition types
        both branches (both survive at runtime, see
        ``_exec_runtime_if``).  The block is pushed on the frame's block
        stack, recording its entry (the pc of this ``If``) so that the
        matching ``Else``/``End`` markers can be found when the branches
        are walked off (see ``_scan_block``)."""
        frame = self._frames[-1]
        entry = frame.pc - 1
        cond = self.operand(inst.cond)
        if isinstance(cond, ComptimeVal):
            if cond.obj:
                # the then branch is chosen: it follows the ``If``
                frame.block_stack.append(BlockFrame(entry, IfBlockData(True)))
                return
            # the else branch is chosen: skip the (dead) then branch
            p_else, p_end = self._scan_block(entry)
            if p_else is None:
                # no else branch either: the whole ``if`` is dead
                frame.pc = p_end + 1
                return
            frame.block_stack.append(BlockFrame(entry, IfBlockData(False)))
            frame.pc = p_else + 1
            return
        self._exec_runtime_if(cond, entry)

    def _exec_else(self) -> None:
        """The walk fell off the end of the then branch and reached the
        ``Else`` marker of the innermost open block."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        data = bf.data
        assert isinstance(data, IfBlockData)
        if data.chosen is None:
            # a runtime ``if``: its then-region fell off its end (it
            # does not return); the else-region is typed next
            data.then_returns = False
            self._emit(mir.Else())
            return
        # a compile-time ``if`` whose chosen branch is the then branch,
        # which fell off its end: the (unchosen) else branch is dead -
        # skip it and close the block
        assert data.chosen
        frame.block_stack.pop()
        _, p_end = self._scan_block(bf.entry)
        frame.pc = p_end + 1

    def _exec_end(self) -> None:
        """The walk fell off the end of a branch and reached the ``End``
        marker of the innermost open block.  A runtime ``if`` whose
        currently-typed region fell off its end - the then-region of an
        ``if`` without an else, or the else-region - is complete: the
        falling branch continues with the code after the ``End``.  Both
        branches falling through (a join) is fine: the MIR's falling
        branches already continue at that shared continuation, and a
        variable a branch *assigns* lives in an enclosing block's slot -
        memory, since an assignment in a branch has to be visible after
        it - so the state crossing the join needs no phi."""
        frame = self._frames[-1]
        data = frame.block_stack[-1].data
        match data:
            case IfBlockData():
                if data.chosen is not None:
                    # a compile-time ``if``: the chosen branch fell off its end
                    frame.block_stack.pop()
                    return
                frame = self._frames[-1]
                bf = frame.block_stack[-1].data
                assert isinstance(bf, IfBlockData)
                assert bf.chosen is None
                self._emit(mir.End())
                frame.block_stack.pop()
            case _:
                raise AssertionError('unreachable')

    # -- runtime ``if`` regions --------------------------------------------

    def _exec_runtime_if(
        self, cond: InterpVal, entry: int
    ) -> None:
        if not isinstance(cond, RuntimeVal) or cond.type != sval.BoolType():
            raise CompileError('runtime if conditions must be boolean values')
        frame = self._frames[-1]
        self._emit(mir.If(cond.value))
        frame.block_stack.append(BlockFrame(entry, IfBlockData()))

    def _cut(self) -> PollResult:
        """The current path of the executing frame ended - a ``return``
        was executed, or the path turned out dead (a runtime ``if``
        whose every branch returned): unwind the open blocks of the
        frame.  A compile-time ``if`` whose (chosen) branch the path ran
        through is just popped (the code after it is dead); a runtime
        ``if`` whose currently-typed region the path ended in continues
        with its sibling region, or - when its else-region ended - is
        complete: a single falling branch resumes the code after the
        ``if``, and when both branches returned the cut keeps unwinding.
        With no open block left, the run of the frame's body ended."""
        while True:
            frame = self._frames[-1]
            if not frame.block_stack:
                # the body run ended in a return: skip the dead code
                # after it
                if len(self._frames) == 1:
                    return PollResult.DONE
                self._pop_frame()
                return PollResult.AGAIN
            bf = frame.block_stack[-1]
            data = bf.data
            match data:
                case IfBlockData():
                    if data.chosen is not None:
                        # the path ran through the chosen branch of a
                        # compile-time ``if`` and returned: dead code after it
                        frame.block_stack.pop()
                        continue
                    if data.then_returns is None:
                        # the path ended inside the then-region: type the
                        # else-region next (it is a fresh path)
                        data.then_returns = True
                        p_else, p_end = self._scan_block(bf.entry)
                        if p_else is not None:
                            self._emit(mir.Else())
                            frame.pc = p_else + 1
                            return PollResult.AGAIN
                        self._emit(mir.End())
                        frame.block_stack.pop()
                        frame.pc = p_end + 1
                        return PollResult.AGAIN
                    # the path ended inside the else-region: the ``if`` is
                    # complete; when every path returned the cut keeps unwinding
                    then_returns = data.then_returns
                    assert then_returns is not None
                    self._emit(mir.End())
                    frame.block_stack.pop()
                    if not then_returns:
                        _, p_end = self._scan_block(bf.entry)
                        frame.pc = p_end + 1
                    if then_returns:
                        continue
                case _:
                    raise AssertionError('unreachable')
            return PollResult.AGAIN

    def _type_var_value(self, obj: sval.AnyValue) -> InterpVal | None:
        """The value a type parameter stands for in the body currently
        being executed.  A name that denotes a type parameter of the
        function (or of the struct a method belongs to) is a compile-time
        value (see ``astgen``); the frame carries the type the call solved
        it to, so ``Foo[T]``, ``spy.typeof(x) == T``, ... see the concrete
        type."""
        if not isinstance(obj, sval.TypeVar):
            return None
        value = self._frames[-1].generic_var_values.get(obj)
        if value is None:
            raise CompileError(f'type parameter {obj.name} is not bound here')
        return value

    def operand(self, value: hir.Value) -> InterpVal:
        regs = self._frames[-1].regs
        match value:
            case hir.Const():
                # the value of an immutable global (or a literal): an
                # embedded Python object that may be a function
                # registered in the host context - reached as the raw
                # function object or through the callable view its
                # decorated name binds to; it is resolved to its entry
                # here, when the reference runs (see
                # ``FunctionResolver.resolve_global``)
                obj = value.value
                type_var_value = self._type_var_value(obj)
                if type_var_value is not None:
                    return type_var_value
                if not isinstance(obj, (int, float, str, bool, pytypes.NoneType)):
                    resolved = self._analyser._resolver.resolve_global(obj)
                    if resolved is not None:
                        return ComptimeVal(resolved)
                return ComptimeVal(sval.as_value(obj))
            case hir.ConstRef():
                # a reference to an immutable global.  At compile time a
                # reference to a global behaves exactly like the value it
                # refers to (its static type is a ``sval.PointerType`` of
                # the referenced object - ``sval.PointerType(
                # sval.type_of(expr), True)`` - but nothing emits a
                # runtime load of a compile-time global yet: it is
                # dereferenced only at compile time (see ``load``), so a
                # reference is otherwise consumed as an identity - the
                # callee of a call).  The
                # referenced object is resolved to its entry like a
                # ``Const`` value.
                obj = value.value
                resolved = self._analyser._resolver.resolve_global(obj)
                return ComptimeVal(sval.ConstRef(resolved if resolved is not None else obj))
            case hir.Arg(index):
                assert len(self._frames) > 0, 'Arg outside of any function frame'
                frame = self._frames[-1]
                assert index < len(frame.arg_values), 'Arg index out of range'
                return frame.arg_values[index]
            case hir.ResultLoc():
                return self._current_result_loc()
            case hir.Inst():
                reg = regs.get(value)
                assert reg is not None, 'register not evaluated'
                return reg
            case _:
                raise CompileError(f"unsupported operand {value}")

    # -- memory instructions -------------------------------------------------

    def load(self, ptr: InterpVal) -> InterpVal:
        """Read the value a slot or a pointer holds."""
        if isinstance(ptr, PendingSlot) and ptr.committed is None:
            raise CompileError('cannot load from a slot before it is committed')

        ptr = _shallow_normalize(ptr)
        type = _type_of(ptr)
        if not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot load from a {type} value")
        unit_value = type.elem.get_unit_value()
        if unit_value is not None:
            return ComptimeVal(unit_value)
        match ptr:
            case ComptimeBox():
                return ptr.value
            case ComptimeVal(obj) if isinstance(obj, sval.ConstRef):
                # a reference to an immutable compile-time global behaves like
                # the value it refers to
                return ComptimeVal(obj.value)
            case RuntimeVal():
                return RuntimeVal(self._emit(mir.Load(ptr.value)), type.elem)
        raise CompileError('cannot load from a compile-time pointer')

    def store(self, ptr: InterpVal, value: InterpVal) -> None:
        """Write ``value`` into the slot or through the pointer ``ptr``.
        A store into a still uncommitted slot only records a store point
        (see :class:`PendingSlot`): the slot's final type is not known
        until it is committed, and the actual store is inserted then."""
        if isinstance(ptr, ComptimeTuple):
            # a destructuring target: a tuple of element *addresses*, one
            # per element of the value it is stored with (a nested tuple
            # target pairs with a nested tuple value)
            todo: list[tuple[InterpVal, InterpVal]] = [(ptr, value)]
            stores: list[tuple[InterpVal, InterpVal]] = []
            while todo:
                target, source = todo.pop()
                if isinstance(target, ComptimeTuple):
                    if not isinstance(source, ComptimeTuple):
                        raise CompileError(
                            f'cannot unpack a value into {len(target.values)} targets'
                        )
                    if len(source.values) != len(target.values):
                        raise CompileError(
                            f'cannot unpack {len(source.values)} values into '
                            f'{len(target.values)} targets'
                        )
                    assert all(a.is_ref or isinstance(a.value, ComptimeTuple) for a in target.values)
                    todo.extend(reversed([(t.value, self._arg_value(s)) for t, s in zip(target.values, source.values)]))
                    continue
                stores.append((target, source))
            for target, source in stores:
                self.store(target, source)
            return

        if isinstance(ptr, PendingSlot) and ptr.committed is None:
            value_type = _type_of(value)
            if value_type is None:
                raise CompileError('cannot store a value that has no spy type')
            self._record_pending_action(
                ptr,
                _PendingStore(
                    type=value_type,
                    is_comptime=_is_comptime_val(value),
                    value=value,
                ),
            )
            return

        ptr = _shallow_normalize(ptr)
        ptr_type = _type_of(ptr)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f"cannot store to a {ptr_type} value")
        elem = ptr_type.elem
        if elem.is_zst():
            return
        coerced = self._coerce(value, elem)
        match ptr:
            case ComptimeBox():
                ptr.value = coerced
            case RuntimeVal():
                self._emit(mir.Store(ptr.value, _to_runtime(coerced)))
            case _:
                raise CompileError('cannot store through a compile-time pointer')

    def _arg_value(self, arg: ArgEntry[InterpVal]) -> InterpVal:
        """The value an argument denotes: a reference argument is loaded
        out of the address it carries."""
        if arg.is_ref:
            return self.load(arg.value)
        return arg.value

    def _auto_deref(self, ev: InterpVal) -> InterpVal:
        t = _type_of(ev)
        if isinstance(t, sval.PointerType) and isinstance(t.elem, sval.PointerType):
            return self.load(ev)
        return ev

    # -- struct and array values ---------------------------------------------

    def exec_field_name_addr(self, ptr: InterpVal, name: str, is_aggregate_init: bool = False) -> InterpVal:
        """The address of the field ``name`` of the struct ``base`` points at.
        Auto-dereferences a base that points at a pointer, unlike
        ``field_index_addr``."""
        ptr = _shallow_normalize(ptr)
        if is_aggregate_init and isinstance(ptr, PendingSlot) and ptr.committed is None:
            # the storage of the aggregate being built has no address yet: the
            # field gets a pending place of its own (see ``finish_struct``)
            return self.alloca(False)
        ptr = self._auto_deref(ptr)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot take field address of {ptr}")
        container_type = type.elem
        if not isinstance(container_type, sval.StructType):
            raise CompileError(f"cannot take field address of {ptr}")
        index = container_type.field_index(name)
        if index is None:
            raise CompileError(
                f"type {container_type} has no field named '{name}'"
            )

        return self.field_index_addr(ptr, _index_value(index))

    def field_index_addr(
        self,
        ptr: InterpVal,
        index: InterpVal,
        is_aggregate_init: bool = False,
        at: mir.Insertion | None = None,
    ) -> InterpVal:
        """The address of the ``index``-th field of the struct - or of the
        ``index``-th element of the array - the place ``ptr`` denotes.  No auto
        deref, unlike ``exec_field_name_addr``: the base is the storage itself
        (a struct field chain, a construction's storage, an array).

        ``is_aggregate_init`` marks a construction's field/element address: when
        its storage is a slot whose type is not decided yet (the aggregate is
        still being built), the place gets a pending slot of its own rather than
        an address, which the ``FinishStruct``/``FinishArray`` closing the
        construction turns into the address of its field/element (see
        ``finish_array``/``finish_struct``).  ``at`` produces the address
        instruction into an insertion block instead of the current position.
        A zero-sized field/element occupies no storage and has no address."""
        ptr = _shallow_normalize(ptr)
        if is_aggregate_init and isinstance(ptr, PendingSlot) and ptr.committed is None:
            # the aggregate's storage has no address yet: a pending place
            return self.alloca(False)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f'cannot take a field or element address of {ptr}')
        container_type = type.elem
        is_const = type.is_const

        if isinstance(container_type, sval.StructType):
            index_int = _comptime_index(index)
            fields = container_type.fields()
            if index_int < 0 or index_int >= len(fields.by_id):
                raise CompileError(f'type {container_type} has no field at index {index_int}')
            field_type = fields.get_by_id(index_int).type
            field_ptr_type = sval.PointerType(field_type, is_const)
            if field_type.is_zst():
                # a zero-sized field occupies no storage and has no address
                return ComptimeVal(sval.Undefined(field_ptr_type))
            match ptr:
                case ComptimeVal():
                    raise CompileError(
                        'cannot take the address of a field of a compile-time value'
                    )
                case RuntimeVal():
                    if not container_type.mirror_is_a_field():
                        mir_index = container_type.get_field_mir_indices()[index_int]
                        assert mir_index is not None, 'a field with storage has a mirror position'
                        return RuntimeVal(
                            self._emit(mir.Gep(ptr.value, mir_index), at), field_ptr_type
                        )
                    # the mirror of the struct is the mirror of its own field
                    # (see ``sval.StructType.mirror_is_a_field``): the field is
                    # the value itself, so it takes no address arithmetic
                    return RuntimeVal(ptr.value, field_ptr_type)
                case _:
                    raise CompileError(f'cannot take field address of {ptr}')

        if isinstance(container_type, sval.ArrayType):
            elem_ptr_type = sval.PointerType(container_type.elem, is_const=is_const)
            if container_type.is_zst():
                # a zero-sized array holds no storage and so has no addresses:
                # every value of one equals the unit value of its element type
                return ComptimeVal(sval.Undefined(elem_ptr_type))
            if not isinstance(ptr, RuntimeVal):
                raise CompileError(
                    f'cannot take the address of an element of {container_type}: '
                    f'the array is a compile-time value'
                )
            return RuntimeVal(
                self._emit(mir.Gep(ptr.value, _mir_index(index)), at), elem_ptr_type
            )

        raise CompileError(f'cannot take a field or element address of {ptr}')

    def _emit(self, inst: mir.Inst, at: mir.Insertion | None = None) -> mir.Value:
        """Append one instruction to the list currently being filled:
        the flat body of the function being typed, or a pending action's
        insertion block while one is delivered.  ``at`` appends to an
        insertion block instead - the instruction then lands at the
        position that block sits at (a slot's storage, see
        ``PendingSlot.insertion``).  A specialization's MIR lands in one
        list, delimited by the ``If``/``Else``/``End`` (and ``Block``)
        markers (there are no separate regions)."""
        if at is not None:
            at.insts.append(inst)
        else:
            assert self._mir_block_stack
            self._mir_block_stack[-1].append(inst)
        return inst

    def alloca(self, allow_comptime: bool = False) -> PendingSlot:
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        return PendingSlot(insertion, allow_comptime)

    # -- helpers -------------------------------------------------------------

    def _coerce(self, ev: InterpVal, target: sval.Type) -> InterpVal:
        """Materialize a value of the spy type ``target``: a compile-time
        value is converted with ``sval.coerce_const``, a runtime value
        gets whatever numeric conversion the target needs - widening or
        narrowing, see ``_convert_inst``.  A committed slot is the address
        of the value it holds (``_shallow_normalize``), which is what an
        operation that takes a value without loading it (taking an address,
        ``ref``) hands over."""
        ev = _shallow_normalize(ev)
        match ev:
            case ComptimeVal(obj):
                return ComptimeVal(sval.coerce_const(obj, target))
            case RuntimeVal(value, type):
                return RuntimeVal(self._convert(value, type, target), target)
            case _:
                raise CompileError('cannot materialize this value')

    def _convert(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Value:
        converted = _convert_inst(value, from_type, to_type)
        if converted is None:
            return value
        return self._emit(converted)

    # -- operators ------------------------------------------------------------

    def _eval_binary(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        # what the operation *is* follows from the types of the operands - a
        # primitive arithmetic instruction, or (later) an overload method
        # that takes the operands by reference (``a + b`` becomes
        # ``a.__add__(b)``, see ``call_method``), so the operands are kept as
        # the references they are and a value is only loaded where one is
        # needed
        if _is_comptime_val(lhs.value) and _is_comptime_val(rhs.value):
            # every operand is compile-time: the operation is evaluated
            # eagerly in Python, whatever the runtime types are
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            assert isinstance(lv, ComptimeVal) and isinstance(rv, ComptimeVal)
            self.store(ret, ComptimeVal(_comptime_py_op(op, lv.obj, rv.obj)))
            return PollResult.AGAIN

        lhs_type = _arg_type_of(lhs)
        rhs_type = _arg_type_of(rhs)

        if lhs_type is None or rhs_type is None:
            raise CompileError(f"cannot apply '{op}' to untyped objects")
        if sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type):
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            type = lhs_type.resolve_peer_type(rhs_type)
            if type is None:
                raise CompileError(f"cannot apply '{op}' to {lhs_type} and {rhs_type}")
            if isinstance(type, sval.IntType):
                if op == '/':
                    raise CompileError(
                        "integer division ('/') is not supported; divide float values instead"
                    )
                if op == '//':
                    raise CompileError("integer floor division ('//') is not supported yet")
                if op == '**':
                    raise CompileError("integer exponentiation ('**') is not supported yet")
                if op not in ('+', '-', '*', '%'):
                    raise CompileError(f"unsupported operator '{op}' for integers")
            else:
                if op == '**':
                    raise CompileError("float exponentiation ('**') is not supported yet")
                if op == '//':
                    raise CompileError("float floor division ('//') is not supported yet")
                if op not in ('+', '-', '*', '/'):
                    raise CompileError(f"unsupported operator '{op}' for floats")
            lc = self._coerce(lv, type)
            rc = self._coerce(rv, type)
            signed = isinstance(type, sval.IntType) and type.signed
            mir_type = type.to_mir_type()
            assert mir_type is not None and not isinstance(mir_type, mir.VoidType)
            value = self._emit(
                mir.Arith(op, signed, _to_runtime(lc), _to_runtime(rc), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        elif isinstance(lhs_type, sval.StructType) or isinstance(rhs_type, sval.StructType):
            # call `__xxx__` methods
            raise NotImplementedError
        else:
            raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")

    def _eval_cmp(self, op: CompareOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        lhs_type = _arg_type_of(lhs)
        rhs_type = _arg_type_of(rhs)

        if _is_comptime_val(lhs.value) and _is_comptime_val(rhs.value):
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            assert isinstance(lv, ComptimeVal) and isinstance(rv, ComptimeVal)
            self._frames[-1].regs[ret_reg] = ComptimeVal(_comptime_py_op(op, lv.obj, rv.obj))
            return PollResult.AGAIN

        if lhs_type is None or rhs_type is None:
            raise CompileError(f"cannot apply '{op}' to untyped objects")
        if sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type):
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            type = lhs_type.resolve_peer_type(rhs_type)
            if type is None:
                raise CompileError(f'cannot compare {lhs_type} and {rhs_type}')
            lc = self._coerce(lv, type)
            rc = self._coerce(rv, type)
            kind = 'int' if isinstance(type, sval.IntType) else 'float'
            signed = isinstance(type, sval.IntType) and type.signed
            value = self._emit(
                mir.Cmp(op, signed, kind, _to_runtime(lc), _to_runtime(rc))
            )
            self._frames[-1].regs[ret_reg] = RuntimeVal(value, sval.BoolType())
            return PollResult.AGAIN
        elif isinstance(lhs_type, sval.StructType) and isinstance(rhs_type, sval.StructType):
            # call `__xxx__` methods
            raise NotImplementedError
        else:
            raise CompileError(f'unsupported operand types: {lhs_type} and {rhs_type}')

    def _eval_boolop(self, op: BoolOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        if _is_comptime_val(lhs.value) and _is_comptime_val(rhs.value):
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            assert isinstance(lv, ComptimeVal) and isinstance(rv, ComptimeVal)
            result = (lv.obj and rv.obj) if op == 'and' else (lv.obj or rv.obj)
            self._frames[-1].regs[ret_reg] = ComptimeVal(bool(result))
            return PollResult.AGAIN
        raise CompileError(
            f"'{op}' between runtime values is not supported yet "
            '(only compile-time operands)'
        )

    def _eval_unary(self, op: UnaryOp, operand: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        if _is_comptime_val(operand.value):
            ev = self._arg_value(operand)
            assert isinstance(ev, ComptimeVal)
            obj = ev.obj
            if op == 'not':
                result: sval.AnyValue = not obj
            elif op == '-':
                negated = sval.negate(obj)
                if negated is None:
                    raise CompileError(f'cannot negate {obj!r} at compile time')
                result = negated
            else:
                raise CompileError(f"unsupported unary operator '{op}'")
            self.store(ret, ComptimeVal(result))
            return PollResult.AGAIN

        type = _arg_type_of(operand)
        if type is None:
            raise CompileError(f"cannot apply unary '{op}' to a value that has no type yet")
        if op == 'not':
            if not isinstance(type, sval.BoolType):
                raise CompileError(f"cannot apply 'not' to a {type} value")
            coerced = self._coerce(self._arg_value(operand), type)
            value = self._emit(
                mir.Cmp('==', False, 'int', _to_runtime(coerced), mir.BoolValue(False))
            )
            self.store(ret, RuntimeVal(value, sval.BoolType()))
            return PollResult.AGAIN
        if op == '-':
            mir_type = type.to_mir_type()
            assert mir_type is not None and not isinstance(mir_type, mir.VoidType)
            if isinstance(type, sval.FloatType):
                assert isinstance(mir_type, mir.FloatType)
                zero: mir.Value = mir.Float(0.0, mir_type)
            elif isinstance(type, sval.IntType):
                assert isinstance(mir_type, mir.IntType)
                zero = mir.Int(0, mir_type)
            else:
                raise CompileError(f'cannot negate a {type} value')
            coerced = self._coerce(self._arg_value(operand), type)
            value = self._emit(
                mir.Arith('-', False, zero, _to_runtime(coerced), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        raise CompileError(f"unsupported unary operator '{op}'")

    def binary_assign(self, op: BinaryOp, left: InterpVal, right: ArgEntry[InterpVal]) -> PollResult:
        # ``x op= y`` is ``x = x op y``: the value the target currently
        # holds and the right operand feed the operator, and its result is
        # stored back into the target
        return self._eval_binary(op, ArgEntry(left, True), right, left)

    # -- calls ----------------------------------------------------------------

    def call(self, callee: InterpVal, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """Resolve one call by its callee value and run it.  Spy
        functions compile to a native ``call`` producing a typed
        register, plain Python functions are inlined, and the spy
        builtins are evaluated at compile time.  The callee constant of a
        registered spy function already resolved to its entry when the
        callee operand was evaluated (see ``operand``).

        Returns ``PollResult.AGAIN`` when the call completed here (an
        inlined callee's body writes into the result location directly),
        or ``PollResult.SUSPEND`` when the callee's specialization was
        just started and must be typed first: the call is then completed
        by ``resume`` when that runner ends (see
        ``_call_function_entry``)."""
        target = _callee_object(callee)
        if target is not None:
            if isinstance(target, FunctionValue):
                return self._call_function_entry(target, args, ret)
            if isinstance(target, sval.BoundMethod):
                # a method of a generic struct resolved from a value: the
                # struct's type-argument values are substituted into the
                # method's signature (see ``_call_function_entry``)
                fn = target.fn
                assert isinstance(fn, FunctionValue), 'a bound method holds a function value'
                return self._call_function_entry(fn, args, ret, target.generic_var_values)
            if isinstance(target, sval.BuiltinFn):
                return self._call_builtin(target, args, ret)
        raise CompileError(
            f"cannot compile a call to {callee!r}; only spy functions, plain Python "
            "functions and the spy builtins can be called"
        )

    def _call_builtin(
        self, fn: sval.BuiltinFn, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """Evaluate one ``spy.*`` builtin at compile time and hand its
        result to the call's result location."""
        if fn.name == 'typeof':
            if len(args.positional) != 1 or len(args.kwargs) > 0:
                raise CompileError('spy.typeof takes exactly one argument')
            type = _arg_type_of(args.positional[0])
            if type is None:
                raise CompileError('cannot determine the type of this value')
            self.store(ret, ComptimeVal(type))
            return PollResult.AGAIN
        if fn.name == 'compile_log':
            parts: list[str] = []
            for arg in args.positional:
                ev = self._arg_value(arg)
                obj = _to_comptime(ev)
                if obj is None:
                    type = _type_of(ev)
                    parts.append(f'<{type}>' if type is not None else '<value>')
                else:
                    parts.append(str(obj))
            for arg in args.kwargs.values():
                ev = self._arg_value(arg)
                obj = _to_comptime(ev)
                parts.append(str(obj) if obj is not None else '<value>')
            print(' '.join(parts))
            self.store(ret, ComptimeVal(sval.Void()))
            return PollResult.AGAIN
        raise CompileError(f"cannot call the spy builtin {fn.name} inside a spy function")

    def _record_pending_action(self, slot: PendingSlot, data: _PendingActionData) -> None:
        """Record one action on a still uncommitted slot, reserving the
        insertion block that will hold the instructions delivering it."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        slot.stores.append(_PendingAction(insertion, data))

    def _commit_pending_slot(
        self, val: InterpVal, type: sval.Type | None = None, ptr: mir.Value | None = None
    ) -> None:
        """Materialize a pending slot.  It becomes a :class:`ComptimeBox`
        when it may inline values and every action is compile-time, or
        when its type is zero-sized (the box then carries the unit value
        and the recorded actions are dropped); with an explicit result
        pointer (``ptr``), or otherwise, it becomes a
        :class:`RuntimeVal` pointer to freshly allocated memory.  The
        type is the pairwise ``resolve_peer_type`` of the action types
        (or the given one).  The recorded actions are delivered through
        ``_exec_pending_actions``, which fills in the instructions that
        must be spliced at their original positions."""
        if not isinstance(val, PendingSlot):
            raise CompileError('can only commit a pending slot')
        if val.committed is not None:
            return
        if type is None:
            type = val.committed_type()

        if ptr is not None:
            self._bind_slot(val, ptr, type)
            return

        if len(val.stores) > 0 and val.is_comptime():
            box = ComptimeBox(type, ComptimeVal(sval.Undefined(type)))
            val.committed = box
            self._exec_pending_actions(val, type)
            return
        unit = type.get_unit_value()
        if unit is not None:
            val.committed = ComptimeBox(type, ComptimeVal(unit))
            return
        mir_type = type.to_mir_type()
        if mir_type is None or isinstance(mir_type, mir.VoidType):
            raise _no_runtime_type(type)
        alloca = mir.Alloca(mir_type)
        val.insertion.insts.append(alloca)
        self._bind_slot(val, alloca, type)

    def _bind_slot(self, slot: PendingSlot, ptr: mir.Value, type: sval.Type) -> None:
        slot.committed = RuntimeVal(ptr, sval.PointerType(type, is_const=False))
        self._exec_pending_actions(slot, type)

    def _exec_pending_actions(self, slot: PendingSlot, type: sval.Type) -> None:
        """Deliver every action a committed slot recorded: an action emits
        the instructions that write its value into the slot, and they land
        in the insertion block that sits at the position the action was
        recorded at."""
        assert slot.committed is not None
        for action in slot.stores:
            self.push_insts(action.insertion.insts)
            self._exec_pending_action(action.data, slot.committed, type)
            self.pop_insts()

    def _exec_pending_action(self, action: _PendingActionData, ptr: InterpVal, type: sval.Type) -> None:
        match action:
            case _PendingStore():
                self.store(ptr, action.value)
            case _PendingPtrConvertion():
                input_ptr = _shallow_normalize(action.input)
                if not (isinstance(input_ptr, RuntimeVal) and isinstance(input_ptr.type, sval.PointerType)):
                    raise CompileError('cannot convert a compile-time pointer')
                action.output.value = _to_runtime(self._convert_result_ptr(input_ptr, action.type))
            case _:
                raise CompileError(f'unsupported pending action {action}')

    def _defer_ptr_convertion(self, slot: PendingSlot, type: sval.Type) -> InterpVal:
        """The address a delivery into ``slot`` writes through, when the slot
        has no address of its own yet: a placeholder insertion that the slot's
        commit fills in with the converted pointer (``_PendingPtrConvertion``),
        so that what the delivery writes through only exists once the slot's
        final type is known.

        A zero-sized ``type`` has no address to write through at all - what is
        delivered is the type's *unit value* - so there is nothing to defer:
        the value is recorded as an ordinary store into the slot, which is what
        gives the slot its type in the first place.  The store takes part in the
        peer resolution like any other, so a destination that also receives a
        wider type (a future ``Option[T]``) widens with it, and the unit value
        is coerced to the final type at the slot's commit."""
        unit = type.get_unit_value()
        if unit is not None:
            self.store(slot, ComptimeVal(unit))
            return ComptimeVal(sval.Undefined(sval.PointerType(type, is_const=False)))
        mir_type = type.to_mir_type()
        if mir_type is None or isinstance(mir_type, mir.VoidType):
            raise _no_runtime_type(type)
        output = mir.Insertion([], None, mir.PointerType(mir_type))
        self._emit(output)
        slot.stores.append(_PendingAction(output, _PendingPtrConvertion(type, slot, output)))
        return RuntimeVal(output, sval.PointerType(type, is_const=False))

    def _convert_result_ptr(self, ptr: InterpVal, to_type: sval.Type) -> InterpVal:
        """The pointer a result-location operation writes through, converted
        to the type it delivers - the result type of a call, or the struct/
        array type a construction builds its fields/elements into.  Not
        implemented yet (the stub performs no conversion): it is only needed
        when the location's final type and the delivered type differ."""
        return ptr

    def as_bool(self, value: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """Use the value as the condition of an ``if`` - a statement's or an
        if-expression's: a ``spy.bool`` value passes through, as the boolean
        register the interpreter branches on.  Spy has no truthiness, so
        nothing else is a condition."""
        type = _arg_type_of(value)
        if isinstance(type, sval.BoolType):
            self._frames[-1].regs[ret] = self._arg_value(value)
            return PollResult.AGAIN

        raise CompileError(f'an if condition must be a bool value, got {type}')

    def subscript(self, base: InterpVal, index: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """``Foo[i32, f64]``: the specialization of the struct template
        ``base`` for the generic arguments ``index`` (one type value, or a
        tuple of them).  The result is the struct *type* itself, a
        compile-time value - the same one the annotation spelling evaluates
        to at the Python level (see ``dsl._RegisteredClass.__getitem__``).

        ``a[i]``: the *place* the i-th element of the array ``base`` points at
        is - a subscript of an array is read and written through like a field
        of a struct (see ``field_index_addr``).  The index is a ``u64`` for now;
        ``usize``, the width of a pointer of the target, will take its place."""
        array_type = _array_elem_type_of(base)
        if array_type is not None:
            index_value = self._coerce(self._arg_value(index), sval.IntType(64, False))
            element_index = _to_comptime(index_value)
            if isinstance(element_index, sval.Int):
                # a compile-time index is checked here: the MIR address of an
                # element is taken with no bounds information at runtime
                length = array_type.length_int
                if length is not None and not 0 <= element_index.value < length:
                    raise CompileError(
                        f'index {element_index.value} is out of bounds for {array_type}'
                    )
            self._frames[-1].regs[ret] = self.field_index_addr(base, index_value)
            return PollResult.AGAIN

        if not (isinstance(base, ComptimeVal) and isinstance(base.obj, sval.ConstRef) and isinstance(base.obj.value, sval.StructTypeHead)):
            raise CompileError(
                f'cannot subscript {base!r}: a subscript is a place in an array, '
                f'or a specialization of a struct template'
            )
        struct = base.obj.value
        args: tuple[InterpVal, ...]
        if isinstance(index.value, ComptimeTuple):
            assert not index.is_ref
            args = tuple(self._arg_value(a) for a in index.value.values)
        else:
            args = (self._arg_value(index),)
        arg_values: list[sval.Value] = []
        for arg in args:
            if not isinstance(arg, ComptimeVal) or not isinstance(arg.obj, sval.Value):
                raise CompileError(f'expected a compile-time type argument, got {arg!r}')
            arg_values.append(arg.obj)

        instance = struct.specialize(tuple(arg_values))
        self._frames[-1].regs[ret] = ComptimeVal(instance)
        return PollResult.AGAIN

    def finish_struct(
        self,
        struct: InterpVal,
        dest: InterpVal,
        indices: tuple[InterpVal, ...],
        names: frozendict[str, InterpVal],
    ) -> None:
        """Close a struct construction (``hir.FinishStruct``): decide the
        struct type, give every field the address it writes through and fill
        the fields that no argument provides.

        The struct type is the one ``struct`` names, or - when it is a generic
        template written without its arguments - the one the storage declares
        or the field values determine (see ``_struct_construction_type``).  A
        positional argument binds the field of the same declaration index, a
        keyword one the field of that name, and every field that no argument
        provides is filled with its default: the unit value of a zero-sized
        field, which occupies no storage.  A field with a runtime
        representation that no argument provides is an error.  Just like
        ``finish_array``, the storage takes the struct type through a deferral,
        so its type is *recorded* on the slot rather than fixed on it, and
        every field place that is still pending becomes the address of its
        field in the storage (its value is written through that address, in
        place)."""
        struct_def = _callee_object(struct)
        if isinstance(struct_def, sval.StructTypeHead):
            field_indices = struct_def.fields.by_key
        elif isinstance(struct_def, sval.StructType):
            field_indices = struct_def.fields().by_key
        else:
            raise CompileError(f'cannot finish the construction of {struct!r}')

        # every provided field, by declaration index: the positional ones in
        # the order they were given, the keyword ones by field name
        provided: dict[int, InterpVal] = {}
        for index, place in enumerate(indices):
            provided[index] = place
        for name, place in names.items():
            index = field_indices.get(name)
            if index is None:
                raise CompileError(f'{struct_def} has no field named {name!r}')
            if index in provided:
                raise CompileError(f'got multiple values for field {name!r}')
            provided[index] = place

        struct_type = _struct_construction_type(struct_def, dest, provided)
        fields = struct_type.fields()
        if len(indices) > len(fields.by_id):
            raise CompileError(
                f'{struct_type} takes {len(fields.by_id)} positional '
                f'argument(s) but {len(indices)} were given'
            )

        if isinstance(dest, PendingSlot) and dest.committed is None:
            # the slot has no address yet: the deferred conversion also
            # records the struct type as the type of the slot
            dest_ptr = self._defer_ptr_convertion(dest, struct_type)
        else:
            dest_ptr = self._convert_result_ptr(_shallow_normalize(dest), struct_type)

        for index, place in provided.items():
            field_type = fields.get_by_id(index).type
            if not (isinstance(place, PendingSlot) and place.committed is None):
                # the field wrote through the address it was given (see
                # ``field_index_addr``), which is already final
                continue
            if field_type.is_zst():
                # a zero-sized field occupies no storage: the place is
                # committed for its value alone, which every value of the
                # field type equals anyway
                self._commit_pending_slot(place, field_type)
                continue
            addr = self.field_index_addr(dest_ptr, _index_value(index), at=place.insertion)
            assert isinstance(addr, RuntimeVal), 'a field with storage has an address'
            self._bind_slot(place, addr.value, field_type)

        for index, field0 in enumerate(fields.values()):
            if index in provided:
                continue
            if field0.type.get_unit_value() is None:
                raise CompileError(f'missing a value for field {field0.name!r}')

    # -- array values ----------------------------------------------------------

    def finish_array(self, array: InterpVal, elements: tuple[InterpVal, ...]) -> None:
        """Close an array construction (``hir.FinishArray``): decide the type
        of the array and give every element the address it writes through.

        The length of the array is the number of elements, and its element type
        the one the storage already has - what is built in a place has to agree
        with the type of the place - or else the common type of the elements
        (see ``_array_construction_type``).  The storage then takes the array
        type through the same deferral a struct construction uses, so its type
        is *recorded* on the slot rather than fixed on it, and every element
        place that is still pending becomes the address of its element in the
        storage (its value is written through that address, in place)."""
        array_type = _array_construction_type(array, elements)
        if isinstance(array, PendingSlot) and array.committed is None:
            array_ptr = self._defer_ptr_convertion(array, array_type)
        else:
            array_ptr = self._convert_result_ptr(_shallow_normalize(array), array_type)
        for index, element in enumerate(elements):
            if not (isinstance(element, PendingSlot) and element.committed is None):
                # the element wrote through the address it was given (see
                # ``field_index_addr``), which is already final
                continue
            if array_type.is_zst():
                # a zero-sized array has nowhere to write an element: the place
                # is committed for its value alone, which every value of the
                # element type equals anyway
                self._commit_pending_slot(element, array_type.elem)
                continue
            ptr = self.field_index_addr(
                array_ptr, _index_value(index), at=element.insertion
            )
            assert isinstance(ptr, RuntimeVal), 'an element of an array with storage has an address'
            self._bind_slot(element, ptr.value, array_type.elem)

    def _method_of(self, struct: sval.StructType, method_name: str) -> sval.AnyValue | None:
        """The value of the method ``method_name`` of the struct type
        ``struct``: its function value - bound with the struct's type-
        argument values when the struct is generic (see
        :class:`sval.BoundMethod`).  This is the ``typeof(a).m`` a method
        call ``a.m(...)`` resolves to.  A future class-name access
        (``Foo[i32].m(x)``) resolves the same way, through the
        specialization the class name denotes."""
        method = struct.get_method(method_name)
        if method is None:
            return None
        resolved = self._analyser._resolver.resolve_global(method)
        if resolved is None:
            return None
        generic_var_values = _struct_generic_var_values(struct)
        if len(generic_var_values) == 0:
            return resolved
        return sval.BoundMethod(resolved, generic_var_values)

    def _resolve_method(self, type: sval.Type, method_name: str) -> sval.AnyValue | None:
        match type:
            case sval.StructType():
                return self._method_of(type, method_name)
            case _:
                return None

    def call_method(self, ptr: InterpVal, method_name: str, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        base = _shallow_normalize(ptr)
        # ``Foo[i32].m(x)`` - a method accessed through the class name - is
        # not supported yet.  Such a base is a compile-time struct *type*
        # rather than a pointer to a value: the future path resolves the
        # method through ``_method_of`` and calls it with no implicit
        # ``self`` (the call passes every argument, ``self`` included).
        if isinstance(base, ComptimeBox) and isinstance(base.value, ComptimeVal) and isinstance(base.value.obj, sval.StructType):
            raise CompileError(
                'calling a method through the class name is not supported yet; '
                'call it on a value of the struct instead'
            )
        ptr = self._auto_deref(base)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f'cannot call a method on a {type} value')

        method = self._resolve_method(type.elem, method_name)
        if method is None:
            raise CompileError(f'type {type.elem} has no method named {method_name}')

        # a method's first parameter is the struct itself: it is passed by
        # reference (the base's address) unless the method declares
        # ``self`` as a pointer type, in which case the base's address is
        # already the pointer value the parameter expects
        fn = method.fn if isinstance(method, sval.BoundMethod) else method
        self_is_ref = True
        if isinstance(fn, FunctionValue):
            first = fn.hir.signature.positional.by_id[0]
            self_is_ref = first.by_ref or not isinstance(first.type, sval.PointerType)

        return self.call(
            ComptimeVal(sval.ConstRef(method)),
            RawArgList((ArgEntry(ptr, self_is_ref),) + args.positional, args.kwargs),
            ret,
        )

    def operand_arglist(self, args: RawArgList[ArgEntry[hir.Value]]) -> RawArgList[ArgEntry[InterpVal]]:
        return RawArgList(
            tuple(self.operand_arg(a) for a in args.positional),
            frozendict((k, self.operand_arg(v)) for k, v in args.kwargs.items()),
        )

    def operand_arg(self, arg: ArgEntry[hir.Value]) -> ArgEntry[InterpVal]:
        return ArgEntry(self.operand(arg.value), arg.is_ref)

    def _call_function_entry(
        self,
        fn: FunctionValue,
        args: RawArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        generic_var_values: frozendict[sval.TypeVar, sval.Value] | None = None,
    ) -> PollResult:
        """A call of a registered spy function with the given (already
        evaluated) argument values - the common tail of an ordinary
        function call and of a method call, whose ``self`` the caller
        prepended to the arguments.  The call is specialized from the
        marshaled argument types (an annotated parameter fixes its type,
        an unannotated one is typed by its argument); a plain Python
        callee (``force_inline``) is inlined into the current stream
        instead of being compiled into a native specialization.

        ``generic_var_values`` are the type-argument values of the struct
        the callee is a method of (see :class:`sval.BoundMethod`): the
        method's signature names the struct's type parameters, and they are
        substituted into it before it is specialized."""
        sig = fn.hir.signature
        if generic_var_values:
            sig = sig.substitute_type_vars(dict(generic_var_values))
        binded_args = sig.bind_arg_pos(args, lambda e: ArgEntry(ComptimeVal(e), False))
        if fn.force_inline:
            # an undecorated plain Python function: its body is inlined into
            # the current stream (it has no native specialization of its own)
            return self._start_inline(fn.hir.body, binded_args, ret, generic_var_values)
        arg_types = binded_args.map(_arg_type_of)
        spec_sig = sig.specialize(arg_types)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature):
            self0._make_runtime_call(fn_mir, binded_args, ret, spec_sig[0], ret_sig)

        self._fn_req_resumer = _resumer
        res = self._analyser._request_function(fn, spec_sig[0], spec_sig[1], generic_var_values)
        if res is not None:
            fn_mir, ret_sig = res
            self.resume(fn_mir, ret_sig)
            return PollResult.AGAIN
        return PollResult.SUSPEND

    def resume(self, fn_mir: mir.Value, ret_sig: ReturnSignature):
        resumer = self._fn_req_resumer
        assert resumer is not None
        self._fn_req_resumer = None
        resumer(self, fn_mir, ret_sig)

    def _make_runtime_call(self, callee: mir.Value, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal | hir.Inst, call_sig: CallSignature, ret_sig: ReturnSignature) -> None:
        """Emit the native call of an already-resolved callee and hand its
        result to the call's result location (or register)."""
        mir_args: list[mir.Value] = []

        def convert_one(arg: ArgEntry[InterpVal], sig_arg: SpecializedFormalArg) -> None:
            if not isinstance(sig_arg, SpecializedRuntimeArg):
                # a compile-time (zero-sized) argument carries no runtime
                # value and is never passed
                return
            if sig_arg.is_ref:
                if arg.is_ref:
                    ev = _shallow_normalize(arg.value)
                    if not (isinstance(ev, RuntimeVal) and isinstance(ev.type, sval.PointerType)):
                        raise CompileError('cannot pass a compile-time reference by pointer')
                    mir_args.append(ev.value)
                else:
                    slot = self.alloca(False)
                    self._commit_pending_slot(slot, sig_arg.type)
                    self.store(slot, arg.value)
                    mir_args.append(_to_runtime(slot))
            else:
                ev = self.load(arg.value) if arg.is_ref else arg.value
                mir_args.append(_to_runtime(self._coerce(ev, sig_arg.type)))

        for arg, (_, sig_arg) in zip(args.positional, call_sig.positional):
            convert_one(arg, sig_arg)

        if call_sig.varargs is not None:
            for arg, sig_arg in zip(args.varargs, call_sig.varargs):
                convert_one(arg, sig_arg)

        if call_sig.kwargs is not None:
            for name, arg in args.kwargs.items():
                convert_one(arg, call_sig.kwargs[name])

        if ret_sig.ret_by_ref:
            assert isinstance(ret, InterpVal)
            if ret is None:
                raise CompileError('a result-pointer call needs a result location')
            if isinstance(ret, PendingSlot) and ret.committed is None:
                # the result pointer of the slot is not known yet: the call
                # writes through a placeholder the slot's commit fills in
                ret = self._defer_ptr_convertion(ret, ret_sig.ret_type)
            ptr = _to_runtime(_shallow_normalize(ret))
            self._emit(mir.Call(callee, (*mir_args, ptr), mir.VOID))
        else:
            ret_type = ret_sig.ret_type.to_mir_type()
            if ret_type is None or isinstance(ret_type, mir.VoidType):
                # a zero-sized result produces no register (the call returns
                # nothing), but it is still delivered to the result location:
                # its *unit value*.  The location of a call whose result is
                # dropped (an expression statement) would otherwise stay
                # untyped, and the type is what makes its slot a compile-time
                # box of the unit value (see ``_commit_pending_slot``)
                self._emit(mir.Call(callee, tuple(mir_args), mir.VOID))
                unit = ret_sig.ret_type.get_unit_value()
                if unit is not None:
                    value = ComptimeVal(unit)
                    if isinstance(ret, InterpVal):
                        self.store(ret, value)
                    else:
                        self._frames[-1].regs[ret] = value
            else:
                call_inst = self._emit(mir.Call(callee, tuple(mir_args), ret_type))
                value = RuntimeVal(call_inst, ret_sig.ret_type)
                if isinstance(ret, InterpVal):
                    self.store(ret, value)
                else:
                    self._frames[-1].regs[ret] = value

    def _start_inline(
        self,
        body: tuple[hir.Inst, ...],
        args: ArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        generic_var_values: frozendict[sval.TypeVar, sval.Value] | None = None,
    ) -> PollResult:
        """Start the inlined body of a plain Python callee: convert its
        bound arguments into addressable values (the callee's ``hir.Arg``
        leaves denote its parameter slots) and push its frame under a
        fresh ``mir.Block`` (a ``return`` inside a runtime branch leaves
        it with a ``mir.Break``; a ``return`` on the body's top level
        just falls off its region, closed by the matching ``mir.End``).
        An argument that is already a reference is forwarded as the
        address it is.  The body now runs under the machine; its return
        statements write into ``ret`` directly (it is the body's result
        location), so no result is handed back here.

        ``generic_var_values`` are the type-argument values of the struct
        the callee is a method of (see :class:`sval.BoundMethod`): the body
        names the struct's type parameters, and the frame resolves them
        (see ``operand``)."""
        if len(self._frames) - 1 >= _MAX_INLINE_DEPTH:
            raise CompileError(
                f'inline recursion or nesting exceeded '
                f'{_MAX_INLINE_DEPTH} levels'
            )
        if len(args.varargs) > 0 or len(args.kwargs) > 0:
            raise CompileError('*args/**kwargs cannot be inlined yet')
        arg_values: list[InterpVal] = []
        for arg in args.positional:
            if arg.is_ref:
                arg_values.append(arg.value)
            else:
                slot = self.alloca(True)
                self.store(slot, arg.value)
                self._commit_pending_slot(slot)
                arg_values.append(slot)
        self._emit(mir.Block())
        frame_values: dict[sval.TypeVar, InterpVal] = {}
        if generic_var_values is not None:
            frame_values = {tv: ComptimeVal(v) for tv, v in generic_var_values.items()}
        frame = InlineFrame(frame_values, tuple(arg_values), ret, body)
        self._frames.append(frame)
        return PollResult.AGAIN

    # -- finishing -------------------------------------------------------------

    def finish(self) -> None:
        """Called when the body of the function proper has been fully
        typed: fix an inferred return convention and splice the deferred
        insertion blocks into the body."""
        self._finish_function()
        mir_fn = self._fn_instance.mir
        mir_fn.insts = mir.normalize(mir_fn.insts)

    def _finish_function(self) -> None:
        """Fix the return convention of a function without a declared
        return type from its result location's store points, and fill in
        the ``mir.Ret`` of every deferred return site."""
        if self.return_sig is None:
            location = self._current_result_loc()
            assert isinstance(location, PendingSlot)
            if len(location.stores) == 0:
                self._fn_instance.mir.ret_type = mir.VOID
                self.return_sig = ReturnSignature(False, sval.VoidType())
            else:
                self._materialize_result_ptr(location.committed_type(), None)
        ret_sig = self.return_sig
        assert ret_sig is not None
        location = self._current_result_loc()
        for block in self._deferred_returns:
            if ret_sig.ret_by_ref or ret_sig.ret_type.is_zst():
                block.insts.append(mir.Ret(None))
            else:
                assert isinstance(location, PendingSlot)
                committed = location.committed
                if isinstance(committed, RuntimeVal):
                    load = mir.Load(committed.value)
                    block.insts.append(load)
                    block.insts.append(mir.Ret(load))
                elif isinstance(committed, ComptimeBox):
                    assert committed.value is not None
                    block.insts.append(mir.Ret(_to_runtime(committed.value)))
                else:
                    raise CompileError('cannot deliver the return value')

class Analyser:
    def __init__(self, resolver: GlobalResolver) -> None:
        self._resolver = resolver
        self._analyse_stack: list[HirRunner] = []
        self._symbol_table = CompileBatch(
            extern_anon_symbols={},
            newly_compiled=set(),
        )

    def _resolve_extern_anon_symbol(self, name: str, fn: NativeFn, type: mir.Type) -> mir.ExternAnonSymbol:
        st = self._symbol_table
        if fn in st.extern_anon_symbols:
            return st.extern_anon_symbols[fn]
        ret = mir.ExternAnonSymbol(name, type)
        st.extern_anon_symbols[fn] = ret
        return ret

    def _request_function(
        self,
        fn_entry: FunctionValue,
        call_sig: CallSignature,
        ret_sig: ReturnSignature | None,
        generic_var_values: frozendict[sval.TypeVar, sval.Value] | None = None,
    ) -> tuple[mir.Value, ReturnSignature] | None:
        """Make sure the specialization ``call_sig`` of ``fn_entry`` is
        compiled (into the module being built) and return its callee
        value and return signature - or ``None`` when the specialization
        was just started, in which case its runner has been pushed and the
        caller must suspend until it completes.

        A specialization that is already compiled is resolved to an
        external symbol (its definition lives in an earlier module); one
        that is still being compiled - a recursive reference - resolves
        to the in-module function being typed.

        ``generic_var_values`` are the type-argument values of the struct
        the callee is a method of (see :class:`sval.BoundMethod`); they are
        merged with the values ``call_sig`` solved the function's own type
        parameters to and handed to the body's frame, so that the body can
        use them as values (see ``operand``)."""
        st = self._symbol_table
        name = f"{fn_entry.name_base}({call_sig})"
        if call_sig in fn_entry.specs:
            instance = fn_entry.specs[call_sig]
            if instance.native_fn is not None:
                # this function was compiled in an earlier round
                assert instance.ret_sig is not None
                return (
                    self._resolve_extern_anon_symbol(
                        name, instance.native_fn, instance.mir.get_type()
                    ),
                    instance.ret_sig,
                )
            if instance.mir.is_complete:
                # compiled in this round
                assert instance.ret_sig is not None
                return instance.mir, instance.ret_sig
            # still being compiled: a recursive reference to the very
            # function being typed; its return type must be known already
            actual = instance.ret_sig or ret_sig
            if actual is None:
                raise CompileError(
                    f"recursive function {fn_entry.hir.name} requires a return type annotation"
                )
            return instance.mir, actual
        mir_fn = mir.Function(name, [], [], mir.VOID, [])
        instance = FunctionInstance(mir_fn)
        st.newly_compiled.add(instance)
        fn_entry.specs[call_sig] = instance
        runner = HirRunner(self, instance)
        frame_generic_values: dict[sval.TypeVar, InterpVal] = {}
        if generic_var_values is not None:
            frame_generic_values.update((tv, ComptimeVal(v)) for tv, v in generic_var_values.items())
        frame_generic_values.update(
            (tv, ComptimeVal(v))
            for tv, v in zip(fn_entry.hir.signature.generic_args, call_sig.generic_args)
        )
        runner.run_function(fn_entry.hir.body, call_sig, ret_sig, frame_generic_values)
        self._analyse_stack.append(runner)
        return None

    def _run(self):
        while self._analyse_stack:
            top = self._analyse_stack[-1]
            if top._run_machine() == PollResult.DONE:
                top.finish()
                assert top.return_sig is not None
                self._analyse_stack.pop()
                instance = top._fn_instance
                instance.ret_sig = top.return_sig
                instance.mir.is_complete = True
                if self._analyse_stack:
                    last_top = self._analyse_stack[-1]
                    last_top.resume(instance.mir, top.return_sig)
                else:
                    return instance.mir, top.return_sig
        return None

    def analyse_function(self, fn_entry: FunctionValue, call_sig: CallSignature, ret_sig: ReturnSignature | None):
        """Type (and thereby compile) the specialization ``call_sig`` of
        ``fn_entry`` if it is not compiled yet."""
        if self._request_function(fn_entry, call_sig, ret_sig) is None:
            self._run()

    def finish(self) -> CompileBatch:
        return self._symbol_table

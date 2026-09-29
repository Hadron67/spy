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
  Python ``None``, and the ``None`` the source writes is the absent value
  of an option, ``sval.Null()``,
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
  value), or
* :class:`ComptimeResult` - the result location of a function: the places
  its normal value, its error code and its payload union are delivered into
  (the interpreter's form of a :class:`sval.ResultType`).

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

An inlined body is emitted into the block the call sits in; the caller's
continuation is a fresh *exit block* that every return of the body jumps
to (a falling end joins it too).  That block is created the first time a
path of the body reaches it: a body whose every path ends elsewhere (a
``raise`` that leaves it) never gets one, and the caller's code after the
call is then dead.  An inlined body may contain runtime ``if`` branches
like the function proper.  Every return of the body stores its value into
the call's result location on its own runtime path, whatever the path is,
so the result location's memory is the join of the paths (its alloca is
hoisted by ``lower`` so every path shares one address).
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
    PartialReturnSignature,
    RawArgList,
    ReturnSignature,
    Signature,
    SpecializedComptimeArg,
    SpecializedFormalArg,
    SpecializedRuntimeArg,
)
from .hir import InlineMode
from .sval import (
    GlobalResolver,
    RetSpec,
    RetTuple,
    RetValue,
    iter_ret_leaves,
    ret_by_value_index,
)
from .util import ArraySet, frozendict

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
    """A comptime-time writable box for non-aggregate values. Note that
    ``value`` does not have to be a comptime-time value: it also can be
    a runtime value :class:`RuntimeVal`. Supports non-aggregate values only,
    for aggregate values, use :class:`ComptimeAggregate` or
    :class:`ComptimeAggregatePtr` instead."""

    type: sval.Type
    value: InterpVal


class _PendingActionData:
    """One action recorded by a :class:`PendingSlot`: how it is delivered
    once the slot has an address is decided by the runner (see
    ``HirRunner._exec_pending_action``), the slot itself only asks for the
    type it contributes and whether it is inline."""

    @abstractmethod
    def info(self) -> tuple[sval.Type, bool]:
        """Returns (type, is_inline)"""
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
    the stored value and whether it may be inlined.  The store itself is
    delivered once the slot's final type is known (the stored value is
    coerced to it then)."""

    type: sval.Type
    is_inline: bool
    value: InterpVal

    def info(self) -> tuple[sval.Type, bool]:
        return self.type, self.is_inline

@dataclass
class _PendingPtrConvertion(_PendingActionData):
    type: sval.Type
    input: InterpVal
    output: mir.Insertion

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return self.type, False

@dataclass
class _PendingTuple(_PendingActionData):
    # one tuple initialization recorded by a PendingSlot (see ``init_tuple``):
    # the places the tuple's elements are written into.  A tuple has no
    # representation of its own, so the slot holds the tuple of places itself
    # (see ``ComptimeTuple``), and its type - the tuple of the types of those
    # places - is read when the slot is committed, once the elements have been
    # written.  Recording it as an action (rather than committing the slot
    # right away) keeps the slot's type resolution - and the conflict it
    # reports against any other store into the slot - intact.
    places: tuple[ArgEntry[InterpVal], ...]

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return _tuple_places_type(self.places), True

@dataclass
class _PendingAggregate(_PendingActionData):
    # one compile-time aggregate initialization recorded by a PendingSlot (see
    # ``HirRunner.finish_array``/``HirRunner.finish_struct``): the places of its
    # fields, in declaration order, or of its elements.  Like a tuple, an
    # aggregate has no representation of its own that a slot could hold: the
    # slot holds the aggregate pointer itself (see ``ComptimeAggregatePtr``),
    # which is why the type recorded here is the struct or array type the
    # construction resolved, and the places are committed together with the slot
    # (see ``HirRunner.init_inline_aggregate``).
    type: sval.Type
    places: tuple[InterpVal, ...]

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return self.type, True

@dataclass(slots=True)
class PendingSlot(InterpVal):
    """The value of an executed ``hir.Alloca`` before it is *committed*.
    In this phase a store (or an RLS call) into the slot only records a
    :class:`_PendingAction`; the slot acquires its final type (the
    pairwise ``resolve_peer_type`` of the action types) and its storage
    when ``hir.CommitSlot`` runs, which materializes it into a
    :class:`RuntimeVal` (a pointer to real memory), a :class:`ComptimeBox`
    or - for an aggregate - a :class:`ComptimeAggregatePtr` (see
    ``HirRunner._commit_pending_slot``).

    ``inline_mode`` says how much of the value may be kept inline: nothing (a
    plain slot), anything but an aggregate (an expression temporary), or
    anything (a ``Comptime`` variable) - see ``InlineMode``.

    ``insertion`` is the position the slot's storage is produced at: a
    :class:`mir.Insertion` emitted where the ``Alloca`` ran, whose
    instructions the slot's commit fills with the :class:`mir.Alloca` (or,
    for an element/field place of an aggregate construction, with the
    ``Gep`` addressing it, see ``finish_array``/``finish_struct``)."""

    insertion: mir.Insertion
    inline_mode: InlineMode
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

    def is_inline(self, type: sval.Type) -> bool:
        """Whether the slot may hold the value of type ``type`` inline (in a
        :class:`ComptimeBox`, or as a :class:`ComptimeAggregatePtr` for an
        aggregate) rather than in memory.  A ``FULL`` slot holds *any* aggregate
        that way, whatever the values of its fields are - a compile-time
        aggregate is its fields' own places (see ``ComptimeAggregatePtr``) -
        unless a delivery into the slot needs an address of its own: a result
        pointer a callee writes through (see ``_defer_ptr_convertion``), which a
        place held by its fields has no single address to hand over.  Otherwise
        the mode has to allow it (see ``InlineMode``) and every store into the
        slot has to be of an inline value (see ``_is_inline_val``) - a runtime
        value written into a ``NON_AGGREGATE`` slot or into the field of a
        compile-time aggregate has to land in memory, since the runtime paths
        that write it (the two branches of a runtime ``if``, e.g.) only join
        there - and a ``NON_AGGREGATE`` slot holds no aggregate at all.  A
        zero-sized value has no runtime representation, so any mode keeps its
        unit value."""
        if self.inline_mode == InlineMode.NONE:
            return False
        if self.inline_mode == InlineMode.FULL and _is_aggregate(type):
            return not any(
                isinstance(store.data, _PendingPtrConvertion) for store in self.stores
            )
        if not all(store.data.info()[1] for store in self.stores):
            return False
        return not (self.inline_mode == InlineMode.NON_AGGREGATE and _is_aggregate(type))

@dataclass(frozen=True, slots=True)
class ComptimeResult(InterpVal):
    """The value form of a :class:`sval.ResultType`: the result location of a
    function - the place its normal result is delivered into, the place of its
    error code and the place of its payload union, in that order.  The places
    are :class:`PendingSlot`s while the location is not committed yet (the code
    and payload widths follow from the exception set at the commit, see
    ``_commit_error_space``), and the materialized places afterwards (a
    ``ComptimeBox`` for a zero-sized one)."""

    value: InterpVal
    code: InterpVal
    payload: InterpVal

@dataclass(frozen=True, slots=True)
class ComptimeTuple(InterpVal):
    values: tuple[ArgEntry[InterpVal], ...]

@dataclass
class ComptimeDict(InterpVal):
    values: dict[str, ArgEntry[InterpVal]]

@dataclass(frozen=True, slots=True)
class ComptimeAggregate(InterpVal):
    type: sval.Type
    values: tuple[InterpVal, ...]

@dataclass(frozen=True, slots=True)
class ComptimeAggregatePtr(InterpVal):
    type: sval.Type
    ptrs: tuple[InterpVal, ...]

@dataclass
class _PendingErrorCodeWrite:
    """One write of the error code of ``exception`` into the function's own
    error location (an error escaping the function, or one raised through a
    result location): the code of an exception depends on the function's
    *result type* - a value-less function has no "no error" code, so its tags
    start at 0 - which may still be inferred while the body runs.  The write is
    therefore deferred to the insertion emitted at its position and filled in
    by ``HirRunner._finish_function``, once the result type is known."""

    exception: sval.Type
    insertion: mir.Insertion

def _is_comptime_val(val: InterpVal) -> bool:
    """Whether the value is *deeply* compile-time: it is known in full while
    the HIR runs, so a computation over it can be folded in Python (see
    ``_eval_binary`` and friends) and a call may take it for a compile-time
    parameter.  A container is deeply compile-time when everything it holds
    is, so a box or a tuple holding a runtime value is not.

    Not to be confused with ``_is_inline_val``, the *shallow* property that
    decides whether a value may live in a :class:`ComptimeBox`."""
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
            case ComptimeBox():
                # a compile-time box is comptime only when the value it holds
                # is (it may hold a runtime value, see ``ComptimeBox``)
                todo.append(val.value)
            case ComptimeTuple():
                todo.extend(a.value for a in val.values)
            case ComptimeDict():
                todo.extend(a.value for a in val.values.values())
            case ComptimeAggregate():
                # a compile-time aggregate is comptime when everything it holds
                # is, like a tuple or a box (see ``ComptimeAggregate``)
                todo.extend(val.values)
            case ComptimeAggregatePtr():
                todo.extend(val.ptrs)
    return True

def _is_inline_val(val: InterpVal) -> bool:
    """Whether the value may be *inlined* - kept as a compile-time value (in a
    :class:`ComptimeBox`, or as a :class:`ComptimeAggregatePtr` for an
    aggregate) rather than written into memory.  This is the *shallow* property
    of the value itself: it is not a runtime value.  A container counts as
    inline even when what it holds is a runtime value - a tuple, a box, ... has
    no runtime representation of its own, so it only exists while the HIR runs
    (a ``Comptime`` variable may hold one, see ``_is_comptime_val`` for the
    deep property)."""
    return not isinstance(_shallow_normalize(val), RuntimeVal)

def _is_aggregate(type: sval.Type) -> bool:
    """Whether ``type`` is an *aggregate*: a struct or an array.  An aggregate
    has no storage of its own here - an inline slot holds it as a
    :class:`ComptimeAggregatePtr`, whose fields or elements are their own
    places - so a :class:`ComptimeBox` never holds one (see ``InlineMode``).

    A union is not one: its storage is a single variant, not a field per
    place."""
    return isinstance(type, (sval.StructType, sval.ArrayType))

def _is_union_unit(val: InterpVal) -> bool:
    """Whether ``val`` is a union value that carries no storage
    (:class:`sval.UnionValue`): it says only which union it belongs to, so a
    storage destination has no variant to write."""
    obj = _to_comptime(val)
    if isinstance(obj, sval.AsValue):
        obj = obj.value
    return isinstance(obj, sval.UnionValue)

def _aggregate_place_types(type: sval.Type) -> tuple[sval.Type, ...]:
    """The type of every place of the aggregate ``type``, in place order: the
    fields of a struct in declaration order, the elements of an array."""
    if isinstance(type, sval.StructType):
        return tuple(field0.type for field0 in type.fields().values())
    if isinstance(type, sval.ArrayType):
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot tell how many elements {type} holds')
        return (type.elem,) * length
    raise CompileError(f'{type} is not an aggregate')

def _as_aggregate(ev: InterpVal) -> ComptimeAggregate | None:
    """The aggregate value an interpreter value denotes, when it denotes one:
    the aggregate value form itself, or an ``sval.AggregateValue`` a
    compile-time value holds - the unit value of a zero-sized aggregate, or an
    aggregate whose field values are all known (see ``ComptimeAggregate``)."""
    if isinstance(ev, ComptimeAggregate):
        return ev
    if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.AggregateValue):
        return ComptimeAggregate(ev.obj.type, tuple(ComptimeVal(value) for value in ev.obj.values))
    return None

class BlockFrameData:
    pass

@dataclass
class IfBlockData(BlockFrameData):
    """The state of one open ``if`` of the HIR.

    ``chosen`` is set for a compile-time ``if`` (the branch the walk went
    into) and None for a runtime one; ``then_returns`` records whether the
    then-region ended by returning - it is None until that region is
    either returned out of or walked off.  ``p_else``/``p_end`` are the
    marker positions found by ``_scan_block``.  The MIR blocks are the
    ones the two branches are built into; a compile-time ``if`` has none
    (only the chosen branch is emitted, straight into the current block)."""

    chosen: bool | None = None
    then_returns: bool | None = None
    p_else: int | None = None
    p_end: int = 0
    then_block: mir.BasicBlock | None = None
    else_block: mir.BasicBlock | None = None
    exit_block: mir.BasicBlock | None = None

@dataclass
class LoopBlockData(BlockFrameData):
    """The state of one open ``loop`` of the HIR.

    ``header_block`` is the loop's head - the block the condition is
    computed in, the target of the back edge the body's falling end and
    every ``continue`` emit.  It is None for a compile-time (``is_inline``)
    loop, which has no back edge: its body is unrolled and the iterations
    fall through one into the next.  ``exit_block`` is the block the code
    after the loop continues in, created on the first ``break`` that reaches
    it: the loop can be left exactly when some ``break`` (a source one, or
    the implicit one the ``while`` lowering puts after its else clause)
    jumped there, so a loop without one (``while True:``) never gets an exit
    and the code after it is dead (see ``_cut``).  ``inline_next`` is the
    entry block of the next unrolled body, created on the first
    ``continue`` of the current iteration (an inline loop only).  ``p_end``
    is the position of the matching ``End`` marker (found by
    ``_scan_block``)."""

    p_end: int
    is_inline: bool = False
    header_block: mir.BasicBlock | None = None
    exit_block: mir.BasicBlock | None = None
    inline_next: mir.BasicBlock | None = None

@dataclass
class TryExceptBlockData(BlockFrameData):
    """The state of one open ``try`` of the HIR.

    ``region`` is the region being walked: 0 is the try body, ``i + 1`` the
    i-th except clause.  Every error raised while the body is typed is
    dispatched straight to the clause that catches it (see
    ``HirRunner._find_catching_clause``); a clause's entry block is created
    lazily, on the first dispatch that reaches it, so a clause no error reaches
    is dead code - its HIR is never typed.  ``payload_phi[i]`` is the ``Phi``
    clause ``i``'s error payload pointer is delivered through (the clause's
    ``binds[i]``, an :class:`hir.ExceptBind`, reads it) and ``code_phi[i]`` the
    ``Phi`` carrying the remapped error code of a *bare* clause.  ``join`` is the
    block the falling regions continue in, ``binds`` holds the ``as`` bind
    of every clause and ``except_types`` every clause's type expression -
    evaluated before the ``Try`` - (see ``hir.Try``)."""

    insts: tuple[hir.Inst, ...]
    p_excepts: list[int]
    p_end: int
    binds: tuple[hir.Value | None, ...]
    except_types: tuple[hir.Value | None, ...]
    join: mir.BasicBlock
    clause_blocks: list[mir.BasicBlock | None] = field(default_factory=list)
    payload_phi: list[mir.Phi | None] = field(default_factory=list)
    code_phi: list[mir.Phi | None] = field(default_factory=list)
    # the dispatches into a *bare* clause, whose union - and so the phi types -
    # is only fixed once the try body has been walked; one (clause, case block,
    # variant pointer, exception type) per dispatch (see
    # ``_begin_except_clause``)
    bare_pending: list[tuple[int, mir.BasicBlock, InterpVal, sval.Type]] = field(default_factory=list)
    region: int = 0
    # the try body has been walked: errors at the current position no longer
    # belong to this try
    body_done: bool = False
    # whether any region (the body or a clause) fell through to ``join``
    fell: bool = False

@dataclass
class BlockFrame:
    entry: int
    data: BlockFrameData

class InlineFrame:
    def __init__(self, generic_var_values: dict[sval.TypeVar, InterpVal], arg_values: tuple[InterpVal, ...], ret_loc: ComptimeResult, insts: tuple[hir.Inst, ...], value_is_empty: bool = False) -> None:
        self.generic_var_values = generic_var_values
        self.arg_values = arg_values
        # the frame's result location: the place its result is delivered into and
        # the function proper's error places (an inlined plain body raises into
        # the enclosing function's error location, so it shares them)
        self.ret_loc: ComptimeResult = ret_loc
        # whether the body of this frame has no value to return (the *declared*
        # one of an inlined plain-Python body; the function proper's own type is
        # asked for when it runs, see ``HirRunner._current_value_is_empty``)
        self.value_is_empty = value_is_empty
        self.insts = insts
        self.pc: int = 0
        self.block_stack: list[BlockFrame] = []
        self.regs: dict[hir.Inst, InterpVal] = {}
        # the block the caller of the inlined body continues in: created by
        # ``continuation`` the first time a path of the body reaches it, so that
        # a body whose every path ends elsewhere (a ``raise`` leaving it) has
        # none - which is what makes the caller's code after the call dead (see
        # ``_pop_frame``); None for the function proper, which has no caller
        self.exit_block: mir.BasicBlock | None = None

    def continuation(self) -> mir.BasicBlock:
        """The block the caller continues in once the inlined body ends,
        created on the first path that reaches it: every ``return`` of the body
        jumps to it and its falling end joins it too (see ``_pop_frame``)."""
        if self.exit_block is None:
            self.exit_block = mir.BasicBlock()
        return self.exit_block

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
        case ComptimeAggregate(type):
            # a struct value held compile-time: the struct type (see
            # ``ComptimeAggregate``)
            return type
        case ComptimeAggregatePtr(type):
            # the compile-time storage of a struct: a pointer to it
            return sval.PointerType(type, is_const=False)
        case RuntimeVal(_, type):
            if isinstance(type, sval.ValueType) and not allow_value_type:
                return sval.type_of(type.value)
            return type
        case ComptimeVal(obj):
            return sval.type_of(obj) if not allow_value_type else sval.ValueType(obj)
        case ComptimeTuple():
            # a tuple of values: its type is the tuple of the types of the
            # values its entries denote, with no runtime representation of
            # its own (see ``sval.TupleType``)
            types: list[sval.Type] = []
            for entry in ev.values:
                entry_type = _arg_type_of(entry)
                if entry_type is None:
                    return None
                types.append(entry_type)
            return sval.TupleType(tuple(types), False)
        case _:
            return None

def _is_null(ev: InterpVal) -> bool:
    """Whether the value ``ev`` denotes is the ``Null`` value (the absent
    value of an option).  A store of a ``None`` records the store type
    :class:`sval.NullType`, which is how an uncommitted slot remembers the
    delivery is absent (the value has no runtime representation of its own)."""
    ev = _shallow_normalize(ev)
    if isinstance(ev, ComptimeVal):
        return isinstance(ev.obj, sval.Null)
    return isinstance(_type_of(ev), sval.NullType)

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
# stateless helpers of field/element access, struct/array construction and the
# result location of a function that returns several values: pure functions
# over the interpreter values they are given (the struct/array structure is
# read off those, not off any runner state)
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

def _tuple_places_type(places: tuple[ArgEntry[InterpVal], ...]) -> sval.Type:
    # the type of a tuple of element places (see ``init_tuple``): a tuple has
    # no representation of its own, so this is the tuple of the types of the
    # places themselves
    types: list[sval.Type] = []
    for place in places:
        type = _place_type(place.value)
        if type is None:
            raise CompileError('a tuple element has no type yet')
        types.append(type)
    return sval.TupleType(tuple(types), False)


def _union_contains(superset: sval.UnionType, subset: sval.UnionType) -> bool:
    """Whether every variant of ``subset`` is a variant of ``superset``: the
    two unions then share their layout (every variant lives at offset 0, and the
    superset's storage holds the subset's), so one's pointer can be used as the
    other's (see ``HirRunner._convert_result_ptr``)."""
    return all(exception in superset.types for exception in subset.types)


def _result_places(location: InterpVal) -> tuple[InterpVal, ...]:
    """The leaf places one result value each is delivered into, in depth-first
    declaration order: the result location of a function whose annotation
    declares several results is a tuple of slots (nested exactly like the
    results), every other function has a single result slot."""
    places: list[InterpVal] = []
    work: list[InterpVal] = [location]
    while work:
        place = _shallow_normalize(work.pop())
        if isinstance(place, ComptimeTuple):
            work.extend(reversed([entry.value for entry in place.values]))
        elif isinstance(place, ComptimeResult):
            # the result spreads into its value, its error code and its payload,
            # in that order (see ``sval.make_ret_spec``)
            work.append(place.payload)
            work.append(place.code)
            work.append(place.value)
        else:
            places.append(place)
    return tuple(places)


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
    type of the value written into it.  A field constrains nothing when its
    place has no type yet."""
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
    of its own: a compile-time-only type (the type of an untyped integer
    literal, a type variable, ...) is called out by name, since the location
    has to declare the type such a value is resolvable at compile time with.
    A zero-sized type never gets here: it has a value (its unit value) and no
    runtime location needs to hold it."""
    if isinstance(type, sval.TupleType):
        return CompileError(
            'a multi-value result must be destructured (``a, b = f()``) or held '
            'by a compile-time variable (``a: Comptime = f()``)'
        )
    if type.classify() == sval.SpecialTypeKind.COMPTIME:
        return CompileError(
            f'{type} is a compile-time-only type: a runtime location cannot '
            f'hold it and must declare its type'
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
        return fn(_comptime_py_value(lhs), _comptime_py_value(rhs))
    except Exception as e:
        raise CompileError(
            f"cannot apply '{op}' to {lhs!r} and {rhs!r} at compile time: {e}"
        ) from e

def _comptime_py_value(value: Any) -> Any:
    # a *typed* scalar constant (the value a compile-time location holds) is
    # operated on as the Python value it is, so that the operators are the
    # ordinary ones (an ``Int``/``Float`` has none of its own); every other
    # compile-time object (a type, a null, ...) is used as it is
    return value.value if isinstance(value, (sval.Int, sval.Float)) else value

def _convert_inst(
    value: mir.Value, from_type: sval.Type, to_type: sval.Type, cache: sval.MirLowerCache
) -> mir.Inst | None:
    """Build (but do not emit) the conversion of ``value`` from
    ``from_type`` to ``to_type``; returns ``None`` when no conversion
    instruction is needed (the types are equal, or both are pointers)."""
    if from_type == to_type:
        return None
    if isinstance(from_type, sval.UnionType) and isinstance(to_type, sval.UnionType):
        # a union value cannot be converted: its storage has to be
        # reinterpreted through a pointer instead (see ``_convert_result_ptr``),
        # so only a store of the very same union is expressible
        raise CompileError(
            f'cannot convert a {from_type} value to {to_type}: the storage of a '
            f'union is written through a pointer to it'
        )
    mir_to_type = to_type.to_mir_type(cache)
    assert mir_to_type is not None and not to_type.is_zst()
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

def _check_comptime_args(
    sig: Signature, args: ArgList[ArgEntry[InterpVal]]
) -> None:
    """Require a *deeply* compile-time value for every parameter declared
    compile-time (``Comptime``/``Comptime[T]``): the callee reads such a
    parameter as a compile-time value whatever it is given, so a runtime
    argument would be silently dropped (see ``SpecializedComptimeArg``).
    Every other parameter may be given a runtime value - a zero-sized one
    is delivered as its unit value whatever it is given (see
    ``Signature.specialize``)."""
    for (name, formal), arg in zip(sig.positional.items(), args.positional):
        if not formal.is_comptime:
            continue
        if not _is_comptime_val(arg.value):
            raise CompileError(
                f"the argument of compile-time parameter '{name}' must be "
                f"a compile-time value"
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
        # the host's MIR-mirror interning table (see ``Analyser``), through
        # which every mirror a type is lowered to is made
        self._mir_cache = analyser.mir_lower_cache
        # the frames of the function bodies under execution: the function
        # proper at the bottom, one frame per inlined plain function
        # above it (see ``_in_function_proper``; each frame carries the
        # HIR of its body, see ``InlineFrame``)
        self._frames: list[InlineFrame] = []
        # the function proper whose body is currently being typed (see
        # ``_materialize_result_ptr``)
        self._fn_instance = fn_instance
        # the basic block currently being built, and the insertion block a
        # pending action is being delivered into (None outside one, see
        # ``_emit``)
        self._cur_block: mir.BasicBlock = fn_instance.mir.entry
        self._insertion: mir.Insertion | None = None
        # the ``hir.Ret`` positions of the function proper whose return
        # convention is not fixed yet (an unannotated return type); their
        # ``mir.Ret`` is filled in by ``_finish_function`` once the result
        # location has been materialized
        self._deferred_returns: list[mir.Insertion] = []
        # the exception set of the function proper's own error location, in
        # first-delivery order (its error codes follow from this order, see
        # ``sval.ResultType``); it grows with every delivery while the set is
        # inferred
        self._error_types: ArraySet[sval.Type] = ArraySet()
        # the writes of those error codes, deferred until the result type (which
        # decides the tags the function uses) is known (see
        # ``_PendingErrorCodeWrite``)
        self._pending_error_code_writes: list[_PendingErrorCodeWrite] = []
        # the return convention the signature declares, with the parts the body
        # still has to infer missing (see ``PartialReturnSignature``)
        self.partial_ret_sig: PartialReturnSignature | None = None
        # the complete return convention, fixed by ``_materialize_ret_sig`` once
        # its parts are known
        self.ret_sig: ReturnSignature | None = None

        self._fn_req_resumer: Callable[[Self, mir.Value, ReturnSignature], PollResult] | None = None
        # the number of compile-time loop bodies unrolled so far, and the limit
        # that keeps a non-terminating loop marked with ``syntax.unroll()`` (one
        # whose condition never becomes false) from unrolling forever (see
        # ``_unroll_inline_loop``)
        self._loop_unrolls: int = 0
        self.max_loop_unroll: int = 1024

    # -- entry point ---------------------------------------------------------

    def run_function(
        self,
        body: tuple[hir.Inst, ...],
        arg_is_ref: tuple[bool, ...],
        sig: CallSignature,
        ret_sig: PartialReturnSignature | None,
        generic_var_values: dict[sval.TypeVar, InterpVal],
    ):
        # reset the per-specialization state; the result location of the
        # function proper is reserved first so its slot sits at a known
        # position in the body
        self.partial_ret_sig = ret_sig
        self.ret_sig = None
        self.resume_info = None
        self._deferred_returns = []
        self._pending_error_code_writes = []
        self._error_types = ArraySet()
        if ret_sig is not None and ret_sig.exceptions is not None:
            # a declared exception set is the function's own from the start: its
            # order fixes the error codes, whatever the value type turns out to be
            for exception in ret_sig.exceptions.values:
                self._error_types.add(exception)
        frame = InlineFrame(
            generic_var_values, (), self._reserve_result_loc(ret_sig), body,
        )
        self._frames.append(frame)
        mir_args = self._fn_instance.mir.args
        args = self._init_args_from_signature(sig, mir_args, arg_is_ref)
        for arg in mir_args:
            assert arg is not None
        frame.arg_values = args
        if ret_sig is not None and ret_sig.is_complete():
            # the whole return convention is declared: fix it before the body
            self._materialize_ret_sig(ret_sig.complete())

    def _reserve_result_loc(self, ret_sig: PartialReturnSignature | None) -> ComptimeResult:
        """The result location of the function proper (see ``ComptimeResult``):
        the place its declared result is delivered into and its error places.
        The places are the ones the effective spec names when the signature
        declares the whole return convention, and fresh slots otherwise - an
        inferred error place has no width until its set is known (see
        ``_materialize_ret_sig``)."""
        declared_value = ret_sig.ret_type_spec if ret_sig is not None else None
        if ret_sig is not None and ret_sig.is_complete():
            # the value and the error part are both declared: the result
            # location is the one the effective spec names
            loc = self._ret_spec_place(ret_sig.complete().ret_spec())
            assert isinstance(loc, ComptimeResult)
            return loc
        if declared_value is not None:
            value_loc = self._ret_spec_place(declared_value)
        else:
            value_loc = self.alloca(InlineMode.NONE)
        return ComptimeResult(
            value_loc, self.alloca(InlineMode.NONE), self.alloca(InlineMode.NONE),
        )

    def _init_one_arg(self, node: SpecializedFormalArg, mir_args: list[mir.Type], arg_is_ref: bool) -> InterpVal:
        match node:
            case SpecializedComptimeArg():
                return ComptimeVal(sval.ConstRef(node.value))
            case SpecializedRuntimeArg():
                mir_type = node.type.to_mir_type(self._mir_cache)
                if mir_type is None or node.type.is_zst():
                    raise _no_runtime_type(node.type)
                index = len(mir_args)
                if node.is_ref:
                    # the signature passes the address of the value as a const
                    # pointer
                    arg_mir = mir.PointerType(mir_type, True)
                    arg_sval = sval.PointerType(node.type, True)
                else:
                    arg_mir = mir_type
                    arg_sval = node.type
                mir_args.append(arg_mir)
                if node.is_ref or arg_is_ref:
                    # the HIR binds the parameter directly to its argument - the
                    # address of the value - and a read of the name loads it.  A
                    # method's ``self`` is such a parameter (its type is already
                    # a pointer, so no extra indirection is added) without being
                    # a by-reference one in the signature
                    return RuntimeVal(mir.Param(index, arg_mir), arg_sval)
                slot = self.alloca()
                self._commit_pending_slot(slot, node.type)
                self.store(slot, RuntimeVal(mir.Param(index, arg_mir), node.type))
                return slot
            case _:
                raise CompileError(f'unsupported specialized argument {node!r}')

    def _init_args_from_signature(
        self,
        signature: CallSignature,
        mir_args: list[mir.Type],
        arg_is_ref: tuple[bool, ...],
    ) -> tuple[InterpVal, ...]:
        arg_values: list[InterpVal] = []

        for (_, arg), is_ref in zip(signature.positional, arg_is_ref):
            arg_values.append(self._init_one_arg(arg, mir_args, is_ref))

        if signature.varargs:
            arg_values.append(ComptimeTuple(tuple(ArgEntry(self._init_one_arg(a, mir_args, False), True) for a in signature.varargs)))
        if signature.kwargs:
            arg_values.append(ComptimeDict({k: ArgEntry(self._init_one_arg(v, mir_args, False), True) for k, v in signature.kwargs.items()}))

        return tuple(arg_values)

    def _ret_spec_place(self, node: RetSpec) -> InterpVal:
        """The result place one declared result is delivered into: a fresh slot
        for a value, a tuple of the places of its elements for a group, and the
        function's result location for a result-type group (its exception set and
        storage are fixed when it is committed, see ``_commit_error_space``)."""
        match node:
            case RetValue():
                return self.alloca(InlineMode.NONE)
            case RetTuple(type=type, values=values):
                if isinstance(type, sval.ResultType):
                    # the function's own result location: its value, its error
                    # code and its payload union (the set and the storage are
                    # fixed when the group is committed, see ``_commit_error_space``)
                    return ComptimeResult(
                        self._ret_spec_place(values[0]),
                        self.alloca(InlineMode.NONE),
                        self.alloca(InlineMode.NONE),
                    )
                entries: list[ArgEntry[InterpVal]] = []
                for child in values:
                    place = self._ret_spec_place(child)
                    entries.append(ArgEntry(place, isinstance(child, RetValue)))
                return ComptimeTuple(tuple(entries))

    def _materialize_ret_sig(
        self, sig: ReturnSignature
    ) -> None:
        """Fix the return convention of the function proper from the spy
        types of the values it returns (its declared annotation, or the peer
        type of all its store points for an inferred single result).  A
        result delivered through a result pointer appends the hidden result
        pointer formal to the lowered signature *after* every declared
        argument and makes the result location the memory of that pointer; a
        result returned by value fixes the MIR return type and leaves its
        location recording the value.  A function whose every result goes
        through a pointer returns void.

        The convention is a property of the result types
        (``sval.make_ret_spec``) unless the signature declares it.
        """
        spec = sig.ret_spec()
        if self.ret_sig is not None:
            if self.ret_sig != sig:
                raise CompileError(
                    f"function returns values of conflicting types "
                    f"{[leaf.type for leaf in iter_ret_leaves(self.ret_sig.ret_spec())]} and "
                    f"{[leaf.type for leaf in iter_ret_leaves(spec)]}"
                )
            return
        self.ret_sig = sig
        self._fn_instance.mir.ret_type = mir.VOID
        self._commit_ret_places(spec, self._current_result_loc())
        if sig.is_noreturn():
            # the function has no value to return and raises nothing: no path
            # of it can ever come back, which its lowered form says outright
            self._fn_instance.mir.ret_type = mir.NORETURN

    def _commit_ret_places(self, node: RetSpec, place: InterpVal) -> None:
        """Commit the result place(s) mirroring the spec node ``node``: a leaf
        place is committed (through a hidden result pointer when the leaf says
        so), and a result-type group - whose place is the result location - is
        committed as a whole, its places being the value, the error code and the
        payload union of the result."""
        match node:
            case RetValue():
                self._commit_pending_slot(place, node.type, ptr=self._ret_leaf_ptr(node))
            case RetTuple(type=type, values=values):
                if isinstance(type, sval.ResultType):
                    assert isinstance(place, ComptimeResult)
                    value, code, payload = values
                    self._commit_result_place(value, place.value)
                    code_ptr = self._ret_leaf_ptr(code)
                    payload_ptr = self._ret_leaf_ptr(payload)
                    self._commit_error_space(place, type, code_ptr, payload_ptr)
                    return
                assert isinstance(place, ComptimeTuple)
                for child, entry in zip(values, place.values):
                    self._commit_ret_places(child, entry.value)

    def _commit_result_place(self, node: RetSpec, place: InterpVal) -> None:
        """Commit the place(s) of the value part of a result (the first element
        of a result-type group): a leaf is committed - through the hidden result
        pointer it is delivered through when it says so - and a nested group of
        values by pairing it with the place tree the caller reserved for it."""
        match node:
            case RetValue():
                self._commit_pending_slot(place, node.type, ptr=self._ret_leaf_ptr(node))
            case RetTuple(type=type, values=values):
                assert not isinstance(type, sval.ResultType)
                assert isinstance(place, ComptimeTuple)
                for child, entry in zip(values, place.values):
                    self._commit_result_place(child, entry.value)

    def _ret_leaf_ptr(self, leaf: RetSpec) -> mir.Value | None:
        """The hidden result pointer a leaf is delivered through (appending the
        formal to the lowered signature), or None when the leaf is returned by
        value (which fixes the MIR return type)."""
        assert isinstance(leaf, RetValue)
        if leaf.type.classify() == sval.SpecialTypeKind.DST:
            # a dynamically-sized type has no runtime value a return could
            # deliver: only a pointer to one can be returned
            raise CompileError(
                f'cannot return a value of the dynamically-sized type {leaf.type}'
            )
        mir_fn = self._fn_instance.mir
        if leaf.via_result_ptr:
            mir_type = leaf.type.to_mir_type(self._mir_cache)
            if mir_type is None or leaf.type.is_zst():
                raise CompileError(f'cannot return {leaf.type} through a result pointer')
            ptr_type = mir.PointerType(mir_type, False)
            index = len(mir_fn.args)
            mir_fn.args.append(ptr_type)
            mir_fn.arg_names.append('$result')
            return mir.Param(index, ptr_type)
        if not leaf.type.is_zst():
            # a zero-sized result is delivered as its unit value; only a
            # result with storage fixes the MIR return type
            mir_ret = leaf.type.to_mir_type(self._mir_cache)
            if mir_ret is None:
                raise _no_runtime_type(leaf.type)
            mir_fn.ret_type = mir_ret
        return None

    # -- return statements ------------------------------------------------

    def _current_result_loc(self) -> ComptimeResult:
        assert len(self._frames) > 0, 'no function result location'
        ret = self._frames[-1].ret_loc
        assert isinstance(ret, ComptimeResult)
        return ret

    def _result_loc(self) -> InterpVal:
        """The place the current frame's result is delivered into: the value
        part of the frame's result location (its error places are separate)."""
        return self._current_result_loc().value

    def _function_result(self) -> ComptimeResult:
        """The function proper's own result location (the frame at the bottom of
        the stack: an inlined body's result location shares its error places)."""
        ret = self._frames[0].ret_loc
        assert isinstance(ret, ComptimeResult)
        return ret

    def _current_value_is_empty(self) -> bool:
        """Whether the body being executed has no value to return, so that no
        path of it may deliver a result (see ``hir.Ret`` and
        ``store_void_retloc``): the function proper's settled value type (the
        empty type for a ``-> Never`` annotation, or a type parameter that solved
        to it), or the declared one of an inlined plain-Python body - an inlined
        body has no convention of its own, so only a declaration can say it."""
        if not self._in_function_proper():
            return self._frames[-1].value_is_empty
        if self.ret_sig is not None:
            return self.ret_sig.value_is_empty()
        partial = self.partial_ret_sig
        return partial is not None and sval.ret_spec_value_is_empty(partial.ret_type_spec)

    def _function_error_inferred(self) -> bool:
        """Whether the function proper's exception set is still to be inferred
        (``@func(exceptions="infer")``, or no annotation at all)."""
        partial = self.partial_ret_sig
        return partial is None or partial.exceptions is None

    def _find_catching_clause(self, exception: sval.Type) -> tuple[TryExceptBlockData, int] | None:
        """The clause that catches ``exception`` at the current position: the
        first clause of the innermost open try that names it (a bare ``except``
        matching anything), or None when no clause catches it and the error
        escapes the function.  The frames are searched innermost-last, so that
        an error of an inlined body belongs to the try blocks enclosing its
        caller."""
        for frame in reversed(self._frames):
            for bf in reversed(frame.block_stack):
                data = bf.data
                if not isinstance(data, TryExceptBlockData) or data.body_done:
                    continue
                for index, type_operand in enumerate(data.except_types):
                    clause_type = self._except_struct_type(type_operand)
                    if clause_type is None or clause_type == exception:
                        return data, index
        return None

    def _defer_return(self) -> None:
        """End a path whose return convention is not fixed yet: a placeholder
        that ``_finish_function`` fills with the ``mir.Ret`` once the effective
        spec is known (see ``hir.Ret``/``hir.Raise``).  The placeholder is not
        a terminator, so ``emit`` cannot end the block on its own - the path
        ends here all the same."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        self._cur_block.is_finished = True
        self._deferred_returns.append(insertion)

    def _emit_function_return(self) -> None:
        """Emit the ``mir.Ret`` that ends one path of the *function proper*
        (whatever inline frame the path sits in): its by-value result is
        loaded out of the function's result location, or none is returned when
        every result goes through a result pointer."""
        sig = self.ret_sig
        assert sig is not None
        spec = sig.ret_spec()
        places = _result_places(self._frames[0].ret_loc)
        index = ret_by_value_index(spec)
        if index is None:
            self._cur_block.emit(mir.Ret(None))
        else:
            self._cur_block.emit(mir.Ret(self._to_runtime(self.load(places[index]))))

    def _set_error_code_zero(self) -> None:
        """Record a successful outcome: the function proper's error code is
        cleared (a ``return`` path is defined to carry no error, and a path that
        returns a value cannot be one of a value-less function).  The store is
        made with at least one bit so that it takes part in the code slot's type
        even when no exception has been delivered yet - an inferred set may
        still grow later, and the successful path must clear the code either
        way."""
        self.store(self._function_result().code, ComptimeVal(sval.Int(0, sval.IntType(0, False))))

    def _end_error_path(self) -> None:
        """End a path at the function boundary (an error escaping the function):
        a typed return, or a deferred one when the convention is not fixed yet."""
        if self.ret_sig is None:
            self._defer_return()
        else:
            self._emit_function_return()

    def _raise(self, slot: InterpVal) -> PollResult:
        """End one path with ``hir.Raise``: the exception has been built into
        the slot ``slot``.  It is dispatched straight to the clause that catches
        its type when there is one, or written into the function's error
        location and returned otherwise; the enclosing HIR blocks are then
        unwound like a ``return``'s (see ``_cut``)."""
        exception = _place_type(slot)
        if not isinstance(exception, sval.StructType):
            raise CompileError(f'cannot raise {exception!r}: an exception must be a struct')
        target = self._find_catching_clause(exception)
        if target is not None:
            data, index = target
            if exception.is_zst():
                incoming = ComptimeVal(sval.Undefined(sval.PointerType(exception, is_const=False)))
            else:
                self._commit_pending_slot(slot, exception)
                incoming = _shallow_normalize(slot)
            self._route_to_clause(data, index, incoming, exception)
        else:
            self._deliver_uncaught_raise(slot, exception)
        return self._cut()

    # -- error locations ---------------------------------------------------

    def _commit_error_space(self, result: ComptimeResult, type: sval.ResultType, code_ptr: mir.Value | None = None, payload_ptr: mir.Value | None = None) -> None:
        """Commit an error location: its code and payload slots are materialized
        (through a hidden result pointer when given, freshly allocated
        otherwise), their widths following from the result type, and the
        deliveries recorded on them run against them.  The exception set becomes
        the function's own (the error codes follow from its order).

        A payload place is always storage (an alloca of its own when the payload
        is not delivered through a result pointer, see ``_ret_leaf_ptr``): the
        variant of an exception is built through its address either way, and a
        by-value payload return loads it back out of it.  A zero-sized payload
        union holds no storage and only records its unit value."""
        for exception in type.types:
            self._error_types.add(exception)
        self._commit_pending_slot(result.code, type.code_type, ptr=code_ptr)
        self._commit_pending_slot(result.payload, type.union, ptr=payload_ptr)

    def _add_function_exception(self, exception: sval.Type) -> None:
        """Record that the function proper may raise ``exception``: an exception
        the (declared) set does not allow is rejected, and an inferred set grows
        to hold it.  Its error code is written later (see
        ``_defer_error_code_write``), because the tags depend on the function's
        result type - which may still be inferred here."""
        if not isinstance(exception, sval.StructType):
            raise CompileError(f'cannot raise {exception}: an exception must be a struct')
        if exception not in self._error_types.values:
            if not self._function_error_inferred():
                raise CompileError(
                    f'{exception} is not an exception of this function: it '
                    f'cannot raise it (declare it with @func(exceptions={{...}}) '
                    f'or @func(exceptions="infer"))'
                )
            self._error_types.add(exception)

    def _defer_error_code_write(self, exception: sval.StructType) -> None:
        """Record the write of ``exception``'s error code into the function's own
        error location at the current position: the insertion is filled in by
        ``_finish_function``, once the result type names the tag."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        self._pending_error_code_writes.append(
            _PendingErrorCodeWrite(exception, insertion)
        )

    def _union_variant_ptr(self, place: InterpVal, struct_type: sval.StructType) -> InterpVal:
        """The address a value of the union variant ``struct_type`` is written
        to (or read from) in the payload union the place ``place`` points at:
        the place reinterpreted as a pointer to the variant type."""
        if struct_type.is_zst():
            return ComptimeVal(sval.Undefined(sval.PointerType(struct_type, is_const=False)))
        place = _shallow_normalize(place)
        ptr_type = _type_of(place)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'cannot take a variant of {place!r}')
        mir_struct = struct_type.to_mir_type(self._mir_cache)
        assert mir_struct is not None and not struct_type.is_zst()
        bitcast = self._emit(mir.BitCast(self._to_runtime(place), mir.PointerType(mir_struct)))
        return RuntimeVal(bitcast, sval.PointerType(struct_type, is_const=False))

    def _error_payload_ptr(self, error: ComptimeResult, struct_type: sval.StructType) -> InterpVal:
        """The address the variant ``struct_type`` is written to inside the
        error location ``error``."""
        return self._union_variant_ptr(error.payload, struct_type)

    def _raise_error(self, result: ComptimeResult, exception: sval.StructType, value: InterpVal) -> None:
        """Deliver one exception value into the error location ``result`` (a
        ``raise`` through a result location): tag it and write it into the
        payload."""
        self._add_function_exception(exception)
        self._defer_error_code_write(exception)
        self.store(self._error_payload_ptr(result, exception), value)

    def _payload_variant_ptr(self, exception: sval.StructType) -> InterpVal:
        """The address the variant ``exception`` lives at in the function's
        error location - a fresh placeholder while its payload has no address
        yet (see ``_defer_ptr_convertion``)."""
        payload = self._function_result().payload
        if isinstance(payload, PendingSlot) and payload.committed is None:
            return self._defer_ptr_convertion(payload, exception)
        return self._convert_result_ptr(payload, exception)

    def _use_ret_payload(self, callee_exceptions: tuple[sval.Type, ...]) -> bool:
        """Whether the callee's error payload is written straight into the
        function's own error location rather than a fresh slot: it is when no
        enclosing ``try`` catches any of the callee's exceptions (so the error
        is only propagated) and the function's declared set already holds them
        (or is still to be inferred, in which case it grows to)."""
        for exception in callee_exceptions:
            if self._find_catching_clause(exception) is not None:
                return False
        if self._function_error_inferred():
            return True
        return all(exception in self._error_types.values for exception in callee_exceptions)

    def _route_to_clause(self, data: TryExceptBlockData, index: int, incoming: InterpVal, exception: sval.StructType) -> None:
        """Route one caught error to clause ``index``: create the clause's entry
        block - and its error-payload ``Phi`` - on the first dispatch that
        reaches it, then have the current block jump to it with ``incoming`` (the
        address of the caught exception) as its payload pointer.  A bare clause's
        union - and so its phi types - is only fixed once the whole try body has
        been walked, so its dispatch is deferred through a case block."""
        block = data.clause_blocks[index]
        if block is None:
            block = mir.BasicBlock()
            data.clause_blocks[index] = block
        if self._clause_is_bare(data, index):
            case_block = mir.BasicBlock()
            self._cur_block.emit(mir.Jmp(case_block))
            data.bare_pending.append((index, case_block, incoming, exception))
            return
        if exception.is_zst():
            # a zero-sized exception has no payload: the clause is entered
            # without a payload pointer (its bind is the unit value)
            self._cur_block.emit(mir.Jmp(block))
            return
        value = self._to_runtime(incoming)
        phi = data.payload_phi[index]
        if phi is None:
            phi = mir.Phi([(value, self._cur_block)])
            data.payload_phi[index] = phi
            block.emit(phi)
        else:
            phi.add_incoming(value, self._cur_block)
        self._cur_block.emit(mir.Jmp(block))

    def _clause_is_bare(self, data: TryExceptBlockData, index: int) -> bool:
        return self._except_struct_type(data.except_types[index]) is None

    def _deliver_uncaught_call(self, exception: sval.StructType, payload_place: InterpVal, use_ret_payload: bool) -> None:
        """Deliver a call's error no clause catches into the function's error
        location and end the path: the callee's tag remapped (written once the
        tags are known, see ``_defer_error_code_write``), the payload already in
        place when the callee wrote straight into the location (no copy) or
        copied from the call's payload slot otherwise."""
        self._add_function_exception(exception)
        self._defer_error_code_write(exception)
        if not use_ret_payload and not exception.is_zst():
            value = self.load(self._union_variant_ptr(payload_place, exception))
            self.store(self._payload_variant_ptr(exception), value)
        self._end_error_path()

    def _deliver_uncaught_raise(self, slot: InterpVal, exception: sval.StructType) -> None:
        """Deliver a ``raise`` no clause catches into the function's error
        location and end the path: the exception's own slot is bound to the
        location's payload variant, so the already built value is written there
        directly (no copy), and the code tagged (once the tags are known, see
        ``_defer_error_code_write``)."""
        self._add_function_exception(exception)
        self._defer_error_code_write(exception)
        if exception.is_zst():
            self._commit_pending_slot(slot, exception)
        else:
            dest = self._payload_variant_ptr(exception)
            assert isinstance(dest, RuntimeVal)
            self._commit_pending_slot(slot, exception, ptr=dest.value)
        self._end_error_path()

    def _in_function_proper(self) -> bool:
        """Whether the instructions currently being executed are those
        of the function proper (whose ``return`` emits a typed return)
        rather than of an inlined plain function (whose ``return`` just
        yields a value to the caller): the frames stack holds the
        function proper at its bottom and one frame per inlined body
        above it, so the innermost body is the function proper exactly
        when it is the only frame."""
        return len(self._frames) == 1

    # -- running the machine -------------------------------------------------

    def _run_machine(self) -> PollResult:
        """The execution loop: walks the HIR of the executing frame (see
        ``_step``) until the run of the function proper ended (``DONE``) or
        a called spy function's specialization was just started and must be
        typed by a runner of its own (``SUSPEND``, see ``Analyser._run``).
        There is no recursion: the walk is linear over the HIR list of one
        frame, and every control state lives in the block stacks of the
        frames while the MIR being emitted grows as a basic-block graph
        rooted at the function's entry block."""
        while True:
            ret = self._step()
            if ret != PollResult.AGAIN:
                return ret

    def _pop_frame(self) -> bool:
        """Leave the innermost inlined body.  Its caller's continuation is the
        frame's exit block, which exists exactly when some path of the body
        reached it (see ``continuation``): the path now sitting at the frame's
        end is a falling one (the block is not closed yet) when it jumps there,
        and otherwise the body ended every path of itself elsewhere - a
        ``raise`` delivering the error out of the body - so the caller's code
        after the call is dead and must not be typed.  Returns whether the
        caller's continuation is live; ``_cut`` keeps unwinding the caller's own
        blocks when it is not, as it does for a dead path."""
        frame = self._frames.pop()
        if not self._cur_block.is_finished:
            # the path fell off the end of the body: it joins the caller
            self._cur_block.emit(mir.Jmp(frame.continuation()))
        exit_block = frame.exit_block
        if exit_block is None:
            return False
        self._cur_block = exit_block
        return True

    def _step(self) -> PollResult:
        """Execute one step of the machine: the instruction at the pc of
        the executing frame, advancing the pc - or the end of the
        frame's instruction list (its body fell off its end)."""
        frame = self._frames[-1]
        if frame.pc >= len(frame.insts):
            # the body fell off its end: every block is closed (see the
            # block transitions) - the run of the frame ended, and its
            # falling path joins the caller (see ``_pop_frame``)
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
        instruction changes the execution state: an ``If`` splits the
        current MIR block (a runtime ``if`` branches into its two branch
        blocks, a compile-time one keeps only the chosen branch, emitted
        straight into the current block), the ``Else``/``End`` markers
        close the branch being walked, a ``return`` cuts the current path
        (see ``_cut``); every other instruction only advances the state of
        the frame (its register table, and the MIR emitted so far)."""

        frame = self._frames[-1]
        regs = frame.regs
        match inst:
            case hir.Ret():
                if self._current_value_is_empty():
                    # the body has no value to return, so nothing ever wrote its
                    # result location: a ``return`` (whose store into that
                    # location is a no-op) is rejected here - the function proper
                    # and an inlined body alike
                    raise CompileError(
                        'a function that returns Never cannot return: it has no '
                        'value to return'
                    )
                if not self._in_function_proper():
                    # an inlined ``return`` delivers its value (already
                    # stored into the result location) and leaves the
                    # inlined body; the caller continues in the exit block
                    if not self._cur_block.is_finished:
                        self._cur_block.emit(mir.Jmp(frame.continuation()))
                    return self._cut()
                self._set_error_code_zero()
                if self.ret_sig is None:
                    # the return convention is not fixed yet (an unannotated
                    # return type, or an inferred exception set): the ``mir.Ret``
                    # is filled in by ``_finish_function`` once it is known
                    self._defer_return()
                    return self._cut()
                self._emit_function_return()
                return self._cut()
            case hir.Raise():
                # the exception was built into the slot the instruction names;
                # the path ends, dispatched to the clause that catches it or
                # returned from the function (see ``_raise``)
                return self._raise(self.operand(inst.value))
            case hir.ExceptBind():
                # the payload pointer of the clause currently being typed: the
                # value of its error-payload ``Phi`` (see ``_begin_except_clause``)
                frame0 = self._frames[-1]
                data0 = frame0.block_stack[-1].data
                assert isinstance(data0, TryExceptBlockData)
                exception = self._except_struct_type(data0.except_types[data0.region - 1])
                assert exception is not None
                phi = data0.payload_phi[data0.region - 1]
                if phi is None:
                    # a zero-sized exception has no payload: its bind is the unit
                    regs[inst] = ComptimeVal(sval.Undefined(sval.PointerType(exception, is_const=False)))
                else:
                    regs[inst] = RuntimeVal(phi, sval.PointerType(exception, is_const=False))
            case hir.AsBool():
                return self.as_bool(self.operand_arg(inst.value), inst)
            case hir.BinaryAssign():
                return self.binary_assign(inst.op, self.operand(inst.lhs), self.operand_arg(inst.rhs))
            case hir.If():
                self._exec_if(inst)
            case hir.Loop():
                self._exec_loop()
            case hir.Try():
                self._exec_try()
            case hir.Except():
                self._exec_except(inst)
            case hir.Break():
                return self._exec_break()
            case hir.Continue():
                return self._exec_continue()
            case hir.Else():
                self._exec_else()
            case hir.End():
                return self._exec_end()
            case hir.Load():
                regs[inst] = self.load(self.operand(inst.ptr))
            case hir.Alloca():
                regs[inst] = self.alloca(inst.inline, self._declared_type(inst.type))
            case hir.Store():
                self.store(self.operand(inst.ptr), self.operand(inst.value))
            case hir.StoreVoidRetloc():
                self.store_void_retloc()
            case hir.Tuple():
                regs[inst] = ComptimeTuple(tuple(self.operand_arg(v) for v in inst.values))
            case hir.InitTuple():
                self.init_tuple(self.operand(inst.tuple_ptr), inst.length)
            case hir.TuplePtrElement():
                regs[inst] = self.tuple_ptr_element(self.operand(inst.tuple_ptr), inst.index)
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
        passed its ``If`` instruction.  A compile-time condition keeps only
        the chosen branch - the other branch is dead, its instructions are
        skipped (never typed or emitted), and the chosen branch continues
        in the current block without any branch of its own; a runtime
        condition splits the current block into the two branch blocks (both
        survive at runtime, see ``_exec_runtime_if``).  The block is pushed
        on the frame's block stack, recording its entry (the pc of this
        ``If``) so that the matching ``Else``/``End`` markers can be found
        when the branches are walked off (see ``_scan_block``)."""
        frame = self._frames[-1]
        entry = frame.pc - 1
        cond = self.operand(inst.cond)
        if isinstance(cond, ComptimeVal):
            p_else, p_end = self._scan_block(entry)
            if cond.obj:
                # the then branch is chosen: it follows the ``If``
                frame.block_stack.append(
                    BlockFrame(entry, IfBlockData(True, p_else=p_else, p_end=p_end))
                )
                return
            # the else branch is chosen: skip the (dead) then branch
            if p_else is None:
                # no else branch either: the whole ``if`` is dead
                frame.pc = p_end + 1
                return
            frame.block_stack.append(
                BlockFrame(entry, IfBlockData(False, p_else=p_else, p_end=p_end))
            )
            frame.pc = p_else + 1
            return
        self._exec_runtime_if(cond, entry)

    def _exec_loop(self) -> None:
        """Open a ``loop`` block (``hir.Loop``): its body follows in the frame's
        flat instruction list, closed by the matching ``hir.End``.  The current
        block jumps to the new header, which the body's falling end and every
        ``continue`` jump back to (a back edge) - the loop is left only by a
        ``break`` (or a ``return``/``raise``), whose jump to the on-demand exit
        block is what tells a later ``_cut`` whether the code after the loop is
        reachable (see ``LoopBlockData``).

        A compile-time loop (``is_inline``) has no back edge and no header: the
        body is unrolled in place, one iteration after another (see
        ``_unroll_inline_loop``), so opening it only pushes the loop's state."""
        frame = self._frames[-1]
        entry = frame.pc - 1
        inst = frame.insts[entry]
        assert isinstance(inst, hir.Loop)
        p_else, p_end = self._scan_block(entry)
        assert p_else is None, 'a loop body has no else marker'
        if inst.is_inline:
            frame.block_stack.append(
                BlockFrame(entry, LoopBlockData(p_end=p_end, is_inline=True))
            )
            return
        header = mir.BasicBlock()
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(header))
        self._cur_block = header
        frame.block_stack.append(
            BlockFrame(entry, LoopBlockData(p_end=p_end, header_block=header))
        )

    def _find_loop(self) -> LoopBlockData:
        """The state of the innermost open ``loop`` of the executing frame - the
        target of a ``break``/``continue``.  An inlined body never crosses a
        loop (Python forbids ``break``/``continue`` in a nested function), so the
        open loops of the executing frame are all of them."""
        for bf in reversed(self._frames[-1].block_stack):
            data = bf.data
            if isinstance(data, LoopBlockData):
                return data
        raise CompileError('break/continue outside of a loop')

    def _unroll_inline_loop(self, frame: InlineFrame, bf: BlockFrame, data: LoopBlockData) -> None:
        """Start the next iteration of a compile-time loop: the current
        iteration is complete (its body fell off its end, or every path of it
        ended elsewhere), so route that falling end into the next body's entry -
        the block a ``continue`` of the iteration jumped to, when there was one -
        and rewind the walk to the head of the body to evaluate the condition
        again.  Unrolling is counted against ``max_loop_unroll``: a condition
        that never turns false (a non-compile-time one, or a loop that makes no
        compile-time progress) is reported instead of unrolling forever."""
        nxt = data.inline_next
        data.inline_next = None
        if nxt is not None:
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(nxt))
            self._cur_block = nxt
        self._loop_unrolls += 1
        if self._loop_unrolls > self.max_loop_unroll:
            raise CompileError(
                f'a compile-time loop unrolled more than {self.max_loop_unroll} '
                'times: its condition must be a compile-time value and the loop '
                'must make compile-time progress'
            )
        # rewind to the head of the body (the condition evaluation), keeping the
        # loop open so its remaining iterations keep unrolling
        frame.pc = bf.entry + 1

    def _exec_break(self) -> PollResult:
        """``hir.Break``: end the current path at the innermost loop's exit block,
        created on demand - the first ``break`` is what makes the code after the
        loop reachable - and unwind the frame's open blocks like any other ended
        path (see ``_cut``).  The loop's exit is exactly where a ``_cut`` that
        reaches the loop continues the walk.  A compile-time loop's ``break``
        works the same way: it leaves the whole unrolled sequence."""
        data = self._find_loop()
        exit_block = data.exit_block
        if exit_block is None:
            exit_block = data.exit_block = mir.BasicBlock()
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(exit_block))
        return self._cut()

    def _exec_continue(self) -> PollResult:
        """``hir.Continue``: end the current path back at the innermost loop's
        head, so the rest of the body and the ``while``'s else clause are skipped
        and the loop's condition is evaluated again; then unwind like any other
        ended path (see ``_cut``).  A runtime loop jumps to its header (the back
        edge); a compile-time loop jumps to the entry of its next unrolled body,
        created on demand and reached by this iteration's falling end too."""
        data = self._find_loop()
        if data.is_inline:
            nxt = data.inline_next
            if nxt is None:
                nxt = data.inline_next = mir.BasicBlock()
        else:
            assert data.header_block is not None
            nxt = data.header_block
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(nxt))
        return self._cut()

    def _exec_else(self) -> None:
        """The walk fell off the end of the then branch and reached the
        ``Else`` marker of the innermost open block."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        data = bf.data
        assert isinstance(data, IfBlockData)
        if data.chosen is None:
            # a runtime ``if``: its then-region fell off its end (it does
            # not return); the then block joins the continuation and the
            # else-region is typed next
            data.then_returns = False
            exit_block = data.exit_block
            assert exit_block is not None and data.else_block is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(exit_block))
            self._cur_block = data.else_block
            return
        # a compile-time ``if`` whose chosen branch is the then branch,
        # which fell off its end: the (unchosen) else branch is dead -
        # skip it and close the block
        assert data.chosen
        frame.block_stack.pop()
        frame.pc = data.p_end + 1

    def _exec_try(self) -> None:
        """Open a ``try`` block (``hir.Try``): reserve its continuation block
        and push it.  A clause's entry block is created lazily, when the first
        error reaches it (see ``_route_to_clause``), so a clause no error
        reaches is dead code (see ``_next_live_clause``)."""
        frame = self._frames[-1]
        entry = frame.pc - 1
        inst = frame.insts[entry]
        assert isinstance(inst, hir.Try)
        p_excepts, p_end = hir.scan_try(frame.insts, entry)
        data = TryExceptBlockData(
            insts=frame.insts,
            p_excepts=p_excepts, p_end=p_end, binds=inst.binds,
            except_types=inst.except_types,
            join=mir.BasicBlock(),
            clause_blocks=[None] * len(p_excepts),
            payload_phi=[None] * len(p_excepts),
            code_phi=[None] * len(p_excepts),
        )
        frame.block_stack.append(BlockFrame(entry, data))

    def _exec_except(self, inst: hir.Except) -> None:
        """The region just walked fell off its end and reached the ``Except``
        marker ``inst``: it joins the continuation, and the clause the marker
        opens is typed next (when some error reached it)."""
        frame = self._frames[-1]
        data = frame.block_stack[-1].data
        assert isinstance(data, TryExceptBlockData)
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(data.join))
        data.fell = True
        data.body_done = True
        index = self._next_live_clause(data, inst.index)
        if index is None:
            frame.pc = data.p_end
            return
        self._begin_except_clause(data, index)
        data.region = index + 1
        frame.pc = data.p_excepts[index] + 1

    def _next_live_clause(self, data: TryExceptBlockData, start: int) -> int | None:
        """The first clause at or after ``start`` that some error reached - its
        entry block exists - or None when there is none: the remaining clauses
        are dead code and are not typed."""
        for index in range(start, len(data.p_excepts)):
            if data.clause_blocks[index] is not None:
                return index
        return None

    def _begin_except_clause(self, data: TryExceptBlockData, index: int) -> None:
        """Start typing the except clause ``index``: its entry block - reached
        from every dispatch the clause caught, with the payload ``Phi`` they
        delivered - is entered, and the clause's ``as`` bind reads that phi (see
        ``hir.ExceptBind``).  A bare clause's union (and so its phis) is fixed
        here, once the whole body has been walked."""
        block = data.clause_blocks[index]
        assert block is not None
        if self._except_struct_type(data.except_types[index]) is None:
            self._finalize_bare_clause(data, index)
        self._cur_block = block

    def _finalize_bare_clause(self, data: TryExceptBlockData, index: int) -> None:
        """Fix a bare clause's error union once the try body has been walked:
        the dispatches into it, deferred while the union was still growing (see
        ``_route_to_clause``), now get their case blocks finished with the union,
        the error code and the payload pointer passed through the clause's phis.
        The clause uses neither for now, so nothing reads them."""
        records = [record for record in data.bare_pending if record[0] == index]
        if len(records) == 0:
            return
        block = data.clause_blocks[index]
        assert block is not None
        union = sval.UnionType(tuple(exception for _, _, _, exception in records))
        assert not union.is_zst(), 'a bare clause has a payload pointer'
        union_mir = union.to_mir_type(self._mir_cache)
        assert isinstance(union_mir, mir.UnionType)
        code_mir = mir.IntType(len(records).bit_length(), False)
        payload_phi: mir.Phi | None = None
        code_phi: mir.Phi | None = None
        saved = self._cur_block
        for tag, (_, case_block, incoming, exception) in enumerate(records, start=1):
            self._cur_block = case_block
            if exception.is_zst():
                # a zero-sized exception has no payload: a null is passed, the
                # clause does not read it
                bitcast = mir.NullValue(mir.PointerType(union_mir))
            else:
                bitcast = self._emit(mir.BitCast(
                    self._to_runtime(incoming), mir.PointerType(union_mir),
                ))
            case_block.emit(mir.Jmp(block))
            if payload_phi is None:
                payload_phi = mir.Phi([(bitcast, case_block)])
                block.emit(payload_phi)
            else:
                payload_phi.add_incoming(bitcast, case_block)
            code_value = mir.Int(tag, code_mir)
            if code_phi is None:
                code_phi = mir.Phi([(code_value, case_block)])
                block.emit(code_phi)
            else:
                code_phi.add_incoming(code_value, case_block)
        self._cur_block = saved
        data.payload_phi[index] = payload_phi
        data.code_phi[index] = code_phi

    def _except_struct_type(self, type_operand: hir.Value | None) -> sval.StructType | None:
        """The exception struct a clause's type expression denotes (None for a
        bare ``except:``): the operand is that expression, evaluated as a value
        before the ``Try`` opened (see ``hir.Try.except_types``)."""
        if type_operand is None:
            return None
        value: InterpVal = self.operand(type_operand)
        if isinstance(value, ComptimeVal) and isinstance(value.obj, sval.StructType):
            return value.obj
        raise CompileError(f'{value!r} is not an exception struct')

    def _exec_end(self) -> PollResult:
        """The walk fell off the end of a region and reached the ``End``
        marker of the innermost open block.  A runtime ``if`` whose
        currently-typed region fell off its end - the then-region of an
        ``if`` without an else, or the else-region - is complete: the
        falling branch jumps to the continuation, the block shared with the
        parallel branch (and, for an ``if`` without an else, with the false
        target of the branch).  Both branches falling through (a join) is
        fine: a variable a branch *assigns* lives in an enclosing block's
        slot - memory, since an assignment in a branch has to be visible
        after it - so the state crossing the join needs no phi.

        A ``loop`` body falling off its end jumps back to the loop header
        (the next iteration) and the code after the loop is typed next (see
        ``_cut``, which decides whether it is reachable); a compile-time
        loop's body falling off its end unrolls the next iteration in place
        (see ``_unroll_inline_loop``)."""
        frame = self._frames[-1]
        data = frame.block_stack[-1].data
        if isinstance(data, LoopBlockData):
            if data.is_inline:
                # the body fell off its end: unroll the next iteration, routing
                # this falling end into the block a ``continue`` jumped to (when
                # there was one)
                self._unroll_inline_loop(frame, frame.block_stack[-1], data)
                return PollResult.AGAIN
            # the loop body fell off its end: jump back to the header.  Whether
            # the code after the loop is live is decided by ``_cut`` from the
            # loop's exit block (created only by a ``break``)
            assert data.header_block is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(data.header_block))
            return self._cut()
        if isinstance(data, TryExceptBlockData):
            # the last except clause fell off its end: the try is complete
            assert data.join is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(data.join))
            self._cur_block = data.join
            frame.block_stack.pop()
            return PollResult.AGAIN
        assert isinstance(data, IfBlockData)
        if data.chosen is not None:
            # a compile-time ``if``: the chosen branch fell off its end
            frame.block_stack.pop()
            return PollResult.AGAIN
        exit_block = data.exit_block
        assert exit_block is not None
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(exit_block))
        self._cur_block = exit_block
        frame.block_stack.pop()
        return PollResult.AGAIN

    # -- runtime ``if`` regions --------------------------------------------

    def _exec_runtime_if(
        self, cond: InterpVal, entry: int
    ) -> None:
        if not isinstance(cond, RuntimeVal) or cond.type != sval.BoolType():
            raise CompileError('runtime if conditions must be boolean values')
        p_else, p_end = self._scan_block(entry)
        # the two branch blocks, and the block the code after the ``if``
        # continues in (the join of the falling branches - also the false
        # target when there is no else branch)
        then_block = mir.BasicBlock()
        exit_block = mir.BasicBlock()
        else_block = mir.BasicBlock() if p_else is not None else None
        self._cur_block.emit(
            mir.Br(cond.value, then_block, else_block if else_block is not None else exit_block)
        )
        frame = self._frames[-1]
        frame.block_stack.append(
            BlockFrame(entry, IfBlockData(
                p_else=p_else, p_end=p_end,
                then_block=then_block, else_block=else_block, exit_block=exit_block,
            ))
        )
        self._cur_block = then_block

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
        A ``loop`` whose body the path ended in is complete too: the walk
        continues in the loop's exit block - the code after the loop - when
        some ``break`` reaches it, and keeps unwinding (the code after the
        loop is dead) when none does; a compile-time loop a ``continue`` of
        the body leads out of unrolls its next iteration instead.  With no
        open block left, the run of the frame's body ended."""
        while True:
            frame = self._frames[-1]
            if not frame.block_stack:
                # the body run ended in a return: skip the dead code after it
                if len(self._frames) == 1:
                    # the walk of the function proper is over: park the pc at the
                    # end of its body, so that a run resumed after the cut (the
                    # analyser resumes a runner whose callee never returns) ends
                    # at once rather than walking on into the dead code
                    frame.pc = len(frame.insts)
                    return PollResult.DONE
                if self._pop_frame():
                    return PollResult.AGAIN
                # no path of the body reached the caller's continuation, so
                # the caller's path ended with the body's: keep unwinding its
                # own blocks (see ``_pop_frame``)
                continue
            bf = frame.block_stack[-1]
            data = bf.data
            if isinstance(data, TryExceptBlockData):
                # the region that ended (the try body, or a clause) returned or
                # raised: a later clause some error reached is typed next;
                # otherwise the try is complete, and the code after it is still
                # reachable when some earlier region fell through to the join
                data.body_done = True
                index = self._next_live_clause(data, data.region)
                if index is not None:
                    self._begin_except_clause(data, index)
                    data.region = index + 1
                    frame.pc = data.p_excepts[index] + 1
                    return PollResult.AGAIN
                frame.block_stack.pop()
                if data.fell:
                    self._cur_block = data.join
                    frame.pc = data.p_end + 1
                    return PollResult.AGAIN
                continue
            if isinstance(data, LoopBlockData):
                # the current path ended inside the loop body (a ``break``, a
                # ``continue``, a ``return`` or a ``raise``) and every enclosing
                # block of the body has been closed: the body is complete.  A
                # compile-time loop whose body a ``continue`` leads out of still
                # has its next iteration to unroll (whatever ended this one) - a
                # ``break`` jumps to the loop's exit block, so the code after the
                # loop is typed there; a loop no ``break`` can leave
                # (``while True:``) never has one, and the code after it is dead -
                # the cut keeps unwinding.
                if data.is_inline and data.inline_next is not None:
                    self._unroll_inline_loop(frame, bf, data)
                    return PollResult.AGAIN
                frame.block_stack.pop()
                exit_block = data.exit_block
                if exit_block is None:
                    continue
                self._cur_block = exit_block
                frame.pc = data.p_end + 1
                return PollResult.AGAIN
            assert isinstance(data, IfBlockData)
            if data.chosen is not None:
                # the path ran through the chosen branch of a compile-time
                # ``if`` and returned: dead code after it
                frame.block_stack.pop()
                continue
            exit_block = data.exit_block
            assert exit_block is not None
            if data.then_returns is None:
                # the path ended inside the then-region: it ends there (the
                # then block is already terminated); type the else-region
                # next, or - without one - continue after the ``if``
                data.then_returns = True
                if data.else_block is not None:
                    self._cur_block = data.else_block
                    assert data.p_else is not None
                    frame.pc = data.p_else + 1
                else:
                    frame.block_stack.pop()
                    if not self._cur_block.is_finished:
                        self._cur_block.emit(mir.Jmp(exit_block))
                    self._cur_block = exit_block
                    frame.pc = data.p_end + 1
                return PollResult.AGAIN
            # the path ended inside the else-region: the ``if`` is
            # complete; when every path returned the cut keeps unwinding
            then_returns = data.then_returns
            frame.block_stack.pop()
            if not then_returns:
                if not self._cur_block.is_finished:
                    self._cur_block.emit(mir.Jmp(exit_block))
                self._cur_block = exit_block
                frame.pc = data.p_end + 1
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
                return ComptimeVal(sval.as_value(obj, resolver=self._analyser._resolver))
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
                # ``hir.ResultLoc`` names the declared results: the value part
                # of the frame's result location (its error part is separate)
                return self._result_loc()
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
            case ComptimeAggregatePtr(aggregate_type, ptrs):
                # an aggregate is held by its fields: loading one loads every
                # field out of its own place (see ``ComptimeAggregatePtr``)
                return ComptimeAggregate(aggregate_type, tuple(self.load(p) for p in ptrs))
            case ComptimeVal(obj) if isinstance(obj, sval.ConstRef):
                # a reference to an immutable compile-time global behaves like
                # the value it refers to
                return ComptimeVal(obj.value)
            case RuntimeVal():
                if type.elem.classify() == sval.SpecialTypeKind.DST:
                    raise CompileError(
                        f'cannot load a value of the dynamically-sized type {type.elem}'
                    )
                return RuntimeVal(self._emit(mir.Load(ptr.value)), type.elem)
        raise CompileError('cannot load from a compile-time pointer')

    def store(self, ptr: InterpVal, value: InterpVal) -> None:
        """Write ``value`` into the slot or through the pointer ``ptr``.
        A store into a still uncommitted slot only records a store point
        (see :class:`PendingSlot`): the slot's final type is not known
        until it is committed, and the actual store is inserted then.

        A zero-sized type has no storage, so no store is *emitted* for one at
        runtime; a compile-time place (a box, the field of a compile-time
        aggregate) still records the value, since the type of such a place may
        have no runtime representation at all."""
        value = _shallow_normalize(value)
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

        ptr = _shallow_normalize(ptr)
        if isinstance(ptr, ComptimeResult):
            # a delivery into the function's result location: the value is either
            # the "no error" tag (``Success``) or an exception value, tagged and
            # written into the payload
            obj = value.obj if isinstance(value, ComptimeVal) else None
            if isinstance(obj, sval.Success):
                self.store(ptr.code, ComptimeVal(0))
                return
            type = _type_of(value)
            if not isinstance(type, sval.StructType):
                raise CompileError(f'cannot raise {type}: an exception must be a struct')
            self._raise_error(ptr, type, value)
            return

        if isinstance(ptr, PendingSlot) and ptr.committed is None:
            value_type = _type_of(value)
            if value_type is None:
                raise CompileError('cannot store a value that has no spy type')
            if ptr.inline_mode == InlineMode.FULL and _is_aggregate(value_type) and not _is_inline_val(value):
                # a *runtime* aggregate value into a compile-time variable: such a
                # variable holds its fields as places of their own (see
                # ``ComptimeAggregatePtr``), so the value is split into one place
                # (and one store) per field - each of which then lands where its
                # own kind says, in memory for a runtime field and in a box
                # otherwise
                recorded = self._pending_aggregate(ptr)
                if recorded is None:
                    self._record_pending_action(
                        ptr,
                        _PendingAggregate(
                            value_type, self._split_runtime_aggregate(value, value_type),
                        ),
                    )
                else:
                    # a second value into the same storage (the branches of an
                    # ``if`` expression): its fields are places already, which
                    # the field values are written into
                    for index, field_value in enumerate(
                        self._runtime_aggregate_field_values(value, value_type)
                    ):
                        self.store(recorded.places[index], field_value)
                return
            self._record_pending_action(
                ptr,
                _PendingStore(
                    type=value_type,
                    is_inline=_is_inline_val(value),
                    value=value,
                ),
            )
            return

        ptr_type = _type_of(ptr)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f"cannot store to a {ptr_type} value")
        elem = ptr_type.elem
        if isinstance(elem, sval.UnionType) and _is_union_unit(value):
            # a union value carries no storage - it says nothing but which union
            # it belongs to - so a storage destination has nothing to write
            return
        if isinstance(elem, sval.OptionType):
            # an option (and the ``T``/``Null`` a store delivers) is written
            # through the representation of ``elem`` (see ``_write_option``)
            match ptr:
                case ComptimeBox():
                    obj = _to_comptime(value)
                    if obj is None:
                        ptr.value = self._coerce_option_value(value, elem)
                    elif isinstance(obj, sval.Value):
                        # an already spy-typed value (the null value, or a value
                        # of the child type) carries its own type
                        ptr.value = ComptimeVal(obj)
                    else:
                        ptr.value = ComptimeVal(sval.coerce_const(obj, elem))
                case RuntimeVal():
                    self._write_option(ptr.value, value, elem)
                case _:
                    raise CompileError('cannot store through a compile-time pointer')
            return
        aggregate = _as_aggregate(value)
        if aggregate is not None and _is_aggregate(elem):
            # a whole aggregate is written place by place, into the place of each
            # field (or element) - memory storage or a compile-time aggregate -
            # which is how a copy into an existing storage works (see
            # ``ComptimeAggregate``).  This comes before any question about the
            # type's runtime representation: an aggregate's places exist whether
            # or not the aggregate has a mirror of its own
            place_types = _aggregate_place_types(elem)
            if len(aggregate.values) != len(place_types):
                raise CompileError(
                    f'cannot store an aggregate of {len(aggregate.values)} place(s) '
                    f'into {elem}'
                )
            for index, place_value in enumerate(aggregate.values):
                self.store(self.field_index_addr(ptr, _index_value(index)), place_value)
            return
        if isinstance(ptr, ComptimeAggregatePtr) and _is_aggregate(elem) and isinstance(value, RuntimeVal):
            # a *runtime* aggregate value into a compile-time aggregate that
            # exists already (a declared ``Comptime[T]``, or one assigned
            # before): every field already is a place, which the field values are
            # written into
            for index, field_value in enumerate(self._runtime_aggregate_field_values(value, elem)):
                self.store(self.field_index_addr(ptr, _index_value(index)), field_value)
            return
        if isinstance(ptr, ComptimeVal) and isinstance(ptr.obj, sval.Undefined):
            # a compile-time pointer with no storage at all: the address of a
            # zero-sized field or element (see ``field_index_addr``), or of a
            # zero-sized exception's variant (see ``_union_variant_ptr``).
            # Every value of the type it points at is the type's unit value, so
            # a store into it records nothing
            return
        unit = elem.get_unit_value()
        match ptr:
            case ComptimeBox():
                if _is_aggregate(ptr.type):
                    # an aggregate is held by its own places, never by a box (see
                    # ``ComptimeAggregatePtr``)
                    raise CompileError(f'a compile-time box cannot hold the aggregate {ptr.type}')
                if unit is not None:
                    # a zero-sized type has one value - its unit value - which
                    # is what a place of it holds whatever is stored into it (no
                    # coercion: the value stored need not even be of the type)
                    ptr.value = ComptimeVal(unit)
                else:
                    ptr.value = self._coerce(value, elem)
            case RuntimeVal():
                if unit is not None:
                    # a zero-sized type has no storage: nothing is emitted at
                    # runtime (a compile-time place still records its value)
                    return
                self._emit(mir.Store(ptr.value, self._to_runtime(self._coerce(value, elem))))
            case _:
                raise CompileError('cannot store through a compile-time pointer')

    def _runtime_aggregate_field_values(self, value: InterpVal, type: sval.Type) -> list[InterpVal]:
        """The value of every field (or element) of the *runtime* aggregate
        ``value``, read out of a copy of it materialized in memory: an aggregate
        value has no address of its own, so a copy is what its fields are read
        from (the same copy a whole-value store of one makes anyway)."""
        src = self.alloca(InlineMode.NONE)
        self._commit_pending_slot(src, type)
        self.store(src, value)
        src_ptr = _shallow_normalize(src)
        return [
            self.load(self.field_index_addr(src_ptr, _index_value(index)))
            for index in range(len(_aggregate_place_types(type)))
        ]

    def _split_runtime_aggregate(self, value: InterpVal, type: sval.Type) -> tuple[InterpVal, ...]:
        """One fresh place per field (or element) of the *runtime* aggregate
        ``value``, written with the field read out of it - the places a
        compile-time aggregate holds (see ``ComptimeAggregatePtr``).  A nested
        aggregate field is split the same way, recursively (its own place is a
        ``FULL`` slot holding a runtime aggregate, see ``store``)."""
        places: list[InterpVal] = []
        for field_value in self._runtime_aggregate_field_values(value, type):
            place = self.alloca(InlineMode.FULL)
            self.store(place, field_value)
            places.append(place)
        return tuple(places)

    def store_void_retloc(self) -> None:
        """Deliver the void unit value into the result location (see
        ``hir.StoreVoidRetloc``): the value a body that falls off its end
        returns.  A function with no value to return cannot fall off its end
        either - that would be a returning path all the same - so a body of the
        empty type is rejected here (its store into the result location would
        otherwise be a silent no-op)."""
        if self._current_value_is_empty():
            raise CompileError(
                'a function that returns Never cannot fall off its end: it has no '
                'value to return'
            )
        location = self._result_loc()
        if isinstance(location, ComptimeTuple):
            raise CompileError('a function that returns several values must return them')
        self.store(location, ComptimeVal(sval.Void()))

    # -- options ---------------------------------------------------------------

    def _write_option(self, dst: mir.Value, value: InterpVal, option: sval.OptionType) -> None:
        """Write the ``Option[T]`` value ``value`` into the memory ``dst`` - a
        MIR pointer to the option's representation (see
        ``sval.OptionType.to_mir_type``).  A present value is the child's own
        representation (coerced to the child), and the absent one the null
        representation: a ``bool`` for a zero-sized child, the tagging pointer
        nulled (``_write_option_null``) for a child that has one, and the tag
        ``false`` for the rest."""
        child = option.child
        if isinstance(value, RuntimeVal) and _type_of(value) == option:
            # the value already is an option of this type (a parameter of one,
            # a load of one): its representation is written as a whole
            self._emit(mir.Store(dst, self._to_runtime(value)))
            return
        null = _is_null(value)
        if child.is_zst():
            # a zero-sized child carries no value: only whether there is one
            self._emit(mir.Store(dst, mir.BoolValue(not null)))
            return
        if sval.find_first_pointer_type_pos(child) is not None:
            if null:
                self._write_option_null(dst, option)
            else:
                self._emit(mir.Store(dst, self._to_runtime(self._coerce(value, child))))
            return
        # a struct of the tag and the value: the tag says whether there is one
        tag = self._emit(mir.Gep(dst, 0))
        self._emit(mir.Store(tag, mir.BoolValue(not null)))
        if not null:
            payload = self._emit(mir.Gep(dst, 1))
            self._emit(mir.Store(payload, self._to_runtime(self._coerce(value, child))))

    def _option_tag_addr(
        self, ptr: mir.Value, option: sval.OptionType
    ) -> tuple[mir.Value, sval.PointerType] | None:
        """The address of the pointer that tags ``option``, and its type - the
        pointer whose nullness makes the option absent - or ``None`` when the
        option's representation carries a ``bool`` tag instead (see
        ``sval.find_first_pointer_type_pos``).

        The position is the one the spy type names, mapped onto the mirror: the
        field positions of a struct are its *mirror* positions (a struct of one
        stored field *is* that field, see ``mirror_is_a_field``), an array
        element is its own position, and stepping into an option costs no
        position (an option that still has a free pointer shares the
        representation of its child)."""
        path = sval.find_first_pointer_type_pos(option.child)
        if path is None:
            return None
        node: sval.Type = option.child
        cur = ptr
        for index in path:
            if isinstance(node, sval.OptionType):
                node = node.child
                continue
            if isinstance(node, sval.StructType):
                if not node.mirror_is_a_field(self._mir_cache):
                    mir_index = node.get_field_mir_indices(self._mir_cache)[index]
                    assert mir_index is not None, 'a field holding a pointer has a mirror position'
                    cur = self._emit(mir.Gep(cur, mir_index))
                node = node.fields().get_by_id(index).type
            elif isinstance(node, sval.ArrayType):
                cur = self._emit(mir.Gep(cur, index))
                node = node.elem
            else:
                raise CompileError(f'cannot take the tag address of {option}')
        if not isinstance(node, sval.PointerType):
            raise CompileError(f'cannot take the tag address of {option}')
        return cur, node

    def _write_option_null(self, dst: mir.Value, option: sval.OptionType) -> None:
        """Write the absent value of ``option`` into the memory ``dst``: null
        the pointer that tags it."""
        tag = self._option_tag_addr(dst, option)
        assert tag is not None, 'the option has a pointer tag'
        tag_ptr, tag_type = tag
        mir_type = tag_type.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.PointerType)
        self._emit(mir.Store(tag_ptr, mir.NullValue(mir_type)))

    def _coerce_option_value(self, ev: InterpVal, option: sval.OptionType) -> InterpVal:
        """Materialize ``ev`` as a value of the option type ``option``: a value
        that already is one passes through, the null value becomes the absent
        one and anything else a present value of the child type.

        The result is a value of the option: for a child that has a free pointer
        (whose representation the option shares) the coerced child itself, a
        ``bool`` or a null pointer for the scalar representations, and the
        ``mir.Load`` of a fresh temporary for the struct representation (a tag
        and the value have no constant form of their own)."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, RuntimeVal) and _type_of(ev) == option:
            return ev
        child = option.child
        null = _is_null(ev)
        if child.is_zst():
            return RuntimeVal(mir.BoolValue(not null), option)
        if sval.find_first_pointer_type_pos(child) is not None:
            if not null:
                return self._coerce(ev, child)
            mir_type = option.to_mir_type(self._mir_cache)
            assert mir_type is not None and not option.is_zst()
            if isinstance(mir_type, mir.PointerType):
                # the option itself is the tagging pointer: a null pointer is
                # the absent value
                return RuntimeVal(mir.NullValue(mir_type), option)
        # build the representation in a temporary and load it back
        mir_type = option.to_mir_type(self._mir_cache)
        assert mir_type is not None and not option.is_zst()
        alloca = self._emit(mir.Alloca(mir_type))
        self._write_option(alloca, ev, option)
        return RuntimeVal(self._emit(mir.Load(alloca)), option)

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
        if is_aggregate_init and isinstance(ptr, ComptimeResult):
            # a field of the exception being raised: its address is decided by
            # the ``FinishStruct`` that closes the construction, in the error
            # location's payload (see ``finish_struct``)
            return self.alloca(InlineMode.NONE)
        if is_aggregate_init and isinstance(ptr, PendingSlot) and ptr.committed is None:
            # the storage of the aggregate being built has no address yet: the
            # field gets a pending place of its own (see ``finish_struct``) -
            # the one a previous construction of the storage recorded, when
            # there is one (the branches of an ``if`` expression)
            recorded = self._pending_aggregate(ptr)
            if recorded is not None and isinstance(recorded.type, sval.StructType):
                index = recorded.type.field_index(name)
                if index is not None:
                    return recorded.places[index]
            return self.alloca(ptr.inline_mode)
        ptr = self._auto_deref(ptr)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot take field address of {ptr}")
        container_type = type.elem
        if is_aggregate_init:
            # the construction may build the child of nested options: its fields
            # are taken in the payload of each of them
            while self._is_option_construction(container_type):
                assert isinstance(container_type, sval.OptionType)
                ptr = self._option_payload_ptr(ptr, container_type)
                type = _type_of(ptr)
                assert isinstance(type, sval.PointerType)
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
        if is_aggregate_init and isinstance(ptr, ComptimeResult):
            # a keyword field of the exception being raised (see
            # ``field_index_addr``)
            return self.alloca(InlineMode.NONE)
        if is_aggregate_init and isinstance(ptr, PendingSlot) and ptr.committed is None:
            # the aggregate's storage has no address yet: the field gets a
            # pending place of its own - the one a previous construction of the
            # same storage recorded, when there is one (the branches of an
            # ``if`` expression build into one storage), see ``finish_struct``
            recorded = self._pending_aggregate(ptr)
            if recorded is not None:
                return recorded.places[_comptime_index(index)]
            return self.alloca(ptr.inline_mode)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f'cannot take a field or element address of {ptr}')
        container_type = type.elem
        is_const = type.is_const
        if is_aggregate_init:
            # the construction may build the child of nested options: the field
            # or element address is taken in the payload of each of them, which
            # marks the option present when its tag is a ``bool``
            while self._is_option_construction(container_type):
                assert isinstance(container_type, sval.OptionType)
                ptr = self._option_payload_ptr(ptr, container_type)
                type = _type_of(ptr)
                assert isinstance(type, sval.PointerType)
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
                case ComptimeAggregatePtr(_, ptrs):
                    # the fields of a compile-time aggregate are their own
                    # places, in declaration order (see ``ComptimeAggregatePtr``)
                    return ptrs[index_int]
                case ComptimeVal():
                    raise CompileError(
                        'cannot take the address of a field of a compile-time value'
                    )
                case RuntimeVal():
                    if not container_type.mirror_is_a_field(self._mir_cache):
                        mir_index = container_type.get_field_mir_indices(self._mir_cache)[index_int]
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
            match ptr:
                case ComptimeAggregatePtr(_, ptrs):
                    # the elements of a compile-time aggregate are their own
                    # places, in element order (see ``ComptimeAggregatePtr``)
                    return ptrs[_comptime_index(index)]
                case RuntimeVal():
                    return RuntimeVal(
                        self._emit(mir.Gep(ptr.value, _mir_index(index)), at), elem_ptr_type
                    )
            raise CompileError(
                f'cannot take the address of an element of {container_type}: '
                f'the array is a compile-time value'
            )

        raise CompileError(f'cannot take a field or element address of {ptr}')

    def _emit(self, inst: mir.Inst, at: mir.Insertion | None = None) -> mir.Value:
        """Append one instruction to the block currently being built, or,
        while a pending action is being delivered, to the insertion block
        that action reserved (``self._insertion``) - or to an explicit
        insertion (``at``, a slot's storage position, see
        ``PendingSlot.insertion``).  Instructions emitted into an insertion
        land at the position it sits at once the body is normalized."""
        target = at if at is not None else self._insertion
        if target is not None:
            target.insts.append(inst)
        else:
            self._cur_block.emit(inst)
        return inst

    def alloca(self, inline: InlineMode = InlineMode.NONE, declared: sval.Type | None = None) -> PendingSlot:
        """Reserve a fresh slot.  ``inline`` is how much of the value may be
        kept inline - nothing, anything but an aggregate, or anything (see
        ``InlineMode``); a ``declared`` type (an annotated variable, see
        ``_declared_type``) fixes the slot's storage right away - a
        compile-time value for a ``Comptime`` variable or a zero-sized type,
        memory (a :class:`RuntimeVal`) for anything else (see ``hir.Alloca``)."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        slot = PendingSlot(insertion, inline)
        if declared is not None:
            if inline != InlineMode.NONE and declared.get_unit_value() is None:
                if _is_aggregate(declared):
                    # a compile-time variable of an aggregate type: the aggregate
                    # is built in place, its fields (or elements) being their own
                    # places
                    slot.committed = self.init_inline_aggregate(declared)
                else:
                    # a compile-time variable of a declared type: a box the
                    # value it is assigned is written into
                    slot.committed = ComptimeBox(declared, ComptimeVal(sval.Undefined(declared)))
            else:
                self._commit_pending_slot(slot, declared)
        return slot

    def _declared_type(self, node: hir.Value | None) -> sval.Type | None:
        """The spy type a variable's annotation declares (the ``type`` operand
        of a typed ``hir.Alloca``), or None when the variable declares none.
        The annotation is a compile-time type value, into which the executing
        frame's type parameters are substituted - a local annotation may name
        them, exactly like a parameter annotation (see ``astgen``)."""
        if node is None:
            return None
        obj = _to_comptime(_shallow_normalize(self.operand(node)))
        if isinstance(obj, sval.ConstRef):
            obj = obj.value
        if not isinstance(obj, sval.Type):
            raise CompileError(f'{obj!r} is not a type')
        reps: dict[sval.TypeVar, sval.AnyValue] = {
            tv: value.obj
            for tv, value in self._frames[-1].generic_var_values.items()
            if isinstance(value, ComptimeVal)
        }
        if len(reps) > 0:
            obj = sval.replace_type_vars_type(obj, reps)
        return obj

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
        if isinstance(target, sval.OptionType):
            # a ``T``/``Null`` value is coerced through the option's
            # representation (see ``_coerce_option_value``)
            return self._coerce_option_value(ev, target)
        if isinstance(target, sval.TupleType):
            # a tuple has no runtime representation to convert to: the
            # compile-time tuple itself is what a location of the type holds
            if not isinstance(ev, ComptimeTuple):
                raise CompileError(f'cannot materialize a {target} from {ev!r}')
            return ev
        match ev:
            case ComptimeVal(obj) if isinstance(obj, sval.AggregateValue):
                # an aggregate held as one compile-time object (see
                # ``_as_aggregate``): it has no runtime representation of its
                # own, so it is materialized like an interpreter aggregate value
                aggregate = _as_aggregate(ev)
                assert aggregate is not None
                if not _is_aggregate(target):
                    raise CompileError(f'cannot materialize a {target} from an aggregate')
                return self.load(self._materialize_aggregate(aggregate, target))
            case ComptimeVal(obj):
                return ComptimeVal(sval.coerce_const(obj, target))
            case RuntimeVal(value, type):
                return RuntimeVal(self._convert(value, type, target), target)
            case ComptimeAggregatePtr():
                # a compile-time aggregate as a value of a pointer type: it *is*
                # a pointer already (see ``_type_of``) - its fields are their own
                # places - so nothing is converted here.  Becoming an address of
                # real memory happens only where a MIR value is actually needed
                # (see ``_to_runtime``)
                if not isinstance(target, sval.PointerType):
                    raise CompileError(
                        f'cannot materialize a {target} from a compile-time aggregate'
                    )
                return ev
            case ComptimeBox():
                # a compile-time box already is a value of a pointer type (see
                # ``_type_of``), and a pointer needs no conversion, just like a
                # runtime one (see ``_convert``)
                if not isinstance(target, sval.PointerType):
                    raise CompileError(
                        f'cannot materialize a {target} from a compile-time box'
                    )
                return ev
            case ComptimeAggregate():
                # an aggregate has no runtime representation to convert to: it
                # is materialized into a temporary and read back as a runtime
                # value
                if not _is_aggregate(target):
                    raise CompileError(f'cannot materialize a {target} from an aggregate')
                return self.load(self._materialize_aggregate(ev, target))
            case _:
                raise CompileError('cannot materialize this value')

    def _materialize_aggregate(
        self, value: ComptimeAggregate, aggregate_type: sval.Type
    ) -> InterpVal:
        """The address of fresh memory the aggregate ``value`` is written into,
        place by place: an aggregate has no runtime representation of its own,
        so a runtime location of its type is built by copying it in."""
        slot = self.alloca(InlineMode.NONE)
        self._commit_pending_slot(slot, aggregate_type)
        self.store(slot, value)
        return _shallow_normalize(slot)

    def _to_runtime(self, ev: InterpVal) -> mir.Value:
        """Materialize a value as a typed MIR value: a runtime value yields its
        MIR object, a compile-time value a constant built from it
        (``_sval_to_runtime``), and a committed slot the value it was
        materialized into.  A compile-time aggregate has no MIR representation
        of its own either: the address of fresh memory it is copied into is
        what a pointer to it delivers (see ``_materialize_aggregate``; a value
        of an aggregate type goes through ``_coerce``, which reads it back).
        An uncommitted slot or a compile-time box is rejected."""
        ev = _shallow_normalize(ev)
        match ev:
            case RuntimeVal():
                return ev.value
            case ComptimeVal():
                return _sval_to_runtime(ev.obj)
            case ComptimeAggregatePtr(aggregate_type, _):
                aggregate = self.load(ev)
                if not isinstance(aggregate, ComptimeAggregate):
                    # a zero-sized aggregate: its value *is* its unit value, which
                    # has no runtime representation at all
                    raise CompileError(
                        f'a value of the zero-sized {aggregate_type} has no runtime value'
                    )
                return self._to_runtime(self._materialize_aggregate(aggregate, aggregate_type))
            case ComptimeBox():
                raise CompileError('cannot use a compile-time box as a runtime value')
            case PendingSlot():
                raise CompileError('cannot use an uncommitted slot as a runtime value')
        raise CompileError('cannot return this value')

    def _convert(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Value:
        converted = _convert_inst(value, from_type, to_type, self._mir_cache)
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
            mir_type = type.to_mir_type(self._mir_cache)
            assert mir_type is not None and not type.is_zst()
            value = self._emit(
                mir.Arith(op, signed, self._to_runtime(lc), self._to_runtime(rc), mir_type)
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
                mir.Cmp(op, signed, kind, self._to_runtime(lc), self._to_runtime(rc))
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
                mir.Cmp('==', False, 'int', self._to_runtime(coerced), mir.BoolValue(False))
            )
            self.store(ret, RuntimeVal(value, sval.BoolType()))
            return PollResult.AGAIN
        if op == '-':
            mir_type = type.to_mir_type(self._mir_cache)
            assert mir_type is not None and not type.is_zst()
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
                mir.Arith('-', False, zero, self._to_runtime(coerced), mir_type)
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
        """Materialize a pending slot.  It becomes a :class:`ComptimeBox` when
        it may inline values and every action may be inlined, a
        :class:`ComptimeAggregatePtr` when the value is an aggregate (which a box
        never holds), or - when its type is zero-sized - the unit value it only
        records; with an explicit result pointer (``ptr``), or otherwise, it
        becomes a :class:`RuntimeVal` pointer to freshly allocated memory.  The
        type is the pairwise ``resolve_peer_type`` of the action types (or the
        given one).  The recorded actions are delivered through
        ``_exec_pending_actions``, which fills in the instructions that must be
        spliced at their original positions."""
        if not isinstance(val, PendingSlot):
            raise CompileError('can only commit a pending slot')
        if val.committed is not None:
            return
        if type is None:
            type = val.committed_type()

        if ptr is not None:
            self._bind_slot(val, ptr, type)
            return

        if len(val.stores) > 0 and val.is_inline(type):
            if _is_aggregate(type):
                # a compile-time aggregate: its fields (or elements) are their
                # own places (see ``init_inline_aggregate``)
                val.committed = self._inline_aggregate_of(val, type)
            else:
                # a single value in a compile-time box
                val.committed = ComptimeBox(type, ComptimeVal(sval.Undefined(type)))
            self._exec_pending_actions(val, type)
            return
        unit = type.get_unit_value()
        if unit is not None:
            if _is_aggregate(type):
                # a zero-sized aggregate holds its unit value, which a box never
                # carries: it is held by its own places like any other aggregate
                val.committed = self._inline_aggregate_of(val, type)
                self._exec_pending_actions(val, type)
                return
            val.committed = ComptimeBox(type, ComptimeVal(unit))
            return
        if type.classify() == sval.SpecialTypeKind.DST:
            # a dynamically-sized type has no size to allocate and no mirror a
            # value of it could live in (only a pointer to one is a value)
            raise CompileError(
                f'cannot allocate a runtime location of the dynamically-sized type {type}'
            )
        mir_type = type.to_mir_type(self._mir_cache)
        if mir_type is None or type.is_zst():
            raise _no_runtime_type(type)
        alloca = mir.Alloca(mir_type)
        val.insertion.insts.append(alloca)
        self._bind_slot(val, alloca, type)

    def _inline_aggregate_of(self, slot: PendingSlot, type: sval.Type) -> ComptimeAggregatePtr:
        """The aggregate storage a slot commits to: the places its construction
        recorded, or fresh ones for a slot that only took whole-value stores
        (see ``init_inline_aggregate``)."""
        recorded = self._pending_aggregate(slot)
        if recorded is None:
            return self.init_inline_aggregate(type)
        return ComptimeAggregatePtr(type, recorded.places)

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
            saved = self._insertion
            self._insertion = action.insertion
            self._exec_pending_action(action.data, slot.committed, type)
            self._insertion = saved

    def _exec_pending_action(self, action: _PendingActionData, ptr: InterpVal, type: sval.Type) -> None:
        match action:
            case _PendingStore():
                self.store(ptr, action.value)
            case _PendingTuple():
                # a tuple location: the element places are committed together
                # with the slot (no ``hir.CommitSlot`` names them), and the slot
                # holds the tuple of places itself (see ``init_tuple``)
                self._commit_tuple_places(ComptimeTuple(action.places))
                if not isinstance(ptr, ComptimeBox):
                    raise CompileError('cannot deliver a tuple into storage')
                ptr.value = ComptimeTuple(action.places)
            case _PendingPtrConvertion():
                input_ptr = _shallow_normalize(action.input)
                if isinstance(input_ptr, ComptimeResult):
                    # a call returning an aggregate is raised: the payload pointer
                    # is handed over (and the error code tagged) at the commit
                    action.output.value = self._to_runtime(self._convert_result_ptr(action.input, action.type))
                else:
                    if not (isinstance(input_ptr, RuntimeVal) and isinstance(input_ptr.type, sval.PointerType)):
                        raise CompileError('cannot convert a compile-time pointer')
                    action.output.value = self._to_runtime(self._convert_result_ptr(input_ptr, action.type))
            case _PendingAggregate():
                # a compile-time aggregate: every field (or element) place is
                # committed with the slot (no ``hir.CommitSlot`` names them) and
                # the slot holds the aggregate itself; a slot materialized into
                # memory instead copies the values in (see
                # ``init_inline_aggregate``)
                place_types = _aggregate_place_types(action.type)
                aggregate = isinstance(ptr, ComptimeAggregatePtr)
                for index, place in enumerate(action.places):
                    if isinstance(place, PendingSlot) and place.committed is None:
                        self._commit_pending_slot(place, place_types[index])
                    if aggregate:
                        continue
                    self.store(
                        self.field_index_addr(ptr, _index_value(index)),
                        self.load(place),
                    )
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
        wider type (an ``Option[T]``, which the null value peers to) widens
        with it, and the unit value is coerced to the final type at the
        slot's commit."""
        unit = type.get_unit_value()
        if unit is not None:
            self.store(slot, ComptimeVal(unit))
            return ComptimeVal(sval.Undefined(sval.PointerType(type, is_const=False)))
        mir_type = type.to_mir_type(self._mir_cache)
        if mir_type is None or type.is_zst():
            raise _no_runtime_type(type)
        output = mir.Insertion([], None, mir.PointerType(mir_type))
        self._emit(output)
        slot.stores.append(_PendingAction(output, _PendingPtrConvertion(type, slot, output)))
        return RuntimeVal(output, sval.PointerType(type, is_const=False))

    def _convert_result_ptr(self, ptr: InterpVal, to_type: sval.Type) -> InterpVal:
        """The pointer a result-location operation writes through, converted
        to the type it delivers - the result type of a call, or the struct/
        array type a construction builds its fields/elements into.  A location
        whose type is an ``Option[T]`` and a delivery of a ``T`` are what makes
        the two differ: the delivery writes through the place the value of a
        present option lives in (``_option_payload_ptr``), which marks the
        option present.  A location that is the function's error location and
        a delivery of an exception ``E`` are the other case: the error is
        tagged and written into the payload (``_error_payload_ptr``)."""
        ptr = _shallow_normalize(ptr)
        if isinstance(ptr, ComptimeResult):
            if not isinstance(to_type, sval.StructType):
                raise CompileError(f'cannot raise {to_type}: an exception must be a struct')
            self._add_function_exception(to_type)
            self._defer_error_code_write(to_type)
            return self._error_payload_ptr(ptr, to_type)
        ptr_type = _type_of(ptr)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'cannot use {ptr!r} as a result location')
        from_type = ptr_type.elem
        if from_type == to_type:
            return ptr
        if isinstance(from_type, sval.UnionType) and isinstance(to_type, sval.StructType):
            # an exception written through the address of an error payload: the
            # union's storage reinterpreted as the variant
            if to_type not in from_type.types:
                raise CompileError(f'{to_type} is not a variant of {from_type}')
            return self._union_variant_ptr(ptr, to_type)
        if isinstance(from_type, sval.UnionType) and isinstance(to_type, sval.UnionType):
            # one union's storage reinterpreted as another's: a subset union's
            # pointer is a pointer to the superset union as well (every variant
            # lives at offset 0, and the superset's storage holds it)
            if not (_union_contains(from_type, to_type) or _union_contains(to_type, from_type)):
                raise CompileError(f'cannot convert a {from_type} pointer to {to_type}')
            to_mir = to_type.to_mir_type(self._mir_cache)
            assert to_mir is not None and not to_type.is_zst()
            bitcast = self._emit(mir.BitCast(self._to_runtime(ptr), mir.PointerType(to_mir)))
            return RuntimeVal(bitcast, sval.PointerType(to_type, is_const=False))
        if isinstance(from_type, sval.OptionType):
            # the delivery goes into the payload of the option - and, when the
            # option's child is an option itself, through each of its layers
            # (a ``T`` converts to ``Option[T]``, and so on outward)
            inner = self._option_payload_ptr(ptr, from_type)
            return self._convert_result_ptr(inner, to_type)
        raise CompileError(
            f'cannot deliver a {to_type} into a location of type {from_type}'
        )

    def _is_option_construction(self, container_type: sval.Type) -> bool:
        """Whether ``container_type`` is an option whose child a construction
        builds in place (the child has storage to address: it is not
        zero-sized)."""
        return (
            isinstance(container_type, sval.OptionType)
            and not container_type.child.is_zst()
        )

    def _option_payload_ptr(self, ptr: InterpVal, option: sval.OptionType) -> InterpVal:
        """The place the value of a present ``Option[T]`` lives in - what a
        delivery of a ``T`` into the option writes through.  The delivery also
        marks the option present: for a child that still has a free pointer the
        option *is* the value (that pointer is the tag, and the value itself
        sets it), and otherwise the tag of the struct representation is set
        here."""
        child = option.child
        if child.is_zst():
            raise CompileError(f'cannot take the address of the value of {option}')
        src = self._to_runtime(ptr)
        if sval.find_first_pointer_type_pos(child) is not None:
            return RuntimeVal(src, sval.PointerType(child, is_const=False))
        tag = self._emit(mir.Gep(src, 0))
        self._emit(mir.Store(tag, mir.BoolValue(True)))
        payload = self._emit(mir.Gep(src, 1))
        return RuntimeVal(payload, sval.PointerType(child, is_const=False))

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

    def _pending_aggregate(self, slot: PendingSlot) -> _PendingAggregate | None:
        # the aggregate initialization a slot recorded, when one did: the places
        # an inline aggregate construction builds into (see
        # ``init_inline_aggregate``)
        for action in slot.stores:
            if isinstance(action.data, _PendingAggregate):
                return action.data
        return None

    def init_inline_aggregate(self, type: sval.Type) -> ComptimeAggregatePtr:
        """Fresh compile-time storage for an aggregate: every field (or array
        element) gets a place of its own - a box holding the type's unit value
        for a zero-sized one, an aggregate pointer of its own for a nested
        aggregate (a box never holds an aggregate), an undefined box otherwise -
        and the places, in declaration (or element) order, make up the result.
        The result is itself a pointer (``PointerType(type)``):
        ``HirRunner.field_index_addr`` takes a place out of it, ``store`` writes
        a place through it and ``load`` reads the whole aggregate (see
        ``ComptimeAggregatePtr`` and ``ComptimeAggregate``)."""
        places: list[InterpVal] = []
        for place_type in _aggregate_place_types(type):
            if _is_aggregate(place_type):
                places.append(self.init_inline_aggregate(place_type))
                continue
            unit = place_type.get_unit_value()
            if unit is not None:
                places.append(ComptimeBox(place_type, ComptimeVal(unit)))
            else:
                places.append(ComptimeBox(place_type, ComptimeVal(sval.Undefined(place_type))))
        return ComptimeAggregatePtr(type, tuple(places))

    def finish_struct(
        self,
        struct: InterpVal,
        dest: InterpVal,
        indices: tuple[InterpVal, ...],
        names: frozendict[str, InterpVal],
    ) -> None:
        """Close a struct construction (``hir.FinishStruct``): decide the
        struct type and give every field the storage it writes through.

        The struct type is the one ``struct`` names, or - when it is a generic
        template written without its arguments - the one the storage declares
        or the field values determine (see ``_struct_construction_type``).  A
        positional argument binds the field of the same declaration index, a
        keyword one the field of that name, and any other field must have a
        default, which the class body declared (see ``sval.StructField``): the
        default value is written into the field, coerced to its type, exactly
        like a provided argument.  A zero-sized field has no storage, so its
        default records nothing (every value of the type equals its unit
        value).  Defaults do not take part in inferring the generic arguments -
        a type parameter only a defaulted field names cannot be inferred.

        Just like ``finish_array``, the storage takes the struct type through a
        deferral, so its type is *recorded* on the slot rather than fixed on it,
        and every field place that is still pending becomes the address of its
        field in the storage (its value is written through that address, in
        place).

        An inline storage (an expression temporary or a ``Comptime`` variable,
        see ``PendingSlot.is_inline``) gets no addresses at all: its fields are
        their own places, which the construction records on the slot for its
        commit to materialize (see ``init_inline_aggregate``), and a storage that
        already is a :class:`ComptimeAggregatePtr` wrote every field through
        its place before."""
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
        for index, field0 in enumerate(fields.values()):
            if index not in provided and field0.default is None:
                # a field may only be left out when it has a default
                raise CompileError(f'missing a value for field {field0.name!r}')

        if isinstance(dest, PendingSlot) and dest.committed is None and dest.is_inline(struct_type):
            # an inline aggregate: its fields are their own places, so the
            # construction only hands them to the storage slot, whose commit
            # materializes the aggregate (see ``init_inline_aggregate``)
            if self._pending_aggregate(dest) is None:
                places = tuple(
                    provided[index] if index in provided
                    else self._default_field_place(dest, index, fields.get_by_id(index))
                    for index in range(len(fields.by_id))
                )
                self._record_pending_action(dest, _PendingAggregate(struct_type, places))
            else:
                # a second construction into the same storage (the branches of
                # an ``if`` expression): the places are recorded already, so
                # only the fields it leaves out are filled
                for index in range(len(fields.by_id)):
                    if index not in provided:
                        self._default_field_place(dest, index, fields.get_by_id(index))
            return
        if isinstance(_shallow_normalize(dest), ComptimeAggregatePtr):
            # the aggregate is already materialized (a ``Comptime`` variable that
            # was assigned before): every field already is a place, which the
            # arguments wrote through (see ``field_index_addr``)
            for index, field0 in enumerate(fields.values()):
                if index not in provided:
                    self._default_field_place(dest, index, field0)
            return

        if isinstance(dest, PendingSlot) and dest.committed is None:
            # the slot has no address yet: the deferred conversion also
            # records the struct type as the type of the slot (an uncommitted
            # error slot included - its commit hands back the payload pointer)
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
            if index not in provided:
                self._default_field_place(dest_ptr, index, field0)

    def _default_field_place(self, dest: InterpVal, index: int, field: sval.StructField) -> InterpVal:
        """The place a left-out field's default is written through: the same
        place a provided argument is generated into (see ``field_index_addr``
        with ``is_aggregate_init``), so a default fills its field exactly like
        an argument - in a compile-time aggregate's own place, at the field's
        address in memory, or in the fresh place an inline construction records
        for its commit (see ``finish_struct``).  The place is returned so that
        the inline storage can hand it to its pending aggregate.  A zero-sized
        field has no storage, so its default records nothing."""
        assert field.default is not None, 'a field without a default is never left out'
        place = self.field_index_addr(dest, _index_value(index), is_aggregate_init=True)
        self.store(place, ComptimeVal(field.default))
        return place

    # -- array values ----------------------------------------------------------

    def finish_array(self, array: InterpVal, elements: tuple[InterpVal, ...]) -> None:
        """Close an array construction (``hir.FinishArray``): decide the type of
        the array and give every element the storage it writes through.

        The length of the array is the number of elements, and its element type
        the one the storage already has - what is built in a place has to agree
        with the type of the place - or else the common type of the elements
        (see ``_array_construction_type``).  The storage then takes the array
        type through the same deferral a struct construction uses, so its type
        is *recorded* on the slot rather than fixed on it, and every element
        place that is still pending becomes the address of its element in the
        storage (its value is written through that address, in place).

        An inline storage (see ``PendingSlot.is_inline``) gets no addresses at
        all: its elements are their own places, which the construction records
        on the slot for its commit to materialize (see
        ``init_inline_aggregate``), and a storage that already is a
        :class:`ComptimeAggregatePtr` wrote every element through its place
        before.  Neither takes an address."""
        array_type = _array_construction_type(array, elements)
        if isinstance(array, PendingSlot) and array.committed is None and array.is_inline(array_type):
            if self._pending_aggregate(array) is None:
                self._record_pending_action(array, _PendingAggregate(array_type, elements))
            return
        if isinstance(_shallow_normalize(array), ComptimeAggregatePtr):
            return

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

    def init_tuple(self, location: InterpVal, length: int) -> None:
        # a tuple has no storage of its own, so initializing one is building the
        # tuple of the *places* its elements are written into.  A location that
        # already is such a tuple - the target tuple of a destructuring, an
        # element of a tuple being initialized - is used as it is; any other
        # location (a ``Comptime`` variable's slot) records a fresh tuple of
        # element places as a pending action, so that the slot's type - and the
        # conflict another store into it would be - is resolved at its commit
        # (see ``_PendingTuple``).  That tuple of places is what
        # ``hir.TuplePtrElement`` then takes the element addresses of.
        location = _shallow_normalize(location)
        if isinstance(location, ComptimeTuple):
            if len(location.values) != length:
                raise CompileError(
                    f'cannot initialize a tuple of {length} element(s) in a '
                    f'location of {len(location.values)} place(s)'
                )
            return
        if isinstance(location, PendingSlot) and location.inline_mode != InlineMode.NONE and location.committed is None:
            self._record_pending_action(
                location,
                _PendingTuple(tuple(ArgEntry(self.alloca(InlineMode.FULL), True) for _ in range(length))),
            )
            return
        raise CompileError(f'cannot initialize a tuple in {location!r}')

    def tuple_ptr_element(self, location: InterpVal, index: int) -> InterpVal:
        # the place the index-th element of the tuple being initialized in
        # ``location`` is written through (see ``init_tuple``)
        location = _shallow_normalize(location)
        places: tuple[ArgEntry[InterpVal], ...] | None = None
        if isinstance(location, ComptimeTuple):
            places = location.values
        elif isinstance(location, PendingSlot):
            # a slot InitTuple gave a fresh tuple of element places: the tuple is
            # the initialization it recorded
            for action in reversed(location.stores):
                if isinstance(action.data, _PendingTuple):
                    places = action.data.places
                    break
        if places is None:
            raise CompileError(f'cannot take an element of {location!r}')
        if index < 0 or index >= len(places):
            raise CompileError(f'the tuple has no element at index {index}')
        place = places[index]
        assert place.is_ref or isinstance(place.value, ComptimeTuple)
        return place.value

    def _commit_tuple_places(self, tuple_value: ComptimeTuple) -> None:
        # the element places of a tuple location are committed with the
        # location: no ``hir.CommitSlot`` of its own names them
        # (see ``init_tuple``)
        todo: list[InterpVal] = [tuple_value]
        while todo:
            value = todo.pop()
            match value:
                case ComptimeTuple():
                    todo.extend(entry.value for entry in value.values)
                case PendingSlot():
                    self._commit_pending_slot(value)
                case _:
                    raise CompileError(f'unsupported tuple element place {value!r}')

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
        _check_comptime_args(sig, binded_args)
        if fn.force_inline:
            # an undecorated plain Python function: its body is inlined into
            # the current stream (it has no native specialization of its own).
            # Its declared return type still says when it can never return a
            # value, which its body must respect (see ``_current_value_is_empty``)
            return self._start_inline(
                fn.hir.body, fn.hir.arg_is_ref, binded_args, ret,
                generic_var_values, value_is_empty=isinstance(sig.ret_type, sval.EmptyType),
            )
        arg_types = binded_args.map(_arg_type_of)
        spec_sig = sig.specialize(arg_types)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            return self0._make_runtime_call(fn_mir, binded_args, ret, spec_sig[0], ret_sig)

        self._fn_req_resumer = _resumer
        res = self._analyser._request_function(fn, spec_sig[0], spec_sig[1], generic_var_values)
        if res is not None:
            fn_mir, ret_sig = res
            return self.resume(fn_mir, ret_sig)
        return PollResult.SUSPEND

    def resume(self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
        resumer = self._fn_req_resumer
        assert resumer is not None
        self._fn_req_resumer = None
        return resumer(self, fn_mir, ret_sig)

    def _make_runtime_call(self, callee: mir.Value, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal, call_sig: CallSignature, ret_sig: ReturnSignature) -> PollResult:
        """Emit the native call of an already-resolved callee and hand its
        result to the call's result location.  A callee that cannot return
        normally (no value and no error, or no value at all) ends the current
        path, which is unwound like any other ended one (see ``_cut``)."""
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
                    slot = self.alloca(InlineMode.NONE)
                    self._commit_pending_slot(slot, sig_arg.type)
                    self.store(slot, arg.value)
                    mir_args.append(self._to_runtime(slot))
            else:
                ev = self.load(arg.value) if arg.is_ref else arg.value
                mir_args.append(self._to_runtime(self._coerce(ev, sig_arg.type)))

        for arg, (_, sig_arg) in zip(args.positional, call_sig.positional):
            convert_one(arg, sig_arg)

        if call_sig.varargs is not None:
            for arg, sig_arg in zip(args.varargs, call_sig.varargs):
                convert_one(arg, sig_arg)

        if call_sig.kwargs is not None:
            for name, arg in args.kwargs.items():
                convert_one(arg, call_sig.kwargs[name])

        spec = ret_sig.ret_spec()
        callee_result = ret_sig.result_type()
        value_is_empty = ret_sig.value_is_empty()
        if len(ret_sig.exceptions) == 0:
            # the callee has no error part in its MIR: deliver its value alone
            # (and when that value is the empty type, nothing comes back at all)
            self._deliver_result(
                callee, mir_args, ret, ret_sig.ret_type_spec,
                noreturn=ret_sig.is_noreturn(),
            )
            if value_is_empty:
                return self._cut()
            return PollResult.AGAIN
        # the callee may raise: it delivers its normal result into ``ret`` and
        # its error code and payload through fresh places, dispatched below
        callee_exceptions = tuple(ret_sig.exceptions.values)
        use_ret_payload = self._use_ret_payload(callee_exceptions)
        code_place = self.alloca(InlineMode.NONE)
        if use_ret_payload:
            # no ``try`` catches the callee's errors: it writes them straight
            # into the function's own error payload (no copy on the way out)
            payload_place: InterpVal = self._function_result().payload
        else:
            payload_place = self.alloca(InlineMode.NONE)
        error_tuple: InterpVal = ComptimeResult(ret, code_place, payload_place)
        self._deliver_result(callee, mir_args, error_tuple, spec)
        self._check_call_error(
            callee_exceptions, callee_result, code_place, payload_place,
            use_ret_payload, value_is_empty,
        )
        if value_is_empty:
            # a callee with no value to return can never come back normally: the
            # code is one of its exceptions, and every path out of the switch
            # ended (in a clause, or in the function's own error location)
            return self._cut()
        return PollResult.AGAIN

    def _check_call_error(
        self,
        callee_exceptions: tuple[sval.Type, ...],
        callee_result: sval.ResultType,
        code_place: InterpVal,
        payload_place: InterpVal,
        use_ret_payload: bool,
        value_is_empty: bool,
    ) -> None:
        """Dispatch the error a call delivered (see ``_make_runtime_call``): a
        ``switch`` on the callee's code sends every exception to the clause that
        catches it - jumped to with the payload pointer handed over through the
        clause's ``Phi`` - or, when nothing catches it, into the function's own
        error location (see ``_deliver_uncaught_call``).  The code ``0`` of a
        callee that returns a value is the successful outcome and continues in a
        fresh block; a value-less callee has no such code, so there is no
        continuation at all - and one with a single exception has no code at all
        (``u0``), so its only error is dispatched statically."""
        self._commit_pending_slot(code_place)
        if not use_ret_payload:
            self._commit_pending_slot(payload_place)
        code_type = callee_result.code_type
        mir_code_type = code_type.to_mir_type(self._mir_cache)
        zero_sized = code_type.is_zst()
        cont_block = mir.BasicBlock() if not value_is_empty else None
        if zero_sized:
            # the callee uses no code at all: its single exception is the only
            # outcome of the call (a value-less callee always delivers an error)
            assert len(callee_exceptions) == 1
            self._dispatch_call_error(callee_exceptions[0], payload_place, use_ret_payload)
        else:
            assert isinstance(mir_code_type, mir.IntType)
            code = self.load(code_place)
            assert isinstance(code, RuntimeVal)
            case_blocks = [mir.BasicBlock() for _ in callee_exceptions]
            cases = [
                (callee_result.code_of(exception), case_blocks[index])
                for index, exception in enumerate(callee_exceptions)
            ]
            if cont_block is not None:
                cases.append((0, cont_block))
            default = case_blocks[0] if cont_block is None else cont_block
            self._cur_block.emit(mir.Switch(code.value, default, tuple(cases)))
            for index, exception in enumerate(callee_exceptions):
                self._cur_block = case_blocks[index]
                self._dispatch_call_error(exception, payload_place, use_ret_payload)
        if cont_block is not None:
            self._cur_block = cont_block

    def _dispatch_call_error(self, exception: sval.Type, payload_place: InterpVal, use_ret_payload: bool) -> None:
        """Route one exception a call may deliver: to the clause that catches
        it, or - when nothing catches it - into the function's own error
        location (which ends the path)."""
        assert isinstance(exception, sval.StructType)
        target = self._find_catching_clause(exception)
        if target is not None:
            data, clause = target
            assert not use_ret_payload
            incoming = self._union_variant_ptr(payload_place, exception)
            self._route_to_clause(data, clause, incoming, exception)
        else:
            self._deliver_uncaught_call(exception, payload_place, use_ret_payload)

    def _deliver_result(
        self,
        callee: mir.Value,
        mir_args: list[mir.Value],
        ret: InterpVal,
        spec: RetSpec,
        noreturn: bool = False,
    ) -> None:
        """Emit the native call of a callee returning the value(s) of ``spec``
        and hand every result to its place.

        The results go either into the places the caller reserved - ``ret`` is
        the place of a single result, or a tuple of places nested exactly like
        the results (see ``_pair_places``) - or, when a result (or a group of
        them) is given a single place of its own, into fresh places whose tuple
        is then stored in that place.  A tuple may be inlined, so such a place
        has to be able to hold its value inline (a ``Comptime`` variable); a
        plain slot rejects the tuple when it is committed (see
        ``_no_runtime_type``).

        A zero-sized result occupies no place of its own: its unit value is
        written into its place.  A ``noreturn`` call (``mir.NoReturn``) ends the
        block it is emitted into."""
        places: list[InterpVal] = []
        packed: list[tuple[InterpVal, ComptimeTuple]] = []
        self._pair_places(spec, ArgEntry(ret, True), places, packed)

        result_args: list[mir.Value] = []
        ret_type: mir.ReturnType = mir.VOID
        by_value: tuple[InterpVal, sval.Type] | None = None
        for leaf, place in zip(iter_ret_leaves(spec), places):
            if leaf.via_result_ptr:
                target = place
                if isinstance(target, PendingSlot) and target.committed is None:
                    # the slot has no address yet: the call writes through a
                    # placeholder its commit fills in
                    target = self._defer_ptr_convertion(target, leaf.type)
                else:
                    # a committed location: the result-location conversion (a
                    # union variant, an option's payload, ...) hands back the
                    # pointer to write through
                    target = self._convert_result_ptr(target, leaf.type)
                result_args.append(self._to_runtime(_shallow_normalize(target)))
                continue
            if leaf.type.is_zst():
                # a zero-sized result produces no register: deliver its unit
                # value.  The location of a call whose result is dropped (an
                # expression statement) would otherwise stay untyped, and the
                # type is what makes its slot a compile-time box of the unit
                # value (see ``_commit_pending_slot``)
                unit = leaf.type.get_unit_value()
                if unit is not None:
                    self.store(place, ComptimeVal(unit))
                continue
            mir_type = leaf.type.to_mir_type(self._mir_cache)
            if mir_type is None:
                raise _no_runtime_type(leaf.type)
            ret_type = mir_type
            by_value = (place, leaf.type)

        if noreturn:
            # the callee never comes back: the call ends the block it is in
            ret_type = mir.NORETURN
        call_inst = self._emit(mir.Call(callee, (*mir_args, *result_args), ret_type))
        if by_value is not None:
            place, type = by_value
            if isinstance(type, sval.UnionType):
                # a union value cannot be converted, and the destination (the
                # function's own payload, a union of its whole exception set) may
                # be a superset: the storage is reinterpreted as the union that
                # arrives (its variants live at offset 0, and a pointer
                # reinterpretation is free)
                if isinstance(place, PendingSlot) and place.committed is None:
                    place = self._defer_ptr_convertion(place, type)
                else:
                    place = self._convert_result_ptr(place, type)
            self.store(place, RuntimeVal(call_inst, type))

        if len(packed) == 0:
            return
        for _, tree in packed:
            self._commit_tuple_places(tree)
        for target, tree in packed:
            self.store(target, tree)

    def _pair_places(
        self,
        node: RetSpec,
        entry: ArgEntry[InterpVal],
        places: list[InterpVal],
        packed: list[tuple[InterpVal, ComptimeTuple]],
    ) -> None:
        """Pair the results of ``node`` with the place tree the caller
        reserved: a single result takes the place ``entry`` denotes, a group
        takes a tuple of places nested exactly like the results (see
        ``astgen._gen_target_tuple``), and the leaves are appended to
        ``places`` in depth-first order.  A group the caller gives a single
        non-tuple place is *packed*: fresh places are reserved and their
        (nested) tuple is stored in that place after the call (see
        ``_deliver_result``)."""
        target = entry.value
        match node:
            case RetValue():
                if isinstance(target, ComptimeTuple):
                    raise CompileError('cannot unpack one value into a tuple target')
                if not entry.is_ref:
                    raise CompileError(
                        'the target of a multi-value result must be addressable'
                    )
                places.append(target)
            case RetTuple(type=type, values=values):
                if isinstance(type, sval.ResultType):
                    # the result group: its value, its error code and its payload
                    # union, in the order of the result location
                    assert isinstance(target, ComptimeResult)
                    for child, sub in zip(
                        values, (target.value, target.code, target.payload),
                    ):
                        self._pair_places(child, ArgEntry(sub, True), places, packed)
                    return
                if isinstance(target, ComptimeTuple):
                    if len(target.values) != len(values):
                        raise CompileError(
                            f'cannot unpack {len(values)} value(s) into '
                            f'{len(target.values)} target(s)'
                        )
                    for child, sub_entry in zip(values, target.values):
                        self._pair_places(child, sub_entry, places, packed)
                    return
                tree = self._fresh_places(node, places)
                assert isinstance(tree, ComptimeTuple)
                packed.append((target, tree))

    def _fresh_places(self, node: RetSpec, places: list[InterpVal]) -> InterpVal:
        """Fresh places for a group packed into a single location: a tuple of
        places mirroring the (possibly nested) shape of the results, with the
        leaf places appended to ``places`` in depth-first order."""
        match node:
            case RetValue():
                place = self.alloca(InlineMode.FULL)
                places.append(place)
                return place
            case RetTuple(values=values):
                entries: list[ArgEntry[InterpVal]] = []
                for child in values:
                    place = self._fresh_places(child, places)
                    entries.append(ArgEntry(place, isinstance(child, RetValue)))
                return ComptimeTuple(tuple(entries))

    def _start_inline(
        self,
        body: tuple[hir.Inst, ...],
        arg_is_ref: tuple[bool, ...],
        args: ArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        generic_var_values: frozendict[sval.TypeVar, sval.Value] | None = None,
        value_is_empty: bool = False,
    ) -> PollResult:
        """Start the inlined body of a plain Python callee: convert its
        bound arguments into addressable values (the callee's ``hir.Arg``
        leaves denote its parameter slots) and push its frame.  The body is
        emitted into the block the call happened in (an inlined body is
        entered unconditionally), and the caller's continuation - the frame's
        *exit block*, which every ``return`` of the body jumps to and its
        falling end joins - is created on the first path of the body that
        reaches it (see ``InlineFrame.continuation`` and ``_pop_frame``).  An
        argument that is already a reference is
        forwarded as the address it is.  The body now runs under the
        machine; its return statements write into ``ret`` directly (it is
        the body's result location), so no result is handed back here.

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
        for (arg, is_ref) in zip(args.positional, arg_is_ref):
            # ``arg_is_ref`` is the *callee's* HIR binding: a method's ``self`` is
            # bound directly to its argument (the address) even though the
            # caller passed it as a value; every other parameter is forwarded as
            # the caller passed it (by reference, or materialized by value)
            if arg_is_ref or arg.is_ref:
                arg_values.append(arg.value)
            else:
                slot = self.alloca(InlineMode.FULL)
                self.store(slot, arg.value)
                self._commit_pending_slot(slot)
                arg_values.append(slot)
        frame_values: dict[sval.TypeVar, InterpVal] = {}
        if generic_var_values is not None:
            frame_values = {tv: ComptimeVal(v) for tv, v in generic_var_values.items()}
        # the body delivers its result into the call's place and shares the
        # function proper's error places: an error of the body leaves the
        # function the same way
        error = self._function_result()
        frame = InlineFrame(
            frame_values, tuple(arg_values),
            ComptimeResult(ret, error.code, error.payload),
            body, value_is_empty=value_is_empty,
        )
        self._frames.append(frame)
        return PollResult.AGAIN

    # -- finishing -------------------------------------------------------------

    def finish(self) -> None:
        """Called when the body of the function proper has been fully
        typed: end the last block (a body that fell off its end returns
        void), fix an inferred return convention and flatten the deferred
        insertion blocks away."""
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Ret(None))
        self._finish_function()
        mir_fn = self._fn_instance.mir
        mir.normalize(mir_fn)

    def _finish_function(self) -> None:
        """Fix the return convention of the function proper - the return type
        and/or the exception set the signature left to be inferred, from the
        stores the body performed - fill in the ``mir.Ret`` of every deferred
        return site, and write the deferred error codes, which the result type
        now names."""
        partial = self.partial_ret_sig
        value_spec = partial.ret_type_spec if partial is not None else None
        if value_spec is None:
            location = self._result_loc()
            assert isinstance(location, PendingSlot)
            # a result location nothing was ever stored into holds no value: its
            # type is the *empty* type, so the function cannot return a value at
            # all (see ``sval.EmptyType``)
            value_spec = sval.make_ret_spec(location.committed_type())
        exceptions = partial.exceptions if partial is not None else None
        if exceptions is None:
            exceptions = ArraySet()
            for exception in self._error_types.values:
                exceptions.add(exception)
        self._materialize_ret_sig(ReturnSignature(value_spec, exceptions))
        sig = self.ret_sig
        assert sig is not None
        result_type = sig.result_type()
        self._write_deferred_error_codes(result_type)
        spec = sig.ret_spec()
        places = _result_places(self._current_result_loc())
        index = ret_by_value_index(spec)
        for block in self._deferred_returns:
            if index is None:
                block.insts.append(mir.Ret(None))
            else:
                slot = _shallow_normalize(places[index])
                if isinstance(slot, RuntimeVal):
                    load = mir.Load(slot.value)
                    block.insts.append(load)
                    block.insts.append(mir.Ret(load))
                elif isinstance(slot, ComptimeBox):
                    assert slot.value is not None
                    block.insts.append(mir.Ret(self._to_runtime(slot.value)))
                else:
                    raise CompileError('cannot deliver the return value')

    def _write_deferred_error_codes(self, result_type: sval.ResultType) -> None:
        """Fill in the deferred writes of the function's error codes (see
        ``_PendingErrorCodeWrite``): the tag of an exception is its position in
        the exception set, offset by the result type's base (a value-less
        function has no "no error" code).  A zero-sized code holds no storage -
        and the code of a value-less function's single exception is always 0 -
        so there is nothing to write then."""
        if len(self._pending_error_code_writes) == 0:
            return
        code_type = result_type.code_type
        if code_type.is_zst():
            # a zero-sized code holds no storage: the only code a value-less
            # function ever writes is 0, which is what ``u0`` is anyway
            return
        mir_code_type = code_type.to_mir_type(self._mir_cache)
        assert isinstance(mir_code_type, mir.IntType)
        code = _shallow_normalize(self._function_result().code)
        assert isinstance(code, RuntimeVal), 'an error code with storage is memory'
        for write in self._pending_error_code_writes:
            tag = result_type.code_of(write.exception)
            write.insertion.insts.append(
                mir.Store(code.value, mir.Int(tag, mir_code_type))
            )

class Analyser:
    def __init__(self, resolver: GlobalResolver, mir_lower_cache: sval.MirLowerCache) -> None:
        self._resolver = resolver
        # the host's MIR-mirror interning table, which every mirror a body
        # creates is made through (see ``sval.MirLowerCache``)
        self.mir_lower_cache = mir_lower_cache
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
        ret_sig: PartialReturnSignature | None,
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
            # function being typed; its return convention must be known already
            actual = instance.ret_sig
            if actual is None and ret_sig is not None and ret_sig.is_complete():
                actual = ret_sig.complete()
            if actual is None:
                raise CompileError(
                    f"recursive function {fn_entry.hir.name} requires a declared "
                    f"return type and exception set"
                )
            return instance.mir, actual
        mir_fn = mir.Function(name, [], [], mir.VOID)
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
        runner.run_function(
            fn_entry.hir.body, fn_entry.hir.arg_is_ref,
            call_sig, ret_sig, frame_generic_values,
        )
        self._analyse_stack.append(runner)
        return None

    def _run(self):
        while self._analyse_stack:
            top = self._analyse_stack[-1]
            if top._run_machine() == PollResult.DONE:
                top.finish()
                assert top.ret_sig is not None
                self._analyse_stack.pop()
                instance = top._fn_instance
                instance.ret_sig = top.ret_sig
                instance.mir.is_complete = True
                if self._analyse_stack:
                    last_top = self._analyse_stack[-1]
                    last_top.resume(instance.mir, top.ret_sig)
                else:
                    return instance.mir, top.ret_sig
        return None

    def analyse_function(self, fn_entry: FunctionValue, call_sig: CallSignature, ret_sig: PartialReturnSignature | None):
        """Type (and thereby compile) the specialization ``call_sig`` of
        ``fn_entry`` if it is not compiled yet."""
        if self._request_function(fn_entry, call_sig, ret_sig) is None:
            self._run()

    def finish(self) -> CompileBatch:
        return self._symbol_table

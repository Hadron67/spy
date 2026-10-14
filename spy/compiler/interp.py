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
  compile-time aggregate snapshot, a ``sval.ComptimePtr`` into compile-time
  storage, a function to call/inline, ...).  "No value" is the unit value
  ``sval.Void()`` - the unique value of the zero-sized void type - never
  Python ``None``, and the ``None`` the source writes is the absent value
  of an option, ``sval.Null()``,
* :class:`RuntimeVal` - the object of an already emitted MIR
  instruction (a typed runtime value),
* :class:`PendingSlot` - an executed ``Alloca`` that is only committed
  (into real memory, or into the ``sval.Box`` a comptime alloca reserves)
  when ``hir.CommitSlot`` runs; a slot whose content type is zero-sized
  never gets memory: it only records its unit value, or
* :class:`ComptimeResultPtr` - the result location of a function: the
  places its normal value, its error code and its payload union are
  delivered into (the interpreter's form of a :class:`sval.ResultType`).
  It is the one interpreter place that is not an :class:`InterpVal` (it
  never flows as a value; see :class:`ComptimeResultPtr`).

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

* calls to the ``spy`` builtins (``spy.compile_log``) are evaluated eagerly,
* calls to other spy functions (jit or aot) become native ``call``
  instructions to the specialization selected by the argument types,
* calls to plain Python functions inline the callee body into the
  current stream.

``spy.typeof`` is not a call at all: it is a ``syntax`` marker the parser
lowers to a *type probe* (see ``hir.TypeOfBegin``), which types its argument
without emitting anything.

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

import math
import operator
import struct
import sys
import types as pytypes
from abc import abstractmethod
from annotationlib import Format
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from typing import Any, Never, Self, override

from . import hir, mir, sval
from .binop import BinaryOp, CompareOp, UnaryOp
from .errors import CoerceError, CompileError
from .fn import (
    ArgEntry,
    ArgList,
    ArgNode,
    CallSignature,
    CompileBatch,
    CompoundArgNode,
    FunctionInstance,
    FunctionIR,
    FunctionValue,
    NativeFn,
    PartialReturnSignature,
    RawArgList,
    ReturnSignature,
    RuntimeArgNode,
    Signature,
    SignatureFormalArg,
    signature_of_fn_type,
)
from .sval import (
    CompileContext,
    RetComptime,
    RetSpec,
    RetTuple,
    RetValue,
    iter_ret_leaves,
    ret_by_value_index,
)
from .util import ArraySet, IndexedMap, TriState, frozendict

_MAX_INLINE_DEPTH = 64

_PY_OPS: dict[str, Any] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '**': operator.pow,
    '|': operator.or_,
    '&': operator.and_,
    '^': operator.xor,
    '<<': operator.lshift,
    '>>': operator.rshift,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

# the magic method of every binary operator: the method a struct's ``a op b``
# resolves to, its reflected counterpart (``b``'s method, called with the
# operands swapped, when ``a`` names none) and its in-place counterpart (what
# an augmented assignment calls first, see ``HirRunner.binary_assign``)
_BINARY_METHODS: dict[str, tuple[str, str, str]] = {
    '+': ('__add__', '__radd__', '__iadd__'),
    '-': ('__sub__', '__rsub__', '__isub__'),
    '*': ('__mul__', '__rmul__', '__imul__'),
    '/': ('__truediv__', '__rtruediv__', '__itruediv__'),
    '//': ('__floordiv__', '__rfloordiv__', '__ifloordiv__'),
    '%': ('__mod__', '__rmod__', '__imod__'),
    '**': ('__pow__', '__rpow__', '__ipow__'),
    '|': ('__or__', '__ror__', '__ior__'),
    '&': ('__and__', '__rand__', '__iand__'),
    '^': ('__xor__', '__rxor__', '__ixor__'),
    '<<': ('__lshift__', '__rlshift__', '__ilshift__'),
    '>>': ('__rshift__', '__rrshift__', '__irshift__'),
}

# the magic method of every comparison: the method the left operand's struct
# answers with, and the one a struct *right* operand answers with when the
# left one has none (the operands swapped: ``a < b`` becomes ``b > a``)
_COMPARE_METHODS: dict[str, tuple[str, str]] = {
    '==': ('__eq__', '__eq__'),
    '!=': ('__ne__', '__ne__'),
    '<': ('__lt__', '__gt__'),
    '<=': ('__le__', '__ge__'),
    '>': ('__gt__', '__lt__'),
    '>=': ('__ge__', '__le__'),
}

# the magic method of every unary operator, and the one ``bool(x)`` uses
_UNARY_METHODS: dict[str, str] = {'-': '__neg__', '~': '__invert__'}
_BOOL_METHOD = '__bool__'
_LEN_METHOD = '__len__'

# the operation of one ``std.atomic.atomic_rmw``, by the name of the
# ``std.atomic.RmwOp`` member that selects it
_ATOMIC_RMW_OPS: dict[str, mir.AtomicRmwOp] = {
    'XCHG': 'xchg',
    'ADD': 'add',
    'SUB': 'sub',
    'AND': 'and',
    'OR': 'or',
    'XOR': 'xor',
}

# how strong each memory ordering is, for the success/failure ordering rule of
# ``atomic_cmpxchg`` (see ``_check_atomic_failure_ordering``)
_ATOMIC_ORDER_STRENGTH: dict[mir.AtomicOrdering, int] = {
    mir.AtomicOrdering.RELAXED: 0,
    mir.AtomicOrdering.ACQUIRE: 1,
    mir.AtomicOrdering.RELEASE: 1,
    mir.AtomicOrdering.ACQ_REL: 2,
    mir.AtomicOrdering.SEQ_CST: 3,
}


def _check_atomic_ordering(
    ordering: mir.AtomicOrdering, allowed: tuple[mir.AtomicOrdering, ...], what: str
) -> mir.AtomicOrdering:
    """Check that an atomic operation with the given allowed memory orderings
    was given one of them (LLVM rejects the others)."""
    if ordering not in allowed:
        names = ', '.join(o.name for o in allowed)
        raise CompileError(
            f'{what} does not accept the {ordering.name} ordering; '
            f'it must be one of {names}'
        )
    return ordering


# the ``std.atomic.RmwOp`` operations each pointee kind accepts: an integer
# takes every operation, a float only the arithmetic ones (LLVM has ``fadd``/
# ``fsub`` but no float bitwise op), and a pointer only ``xchg``
_ATOMIC_RMW_OPS_BY_KIND: dict[str, tuple[mir.AtomicRmwOp, ...]] = {
    'float': ('xchg', 'add', 'sub'),
    'pointer': ('xchg',),
    'integer': ('xchg', 'add', 'sub', 'and', 'or', 'xor'),
}


def _check_atomic_rmw_op(elem: sval.Type, op: mir.AtomicRmwOp) -> None:
    """Check that the pointee kind accepts the ``std.atomic.RmwOp`` operation
    ``op`` (LLVM supports only the arithmetic atomicrmw operations on floats,
    and only ``xchg`` on pointers; a complex number is handled part by part, so
    it takes what a float takes)."""
    if isinstance(elem, (sval.FloatType, sval.ComplexType)):
        kind = 'float'
    elif isinstance(elem, sval.PointerType):
        kind = 'pointer'
    else:
        kind = 'integer'
    allowed = _ATOMIC_RMW_OPS_BY_KIND[kind]
    if op not in allowed:
        names = ', '.join(o.upper() for o in allowed)
        raise CompileError(
            f'std.atomic.atomic_rmw supports only {names} on a {elem} pointee, '
            f'not {op.upper()}'
        )


def _check_atomic_failure_ordering(
    success: mir.AtomicOrdering, failure: mir.AtomicOrdering
) -> None:
    """Check the ordering pair of a ``cmpxchg`` against LLVM's rules: the
    failure ordering may not be ``release``/``acq_rel``, and may not be
    stronger than the success ordering."""
    if failure in (mir.AtomicOrdering.RELEASE, mir.AtomicOrdering.ACQ_REL):
        raise CompileError(
            'the failure ordering of std.atomic.atomic_cmpxchg cannot be '
            'release or acq_rel'
        )
    if _ATOMIC_ORDER_STRENGTH[failure] > _ATOMIC_ORDER_STRENGTH[success]:
        raise CompileError(
            'the failure ordering of std.atomic.atomic_cmpxchg cannot be '
            'stronger than its success ordering'
        )


@dataclass(slots=True)
class CompileVars:
    """The compile-time behaviours a frame can configure (see
    ``InlineFrame.compile_vars_stack``).  Every field has the default the
    compiler uses; a frame inherits the configuration of its caller."""

    # the float type an integer ``/`` (and a division-like operation that
    # promotes an integer) is computed in
    int_div_type: sval.FloatType = field(default_factory=lambda: sval.FloatType(64))
    # whether integer ``//`` truncates towards zero (True) or floors (False,
    # the Python behaviour)
    int_trunc_div: bool = False
    # the largest magnitude of a compile-time integer exponent that is
    # unfolded; a larger one becomes a runtime loop
    max_exp_unroll: int = 4096

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
class _PendingActionData:
    """One action recorded by a :class:`PendingSlot`: how it is delivered
    once the slot has an address is decided by the runner (see
    ``HirRunner._exec_pending_action``), the slot itself only asks for the
    type it contributes and whether it is inline."""

    @abstractmethod
    def info(self) -> tuple[sval.Type, bool]:
        """Returns (type, may_be_inlined) of the value the action delivers."""
        ...

    def is_comptime(self) -> bool:
        """Whether the action delivers its value entirely at compile time - what
        a slot whose storage is still inferred reads to decide between a ``Box``
        and memory (see ``PendingSlot.use_comptime_storage``)."""
        return False

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
    the stored value and whether it may be kept inline - a store that crossed a
    runtime block boundary since the slot's own position may not (see
    ``HirRunner._record_pending_store``).  The store itself is delivered once
    the slot's final type is known (the stored value is coerced to it then)."""

    type: sval.Type
    is_inline: bool
    value: InterpVal

    def info(self) -> tuple[sval.Type, bool]:
        return self.type, self.is_inline

    @override
    def is_comptime(self) -> bool:
        return self.is_inline

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
    # the places the tuple's elements are written into.  The slot holds the tree
    # of element places itself, and its type - the tuple of the types of those
    # places - is read when the slot is committed, once the elements have been
    # written.  Recording it as an action (rather than committing the slot
    # right away) keeps the slot's type resolution - and the conflict it
    # reports against any other store into the slot - intact.
    places: tuple[InterpVal, ...]

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return _tuple_places_type(self.places), True

    @override
    def is_comptime(self) -> bool:
        return all(_place_is_comptime(place) for place in self.places)

@dataclass
class _PendingAggregate(_PendingActionData):
    # one aggregate initialization recorded by a PendingSlot (see
    # ``HirRunner.finish_array``/``HirRunner.finish_struct``): the places of its
    # fields, in declaration order, or of its elements.  The type recorded here
    # is the struct or array type the construction resolved, and the places are
    # committed together with the slot (see ``HirRunner.init_inline_aggregate``).
    type: sval.Type
    places: tuple[InterpVal, ...]

    @override
    def info(self) -> tuple[sval.Type, bool]:
        return self.type, True

    @override
    def is_comptime(self) -> bool:
        return all(_place_is_comptime(place) for place in self.places)

@dataclass(slots=True)
class PendingSlot(InterpVal):
    """The value of an executed ``hir.Alloca`` before it is *committed*.
    In this phase a store (or an RLS call) into the slot only records a
    :class:`_PendingAction`; the slot acquires its final type (the
    pairwise ``resolve_peer_type`` of the action types) and its storage
    when ``hir.CommitSlot`` runs, which materializes it into a
    :class:`RuntimeVal` (a pointer to real memory) or into the ``sval.Box``
    a comptime alloca reserves (see ``HirRunner._commit_pending_slot``).

    ``is_comptime`` says where the value is kept: ``True``/``False`` when it is
    fixed, ``None`` when it is still inferred from the stores (see
    ``use_comptime_storage``).

    ``runtime_pos`` is the *runtime* block the ``Alloca`` ran in - the MIR block
    the walk was building, recorded so that a store into the slot knows whether
    the walk has since crossed a runtime block boundary (see
    ``_record_pending_store``).

    ``insertion`` is the position the slot's storage is produced at: a
    :class:`mir.Insertion` emitted where the ``Alloca`` ran, whose
    instructions the slot's commit fills with the :class:`mir.Alloca` (or,
    for an element/field place of an aggregate construction, with the
    ``Gep`` addressing it, see ``finish_array``/``finish_struct``)."""

    insertion: mir.Insertion
    is_comptime: bool | None
    is_const: bool = False
    runtime_pos: mir.BasicBlock | None = None
    stores: list[_PendingAction] = field(default_factory=list)
    committed: InterpVal | None = None
    declared: sval.Type | None = None

    def committed_type(self) -> sval.Type:
        if self.declared is not None:
            return self.declared
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

    def use_comptime_storage(self) -> bool:
        """Whether the slot holds its value in compile-time storage (a
        ``sval.Box``) rather than memory: ``True``/``False`` when ``is_comptime``
        fixes it, otherwise inferred - compile-time when *every* store into it is
        an inline compile-time one (a store that crossed a runtime block boundary
        since the slot's own position is not, see ``_record_pending_store``)."""
        if self.is_comptime is not None:
            return self.is_comptime
        return all(store.data.is_comptime() for store in self.stores)

@dataclass(frozen=True, slots=True)
class ComptimeResultPtr:
    """The *pointer* form of a :class:`sval.ResultType`: the result location of
    a function - the place its normal result is delivered into, the place of its
    error code and the place of its payload union, in that order.  The places
    are :class:`PendingSlot`s while the location is not committed yet (the code
    and payload widths follow from the exception set at the commit, see
    ``_commit_error_space``), and the materialized places afterwards.

    Unlike every other interpreter value, a result location is *not* an
    :class:`InterpVal`: it is a composite place that only ever occurs as the
    frame's own result location (see ``InlineFrame.ret_loc``), never as a value
    an expression produces or consumes, so it has no place in ``_type_of``'s
    world.  A call hands its own result location over as the separate
    ``ret_loc``/``err_loc`` places (see ``_deliver_result``)."""

    value: ResultLocTree | None
    code: InterpVal
    payload: InterpVal

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
    """Whether the value is a *compile-time* value: one the interpreter holds
    fully in Python.  A ``ComptimeVal`` always is (its ``sval`` value - a
    scalar, an aggregate snapshot, an ``sval.ComptimePtr``/``sval.CastedPtr`` -
    is known in full); a ``RuntimeVal`` **never** is, whatever it refers to (the
    old "a runtime value that refers to no register" relaxation is gone, see
    §1 of the behaviour doc); an uncommitted slot is decided by its stores."""
    todo = [val]
    while todo:
        val = todo.pop()
        match val:
            case ComptimeVal():
                # an sval value (a scalar, an aggregate snapshot, a
                # ComptimePtr/CastedPtr) is compile-time in full
                continue
            case RuntimeVal():
                return False
            case PendingSlot():
                if val.committed is not None:
                    todo.append(val.committed)
                elif val.declared is not None or val.stores:
                    # a declared-but-uncommitted slot is compile-time when every
                    # store into it is an inline compile-time one
                    for action in val.stores:
                        if not action.data.is_comptime():
                            return False
                        if isinstance(action.data, _PendingStore):
                            todo.append(action.data.value)
                else:
                    return False
            case _:
                return False
    return True

def _place_is_comptime(place: InterpVal) -> bool:
    """Whether the place ``place`` holds its value at compile time - the
    storage decision a slot's aggregate action reads for its field/element
    places (see ``_PendingAggregate``/``_PendingTuple``)."""
    todo = [place]
    while todo:
        value = todo.pop()
        if isinstance(value, ComptimeVal):
            continue
        if isinstance(value, RuntimeVal):
            return False
        if isinstance(value, PendingSlot):
            if value.committed is not None:
                todo.append(value.committed)
            else:
                for action in value.stores:
                    if not action.data.is_comptime():
                        return False
            continue
        return False
    return True

def _references_register(value: mir.Value) -> bool:
    """Whether the MIR value tree ``value`` refers to a *register* - a
    ``mir.Inst`` (the result of an instruction) or a ``mir.Param`` (a function
    argument).  A global initializer may hold only constants and references to
    globals, so a value that does (a load, a computation, an aggregate built by
    an ``InsertValue`` chain, ...) cannot stand in one.  A ``mir.GlobalValue``
    is a reference to a global and is not looked into (its body is not part of
    the value it denotes)."""
    todo: list[mir.Value] = [value]
    while todo:
        item = todo.pop()
        if isinstance(item, (mir.Inst, mir.Param)):
            return True
        if isinstance(item, mir.GlobalValue):
            continue
        children = item.get_children()
        if children is not None:
            todo.extend(children)
    return False

def _is_aggregate(type: sval.Type) -> bool:
    """Whether ``type`` is an *aggregate* type (a struct, array, tuple, string
    dictionary, complex, option, tagged union or payload union): a value of one
    is flattened into a cell per scalar leaf when it is stored in a ``Box`` (see
    ``sval.scalar_leaves``)."""
    return isinstance(type, sval.AggregateType)

def _is_union_unit(val: InterpVal) -> bool:
    """Whether ``val`` is a union value that carries no storage
    (:class:`sval.UnionValue`): it says only which union it belongs to, so a
    storage destination has no variant to write."""
    obj = _to_comptime(val)
    return isinstance(obj, sval.UnionValue)

def _is_undefined_val(val: InterpVal) -> bool:
    """Whether the value is the undefined literal - the untyped
    ``sval.UntypedUndefined`` ``std.core.undefined`` evaluates to, or the typed
    ``sval.UndefinedScalar``/``sval.UndefinedAggregate`` it becomes once it is
    coerced to a type: a store of one leaves its destination undefined (see
    ``HirRunner.store``).

    A typed undefined of a *pointer* type is not one: null is not a legal value
    of a non-null pointer, so it stands for a dangling - but defined - pointer,
    and a store of it writes that pointer (see ``mir.Dangling``)."""
    obj = _to_comptime(_shallow_normalize(val))
    if isinstance(obj, sval.UndefinedAggregate):
        return True
    if isinstance(obj, sval.UndefinedScalar):
        return not isinstance(obj.type, sval.PointerType)
    return isinstance(obj, sval.UntypedUndefined)

def _aggregate_place_types(type: sval.Type) -> tuple[sval.Type, ...]:
    """The type of every place of the aggregate ``type``, in place order: the
    fields of a struct in declaration order, the elements of a tuple/array, the
    entries of a ``dict[str, T]``, the real and imaginary parts of a complex
    number."""
    if isinstance(type, sval.StructType):
        return tuple(field0.type for field0 in type.fields().values())
    if isinstance(type, sval.TupleType):
        return type.types
    if isinstance(type, sval.StrDictType):
        return tuple(type.values.values())
    if isinstance(type, sval.ArrayType):
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot tell how many elements {type} holds')
        return (type.elem,) * length
    if isinstance(type, sval.ComplexType):
        return (type.elem, type.elem)
    raise CompileError(f'{type} is not an aggregate')

def _as_aggregate(ev: InterpVal) -> sval.AggregateValue | None:
    """The ``sval.AggregateValue`` the interpreter value denotes, when it
    denotes one (a compile-time aggregate value - the unit value of a zero-sized
    aggregate, or one whose field values are all known)."""
    if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.AggregateValue):
        return ev.obj
    return None

class BlockFrameData:
    pass


@dataclass(frozen=True, slots=True)
class _DeferEntry:
    """One ``with syntax.defer():`` region declared in an open region: the
    exits that trigger it (``flags``, a ``hir.DeferPath``) and the entry block
    of the deferred body the interpreter emitted (the template every transfer
    that triggers it refers to, see ``mir.EndDefer``/``Terminator.get_defer_blocks``)."""

    flags: hir.DeferPath
    entry: mir.BasicBlock


@dataclass
class DeferBlockData(BlockFrameData):
    """The state of the defer body currently being walked (a
    ``with syntax.defer():`` region).  ``flags`` is the exits the defer
    triggers (``hir.DeferPath``), ``entry`` the entry block of the body
    (appended to the enclosing region's defer list), ``body_defers`` the
    defers declared inside the body itself - they are what its
    ``mir.EndDefer`` triggers - and ``saved_block`` the block the walk
    continues in after the body, which is emitted detached from the current
    one (see ``HirRunner._exec_defer``)."""

    flags: hir.DeferPath
    entry: mir.BasicBlock
    body_defers: list[_DeferEntry]
    saved_block: mir.BasicBlock


@dataclass
class TypeOfBlockData(BlockFrameData):
    """The state of the ``syntax.typeof`` type probe currently being walked.

    The probe's instruction stream follows the ``hir.TypeOfBegin`` that opened
    it, closed by the matching ``hir.TypeOfEnd``.  The interpreter builds the
    probe into a *detached* block - never linked to the function's CFG, so it is
    never lowered - rather than into the block being built, which is what makes
    the probe emit no code; ``saved_block`` is the block the walk continues in
    once the probe closes.  A transfer that would leave the probe is rejected
    (see ``_collect_exit_defers``/``_cut``): the probe only ever types."""

    saved_block: mir.BasicBlock


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
    # which branch is being walked: 0 the then-region, 1 the else-region (see
    # ``HirRunner._region_defers`` and the region-exit collection)
    region: int = 0
    # the deferred bodies declared directly in each region (see ``hir.Defer``)
    then_defers: list[_DeferEntry] = field(default_factory=list)
    else_defers: list[_DeferEntry] = field(default_factory=list)

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
    # the deferred bodies declared directly in the loop body (see ``hir.Defer``)
    body_defers: list[_DeferEntry] = field(default_factory=list)

@dataclass
class PlainBlockData(BlockFrameData):
    """The state of one open ``hir.Block`` of the HIR.

    ``exit_block`` is the block the code after the block's ``End`` is typed in
    - the target of a ``hir.BreakIf`` that leaves it - created on the first
    break (or by the falling end, whichever comes first).  ``p_end`` is the
    position of the matching ``End`` (found by ``_scan_block``).  Unlike a
    loop, the code after the block is reachable even without a break: the
    falling end reaches the same ``exit_block``."""

    p_end: int
    exit_block: mir.BasicBlock | None = None
    # the deferred bodies declared directly in the block body (see ``hir.Defer``)
    body_defers: list[_DeferEntry] = field(default_factory=list)

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
    # the deferred bodies declared directly in each region (the try body and
    # every except clause), indexed like ``region`` (see ``hir.Defer``)
    region_defers: list[list[_DeferEntry]] = field(default_factory=list)
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
    def __init__(self, generic_var_values: dict[sval.TypeVar, InterpVal], arg_values: tuple[InterpVal, ...], ret_loc: ComptimeResultPtr, insts: tuple[hir.Inst, ...], fn_name: str, value_is_empty: bool = False, on_done: Callable[[], None] | None = None, compile_vars: CompileVars | None = None, closure_values: tuple[InterpVal, ...] = ()) -> None:
        self.generic_var_values = generic_var_values
        self.arg_values = arg_values
        # the name of the function this frame executes (the specialization's
        # name, with its signature, for the function proper; the closure's or
        # the inlined function's own name otherwise): a nested closure created
        # here prefixes its native name with it (see ``exec_make_closure``)
        self.fn_name = fn_name
        # the capture places of a closure body (see ``hir.Closure``): the
        # enclosing slots an inlined closure captures, or the pointers a
        # compiled closure was passed - empty for an ordinary body
        self.closure_values = closure_values
        # the compile-time configuration of this body, as a stack so a nested
        # scope can push an override and pop it again; the bottom entry is
        # inherited from the caller (a fresh default for the function proper)
        self.compile_vars_stack: list[CompileVars] = [CompileVars() if compile_vars is None else compile_vars]
        # the frame's result location: the place its result is delivered into and
        # the function proper's error places (an inlined plain body raises into
        # the enclosing function's error location, so it shares them)
        self.ret_loc: ComptimeResultPtr = ret_loc
        # whether the body of this frame has no value to return (the *declared*
        # one of an inlined plain-Python body; the function proper's own type is
        # asked for when it runs, see ``HirRunner._current_value_is_empty``)
        self.value_is_empty = value_is_empty
        # what the call the body was inlined into has to do once the body
        # delivered its result (a subscript resolved through a method needs to
        # move it into the instruction's register, see ``HirRunner.subscript``);
        # None for the function proper and for a call that wants nothing more
        self.on_done: Callable[[], None] | None = on_done
        self.insts = insts
        self.pc: int = 0
        self.block_stack: list[BlockFrame] = []
        self.regs: dict[hir.Inst, InterpVal] = {}
        # the deferred bodies declared directly in the body's top-level region
        # (the function body, or the body an inlined plain function; see
        # ``hir.Defer`` and ``HirRunner._region_defers``)
        self.body_defers: list[_DeferEntry] = []
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


def _append_defers(out: list[mir.BasicBlock], defers: list[_DeferEntry], path: hir.DeferPath) -> None:
    """Append the entries of ``defers`` a transfer that exits along ``path``
    triggers, in the order they run - reverse declaration order (see
    ``hir.DeferPath``): an entry runs when its flags hold the exit's bit."""
    for entry in reversed(defers):
        if entry.flags & path:
            out.append(entry.entry)


def _normal_defers(defers: list[_DeferEntry]) -> tuple[mir.BasicBlock, ...]:
    """The entries of ``defers`` a *normal* exit of the region they were
    declared in triggers (``defer`` and ``okdefer``), in the order they run."""
    out: list[mir.BasicBlock] = []
    _append_defers(out, defers, hir.DeferPath.OK)
    return tuple(out)


def _sval_to_runtime(value: sval.AnyValue, cache: sval.MirLowerCache) -> mir.Value:
    match value:
        case bool():
            return mir.BoolValue(value)
        case sval.Int():
            return mir.Int(value.value, mir.IntType(value.type.bits, value.type.signed))
        case sval.IntEnumValue():
            # an enumeration member: the integer is read off its type, and the
            # MIR type is the smallest integer type the whole enumeration needs
            enum_type = value.type
            int_type = enum_type.int_type
            return mir.Int(enum_type.value_of(value.index), mir.IntType(int_type.bits, int_type.signed))
        case sval.Float():
            return mir.Float(value.value, mir.FloatType(value.type.bits))
        case sval.DeclareFunction():
            # an external function is a global symbol: a function pointer is a
            # reference to it (see ``mir.ExternSymbol`` and ``lower``)
            mir_type = value.get_type().to_mir_type(cache)
            assert isinstance(mir_type, mir.PointerType)
            return mir.ExternSymbol(value.linkname, mir_type)
        case sval.UndefinedScalar():
            # an undefined value has no defined content.  For a pointer it may
            # not be LLVM's ``undef``: null is not a legal value of a non-null
            # ``Ptr[T]``, so an undefined pointer is a dangling (non-null,
            # aligned) address instead (see ``mir.Dangling``)
            mir_type = value.type.to_mir_type(cache)
            if mir_type is None:
                raise _no_runtime_type(value.type)
            if isinstance(value.type, sval.PointerType):
                assert isinstance(mir_type, mir.PointerType)
                return mir.Dangling(mir_type)
            return mir.UndefValue(mir_type)
        case _:
            raise CompileError(f"cannot return the compile-time value {value!r}")


def _flatten_natural(value: sval.AnyValue, type: sval.Type) -> list[sval.ScalarValue]:
    """The natural (unpadded) flattened cells of the compile-time ``value`` of
    ``type`` (see :func:`_flatten_value`)."""
    if isinstance(type, sval.ScalarType):
        return [value]
    if isinstance(value, sval.AggregateValue):
        values = value.values
    else:
        raise CompileError(f'cannot flatten {value!r} as {type}')
    if isinstance(type, sval.StructType):
        cells: list[sval.ScalarValue] = []
        for index, field in enumerate(type.fields().values()):
            if field.is_ref:
                # the cell holds the compile-time pointer itself
                cells.append(values[index])
            else:
                cells.extend(_flatten_value(values[index], field.type))
        return cells
    if isinstance(type, sval.TupleType):
        cells = []
        for index, elem in enumerate(type.types):
            cells.extend(_flatten_value(values[index], elem))
        return cells
    if isinstance(type, sval.StrDictType):
        cells = []
        for index, elem in enumerate(type.values.values()):
            cells.extend(_flatten_value(values[index], elem))
        return cells
    if isinstance(type, sval.ArrayType):
        cells = []
        for elem_value in values:
            cells.extend(_flatten_value(elem_value, type.elem))
        return cells
    if isinstance(type, sval.ComplexType):
        return [values[0], values[1]]
    raise CompileError(f'cannot flatten {value!r} as {type}')


def _flatten_value(value: sval.AnyValue, type: sval.Type) -> list[sval.ScalarValue]:
    """The flattened cell sequence of the compile-time ``value`` of ``type``
    (see ``sval.scalar_leaves``), padded to the type's width with an
    ``UndefinedScalar`` of each remaining leaf type."""
    if isinstance(value, (sval.UndefinedAggregate, sval.UndefinedScalar, sval.UntypedUndefined)):
        # an undefined value flattens to undefined cells (one per leaf)
        return [sval.UndefinedScalar(leaf) for leaf in sval.scalar_leaves(type)]
    if isinstance(type, sval.OptionType):
        if isinstance(value, sval.OptionValue):
            cells: list[sval.ScalarValue] = [True]
            cells.extend(_flatten_value(value.value, type.child))
        elif isinstance(value, (sval.Null, sval.TypedNull)):
            cells = [False]
        else:
            raise CompileError(f'cannot flatten the option value {value!r}')
    elif isinstance(type, sval.TaggedUnionType):
        if not isinstance(value, sval.TaggedUnionValue):
            raise CompileError(f'cannot flatten the tagged union value {value!r}')
        cells = [sval.Int(value.index, type.tag_type())]
        cells.extend(_flatten_value(value.value, type.types[value.index]))
    else:
        cells = _flatten_natural(value, type)
    leaves = sval.scalar_leaves(type)
    while len(cells) < len(leaves):
        cells.append(sval.UndefinedScalar(leaves[len(cells)]))
    if len(cells) != len(leaves):
        raise CompileError(f'cannot flatten {value!r} as {type}')
    return cells


def _unflatten_cells(cells: list[sval.ScalarValue], offset: int, type: sval.Type) -> sval.AnyValue:
    """Reconstruct a compile-time value of ``type`` from the flattened cells at
    ``offset`` (the inverse of :func:`_flatten_value`)."""
    if isinstance(type, sval.ScalarType):
        return cells[offset]
    if isinstance(type, sval.StructType):
        values: list[sval.AnyValue] = []
        for index, field in enumerate(type.fields().values()):
            field_offset = offset + sval.field_scalar_offset(type, index)
            if field.is_ref:
                values.append(cells[field_offset])
            elif field.is_comptime:
                cell = cells[field_offset]
                # an unset compile-time field reads its default
                values.append(field.default if isinstance(cell, sval.UndefinedScalar) else cell)
            else:
                values.append(_unflatten_cells(cells, field_offset, field.type))
        return sval.AggregateValue(tuple(values), type)
    if isinstance(type, sval.TupleType):
        return sval.AggregateValue(
            tuple(
                _unflatten_cells(cells, offset + sval.field_scalar_offset(type, index), elem)
                for index, elem in enumerate(type.types)
            ),
            type,
        )
    if isinstance(type, sval.StrDictType):
        return sval.AggregateValue(
            tuple(
                _unflatten_cells(cells, offset + sval.field_scalar_offset(type, index), elem)
                for index, elem in enumerate(type.values.values())
            ),
            type,
        )
    if isinstance(type, sval.ArrayType):
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot flatten the unsized array {type}')
        step = sval.scalar_count(type.elem)
        return sval.AggregateValue(
            tuple(
                _unflatten_cells(cells, offset + index * step, type.elem)
                for index in range(length)
            ),
            type,
        )
    if isinstance(type, sval.ComplexType):
        return sval.AggregateValue((cells[offset], cells[offset + 1]), type)
    if isinstance(type, sval.OptionType):
        if cells[offset] is True:
            return sval.OptionValue(_unflatten_cells(cells, offset + 1, type.child))
        return sval.TypedNull(type.child)
    if isinstance(type, sval.TaggedUnionType):
        tag = cells[offset]
        index = tag.value if isinstance(tag, sval.Int) else 0
        return sval.TaggedUnionValue(
            type, index, _unflatten_cells(cells, offset + 1, type.types[index]),
        )
    if isinstance(type, sval.UnionType):
        unit = type.get_unit_value()
        if unit is None:
            raise CompileError(f'cannot build a value of the union {type}')
        return unit
    raise CompileError(f'cannot unflatten {type}')


def _fill_comptime_defaults(box: sval.Box, type: sval.Type, offset: int) -> None:
    """Fill the cell of every ``is_comptime`` struct field reachable from
    ``(type, offset)`` with that field's declared default: a box's cells start as
    the flattened undefined, so a compile-time field would otherwise read
    undefined instead of its default (see ``_comptime_field_place``).  Iterative
    (an explicit stack), so type nesting costs no Python stack."""
    stack: list[tuple[sval.Type, int]] = [(type, offset)]
    while stack:
        current, base = stack.pop()
        if isinstance(current, sval.StructType):
            for index, field0 in enumerate(current.fields().values()):
                field_offset = base + sval.field_scalar_offset(current, index)
                if field0.is_comptime:
                    if field0.default is not None:
                        decl = (
                            sval.PointerType(field0.type, is_const=True)
                            if field0.is_ref else field0.type
                        )
                        cells = _flatten_value(field0.default, decl)
                        box.cells[field_offset:field_offset + len(cells)] = cells
                elif not field0.is_ref:
                    stack.append((field0.type, field_offset))
        elif isinstance(current, sval.TupleType):
            for index, elem in enumerate(current.types):
                stack.append((elem, base + sval.field_scalar_offset(current, index)))
        elif isinstance(current, sval.StrDictType):
            for index, elem in enumerate(current.values.values()):
                stack.append((elem, base + sval.field_scalar_offset(current, index)))
        elif isinstance(current, sval.ArrayType):
            length = current.length_int
            if length is None:
                continue
            step = sval.scalar_count(current.elem)
            for index in range(length):
                stack.append((current.elem, base + index * step))

def _load_box_cells(box: sval.Box | bytes, offset: int, type: sval.Type) -> sval.AnyValue:
    """Read a compile-time value of ``type`` out of the compile-time storage
    ``box`` at ``offset`` (a byte-string store holds raw bytes)."""
    if isinstance(box, bytes):
        if isinstance(type, sval.BytesType):
            # a byte-string store holds the bytes themselves
            return box[offset:]
        if isinstance(type, sval.IntType):
            return sval.Int(box[offset], type)
        raise CompileError(f'cannot read {type} out of a byte string')
    return _unflatten_cells(box.cells, offset, type)


def _to_comptime(value: InterpVal) -> sval.AnyValue | None:
    match value:
        case ComptimeVal():
            return value.obj
        case _:
            return None

def _shallow_normalize(ev: InterpVal) -> InterpVal:
    """Follow a committed :class:`PendingSlot` to the pointer value it was
    materialized into (a :class:`RuntimeVal` pointer or a
    :class:`ComptimeVal` wrapping a :class:`sval.ComptimePtr`), so that pointer
    operations see one shape.  An uncommitted slot is left alone for
    ``load``/``store`` to handle."""
    if isinstance(ev, PendingSlot) and ev.committed is not None:
        return ev.committed
    return ev

def _type_of(ev: InterpVal, allow_value_type: bool = False) -> sval.Type | None:
    """The spy type of the value ``ev`` denotes, or None when it has
    no spy representation (an un-typable compile-time object).
    Should work on non-normalized values."""
    match ev:
        case PendingSlot():
            if ev.committed is not None:
                return _type_of(ev.committed, allow_value_type)
            if ev.declared is not None:
                return sval.PointerType(ev.declared, is_const=False)
            return None
        case RuntimeVal(_, type):
            if isinstance(type, sval.ValueType) and not allow_value_type:
                return sval.type_of(type.value)
            return type
        case ComptimeVal(obj):
            # an sval value; a ``ComptimePtr``/``CastedPtr`` reports its pointer
            # type through ``get_type`` (see ``sval.type_of``)
            return sval.type_of(obj) if not allow_value_type else sval.ValueType(obj)
        case _:
            return None

def _enum_member_index(enum_type: sval.IntEnumType, value: sval.AnyValue) -> int:
    """The index of the member of ``enum_type`` the compile-time value ``value``
    denotes: a typed :class:`sval.IntEnumValue` of it, or one of its Python
    members.  Raises when the value is not one of its members."""
    if isinstance(value, sval.IntEnumValue):
        return value.index
    if sval.is_int_enum_member(value):
        index = enum_type.index_of(value.name)
        if index is not None and enum_type.value_of(index) == value.value:
            return index
    raise CompileError(f'{value!r} is not a member of {enum_type}')


def _comptime_int(ev: InterpVal) -> int | None:
    """The Python integer a compile-time integer value denotes, or None when
    the value is not one at all (a runtime value, or no integer)."""
    if isinstance(ev, ComptimeVal):
        if isinstance(ev.obj, sval.Int):
            return ev.obj.value
        if isinstance(ev.obj, int):
            return ev.obj
    return None



def _comptime_bool(ev: InterpVal, what: str = 'a bool') -> bool:
    """The Python bool a compile-time bool *value* denotes (a decision a caller
    has to make while the HIR runs); a runtime value has none, so it is rejected
    (``what`` names it in the error)."""
    ev = _shallow_normalize(ev)
    if isinstance(ev, ComptimeVal) and isinstance(ev.obj, bool):
        return ev.obj
    raise CompileError(f'{what} has to be known at compile time')

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


def _vararg_field(child: ArgNode) -> tuple[sval.Type, sval.AnyValue | None, bool, bool]:
    """The ``(type, default, is_comptime, is_ref)`` a flattened ``tuple``/
    ``dict`` element contributes to the synthesized struct field that carries it
    at the callee (see §8.1): a runtime element its spy type (``is_ref`` mirroring
    how it is passed), a nested ``tuple``/``dict`` the nested synthetic struct,
    and a compile-time value its type with the value baked into the field (an
    ``is_comptime`` field with a default)."""
    if isinstance(child, RuntimeArgNode):
        return child.type, None, False, child.is_ref
    if isinstance(child, (tuple, frozendict)):
        return _vararg_struct_type(child), None, False, False
    type = child.type if isinstance(child, sval.ComptimePtr) else sval.type_of(child)
    if type is None:
        raise CompileError(f'cannot rebuild the compile-time argument {child!r}')
    return type, child, True, False


def _vararg_struct_type(elems: tuple[ArgNode, ...] | frozendict[str, ArgNode]) -> sval.StructType:
    """The synthesized ``StructType`` a flattened ``tuple``/``dict`` argument is
    rebuilt as at the callee (see §8.1): one positional field per element (``_i``)
    or one field per entry (its name).  The head is marked ``$args``/``$kwargs`` so
    a subscript of the struct is read as a tuple/dict element (see
    ``HirRunner.subscript``)."""
    is_named = isinstance(elems, frozendict)
    head = sval.StructTypeHead('$kwargs' if is_named else '$args')
    if is_named:
        items: list[tuple[str, ArgNode]] = list(elems.items())
    else:
        items = [(f'_{index}', child) for index, child in enumerate(elems)]
    for name, child in items:
        field_type, default, is_comptime, is_ref = _vararg_field(child)
        head.add_field(name, field_type, default, is_comptime, is_ref)
    return head.specialize(())


def _arg_node_types(node: ArgNode) -> Iterator[sval.Type]:
    """The spy type of every runtime leaf of a provided :data:`ArgNode` (and the
    type of a compile-time leaf), in order - what a check over the arguments of
    a call reads (see ``_call_function_entry``)."""
    if isinstance(node, RuntimeArgNode):
        yield node.type
    elif isinstance(node, tuple):
        for child in node:
            yield from _arg_node_types(child)
    elif isinstance(node, frozendict):
        for child in node.values():
            yield from _arg_node_types(child)
    elif isinstance(node, CompoundArgNode):
        for child in node.elems:
            yield from _arg_node_types(child)
    else:
        type = sval.type_of(node)
        if type is not None:
            yield type


def _arg_node_needs_mir(node: ArgNode) -> bool:
    """Whether the argument tree ``node`` consumes a MIR argument: it holds a
    runtime leaf anywhere (a ``RuntimeArgNode``).  It is what a caller that can
    only take a runtime-pointer-free argument reads (see ``exec_as_func_ptr``)."""
    if isinstance(node, RuntimeArgNode):
        return True
    if isinstance(node, tuple):
        return any(_arg_node_needs_mir(child) for child in node)
    if isinstance(node, frozendict):
        return any(_arg_node_needs_mir(child) for child in node.values())
    if isinstance(node, CompoundArgNode):
        return any(_arg_node_needs_mir(child) for child in node.elems)
    return False


def _union_variant_type(union_type: sval.TaggedUnionType, tag: InterpVal) -> sval.Type:
    """The variant type the tag value ``tag`` selects in ``union_type`` - or the
    payload union storage type when the tag is only known at runtime."""
    obj = _to_comptime(_shallow_normalize(tag))
    if isinstance(obj, sval.Int):
        return union_type.types[obj.value]
    return union_type.payload_type()


def _provided_node_of_type(type: sval.Type) -> ArgNode:
    """A provided :data:`ArgNode` for every leaf of ``type`` (see
    ``Signature.solve_param_types``): a compile-time-only aggregate is provided
    element by element, any other type as one runtime argument.  It is what a
    call that specializes a function pointer without passing values uses (see
    ``exec_as_func_ptr``)."""
    if isinstance(type, sval.TupleType):
        return tuple(_provided_node_of_type(child) for child in type.types)
    if isinstance(type, sval.StrDictType):
        return frozendict((key, _provided_node_of_type(child)) for key, child in type.values.items())
    return RuntimeArgNode(type, False)


# ---------------------------------------------------------------------------
# stateless helpers of field/element access, struct/array construction and the
# result location of a function that returns several values: pure functions
# over the interpreter values they are given (the struct/array structure is
# read off those, not off any runner state)
# ---------------------------------------------------------------------------


def _index_value(index: int) -> InterpVal:
    """A compile-time field/element index as an interpreter value."""
    return ComptimeVal(sval.Int(index, sval.IntType(64, False)))

def _try_comptime_index(index: InterpVal) -> int | None:
    """The Python integer a compile-time index denotes, or None when the index
    is a runtime value (an array element may be addressed by a runtime one, see
    ``field_index_addr``)."""
    if isinstance(index, ComptimeVal) and isinstance(index.obj, sval.Int):
        return index.obj.value
    return None

def _comptime_index(index: InterpVal) -> int:
    """The Python integer a compile-time index denotes (a field or element of
    an aggregate held compile-time is always named by a compile-time index)."""
    result = _try_comptime_index(index)
    if result is None:
        raise CompileError(
            'cannot pick a field or element with a runtime index: it has to be a '
            'compile-time integer'
        )
    return result

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
                if isinstance(obj, sval.ComptimePtr):
                    # a callee boxed by ``_as_ref`` (a struct template, a builtin
                    # kept in a compile-time variable): read it back out of its box
                    todo.append(ComptimeVal(_load_box_cells(obj.box, obj.offset, obj.type.elem)))
                    continue
                return obj
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

def _tuple_places_type(places: tuple[InterpVal, ...]) -> sval.Type:
    # the type of a tuple of element places (see ``init_tuple``): a tuple has
    # no representation of its own, so this is the tuple of the types of the
    # places themselves
    types: list[sval.Type] = []
    for place in places:
        type = _place_type(place)
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


def _flatten_result_tree(tree: InterpVal | tuple) -> tuple[InterpVal, ...]:
    """The leaf places of a (nested) result-location tree, in depth-first order."""
    out: list[InterpVal] = []
    work: list[InterpVal | tuple] = [tree]
    while work:
        item = work.pop()
        if isinstance(item, tuple):
            work.extend(reversed(item))
        else:
            out.append(item)
    return tuple(out)

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
    if type.classify() == sval.SpecialTypeKind.DST:
        return CompileError(
            f'{type} is a dynamically-sized type: only a pointer to it is a value'
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

def _truncate_div(a: int, b: int) -> int:
    """Integer division truncated towards zero (C's ``/``), for the compile-time
    fold of ``//`` when ``CompileVars.int_trunc_div`` is set."""
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient

def _convert_inst(
    value: mir.Value, from_type: sval.Type, to_type: sval.Type, cache: sval.MirLowerCache,
) -> mir.Inst | None:
    """Build (but do not emit) the conversion of ``value`` from
    ``from_type`` to ``to_type``; returns ``None`` when no conversion
    instruction is needed (the types are equal, or one pointer converts to
    the other - see ``sval.PointerType.is_subtype_of``)."""
    if from_type == to_type:
        return None
    if isinstance(from_type, sval.IntEnumType) or isinstance(to_type, sval.IntEnumType):
        # an enumeration is not an integer: it converts to neither an integer nor
        # another enumeration (only a value of the very same enumeration is one)
        raise CoerceError(
            f'cannot convert a {from_type} value to {to_type}: an enumeration is '
            f'not an integer'
        )
    if isinstance(from_type, sval.UnionType) and isinstance(to_type, sval.UnionType):
        # a union value cannot be converted: its storage has to be
        # reinterpreted through a pointer instead (see ``_convert_result_ptr``),
        # so only a store of the very same union is expressible
        raise CoerceError(
            f'cannot convert a {from_type} value to {to_type}: the storage of a '
            f'union is written through a pointer to it'
        )
    mir_to_type = to_type.to_mir_type(cache)
    if mir_to_type is None or to_type.is_zst():
        # a dynamically-sized or zero-sized target has no value to convert to
        # (and a dynamically-sized one is not lowerable on its own)
        raise CoerceError(f'cannot convert a {from_type} value to {to_type}')
    if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.IntType):
        if from_type.bits == to_type.bits:
            # same width: only the signedness differs, which is a compile-time
            # notion with no runtime effect (both are the same ``i{bits}``), so
            # the value is already the one the target names
            return None
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
        # the address itself is what converts: the two are the same value
        # whenever the pointers convert at all, so what is left is to reject
        # the pairs that do not - dropping the constness, changing the pointee
        # type and *adding* the multi form, which indexing needs (see
        # ``sval.PointerType.is_subtype_of``)
        if not from_type.is_subtype_of(to_type):
            raise CoerceError(
                f'cannot convert a {from_type} value to {to_type}'
            )
        return None
    raise CoerceError(
        f"cannot convert a {from_type} value to {to_type}"
    )

_FLOAT_STRUCT_FMT = {32: '=f', 64: '=d'}

def _scalar_bit_width(t: sval.Type) -> int | None:
    """The storage width in bits of a scalar type ``std.core.bitcast`` accepts
    (a boolean is a one-bit integer), or None for every other type."""
    match t:
        case sval.BoolType():
            return 1
        case sval.IntType():
            return t.bits
        case sval.FloatType():
            return t.bits
        case _:
            return None

def _scalar_to_bits(obj: sval.AnyValue, t: sval.Type) -> int:
    """The integer bit pattern of the scalar compile-time constant ``obj`` of
    the scalar type ``t`` (see ``HirRunner._builtin_bitcast``)."""
    match t:
        case sval.BoolType():
            assert isinstance(obj, bool), f'expected a bool constant, got {obj!r}'
            return 1 if obj else 0
        case sval.IntType():
            assert isinstance(obj, sval.Int), f'expected an integer constant, got {obj!r}'
            return obj.value & ((1 << t.bits) - 1) if t.bits > 0 else 0
        case sval.FloatType():
            assert isinstance(obj, sval.Float), f'expected a float constant, got {obj!r}'
            return int.from_bytes(struct.pack(_FLOAT_STRUCT_FMT[t.bits], obj.value), sys.byteorder)
        case _:
            raise AssertionError(f'{t} is not a scalar type')

def _scalar_of_bits(bits: int, t: sval.Type) -> sval.AnyValue:
    """The scalar compile-time constant of the scalar type ``t`` whose bit
    pattern is ``bits`` (the inverse of ``_scalar_to_bits``)."""
    match t:
        case sval.BoolType():
            return bool(bits & 1)
        case sval.IntType():
            if t.bits == 0:
                return sval.Int(0, t)
            if t.signed and (bits >> (t.bits - 1)) & 1:
                bits -= 1 << t.bits
            return sval.Int(bits, t)
        case sval.FloatType():
            data = bits.to_bytes(t.bits // 8, sys.byteorder)
            return sval.Float(struct.unpack(_FLOAT_STRUCT_FMT[t.bits], data)[0], t)
        case _:
            raise AssertionError(f'{t} is not a scalar type')

def _check_comptime_args(
    sig: Signature, args: ArgList[ArgEntry[InterpVal]]
) -> None:
    """Require a *deeply* compile-time value for every parameter declared
    compile-time (``Comptime``/``Comptime[T]``): the callee reads such a
    parameter as a compile-time value whatever it is given, so a runtime
    argument would be silently dropped (see ``RuntimeArgNode``/
    ``Signature.specialize``).
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
        # location has been materialized, together with the deferred bodies the
        # return runs first
        self._deferred_returns: list[tuple[mir.Insertion, tuple[mir.BasicBlock, ...]]] = []
        # the function-wide block a panic resumes in after a ``CallMayPanic``'s
        # unwind edge ran its deferred bodies (see ``_resume_block``); None until
        # the first may-panic call needs it
        self._unwind_resume_block: mir.BasicBlock | None = None
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

    @property
    def _special_type(self) -> sval.SpecialTypes:
        """The ``std`` types the type rules need, of the host the body is
        compiled for (the host caches them, see ``sval.SpecialTypes``)."""
        return self._analyser._resolver.special_types()

    def _usize_type(self) -> sval.IntType:
        """``usize``: the unsigned integer of the target's pointer width - the
        type of a length, an index and a ``SlicePtr``'s own length."""
        return sval.IntType(self._analyser._resolver.target_info().usize_bits, False)

    def _isize_type(self) -> sval.IntType:
        """``isize``: the signed integer of the target's pointer width - the
        type of a pointer offset, so that a negative one walks backwards."""
        return sval.IntType(self._analyser._resolver.target_info().usize_bits, True)

    @property
    def _compile_vars(self) -> CompileVars:
        """The compile-time configuration of the innermost executing body (the
        top of its ``compile_vars_stack``, see :class:`InlineFrame`)."""
        return self._frames[-1].compile_vars_stack[-1]

    def _inherit_compile_vars(self) -> CompileVars:
        """A copy of the current configuration, for a body about to be inlined:
        an inlined callee starts from its caller's configuration."""
        vars = self._compile_vars
        return CompileVars(vars.int_div_type, vars.int_trunc_div, vars.max_exp_unroll)

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
        self._unwind_resume_block = None
        if ret_sig is not None and ret_sig.exceptions is not None:
            # a declared exception set is the function's own from the start: its
            # order fixes the error codes, whatever the value type turns out to be
            for exception in ret_sig.exceptions.values:
                self._error_types.add(exception)
        frame = InlineFrame(
            generic_var_values, (), self._reserve_result_loc(ret_sig), body,
            self._fn_instance.mir.name_base,
        )
        self._frames.append(frame)
        mir_args = self._fn_instance.mir.args
        args = self._init_args_from_signature(sig, mir_args, arg_is_ref)
        # a closure's captures are its anonymous struct's fields now (§9): a
        # body reached from a call has no separate capture places
        frame.closure_values = ()
        for arg in mir_args:
            assert arg is not None
        frame.arg_values = args
        if ret_sig is not None and ret_sig.is_complete():
            # the whole return convention is declared: fix it before the body
            self._materialize_ret_sig(ret_sig.complete())

    def _reserve_result_loc(self, ret_sig: PartialReturnSignature | None) -> ComptimeResultPtr:
        """The result location of the function proper (see ``ComptimeResultPtr``):
        the place its declared result is delivered into and its error places.
        The places are the ones the effective spec names when the signature
        declares the whole return convention, and fresh slots otherwise - an
        inferred error place has no width until its set is known (see
        ``_materialize_ret_sig``)."""
        declared_value = ret_sig.ret_type_spec if ret_sig is not None else None
        if ret_sig is not None and ret_sig.is_complete():
            # the value and the error part are both declared: the result
            # location's value is the one the effective spec names (the error
            # places are committed later, see ``_commit_error_space``)
            spec = ret_sig.complete().ret_spec(self._mir_cache)
            assert isinstance(spec, RetTuple) and isinstance(spec.type, sval.ResultType)
            value_loc = self._ret_spec_place(spec.values[0])
            return ComptimeResultPtr(
                value_loc, self.alloca(False), self.alloca(False),
            )
        if declared_value is not None:
            value_loc = self._ret_spec_place(declared_value)
        else:
            # the value type is inferred from the body: the result location has
            # to be able to hold a *tuple* of places (a multi-value return, see
            # ``init_tuple``/``_deliver_packed_tuple``); its storage is decided
            # by the stores
            value_loc = self.alloca()
        return ComptimeResultPtr(
            value_loc, self.alloca(False), self.alloca(False),
        )

    def _init_arg_node(self, node: ArgNode, mir_args: list[mir.Type], arg_is_ref: bool) -> InterpVal:
        """The place one specialized formal argument is bound to: one form of
        the argument tree (see :class:`~spy.compiler.fn.ArgNode`).  A
        compile-time leaf is a const box holding its value and consumes no MIR
        argument; a runtime leaf consumes one and becomes a slot (or the
        argument itself when it is passed by address); a ``tuple``/``dict`` is
        rebuilt as the synthesized struct of its elements (see §8.1)."""
        match node:
            case RuntimeArgNode():
                index = len(mir_args)
                if node.is_ref:
                    # the signature passes the address of the value as a const
                    # pointer: the pointer's own mirror is always there, even
                    # when the pointee is dynamically sized (an opaque type or
                    # an unsized array has no mirror of its own)
                    arg_sval = sval.PointerType(node.type, True)
                    arg_mir = arg_sval.to_mir_type(self._mir_cache)
                    if arg_mir is None:
                        raise _no_runtime_type(node.type)
                else:
                    mir_type = node.type.to_mir_type(self._mir_cache)
                    # ZSTs are never present in RuntimeArgNode
                    assert not node.type.is_zst()
                    if mir_type is None:
                        raise _no_runtime_type(node.type)
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
            case tuple() | frozendict():
                # a flattened ``*args``/``**kwargs``: the callee rebuilds it as a
                # synthesized struct, one field per element/entry (see §8.1)
                return self._build_vararg_struct(node, mir_args)
            case _:
                # a compile-time leaf (an ``sval.Value`` or a plain Python
                # scalar): a const box holding the value, which is its place (no
                # MIR argument is consumed)
                type = node.type if isinstance(node, sval.ComptimePtr) else sval.type_of(node)
                if type is None:
                    raise CompileError(
                        f'cannot determine the type of the compile-time argument {node!r}'
                    )
                return self._const_box_place(node, type)

    def _vararg_struct_from_entries(
        self, entries: tuple[ArgEntry[InterpVal], ...] | frozendict[str, ArgEntry[InterpVal]]
    ) -> InterpVal:
        """The synthesized struct an *inlined* callee's ``*args``/``**kwargs`` is
        bound to, from the already-evaluated argument places (the mirror of
        ``_build_vararg_struct``, see §8.1)."""
        is_named = isinstance(entries, frozendict)
        items: list[tuple[str, ArgEntry[InterpVal]]] = (
            list(entries.items()) if is_named
            else [(f'_{index}', entry) for index, entry in enumerate(entries)]
        )
        head = sval.StructTypeHead('$kwargs' if is_named else '$args')
        for name, entry in items:
            if entry.is_ref:
                head.add_field(name, _arg_type_of(entry), is_ref=True)
                continue
            obj = _to_comptime(entry.value)
            if obj is not None:
                type = sval.type_of(obj)
                if type is None:
                    raise CompileError(f'cannot rebuild the compile-time argument {obj!r}')
                head.add_field(name, type, default=obj, is_comptime=True)
                continue
            type = _type_of(entry.value)
            if type is None:
                raise CompileError('cannot determine the type of an argument of a call')
            head.add_field(name, type)
        struct_type = head.specialize(())
        is_comptime = struct_type.classify() != sval.SpecialTypeKind.NONE
        place = self._aggregate_place(struct_type, is_comptime=is_comptime)
        for index, (_name, entry) in enumerate(items):
            field0 = struct_type.fields().get_by_id(index)
            if field0.is_comptime:
                continue
            if field0.is_ref:
                self.store(self._field_storage_addr(place, struct_type, index), entry.value)
            else:
                value = self.load(entry.value) if entry.is_ref else entry.value
                self.store(self.field_index_addr(place, _index_value(index)), value)
        return place

    def _build_vararg_struct(
        self, elems: tuple[ArgNode, ...] | frozendict[str, ArgNode], mir_args: list[mir.Type]
    ) -> InterpVal:
        """The place a flattened ``*args``/``**kwargs`` is bound to at the callee:
        a single synthesized struct (see §8.1), its fields written from the MIR
        arguments the elements consumed (one ``alloca`` plus recursive writes)."""
        struct_type = _vararg_struct_type(elems)
        is_comptime = struct_type.classify() != sval.SpecialTypeKind.NONE
        place = self._aggregate_place(struct_type, is_comptime=is_comptime)
        if isinstance(elems, frozendict):
            items: list[tuple[str, ArgNode]] = list(elems.items())
        else:
            items = [(f'_{index}', child) for index, child in enumerate(elems)]
        for index, (_name, child) in enumerate(items):
            self._write_vararg_field(place, struct_type, index, child, mir_args)
        return place

    def _write_vararg_field(
        self, place: InterpVal, struct_type: sval.StructType, index: int, child: ArgNode, mir_args: list[mir.Type]
    ) -> None:
        """Write one field of the synthesized varargs struct ``place`` from the
        element ``child``: a compile-time value is baked into the field's type (no
        write), a ``is_ref`` element writes the pointer itself into the field's
        storage, and any other runtime element writes the value the MIR argument
        delivers."""
        field0 = struct_type.fields().get_by_id(index)
        if field0.is_comptime:
            return
        if isinstance(child, (tuple, frozendict)):
            # a nested ``tuple``/``dict``: embedded by value, so its fields are
            # written at their offset in the outer field
            assert isinstance(field0.type, sval.StructType)
            field_place = self.field_index_addr(place, _index_value(index))
            if isinstance(child, frozendict):
                items: list[tuple[str, ArgNode]] = list(child.items())
            else:
                items = [(f'_{sub_index}', sub) for sub_index, sub in enumerate(child)]
            for sub_index, (_name, sub) in enumerate(items):
                self._write_vararg_field(field_place, field0.type, sub_index, sub, mir_args)
            return
        if field0.is_ref:
            pointer = self._init_arg_node(child, mir_args, True)
            self.store(self._field_storage_addr(place, struct_type, index), pointer)
            return
        src = self._init_arg_node(child, mir_args, False)
        self.store(self.field_index_addr(place, _index_value(index)), self.load(src))

    def _field_storage_addr(self, place: InterpVal, struct_type: sval.StructType, index: int) -> InterpVal:
        """The storage of the ``index``-th field of the struct place ``place``,
        typed as the field's own storage (``*const T`` for an ``is_ref`` field) -
        what a construction writes the field's value through, without the extra
        load an ``is_ref`` read takes (see ``field_index_addr``)."""
        place = _shallow_normalize(place)
        ptype = _type_of(place)
        assert isinstance(ptype, sval.PointerType)
        field0 = struct_type.fields().get_by_id(index)
        base_type = (
            sval.PointerType(sval.PointerType(field0.type, is_const=True), ptype.is_const)
            if field0.is_ref else sval.PointerType(field0.type, ptype.is_const)
        )
        if isinstance(place, ComptimeVal) and isinstance(place.obj, sval.ComptimePtr):
            offset = place.obj.offset + sval.field_scalar_offset(struct_type, index)
            return ComptimeVal(sval.ComptimePtr(place.obj.box, offset, base_type))
        if isinstance(place, RuntimeVal):
            mir_index = struct_type.get_field_mir_indices(self._mir_cache)[index]
            return RuntimeVal(self._emit(mir.Gep(place.value, mir_index)), base_type)
        raise CompileError(f'cannot take the field storage of {place!r}')

    def _init_args_from_signature(
        self,
        signature: CallSignature,
        mir_args: list[mir.Type],
        arg_is_ref: tuple[bool, ...],
    ) -> tuple[InterpVal, ...]:
        arg_values: list[InterpVal] = []

        for (_, arg), is_ref in zip(signature.positional, arg_is_ref):
            arg_values.append(self._init_arg_node(arg, mir_args, is_ref))

        if signature.varargs is not None:
            arg_values.append(self._build_vararg_struct(signature.varargs, mir_args))
        if signature.kwargs is not None:
            arg_values.append(self._build_vararg_struct(signature.kwargs, mir_args))

        return tuple(arg_values)

    def _ret_spec_place(self, node: RetSpec) -> InterpVal | tuple:
        """The result place the *value* part ``node`` of a declared result is
        delivered into: a fresh slot for a value (a compile-time one for a
        ``RetComptime`` leaf, see §7.3), a (nested) Python tuple of the places of
        its elements for a group.  A result-type group is not a value part - the
        function's result location is built around it (see
        ``_reserve_result_loc``/``ComptimeResultPtr``)."""
        match node:
            case RetComptime():
                return self.alloca(True)
            case RetValue():
                return self.alloca(False)
            case RetTuple(type=type, values=values):
                assert not isinstance(type, sval.ResultType)
                return tuple(self._ret_spec_place(child) for child in values)
        raise CompileError(f'cannot reserve a result place for {node!r}')

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
        if sig.callconv != 'default':
            # a C function may return only one value and may not raise (the
            # declared parts were checked when the signature was specialized;
            # an inferred one is checked here, once its type is known)
            if isinstance(sig.ret_type_spec.type, sval.TupleType):
                raise CompileError(
                    'a non-default-callconv function may return only one value'
                )
            if len(sig.exceptions) > 0:
                raise CompileError('a non-default-callconv function may not raise')
        spec = sig.ret_spec(self._mir_cache)
        if self.ret_sig is not None:
            if self.ret_sig != sig:
                raise CompileError(
                    f"function returns values of conflicting types "
                    f"{[leaf.type for leaf in iter_ret_leaves(self.ret_sig.ret_spec(self._mir_cache))]} and "
                    f"{[leaf.type for leaf in iter_ret_leaves(spec)]}"
                )
            return
        self.ret_sig = sig
        self._fn_instance.mir.ret_type = mir.VOID
        ret_loc = self._current_result_loc()
        # the effective spec's root is always the result-type group: its value is
        # committed through the places ``_ret_spec_place`` reserved, its error
        # code and payload union are the result location's own (see
        # ``_commit_error_space``)
        assert isinstance(spec, RetTuple) and isinstance(spec.type, sval.ResultType)
        value_spec, code_spec, payload_spec = spec.values
        self._commit_ret_places(value_spec, ret_loc.value)
        code_ptr = self._ret_leaf_ptr(code_spec)
        payload_ptr = self._ret_leaf_ptr(payload_spec)
        self._commit_error_space(ret_loc, spec.type, code_ptr, payload_ptr)
        if sig.is_noreturn():
            # the function has no value to return and raises nothing: no path
            # of it can ever come back, which its lowered form says outright
            self._fn_instance.mir.ret_type = mir.NORETURN

    def _commit_ret_places(self, node: RetSpec, place: InterpVal) -> None:
        """Commit the value place(s) mirroring the spec node ``node``: a leaf
        place is committed (through a hidden result pointer when the leaf says
        so), a group by pairing it with the place tree the caller reserved for
        it.  The root result-type group is handled by ``_materialize_ret_sig``,
        so ``node`` is never one here."""
        match node:
            case RetComptime():
                # a compile-time result leaf: the place is a compile-time box of
                # the leaf's type, committed so its value can be read back (§7.3)
                self._commit_pending_slot(place, node.type)
            case RetValue():
                self._commit_pending_slot(place, node.type, ptr=self._ret_leaf_ptr(node))
            case RetTuple(type=type, values=values):
                assert not isinstance(type, sval.ResultType)
                # an inferred multi-value result location is still an uncommitted
                # slot typed by the first delivery as a tuple of element places
                # (a ``_PendingTuple``, see ``_store_return_loc``): its tree *is*
                # those places
                pending = self._pending_tuple(place) if isinstance(place, PendingSlot) else None
                if pending is not None:
                    place = pending.places
                assert isinstance(place, tuple)
                for child, sub in zip(values, place):
                    self._commit_ret_places(child, sub)

    def _ret_leaf_ptr(self, leaf: RetSpec) -> mir.Value | None:
        """The hidden result pointer a leaf is delivered through (appending the
        formal to the lowered signature), or None when the leaf is returned by
        value (which fixes the MIR return type).  A dynamically-sized leaf has
        no value of its own: it is always delivered through the result pointer,
        the pointer to it (a ``void*`` for an opaque type, a plain pointer to
        the elements of an unsized array)."""
        assert isinstance(leaf, RetValue)
        mir_fn = self._fn_instance.mir
        if leaf.via_result_ptr:
            if leaf.type.is_zst():
                raise CompileError(f'cannot return {leaf.type} through a result pointer')
            # the pointer to the value: for a dynamically-sized type that is the
            # only form a value of it has (see ``sval.PointerType.to_mir_type``)
            ptr_type = sval.PointerType(leaf.type, is_const=False).to_mir_type(self._mir_cache)
            if ptr_type is None:
                raise CompileError(f'cannot return {leaf.type} through a result pointer')
            index = len(mir_fn.args)
            mir_fn.args.append(ptr_type)
            mir_fn.arg_names.append('$result')
            return mir.Param(index, ptr_type)
        if not leaf.type.is_copyable():
            # a non-copyable leaf is only ever delivered through the result
            # pointer: reaching here means a convention forced it by value
            raise CompileError(
                f'cannot return a value of the non-copyable type {leaf.type} by value'
            )
        if not leaf.type.is_zst():
            # a zero-sized result is delivered as its unit value; only a
            # result with storage fixes the MIR return type
            mir_ret = leaf.type.to_mir_type(self._mir_cache)
            if mir_ret is None:
                raise _no_runtime_type(leaf.type)
            mir_fn.ret_type = mir_ret
        return None

    # -- return statements ------------------------------------------------

    def _current_result_loc(self) -> ComptimeResultPtr:
        assert len(self._frames) > 0, 'no function result location'
        ret = self._frames[-1].ret_loc
        assert isinstance(ret, ComptimeResultPtr)
        return ret

    def _result_loc(self) -> InterpVal:
        """The place the current frame's result is delivered into: the value
        part of the frame's result location (its error places are separate)."""
        return self._current_result_loc().value

    def _function_result(self) -> ComptimeResultPtr:
        """The function proper's own result location (the frame at the bottom of
        the stack: an inlined body's result location shares its error places)."""
        ret = self._frames[0].ret_loc
        assert isinstance(ret, ComptimeResultPtr)
        return ret

    def _result_places(self, location: ComptimeResultPtr) -> tuple[InterpVal, ...]:
        """The leaf places one result value each is delivered into, in depth-first
        declaration order: the result location's value may itself be a tree of
        slots (a Python tuple, nested exactly like the results - see §7.2 -
        followed by the error code and the payload union, see ``sval.make_ret_spec``).
        An untyped multi-value location (a ``_PendingTuple``) spreads into its
        element places."""
        places: list[InterpVal] = []
        # the result spreads into its value, its error code and its payload, in
        # that order (the stack is popped last-in-first-out, so it is pushed in
        # reverse)
        work: list[object] = [location.payload, location.code, location.value]
        while work:
            place = work.pop()
            if isinstance(place, tuple):
                work.extend(reversed(place))
                continue
            place = _shallow_normalize(place)
            if isinstance(place, PendingSlot):
                recorded = self._pending_tuple(place)
                if recorded is not None:
                    work.extend(reversed(recorded.places))
                    continue
            places.append(place)
        return tuple(places)

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

    def _defer_return(self, defer_blocks: tuple[mir.BasicBlock, ...] = ()) -> None:
        """End a path whose return convention is not fixed yet: a placeholder
        that ``_finish_function`` fills with the ``mir.Ret`` once the effective
        spec is known (see ``hir.Ret``/``hir.Raise``).  The placeholder is not
        a terminator, so ``emit`` cannot end the block on its own - the path
        ends here all the same.  ``defer_blocks`` are the deferred bodies the
        return runs first."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        self._cur_block.is_finished = True
        self._deferred_returns.append((insertion, defer_blocks))

    def _emit_function_return(self, defer_blocks: tuple[mir.BasicBlock, ...] = ()) -> None:
        """Emit the ``mir.Ret`` that ends one path of the *function proper*
        (whatever inline frame the path sits in): its by-value result is
        loaded out of the function's result location, or none is returned when
        every result goes through a result pointer.  ``defer_blocks`` are the
        deferred bodies the return runs first."""
        sig = self.ret_sig
        assert sig is not None
        spec = sig.ret_spec(self._mir_cache)
        places = self._result_places(self._frames[0].ret_loc)
        # record the compile-time result leaves (see §7.3): each ``RetComptime``
        # leaf's place holds the value this path returns - the last path's wins
        for leaf, place in zip(iter_ret_leaves(spec), places):
            if not isinstance(leaf, RetComptime):
                continue
            self._record_ret_comptime_value(leaf, place)
        index = ret_by_value_index(spec)
        if index is None:
            self._cur_block.emit(mir.Ret(None, defer_blocks))
        else:
            self._cur_block.emit(mir.Ret(self._to_runtime(self.load(places[index])), defer_blocks))

    def _record_ret_comptime_value(self, leaf: RetComptime, place: InterpVal) -> None:
        """Record the value a ``RetComptime`` leaf holds at a return site - the
        path's own return (``_emit_function_return``), or the deferred return
        ``_finish_function`` fills in.  The callee's per-``FunctionInstance`` list
        is what a non-inline caller reads to deliver the value (see §7.3); the
        last path's value wins."""
        value = _to_comptime(self.load(place))
        values = self._fn_instance.ret_comptime_values
        if values is None:
            values = self._fn_instance.ret_comptime_values = []
        while len(values) <= leaf.index:
            values.append(None)
        values[leaf.index] = value

    def _end_error_path(self, defer_blocks: tuple[mir.BasicBlock, ...] = ()) -> None:
        """End a path at the function boundary (an error escaping the function):
        a typed return, or a deferred one when the convention is not fixed yet.
        ``defer_blocks`` are the deferred bodies the error exit runs first."""
        if self.ret_sig is None:
            self._defer_return(defer_blocks)
        else:
            self._emit_function_return(defer_blocks)

    def _raise(self, slot: InterpVal) -> PollResult:
        """End one path with ``hir.Raise``: the exception has been built into
        the slot ``slot``.  It is dispatched straight to the clause that catches
        its type when there is one, or written into the function's error
        location and returned otherwise; the enclosing HIR blocks are then
        unwound like a ``return``'s (see ``_cut``)."""
        self._dispatch_exception(slot)
        return self._cut()

    def _dispatch_exception(self, slot: InterpVal) -> None:
        """Route an exception built into the slot ``slot`` to the clause that
        catches its type, or into the function's own error location (see
        ``_raise``).  Ends the current block either way but does not unwind the
        interpreter's open regions: the ``catch_unwind`` builtin dispatches its
        caught exception from a detached catch block and then continues in the
        normal one (see ``_make_runtime_call``)."""
        exception = _place_type(slot)
        if not isinstance(exception, sval.StructType):
            raise CompileError(f'cannot raise {exception!r}: an exception must be a struct')
        target = self._find_catching_clause(exception)
        if target is not None:
            data, index = target
            defer_blocks = self._collect_exit_defers(
                hir.DeferPath.RAISE, data, inclusive=True, current_frame_only=False,
            )
            if exception.is_zst():
                incoming = ComptimeVal(sval.UndefinedScalar(sval.PointerType(exception, is_const=False)))
            else:
                self._commit_pending_slot(slot, exception)
                incoming = _shallow_normalize(slot)
            self._route_to_clause(data, index, incoming, exception, defer_blocks)
        else:
            defer_blocks = self._collect_exit_defers(
                hir.DeferPath.RAISE, None, inclusive=False, current_frame_only=False,
            )
            self._add_function_exception(exception)
            self._defer_error_code_write(exception)
            if exception.is_zst():
                self._commit_pending_slot(slot, exception)
            else:
                dest = self._payload_variant_ptr(exception)
                assert isinstance(dest, RuntimeVal)
                self._commit_pending_slot(slot, exception, ptr=dest.value)
            self._end_error_path(defer_blocks)

    def _build_catch_path(self, catch_data: InterpVal) -> None:
        """Build the catch block of a ``catch_unwind`` (see ``mir.CatchUnwind``):
        the landing pad has written the caught ``PanicData`` pointer into the
        place ``catch_data`` points at, so it is read back and wrapped in an
        ``UnwindException`` through the ordinary construction path - the value is
        materialized with ``_to_runtime`` from an aggregate whose mirror is its
        single ``data`` field (see ``sval.StructType.mirror_is_a_field``) - and
        then dispatched as a raise.  The current block is the catch block when
        this runs."""
        exception = self._unwind_exception_type()
        pointer = self.load(catch_data)          # the ``Ptr[PanicData]`` caught
        data = self.load(pointer)                # the caught ``PanicData``
        slot = self.alloca(False)
        self._commit_pending_slot(slot, exception)
        self.store(self.field_index_addr(slot, _index_value(0)), data)
        self._dispatch_exception(slot)

    # -- error locations ---------------------------------------------------

    def _commit_error_space(self, result: ComptimeResultPtr, type: sval.ResultType, code_ptr: mir.Value | None = None, payload_ptr: mir.Value | None = None) -> None:
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
                    f'cannot raise it (declare it with @func(exceptions=(...)) '
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

    def _union_variant_ptr(self, place: InterpVal, struct_type: sval.Type) -> InterpVal:
        """The address a value of the union variant ``struct_type`` is written
        to (or read from) in the payload union the place ``place`` points at:
        the place reinterpreted as a pointer to the variant type."""
        if struct_type.is_zst():
            return ComptimeVal(sval.UndefinedScalar(sval.PointerType(struct_type, is_const=False)))
        place = _shallow_normalize(place)
        ptr_type = _type_of(place)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'cannot take a variant of {place!r}')
        mir_struct = struct_type.to_mir_type(self._mir_cache)
        assert mir_struct is not None and not struct_type.is_zst()
        bitcast = self._emit(mir.BitCast(self._to_runtime(place), mir.PointerType(mir_struct)))
        return RuntimeVal(bitcast, sval.PointerType(struct_type, is_const=False))


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

    def _route_to_clause(self, data: TryExceptBlockData, index: int, incoming: InterpVal, exception: sval.StructType, defer_blocks: tuple[mir.BasicBlock, ...] = ()) -> None:
        """Route one caught error to clause ``index``: create the clause's entry
        block - and its error-payload ``Phi`` - on the first dispatch that
        reaches it, then have the current block jump to it with ``incoming`` (the
        address of the caught exception) as its payload pointer.  A bare clause's
        union - and so its phi types - is only fixed once the whole try body has
        been walked, so its dispatch is deferred through a case block.
        ``defer_blocks`` are the deferred bodies the error exit runs first."""
        block = data.clause_blocks[index]
        if block is None:
            block = mir.BasicBlock()
            data.clause_blocks[index] = block
        if self._except_struct_type(data.except_types[index]) is None:
            case_block = mir.BasicBlock()
            self._cur_block.emit(mir.Jmp(case_block, defer_blocks))
            data.bare_pending.append((index, case_block, incoming, exception))
            return
        if exception.is_zst():
            # a zero-sized exception has no payload: the clause is entered
            # without a payload pointer (its bind is the unit value)
            self._cur_block.emit(mir.Jmp(block, defer_blocks))
            return
        value = self._to_runtime(incoming)
        phi = data.payload_phi[index]
        if phi is None:
            phi = mir.Phi([(value, self._cur_block)])
            data.payload_phi[index] = phi
            block.emit(phi)
        else:
            phi.add_incoming(value, self._cur_block)
        self._cur_block.emit(mir.Jmp(block, defer_blocks))


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
            # the path fell off the end of the body: it joins the caller, after
            # running the defers of the body's top-level region
            self._cur_block.emit(mir.Jmp(frame.continuation(), _normal_defers(frame.body_defers)))
        exit_block = frame.exit_block
        if exit_block is None:
            return False
        self._cur_block = exit_block
        if frame.on_done is not None:
            # the body returned normally and delivered its result: finish what
            # the call it was inlined into had to do with it (see
            # ``subscript``)
            frame.on_done()
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

    # -- deferred bodies -----------------------------------------------------

    def _region_defers(self, data: BlockFrameData) -> list[_DeferEntry]:
        """The deferred bodies declared directly in the region whose state is
        ``data`` - the region currently being walked (an ``if`` walks one branch
        at a time, a ``try`` one clause, ...)."""
        if isinstance(data, DeferBlockData):
            return data.body_defers
        if isinstance(data, IfBlockData):
            return data.then_defers if data.region == 0 else data.else_defers
        if isinstance(data, (LoopBlockData, PlainBlockData)):
            return data.body_defers
        if isinstance(data, TypeOfBlockData):
            # a type probe declares no deferred bodies of its own
            return []
        assert isinstance(data, TryExceptBlockData)
        return data.region_defers[data.region]

    def _current_defers(self) -> list[_DeferEntry]:
        """The list a ``with syntax.defer():`` region being opened appends its
        entry to: the defer list of the region currently being walked."""
        frame = self._frames[-1]
        if len(frame.block_stack) == 0:
            return frame.body_defers
        return self._region_defers(frame.block_stack[-1].data)

    def _collect_exit_defers(
        self,
        path: hir.DeferPath,
        stop_data: BlockFrameData | None,
        inclusive: bool,
        current_frame_only: bool,
    ) -> tuple[mir.BasicBlock, ...]:
        """The deferred bodies a transfer out of the currently walked regions
        triggers, in the order they run: from the innermost region outwards,
        every region's deferred bodies whose ``flags`` hold ``path``'s bit, each
        in reverse declaration order.  ``path`` is ``OK`` for a normal exit (a
        ``return``/``break``/``continue``/falling off), ``RAISE`` for a spy
        error leaving the function and ``UNWIND`` for a panic unwinding
        through it.  The walk stops at the region of ``stop_data`` - included
        or not per ``inclusive`` - and, when ``current_frame_only`` is set, at
        the current inline frame's own body (an inlined ``return``); otherwise
        it continues into the caller's frames, so that an error escaping an
        inlined body runs the defers of the caller's regions too.  A transfer
        that would leave a defer body is rejected."""
        ret: list[mir.BasicBlock] = []
        for frame in reversed(self._frames):
            for bf in reversed(frame.block_stack):
                if bf.data is stop_data:
                    if inclusive:
                        _append_defers(ret, self._region_defers(bf.data), path)
                    return tuple(ret)
                if isinstance(bf.data, DeferBlockData):
                    if path == hir.DeferPath.UNWIND:
                        # a panic unwinding out of a defer body leaves it: the
                        # defers declared *inside* the body run, then the walk
                        # stops - the body's own defer (registered in the region
                        # it was declared in) must not run a second time
                        _append_defers(ret, self._region_defers(bf.data), path)
                        return tuple(ret)
                    raise CompileError('cannot jump out of a defer block')
                if isinstance(bf.data, TypeOfBlockData):
                    raise CompileError('cannot jump out of a typeof block')
                _append_defers(ret, self._region_defers(bf.data), path)
            _append_defers(ret, frame.body_defers, path)
            if current_frame_only:
                return tuple(ret)
        if stop_data is not None:
            raise CompileError('the target of a transfer is not an open region')
        return tuple(ret)

    def _unwind_defers(self) -> tuple[mir.BasicBlock, ...]:
        """The deferred bodies a panic raised at the current position would run
        on its way out of the function (see ``_collect_exit_defers`` with
        ``UNWIND``).  A ``syntax.typeof`` probe emits no code, so a may-panic
        call inside one has no landing pad and this is empty."""
        for frame in reversed(self._frames):
            for bf in reversed(frame.block_stack):
                if isinstance(bf.data, TypeOfBlockData):
                    return ()
        return self._collect_exit_defers(
            hir.DeferPath.UNWIND, None, inclusive=False, current_frame_only=False,
        )

    def _resume_block(self) -> mir.BasicBlock:
        """The single block per function that carries a panic on after the
        deferred bodies of every ``CallMayPanic`` unwind edge have run (the
        ``unwind_path``), created on first use.  It holds one ``mir.Resume``;
        ``lower`` loads the exception the landing pad stored into the function's
        unwind slot and resumes with it."""
        if self._unwind_resume_block is None:
            block = mir.BasicBlock()
            block.emit(mir.Resume())
            self._unwind_resume_block = block
        return self._unwind_resume_block

    def _run_defers_on_fallthrough(self, defers: list[_DeferEntry], path: hir.DeferPath) -> None:
        """A region ended by falling off its end into a continuation that has no
        transfer instruction of its own (a compile-time ``if`` whose chosen
        branch fell through): when its deferred bodies have to run, split the
        block - jump through them to a fresh block and continue there."""
        out: list[mir.BasicBlock] = []
        _append_defers(out, defers, path)
        if len(out) == 0:
            return
        cont = mir.BasicBlock()
        self._cur_block.emit(mir.Jmp(cont, tuple(out)))
        self._cur_block = cont

    def _exec_defer(self, inst: hir.Defer) -> None:
        """Open a ``with syntax.defer():`` region (``hir.Defer``): the body
        follows in the flat instruction list, closed by the matching ``hir.End``.
        The body is *not* executed here - it is emitted into a block tree of its
        own, detached from the current block, and its entry is recorded in the
        enclosing region's defer list (see ``_current_defers``), so that only the
        transfers leaving that region run it.  The walk continues in the
        detached block until the body's ``End`` emits ``mir.EndDefer`` and
        restores the block the region was opened in (see ``_exec_end``).  A
        defer inside a compile-time (``syntax.unroll``) loop's body is registered
        in the region that wraps that body - the compile-time ``if`` a ``while``
        lowers to, or the ``try`` a ``for`` lowers to - so it runs at the end of
        each unrolled iteration, not at the loop's own unrolling edge.
        ``inst.flags`` is evaluated here - its instructions were run just before
        the ``Defer`` - and has to be a compile-time integer."""
        flags_value = self.operand(inst.flags)
        flags_int = _comptime_int(flags_value)
        if flags_int is None:
            raise CompileError('the flags of syntax.defer must be known at compile time')
        flags = hir.DeferPath(flags_int)
        frame = self._frames[-1]
        entry = frame.pc - 1
        p_else, _p_end = self._scan_block(entry)
        assert p_else is None, 'a defer body has no else marker'
        template = mir.BasicBlock()
        self._current_defers().append(_DeferEntry(flags, template))
        frame.block_stack.append(
            BlockFrame(entry, DeferBlockData(flags, template, [], self._cur_block))
        )
        self._cur_block = template

    def _exec_typeof_begin(self) -> None:
        """Open a ``syntax.typeof`` *type probe* (``hir.TypeOfBegin``): the
        probe's instructions follow in the flat list, closed by the matching
        ``hir.TypeOfEnd``.  The probe is built into a detached block - never
        linked to the function's CFG, so it is never lowered - which is what
        makes it emit no code: its instructions only *type* the probed
        expression (compiling whatever functions it names) and are discarded.
        The block the probe was opened in is restored by ``_exec_typeof_end``."""
        frame = self._frames[-1]
        frame.block_stack.append(
            BlockFrame(frame.pc - 1, TypeOfBlockData(self._cur_block))
        )
        self._cur_block = mir.BasicBlock()

    def _exec_typeof_end(self, inst: hir.TypeOfEnd) -> None:
        """Close the type probe opened by ``hir.TypeOfBegin``: the probed
        expression is the place ``inst.value`` denotes, and the instruction's
        register is its spy type as a compile-time value.  The probe's detached
        block is abandoned and the walk resumes in the block the probe was
        opened in."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        data = bf.data
        assert isinstance(data, TypeOfBlockData)
        type = _arg_type_of(ArgEntry(self.operand(inst.value), inst.is_ref))
        if type is None:
            raise CompileError('cannot determine the type of this value')
        frame.block_stack.pop()
        self._cur_block = data.saved_block
        frame.regs[inst] = ComptimeVal(type)

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
                    # inlined body; the caller continues in the exit block.  It
                    # runs the defers of the body's own regions first
                    if not self._cur_block.is_finished:
                        self._cur_block.emit(mir.Jmp(
                            frame.continuation(),
                            self._collect_exit_defers(
                                hir.DeferPath.OK, None, inclusive=False, current_frame_only=True,
                            ),
                        ))
                    return self._cut()
                # a ``return`` of the function proper runs the defers of every
                # region it leaves, up to the function body
                defer_blocks = self._collect_exit_defers(
                    hir.DeferPath.OK, None, inclusive=False, current_frame_only=False,
                )
                self.store(self._function_result().code, ComptimeVal(sval.Int(0, sval.IntType(0, False))))
                if self.ret_sig is None:
                    # the return convention is not fixed yet (an unannotated
                    # return type, or an inferred exception set): the ``mir.Ret``
                    # is filled in by ``_finish_function`` once it is known
                    self._defer_return(defer_blocks)
                    return self._cut()
                self._emit_function_return(defer_blocks)
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
                    regs[inst] = ComptimeVal(sval.UndefinedScalar(sval.PointerType(exception, is_const=False)))
                else:
                    regs[inst] = RuntimeVal(phi, sval.PointerType(exception, is_const=False))
            case hir.AsBool():
                return self.as_bool(self.operand_arg(inst.value), inst)
            case hir.Len():
                return self.exec_len(inst)
            case hir.IsNull():
                value = self._arg_value(self.operand_arg(inst.opt))
                value_type = _type_of(value)
                if not isinstance(value_type, (sval.OptionType, sval.NullType)):
                    raise CompileError(f'``is None`` needs an option, got {value_type}')
                regs[inst] = self._is_null(value)
            case hir.OptionPayloadPtr():
                place = self.operand(inst.ptr)
                ptr_type = _type_of(place)
                if not isinstance(ptr_type, sval.PointerType) or not isinstance(ptr_type.elem, sval.OptionType):
                    raise CompileError(f'a payload address needs an option, got {ptr_type}')
                regs[inst] = self._option_payload_ptr(place, write_tag=False)
            case hir.IsInstance():
                value = self._arg_value(self.operand_arg(inst.value))
                union_type = _type_of(value)
                if not isinstance(union_type, sval.TaggedUnionType):
                    raise CompileError(f'isinstance needs a tagged union, got {union_type}')
                variant = self.type_operand(inst.type, 'the isinstance type')
                index = union_type.variant_index(variant)
                if index is None:
                    raise CompileError(f'{variant} is not a variant of {union_type}')
                tag = self._tagged_union_tag(value)
                tag_obj = _to_comptime(tag)
                if isinstance(tag_obj, sval.Int):
                    regs[inst] = ComptimeVal(tag_obj.value == index)
                else:
                    tag_mir = union_type.tag_type().to_mir_type(self._mir_cache)
                    assert isinstance(tag_mir, mir.IntType)
                    cond = self._emit(mir.Cmp('==', self._to_runtime(tag), mir.Int(index, tag_mir)))
                    regs[inst] = RuntimeVal(cond, sval.BoolType())
            case hir.TaggedUnionPayloadPtr():
                place = self.operand(inst.ptr)
                ptr_type = _type_of(place)
                if not isinstance(ptr_type, sval.PointerType) or not isinstance(ptr_type.elem, sval.TaggedUnionType):
                    raise CompileError(f'a payload address needs a tagged union, got {ptr_type}')
                variant = self.type_operand(inst.type, 'the payload type')
                regs[inst] = self._tagged_union_payload_ptr(place, variant)
            case hir.BinaryAssign():
                return self.binary_assign(inst.op, self.operand(inst.lhs), self.operand_arg(inst.rhs))
            case hir.If():
                self._exec_if(inst)
            case hir.Defer():
                self._exec_defer(inst)
            case hir.Loop():
                self._exec_loop()
            case hir.Try():
                self._exec_try()
            case hir.Except():
                self._exec_except(inst)
            case hir.BreakLoop():
                return self._exec_break_loop()
            case hir.Continue():
                return self._exec_continue()
            case hir.Block():
                self._exec_block()
            case hir.BreakIf():
                return self._exec_break_if(inst)
            case hir.Else():
                self._exec_else()
            case hir.End():
                return self._exec_end()
            case hir.Load():
                regs[inst] = self.load(self.operand(inst.ptr))
            case hir.Alloca():
                regs[inst] = self.alloca(inst.is_comptime, inst.is_const, self._declared_type(inst.type), commit_declared=False)
            case hir.Store():
                if isinstance(inst.ptr, hir.ReturnLoc):
                    self._store_return_loc(self.operand(inst.value))
                else:
                    self.store(self.operand(inst.ptr), self.operand(inst.value))
            case hir.StoreVoidRetloc():
                self.store_void_retloc()
            case hir.Tuple():
                regs[inst] = self._build_tuple_value(
                    tuple(self.operand_arg(v) for v in inst.values)
                )
            case hir.InitTuple():
                self.init_tuple(self._loc_operand(inst.result_loc), inst.length)
            case hir.TuplePtrElement():
                regs[inst] = self.tuple_ptr_element(self._loc_operand(inst.tuple_ptr), inst.index)
            case hir.Binary():
                return self._eval_binary(inst.op, self.operand_arg(inst.lhs), self.operand_arg(inst.rhs), self._loc_operand(inst.ret))
            case hir.Compare():
                return self._eval_cmp(inst.op, self.operand_arg(inst.lhs), self.operand_arg(inst.rhs), inst)
            case hir.Not():
                return self._eval_not(self.operand(inst.value), inst)
            case hir.Unary():
                return self._eval_unary(inst.op, self.operand_arg(inst.operand), self._loc_operand(inst.ret))
            case hir.Ord():
                return self._eval_ord(self.operand(inst.operand), self._loc_operand(inst.ret))
            case hir.CallInplace():
                return self.call(self.operand(inst.callee), self.eval_call_args(inst.args), self._loc_operand(inst.ret))
            case hir.CallMethodInplace():
                return self.call_method(self.operand(inst.base), inst.name, self.eval_call_args(inst.args), self._loc_operand(inst.ret))
            case hir.FieldAddr():
                regs[inst] = self.exec_field_name_addr(self._loc_operand(inst.base), inst.name, inst.is_aggregate_init)
            case hir.FieldIndexAddr():
                regs[inst] = self.field_index_addr(
                    self._loc_operand(inst.base), _index_value(inst.index), inst.is_aggregate_init
                )
            case hir.FinishStruct():
                self.finish_struct(
                    self.operand(inst.struct),
                    self._loc_operand(inst.dest),
                    tuple(self.operand(v) for v in inst.indices),
                    frozendict((k, self.operand(v)) for k, v in inst.names.items()),
                )
            case hir.FinishArray():
                self.finish_array(self._loc_operand(inst.array), tuple(self.operand(e) for e in inst.elements))
            case hir.CommitSlot():
                self._commit_pending_slot(self.operand(inst.slot))
            case hir.Slice():
                # the slice object a slice subscript carries (see ``hir.Slice``)
                regs[inst] = self.slice_object(inst)
            case hir.Subscript():
                return self.subscript(self.operand(inst.base), self.operand_arg(inst.index), inst)
            case hir.PointerType():
                regs[inst] = ComptimeVal(sval.PointerType(
                    self.type_operand(inst.elem, 'the element type of a pointer'),
                    inst.is_const,
                    sval.PointerVariant.MULTI if inst.is_multi else sval.PointerVariant.SINGLE,
                ))
            case hir.ArrayType():
                regs[inst] = ComptimeVal(sval.ArrayType(
                    self.type_operand(inst.elem, 'the element type of an array'),
                    None if inst.length is None else self._operand_comptime_value(inst.length),
                ))
            case hir.OptionType():
                regs[inst] = ComptimeVal(sval.OptionType(
                    self.type_operand(inst.child, 'the child type of an option'),
                ))
            case hir.PtrCast():
                regs[inst] = self.exec_ptr_cast(self.operand(inst.value), self.operand(inst.type))
            case hir.AsFuncPtr():
                return self.exec_as_func_ptr(inst)
            case hir.TypeOfBegin():
                self._exec_typeof_begin()
            case hir.TypeOfEnd():
                self._exec_typeof_end(inst)
            case hir.MakeClosure():
                return self.exec_make_closure(inst)
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
        if _is_undefined_val(cond):
            raise CompileError('the condition of an if is undefined')
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
                BlockFrame(entry, IfBlockData(False, p_else=p_else, p_end=p_end, region=1))
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
            if isinstance(data, DeferBlockData):
                raise CompileError('break/continue cannot leave a defer block')
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

    def _exec_break_loop(self) -> PollResult:
        """``hir.BreakLoop``: end the current path at the innermost loop's exit
        block, created on demand - the first ``break`` is what makes the code
        after the loop reachable - and unwind the frame's open blocks like any
        other ended path (see ``_cut``).  The loop's exit is exactly where a
        ``_cut`` that reaches the loop continues the walk.  A compile-time
        loop's ``break`` works the same way: it leaves the whole unrolled
        sequence."""
        data = self._find_loop()
        exit_block = data.exit_block
        if exit_block is None:
            exit_block = data.exit_block = mir.BasicBlock()
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(
                exit_block,
                self._collect_exit_defers(hir.DeferPath.OK, data, inclusive=True, current_frame_only=False),
            ))
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
            self._cur_block.emit(mir.Jmp(
                nxt,
                self._collect_exit_defers(hir.DeferPath.OK, data, inclusive=True, current_frame_only=False),
            ))
        return self._cut()

    def _exec_block(self) -> None:
        """Open a ``hir.Block`` (``hir.Block``): its body follows in the flat
        instruction list, closed by the matching ``hir.End``.  Unlike an
        ``if`` there is nothing to split here - the current block continues as
        the body - so a ``hir.BreakIf`` of the body splits it itself (see
        ``_exec_break_if``), and the falling end and every break join in the
        block's exit block (see ``PlainBlockData``)."""
        frame = self._frames[-1]
        entry = frame.pc - 1
        p_else, p_end = self._scan_block(entry)
        assert p_else is None, 'a block body has no else marker'
        frame.block_stack.append(BlockFrame(entry, PlainBlockData(p_end=p_end)))

    def _block_exit(self, data: PlainBlockData) -> mir.BasicBlock:
        """The block the code after a ``hir.Block``'s ``End`` is typed in -
        created on demand.

        **This has a side effect**: creating the exit is what marks the code
        after the block as *reachable* - ``_cut`` continues the walk in it once
        ``exit_block`` is set (see ``_cut`` and ``PlainBlockData``).  A caller
        must therefore only invoke it when the code after the block really is
        reachable (a taken ``break``, or the body falling off its ``End``), and
        never for a break that turns out not to be taken (a compile-time-false
        ``hir.BreakIf``, see ``_exec_break_if``)."""
        if data.exit_block is None:
            data.exit_block = mir.BasicBlock()
        return data.exit_block

    def _find_block(self, levels: int) -> BlockFrame:
        """The ``levels``-th enclosing ``hir.Block`` of the executing frame (1
        is the innermost) - the target of a ``hir.BreakIf``.  Only ``hir.Block``
        frames count: a ``break``/``continue`` of a loop is unaffected by the
        blocks a condition of it opened (see ``_find_loop``)."""
        count = 0
        for bf in reversed(self._frames[-1].block_stack):
            if isinstance(bf.data, DeferBlockData):
                raise CompileError('break cannot leave a defer block')
            if isinstance(bf.data, PlainBlockData):
                count += 1
                if count == levels:
                    return bf
        raise CompileError('break out of more blocks than are open')

    def _exec_break_if(self, inst: hir.BreakIf) -> PollResult:
        """``hir.BreakIf``: leave ``inst.levels`` enclosing ``hir.Block``s when
        ``inst.cond`` holds (unconditionally when it is ``None``).  A compile-
        time condition folds: a ``false`` one does nothing - and, in
        particular, does *not* make the code after the block reachable, so the
        block's exit is left uncreated (see ``_block_exit``) - while any other
        taken break ends the path at the target block's exit block like
        ``break_loop``/``continue`` do (see ``_cut``).  A *runtime* condition
        splits the block being typed instead: the taken edge jumps to the exit
        and the walk continues in a fresh block for the rest of the body."""
        target_bf = self._find_block(inst.levels)
        target = target_bf.data
        assert isinstance(target, PlainBlockData)
        runtime_cond: mir.Value | None = None
        if inst.cond is not None:
            cond = self.operand(inst.cond)
            if _is_undefined_val(cond):
                raise CompileError('the condition of a break_if is undefined')
            if isinstance(cond, ComptimeVal):
                if not cond.obj:
                    # the break is not taken: leave the block's exit uncreated,
                    # so ``_cut`` keeps the code after the block dead (it stays
                    # reachable only through a fall-through or another taken
                    # break, see ``_exec_end``)
                    return PollResult.AGAIN
            else:
                runtime_cond = self._to_runtime(cond)
        # the break is taken (unconditionally, on a compile-time true condition,
        # or on the true edge of a runtime one): the code after the block now is
        # reachable
        exit_block = self._block_exit(target)
        defer_blocks = self._collect_exit_defers(
            hir.DeferPath.OK, target, inclusive=True, current_frame_only=False,
        )
        if runtime_cond is not None:
            cont_block = mir.BasicBlock()
            self._cur_block.emit(
                mir.Br(runtime_cond, exit_block, cont_block, if_true_defer_blocks=defer_blocks)
            )
            self._cur_block = cont_block
            return PollResult.AGAIN
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(exit_block, defer_blocks))
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
            # not return); the then block joins the continuation (after running
            # the then-region's defers) and the else-region is typed next
            data.then_returns = False
            exit_block = data.exit_block
            assert exit_block is not None and data.else_block is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(exit_block, _normal_defers(data.then_defers)))
            self._cur_block = data.else_block
            data.region = 1
            return
        # a compile-time ``if`` whose chosen branch is the then branch,
        # which fell off its end: the (unchosen) else branch is dead - the
        # then-region's defers still run before the code after the ``if``
        assert data.chosen
        self._run_defers_on_fallthrough(self._region_defers(data), hir.DeferPath.OK)
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
            region_defers=[[] for _ in range(len(p_excepts) + 1)],
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
            self._cur_block.emit(mir.Jmp(data.join, _normal_defers(self._region_defers(data))))
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
        union = sval.UnionType(frozenset(exception for _, _, _, exception in records))
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
        if isinstance(data, DeferBlockData):
            # the defer body fell off its end: it ends in ``mir.EndDefer``, which
            # triggers the defers declared inside the body (they run when the
            # body completes, a normal exit), and the walk continues in the
            # block the body was detached from
            self._cur_block.emit(mir.EndDefer(_normal_defers(data.body_defers)))
            frame.block_stack.pop()
            self._cur_block = data.saved_block
            return PollResult.AGAIN
        if isinstance(data, LoopBlockData):
            if data.is_inline:
                # the body fell off its end: unroll the next iteration, routing
                # this falling end into the block a ``continue`` jumped to (when
                # there was one)
                self._unroll_inline_loop(frame, frame.block_stack[-1], data)
                return PollResult.AGAIN
            # the loop body fell off its end: jump back to the header, running
            # the body's defers on the way.  Whether the code after the loop is
            # live is decided by ``_cut`` from the loop's exit block (created
            # only by a ``break``)
            assert data.header_block is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(data.header_block, _normal_defers(data.body_defers)))
            return self._cut()
        if isinstance(data, TryExceptBlockData):
            # the last except clause fell off its end: the try is complete
            assert data.join is not None
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(data.join, _normal_defers(self._region_defers(data))))
            self._cur_block = data.join
            frame.block_stack.pop()
            return PollResult.AGAIN
        if isinstance(data, PlainBlockData):
            # a ``hir.Block`` body fell off its end: it joins the code after the
            # block - the block the breaks of the body jump to (see
            # ``PlainBlockData``) - after running the body's defers
            exit_block = self._block_exit(data)
            if not self._cur_block.is_finished:
                self._cur_block.emit(mir.Jmp(exit_block, _normal_defers(data.body_defers)))
            self._cur_block = exit_block
            frame.block_stack.pop()
            return PollResult.AGAIN
        assert isinstance(data, IfBlockData)
        if data.chosen is not None:
            # a compile-time ``if``: the chosen branch fell off its end, a
            # normal exit of its region
            self._run_defers_on_fallthrough(self._region_defers(data), hir.DeferPath.OK)
            frame.block_stack.pop()
            return PollResult.AGAIN
        exit_block = data.exit_block
        assert exit_block is not None
        if not self._cur_block.is_finished:
            self._cur_block.emit(mir.Jmp(exit_block, _normal_defers(self._region_defers(data))))
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
            if isinstance(data, DeferBlockData):
                # the current path ended inside a defer body and every path of the
                # body ended elsewhere: the body never completes, so the defer
                # would never run
                raise CompileError('the body of a defer must complete')
            if isinstance(data, TypeOfBlockData):
                # the probed expression ended its path (a call that never
                # returns, or an error that no clause caught): a type probe only
                # ever types, so it cannot leave the block
                raise CompileError(
                    'the argument of typeof must produce a value: it cannot end '
                    'the path'
                )
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
            if isinstance(data, PlainBlockData):
                # the current path ended inside a ``hir.Block`` (a ``return``, a
                # ``raise``, or a ``break``/``continue`` of an enclosing loop):
                # the block is complete.  The code after its ``End`` is still
                # reachable when some ``hir.BreakIf`` of the body reaches the
                # block's exit, and dead otherwise - exactly the loop's case
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
                    data.region = 1
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
        type.  A type parameter that is *not* bound here stands for itself:
        that is how a closure's own type parameter is evaluated while its
        annotations are read at its creation site (a closure's ``[T]`` is
        only solved when the closure is called, not when it is created)."""
        if not isinstance(obj, sval.TypeVar):
            return None
        value = self._frames[-1].generic_var_values.get(obj)
        if value is None:
            return ComptimeVal(obj)
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
                if not isinstance(obj, (int, float, bytes, bool, pytypes.NoneType)):
                    resolved = self._analyser._resolver.resolve_global(obj)
                    if resolved is not None:
                        return ComptimeVal(resolved)
                return ComptimeVal(sval.as_value(obj, ctx=self._analyser._resolver))
            case hir.Arg(index):
                assert len(self._frames) > 0, 'Arg outside of any function frame'
                frame = self._frames[-1]
                assert index < len(frame.arg_values), 'Arg index out of range'
                return frame.arg_values[index]
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
        if not type.elem.is_copyable():
            # reading a value out of its place is a copy: a non-copyable type
            # (a dynamically-sized one, or a struct whose copy is forbidden) has
            # no such value
            raise CompileError(f'cannot copy a value of the non-copyable type {type.elem}')
        unit_value = type.elem.get_unit_value()
        if unit_value is not None:
            return ComptimeVal(unit_value)
        match ptr:
            case ComptimeVal(sval.ComptimePtr(box, offset, _)):
                # a compile-time place: rebuild the value from the box's cells
                return ComptimeVal(_load_box_cells(box, offset, type.elem))
            case ComptimeVal(sval.CastedPtr()):
                # a pointer reinterpreted by ``ptr_cast`` cannot be read through
                raise CompileError('cannot read through a pointer cast')
            case RuntimeVal():
                return RuntimeVal(self._emit(mir.Load(ptr.value)), type.elem)
        raise CompileError('cannot load from a compile-time pointer')

    def store(self, ptr: InterpVal, value: InterpVal) -> None:
        """Write ``value`` into the slot or through the pointer ``ptr``.
        A store into a still uncommitted slot only records a store point
        (see :class:`PendingSlot`): the slot's final type is not known
        until it is committed, and the actual store is inserted then.

        ``ptr`` is a *place* and ``value`` a *value* of its element type - a
        place *is* a pointer here (see ``_type_of``), so the two are different
        kinds of thing and the value is never read out of what it points at:
        whoever hands a place over where a value is needed reads it (``astgen``
        emits the ``Load`` of a name or a subscript, see ``_as_value``).

        A zero-sized type has no storage, so no store is *emitted* for one at
        runtime; a compile-time place (a ``Box``) records the value in its
        flattened cells instead.

        A runtime value never goes into a compile-time place: that is the
        ``cannot store a runtime value into a compile-time variable`` error (see
        the behaviour doc § 13)."""
        value = _shallow_normalize(value)
        ptr = _shallow_normalize(ptr)

        if isinstance(ptr, PendingSlot) and ptr.committed is None:
            value_type = _type_of(value)
            if value_type is None:
                raise CompileError('cannot store a value that has no spy type')
            self._record_pending_store(ptr, value_type, value)
            return

        ptr_type = _type_of(ptr)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f"cannot store to a {ptr_type} value")
        elem = ptr_type.elem
        if isinstance(elem, sval.UnionType) and _is_union_unit(value):
            # a union value carries no storage - it says nothing but which union
            # it belongs to - so a storage destination has nothing to write
            return

        if _is_undefined_val(value):
            self._store_undefined(ptr, elem)
            return

        # the value is converted to the place's element type once, up front
        value = self._coerce(value, elem)

        if isinstance(ptr, ComptimeVal) and isinstance(ptr.obj, sval.UndefinedScalar):
            # a compile-time pointer with no storage at all: the address of a
            # zero-sized field or element (see ``field_index_addr``).  There is
            # no storage to write, but the value still has to be of the
            # (single-valued) type the place points at
            return

        match ptr:
            case ComptimeVal(sval.ComptimePtr(box, offset, _)):
                if not ptr.obj.is_writable():
                    raise CompileError(
                        f'cannot store through the read-only pointer {ptr.obj.type}'
                    )
                if isinstance(box, bytes):
                    raise CompileError('cannot write through a byte-string pointer')
                if not _is_comptime_val(value):
                    raise CompileError(
                        f'cannot store a runtime value into a compile-time variable '
                        f'of type {elem}'
                    )
                obj = _to_comptime(value)
                assert obj is not None
                cells = _flatten_value(obj, elem)
                box.cells[offset:offset + len(cells)] = cells
            case ComptimeVal(sval.CastedPtr()):
                raise CompileError('cannot write through a pointer cast')
            case RuntimeVal():
                if ptr_type.is_const:
                    # a read-only view (a slice built by subscripting, say) is not
                    # storage to write through
                    raise CompileError(f'cannot store through the const pointer {ptr_type}')
                if elem.get_unit_value() is not None:
                    # a zero-sized type has no storage: nothing is emitted
                    return
                self._emit(mir.Store(ptr.value, self._to_runtime(value)))
            case _:
                raise CompileError('cannot store through a compile-time pointer')

    def _store_undefined(self, ptr: InterpVal, elem: sval.Type) -> None:
        """Leave the destination ``ptr`` of element type ``elem`` undefined: a
        zero-sized place has no storage, a runtime location is filled with
        ``undef``, and a compile-time ``Box`` records the typed undefined in its
        cells."""
        elem_unit = elem.get_unit_value()
        match ptr:
            case ComptimeVal(sval.ComptimePtr(box, offset, _)):
                if not ptr.obj.is_writable():
                    raise CompileError(
                        f'cannot store through the read-only pointer {ptr.obj.type}'
                    )
                if isinstance(box, bytes):
                    raise CompileError('cannot write through a byte-string pointer')
                cells = _flatten_value(sval.coerce_const(sval.UntypedUndefined(), elem), elem)
                box.cells[offset:offset + len(cells)] = cells
            case RuntimeVal():
                if elem_unit is not None:
                    # a zero-sized type has no storage to write
                    return
                mir_type = elem.to_mir_type(self._mir_cache)
                if mir_type is None:
                    raise _no_runtime_type(elem)
                self._emit(mir.Store(ptr.value, mir.UndefValue(mir_type)))
            case ComptimeVal(sval.UndefinedScalar()):
                # a zero-sized place: nothing to leave undefined
                return
            case _:
                raise CompileError(f'cannot leave a {elem} undefined')

    def _extract_aggregate_value(self, value: InterpVal, index: int) -> InterpVal:
        """The ``index``-th field (or element) of the aggregate ``value``, in
        declaration (element) order: the place it holds for a compile-time
        aggregate, and the value read out of a *runtime* one with
        ``mir.ExtractValue`` - a zero-sized field holds its unit value, and the
        mirror of a struct of one stored field *is* that field (see
        ``StructType.mirror_is_a_field``), so it is the value itself.  A value
        that is not an aggregate at all (an ``Option``, say) is rejected."""
        value = _shallow_normalize(value)
        aggregate = _as_aggregate(value)
        if aggregate is not None:
            return ComptimeVal(aggregate.values[index])
        type = _type_of(value)
        if not isinstance(type, (sval.StructType, sval.ArrayType, sval.ComplexType, sval.TupleType)):
            raise CompileError(f'{value!r} is not an aggregate')
        field_type = _aggregate_place_types(type)[index]
        if field_type.classify() == sval.SpecialTypeKind.DST:
            # a dynamically-sized field has no value to read: only its address
            # can be taken (see ``field_index_addr``)
            raise CompileError(
                f'cannot read the dynamically-sized field of {type} by value'
            )
        unit = field_type.get_unit_value()
        if unit is not None:
            return ComptimeVal(unit)
        if not isinstance(value, RuntimeVal):
            raise CompileError(f'cannot read a field of {value!r}')
        if isinstance(type, (sval.ArrayType, sval.ComplexType)):
            # an array's elements and a complex number's parts sit in element
            # order in the mirror (a complex number mirrors to a struct of its
            # real and imaginary parts, with no reordering)
            mir_index = index
        elif type.mirror_is_a_field(self._mir_cache):
            return RuntimeVal(value.value, field_type)
        else:
            mir_index = type.get_field_mir_indices(self._mir_cache)[index]
            assert mir_index is not None, 'a field with storage has a mirror position'
        return RuntimeVal(self._emit(mir.ExtractValue(value.value, mir_index)), field_type)

    def as_comptime_aggregate(self, ev: InterpVal) -> sval.AggregateValue:
        """The ``sval.AggregateValue`` form of the aggregate value ``ev`` (whose
        type it is read off): a compile-time aggregate - held as one object - is
        returned as it is, and a value with a runtime part is rejected (a genuine
        aggregate of runtime parts is a place, not a value form)."""
        ev = _shallow_normalize(ev)
        aggregate = _as_aggregate(ev)
        if aggregate is not None:
            return aggregate
        type = _type_of(ev)
        if not isinstance(type, (sval.StructType, sval.ArrayType, sval.ComplexType)):
            raise CompileError(f'{ev!r} is not an aggregate')
        objs: list[sval.AnyValue] = []
        for index in range(sval.aggregate_type_length(type)):
            obj = _to_comptime(self._extract_aggregate_value(ev, index))
            if obj is None:
                raise CompileError(f'{ev!r} is not a compile-time aggregate')
            objs.append(obj)
        return sval.AggregateValue(tuple(objs), type)

    def _complex_parts(self, ev: InterpVal) -> tuple[InterpVal, InterpVal]:
        """The real and the imaginary part of the complex number ``ev``, as
        values (either may still be runtime): the two field values of a
        compile-time complex number, or ``mir.ExtractValue`` of a runtime one (a
        complex number mirrors to a struct of its parts)."""
        ev = _shallow_normalize(ev)
        return self._extract_aggregate_value(ev, 0), self._extract_aggregate_value(ev, 1)

    def _complex_value(self, type: sval.ComplexType, re: InterpVal, im: InterpVal) -> InterpVal:
        """A complex number of the type ``type`` built from its two parts: a
        compile-time value when both are, a ``mir.InsertValue`` chain otherwise."""
        re_obj = _to_comptime(re)
        im_obj = _to_comptime(im)
        if re_obj is not None and im_obj is not None:
            return ComptimeVal(sval.AggregateValue((re_obj, im_obj), type))
        mir_type = type.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.StructType)
        result = self._emit(mir.InsertValue(mir.UndefValue(mir_type), self._to_runtime(re), 0))
        result = self._emit(mir.InsertValue(result, self._to_runtime(im), 1))
        return RuntimeVal(result, type)

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
        if isinstance(location, tuple) or (
            isinstance(location, PendingSlot) and self._pending_tuple(location) is not None
        ):
            raise CompileError('a function that returns several values must return them')
        self.store(location, ComptimeVal(sval.Void()))

    def _loc_operand(self, loc: hir.FullResultLoc) -> InterpVal | tuple:
        """The place (or place tree) a HIR result location denotes: a
        ``ReturnLoc`` resolves to the current frame's result location tree (see
        §7.2), a Python tuple is a ``ResultLocTree`` whose leaves each resolve in
        turn, and any other location is an ordinary place operand."""
        if isinstance(loc, hir.ReturnLoc):
            return self._result_loc()
        if isinstance(loc, tuple):
            return tuple(self._loc_operand(child) for child in loc)
        return self.operand(loc)

    def _store_return_loc(self, value: InterpVal) -> None:
        """Deliver ``value`` into the current frame's result location (a ``hir.Store``
        whose place is ``ReturnLoc``): a tuple value is splatted element by element
        into the result tree, anything else goes into the single result place (see
        §7.2)."""
        target = self._result_loc()
        value = _shallow_normalize(value)
        value_type = _type_of(value)
        if isinstance(value_type, sval.TupleType):
            leaves = tuple(
                self._extract_aggregate_value(value, index)
                for index in range(len(value_type.types))
            )
            if isinstance(target, tuple):
                if len(target) != len(leaves):
                    raise CompileError(
                        f'cannot return {len(leaves)} value(s) from a function '
                        f'returning {len(target)}'
                    )
                for place, leaf in zip(target, leaves):
                    self.store(place, leaf)
                return
            if isinstance(target, PendingSlot) and target.committed is None:
                # an untyped result location: type it as the leaves' places
                places = tuple(self.alloca() for _ in leaves)
                self._record_pending_action(target, _PendingTuple(places))
                for place, leaf in zip(places, leaves):
                    self.store(place, leaf)
                return
            raise CompileError('cannot return several values into a single result location')
        if isinstance(target, tuple):
            raise CompileError('cannot return a single value from a function returning several')
        self.store(target, value)

    # -- options ---------------------------------------------------------------

    def _not_bool(self, value: InterpVal) -> mir.Value:
        """The negation of a boolean value as a MIR ``bool``: a compile-time one
        folds, a runtime one becomes a comparison against ``false``."""
        obj = _to_comptime(value)
        if isinstance(obj, bool):
            return mir.BoolValue(not obj)
        return self._emit(mir.Cmp('==', self._to_runtime(value), mir.BoolValue(False)))

    def _coerce_option_value(self, ev: InterpVal, option: sval.OptionType) -> InterpVal:
        """Materialize ``ev`` as a value of the option type ``option``: a value
        that already is one passes through, the null value becomes the absent one
        and anything else a present value of the child type.

        A fully compile-time result is the value form - ``sval.OptionValue`` for a
        present option, ``sval.TypedNull`` for an absent one (see §5.3); a result
        with a runtime part is materialized into the option's runtime
        representation (see ``_option_parts_to_runtime``)."""
        ev = _shallow_normalize(ev)
        if _type_of(ev) == option:
            return ev
        child = option.child
        is_null = self._is_null(ev)
        if _to_comptime(is_null) is True:
            # absent: the payload is discarded, and the child type is only named
            if isinstance(is_null, ComptimeVal):
                return ComptimeVal(sval.TypedNull(child))
            if child.is_zst():
                unit = child.get_unit_value()
                assert unit is not None
                payload_value = ComptimeVal(unit)
            else:
                payload_value = ComptimeVal(sval.UndefinedScalar(child))
            return RuntimeVal(self._option_parts_to_runtime(option, is_null, payload_value), option)
        payload = _to_comptime(ev)
        if payload is not None:
            if isinstance(payload, sval.OptionValue):
                payload = payload.value
            coerced = _to_comptime(self._coerce(ComptimeVal(payload), child))
            assert coerced is not None
            return ComptimeVal(sval.OptionValue(coerced))
        return RuntimeVal(self._option_parts_to_runtime(option, is_null, ev), option)

    # -- tagged unions -------------------------------------------------------

    def _tagged_union_int(self, tag: InterpVal) -> int:
        """The compile-time int a tag value denotes."""
        obj = _to_comptime(tag)
        if not isinstance(obj, sval.Int):
            raise CompileError(f'expects a tag known at compile time, got {tag!r}')
        return obj.value

    def _tagged_union_index(self, type: sval.TaggedUnionType, variant_type: sval.Type) -> int:
        index = type.variant_index(variant_type)
        if index is None:
            raise CompileError(f'{variant_type} is not a variant of {type}')
        return index

    def _tagged_union_variant_index(self, ev: InterpVal, type: sval.TaggedUnionType) -> int:
        """Which variant of ``type`` the value ``ev`` is: the variant its spy
        type is (or a subtype of), or - for an untyped compile-time literal -
        the first variant it coerces to."""
        ev_type = _type_of(ev)
        if ev_type is not None:
            index = type.variant_index_for(ev_type)
            if index is not None:
                return index
        if isinstance(_shallow_normalize(ev), ComptimeVal):
            for index, variant in enumerate(type.types):
                try:
                    self._coerce(ev, variant)
                except (CoerceError, CompileError):
                    # a variant the literal does not fit (a pointer, say)
                    continue
                return index
        raise CoerceError(f'cannot materialize a {type} from {ev!r}')

    def _tagged_union_tag(self, ev: InterpVal) -> InterpVal:
        """The tag of the tagged union value ``ev``, as an interpreter int: the
        compile-time tag of a compile-time value (a ``sval.TaggedUnionValue``, or
        the place of one - a ``ComptimePtr``), 0 for a single-variant union, the
        value itself when the payload is zero-sized (the union *is* the tag) and
        the first field of the struct otherwise."""
        ev = _shallow_normalize(ev)
        obj = _to_comptime(ev)
        if isinstance(obj, sval.ComptimePtr):
            obj = _to_comptime(self.load(ev))
        if isinstance(obj, sval.TaggedUnionValue):
            return ComptimeVal(sval.Int(obj.index, obj.type.tag_type()))
        type = _type_of(ev)
        assert isinstance(type, sval.TaggedUnionType)
        shape = sval.tagged_union_shape(type, self._mir_cache)
        if shape == 'single':
            return ComptimeVal(sval.Int(0, type.tag_type()))
        assert isinstance(ev, RuntimeVal)
        if shape == 'tag_only':
            return RuntimeVal(ev.value, type.tag_type())
        return RuntimeVal(self._emit(mir.ExtractValue(ev.value, 0)), type.tag_type())

    def _tagged_union_field_ptr(self, ptr: InterpVal) -> InterpVal:
        """The address of the payload field of the tagged union place ``ptr``,
        typed as the payload union."""
        place = _shallow_normalize(ptr)
        ptr_type = _type_of(place)
        assert isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.TaggedUnionType)
        field = self._emit(mir.Gep(self._to_runtime(place), 1))
        return RuntimeVal(field, sval.PointerType(ptr_type.elem.payload_type(), ptr_type.is_const))

    def _tagged_union_payload_ptr(self, ptr: InterpVal, variant_type: sval.Type) -> InterpVal:
        """The place of the payload of the variant ``variant_type`` in the
        tagged union place ``ptr``: the variant's own place for a compile-time
        storage, and the storage reinterpreted as the variant for a runtime one
        (see ``_union_variant_ptr``).  A zero-sized variant has no place."""
        place = _shallow_normalize(ptr)
        ptr_type = _type_of(place)
        assert isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.TaggedUnionType)
        type = ptr_type.elem
        self._tagged_union_index(type, variant_type)
        if variant_type.get_unit_value() is not None:
            return ComptimeVal(sval.UndefinedScalar(sval.PointerType(variant_type, is_const=ptr_type.is_const)))
        if isinstance(place, ComptimeVal) and isinstance(place.obj, sval.ComptimePtr):
            # a compile-time union place: the payload starts at cell ``1`` (the box
            # layout is ``[tag, max(variant)]``, see §0.2), whichever variant it is
            cptr = place.obj
            return ComptimeVal(sval.ComptimePtr(
                cptr.box, cptr.offset + 1, sval.PointerType(variant_type, cptr.type.is_const),
            ))
        if sval.tagged_union_shape(type, self._mir_cache) == 'single':
            return RuntimeVal(self._to_runtime(place), sval.PointerType(variant_type, is_const=ptr_type.is_const))
        return self._union_variant_ptr(self._tagged_union_field_ptr(place), variant_type)

    def _write_tagged_union_tag(self, ptr: InterpVal, type: sval.TaggedUnionType, index: int) -> InterpVal:
        """Set the tag of the tagged union place ``ptr`` to the variant ``index``
        (a compile-time storage writes the tag cell) and return the place to write
        the payload through."""
        place = _shallow_normalize(ptr)
        if isinstance(place, ComptimeVal) and isinstance(place.obj, sval.ComptimePtr):
            cptr = place.obj
            if isinstance(cptr.box, sval.Box):
                cptr.box.cells[cptr.offset] = sval.Int(index, type.tag_type())
            return place
        assert isinstance(place, RuntimeVal)
        shape = sval.tagged_union_shape(type, self._mir_cache)
        if shape == 'single':
            return place
        tag_mir = type.tag_type().to_mir_type(self._mir_cache)
        assert isinstance(tag_mir, mir.IntType)
        if shape == 'tag_only':
            self._emit(mir.Store(place.value, mir.Int(index, tag_mir)))
            return place
        tag_ptr = self._emit(mir.Gep(place.value, 0))
        self._emit(mir.Store(tag_ptr, mir.Int(index, tag_mir)))
        return place

    def _tagged_union_parts_to_runtime(self, type: sval.TaggedUnionType, tag: InterpVal, value: InterpVal) -> mir.Value:
        """Build the runtime value of a tagged union from its tag ``tag`` (which
        may be a runtime value) and its payload ``value``: a compile-time tag
        makes the payload a variant value, joined into the payload union with an
        ``AsUnion``, while a runtime tag means the payload already *is* the
        payload union storage (a value read out of a runtime union).  A
        zero-sized payload keeps the tag alone, and a single-variant union is the
        variant itself."""
        shape = sval.tagged_union_shape(type, self._mir_cache)
        if shape == 'single':
            return self._to_runtime(self._coerce(value, type.types[0]))
        if shape == 'tag_only':
            return self._to_runtime(tag)
        struct_mir = type.to_mir_type(self._mir_cache)
        union_mir = type.payload_type().to_mir_type(self._mir_cache)
        assert struct_mir is not None and union_mir is not None
        union_value: mir.Value
        index = _comptime_int(tag)
        if index is None:
            # a tag only known at runtime: the payload already is the union storage
            if isinstance(value, ComptimeVal) and isinstance(value.obj, (sval.UndefinedScalar, sval.UndefinedAggregate)):
                union_value = mir.UndefValue(union_mir)
            else:
                union_value = self._to_runtime(value)
        else:
            variant_value = self._to_runtime(self._coerce(value, type.types[index]))
            union_value = self._emit(mir.AsUnion(variant_value, union_mir))
        with_tag = self._emit(mir.InsertValue(mir.UndefValue(struct_mir), self._to_runtime(tag), 0))
        return self._emit(mir.InsertValue(with_tag, union_value, 1))

    def _tagged_union_reindex(self, tag: InterpVal, from_type: sval.TaggedUnionType, to_type: sval.TaggedUnionType) -> InterpVal:
        """The tag ``tag`` of ``from_type`` remapped to ``to_type``'s variant
        order (a runtime tag becomes a chain of ``mir.Select``)."""
        mapping: list[tuple[int, int]] = []
        for index, variant in enumerate(from_type.types):
            to_index = to_type.variant_index(variant)
            assert to_index is not None, f'{variant} is not a variant of {to_type}'
            mapping.append((index, to_index))
        obj = _to_comptime(tag)
        if isinstance(obj, sval.Int):
            return ComptimeVal(sval.Int(dict(mapping)[obj.value], to_type.tag_type()))
        from_tag_mir = from_type.tag_type().to_mir_type(self._mir_cache)
        to_tag_mir = to_type.tag_type().to_mir_type(self._mir_cache)
        assert isinstance(from_tag_mir, mir.IntType) and isinstance(to_tag_mir, mir.IntType)
        src = self._to_runtime(tag)
        acc: mir.Value = mir.Int(mapping[0][1], to_tag_mir)
        for index, to_index in mapping[1:]:
            cond = self._emit(mir.Cmp('==', src, mir.Int(index, from_tag_mir)))
            acc = self._emit(mir.Select(cond, mir.Int(to_index, to_tag_mir), acc))
        return RuntimeVal(acc, to_type.tag_type())

    def _convert_union_runtime(self, ev: RuntimeVal, from_type: sval.TaggedUnionType, to_type: sval.TaggedUnionType) -> InterpVal:
        """The runtime tagged union ``ev`` (of the compatible union ``from_type``)
        converted to ``to_type``: the tag remapped and the payload union storage
        reinterpreted (see ``_convert_union_storage``)."""
        to_shape = sval.tagged_union_shape(to_type, self._mir_cache)
        from_shape = sval.tagged_union_shape(from_type, self._mir_cache)
        if to_shape == 'single':
            # a subset of a single-variant union is that variant
            return RuntimeVal(ev.value, to_type)
        tag = self._tagged_union_reindex(self._tagged_union_tag(ev), from_type, to_type)
        if to_shape == 'tag_only':
            return RuntimeVal(self._to_runtime(tag), to_type)
        struct_mir = to_type.to_mir_type(self._mir_cache)
        assert isinstance(struct_mir, mir.StructType)
        union_mir = struct_mir.fields[1].type
        if from_shape == 'tag_payload':
            storage = self._convert_union_storage(
                RuntimeVal(self._emit(mir.ExtractValue(ev.value, 1)), from_type.payload_type()),
                to_type.payload_type(),
            )
            storage_mir = self._to_runtime(storage)
        else:
            # the source variants are all zero-sized: the active payload has no
            # storage
            storage_mir = mir.UndefValue(union_mir)
        with_tag = self._emit(mir.InsertValue(mir.UndefValue(struct_mir), self._to_runtime(tag), 0))
        return RuntimeVal(self._emit(mir.InsertValue(with_tag, storage_mir, 1)), to_type)

    def _convert_union_storage(self, value: InterpVal, to_union: sval.UnionType) -> InterpVal:
        """The union storage value ``value`` as a value of the union
        ``to_union``: the two share their storage (every variant lives at offset
        0), so the value is reinterpreted - a union without storage is the
        target's unit value, an undefined (or storage-less) one stays undefined,
        and any other runtime value is cast with ``mir.UnionCast``."""
        unit = to_union.get_unit_value()
        if unit is not None:
            return ComptimeVal(unit)
        if isinstance(value, ComptimeVal) and isinstance(value.obj, (sval.UndefinedScalar, sval.UndefinedAggregate, sval.UnionValue)):
            return ComptimeVal(sval.UndefinedScalar(to_union))
        assert isinstance(value, RuntimeVal)
        if value.type == to_union:
            return value
        to_mir = to_union.to_mir_type(self._mir_cache)
        assert to_mir is not None
        return RuntimeVal(self._emit(mir.UnionCast(value.value, to_mir)), to_union)

    def _coerce_tagged_union_value(self, ev: InterpVal, type: sval.TaggedUnionType) -> InterpVal:
        """Materialize ``ev`` as a value of the tagged union ``type``: a value of
        a compatible union is converted, anything else is the variant it belongs
        to, tagged.

        A fully compile-time result is ``sval.TaggedUnionValue`` (see §5.3); one
        with a runtime part is materialized into the union's representation."""
        ev = _shallow_normalize(ev)
        ev_type = _type_of(ev)
        if ev_type == type:
            return ev
        obj = _to_comptime(ev)
        if isinstance(obj, sval.ComptimePtr):
            return self._coerce_tagged_union_value(self.load(ev), type)
        if isinstance(obj, sval.TaggedUnionValue):
            # a compile-time union of a compatible type: remap the tag and re-coerce
            # the variant value
            if isinstance(ev_type, sval.TaggedUnionType):
                index = self._tagged_union_index(type, ev_type.types[obj.index])
            else:
                index = self._tagged_union_variant_index(ComptimeVal(obj.value), type)
            variant_value = self._coerce(ComptimeVal(obj.value), type.types[index])
            variant_obj = _to_comptime(variant_value)
            assert variant_obj is not None
            return ComptimeVal(sval.TaggedUnionValue(type, index, variant_obj))
        if isinstance(ev, RuntimeVal) and isinstance(ev_type, sval.TaggedUnionType):
            return self._convert_union_runtime(ev, ev_type, type)
        index = self._tagged_union_variant_index(ev, type)
        variant_value = self._coerce(ev, type.types[index])
        variant_obj = _to_comptime(variant_value)
        if variant_obj is not None:
            return ComptimeVal(sval.TaggedUnionValue(type, index, variant_obj))
        return RuntimeVal(
            self._tagged_union_parts_to_runtime(type, ComptimeVal(sval.Int(index, type.tag_type())), variant_value),
            type,
        )

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
        place = _shallow_normalize(ev)
        # a compile-time place declared to hold no fixed type (an ``Any`` variable)
        # may yet hold a pointer - a function pointer, say (``comptime(); fn: Any =
        # foo``).  What it holds is what a callee/field access needs, so it is
        # dereferenced like a pointer that points at a pointer
        if (
            isinstance(t, sval.PointerType)
            and isinstance(place, ComptimeVal)
            and isinstance(place.obj, sval.ComptimePtr)
        ):
            loaded = self.load(place)
            if isinstance(_type_of(loaded), sval.PointerType):
                return loaded
        return ev

    # -- struct and array values ---------------------------------------------

    def exec_field_name_addr(self, ptr: InterpVal, name: str, is_aggregate_init: bool = False) -> InterpVal:
        """The address of the field ``name`` of the struct ``base`` points at.
        Auto-dereferences a base that points at a pointer, unlike
        ``field_index_addr``."""
        ptr = _shallow_normalize(ptr)
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
            return self.alloca()
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
                ptr = self._option_payload_ptr(ptr)
                type = _type_of(ptr)
                assert isinstance(type, sval.PointerType)
                container_type = type.elem
        if isinstance(container_type, sval.ComplexType):
            # a complex number's parts are its fields (``real`` and ``imag``)
            index = container_type.field_index(name)
            if index is None:
                raise CompileError(f"type {container_type} has no field named '{name}'")
            return self.field_index_addr(ptr, _index_value(index))
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
            # the aggregate's storage has no address yet: the field gets a
            # pending place of its own - the one a previous construction of the
            # same storage recorded, when there is one (the branches of an
            # ``if`` expression build into one storage), see ``finish_struct``
            recorded = self._pending_aggregate(ptr)
            if recorded is not None:
                index_int = _comptime_index(index)
                if index_int is not None and 0 <= index_int < len(recorded.places):
                    return recorded.places[index_int]
            return self.alloca()
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
                ptr = self._option_payload_ptr(ptr)
                type = _type_of(ptr)
                assert isinstance(type, sval.PointerType)
                container_type = type.elem
                is_const = type.is_const

        if is_aggregate_init and isinstance(container_type, sval.TaggedUnionType):
            # a construction into a tagged union: which variant the storage holds
            # is only known from the struct being built (see ``finish_struct``),
            # so the field gets a pending place the closing ``FinishStruct`` binds
            # to the variant's field address.  The place is inline: a compile-time
            # field value stays compile-time (it is not needlessly materialized
            # and read back as a register, see ``_finish_struct_into_tagged_union``)
            return self.alloca(True)

        if isinstance(container_type, sval.AggregateType):
            is_array = isinstance(container_type, sval.ArrayType)
            index_int = _try_comptime_index(index)
            if not is_array and index_int is None:
                # only an array takes a runtime index; every other aggregate
                # element is named by a compile-time integer
                raise CompileError(
                    'cannot pick a field or element with a runtime index: '
                    'it has to be a compile-time integer'
                )
            field_is_ref = False
            if isinstance(container_type, sval.StructType):
                assert index_int is not None
                fields = container_type.fields()
                if index_int < 0 or index_int >= len(fields.by_id):
                    raise CompileError(f'type {container_type} has no field at index {index_int}')
                field0 = fields.get_by_id(index_int)
                if field0.is_comptime:
                    # a compile-time field does not enter the MIR layout: a
                    # runtime struct reads its default value, a compile-time
                    # (box) struct the cell's current value (see §2.4/§5.1).  A
                    # by-reference compile-time capture bakes the captured
                    # variable's pointer as its default, which *is* the place the
                    # body acts through (see §9.3)
                    if field0.is_ref and isinstance(field0.default, sval.ComptimePtr):
                        return ComptimeVal(field0.default)
                    return self._comptime_field_place(ptr, container_type, index_int, field0)
                field_type = field0.type
                field_is_ref = field0.is_ref
            elif isinstance(container_type, sval.TupleType):
                assert index_int is not None
                if index_int < 0 or index_int >= len(container_type.types):
                    raise CompileError(f'type {container_type} has no element at index {index_int}')
                field_type = container_type.types[index_int]
            elif isinstance(container_type, sval.StrDictType):
                assert index_int is not None
                elem_types = tuple(container_type.values.values())
                if index_int < 0 or index_int >= len(elem_types):
                    raise CompileError(f'type {container_type} has no element at index {index_int}')
                field_type = elem_types[index_int]
            elif isinstance(container_type, sval.ComplexType):
                assert index_int is not None
                if index_int < 0 or index_int > 1:
                    raise CompileError(f'type {container_type} has no field at index {index_int}')
                field_type = container_type.elem
            elif is_array:
                field_type = container_type.elem
            else:
                raise CompileError(f'cannot take a field or element address of {ptr}')
            field_ptr_type = sval.PointerType(field_type, is_const)
            if not field_is_ref and field_type.is_zst():
                # a zero-sized field occupies no storage and has no address
                return ComptimeVal(sval.UndefinedScalar(field_ptr_type))
            match ptr:
                case ComptimeVal(sval.ComptimePtr(box, offset, _)):
                    if isinstance(box, bytes):
                        raise CompileError('cannot take a field address of a byte string')
                    if index_int is None:
                        raise CompileError(
                            'cannot index compile-time storage with a runtime index: '
                            'it has to be a compile-time integer'
                        )
                    field_offset = offset + sval.field_scalar_offset(container_type, index_int)
                    if field_is_ref:
                        # an ``is_ref`` field stores a ``*const T``: the cell holds
                        # the compile-time pointer the read acts through
                        return ComptimeVal(box.cells[field_offset])
                    return ComptimeVal(sval.ComptimePtr(box, field_offset, field_ptr_type))
                case RuntimeVal():
                    if is_array:
                        field_addr = RuntimeVal(self._emit(mir.Gep(ptr.value, _mir_index(index)), at), field_ptr_type)
                    elif isinstance(container_type, (sval.StructType, sval.TupleType, sval.StrDictType)) and container_type.mirror_is_a_field(self._mir_cache):
                        # the mirror of the aggregate is the mirror of its own
                        # single stored field: it takes no address arithmetic
                        field_addr = RuntimeVal(ptr.value, field_ptr_type)
                    else:
                        assert index_int is not None
                        mir_indices = container_type.get_field_mir_indices(self._mir_cache)
                        mir_index = mir_indices[index_int]
                        if mir_index is None:
                            # only the first dynamically-sized field (the struct's
                            # flexible member) has a mirror position; every further
                            # one is inaccessible (see ``get_field_mir_indices``)
                            raise CompileError(
                                f"only the first dynamically-sized field of {container_type} "
                                f"can be accessed"
                            )
                        field_addr = RuntimeVal(self._emit(mir.Gep(ptr.value, mir_index), at), field_ptr_type)
                    if field_is_ref:
                        # the field stores a ``*const T``: load it to get the
                        # pointee address (its constness follows the container's,
                        # so a mutable capture may be written through)
                        return RuntimeVal(self._emit(mir.Load(field_addr.value), at), sval.PointerType(field_type, is_const=is_const))
                    return field_addr
                case _:
                    raise CompileError(f'cannot take field address of {ptr}')

        raise CompileError(f'cannot take a field or element address of {ptr}')

    def _comptime_field_place(self, ptr: InterpVal, container_type: sval.StructType, index: int, field: sval.StructField) -> InterpVal:
        """The place an ``is_comptime`` struct field is read/written through (see
        §2.4/§5.1): the current box cell for a compile-time (box) struct - the box
        is created with the field's default already filled in (see ``_new_box``),
        so a read gives it and a construction can override it - and the declared
        default for a runtime struct (which has no storage for the field at all).
        The fallback place is read-only, so a runtime struct's compile-time field
        cannot be written."""
        if isinstance(ptr, ComptimeVal) and isinstance(ptr.obj, sval.ComptimePtr) and isinstance(ptr.obj.box, sval.Box):
            box = ptr.obj.box
            offset = ptr.obj.offset + sval.field_scalar_offset(container_type, index)
            is_const = ptr.obj.type.is_const or box.is_const
            return ComptimeVal(sval.ComptimePtr(box, offset, sval.PointerType(field.type, is_const=is_const)))
        if field.default is None:
            raise CompileError(
                f"compile-time field '{field.name}' of {container_type} has no value"
            )
        return self._const_box_place(field.default, field.type)

    def _const_box_place(self, value: sval.AnyValue, type: sval.Type) -> InterpVal:
        """A read-only compile-time place holding the constant ``value``: the
        const ``Box`` a compile-time field's default (or a global's value) is read
        through (see §4.3)."""
        box = self._new_box(type, is_const=True)
        box.cells = _flatten_value(value, type)
        return ComptimeVal(sval.ComptimePtr(box, 0, sval.PointerType(type, is_const=True)))

    def _new_box(self, type: sval.Type, is_const: bool) -> sval.Box:
        """A fresh ``Box`` of ``type``: its cells start as the flattened undefined
        and every ``is_comptime`` field cell is then filled with its declared
        default (see ``_fill_comptime_defaults``)."""
        box = sval.Box(self._frames[-1].fn_name, type, is_const=is_const)
        box.reset(type)
        _fill_comptime_defaults(box, type, 0)
        return box

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

    def alloca(self, is_comptime: bool | None = None, is_const: bool = False, declared: sval.Type | None = None, commit_declared: bool = True) -> PendingSlot:
        """Reserve a fresh slot.  ``is_comptime`` is the three-state storage
        decision (see ``PendingSlot``): ``True`` a compile-time variable (a
        ``sval.Box``), ``False`` a runtime location (memory), ``None`` inferred
        from the stores.  A ``declared`` type (an annotated variable, see
        ``_declared_type``) fixes the slot's declared type; ``is_const`` marks a
        ``ConstComptime`` variable (its box becomes read-only after the commit).
        An ordinary annotated variable (``is_comptime=False``) is materialized
        right away (``commit_declared``); a compile-time one commits at its
        ``CommitSlot``."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        slot = PendingSlot(insertion, is_comptime, is_const, self._runtime_position())
        if declared is not None:
            slot.declared = declared
            if commit_declared and is_comptime is False:
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

    def _try_coerce(self, ev: InterpVal, target: sval.Type) -> InterpVal | None:
        """``_coerce`` without the error: ``None`` when the value cannot be
        materialized as ``target`` (see :class:`~spy.errors.CoerceError`).  It
        is what a caller that only has to decide *whether* a coercion is
        possible uses (``exec_ptr_cast``)."""
        try:
            return self._coerce(ev, target)
        except CoerceError:
            return None

    def _coerce(self, ev: InterpVal, target: sval.Type) -> InterpVal:
        """A value of the spy type ``target``: a compile-time value is
        converted with ``sval.coerce_const``, a runtime value gets whatever
        numeric conversion the target needs - widening or narrowing, see
        ``_convert_inst``.  A committed slot is the address of the value it
        holds (``_shallow_normalize``), which is what an operation that takes a
        value without loading it (taking an address, ``ref``) hands over.

        An aggregate has no runtime representation of its own, so a
        compile-time aggregate is only *type-checked* here and returned as it
        is; building the runtime value of one is ``_to_runtime``'s job (the
        callers that need a runtime value wrap the result with it).

        Raises :class:`~spy.errors.CoerceError` when the value has no
        materialization as ``target`` (see ``_try_coerce`` for the failing
        variant)."""
        ev = _shallow_normalize(ev)
        if isinstance(target, sval.OptionType):
            # a ``T``/``Null`` value is coerced through the option's
            # representation (see ``_coerce_option_value``)
            return self._coerce_option_value(ev, target)
        if isinstance(target, sval.TaggedUnionType):
            # a variant value (or a compatible tagged union value) is coerced
            # through the union's representation (see
            # ``_coerce_tagged_union_value``)
            return self._coerce_tagged_union_value(ev, target)
        if isinstance(target, sval.TupleType):
            # a tuple is a real aggregate now (see §7.1): a value or place of the
            # type is its own type
            ev_type = _type_of(ev)
            base = ev_type.elem if isinstance(ev_type, sval.PointerType) else ev_type
            if base == target:
                return ev
            raise CoerceError(f'cannot materialize a {target} from {ev!r}')
        if isinstance(target, sval.StrDictType):
            # likewise for a ``dict[str, T]`` (see §8)
            ev_type = _type_of(ev)
            base = ev_type.elem if isinstance(ev_type, sval.PointerType) else ev_type
            if base == target:
                return ev
            raise CoerceError(f'cannot materialize a {target} from {ev!r}')
        if isinstance(target, sval.ComplexType):
            # a complex number is an aggregate: a real value converts to one with
            # a zero imaginary part and a complex one to a wider element type
            # (see ``_coerce_complex``)
            return self._coerce_complex(ev, target)
        match ev:
            case ComptimeVal(obj=sval.ComptimePtr()):
                # a compile-time pointer as a value of a pointer type: it *is* a
                # pointer already (see ``_type_of``); its pointee is compile-time
                # storage, so becoming an address of real memory happens only
                # where a MIR value is needed (see ``_to_runtime``/§4.3)
                if not isinstance(target, sval.PointerType):
                    raise CoerceError(
                        f'cannot materialize a {target} from a compile-time pointer'
                    )
                return ev
            case ComptimeVal(obj=sval.CastedPtr()):
                # a pointer reinterpreted by ``ptr_cast``: reading or writing
                # through it is not supported yet (see ``sval.CastedPtr``)
                raise CoerceError(
                    f'cannot materialize a {target} from a pointer cast'
                )
            case ComptimeVal(obj) if isinstance(obj, (FunctionValue, sval.DeclareFunction)):
                # a function value or an external declaration *is* a const function
                # pointer (see ``FunctionValue.get_type``): its concrete type is
                # checked where it is lowered (see ``_function_value_to_mir``)
                if not isinstance(target, (sval.PointerType, sval.AnyType)):
                    raise CoerceError(
                        f'cannot materialize a {target} from a function value'
                    )
                return ev
            case ComptimeVal(obj) if isinstance(obj, sval.AggregateValue):
                # an aggregate held as one compile-time object (see
                # ``_as_aggregate``): like an interpreter aggregate value, it is
                # only type-checked and returned (see ``_coerce_aggregate``)
                aggregate = _as_aggregate(ev)
                assert aggregate is not None
                return self._coerce_aggregate(aggregate, target)
            case ComptimeVal(obj) if isinstance(obj, sval.Value):
                # a compile-time object that already *is* a value of the target
                # type (a closure, ...): there is nothing to build a constant
                # from - it is its own value.  Only a compile-time-only type has
                # such a value; a dynamically-sized one (a function type) is
                # still rejected by ``coerce_const``
                value_type = sval.type_of(obj)
                if value_type == target and target.classify() == sval.SpecialTypeKind.COMPTIME:
                    return ev
                return ComptimeVal(sval.coerce_const(obj, target))
            case ComptimeVal(obj):
                return ComptimeVal(sval.coerce_const(obj, target))
            case RuntimeVal(value, type):
                return RuntimeVal(self._convert(value, type, target), target)
            case _:
                raise CoerceError('cannot materialize this value')

    def _coerce_aggregate(self, aggregate: sval.AggregateValue, target: sval.Type) -> InterpVal:
        """A compile-time aggregate as a value of the aggregate type ``target``:
        only the type is checked here - an aggregate is ever only its own type -
        and the value is returned as it is; its runtime representation is built
        by ``_to_runtime`` where one is actually needed."""
        if not _is_aggregate(target) or aggregate.type != target:
            raise CoerceError(f'cannot convert a {aggregate.type} value to {target}')
        return ComptimeVal(aggregate)

    def _coerce_complex(self, ev: InterpVal, target: sval.ComplexType) -> InterpVal:
        """A value of the complex type ``target``: a compile-time value is
        converted with ``sval.coerce_const`` (an ``AggregateValue`` of the type);
        a runtime value is rebuilt part by part - a real (float or integer) value
        becomes a complex one with a zero imaginary part, and a complex one of a
        narrower element type is widened.  This is the implicit ``T ->
        Complex[T]`` conversion.

        Raises :class:`~spy.errors.CoerceError` when the value has no
        materialization as ``target``."""
        ev = _shallow_normalize(ev)
        match ev:
            case ComptimeVal(obj):
                return ComptimeVal(sval.coerce_const(obj, target))
            case RuntimeVal(_, from_type):
                if from_type == target:
                    return ev
                re: InterpVal
                im: InterpVal
                if isinstance(from_type, sval.ComplexType):
                    re, im = self._complex_parts(ev)
                    re = self._coerce(re, target.elem)
                    im = self._coerce(im, target.elem)
                elif sval.is_numeric_type(from_type):
                    # a real value: the real part is the value, the imaginary one
                    # is zero
                    re = self._coerce(ev, target.elem)
                    im = ComptimeVal(sval.coerce_const(0.0, target.elem))
                else:
                    raise CoerceError(f'cannot convert a {from_type} value to {target}')
                return self._complex_value(target, re, im)
            case _:
                raise CoerceError(f'cannot materialize a {target} from {ev!r}')

    def _zst_pointer(self, type: sval.Type) -> mir.Value:
        """The runtime value of a pointer to the zero-sized type ``type``: the
        pointee has no storage, so the pointer is only ever passed around (never
        dereferenced into storage) - it materializes as an undefined pointer."""
        mir_type = sval.PointerType(type, is_const=False).to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.PointerType)
        return mir.UndefValue(mir_type)

    def _to_runtime(self, ev: InterpVal) -> mir.Value:
        """Materialize a value as a typed MIR value: a runtime value yields its
        MIR object, a compile-time value a constant built from it
        (``_sval_to_runtime``), and a committed slot the value it was
        materialized into.  A compile-time aggregate has no MIR value of its own
        (``_aggregate_to_runtime`` builds the ``InsertValue`` chain a by-value
        use needs).  An uncommitted slot, a compile-time box and the aggregate
        of a zero-sized type (which has no runtime value at all) are rejected."""
        ev = _shallow_normalize(ev)
        match ev:
            case RuntimeVal():
                return ev.value
            case ComptimeVal(obj):
                return self._comptime_value_to_runtime(obj)
            case PendingSlot():
                raise CompileError('cannot use an uncommitted slot as a runtime value')
        raise CompileError('cannot return this value')

    def _comptime_value_to_runtime(self, obj: sval.AnyValue) -> mir.Value:
        """The runtime MIR value of a compile-time object: a scalar constant, the
        address of fresh memory a compile-time aggregate/option/union is written
        into, or - for a ``ComptimePtr`` - the automatically lowered address of
        the constant it points at (see §4.3)."""
        if isinstance(obj, sval.ComptimePtr):
            if obj.type.elem.is_zst():
                # a pointer to zero-sized storage: the pointee occupies no
                # storage, so the address is only ever passed around - it
                # materializes as an undefined pointer, like every other
                # zero-sized address
                return self._zst_pointer(obj.type.elem)
            return self._comptime_pointer_value(obj)
        if isinstance(obj, sval.CastedPtr):
            raise CompileError('cannot use a pointer cast as a runtime value')
        if isinstance(obj, sval.AggregateValue):
            return self._aggregate_to_runtime(obj)
        if isinstance(obj, sval.OptionValue):
            option = obj.get_type()
            assert isinstance(option, sval.OptionType)
            return self._option_parts_to_runtime(option, ComptimeVal(False), ComptimeVal(obj.value))
        if isinstance(obj, sval.TaggedUnionValue):
            return self._tagged_union_parts_to_runtime(
                obj.type, ComptimeVal(sval.Int(obj.index, obj.type.tag_type())), ComptimeVal(obj.value),
            )
        if isinstance(obj, (sval.Null, sval.TypedNull)):
            option = sval.type_of(obj)
            if not isinstance(option, sval.OptionType):
                raise CompileError('cannot use a null value as a runtime value')
            child = option.child
            payload = ComptimeVal(child.get_unit_value()) if child.is_zst() else ComptimeVal(sval.UndefinedScalar(child))
            return self._option_parts_to_runtime(option, ComptimeVal(True), payload)
        if isinstance(obj, FunctionValue):
            # a source function value lowers to the address of its compiled
            # specialization (see §4.3)
            return self._function_value_to_mir(obj)
        return _sval_to_runtime(obj, self._mir_cache)

    def _comptime_pointer_value(self, cptr: sval.ComptimePtr) -> mir.Value:
        """The runtime pointer value of a compile-time pointer (see §4.2):

        * a **const view of a const box** lowers to the global constant the box
          holds (see ``_const_pointer_to_mir``);
        * a **const view of a mutable box** is snapshotted into a fresh read-only
          box first, so the lowered global is a constant;
        * a **mutable (non-const) view** materializes a *copy* of the box's
          contents into fresh memory, whose address is the result - a write
          through it is lost (the compile-time variable itself is not aliased)."""
        if cptr.type.is_const or isinstance(cptr.box, bytes):
            if isinstance(cptr.box, sval.Box) and not cptr.box.is_const:
                snapshot = sval.Box(cptr.box.name_base, cptr.box.type, is_const=True)
                snapshot.cells = list(cptr.box.cells)
                cptr = sval.ComptimePtr(snapshot, cptr.offset, cptr.type)
            return self._const_pointer_to_mir(ComptimeVal(cptr), cptr.type)
        mir_type = cptr.type.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.PointerType)
        ptr = self._emit(mir.Alloca(mir_type.elem))
        self.store(RuntimeVal(ptr, cptr.type), self.load(ComptimeVal(cptr)))
        return ptr

    def _function_value_to_mir(self, fn: FunctionValue) -> mir.Value:
        """The runtime ``mir`` value of the function value ``fn``: the in-module
        ``mir.Function`` of its default specialization, or an external symbol when
        it was compiled in an earlier module (see §4.3).  A generic function has no
        default specialization and an inlined Python one is no runtime function at
        all, so both are rejected."""
        if fn.force_inline:
            raise CompileError(
                f'cannot lower {fn.hir.name}: an inlined Python function is not a runtime function'
            )
        sig = fn.hir.signature
        if sig.is_generic():
            raise CompileError(
                f'cannot lower the generic function {fn.hir.name}: specialize it first'
            )
        provided_args: list[ArgNode] = []
        for arg in sig.positional.by_id:
            assert arg.type is not None
            provided_args.append(_provided_node_of_type(arg.type))
        provided = ArgList(tuple(provided_args), (), frozendict())
        call_sig, partial_ret_sig = sig.specialize(provided, self._mir_cache)
        res = self._analyser._request_function(fn, call_sig, partial_ret_sig)
        if res is None:
            raise CompileError(
                f'the function value {fn.hir.name} is not compiled yet; '
                f'call std.core.compile on it first'
            )
        fn_mir, _ret_sig = res
        return fn_mir

    def _builtin_compile(self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """``std.core.compile(fn)``: compile the non-generic function value ``fn``
        ahead of time (its default specialization, built from its declared types),
        so that a later store or conversion of it can be lowered to a
        ``mir.Function`` (see ``_function_value_to_mir``)."""
        obj = _to_comptime(_shallow_normalize(self._arg_value(args.positional[0])))
        if not isinstance(obj, FunctionValue):
            raise CompileError('std.core.compile expects a spy function')
        if obj.force_inline:
            raise CompileError('std.core.compile cannot compile an inlined Python function')
        sig = obj.hir.signature
        if sig.is_generic():
            raise CompileError('std.core.compile does not accept a generic function')
        provided_args: list[ArgNode] = []
        for arg in sig.positional.by_id:
            assert arg.type is not None
            provided_args.append(_provided_node_of_type(arg.type))
        provided = ArgList(tuple(provided_args), (), frozendict())
        call_sig, partial_ret_sig = sig.specialize(provided, self._mir_cache)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            self.store(ret, ComptimeVal(obj))
            return PollResult.AGAIN

        return self._request_function(obj, call_sig, partial_ret_sig, _resumer)

    def _aggregate_to_runtime(self, ev: sval.AggregateValue) -> mir.Value:
        """Build the runtime value of the aggregate ``ev`` from its fields,
        with a chain of ``mir.InsertValue``: a struct mirror that *is* one of
        its fields (see ``sval.StructType.mirror_is_a_field``) yields that
        field's value itself, and a zero-sized aggregate has no runtime value
        at all."""
        aggregate_type = ev.type
        if aggregate_type.is_zst():
            raise CompileError(
                f'a value of the zero-sized {aggregate_type} has no runtime value'
            )
        mir_type = aggregate_type.to_mir_type(self._mir_cache)
        if mir_type is None:
            raise _no_runtime_type(aggregate_type)
        if isinstance(aggregate_type, sval.StructType):
            indices = aggregate_type.get_field_mir_indices(self._mir_cache)
            if aggregate_type.mirror_is_a_field(self._mir_cache):
                # the mirror of the struct is the mirror of its single stored
                # field: the value *is* that field's value
                for index, mir_index in enumerate(indices):
                    if mir_index is not None:
                        return self._to_runtime(ComptimeVal(ev.values[index]))
                raise CompileError(f'{aggregate_type} has no stored field')
            result: mir.Value = mir.UndefValue(mir_type)
            for index, mir_index in enumerate(indices):
                if mir_index is None:
                    continue
                result = self._emit(mir.InsertValue(result, self._to_runtime(ComptimeVal(ev.values[index])), mir_index))
            return result
        # an array/tuple: the elements sit in element order
        result = mir.UndefValue(mir_type)
        for index, value in enumerate(ev.values):
            result = self._emit(mir.InsertValue(result, self._to_runtime(ComptimeVal(value)), index))
        return result

    def _option_parts_to_runtime(self, option: sval.OptionType, is_null: InterpVal, value: InterpVal) -> mir.Value:
        """Build the runtime value of an option from its tag ``is_null`` (which
        may be a runtime value) and its payload ``value``: the two are joined
        through the representation the child chooses (see
        ``sval.OptionType.to_mir_type``).  The ``(bool, T)`` representation
        inserts both into an ``undef`` struct, a zero-sized child only keeps
        whether there is one, and a pointer tag nulls the tagging pointer of the
        payload with a ``mir.Select`` when the option is absent."""
        child = option.child
        null_const = _to_comptime(is_null)
        if child.is_zst():
            # the option *is* the "is there a value" bool
            if isinstance(null_const, bool):
                return mir.BoolValue(not null_const)
            return self._emit(mir.Select(self._to_runtime(is_null), mir.BoolValue(False), mir.BoolValue(True)))
        tag_path = sval.find_first_pointer_type_pos(child)
        if tag_path is None:
            # the (bool, T) representation: the tag and the value are inserted
            # into an undefined struct in turn
            mir_type = option.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.StructType)
            if isinstance(null_const, bool):
                tag: mir.Value = mir.BoolValue(not null_const)
                payload: mir.Value = (
                    mir.UndefValue(mir_type.fields[1].type) if null_const
                    else self._to_runtime(self._coerce(value, child))
                )
            else:
                # the tag is only known at runtime: the payload is always there
                tag = self._not_bool(is_null)
                payload = self._to_runtime(self._coerce(value, child))
            with_tag = self._emit(mir.InsertValue(mir.UndefValue(mir_type), tag, 0))
            return self._emit(mir.InsertValue(with_tag, payload, 1))
        # a pointer tag: the option shares the child's representation
        if isinstance(null_const, bool) and null_const:
            return self._absent_pathed_runtime_option_value(option.child, tag_path)
        present = self._to_runtime(self._coerce(value, child))
        if isinstance(null_const, bool):
            return present
        absent = self._absent_pathed_runtime_option_value(option.child, tag_path)
        return self._emit(mir.Select(self._to_runtime(is_null), absent, present))

    def _absent_pathed_runtime_option_value(self, child: sval.Type, path: tuple[int, ...]) -> mir.Value:
        """The runtime value an absent option of type ``option`` has: the child
        value with the pointer that tags the option (its first pointer, see
        ``sval.find_first_pointer_type_pos``) nulled.  The other fields of the
        child are undefined - nothing may read the payload of an absent
        option."""
        assert path is not None, 'the option has a pointer tag'
        node: sval.Type = child
        # the aggregate mirrors along the path, outermost first, that the value
        # is rebuilt through: a struct whose mirror is a field of its own (see
        # ``mirror_is_a_field``) or an option layer is stepped through without
        # one
        layers: list[tuple[mir.Type, int]] = []
        for index in path:
            if isinstance(node, sval.OptionType):
                node = node.child
                continue
            mir_index: int | None
            if isinstance(node, sval.StructType):
                if node.mirror_is_a_field(self._mir_cache):
                    mir_index = None
                else:
                    mir_index = node.get_field_mir_indices(self._mir_cache)[index]
            elif isinstance(node, sval.ArrayType):
                mir_index = index
            else:
                raise CompileError(f'cannot take the tag pointer of Option[{child}]')
            if mir_index is not None:
                mir_type = node.to_mir_type(self._mir_cache)
                assert mir_type is not None
                layers.append((mir_type, mir_index))
            node = node.get_type_children()[index]
        assert isinstance(node, sval.PointerType)
        mir_type = node.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.PointerType)
        result: mir.Value = mir.NullValue(mir_type)
        for outer_mir_type, mir_index in reversed(layers):
            result = self._emit(mir.InsertValue(mir.UndefValue(outer_mir_type), result, mir_index))
        return result

    def _convert(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Value:
        converted = _convert_inst(value, from_type, to_type, self._mir_cache)
        if converted is None:
            return value
        return self._emit(converted)

    def _operand_comptime_value(self, value: hir.Value) -> sval.AnyValue:
        """The compile-time object an operand denotes (a type value, or the
        value a type parameter stands for): what a ``syntax`` type constructor
        and ``ptr_cast`` take their arguments as."""
        obj = _to_comptime(_shallow_normalize(self.operand(value)))
        if obj is None:
            raise CompileError('a type expression must be a compile-time value')
        return obj

    def type_operand(self, value: hir.Value, what: str | None = None) -> sval.Type:
        """The spy type a type-valued operand denotes; ``what`` names it in the
        error a non-type raises."""
        obj = self._operand_comptime_value(value)
        if not isinstance(obj, sval.Type):
            raise CompileError(f'{what} is not a type: {obj!r}')
        return obj

    def exec_ptr_cast(self, value: InterpVal, target: InterpVal) -> InterpVal:
        """``syntax.ptr_cast(ptr, T)``: reinterpret the pointer ``value`` as the
        pointer type ``T`` names.

        A runtime pointer of another type is ``bit_cast`` (the address is what
        it is - only the pointee type changes, see ``mir.BitCast``) and one of
        the very same type is left alone.  A *compile-time* pointer is
        materialized with ``_coerce`` when its place converts to the target and
        wrapped in a :class:`sval.CastedPtr` otherwise; recasting one of those
        drops the wrapper when the target converts the place it wraps."""
        target_obj = _to_comptime(_shallow_normalize(target))
        if not isinstance(target_obj, sval.PointerType):
            raise CompileError(f'ptr_cast needs a pointer type, got {target_obj!r}')
        from_type = _type_of(value)
        if from_type == target_obj:
            return value
        ev = _shallow_normalize(value)
        if isinstance(ev, RuntimeVal):
            if not isinstance(from_type, sval.PointerType):
                raise CompileError(f'cannot ptr_cast a {from_type} value')
            mir_type = target_obj.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.PointerType)
            return RuntimeVal(self._emit(mir.BitCast(ev.value, mir_type)), target_obj)
        if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.CastedPtr):
            # a pointer already reinterpreted once: when the new target converts
            # the place it wraps, the place itself is the value (the wrapper is
            # dropped); otherwise it is wrapped with the new type
            inner = ev.obj.ptr
            if inner.type.is_subtype_of(target_obj):
                return ComptimeVal(sval.ComptimePtr(inner.box, inner.offset, target_obj))
            return ComptimeVal(sval.CastedPtr(inner, target_obj))
        if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.ComptimePtr):
            # a compile-time pointer: the view type is what changes
            inner = ev.obj
            if inner.type.is_subtype_of(target_obj):
                return ComptimeVal(sval.ComptimePtr(inner.box, inner.offset, target_obj))
            return ComptimeVal(sval.CastedPtr(inner, target_obj))
        if isinstance(ev, ComptimeVal) and isinstance(ev.obj, bytes):
            # a byte string: the pointer points into the byte-string store (see
            # the byte-string lowering, ``_string_global``)
            return ComptimeVal(sval.ComptimePtr(ev.obj, 0, target_obj))
        coerced = self._try_coerce(ev, target_obj)
        if coerced is not None:
            return coerced
        raise CompileError(f'cannot ptr_cast a {from_type} value')

    def exec_make_closure(self, inst: hir.MakeClosure) -> PollResult:
        """Create a closure value (see ``hir.MakeClosure`` and §9): evaluate the
        nested def's parameter annotations, defaults and return annotation into
        the concrete signature, then the capture places, and build an *anonymous
        struct* whose fields are the captures (``_i``) and whose ``__call__`` is
        the closure's own body (with ``self`` prepended to its parameters).  A
        closure is thus an ordinary struct value holding its captures - it can be
        stored, passed, returned and called through ``__call__`` like any other."""
        fn = inst.fn
        positional: IndexedMap[str, SignatureFormalArg] = IndexedMap()
        for i, name in enumerate(fn.param_names):
            declared: sval.Type | None = None
            annotation = inst.annotations[i]
            if annotation is not None:
                declared = self._declared_type(annotation)
            default_value: sval.AnyValue | None = None
            default = inst.defaults[i]
            if default is not None:
                default_value = self._operand_comptime_value(default)
            positional.add(name, SignatureFormalArg(
                declared, fn.is_comptime[i], default_value, TriState.UNKNOWN,
                fn.is_type_value[i],
            ))
        ret_type: sval.Type | None = None
        if inst.ret_annotation is not None:
            ret_type = self.type_operand(inst.ret_annotation, 'the closure return type')
        varargs: SignatureFormalArg | None = None
        if fn.vararg_name is not None:
            vararg_type = (
                None if inst.vararg_annotation is None
                else self._declared_type(inst.vararg_annotation)
            )
            varargs = SignatureFormalArg(
                vararg_type, fn.vararg_is_comptime, None, TriState.UNKNOWN, False,
            )
        kwargs: SignatureFormalArg | None = None
        if fn.kwarg_name is not None:
            kwarg_type = (
                None if inst.kwarg_annotation is None
                else self._declared_type(inst.kwarg_annotation)
            )
            kwargs = SignatureFormalArg(
                kwarg_type, fn.kwarg_is_comptime, None, TriState.UNKNOWN, False,
            )
        if fn.callconv != 'default' and (varargs is not None or kwargs is not None):
            # a C-variadic closure is a C-variadic function *definition*, which
            # is not implemented yet (see ``astgen.parse_function``)
            raise CompileError(
                f'a non-default-callconv closure {fn.name} may not declare *args/**kwargs'
            )
        sig = Signature(
            fn.generic_args, positional, varargs, kwargs, ret_type, fn.exceptions,
            fn.callconv, fn.may_panic,
        )
        # the native name is prefixed with the creating frame's function name
        # (the enclosing specialization, its signature included) - known only
        # here, not at parse time
        name_base = f'{self._frames[-1].fn_name}.{fn.local_name}'
        captures = tuple(self._closure_capture_place(capture) for capture in inst.captures)
        move = inst.move
        # the anonymous struct's fields: one per capture, named ``_i`` (see §9.1).
        # A compile-time capture is baked into the field's type as an
        # ``is_comptime`` field (``move=False`` keeps the *pointer*, dereferenced
        # on a read - ``is_ref``; ``move=True`` bakes the value), a runtime
        # capture is a plain field (``move=False`` stores the pointer - ``is_ref``
        # - ``move=True`` stores a copy)
        head = sval.StructTypeHead(f'{name_base}$closure')
        field_defaults: list[sval.AnyValue | None] = []
        for index, capture in enumerate(captures):
            obj = _to_comptime(_shallow_normalize(capture))
            if obj is not None:
                if isinstance(obj, sval.ComptimePtr):
                    type = obj.type.elem
                else:
                    type = sval.type_of(obj)
                    if type is None:
                        raise CompileError(f'cannot capture the compile-time value {obj!r}')
                by_ref = isinstance(obj, sval.ComptimePtr) and not move
                head.add_field(f'_{index}', type, obj, True, by_ref)
                field_defaults.append(obj)
                continue
            type = _place_type(capture)
            if type is None:
                raise CompileError('cannot capture a value whose type is not known')
            head.add_field(f'_{index}', type, None, False, not move)
            field_defaults.append(None)
        struct_type = head.specialize(())
        # the ``__call__`` method: the closure's own body, with ``self`` (a
        # pointer to the anonymous struct) prepended to its parameters (the body
        # already reads ``Arg(0)`` as its self and the captures as its fields)
        call_positional: IndexedMap[str, SignatureFormalArg] = IndexedMap()
        call_positional.add('self', SignatureFormalArg(
            sval.PointerType(struct_type, is_const=False), False, None, TriState.UNKNOWN, False,
        ))
        for name, param in sig.positional.items():
            call_positional.add(name, param)
        call_sig = Signature(
            sig.generic_args, call_positional, sig.varargs, sig.kwargs, sig.ret_type,
            sig.exceptions, sig.callconv, sig.may_panic,
        )
        call_ir = FunctionIR(f'{name_base}.__call__', call_sig, (True,) + fn.arg_is_ref, fn.body)
        head.methods['__call__'] = FunctionValue(
            f'{name_base}.__call__', call_ir, force_inline=fn.force_inline,
        )
        # the closure value: an instance of the anonymous struct holding the
        # runtime captures (a compile-time one is baked into the field's default)
        is_comptime = struct_type.classify() != sval.SpecialTypeKind.NONE
        place = self._aggregate_place(struct_type, is_comptime=is_comptime)
        for index, capture in enumerate(captures):
            if field_defaults[index] is not None:
                continue
            field0 = struct_type.fields().get_by_id(index)
            if field0.is_ref:
                self.store(self._field_storage_addr(place, struct_type, index), capture)
            else:
                self.store(self.field_index_addr(place, _index_value(index)), self.load(capture))
        regs = self._frames[-1].regs
        regs[inst] = place
        return PollResult.AGAIN

    def _closure_capture_place(self, capture: hir.Value) -> InterpVal:
        """The place a closure captures (see §9.3): an uncommitted *runtime* slot
        (a declared variable a forward reference captures before its own
        ``CommitSlot``) is committed first, so that its storage - the address the
        ``is_ref`` capture field holds - exists when the closure is built."""
        ev = _shallow_normalize(self.operand(capture))
        if isinstance(ev, PendingSlot) and ev.committed is None and ev.is_comptime is False:
            declared = ev.committed_type()
            if declared is not None:
                self._commit_pending_slot(ev, declared)
        return _shallow_normalize(ev)

    def exec_as_func_ptr(self, inst: hir.AsFuncPtr) -> PollResult:
        """``syntax.as_func_ptr(T, closure)``: the runtime function pointer of a
        *closure* - an anonymous struct with a ``__call__`` - as a constant
        function pointer of the closure's own signature.  The first argument is a
        type only the Python type checker reads and is **ignored** (see §9.5).  A
        closure is one struct; the pointer is a *self-less* ``__call__`` built by
        wrapping an undefined instance of the (zero-sized) closure struct, whose
        compile-time captures live in the *type* (see ``_as_func_ptr_wrapper``),
        so it is an ordinary runtime function the automatic lowering turns into a
        ``mir.Function``.  A declared external function already is a pointer and
        is taken as it is."""
        obj = self._operand_comptime_value(inst.obj)
        regs = self._frames[-1].regs
        if isinstance(obj, sval.DeclareFunction):
            # a declared external function already is a function pointer
            regs[inst] = ComptimeVal(obj)
            return PollResult.AGAIN
        struct_type: sval.Type | None = None
        if isinstance(obj, sval.ComptimePtr):
            struct_type = obj.type.elem
        elif isinstance(obj, sval.AggregateValue):
            struct_type = obj.type
        if not isinstance(struct_type, sval.StructType):
            raise CompileError(f'as_func_ptr expects a closure, got {obj!r}')
        method = struct_type.head.methods.get('__call__')
        if method is None:
            raise CompileError(f'as_func_ptr expects a closure (a struct with a __call__), got {struct_type}')
        if not struct_type.is_zst():
            raise CompileError(
                f'as_func_ptr expects a zero-sized closure, but {struct_type} '
                f'captures runtime values'
            )
        if method.force_inline:
            raise CompileError('cannot take the function pointer of an inlined closure')
        wrapper = self._as_func_ptr_wrapper(struct_type, method)
        # compile the wrapper eagerly, so that the automatic lowering can turn
        # the function value into a ``mir.Function`` at its use (see §4.3); the
        # instruction still yields the *value* (a ``ConstPtr[fn_type]``)
        sig = wrapper.hir.signature
        provided_args: list[ArgNode] = []
        for arg in sig.positional.by_id:
            assert arg.type is not None
            provided_args.append(_provided_node_of_type(arg.type))
        provided = ArgList(tuple(provided_args), (), frozendict())
        call_sig, partial_ret_sig = sig.specialize(provided, self._mir_cache)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            regs[inst] = ComptimeVal(wrapper)
            return PollResult.AGAIN

        return self._request_function(wrapper, call_sig, partial_ret_sig, _resumer)

    def _as_func_ptr_wrapper(self, struct_type: sval.StructType, method: FunctionValue) -> FunctionValue:
        """The self-less function ``as_func_ptr`` turns a zero-sized closure into
        (see §9.5, Q2): it has the closure's own parameters, constructs an
        *undefined* instance of the closure struct (``typed_undefined(T)``, whose
        ``is_comptime`` field defaults carry the compile-time captures) and calls
        the closure's ``__call__`` on it, so the captures fold to constants and
        the result is an ordinary runtime function."""
        call_sig = method.hir.signature
        positional: IndexedMap[str, SignatureFormalArg] = IndexedMap()
        for name, formal in call_sig.positional.items():
            if name != 'self':
                positional.add(name, formal)
        sig = Signature(
            call_sig.generic_args, positional, call_sig.varargs, call_sig.kwargs,
            call_sig.ret_type, call_sig.exceptions, call_sig.callconv, call_sig.may_panic,
        )
        unit = struct_type.get_unit_value()
        if unit is None:
            raise CompileError(f'cannot take the function pointer of the non-zero-sized closure {struct_type}')
        slot = hir.Alloca(is_comptime=True)
        args = tuple(ArgEntry(hir.Arg(i), True) for i in range(len(positional.by_id)))
        body: tuple[hir.Inst, ...] = (
            slot,
            hir.Store(slot, hir.Const(unit)),
            hir.CommitSlot(slot),
            hir.CallInplace(slot, hir.CallArgs(args, ()), hir.ReturnLoc()),
            hir.Ret(),
        )
        name = f'{method.name_base}$as_func_ptr'
        hir_fn = FunctionIR(name, sig, method.hir.arg_is_ref[1:], body)
        return FunctionValue(name, hir_fn)

    # -- operators ------------------------------------------------------------

    def _eval_binary(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        # what the operation *is* follows from the types of the operands: a
        # primitive instruction, a struct's magic method (``a + b`` becomes
        # ``a.__add__(b)``, see ``call_method``) or the tagged-union type
        # spelling of ``|``.  The operands are kept as the references they are
        # and a value is only loaded where one is needed.
        lhs_type = _arg_type_of(lhs)
        rhs_type = _arg_type_of(rhs)

        # ``a | b`` is also the tagged-union type spelling: two compile-time
        # type values build the union (the interpreter is what decides which of
        # the two meanings the shared syntax has)
        if (
            op == '|'
            and isinstance(lhs_type, sval.TypeType)
            and isinstance(rhs_type, sval.TypeType)
        ):
            lv = _to_comptime(_shallow_normalize(self._arg_value(lhs)))
            rv = _to_comptime(_shallow_normalize(self._arg_value(rhs)))
            assert isinstance(lv, sval.Type) and isinstance(rv, sval.Type)
            variants: list[sval.Type] = []
            for variant in (lv, rv):
                if isinstance(variant, sval.TaggedUnionType):
                    variants.extend(variant.types)
                else:
                    variants.append(variant)
            self.store(ret, ComptimeVal(sval.tagged_union_of(tuple(variants))))
            return PollResult.AGAIN

        if isinstance(lhs_type, sval.StructType) or isinstance(rhs_type, sval.StructType):
            return self._binary_overload(op, lhs, rhs, lhs_type, rhs_type, ret)

        if isinstance(lhs_type, sval.ComplexType) or isinstance(rhs_type, sval.ComplexType):
            return self._eval_complex(op, lhs, rhs, lhs_type, rhs_type, ret)

        # the operators whose operands do not simply share one peer type are
        # handled on their own: a division promotes to float, an exponent's
        # type follows the base, and a shift's result is the left operand's
        if op == '/':
            return self._eval_divide(lhs, rhs, lhs_type, rhs_type, ret)
        if op == '//':
            return self._eval_floor_divide(lhs, rhs, lhs_type, rhs_type, ret)
        if op == '**':
            return self._eval_pow(lhs, rhs, lhs_type, rhs_type, ret)
        if op in ('|', '&', '^', '<<', '>>'):
            return self._eval_bitwise(op, lhs, rhs, lhs_type, rhs_type, ret)

        if op == '%' and (
            isinstance(lhs_type, sval.FloatType) or isinstance(rhs_type, sval.FloatType)
        ):
            # the float modulo is not implemented: reject it here too, so a
            # compile-time pair does not fold where a runtime one errors
            raise CompileError("unsupported operator '%' for floats")

        if _is_comptime_val(lhs.value) and _is_comptime_val(rhs.value):
            # every operand is compile-time: the operation is evaluated
            # eagerly in Python, whatever the runtime types are
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            assert isinstance(lv, ComptimeVal) and isinstance(rv, ComptimeVal)
            obj = _comptime_py_op(op, lv.obj, rv.obj)
            if (
                lhs_type is not None and rhs_type is not None
                and sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type)
            ):
                peer = lhs_type.resolve_peer_type(rhs_type)
                if peer is not None:
                    obj = sval.coerce_const(obj, peer)
            self.store(ret, ComptimeVal(obj))
            return PollResult.AGAIN

        if lhs_type is None or rhs_type is None:
            raise CompileError(f"cannot apply '{op}' to untyped objects")
        if isinstance(lhs_type, sval.PointerType) or isinstance(rhs_type, sval.PointerType):
            return self._eval_pointer_arith(op, lhs, rhs, lhs_type, rhs_type, ret)
        if sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type):
            lv = self._arg_value(lhs)
            rv = self._arg_value(rhs)
            type = lhs_type.resolve_peer_type(rhs_type)
            if type is None:
                raise CompileError(f"cannot apply '{op}' to {lhs_type} and {rhs_type}")
            if isinstance(type, sval.IntType):
                if op not in ('+', '-', '*', '%'):
                    raise CompileError(f"unsupported operator '{op}' for integers")
            else:
                if op not in ('+', '-', '*'):
                    raise CompileError(f"unsupported operator '{op}' for floats")
            self._fold_or_emit_arith(op, lv, rv, type, ret)
            return PollResult.AGAIN
        raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")

    def _as_float_type(self, a: sval.Type, b: sval.Type) -> sval.FloatType | None:
        """The float type an operation on ``a`` and ``b`` is computed in when
        either operand is a float: the wider of the two, with the integer side
        converted.  ``None`` when neither is a float."""
        if isinstance(a, sval.FloatType) and isinstance(b, sval.FloatType):
            return sval.FloatType(max(a.bits, b.bits))
        if isinstance(a, sval.FloatType):
            return a
        if isinstance(b, sval.FloatType):
            return b
        return None

    def _one_constant(self, type: sval.Type) -> mir.Value:
        """The multiplicative identity of the numeric type ``type``."""
        mir_type = type.to_mir_type(self._mir_cache)
        assert mir_type is not None and not type.is_zst()
        if isinstance(mir_type, mir.FloatType):
            return mir.Float(1.0, mir_type)
        assert isinstance(mir_type, mir.IntType)
        return mir.Int(1, mir_type)

    def _fold_or_emit_arith(self, op: BinaryOp, lv: InterpVal, rv: InterpVal, target: sval.Type, ret: InterpVal) -> None:
        """Compute ``lv op rv`` as ``target`` and store it into ``ret`` (see
        ``_fold_or_emit_arith_value``)."""
        self.store(ret, self._fold_or_emit_arith_value(op, lv, rv, target))

    def _fold_or_emit_arith_value(self, op: BinaryOp, lv: InterpVal, rv: InterpVal, target: sval.Type) -> InterpVal:
        """Compute ``lv op rv`` as ``target``: a compile-time pair folds in
        Python (and the result is re-tagged as ``target``), a runtime one is a
        ``mir.Arith`` over operands coerced to ``target``."""
        if _is_comptime_val(lv) and _is_comptime_val(rv):
            lobj = _to_comptime(lv)
            robj = _to_comptime(rv)
            assert lobj is not None and robj is not None
            obj = _comptime_py_op(op, lobj, robj)
            return ComptimeVal(sval.coerce_const(obj, target))
        lc = self._coerce(lv, target)
        rc = self._coerce(rv, target)
        mir_type = target.to_mir_type(self._mir_cache)
        assert mir_type is not None and not target.is_zst()
        value = self._emit(mir.Arith(op, self._to_runtime(lc), self._to_runtime(rc), mir_type))
        return RuntimeVal(value, target)

    def _eval_complex(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """``a op b`` with a complex operand: one of the four arithmetic
        operations on complex numbers, computed part by part (inline
        ``mir.Arith`` on the real and the imaginary part).  A real (float or
        integer) operand is taken as a complex one with a zero imaginary part,
        widened to the other operand's element type (the implicit ``T ->
        Complex[T]`` conversion).

        Both operands are first turned into their complex number form (see
        ``_complex_parts``), so a compile-time pair folds and a runtime one emits
        the arithmetic."""
        if lhs_type is None or rhs_type is None:
            raise CompileError(f"cannot apply '{op}' to untyped objects")
        if op not in ('+', '-', '*', '/'):
            raise CompileError(f"unsupported operator '{op}' for complex numbers")
        complex_type = lhs_type if isinstance(lhs_type, sval.ComplexType) else rhs_type
        other = rhs_type if isinstance(lhs_type, sval.ComplexType) else lhs_type
        assert isinstance(complex_type, sval.ComplexType)
        target = complex_type.resolve_peer_type(other)
        if not isinstance(target, sval.ComplexType):
            raise CompileError(f"cannot apply '{op}' to {lhs_type} and {rhs_type}")
        elem = target.elem

        # the values are turned into their two parts first: the real and
        # imaginary parts of a compile-time value are compile-time, those of a
        # runtime one are read out of the value itself (see ``_complex_parts``)
        a = self._complex_parts(self._coerce(self._arg_value(lhs), target))
        b = self._complex_parts(self._coerce(self._arg_value(rhs), target))
        ar, ai = a
        br, bi = b

        def arith(op: BinaryOp, x: InterpVal, y: InterpVal) -> InterpVal:
            return self._fold_or_emit_arith_value(op, x, y, elem)

        re: InterpVal
        im: InterpVal
        if op == '+':
            re = arith('+', ar, br)
            im = arith('+', ai, bi)
        elif op == '-':
            re = arith('-', ar, br)
            im = arith('-', ai, bi)
        elif op == '*':
            # (ar + ai i)(br + bi i) = (ar br - ai bi) + (ar bi + ai br) i
            re = arith('-', arith('*', ar, br), arith('*', ai, bi))
            im = arith('+', arith('*', ar, bi), arith('*', ai, br))
        else:
            # a / b = a * conj(b) / |b|^2
            denom = arith('+', arith('*', br, br), arith('*', bi, bi))
            re = arith('/', arith('+', arith('*', ar, br), arith('*', ai, bi)), denom)
            im = arith('/', arith('-', arith('*', ai, br), arith('*', ar, bi)), denom)
        self.store(ret, self._complex_value(target, re, im))
        return PollResult.AGAIN

    def _eval_divide(self, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """``a / b``: true division.  A float operand picks the wider float
        type; two integers are divided in ``CompileVars.int_div_type`` (f64 by
        default), so ``a / b`` of two integers is a float, like Python's."""
        target: sval.Type | None = None
        if lhs_type is not None and rhs_type is not None:
            target = self._as_float_type(lhs_type, rhs_type)
            if target is None and sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type):
                target = self._compile_vars.int_div_type
        if target is None:
            raise CompileError(f"cannot apply '/' to {lhs_type} and {rhs_type}")
        self._fold_or_emit_arith('/', self._arg_value(lhs), self._arg_value(rhs), target, ret)
        return PollResult.AGAIN

    def _eval_floor_divide(self, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """``a // b``.  A float operand floors the float quotient; two integers
        floor the integer quotient (or truncate it when
        ``CompileVars.int_trunc_div`` says so)."""
        ftype: sval.FloatType | None = None
        if lhs_type is not None and rhs_type is not None:
            ftype = self._as_float_type(lhs_type, rhs_type)
        lv = self._arg_value(lhs)
        rv = self._arg_value(rhs)
        if ftype is not None:
            if _is_comptime_val(lv) and _is_comptime_val(rv):
                a = _comptime_py_value(_to_comptime(lv))
                b = _comptime_py_value(_to_comptime(rv))
                try:
                    result = float(math.floor(a / b))
                except Exception as e:
                    raise CompileError(f"cannot apply '//' at compile time: {e}") from e
                self.store(ret, ComptimeVal(sval.coerce_const(result, ftype)))
                return PollResult.AGAIN
            lc = self._coerce(lv, ftype)
            rc = self._coerce(rv, ftype)
            mir_type = ftype.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.FloatType)
            quotient = self._emit(mir.Arith('/', self._to_runtime(lc), self._to_runtime(rc), mir_type))
            self.store(ret, RuntimeVal(self._emit(mir.Floor(quotient)), ftype))
            return PollResult.AGAIN

        if not (
            lhs_type is not None and rhs_type is not None
            and sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type)
        ):
            raise CompileError(f"cannot apply '//' to {lhs_type} and {rhs_type}")
        target = lhs_type.resolve_peer_type(rhs_type)
        if not isinstance(target, (sval.IntType, sval.AnyIntType)):
            raise CompileError(f"cannot apply '//' to {lhs_type} and {rhs_type}")

        if _is_comptime_val(lv) and _is_comptime_val(rv):
            a = _comptime_py_value(_to_comptime(lv))
            b = _comptime_py_value(_to_comptime(rv))
            if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool) or isinstance(b, bool):
                raise CompileError(f"cannot apply '//' to {a!r} and {b!r} at compile time")
            if b == 0:
                raise CompileError("integer division by zero at compile time")
            result = _truncate_div(a, b) if self._compile_vars.int_trunc_div else a // b
            self.store(ret, ComptimeVal(sval.coerce_const(result, target)))
            return PollResult.AGAIN

        if self._compile_vars.int_trunc_div:
            self._fold_or_emit_arith('/', lv, rv, target, ret)
            return PollResult.AGAIN

        # only two untyped compile-time literals could leave ``target`` as the
        # compile-time-only ``AnyIntType``, and they folded above
        assert isinstance(target, sval.IntType)
        lc = self._coerce(lv, target)
        rc = self._coerce(rv, target)
        mir_type = target.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.IntType)
        lc_runtime = self._to_runtime(lc)
        rc_runtime = self._to_runtime(rc)
        quotient = self._emit(mir.Arith('/', lc_runtime, rc_runtime, mir_type))
        if target.signed:
            # floor division: correct the truncating quotient when the remainder
            # is non-zero and its sign differs from the divisor's
            remainder = self._emit(mir.Arith('%', lc_runtime, rc_runtime, mir_type))
            zero = mir.Int(0, mir_type)
            r_nonzero = self._emit(mir.Cmp('!=', remainder, zero))
            r_negative = self._emit(mir.Cmp('<', remainder, zero))
            d_negative = self._emit(mir.Cmp('<', rc_runtime, zero))
            signs_differ = self._emit(mir.Cmp('!=', r_negative, d_negative))
            adjust = self._emit(mir.Select(r_nonzero, signs_differ, mir.BoolValue(False)))
            decremented = self._emit(mir.Arith('-', quotient, mir.Int(1, mir_type), mir_type))
            quotient = self._emit(mir.Select(adjust, decremented, quotient))
        self.store(ret, RuntimeVal(quotient, target))
        return PollResult.AGAIN

    def _eval_pow(self, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """``a ** b``.

        A compile-time integer exponent is unfolded (exponentiation by
        squaring) in the base's own type when non-negative, or in ``f64``
        followed by a reciprocal when negative; one too large to unfold
        (``CompileVars.max_exp_unroll``) becomes a runtime loop.  A runtime
        integer exponent is a runtime loop in ``f64`` (its sign decides
        whether a reciprocal follows).  A float exponent is a ``mir.Pow`` over
        floats."""
        if lhs_type is None or rhs_type is None:
            raise CompileError("cannot apply '**' to untyped objects")
        if not sval.is_numeric_type(lhs_type):
            raise CompileError(f"cannot raise a {lhs_type} to a power")
        if not sval.is_numeric_type(rhs_type):
            raise CompileError(f"cannot use a {rhs_type} as an exponent")

        lv = self._arg_value(lhs)
        rv = self._arg_value(rhs)
        exponent = _comptime_int(rv)

        if _is_comptime_val(lv) and _is_comptime_val(rv):
            # both compile-time: fold in Python (a negative exponent yields a
            # float, exactly like the runtime rule)
            lobj = _to_comptime(lv)
            robj = _to_comptime(rv)
            assert lobj is not None and robj is not None
            result = _comptime_py_op('**', lobj, robj)
            target: sval.Type = lhs_type if exponent is not None and exponent >= 0 else sval.FloatType(64)
            self.store(ret, ComptimeVal(sval.coerce_const(result, target)))
            return PollResult.AGAIN

        if exponent is not None:
            if abs(exponent) > self._compile_vars.max_exp_unroll:
                # too large to unfold: a runtime loop with a constant count
                count_type = rhs_type if isinstance(rhs_type, sval.IntType) else sval.IntType(64, exponent < 0)
                self._pow_loop(lv, ComptimeVal(sval.coerce_const(exponent, count_type)), count_type, ret)
                return PollResult.AGAIN
            if exponent >= 0:
                base_type: sval.Type = lhs_type
                if isinstance(base_type, sval.AnyIntType):
                    # an untyped literal has no runtime type of its own
                    base_type = sval.FloatType(64)
                self._pow_unrolled(lv, exponent, base_type, False, ret)
            else:
                self._pow_unrolled(lv, exponent, sval.FloatType(64), True, ret)
            return PollResult.AGAIN

        if isinstance(rhs_type, sval.IntType):
            self._pow_loop(lv, rv, rhs_type, ret)
            return PollResult.AGAIN

        if isinstance(rhs_type, sval.FloatType):
            target = self._as_float_type(lhs_type, rhs_type) or sval.FloatType(64)
            lc = self._coerce(lv, target)
            rc = self._coerce(rv, target)
            mir_type = target.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.FloatType)
            value = self._emit(mir.Pow(self._to_runtime(lc), self._to_runtime(rc), mir_type))
            self.store(ret, RuntimeVal(value, target))
            return PollResult.AGAIN

        raise CompileError(f"cannot use a {rhs_type} as an exponent")

    def _pow_unrolled(self, base: InterpVal, exponent: int, type: sval.Type, negative: bool, ret: InterpVal) -> None:
        """``base ** exponent`` with a compile-time exponent: exponentiation by
        squaring in ``type`` (``f64`` for a negative exponent), followed by a
        reciprocal when ``negative``."""
        mir_type = type.to_mir_type(self._mir_cache)
        assert mir_type is not None and not type.is_zst()
        power = self._to_runtime(self._coerce(base, type))
        result = self._one_constant(type)
        remaining = abs(exponent)
        while remaining > 0:
            if remaining & 1:
                result = self._emit(mir.Arith('*', result, power, mir_type))
            remaining >>= 1
            if remaining > 0:
                power = self._emit(mir.Arith('*', power, power, mir_type))
        if negative:
            result = self._emit(mir.Arith('/', self._one_constant(sval.FloatType(64)), result, mir_type))
        self.store(ret, RuntimeVal(result, type))

    def _pow_loop(self, base: InterpVal, exponent: InterpVal, exponent_type: sval.IntType, ret: InterpVal) -> None:
        """``base ** exponent`` with a runtime integer exponent: an ``f64`` loop
        that squares the running power and multiplies the running result by it
        whenever the current bit of the exponent is set.  A signed exponent is
        made absolute first and the result is reciprocated at the end when it
        was negative; an unsigned one loops directly.  The loop's carried
        values are ``mir.Phi``s, so no alloca is needed."""
        f64 = sval.FloatType(64)
        mir_f64 = mir.FloatType(64)
        base_runtime = self._to_runtime(self._coerce(base, f64))
        count_mir = exponent_type.to_mir_type(self._mir_cache)
        assert isinstance(count_mir, mir.IntType)
        count = self._to_runtime(self._coerce(exponent, exponent_type))
        zero = mir.Int(0, count_mir)
        one_i = mir.Int(1, count_mir)
        negative: mir.Value
        if exponent_type.signed:
            negative = self._emit(mir.Cmp('<', count, zero))
            negative_count = self._emit(mir.Arith('-', zero, count, count_mir))
            count = self._emit(mir.Select(negative, negative_count, count))
        else:
            negative = mir.BoolValue(False)

        preheader = self._cur_block
        header = mir.BasicBlock()
        body = mir.BasicBlock()
        exit_block = mir.BasicBlock()
        preheader.emit(mir.Jmp(header))

        e_phi = mir.Phi([(count, preheader)])
        acc_phi = mir.Phi([(mir.Float(1.0, mir_f64), preheader)])
        base_phi = mir.Phi([(base_runtime, preheader)])
        header.emit(e_phi)
        header.emit(acc_phi)
        header.emit(base_phi)
        header.emit(mir.Br(header.emit(mir.Cmp('!=', e_phi, zero)), body, exit_block))

        bit = body.emit(mir.Arith('&', e_phi, one_i, count_mir))
        factor = body.emit(mir.Select(body.emit(mir.Cmp('!=', bit, zero)), base_phi, mir.Float(1.0, mir_f64)))
        next_acc = body.emit(mir.Arith('*', acc_phi, factor, mir_f64))
        next_base = body.emit(mir.Arith('*', base_phi, base_phi, mir_f64))
        next_e = body.emit(mir.Arith('>>', e_phi, one_i, count_mir))
        body.emit(mir.Jmp(header))
        e_phi.add_incoming(next_e, body)
        acc_phi.add_incoming(next_acc, body)
        base_phi.add_incoming(next_base, body)

        self._cur_block = exit_block
        reciprocal = self._emit(mir.Arith('/', mir.Float(1.0, mir_f64), acc_phi, mir_f64))
        self.store(ret, RuntimeVal(self._emit(mir.Select(negative, reciprocal, acc_phi)), f64))

    def _eval_bitwise(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """``a op b`` for the integer bitwise/shift operators.  A shift keeps
        the left operand's type (the count is converted to it); the others use
        the operands' peer type, like arithmetic."""
        if lhs_type is None or rhs_type is None:
            raise CompileError(f"cannot apply '{op}' to untyped objects")
        if not (sval.is_numeric_type(lhs_type) and sval.is_numeric_type(rhs_type)):
            raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")
        target: sval.Type | None
        if op in ('<<', '>>'):
            if isinstance(lhs_type, sval.IntType):
                target = lhs_type
            elif isinstance(rhs_type, sval.IntType):
                target = rhs_type
            else:
                target = lhs_type.resolve_peer_type(rhs_type)
        else:
            target = lhs_type.resolve_peer_type(rhs_type)
        if not isinstance(target, (sval.IntType, sval.AnyIntType)):
            raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")
        self._fold_or_emit_arith(op, self._arg_value(lhs), self._arg_value(rhs), target, ret)
        return PollResult.AGAIN

    def _binary_overload(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret: InterpVal) -> PollResult:
        """A binary operator with a struct operand: the left operand's forward
        magic method, or the right one's reflected method with the operands
        swapped when the left names none."""
        forward, reflected, _ = _BINARY_METHODS[op]
        if isinstance(lhs_type, sval.StructType) and self._resolve_method(lhs_type, forward) is not None:
            return self.call_method(self._operand_place(lhs), forward, RawArgList((rhs,), frozendict()), ret)
        if isinstance(rhs_type, sval.StructType) and self._resolve_method(rhs_type, reflected) is not None:
            return self.call_method(self._operand_place(rhs), reflected, RawArgList((lhs,), frozendict()), ret)
        raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")

    def _operand_place(self, arg: ArgEntry[InterpVal]) -> InterpVal:
        """The place a struct operand is addressed by: a reference argument is
        the address it already carries, any other value is written into a fresh
        slot first (like ``astgen._as_ref``)."""
        if arg.is_ref:
            return arg.value
        slot = self.alloca(None)
        self.store(slot, arg.value)
        self._commit_pending_slot(slot)
        return _shallow_normalize(slot)

    def _eval_pointer_arith(self, op: BinaryOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type, rhs_type: sval.Type, ret: InterpVal) -> PollResult:
        """One operation on a pointer: ``mptr + n``, the address of the n-th
        element after the one the pointer carries.  Only a *multi* pointer may
        be offset (a single one names one place only, see ``subscript``), and
        the index is a signed pointer-width integer, so a negative offset walks
        backwards like C's.  It is what makes ``ref(mptr[n]) == mptr + n``."""
        if not isinstance(lhs_type, sval.PointerType):
            raise CompileError(f"cannot add a pointer to {lhs_type}")
        if lhs_type.variant != sval.PointerVariant.MULTI:
            raise CompileError(
                f"cannot apply '{op}' to {lhs_type}: only a multi pointer "
                f"supports pointer arithmetic"
            )
        if op != '+':
            raise CompileError(f"unsupported pointer operator '{op}'")
        if not isinstance(rhs_type, (sval.IntType, sval.AnyIntType)):
            raise CompileError(f'cannot offset a pointer by {rhs_type}')
        base = self._arg_value(lhs)
        if not isinstance(base, RuntimeVal):
            raise CompileError('cannot offset a compile-time pointer')
        offset = self._to_runtime(self._coerce(self._arg_value(rhs), self._isize_type()))
        self.store(ret, RuntimeVal(self._emit(mir.Gep(base.value, offset)), lhs_type))
        return PollResult.AGAIN

    def _eval_cmp(self, op: CompareOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        lhs_type = _arg_type_of(lhs)
        rhs_type = _arg_type_of(rhs)

        if isinstance(lhs_type, sval.StructType) or isinstance(rhs_type, sval.StructType):
            return self._cmp_overload(op, lhs, rhs, lhs_type, rhs_type, ret_reg)

        if isinstance(lhs_type, sval.IntEnumType) or isinstance(rhs_type, sval.IntEnumType):
            return self._eval_enum_cmp(op, lhs, rhs, lhs_type, rhs_type, ret_reg)

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
            value = self._emit(
                mir.Cmp(op, self._to_runtime(lc), self._to_runtime(rc))
            )
            self._frames[-1].regs[ret_reg] = RuntimeVal(value, sval.BoolType())
            return PollResult.AGAIN
        raise CompileError(f'unsupported operand types: {lhs_type} and {rhs_type}')

    def _eval_enum_cmp(self, op: CompareOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret_reg: hir.Inst) -> PollResult:
        """``a == b`` / ``a != b`` of two values of the *same* integer
        enumeration.  An enumeration is not an integer: it is only ever compared
        to its own type (another enumeration or an ``int`` is rejected), and only
        for equality (it has no order)."""
        if not (
            isinstance(lhs_type, sval.IntEnumType)
            and isinstance(rhs_type, sval.IntEnumType)
            and lhs_type is rhs_type
        ):
            raise CompileError(f'cannot compare {lhs_type} and {rhs_type}')
        if op not in ('==', '!='):
            raise CompileError(
                f"unsupported comparison '{op}' for the enumeration {lhs_type}"
            )
        lv = self._arg_value(lhs)
        rv = self._arg_value(rhs)
        if _is_comptime_val(lv) and _is_comptime_val(rv):
            # both compile-time: the two members are compared by their index in
            # the enumeration (the value is read off the type)
            lo = _to_comptime(lv)
            ro = _to_comptime(rv)
            assert lo is not None and ro is not None
            equal = _enum_member_index(lhs_type, lo) == _enum_member_index(lhs_type, ro)
            self._frames[-1].regs[ret_reg] = ComptimeVal(equal if op == '==' else not equal)
            return PollResult.AGAIN
        lc = self._coerce(lv, lhs_type)
        rc = self._coerce(rv, lhs_type)
        value = self._emit(
            mir.Cmp(op, self._to_runtime(lc), self._to_runtime(rc))
        )
        self._frames[-1].regs[ret_reg] = RuntimeVal(value, sval.BoolType())
        return PollResult.AGAIN

    def _cmp_overload(self, op: CompareOp, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], lhs_type: sval.Type | None, rhs_type: sval.Type | None, ret_reg: hir.Inst) -> PollResult:
        """A comparison with a struct operand: the left operand's magic method,
        or the right one's reflected method with the operands swapped when the
        left names none (``a < b`` becomes ``b > a``).  The method's boolean
        result is written into the comparison's register (``Compare`` produces
        a register, not a result location, so the call delivers into a fresh
        slot that is then loaded)."""
        forward, reflected = _COMPARE_METHODS[op]
        base: ArgEntry[InterpVal]
        other: ArgEntry[InterpVal]
        name: str
        if isinstance(lhs_type, sval.StructType) and self._resolve_method(lhs_type, forward) is not None:
            base, other, name = lhs, rhs, forward
        elif isinstance(rhs_type, sval.StructType) and self._resolve_method(rhs_type, reflected) is not None:
            base, other, name = rhs, lhs, reflected
        else:
            raise CompileError(f'unsupported operand types: {lhs_type} and {rhs_type}')
        slot = self.alloca(None)
        regs = self._frames[-1].regs

        def on_return() -> None:
            self._commit_pending_slot(slot)
            regs[ret_reg] = self.load(slot)

        return self.call_method(
            self._operand_place(base), name, RawArgList((other,), frozendict()), slot, on_return,
        )

    def _eval_not(self, operand: InterpVal, ret: hir.Inst) -> PollResult:
        """Boolean negation (``hir.Not``, value -> value): the negation of the
        boolean *value* the operand holds (the ``AsBool`` of the source operand,
        so it is already a ``bool``).  A compile-time operand folds in Python, a
        runtime one becomes a comparison against ``false``."""
        if _is_comptime_val(operand):
            assert isinstance(operand, ComptimeVal)
            self._frames[-1].regs[ret] = ComptimeVal(not operand.obj)
            return PollResult.AGAIN
        type = _type_of(operand)
        if not isinstance(type, sval.BoolType):
            raise CompileError(f"cannot apply 'not' to a {type} value")
        coerced = self._coerce(operand, type)
        value = self._emit(
            mir.Cmp('==', self._to_runtime(coerced), mir.BoolValue(False))
        )
        self._frames[-1].regs[ret] = RuntimeVal(value, sval.BoolType())
        return PollResult.AGAIN

    def _eval_unary(self, op: UnaryOp, operand: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        type = _arg_type_of(operand)
        if isinstance(type, sval.StructType):
            name = _UNARY_METHODS.get(op)
            if name is not None and self._resolve_method(type, name) is not None:
                return self.call_method(self._operand_place(operand), name, RawArgList((), frozendict()), ret)

        if _is_comptime_val(operand.value):
            ev = self._arg_value(operand)
            assert isinstance(ev, ComptimeVal)
            obj = ev.obj
            if op == '-':
                negated = sval.negate(obj)
                if negated is None:
                    raise CompileError(f'cannot negate {obj!r} at compile time')
                self.store(ret, ComptimeVal(negated))
                return PollResult.AGAIN
            if op == '~':
                if isinstance(obj, sval.Int):
                    complemented = ~obj.value if obj.type.signed else (~obj.value) & ((1 << obj.type.bits) - 1)
                    self.store(ret, ComptimeVal(sval.Int(complemented, obj.type)))
                elif isinstance(obj, int) and not isinstance(obj, bool):
                    self.store(ret, ComptimeVal(~obj))
                else:
                    raise CompileError(f'cannot complement {obj!r} at compile time')
                return PollResult.AGAIN
            raise CompileError(f"unsupported unary operator '{op}'")

        if type is None:
            raise CompileError(f"cannot apply unary '{op}' to a value that has no type yet")
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
                mir.Arith('-', zero, self._to_runtime(coerced), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        if op == '~':
            if not isinstance(type, sval.IntType):
                raise CompileError(f'cannot complement a {type} value')
            mir_type = type.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.IntType)
            coerced = self._coerce(self._arg_value(operand), type)
            value = self._emit(
                mir.Arith('^', self._to_runtime(coerced), mir.Int(-1, mir_type), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        raise CompileError(f"unsupported unary operator '{op}'")

    def _eval_ord(self, value: InterpVal, ret: InterpVal) -> PollResult:
        """``ord(x)`` (``hir.Ord``): the encoding of the byte the compile-time
        byte string ``value`` holds.  The operand has to be a compile-time
        ``bytes`` of exactly one byte; the result is an *untyped* integer, so the
        result location gives it its type."""
        obj = _to_comptime(_shallow_normalize(value))
        if not isinstance(obj, bytes):
            type = _type_of(value)
            raise CompileError(
                f'ord expects a compile-time byte string, got '
                f'{type if type is not None else value!r}'
            )
        if len(obj) != 1:
            raise CompileError(
                f'ord expects a byte string of exactly one byte, got {len(obj)}'
            )
        self.store(ret, ComptimeVal(obj[0]))
        return PollResult.AGAIN

    def binary_assign(self, op: BinaryOp, left: InterpVal, right: ArgEntry[InterpVal]) -> PollResult:
        # ``x op= y`` calls the target's in-place magic method when its struct
        # declares one (``__iadd__``, ...); otherwise it is ``x = x op y``
        left_type = _place_type(left)
        if isinstance(left_type, sval.TupleType):
            # a tuple ``+=`` is not supported: a tuple now has a runtime
            # representation, so there is no in-place extension to speak of (see
            # §7.5) - accumulate into a tuple type built field by field instead
            raise CompileError('a tuple does not support ``+=``')
        if isinstance(left_type, sval.StructType):
            _, _, inplace = _BINARY_METHODS[op]
            if self._resolve_method(left_type, inplace) is not None:
                return self.call_method(left, inplace, RawArgList((right,), frozendict()), left)
        return self._eval_binary(op, ArgEntry(left, True), right, left)

    # -- calls ----------------------------------------------------------------

    def call(self, callee: InterpVal, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal, on_return: Callable[[], None] | None = None) -> PollResult:
        """Resolve one call by its callee value and run it.  Spy
        functions compile to a native ``call`` producing a typed
        register, plain Python functions are inlined, and the spy
        builtins are evaluated at compile time.  The callee constant of a
        registered spy function already resolved to its entry when the
        callee operand was evaluated (see ``operand``).

        ``on_return`` is invoked once the callee delivered its result
        (see ``_call_function_entry``); None when the caller wants nothing
        more.

        Returns ``PollResult.AGAIN`` when the call completed here (an
        inlined callee's body writes into the result location directly),
        or ``PollResult.SUSPEND`` when the callee's specialization was
        just started and must be typed first: the call is then completed
        by ``resume`` when that runner ends (see
        ``_call_function_entry``)."""
        callee = self._auto_deref(callee)
        target = _callee_object(callee)
        if target is not None:
            if isinstance(target, FunctionValue):
                return self._call_function_entry(target, args, ret, on_return=on_return)
            if isinstance(target, sval.BoundMethod):
                # a method of a generic struct resolved from a value: the
                # struct's type-argument values are substituted into the
                # method's signature (see ``_call_function_entry``)
                fn = target.fn
                assert isinstance(fn, FunctionValue), 'a bound method holds a function value'
                return self._call_function_entry(fn, args, ret, target.generic_var_values, on_return=on_return)
            if isinstance(target, sval.BuiltinFn):
                res = self._call_builtin(target, args, ret)
                if res == PollResult.AGAIN and on_return is not None:
                    on_return()
                return res
        callee_type = _type_of(callee)
        # a call of a *struct value* through its ``__call__`` method (a closure is
        # such a struct, see §9): the callee is a place, so its type is a pointer
        # to the struct, and the address is the receiver the method is called on
        if (
            isinstance(callee_type, sval.PointerType)
            and isinstance(callee_type.elem, sval.StructType)
            and callee_type.elem.get_method('__call__') is not None
        ):
            return self.call_method(callee, '__call__', args, ret, on_return)
        # a runtime function pointer: a value of a pointer-to-function type,
        # called through the pointer the value carries (a function type alone
        # is dynamically sized, so only a pointer to one is a value)
        if isinstance(callee_type, sval.PointerType) and isinstance(callee_type.elem, sval.FunctionType):
            return self._call_fn_ptr(callee, callee_type.elem, args, ret, on_return)
        raise CompileError(
            f"cannot compile a call to {callee!r}; only spy functions, plain Python "
            "functions and the spy builtins can be called"
        )

    def _call_builtin(
        self, fn: sval.BuiltinFn, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """Evaluate one ``spy.*`` builtin at compile time and hand its
        result to the call's result location.  The arguments are first bound
        to the builtin's signature (see ``sval.BuiltinSignature.bind``), so
        every handler below reads a normalized ``ArgList``: exactly one entry
        per declared positional parameter (defaults filled in), the excess
        positionals in ``varargs`` and the trailing keywords in ``kwargs``."""
        binded = fn.signature.bind(args, lambda e: ArgEntry(ComptimeVal(e), False))
        if fn.name == 'compile_log':
            parts: list[str] = []
            for arg in binded.varargs:
                ev = self._arg_value(arg)
                obj = _to_comptime(ev)
                if obj is None:
                    type = _type_of(ev)
                    parts.append(f'<{type}>' if type is not None else '<value>')
                elif isinstance(obj, bytes):
                    # a byte string is rendered as its text, not as a ``b'...'`` repr
                    parts.append(obj.decode(errors='replace'))
                else:
                    parts.append(str(obj))
            for arg in binded.kwargs.values():
                ev = self._arg_value(arg)
                obj = _to_comptime(ev)
                if isinstance(obj, bytes):
                    parts.append(obj.decode(errors='replace'))
                else:
                    parts.append(str(obj) if obj is not None else '<value>')
            print(' '.join(parts))
            self.store(ret, ComptimeVal(sval.Void()))
            return PollResult.AGAIN
        if fn.name == 'type_info':
            obj = _to_comptime(_shallow_normalize(self._arg_value(binded.positional[0])))
            if not isinstance(obj, sval.Type):
                raise CompileError(f'spy.type_info takes a type, got {obj!r}')
            self.store(ret, self._build_type_info(obj))
            return PollResult.AGAIN
        if fn.name == 'reify':
            return self._builtin_reify(binded, ret)
        if fn.name == 'undefined':
            # ``std.core.undefined``: the undefined literal, the value of any
            # type (see ``sval.UndefinedType``); the store into the result
            # location coerces it to that location's type
            self.store(ret, ComptimeVal(sval.UntypedUndefined()))
            return PollResult.AGAIN
        if fn.name == 'layout_of':
            return self._layout_builtin(binded, ret)
        if fn.name == 'panic':
            return self._builtin_panic(binded)
        if fn.name == 'catch_unwind':
            return self._builtin_catch_unwind(binded, ret)
        if fn.name == 'compile':
            return self._builtin_compile(binded, ret)
        if fn.name == 'const_array':
            return self._builtin_const_array(binded, ret)
        if fn.name == 'bitcast':
            return self._builtin_bitcast(binded, ret)
        if fn.name == 'truncate':
            return self._builtin_truncate(binded, ret)
        if fn.name == 'coerce':
            return self._builtin_coerce(binded, ret)
        if fn.name == 'compile_error':
            return self._builtin_compile_error(binded)
        if fn.name == 'atomic_load':
            return self._builtin_atomic_load(binded, ret)
        if fn.name == 'atomic_store':
            return self._builtin_atomic_store(binded, ret)
        if fn.name == 'atomic_rmw':
            return self._builtin_atomic_rmw(binded, ret)
        if fn.name == 'atomic_cmpxchg':
            return self._builtin_atomic_cmpxchg(binded, ret)
        if fn.name == 'atomic_fence':
            return self._builtin_atomic_fence(binded, ret)
        raise CompileError(f"cannot call the spy builtin {fn.name} inside a spy function")

    # -- byte-string builtins ------------------------------------------------

    def _bytes_builtin_arg(self, args: ArgList[ArgEntry[InterpVal]], what: str) -> bytes:
        """The compile-time ``bytes`` value the one argument of a byte-string
        builtin holds."""
        obj = _to_comptime(_shallow_normalize(self._arg_value(args.positional[0])))
        if not isinstance(obj, bytes):
            raise CompileError(f'{what} expects a compile-time byte string')
        return obj

    def _string_global(self, data: bytes) -> mir.GlobalStringValue:
        """The global static constant holding the bytes ``data`` plus a trailing
        NUL (the C-style terminator).  Identical byte strings share one global
        through the analyser's cache, so a body that names the same bytes twice
        emits one location."""
        cache = self._analyser._string_globals
        global_value = cache.get(data)
        if global_value is None:
            global_value = mir.GlobalStringValue(data + b'\0')
            cache[data] = global_value
        return global_value

    # -- ``std.core.const_array`` --------------------------------------------

    def _builtin_const_array(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.core.const_array(elem, length)``: build an array of ``length``
        (a compile-time integer) elements, every one equal to ``elem`` - the
        single element expression, evaluated once by the call and written into
        each element place.  ``elem`` may be a runtime value (one store per
        element) or a compile-time one (a compile-time array, no runtime code),
        exactly like the ``array(...)`` construction: the elements are the
        storage's places (``field_index_addr``), and ``finish_array`` closes the
        construction (resolving the array type and binding the pending places)."""
        elem = self._arg_value(args.positional[0])
        length = self._arg_value(args.positional[1])
        count = _comptime_int(length)
        if count is None:
            raise CompileError('the length of std.core.const_array must be a compile-time integer')
        if count < 0:
            raise CompileError('the length of std.core.const_array must not be negative')
        places: list[InterpVal] = []
        for index in range(count):
            place = self.field_index_addr(ret, _index_value(index), is_aggregate_init=True)
            self.store(place, elem)
            places.append(place)
        self.finish_array(ret, tuple(places))
        return PollResult.AGAIN

    # -- ``std.core.compile_error`` ------------------------------------------

    def _builtin_compile_error(self, args: ArgList[ArgEntry[InterpVal]]) -> Never:
        """``std.core.compile_error(msg)``: abort the compilation with ``msg`` -
        a compile-time byte string (a string literal is encoded to bytes at parse
        time, see ``astgen``) - as the error message."""
        data = self._bytes_builtin_arg(args, 'std.core.compile_error')
        raise CompileError(data.decode(errors='replace'))

    # -- ``std.core.bitcast`` ------------------------------------------------

    def _builtin_bitcast(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.core.bitcast(value, T)``: reinterpret the bits of the scalar
        ``value`` as the scalar type ``T``.  Only booleans, integers and floats
        are accepted, and the two types must have the same width (a boolean is a
        one-bit integer, so ``bool <-> i1/u1`` works).

        A compile-time scalar is reinterpreted in Python and stays
        compile-time; a runtime value is ``mir.BitCast`` - which lowers to a
        real ``llvm.bitcast`` between an integer and a float, and to a no-op
        when the two share the LLVM type (a signedness change, or ``bool`` and
        its ``i1``/``u1`` counterpart, whose MIR types differ but whose LLVM
        type is the same ``i1``)."""
        target = self._type_value_arg(
            args.positional[1], 'the type argument of std.core.bitcast'
        )
        ev = self._arg_value(args.positional[0])
        source = _type_of(ev)
        if source is None:
            raise CompileError(
                'cannot determine the type of the value passed to std.core.bitcast'
            )
        source_bits = _scalar_bit_width(source)
        target_bits = _scalar_bit_width(target)
        if source_bits is None or target_bits is None:
            raise CompileError(
                f'std.core.bitcast supports only integers, floats and booleans, '
                f'got {source} and {target}'
            )
        if source_bits != target_bits:
            raise CompileError(
                f'cannot bitcast a {source} ({source_bits} bits) to a {target} '
                f'({target_bits} bits)'
            )
        if source == target:
            self.store(ret, ev)
            return PollResult.AGAIN
        obj = ev.obj if isinstance(ev, ComptimeVal) else None
        if isinstance(obj, (sval.Int, sval.Float, bool)):
            # a compile-time scalar: reinterpret it here and keep it compile-time
            self.store(ret, ComptimeVal(_scalar_of_bits(_scalar_to_bits(obj, source), target)))
            return PollResult.AGAIN
        target_mir = target.to_mir_type(self._mir_cache)
        assert target_mir is not None
        self.store(ret, RuntimeVal(
            self._emit(mir.BitCast(self._to_runtime(ev), target_mir)), target
        ))
        return PollResult.AGAIN

    # -- ``std.int.truncate`` ------------------------------------------------

    def _builtin_truncate(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.int.truncate(value, T)``: narrow the integer ``value`` to the
        integer type ``T``, which must have the same signedness and fewer bits.
        A compile-time value is truncated in Python and stays compile-time; a
        runtime value is ``mir.Convert('trunc')``."""
        target = self._type_value_arg(
            args.positional[1], 'the type argument of std.int.truncate'
        )
        ev = self._arg_value(args.positional[0])
        source = _type_of(ev)
        if not isinstance(source, sval.IntType) or not isinstance(target, sval.IntType):
            raise CompileError(
                f'std.int.truncate supports only integers, got {source} and {target}'
            )
        if source.signed != target.signed:
            raise CompileError(
                f'cannot truncate a {source} to a {target}: the signedness must match'
            )
        if target.bits < 1 or target.bits >= source.bits:
            raise CompileError(
                f'cannot truncate a {source} to {target}: the target must have '
                f'fewer bits'
            )
        obj = ev.obj if isinstance(ev, ComptimeVal) else None
        if isinstance(obj, sval.Int):
            bits = obj.value & ((1 << target.bits) - 1)
            self.store(ret, ComptimeVal(_scalar_of_bits(bits, target)))
            return PollResult.AGAIN
        target_mir = target.to_mir_type(self._mir_cache)
        assert target_mir is not None
        self.store(ret, RuntimeVal(
            self._emit(mir.Convert('trunc', self._to_runtime(ev), target_mir)), target
        ))
        return PollResult.AGAIN

    # -- ``std.core.coerce`` -------------------------------------------------

    def _builtin_coerce(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.core.coerce(value, T)``: materialize ``value`` as the spy type
        ``T`` - the same conversion a store into a location of ``T`` performs
        (see ``_coerce``).  A compile-time value is converted in Python and
        stays compile-time; a runtime value gets whatever numeric conversion
        ``T`` needs (widening or narrowing).  The result is written into the
        call's result location, which the store commits to the coerced type."""
        ev = self._arg_value(args.positional[0])
        target = self._type_value_arg(
            args.positional[1], 'the type argument of std.core.coerce'
        )
        self.store(ret, self._coerce(ev, target))
        return PollResult.AGAIN

    # -- ``std.atomic`` ------------------------------------------------------

    def _atomic_pointer(
        self, arg: ArgEntry[InterpVal], what: str
    ) -> tuple[mir.Value, sval.Type]:
        """The ``(address, pointee spy type)`` of the pointer argument ``arg``
        of one atomic operation (see ``_atomic_pointer_value``)."""
        return self._atomic_pointer_value(self._arg_value(arg), what)

    def _atomic_pointer_value(
        self, ev: InterpVal, what: str
    ) -> tuple[mir.Value, sval.Type]:
        """The ``(address, pointee spy type)`` of the pointer value ``ev`` of an
        atomic operation.  The pointee has to be an integer, a float or a pointer
        - the scalar types LLVM can access atomically, byte-sized (a ``bool`` is
        an ``i1``, which LLVM will not access atomically, and an integer's width
        must be a byte-sized power of two).  A complex pointee is not a scalar:
        only ``atomic_rmw`` takes one, and handles it part by part (see
        ``_atomic_rmw_complex``)."""
        ptr_type = _type_of(ev)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'{what} expects a pointer, got {ptr_type}')
        elem = ptr_type.elem
        if not isinstance(elem, (sval.IntType, sval.FloatType, sval.PointerType)):
            raise CompileError(
                f'{what} expects a pointer to an integer, a float or a pointer, '
                f'got {ptr_type}'
            )
        if isinstance(elem, sval.IntType) and not (
            elem.bits > 0 and elem.bits & (elem.bits - 1) == 0 and elem.bits % 8 == 0
        ):
            raise CompileError(
                f'{what} expects a pointer to an integer whose width is a '
                f'byte-sized power of two, got {ptr_type}'
            )
        return self._to_runtime(ev), elem

    def _atomic_ordering(self, arg: ArgEntry[InterpVal]) -> mir.AtomicOrdering:
        """The :class:`mir.AtomicOrdering` the compile-time ``std.atomic``
        ``MemoryOrder`` member ``arg`` denotes (a member passed explicitly, or
        one that filled a parameter's default)."""
        from ..std.atomic import MemoryOrder
        enum_type = self._analyser._resolver.resolve_global(MemoryOrder)
        assert isinstance(enum_type, sval.IntEnumType)
        obj = _to_comptime(_shallow_normalize(self._arg_value(arg)))
        assert obj is not None, 'the ordering argument has no compile-time value'
        index = _enum_member_index(enum_type, obj)
        name = enum_type.members.by_id[index][0]
        return mir.AtomicOrdering[name]

    def _atomic_rmw_op(self, arg: ArgEntry[InterpVal]) -> mir.AtomicRmwOp:
        """The operation the compile-time ``std.atomic`` ``RmwOp`` member
        ``arg`` denotes."""
        from ..std.atomic import RmwOp
        enum_type = self._analyser._resolver.resolve_global(RmwOp)
        assert isinstance(enum_type, sval.IntEnumType)
        obj = _to_comptime(_shallow_normalize(self._arg_value(arg)))
        assert obj is not None, 'the operation argument has no compile-time value'
        index = _enum_member_index(enum_type, obj)
        name = enum_type.members.by_id[index][0]
        return _ATOMIC_RMW_OPS[name]

    def _atomic_volatile(self, arg: ArgEntry[InterpVal]) -> bool:
        """The compile-time ``volatile`` flag of one atomic operation."""
        obj = _to_comptime(_shallow_normalize(self._arg_value(arg)))
        if not isinstance(obj, bool):
            raise CompileError(
                'the volatile argument of an atomic operation must be a '
                'compile-time bool'
            )
        return obj

    def _builtin_atomic_load(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.atomic.atomic_load(ptr, ordering, volatile)``: the value the
        pointer ``ptr`` names, read atomically (see ``mir.AtomicLoad``)."""
        ptr, elem = self._atomic_pointer(args.positional[0], 'std.atomic.atomic_load')
        ordering = _check_atomic_ordering(
            self._atomic_ordering(args.positional[1]),
            (mir.AtomicOrdering.RELAXED, mir.AtomicOrdering.ACQUIRE, mir.AtomicOrdering.SEQ_CST),
            'std.atomic.atomic_load',
        )
        volatile = self._atomic_volatile(args.positional[2])
        self.store(ret, RuntimeVal(
            self._emit(mir.AtomicLoad(ptr, ordering, volatile)), elem
        ))
        return PollResult.AGAIN

    def _builtin_atomic_store(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.atomic.atomic_store(ptr, value, ordering, volatile)``: write
        ``value`` through the pointer ``ptr`` atomically (see
        ``mir.AtomicStore``)."""
        ptr, elem = self._atomic_pointer(args.positional[0], 'std.atomic.atomic_store')
        value = self._coerce(self._arg_value(args.positional[1]), elem)
        ordering = _check_atomic_ordering(
            self._atomic_ordering(args.positional[2]),
            (mir.AtomicOrdering.RELAXED, mir.AtomicOrdering.RELEASE, mir.AtomicOrdering.SEQ_CST),
            'std.atomic.atomic_store',
        )
        volatile = self._atomic_volatile(args.positional[3])
        self._emit(mir.AtomicStore(ptr, self._to_runtime(value), ordering, volatile))
        self.store(ret, ComptimeVal(sval.Void()))
        return PollResult.AGAIN

    def _builtin_atomic_rmw(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.atomic.atomic_rmw(ptr, op, value, ordering, volatile)``: apply
        ``op`` to the pointee of ``ptr`` and ``value`` atomically and deliver
        the old value (see ``mir.AtomicRmw``).  A complex pointee is operated on
        part by part - one atomicrmw on the real part and one on the imaginary
        part (see ``_atomic_rmw_complex``)."""
        op = self._atomic_rmw_op(args.positional[1])
        ordering = self._atomic_ordering(args.positional[3])
        volatile = self._atomic_volatile(args.positional[4])
        ev = self._arg_value(args.positional[0])
        ptr_type = _type_of(ev)
        if isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.ComplexType):
            return self._atomic_rmw_complex(
                ptr_type.elem, ev, args.positional[2], op, ordering, volatile, ret
            )
        ptr, elem = self._atomic_pointer_value(ev, 'std.atomic.atomic_rmw')
        _check_atomic_rmw_op(elem, op)
        value = self._coerce(self._arg_value(args.positional[2]), elem)
        self.store(ret, RuntimeVal(
            self._emit(mir.AtomicRmw(op, ptr, self._to_runtime(value), ordering, volatile)),
            elem,
        ))
        return PollResult.AGAIN

    def _atomic_rmw_complex(
        self, complex_type: sval.ComplexType, ptr_ev: InterpVal, value_arg: ArgEntry[InterpVal],
        op: mir.AtomicRmwOp, ordering: mir.AtomicOrdering, volatile: bool, ret: InterpVal,
    ) -> PollResult:
        """One ``atomic_rmw`` on a complex pointee: the real and the imaginary
        part (each a float) are read-modified-written by their own
        ``mir.AtomicRmw``, and the old part values are delivered as the old
        complex number.  It is not one atomic operation - the two parts are not
        updated as a unit - so only a use that needs no such guarantee (a
        per-number counter, say) is served by it."""
        _check_atomic_rmw_op(complex_type, op)
        elem = complex_type.elem
        ptr = self._to_runtime(ptr_ev)
        parts = self._complex_parts(self._coerce(self._arg_value(value_arg), complex_type))
        old: list[InterpVal] = []
        for index in (0, 1):
            part_ptr = self._emit(mir.Gep(ptr, index))
            part_value = self._to_runtime(parts[index])
            old.append(RuntimeVal(
                self._emit(mir.AtomicRmw(op, part_ptr, part_value, ordering, volatile)),
                elem,
            ))
        self.store(ret, self._complex_value(complex_type, old[0], old[1]))
        return PollResult.AGAIN

    def _builtin_atomic_cmpxchg(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.atomic.atomic_cmpxchg(ptr, expected, desired, success, failure,
        volatile)``: compare the pointee of ``ptr`` against ``expected`` and, on
        a match, store ``desired``; deliver ``(old_value, swapped)``."""
        ptr, elem = self._atomic_pointer(args.positional[0], 'std.atomic.atomic_cmpxchg')
        if isinstance(elem, sval.FloatType):
            # LLVM's cmpxchg operates on integers and pointers only
            raise CompileError(
                'std.atomic.atomic_cmpxchg does not support a float pointee'
            )
        expected = self._coerce(self._arg_value(args.positional[1]), elem)
        desired = self._coerce(self._arg_value(args.positional[2]), elem)
        success = self._atomic_ordering(args.positional[3])
        failure = self._atomic_ordering(args.positional[4])
        volatile = self._atomic_volatile(args.positional[5])
        _check_atomic_failure_ordering(success, failure)
        exchange = self._emit(mir.AtomicCmpxchg(
            ptr, self._to_runtime(expected), self._to_runtime(desired),
            success, failure, volatile,
        ))
        old = self._emit(mir.ExtractValue(exchange, 0))
        ok = self._emit(mir.ExtractValue(exchange, 1))
        results: tuple[InterpVal, ...] = (
            RuntimeVal(old, elem),
            RuntimeVal(ok, sval.BoolType()),
        )
        if isinstance(ret, tuple):
            # a destructuring target (``old, ok = ...``): the two values are
            # delivered into the tree's leaves, like any other tuple result
            if len(ret) != len(results):
                raise CompileError(
                    f'cannot deliver {len(results)} value(s) into a result tree '
                    f'of {len(ret)} place(s)'
                )
            for place, result in zip(ret, results):
                self.store(place, result)
        else:
            self.store(ret, self._build_tuple_value(
                tuple(ArgEntry(result, False) for result in results),
            ))
        return PollResult.AGAIN

    def _builtin_atomic_fence(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.atomic.atomic_fence(ordering)``: a memory ordering fence (see
        ``mir.Fence``)."""
        ordering = _check_atomic_ordering(
            self._atomic_ordering(args.positional[0]),
            (mir.AtomicOrdering.ACQUIRE, mir.AtomicOrdering.RELEASE,
             mir.AtomicOrdering.ACQ_REL, mir.AtomicOrdering.SEQ_CST),
            'std.atomic.atomic_fence',
        )
        self._emit(mir.Fence(ordering))
        self.store(ret, ComptimeVal(sval.Void()))
        return PollResult.AGAIN

    # -- ``std.core.panic`` / ``std.core.catch_unwind`` ----------------------

    def _panic_data_type(self) -> sval.Type:
        """The ``std.core.PanicData`` tagged union the host resolves in its own
        context (an alias, whose underlying type is the union of the panic
        payloads)."""
        from ..std.core import PanicData
        ret = sval.as_value(PanicData, self._analyser._resolver)
        assert isinstance(ret, sval.Type)
        return ret

    def _unwind_exception_type(self) -> sval.StructType:
        """The ``std.core.UnwindException`` struct of the host's context (the
        panic a ``catch_unwind`` raises as a spy error)."""
        from ..std.core import UnwindException
        ret = sval.as_value(UnwindException, self._analyser._resolver)
        assert isinstance(ret, sval.StructType)
        return ret

    def _builtin_panic(self, args: ArgList[ArgEntry[InterpVal]]) -> PollResult:
        """``std.core.panic(data)``: throw a panic carrying ``data`` (see
        ``mir.Panic``).  It never returns, so it ends the current path."""
        panic_type = self._panic_data_type()
        value = self._arg_value(args.positional[0])
        slot = self.alloca(False)
        self._commit_pending_slot(slot, panic_type)
        self.store(slot, value)
        pointer = self._to_runtime(_shallow_normalize(slot))
        # a panic runs the enclosing regions' deferred bodies on its way out
        # (like a may-panic call: see ``mir.Panic``); when there are none the
        # throw needs no landing pad
        defers = self._unwind_defers()
        unwind_path = self._resume_block() if len(defers) > 0 else None
        self._emit(mir.Panic(pointer, unwind_path, defers))
        return self._cut()

    def _builtin_catch_unwind(self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """``std.core.catch_unwind(fn)``: call the closure ``fn`` (an anonymous
        struct with a ``__call__``) and catch a panic it raises (see
        ``mir.CatchUnwind``); the caught payload is delivered as an
        ``UnwindException`` through the ordinary error path."""
        callable_place = self._arg_value(args.positional[0])
        ptr = self._auto_deref(callable_place)
        ptr_type = _type_of(ptr)
        if not (isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.StructType)):
            raise CompileError('the argument of std.core.catch_unwind must be a closure')
        method = self._resolve_method(ptr_type.elem, '__call__')
        if method is None:
            raise CompileError('the argument of std.core.catch_unwind must be a closure')
        fn = method.fn if isinstance(method, sval.BoundMethod) else method
        if not isinstance(fn, FunctionValue):
            raise CompileError('the argument of std.core.catch_unwind must be a closure')
        call_args: RawArgList[ArgEntry[InterpVal]] = RawArgList((ArgEntry(ptr, False),), frozendict())
        return self._call_function_entry(fn, call_args, ret, catch_unwind=True)

    def _const_to_mir(self, ev: InterpVal, type: sval.Type) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        """The constant MIR value of the value ``ev`` of spy type ``type``: a
        plain ``mir.Value`` for a scalar or a pointer/global reference, or an
        ``AggregateConstant`` / ``UnionConstant`` for an aggregate (which cannot
        be an instruction operand, see ``mir``).  Every representation follows
        the MIR representation of ``type``; a value that refers to a register is
        rejected (see ``_const_runtime_to_mir``)."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, RuntimeVal):
            return self._const_runtime_to_mir(ev, type)
        if isinstance(type, sval.PointerType):
            # the value *is* the pointer (a pointer to compile-time storage): it
            # is not followed here - the value it points at becomes a global of
            # its own and the pointer points at it (see ``_const_pointer_to_mir``)
            return self._const_pointer_to_mir(ev, type)
        if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.ComptimePtr):
            ev = self.load(ev)
        match type:
            case sval.BoolType():
                return mir.BoolValue(_comptime_bool(ev, 'the value lowered to a constant'))
            case sval.IntType():
                obj = _to_comptime(ev)
                if not isinstance(obj, sval.Int):
                    raise CompileError(f'expected a compile-time integer, got {ev!r}')
                return mir.Int(obj.value, mir.IntType(type.bits, type.signed))
            case sval.FloatType():
                obj = _to_comptime(ev)
                if not isinstance(obj, sval.Float):
                    raise CompileError(f'expected a compile-time float, got {ev!r}')
                return mir.Float(obj.value, mir.FloatType(type.bits))
            case sval.OptionType():
                return self._const_option_to_mir(ev, type)
            case sval.TaggedUnionType():
                return self._const_tagged_union_to_mir(ev, type)
            case sval.StructType():
                return self._const_struct_to_mir(ev, type)
            case sval.TupleType():
                return self._const_tuple_to_mir(ev, type)
            case sval.StrDictType():
                return self._const_strdict_to_mir(ev, type)
            case sval.ArrayType():
                return self._const_array_to_mir(ev, type)
            case _:
                raise CompileError(f'cannot make a static pointer to a value of {type}')

    def _const_runtime_to_mir(self, ev: RuntimeVal, type: sval.Type) -> mir.Value:
        """The constant of the runtime value ``ev``: only a value that refers
        to no register may stand in a global initializer - a constant leaf, or a
        reference to a global (another static constant, a function, an external
        symbol).  Anything computed at runtime (a ``mir.Load``, a ``mir.Gep``,
        an ``InsertValue`` chain, ...) is rejected."""
        if _references_register(ev.value):
            raise CompileError(
                f'cannot make a static pointer to a {type} value that refers to a runtime register'
            )
        return ev.value

    def _const_pointer_to_mir(self, ev: InterpVal, type: sval.PointerType) -> mir.Value:
        """The constant of a pointer value: a null pointer, a function pointer,
        a function value (its symbol, see §4.3), an undefined pointer, or - for a
        pointer to compile-time storage - a reference to a fresh global holding
        the content it points at."""
        mir_type = type.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.PointerType)
        obj = _to_comptime(ev)
        if isinstance(obj, (sval.Null, sval.TypedNull)):
            return mir.NullValue(mir_type)
        if isinstance(obj, sval.DeclareFunction):
            return _sval_to_runtime(obj, self._mir_cache)
        if isinstance(obj, FunctionValue):
            # a spy function value lowers to the symbol of its compiled
            # specialization, which is a (constant) function pointer itself
            return self._function_value_to_mir(obj)
        if isinstance(obj, (sval.UndefinedScalar, sval.UndefinedAggregate)):
            return mir.UndefValue(mir_type)
        if isinstance(ev, ComptimeVal) and isinstance(ev.obj, sval.ComptimePtr):
            # the pointer names compile-time storage: the content becomes a global
            # of its own, and the pointer is a reference to it
            cptr = ev.obj
            elem = type.elem
            if isinstance(cptr.box, bytes):
                # a byte-string store: the pointer is a global string constant
                # (the content plus a trailing NUL, see ``_string_global``)
                return self._string_global(cptr.box)
            elem_mir = elem.to_mir_type(self._mir_cache)
            if elem_mir is None or elem.classify() == sval.SpecialTypeKind.DST:
                raise CompileError(f'cannot make a static pointer to a pointer to {elem}')
            content = self._const_to_mir(self.load(ev), elem)
            return mir.GlobalConstant(content, mir.PointerType(elem_mir, mir_type.is_const))
        raise CompileError(f'cannot make a static pointer to the pointer constant {ev!r}')

    def _const_struct_to_mir(self, ev: InterpVal, type: sval.StructType) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        aggregate = _as_aggregate(ev)
        if aggregate is None:
            raise CompileError(f'cannot make a static pointer to the struct value {ev!r}')
        indices = type.get_field_mir_indices(self._mir_cache)
        fields = list(type.fields().values())
        if type.mirror_is_a_field(self._mir_cache):
            for index, mir_index in enumerate(indices):
                if mir_index is not None:
                    return self._const_to_mir(ComptimeVal(aggregate.values[index]), fields[index].type)
            raise CompileError(f'{type} has no stored field')
        mir_type = type.to_mir_type(self._mir_cache)
        if not isinstance(mir_type, mir.StructType):
            raise _no_runtime_type(type)
        members: dict[int, mir.Value | mir.AggregateConstant | mir.UnionConstant] = {}
        for index, mir_index in enumerate(indices):
            if mir_index is None:
                continue
            members[mir_index] = self._const_to_mir(ComptimeVal(aggregate.values[index]), fields[index].type)
        return mir.AggregateConstant(mir_type, tuple(members[i] for i in range(len(mir_type.fields))))

    def _const_tuple_to_mir(self, ev: InterpVal, type: sval.TupleType) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        """A compile-time tuple lowers as its mirror struct (a tuple mirrors to a
        synthesized struct, see ``MirLowerCache.tuple_struct``), element by
        element, exactly like a struct."""
        if type.classify() == sval.SpecialTypeKind.COMPTIME:
            raise _no_runtime_type(type)
        return self._const_struct_to_mir(ev, self._mir_cache.tuple_struct(type))

    def _const_strdict_to_mir(self, ev: InterpVal, type: sval.StrDictType) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        """A compile-time ``dict[str, T]`` lowers as its mirror struct (see
        ``MirLowerCache.strdict_struct``), one member per entry."""
        if type.classify() == sval.SpecialTypeKind.COMPTIME:
            raise _no_runtime_type(type)
        return self._const_struct_to_mir(ev, self._mir_cache.strdict_struct(type))

    def _const_array_to_mir(self, ev: InterpVal, type: sval.ArrayType) -> mir.AggregateConstant:
        aggregate = _as_aggregate(ev)
        if aggregate is None:
            raise CompileError(f'cannot make a static pointer to the array value {ev!r}')
        mir_type = type.to_mir_type(self._mir_cache)
        if not isinstance(mir_type, mir.ArrayType):
            raise _no_runtime_type(type)
        length = type.length_int
        if length is None:
            raise CompileError(f'cannot tell how many elements {type} holds')
        members = tuple(self._const_to_mir(ComptimeVal(aggregate.values[index]), type.elem) for index in range(length))
        return mir.AggregateConstant(mir_type, members)

    def _const_option_to_mir(self, ev: InterpVal, type: sval.OptionType) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        obj = _to_comptime(ev)
        value: InterpVal | None = None
        if isinstance(obj, sval.OptionValue):
            absent = False
            value = ComptimeVal(obj.value)
        elif isinstance(obj, (sval.Null, sval.TypedNull)):
            absent = True
        else:
            raise CompileError(f'cannot make a static pointer to the option value {ev!r}')
        child = type.child
        if child.is_zst():
            return mir.BoolValue(not absent)
        tag_path = sval.find_first_pointer_type_pos(child)
        if tag_path is None:
            mir_type = type.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.StructType)
            payload: mir.Value | mir.AggregateConstant | mir.UnionConstant
            if absent:
                payload = mir.UndefValue(mir_type.fields[1].type)
            else:
                assert value is not None
                payload = self._const_to_mir(self._coerce(value, child), child)
            return mir.AggregateConstant(mir_type, (mir.BoolValue(not absent), payload))
        if absent:
            return self._absent_path_constant(child, tag_path)
        assert value is not None
        return self._const_to_mir(self._coerce(value, child), child)

    def _absent_path_constant(self, child: sval.Type, path: tuple[int, ...]) -> mir.Value | mir.AggregateConstant:
        """The constant an absent option with a pointer tag has: the child
        constant with the pointer that tags the option (its first pointer, see
        ``sval.find_first_pointer_type_pos``) left null.  The other fields are
        undefined - nothing may read the payload of an absent option."""
        node: sval.Type = child
        layers: list[tuple[mir.Type, int]] = []
        for index in path:
            if isinstance(node, sval.OptionType):
                node = node.child
                continue
            mir_index: int | None
            if isinstance(node, sval.StructType):
                if node.mirror_is_a_field(self._mir_cache):
                    mir_index = None
                else:
                    mir_index = node.get_field_mir_indices(self._mir_cache)[index]
            elif isinstance(node, sval.ArrayType):
                mir_index = index
            else:
                raise CompileError(f'cannot take the tag pointer of Option[{child}]')
            if mir_index is not None:
                mir_type = node.to_mir_type(self._mir_cache)
                assert mir_type is not None
                layers.append((mir_type, mir_index))
            node = node.get_type_children()[index]
        assert isinstance(node, sval.PointerType)
        mir_pointer = node.to_mir_type(self._mir_cache)
        assert isinstance(mir_pointer, mir.PointerType)
        result: mir.Value | mir.AggregateConstant = mir.NullValue(mir_pointer)
        for outer_type, mir_index in reversed(layers):
            if isinstance(outer_type, mir.StructType):
                members = tuple(
                    result if i == mir_index else mir.UndefValue(field.type)
                    for i, field in enumerate(outer_type.fields)
                )
            else:
                assert isinstance(outer_type, mir.ArrayType)
                members = tuple(
                    result if i == mir_index else mir.UndefValue(outer_type.elem)
                    for i in range(outer_type.length)
                )
            result = mir.AggregateConstant(outer_type, members)
        return result

    def _const_tagged_union_to_mir(self, ev: InterpVal, type: sval.TaggedUnionType) -> mir.Value | mir.AggregateConstant | mir.UnionConstant:
        obj = _to_comptime(ev)
        if not isinstance(obj, sval.TaggedUnionValue):
            raise CompileError(f'cannot make a static pointer to the union value {ev!r}')
        tag = obj.index
        variant_value = ComptimeVal(obj.value)
        shape = sval.tagged_union_shape(type, self._mir_cache)
        if shape == 'single':
            return self._const_to_mir(self._coerce(variant_value, type.types[0]), type.types[0])
        if shape == 'tag_only':
            mir_type = type.to_mir_type(self._mir_cache)
            assert isinstance(mir_type, mir.IntType)
            return mir.Int(tag, mir_type)
        mir_type = type.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.StructType)
        union = mir_type.fields[1].type
        assert isinstance(union, mir.UnionType)
        variant_type = type.types[tag]
        variant = variant_type.to_mir_type(self._mir_cache)
        if variant is None:
            payload: mir.Value | mir.AggregateConstant | mir.UnionConstant = mir.UndefValue(union)
        else:
            payload = mir.UnionConstant(union, variant, self._const_to_mir(self._coerce(variant_value, variant_type), variant_type))
        tag_type = mir_type.fields[0].type
        assert isinstance(tag_type, mir.IntType)
        return mir.AggregateConstant(mir_type, (mir.Int(tag, tag_type), payload))

    # -- ``std.mem.layout_of`` -----------------------------------------------

    def _layout_builtin(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.mem.layout_of``: evaluate the layout of the queried type and
        hand it to the call's result location (see :meth:`_layout_of`)."""
        obj = _to_comptime(_shallow_normalize(self._arg_value(args.positional[0])))
        if not isinstance(obj, sval.Type):
            raise CompileError(f'std.mem.layout_of takes a type, got {obj!r}')
        self.store(ret, self._layout_of(obj))
        return PollResult.AGAIN

    def _layout_of(self, type: sval.Type) -> InterpVal:
        """The ``std.mem.Layout`` value of the spy type ``type``: a struct of
        its ``size`` and its ``align``.  Both are measured by the lowerer (the
        ``mir.Sizeof``/``mir.Alignof`` instructions), since a spy type has no
        compile-time size; only a type with a trivially known layout - a
        zero-sized type and an unsized array - is folded to a constant here."""
        size = self._sizeof_value(type)
        align = self._alignof_value(type)
        layout = self._resolve_layout_type()
        parts = (size, align)
        objs = tuple(_to_comptime(part) for part in parts)
        if all(obj is not None for obj in objs):
            return ComptimeVal(sval.AggregateValue(objs, layout))
        mir_type = layout.to_mir_type(self._mir_cache)
        assert isinstance(mir_type, mir.StructType)
        indices = layout.get_field_mir_indices(self._mir_cache)
        result: mir.Value = mir.UndefValue(mir_type)
        for index, mir_index in enumerate(indices):
            if mir_index is None:
                continue
            result = self._emit(mir.InsertValue(result, self._to_runtime(parts[index]), mir_index))
        return RuntimeVal(result, layout)

    def _sizeof_value(self, type: sval.Type) -> InterpVal:
        """The ``size`` field of ``layout_of(type)``: 0 for a zero-sized type
        and for an unsized array (which has no size of its own), an emitted
        ``mir.Sizeof`` otherwise.  A type with no fixed size (an opaque type or
        a function type) and a compile-time-only type are rejected."""
        kind = type.classify()
        if kind == sval.SpecialTypeKind.COMPTIME:
            raise CompileError(
                f'cannot take the layout of {type}: it is a compile-time-only type'
            )
        if kind == sval.SpecialTypeKind.ZST:
            return self._usize_const(0)
        if isinstance(type, sval.ArrayType) and type.length is None:
            return self._usize_const(0)
        if kind == sval.SpecialTypeKind.DST and not isinstance(type, sval.StructType):
            raise CompileError(f'cannot take the layout of {type}: it has no fixed size')
        mir_type = type.to_mir_type(self._mir_cache)
        if mir_type is None:
            raise CompileError(f'cannot take the layout of {type}')
        return RuntimeVal(
            self._emit(mir.Sizeof(mir_type, self._usize_type().bits)), self._usize_type()
        )

    def _alignof_value(self, type: sval.Type) -> InterpVal:
        """The ``align`` field of ``layout_of(type)``.  An unsized array aligns
        to its element (``align_of([?]T) == align_of(T)``); a zero-sized type -
        which has no MIR mirror to measure - is folded to the alignment its
        structure gives it (see ``sval.alignment_of``), and every other type is
        measured by an emitted ``mir.Alignof``."""
        peeled = type
        while isinstance(peeled, sval.ArrayType) and peeled.length is None:
            peeled = peeled.elem
        if peeled.is_zst():
            align = sval.alignment_of(peeled, self._mir_cache)
            return self._usize_const(align)
        kind = peeled.classify()
        if kind == sval.SpecialTypeKind.COMPTIME:
            raise CompileError(
                f'cannot take the layout of {peeled}: it is a compile-time-only type'
            )
        if kind == sval.SpecialTypeKind.DST and not isinstance(peeled, sval.StructType):
            raise CompileError(f'cannot take the layout of {peeled}: it is dynamically sized')
        mir_type = peeled.to_mir_type(self._mir_cache)
        if mir_type is None:
            raise CompileError(f'cannot take the layout of {peeled}')
        return RuntimeVal(
            self._emit(mir.Alignof(mir_type, self._usize_type().bits)), self._usize_type()
        )

    def _usize_const(self, value: int) -> InterpVal:
        """A compile-time ``usize`` value (the folded layout part)."""
        return ComptimeVal(sval.Int(value, self._usize_type()))

    def _resolve_layout_type(self) -> sval.StructType:
        """The ``std.mem.Layout`` struct type, resolved in the host context this
        body is compiled for (so that every context reflects into its own copy
        of the struct, like any other ``std`` type)."""
        from ..std import mem
        resolved = self._analyser._resolver.resolve_global(mem.Layout)
        assert isinstance(resolved, sval.StructType), 'std.mem.Layout is not a struct'
        return resolved

    # -- compile-time reflection (``std.reflect``) ---------------------------

    def _reflect_struct(self, name: str) -> sval.StructType:
        """The ``std.reflect`` struct type ``name`` declares, resolved in the
        host context this body is compiled for (so that every context reflects
        into its own copy of the struct, like any other ``std`` type)."""
        from ..std import reflect
        resolved = self._analyser._resolver.resolve_global(getattr(reflect, name))
        assert isinstance(resolved, sval.StructType), f'reflect.{name} is not a struct'
        return resolved

    def _reflect_type_info(self) -> sval.TaggedUnionType:
        """The ``std.reflect.TypeInfo`` tagged union, from the ``type`` alias
        itself: evaluating the alias yields its ``IntType | ...`` expression,
        which ``as_value`` resolves in the host context (so the variants are
        this context's own copies, in the order the alias declares them - the
        order is the tag)."""
        from ..std import reflect
        union = sval.as_value(reflect.TypeInfo.evaluate_value(Format.VALUE), ctx=self._analyser._resolver)
        assert isinstance(union, sval.TaggedUnionType), 'TypeInfo is not a tagged union'
        return union

    def _build_type_info(self, ty: sval.Type) -> InterpVal:
        """The ``TypeInfo`` value that describes the compile-time type ``ty``:
        the variant struct that matches it, tagged in the union."""
        fields: tuple[sval.AnyValue, ...]
        if isinstance(ty, sval.IntType):
            variant = 'IntType'
            fields = (ty.bits, ty.signed)
        elif isinstance(ty, sval.PointerType):
            variant = 'PointerType'
            fields = (ty.elem, ty.is_const)
        elif isinstance(ty, sval.ArrayType):
            variant = 'ArrayType'
            fields = (ty.elem, self._size_option(ty.length_int))
        elif isinstance(ty, sval.OptionType):
            variant = 'OptionType'
            fields = (ty.child,)
        elif isinstance(ty, sval.ComplexType):
            variant = 'ComplexType'
            fields = (ty.elem,)
        elif isinstance(ty, sval.StructType):
            variant = 'StructType'
            fields = (
                self._const_slice(
                    self._reflect_struct('StructField'),
                    tuple(self._build_struct_field(field) for field in ty.fields().values()),
                ),
                self._head_option(ty),
            )
        elif isinstance(ty, sval.TaggedUnionType):
            variant = 'TaggedUnionType'
            fields = (self._type_slice(tuple(ty.types)),)
        elif isinstance(ty, sval.UnionType):
            variant = 'UnionType'
            # the variants of a union are a set: order them by their rendered
            # form, so that the reflected order is stable
            fields = (self._type_slice(tuple(sorted(ty.types, key=str))),)
        elif isinstance(ty, sval.StrDictType):
            variant = 'StrDictType'
            fields = (
                self._const_slice(
                    self._reflect_struct('DictEntry'),
                    tuple(self._build_dict_entry(name, child) for name, child in ty.values.items()),
                ),
            )
        elif isinstance(ty, sval.TupleType):
            variant = 'TupleType'
            fields = (self._type_slice(ty.types), ty.has_ellipsis)
        else:
            raise CompileError(f'cannot reflect the type {ty}')
        union = self._reflect_type_info()
        struct_type = self._reflect_struct(variant)
        index = union.variant_index(struct_type)
        assert index is not None, f'{struct_type} is not a TypeInfo variant'
        aggregate = sval.AggregateValue(fields, struct_type)
        return ComptimeVal(sval.TaggedUnionValue(union, index, aggregate))

    def _build_struct_field(self, field: sval.StructField) -> sval.AggregateValue:
        """The ``std.reflect.StructField`` value describing one field of a
        struct: its name as a compile-time ``bytes`` value, its type, its default
        value (absent when the field declares none) and its ``is_comptime``/
        ``is_ref`` markers."""
        default: sval.AnyValue
        if field.default is None:
            default = sval.TypedNull(sval.AnyType())
        else:
            coerced = _to_comptime(self._coerce(ComptimeVal(field.default), sval.AnyType()))
            assert coerced is not None
            default = sval.OptionValue(coerced)
        return sval.AggregateValue((
            field.name.encode(),
            field.type,
            default,
            field.is_comptime,
            field.is_ref,
        ), self._reflect_struct('StructField'))

    def _build_dict_entry(self, name: str, child: sval.Type) -> sval.AggregateValue:
        """The ``std.reflect.DictEntry`` value describing one entry of a
        ``dict[str, T]`` shape: its name as a compile-time ``bytes`` value and
        its type."""
        return sval.AggregateValue((
            name.encode(),
            child,
        ), self._reflect_struct('DictEntry'))

    def _head_option(self, ty: sval.StructType) -> sval.AnyValue:
        """The ``Option[Any]`` a reflected struct's ``head`` is: the template
        head (:class:`sval.StructTypeHead`) of a struct that declares generic
        parameters, the absent option of a non-generic one."""
        if len(ty.head.generic_args) == 0:
            return sval.TypedNull(sval.AnyType())
        coerced = _to_comptime(self._coerce(ComptimeVal(ty.head), sval.AnyType()))
        assert coerced is not None
        return sval.OptionValue(coerced)

    def _size_option(self, length: int | None) -> sval.AnyValue:
        """The ``Option[int]`` a reflected array size is: an unsized array
        (``None``) as the absent option, a length as the present one."""
        if length is None:
            return sval.TypedNull(sval.AnyIntType())
        return sval.OptionValue(length)

    def _type_slice(self, types: tuple[sval.Type, ...]) -> sval.AggregateValue:
        """A ``ConstSlicePtr[type]`` of the compile-time types ``types``."""
        return self._const_slice(sval.TYPE_TYPE, types)

    def _const_slice(self, elem: sval.Type, values: tuple[sval.AnyValue, ...]) -> sval.AggregateValue:
        """A ``ConstSlicePtr[elem]`` compile-time value over ``values``: the
        values in a fresh (read-only) ``Box`` of their own and the (const)
        pointer to it, so that a reflected slice is a plain compile-time value
        (see §4.2)."""
        array_type = sval.ArrayType(elem, len(values))
        box = sval.Box('slice', array_type, is_const=True)
        box.cells = _flatten_value(sval.AggregateValue(tuple(values), array_type), array_type)
        ptr = sval.ComptimePtr(
            box, 0,
            sval.PointerType(array_type, is_const=True, variant=sval.PointerVariant.MULTI),
        )
        slice_type = self._special_type.slice_ptr_of(elem, True)
        return sval.AggregateValue((ptr, sval.Int(len(values), self._usize_type())), slice_type)

    # -- ``std.reflect.reify`` -----------------------------------------------

    def _builtin_reify(
        self, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal
    ) -> PollResult:
        """``std.reflect.reify(info)``: the compile-time type a ``TypeInfo`` value
        describes - the inverse of ``type_info``.  It accepts a whole ``TypeInfo``
        (the tagged union ``type_info`` builds) or one of its variant values (a
        variant struct a body constructs, or the payload an ``isinstance`` unwrap
        bound).  A reified *struct* is built fresh every time, so it never has the
        identity of the struct it was reflected from (see ``_reify_struct``)."""
        info = self._arg_value(args.positional[0])
        self.store(ret, ComptimeVal(self._reify_type_info(info)))
        return PollResult.AGAIN

    def _reify_type_info(self, ev: InterpVal) -> sval.Type:
        """The compile-time type a reflected value describes: a whole ``TypeInfo``
        value, or a bare ``TypeInfo`` variant value."""
        ev = _shallow_normalize(ev)
        obj = _to_comptime(ev)
        if isinstance(obj, sval.ComptimePtr):
            obj = _to_comptime(self.load(ev))
        if isinstance(obj, sval.TaggedUnionValue):
            variant = obj.type.types[obj.index]
            payload = obj.value
            if isinstance(payload, sval.AggregateValue):
                return self._reify_variant(variant, payload.values)
            raise CompileError('std.reflect.reify expects a TypeInfo variant value')
        if isinstance(obj, sval.AggregateValue):
            return self._reify_variant(obj.type, obj.values)
        raise CompileError('std.reflect.reify expects a TypeInfo or one of its variants')

    def _reify_variant(self, variant: sval.Type, fields: tuple[sval.AnyValue, ...]) -> sval.Type:
        """The compile-time type the ``TypeInfo`` variant ``variant`` (whose
        field values are ``fields``, in declaration order) describes."""
        from ..std import reflect
        resolve = self._analyser._resolver.resolve_global
        if variant is resolve(reflect.IntType):
            return sval.IntType(self._reify_int(fields[0]), self._reify_bool(fields[1]))
        if variant is resolve(reflect.PointerType):
            # the pointer variant (single/multi) is not reflected, so the
            # reified pointer is always a single one
            return sval.PointerType(self._reify_type(fields[0]), self._reify_bool(fields[1]))
        if variant is resolve(reflect.ArrayType):
            return sval.ArrayType(self._reify_type(fields[0]), self._reify_size(fields[1]))
        if variant is resolve(reflect.OptionType):
            return sval.OptionType(self._reify_type(fields[0]))
        if variant is resolve(reflect.ComplexType):
            elem = self._reify_type(fields[0])
            if not isinstance(elem, sval.FloatType):
                raise CompileError(f'the element of a complex type is not a float, got {elem}')
            return sval.ComplexType(elem)
        if variant is resolve(reflect.StructType):
            return self._reify_struct(fields[0], fields[1])
        if variant is resolve(reflect.UnionType):
            return sval.UnionType(frozenset(self._reify_type_slice(fields[0])))
        if variant is resolve(reflect.TaggedUnionType):
            return sval.tagged_union_of(self._reify_type_slice(fields[0]))
        if variant is resolve(reflect.TupleType):
            return sval.TupleType(
                self._reify_type_slice(fields[0]), self._reify_bool(fields[1])
            )
        if variant is resolve(reflect.StrDictType):
            return self._reify_strdict(fields[0])
        raise CompileError(f'{variant} is not a TypeInfo variant')

    def _reify_type(self, obj: sval.AnyValue) -> sval.Type:
        """The spy type a reflected ``type`` field holds."""
        if not isinstance(obj, sval.Type):
            raise CompileError('a reflected field that names a type is not a type')
        return obj

    def _reify_int(self, obj: sval.AnyValue) -> int:
        """The Python integer a reflected integer field holds."""
        if isinstance(obj, bool):
            raise CompileError('a reflected integer is not known at compile time')
        if isinstance(obj, sval.Int):
            return obj.value
        if isinstance(obj, int):
            return obj
        raise CompileError('a reflected integer is not known at compile time')

    def _reify_bool(self, obj: sval.AnyValue) -> bool:
        """The Python bool a reflected bool field holds."""
        if isinstance(obj, bool):
            return obj
        raise CompileError('a reflected bool has to be known at compile time')

    def _reify_option(self, obj: sval.AnyValue) -> tuple[bool, sval.AnyValue | None]:
        """The tag and the payload of a reflected option field: whether it is
        absent and the value it wraps."""
        if isinstance(obj, sval.OptionValue):
            return False, obj.value
        if isinstance(obj, (sval.Null, sval.TypedNull)):
            return True, None
        raise CompileError('a reflected option is not a compile-time option')

    def _reify_size(self, obj: sval.AnyValue) -> int | None:
        """The element count a reflected ``ArrayType.size`` holds: the absent
        option of an unsized array."""
        is_absent, value = self._reify_option(obj)
        if is_absent:
            return None
        assert value is not None
        return self._reify_int(value)

    def _reify_slice(self, obj: sval.AnyValue) -> tuple[sval.AnyValue, ...]:
        """The element values a reflected ``ConstSlicePtr`` field holds."""
        if not isinstance(obj, sval.AggregateValue):
            raise CompileError('a reflected slice is not a compile-time slice')
        ptr = obj.values[0]
        if not (isinstance(ptr, sval.ComptimePtr) and isinstance(ptr.type.elem, sval.ArrayType)):
            raise CompileError('a reflected slice does not point at a compile-time array')
        array = _load_box_cells(ptr.box, ptr.offset, ptr.type.elem)
        if not isinstance(array, sval.AggregateValue):
            raise CompileError('a reflected slice is not a compile-time slice')
        return array.values

    def _reify_type_slice(self, obj: sval.AnyValue) -> tuple[sval.Type, ...]:
        """The types a reflected ``ConstSlicePtr[type]`` field holds."""
        return tuple(self._reify_type(item) for item in self._reify_slice(obj))

    def _reify_struct(self, field_slice: sval.AnyValue, head: sval.AnyValue) -> sval.StructType:
        """The struct type a reflected ``StructType`` describes: a *fresh* struct,
        built field by field, so that every reify of the same struct is a type of
        its own (see ``_builtin_reify``).  A reflected template head only lends its
        name; its methods and modifiers are not reflected and so are not restored."""
        name_base = 'struct'
        is_absent, head_value = self._reify_option(head)
        if not is_absent:
            assert head_value is not None
            if isinstance(head_value, sval.StructTypeHead):
                name_base = head_value.name_base
        new_head = sval.StructTypeHead(name_base)
        for item in self._reify_slice(field_slice):
            if not isinstance(item, sval.AggregateValue):
                raise CompileError('a reflected struct field is not a struct value')
            name = item.values[0]
            if not isinstance(name, bytes):
                raise CompileError('a reflected field name is not a byte string')
            is_default_absent, default_value = self._reify_option(item.values[2])
            default = None if is_default_absent else default_value
            new_head.add_field(
                name.decode(errors='replace'),
                self._reify_type(item.values[1]),
                default,
                self._reify_bool(item.values[3]),
                self._reify_bool(item.values[4]),
            )
        return new_head.specialize(())

    def _reify_strdict(self, entry_slice: sval.AnyValue) -> sval.StrDictType:
        """The ``**kwargs``/``dict[str, T]`` type a reflected ``StrDictType``
        describes."""
        values: dict[str, sval.Type] = {}
        for item in self._reify_slice(entry_slice):
            if not isinstance(item, sval.AggregateValue):
                raise CompileError('a reflected dict entry is not a struct value')
            name = item.values[0]
            if not isinstance(name, bytes):
                raise CompileError('a reflected entry name is not a byte string')
            values[name.decode(errors='replace')] = self._reify_type(item.values[1])
        return sval.StrDictType(frozendict(values))

    def _call_fn_ptr(
        self,
        callee: InterpVal,
        fn_type: sval.FunctionType,
        args: RawArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        on_return: Callable[[], None] | None,
    ) -> PollResult:
        """A call through a runtime function pointer: the callee value is an
        address of the function type's signature, so the call is emitted like a
        call of a compiled function, reusing ``_make_runtime_call``.  The call
        signature and the return convention are rebuilt from the function type
        (see ``signature_of_fn_type``)."""
        sig = signature_of_fn_type(fn_type)
        binded_args = sig.bind_arg_pos(args, lambda e: ArgEntry(ComptimeVal(e), False))
        _check_comptime_args(sig, binded_args)
        call_sig, partial_ret_sig = sig.specialize(self._provided_types(sig, binded_args), self._mir_cache)
        ret_sig = partial_ret_sig.complete()
        res = self._make_runtime_call(
            self._to_runtime(callee), binded_args, ret, call_sig, ret_sig,
        )
        if on_return is not None and not ret_sig.value_is_empty():
            on_return()
        return res

    def _runtime_position(self) -> mir.BasicBlock:
        """The *runtime* block the walk currently sits in: the MIR block being
        built.  A store into a slot may be kept inline only while this is the
        block the slot recorded when its ``Alloca`` ran (see
        ``_record_pending_store``): a runtime value is a MIR register that only
        dominates the paths *after* it inside the block it is defined in, so a
        store that crossed a runtime block boundary has to land in memory.  The
        block changes exactly when the walk emits runtime control flow (a
        runtime ``if`` branch or loop header, the taken edge of a runtime
        ``break_if``, the join an ended path continues in); a compile-time
        ``if``/``loop`` and a ``hir.Block`` entry emit none and keep it."""
        return self._cur_block

    def _record_pending_action(self, slot: PendingSlot, data: _PendingActionData) -> None:
        """Record one action on a still uncommitted slot, reserving the
        insertion block that will hold the instructions delivering it."""
        insertion = mir.Insertion([], None)
        self._emit(insertion)
        slot.stores.append(_PendingAction(insertion, data))

    def _record_pending_store(self, slot: PendingSlot, value_type: sval.Type, value: InterpVal) -> None:
        """Record a store into a still uncommitted slot.  A compile-time variable
        only takes compile-time values (a runtime value into one is the
        ``cannot store a runtime value into a compile-time variable`` error, see
        the behaviour doc § 13); the value may be kept inline only while the walk
        has not crossed a runtime block boundary since the slot was created."""
        is_comptime_value = _is_comptime_val(value)
        if slot.is_comptime is True and not is_comptime_value:
            raise CompileError(
                f'cannot store a runtime value into a compile-time variable '
                f'of type {value_type}'
            )
        self._record_pending_action(
            slot,
            _PendingStore(
                type=value_type,
                is_inline=self._runtime_position() == slot.runtime_pos and is_comptime_value,
                value=value,
            ),
        )

    def _commit_pending_slot(
        self, val: InterpVal, type: sval.Type | None = None, ptr: mir.Value | None = None
    ) -> None:
        """Materialize a pending slot.  It becomes a compile-time ``sval.Box``
        (held as a ``ComptimeVal(sval.ComptimePtr)``) when it holds its value in
        compile-time storage, the unit value when its type is zero-sized, or a
        ``RuntimeVal`` pointer to freshly allocated memory.  The type is the
        pairwise ``resolve_peer_type`` of the action types (or the given one).
        The recorded actions are delivered through ``_exec_pending_actions``,
        which fills in the instructions spliced at their original positions."""
        if not isinstance(val, PendingSlot):
            raise CompileError('can only commit a pending slot')
        if val.committed is not None:
            # an already materialized storage may still hold unmaterialized leaf
            # places
            self._commit_place_tree(val.committed)
            return
        if type is None:
            type = val.committed_type()
        if ptr is not None:
            self._bind_slot(val, ptr, type)
            return

        if val.use_comptime_storage() and (len(val.stores) > 0 or val.declared is not None):
            # a compile-time variable: a fresh ``Box`` the stored values are
            # flattened into
            self._commit_box_slot(val, type)
            self._commit_place_tree(val.committed)
            return
        unit = type.get_unit_value()
        if unit is not None:
            # a zero-sized value: its unit value is all it holds.  It is still a
            # *place* (a compile-time ``Box`` holding the flattened undefined),
            # so that a store into it (the no-error code of a value-less
            # function, the void result of a ``return``) and a load out of it
            # both find a pointer (see ``store``/``load``)
            self._commit_box_slot(val, type)
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

    def _commit_box_slot(self, val: PendingSlot, type: sval.Type) -> None:
        """Commit a slot to a compile-time ``Box``.  The box is *writable* while
        the recorded actions are delivered (the initializing store of a
        ``ConstComptime`` variable lands in it); only afterwards does it become
        read-only - and the committed place const - when ``is_const`` says so."""
        box = self._new_box(type, is_const=False)
        val.committed = ComptimeVal(
            sval.ComptimePtr(box, 0, sval.PointerType(type, is_const=False))
        )
        self._exec_pending_actions(val, type)
        if val.is_const:
            box.is_const = True
            val.committed = ComptimeVal(
                sval.ComptimePtr(box, 0, sval.PointerType(type, is_const=True))
            )

    def _aggregate_place(self, type: sval.Type, is_comptime: bool = True) -> InterpVal:
        """A fresh place of the aggregate ``type``: a compile-time ``Box``
        (``is_comptime``) or fresh memory.  Its fields/elements are addressed with
        ``field_index_addr`` at their flattened offsets (see §0.2)."""
        if is_comptime:
            box = self._new_box(type, is_const=False)
            return ComptimeVal(sval.ComptimePtr(box, 0, sval.PointerType(type, is_const=False)))
        slot = self.alloca(False)
        self._commit_pending_slot(slot, type)
        return _shallow_normalize(slot)

    def _result_place_tree(self, place: InterpVal) -> InterpVal | tuple:
        """The result tree a multi-value result location is made of (see
        ``_finish_function``): the element places an uncommitted slot recorded (a
        ``_PendingTuple``), or the place as it is (a single value) - the shape
        ``_commit_ret_places`` then pairs with the spec."""
        place = _shallow_normalize(place)
        if isinstance(place, PendingSlot) and place.committed is None:
            recorded = self._pending_tuple(place)
            if recorded is not None:
                return recorded.places
        return place

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
                # a tuple location: each element is written through the address of
                # its element in the tuple's own storage
                element_types = tuple(_place_type(place) for place in action.places)
                for index, place in enumerate(action.places):
                    element_type = element_types[index]
                    if element_type is None:
                        raise CompileError('a tuple element has no type yet')
                    self._store_into_element(ptr, index, place, element_type)
            case _PendingPtrConvertion():
                input_ptr = _shallow_normalize(action.input)
                if not (isinstance(input_ptr, RuntimeVal) and isinstance(input_ptr.type, sval.PointerType)):
                    raise CompileError('cannot convert a compile-time pointer')
                action.output.value = self._to_runtime(self._convert_result_ptr(input_ptr, action.type))
            case _PendingAggregate():
                # an aggregate construction: every field/element is written
                # through its address in the aggregate's own storage (a box cell
                # for a compile-time one, ``mir.Gep`` for memory)
                ptr_type = _type_of(ptr)
                if isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.TaggedUnionType):
                    # the storage is a tagged union: the struct is one of its
                    # variants - set the tag and write the fields into that
                    # variant's storage (the same layout a construction written
                    # straight into the union takes, see
                    # ``_finish_struct_into_tagged_union``)
                    union = ptr_type.elem
                    index0 = self._tagged_union_index(union, action.type)
                    dest = self._write_tagged_union_tag(ptr, union, index0)
                    ptr = self._tagged_union_payload_ptr(dest, action.type)
                place_types = _aggregate_place_types(action.type)
                for index, place in enumerate(action.places):
                    self._store_into_element(ptr, index, place, place_types[index])
            case _:
                raise CompileError(f'unsupported pending action {action}')

    def _store_into_element(self, ptr: InterpVal, index: int, place: InterpVal, type: sval.Type) -> None:
        """Write the field/element place ``place`` into the ``index``-th
        `field/element of the aggregate storage ``ptr``: an uncommitted place is
        bound to the element's address - writing through it in place - when the
        storage is memory, and materialized and stored when it is a box.  A
        construction that was written into an ``Option[T]``'s payload (see
        ``field_index_addr`` with ``is_aggregate_init``) writes its fields there
        again, so the option layer is stepped through first."""
        ptr = _shallow_normalize(ptr)
        ptr_type = _type_of(ptr)
        while (
            isinstance(ptr_type, sval.PointerType)
            and self._is_option_construction(ptr_type.elem)
        ):
            ptr = self._option_payload_ptr(ptr)
            ptr_type = _type_of(ptr)
        addr = self.field_index_addr(ptr, _index_value(index))
        if isinstance(addr, RuntimeVal) and isinstance(place, PendingSlot) and place.committed is None:
            self._bind_slot(place, addr.value, type)
            return
        if isinstance(place, PendingSlot) and place.committed is None:
            self._commit_pending_slot(place, type)
        self.store(addr, self.load(place))

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
            return ComptimeVal(sval.UndefinedScalar(sval.PointerType(type, is_const=False)))
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
        tagged and written into the payload (``_union_variant_ptr``)."""
        ptr = _shallow_normalize(ptr)
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
            inner = self._option_payload_ptr(ptr)
            return self._convert_result_ptr(inner, to_type)
        if isinstance(from_type, sval.TaggedUnionType) and from_type.variant_index(to_type) is not None:
            # a variant written through the address of a tagged union: tag it and
            # hand back the payload place, which a construction builds its fields
            # in (see ``_write_tagged_union_tag``)
            index = self._tagged_union_index(from_type, to_type)
            ptr = self._write_tagged_union_tag(ptr, from_type, index)
            return self._tagged_union_payload_ptr(ptr, to_type)
        if isinstance(from_type, sval.TaggedUnionType) and isinstance(to_type, sval.TaggedUnionType):
            raise CompileError(
                f'cannot deliver a {to_type} through a result pointer of {from_type}'
            )
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

    def _option_payload_ptr(self, ptr: InterpVal, write_tag: bool = True) -> InterpVal:
        """The place the value of a present ``Option[T]`` lives in - what a
        delivery of a ``T`` into the option writes through.  The delivery also
        marks the option present: for a child that still has a free pointer the
        option *is* the value (that pointer is the tag, and the value itself
        sets it), and otherwise the tag of the struct representation is set
        here.

        ``write_tag`` says whether taking the address also marks the option
        present: a construction delivering a ``T`` does, while the ``:=``
        unwrap does not - the option may still be absent when the address is
        taken, and its tag must not be overwritten before it is tested."""
        type = _type_of(ptr)
        assert isinstance(type, sval.PointerType)
        option = type.elem
        assert isinstance(option, sval.OptionType)
        child = option.child
        if child.is_zst():
            # a zero-sized child has no storage to address: its value is the
            # type's unit value, which the place of an undefined pointer names
            # (a store into it is a no-op, see ``store``)
            return ComptimeVal(sval.UndefinedScalar(sval.PointerType(child, type.is_const)))
        ptr = _shallow_normalize(ptr)
        if isinstance(ptr, ComptimeVal) and isinstance(ptr.obj, sval.ComptimePtr):
            # a compile-time option place: the tag is cell 0, the payload cells
            # from 1 (see §0.2); taking a present payload's address marks the
            # option present by writing the tag cell
            cptr = ptr.obj
            if isinstance(cptr.box, sval.Box) and write_tag:
                cptr.box.cells[cptr.offset] = True
            return ComptimeVal(sval.ComptimePtr(
                cptr.box, cptr.offset + 1, sval.PointerType(child, cptr.type.is_const),
            ))
        src = self._to_runtime(ptr)
        if sval.find_first_pointer_type_pos(child) is not None:
            return RuntimeVal(src, sval.PointerType(child, type.is_const))
        tag = self._emit(mir.Gep(src, 0))
        if write_tag:
            self._emit(mir.Store(tag, mir.BoolValue(True)))
        payload = self._emit(mir.Gep(src, 1))
        return RuntimeVal(payload, sval.PointerType(child, type.is_const))

    def _comptime_option_is_null(self, ev: ComptimeVal) -> bool:
        """Whether the compile-time value ``ev`` is an absent option: a
        compile-time option *place* (a ``ComptimePtr``) is loaded first, then the
        value form is read - an ``sval.OptionValue`` is present, ``Null``/
        ``TypedNull`` absent (see §5.3)."""
        obj = ev.obj
        if isinstance(obj, sval.ComptimePtr):
            obj = _to_comptime(self.load(ev))
        return isinstance(obj, (sval.Null, sval.TypedNull))

    def _is_null(self, ev: InterpVal) -> InterpVal:
        """Whether ``ev`` is the *absent* value of an option, as a bool value.

        The compile-time absent values - the untyped ``Null`` the Python literal
        ``None`` evaluates to, and the typed ``TypedNull`` - answer
        ``ComptimeVal(True)``, every other compile-time value
        ``ComptimeVal(False)``.  A *runtime* option value is only known at
        runtime: the answer is a comparison read off the representation the
        child chooses (see ``sval.OptionType.to_mir_type``) - a zero-sized child
        *is* the "is there a value" bool, a child that still has a free pointer
        *is* the tagging pointer (null when absent, see ``_option_tag_value``),
        and any other child tags a ``(bool, T)`` struct whose first field is the
        tag."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, ComptimeVal):
            return ComptimeVal(self._comptime_option_is_null(ev))
        type = _type_of(ev)
        if isinstance(type, sval.NullType):
            return ComptimeVal(True)
        if not (isinstance(ev, RuntimeVal) and isinstance(type, sval.OptionType)):
            # a present value of the child type (or anything that is not an
            # option at all) is not the absent value
            return ComptimeVal(False)
        child = type.child
        if child.is_zst():
            # the option *is* the "is there a value" bool: absent when it is false
            cmp = mir.Cmp('==', ev.value, mir.BoolValue(False))
        elif sval.find_first_pointer_type_pos(child) is None:
            # the (bool, T) representation: the tag is its first field
            tag = self._emit(mir.ExtractValue(ev.value, 0))
            cmp = mir.Cmp('==', tag, mir.BoolValue(False))
        else:
            # the option *is* the child's value (they share the representation):
            # the tag is the child's first pointer, which is read out field by
            # field and compared against the null pointer
            tag, tag_type = self._option_tag_value(RuntimeVal(ev.value, child), child)
            tag_mir_type = tag_type.to_mir_type(self._mir_cache)
            assert isinstance(tag_mir_type, mir.PointerType)
            cmp = mir.Cmp('==', self._to_runtime(tag), mir.NullValue(tag_mir_type))
        return RuntimeVal(self._emit(cmp), sval.BoolType())

    def _option_tag_value(self, value: InterpVal, child: sval.Type) -> tuple[InterpVal, sval.PointerType]:
        """The pointer *value* that tags the option whose child is ``child``,
        read out of the option value ``value`` (which is typed as ``child``,
        the two sharing their representation) - the value counterpart of taking
        the tag address: the first pointer of ``child`` (see
        ``find_first_pointer_type_pos``), read field by field with
        ``_extract_aggregate_value`` (an option layer costs no field, since it
        shares its child's representation)."""
        path = sval.find_first_pointer_type_pos(child)
        assert path is not None, 'the option has a pointer tag'
        node = child
        tag = value
        for index in path:
            if isinstance(node, sval.OptionType):
                # an option shares its child's representation: view the value as
                # the child
                assert isinstance(tag, RuntimeVal)
                tag = RuntimeVal(tag.value, node.child)
                node = node.child
                continue
            tag = self._extract_aggregate_value(tag, index)
            node = node.get_type_children()[index]
        assert isinstance(node, sval.PointerType)
        return tag, node

    def _option_payload(self, ev: InterpVal, option: sval.OptionType) -> InterpVal:
        """The payload ``T`` of the *present* option value ``ev``.

        A compile-time present option holds its payload as an
        ``sval.OptionValue``.  A runtime option value is read off its
        representation: a zero-sized child has no payload but its unit value, a
        child that still has a free pointer shares the option's representation
        (the value *is* the payload), and the ``(bool, T)`` representation holds
        the payload in its second field, which ``mir.ExtractValue`` reads out of
        the value.  An absent value has no payload and is rejected."""
        ev = _shallow_normalize(ev)
        child = option.child
        if isinstance(ev, ComptimeVal):
            # a compile-time option value (an ``OptionValue`` present, a
            # ``Null``/``TypedNull`` absent) or the place of one (a ``ComptimePtr``)
            obj = ev.obj
            if isinstance(obj, sval.ComptimePtr):
                obj = _to_comptime(self.load(ev))
            if isinstance(obj, sval.OptionValue):
                return ComptimeVal(obj.value)
            if isinstance(obj, (sval.Null, sval.TypedNull)):
                raise CompileError(f'a present value is required here: {option} is absent')
            return self._coerce(ev, child)
        if isinstance(ev, RuntimeVal) and _type_of(ev) == option:
            if child.is_zst():
                unit = child.get_unit_value()
                assert unit is not None
                return ComptimeVal(unit)
            if sval.find_first_pointer_type_pos(child) is None:
                # the (bool, T) representation: the payload is the second field
                return RuntimeVal(self._emit(mir.ExtractValue(ev.value, 1)), child)
            # the option *is* the value: the present child itself
            return RuntimeVal(ev.value, child)
        if _comptime_bool(self._is_null(ev), 'an option value'):
            raise CompileError(f'a present value is required here: {option} is absent')
        return self._coerce(ev, child)

    def as_bool(self, value: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """Use the value as the condition of an ``if`` - a statement's or an
        if-expression's: a ``spy.bool`` value passes through, as the boolean
        register the interpreter branches on, and a struct that declares
        ``__bool__`` answers through it.  Spy has no truthiness, so nothing
        else is a condition."""
        type = _arg_type_of(value)
        if isinstance(type, sval.BoolType):
            ev = self._arg_value(value)
            self._frames[-1].regs[ret] = ev
            return PollResult.AGAIN
        if isinstance(type, sval.StructType) and self._resolve_method(type, _BOOL_METHOD) is not None:
            if _is_undefined_val(self._arg_value(value)):
                raise CompileError('an if condition cannot be undefined')
            slot = self.alloca(None)
            regs = self._frames[-1].regs

            def on_return() -> None:
                self._commit_pending_slot(slot)
                regs[ret] = self.load(slot)

            return self.call_method(
                self._operand_place(value), _BOOL_METHOD, RawArgList((), frozendict()), slot, on_return,
            )

        raise CompileError(f'an if condition must be a bool value, got {type}')

    def exec_len(self, inst: hir.Len) -> PollResult:
        """``len(x)`` (see ``hir.Len``): the element count of a tuple, the length
        of an array, or the answer of a struct's own ``__len__``.  The operand is
        logically a *value*: a tuple yields its length directly (part of its
        type now, see §7.1), an array its own length (part of its type, a
        compile-time ``usize``), and a struct is handed to its method through its
        own place - no copy for a by-reference one, and a value is materialized
        into a fresh place first."""
        operand = self.operand_arg(inst.value)
        type = _arg_type_of(operand)
        if isinstance(type, sval.TupleType):
            # a tuple's length is part of its type
            self._frames[-1].regs[inst] = ComptimeVal(sval.Int(len(type.types), self._usize_type()))
            return PollResult.AGAIN
        if isinstance(type, sval.StructType) and type.head.name_base in ('$args', '$kwargs'):
            # a synthesized ``*args``/``**kwargs`` struct (see §8.1)
            self._frames[-1].regs[inst] = ComptimeVal(sval.Int(len(type.fields().by_id), self._usize_type()))
            return PollResult.AGAIN
        if isinstance(type, sval.ArrayType):
            # an array's length is part of its type: a compile-time ``usize``
            # (the value ``sstr`` gives), rejected when it is not known yet
            length = type.length_int
            if length is None:
                raise CompileError(
                    f'cannot take the length of {type}: an array of unknown '
                    f'length has no element count'
                )
            self._frames[-1].regs[inst] = ComptimeVal(sval.Int(length, self._usize_type()))
            return PollResult.AGAIN
        if isinstance(type, sval.BytesType):
            # a compile-time byte string: its length is the number of bytes (no
            # trailing NUL is counted, see ``sstr``)
            obj = _to_comptime(self._arg_value(operand))
            assert isinstance(obj, bytes)
            self._frames[-1].regs[inst] = ComptimeVal(sval.Int(len(obj), self._usize_type()))
            return PollResult.AGAIN
        if isinstance(type, sval.StructType) and self._resolve_method(type, _LEN_METHOD) is not None:
            slot = self.alloca(None)
            regs = self._frames[-1].regs

            def on_return() -> None:
                self._commit_pending_slot(slot)
                regs[inst] = self.load(slot)

            return self.call_method(
                self._operand_place(operand), _LEN_METHOD, RawArgList((), frozendict()), slot, on_return,
            )
        raise CompileError(f'cannot take the length of {type}')

    def subscript(self, base: InterpVal, index: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """``Foo[i32, f64]``: the specialization of the struct template
        ``base`` for the generic arguments ``index`` (one type value, or a
        tuple of them).  The result is the struct *type* itself, a
        compile-time value - the same one the annotation spelling evaluates
        to at the Python level (see ``dsl._RegisteredClass.__getitem__``).

        ``a[i]``: the *place* the i-th element of the array ``base`` points at
        is - a subscript of an array is read and written through like a field
        of a struct (see ``field_index_addr``).

        ``mptr[i]``: the *place* the i-th element of the multi pointer is, an
        address ``mptr + i`` away (see ``_eval_pointer_arith``); a single
        pointer has only one place and is dereferenced with ``p[...]`` instead
        (see ``astgen``).

        ``mptr[a:b]``: the ``SlicePtr`` of the elements the slice names, built
        as a compile-time aggregate pointer (see ``slice_ptr``) - a subscript
        always yields a pointer.

        ``s[i]`` where ``s`` is a struct value that defines
        ``__spy_getitemptr__``: the place that method returns, the struct's own
        definition of what an element of it is (e.g. ``std.SlicePtr``).  The
        method is called with the subscript's own ``ret`` register as a
        callback target, since a method call delivers into a result location
        rather than a register (see ``_call_function_entry``).

        ``b[i]`` / ``b[a:c]`` where ``b`` is a compile-time byte string: the
        substring the subscript names, held as a reference (see
        ``_subscript_bytes``).

        ``t[i]`` where ``t`` is a compile-time tuple place tree (a varargs
        parameter, or a ``Comptime`` variable holding a multi-value result):
        the place of its i-th element (see ``_subscript_comptime_tuple``).

        ``d[key]`` where ``d`` is a compile-time ``**kwargs`` dictionary place
        tree: the place of the argument the byte-string key names (see
        ``_subscript_comptime_dict``)."""
        base_ev = _shallow_normalize(base)

        array_type = _array_elem_type_of(base)
        if array_type is not None:
            if array_type.length is None:
                raise CompileError(
                    f'cannot subscript {array_type}: a dynamically-sized array has '
                    f'no elements to index (use a ``MultiPtr`` to its elements)'
                )
            index_value = self._coerce(self._arg_value(index), self._usize_type())
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

        base_type = _type_of(base)
        if isinstance(base_type, sval.PointerType) and isinstance(base_type.elem, sval.TupleType):
            # a tuple place: the i-th element (a compile-time integer, checked)
            element_index = _comptime_int(self._arg_value(index))
            if element_index is None:
                raise CompileError('a tuple element index has to be a compile-time integer')
            if not 0 <= element_index < len(base_type.elem.types):
                raise CompileError(
                    f'index {element_index} is out of bounds for {base_type.elem}'
                )
            self._frames[-1].regs[ret] = self.field_index_addr(base, _index_value(element_index))
            return PollResult.AGAIN
        if isinstance(base_type, sval.PointerType) and isinstance(base_type.elem, sval.StrDictType):
            # a ``dict[str, T]`` place: the entry a compile-time byte-string key names
            key_value = _to_comptime(self._arg_value(index))
            if not isinstance(key_value, bytes):
                raise CompileError('a dict element key has to be a compile-time byte string')
            name = key_value.decode()
            entry_index = base_type.elem.values.by_key.get(name) if hasattr(base_type.elem.values, 'by_key') else None
            if entry_index is None:
                candidates = list(base_type.elem.values.keys())
                if name not in candidates:
                    raise CompileError(f'the dict has no element named {name!r}')
                entry_index = candidates.index(name)
            self._frames[-1].regs[ret] = self.field_index_addr(base, _index_value(entry_index))
            return PollResult.AGAIN
        if (
            isinstance(base_type, sval.PointerType)
            and isinstance(base_type.elem, sval.StructType)
            and base_type.elem.head.name_base == '$args'
        ):
            # a synthesized ``*args`` struct (see §8.1): ``args[i]`` is its i-th
            # element
            element_index = _comptime_int(self._arg_value(index))
            if element_index is None:
                raise CompileError('a tuple element index has to be a compile-time integer')
            if not 0 <= element_index < len(base_type.elem.fields().by_id):
                raise CompileError(
                    f'index {element_index} is out of bounds for the argument tuple'
                )
            self._frames[-1].regs[ret] = self.field_index_addr(base, _index_value(element_index))
            return PollResult.AGAIN
        if (
            isinstance(base_type, sval.PointerType)
            and isinstance(base_type.elem, sval.StructType)
            and base_type.elem.head.name_base == '$kwargs'
        ):
            # a synthesized ``**kwargs`` struct (see §8.1): ``kwargs['k']`` is the
            # entry the compile-time byte-string key names
            key_value = _to_comptime(self._arg_value(index))
            if not isinstance(key_value, bytes):
                raise CompileError('a dict element key has to be a compile-time byte string')
            entry_index = base_type.elem.fields().by_key.get(key_value.decode())
            if entry_index is None:
                raise CompileError(f'the dict has no element named {key_value.decode()!r}')
            self._frames[-1].regs[ret] = self.field_index_addr(base, _index_value(entry_index))
            return PollResult.AGAIN
        if isinstance(base_type, sval.PointerType) and isinstance(base_type.elem, sval.BytesType):
            return self._subscript_bytes(base, index, ret)
        if isinstance(base_type, sval.PointerType) and isinstance(base_type.elem, sval.PointerType):
            return self._subscript_pointer(base, base_type.elem, index, ret)

        if (
            isinstance(base_type, sval.PointerType)
            and isinstance(base_type.elem, sval.StructType)
            and self._resolve_method(base_type.elem, '__spy_getitemptr__') is not None
        ):
            # the struct overloads the subscript: the place its own
            # ``__spy_getitemptr__`` returns is what an element of it is.  A
            # method call delivers into a result location, not a register, so
            # the result is written into a fresh slot and moved into the
            # subscript's register once the call returned - whether the method
            # is inlined or compiled into a specialization of its own
            slot = self.alloca()
            regs = self._frames[-1].regs
            struct = base_type.elem

            def on_return() -> None:
                self._commit_pending_slot(slot)
                place = self.load(slot)
                place_type = _type_of(place)
                if not isinstance(place_type, sval.PointerType):
                    raise CompileError(
                        f'__spy_getitemptr__ of {struct} must return a pointer '
                        f'(the place of the element), got {place_type}'
                    )
                regs[ret] = place

            return self.call_method(
                base, '__spy_getitemptr__', RawArgList((index,), frozendict()), slot, on_return,
            )

        if not (isinstance(base_ev, ComptimeVal) and isinstance(base_ev.obj, sval.ComptimePtr)):
            raise CompileError(
                f'cannot subscript {base!r}: a subscript is a place in an array, '
                f'an element of a pointer, or a specialization of a struct template'
            )
        loaded = _to_comptime(self.load(base_ev))
        if not isinstance(loaded, sval.StructTypeHead):
            raise CompileError(
                f'cannot subscript {base!r}: a subscript is a place in an array, '
                f'an element of a pointer, or a specialization of a struct template'
            )
        struct = loaded
        args: tuple[InterpVal, ...]
        obj = _to_comptime(index.value)
        if isinstance(obj, sval.AggregateValue):
            # a tuple of type arguments (``Foo[A, B]``)
            args = tuple(ComptimeVal(v) for v in obj.values)
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


    def _subscript_bytes(self, base: InterpVal, index: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """``b[i]`` / ``b[a:c]`` of the compile-time byte string the place
        ``base`` holds: the substring the subscript names, held as a reference
        (:class:`sval.ComptimePtr`) - a subscript yields a place.  An integer index
        picks one byte (checked against the length), a ``std.slice`` object a
        range (a missing lower bound is 0, a missing upper bound the length, and
        a step has to be 1); every bound has to be compile-time."""
        data = _to_comptime(_shallow_normalize(self.load(base)))
        if not isinstance(data, bytes):
            raise CompileError('cannot subscript a value that is not a byte string')
        index_ev = self._arg_value(index)
        index_type = _type_of(index_ev)
        if isinstance(index_type, sval.StructType) and index_type.head is self._special_type.slice_type:
            start, end = self._bytes_slice_bounds(self.as_comptime_aggregate(index_ev), len(data))
            piece = data[start:end]
        else:
            index_value = self._coerce(index_ev, self._usize_type())
            index_int = _comptime_int(index_value)
            if index_int is None:
                raise CompileError('a byte string needs a constant index')
            if not 0 <= index_int < len(data):
                raise CompileError(
                    f'the index {index_int} is out of bounds for a byte string of '
                    f'length {len(data)}'
                )
            piece = data[index_int:index_int + 1]
        self._frames[-1].regs[ret] = ComptimeVal(
            sval.ComptimePtr(piece, 0, sval.PointerType(sval.BytesType(), is_const=True))
        )
        return PollResult.AGAIN

    def _bytes_slice_bounds(self, slice_obj: sval.AggregateValue, length: int) -> tuple[int, int]:
        """The ``(start, end)`` a ``std.slice`` object names over a byte string
        of ``length`` bytes: a missing lower bound is 0, a missing upper bound
        the length, and a step (if written) has to be 1.  Every bound has to be
        a compile-time integer, and the range is bounds-checked."""
        start_raw, end_raw, step_raw = slice_obj.values
        start_ev = self._slice_bound_value(start_raw, 'the start of a slice')
        start = 0 if start_ev is None else _comptime_int(start_ev)
        end_ev = self._slice_bound_value(end_raw, 'the end of a slice')
        end = length if end_ev is None else _comptime_int(end_ev)
        step_ev = self._slice_bound_value(step_raw, 'the step of a slice')
        if step_ev is not None and _comptime_int(step_ev) != 1:
            raise CompileError('a slice of a byte string has no step')
        if start is None or end is None:
            raise CompileError('a slice of a byte string needs constant bounds')
        if end < start or end > length:
            raise CompileError(
                f'the slice {start}:{end} is out of bounds for a byte string of '
                f'length {length}'
            )
        return start, end

    def _subscript_pointer(self, base: InterpVal, ptr_type: sval.PointerType, index: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """The place a subscript of the pointer ``ptr_type`` - the value the
        place ``base`` holds - names: the ``SlicePtr`` of a slice, or the
        element of a multi pointer, an address ``p + i`` away (the element's own
        place when the pointer names compile-time storage).  A single pointer
        has one place only, its pointee, and is dereferenced with ``p[...]``."""
        index_ev = self._arg_value(index)
        index_type = _type_of(index_ev)
        if isinstance(index_type, sval.StructType) and index_type.head is self._special_type.slice_type:
            self._frames[-1].regs[ret] = self.slice_ptr(
                base, ptr_type, self.as_comptime_aggregate(index_ev)
            )
            return PollResult.AGAIN
        if ptr_type.variant != sval.PointerVariant.MULTI:
            raise CompileError(
                f'cannot index {ptr_type}: a single pointer has to be '
                f'dereferenced with ``p[...]`` (index a ``MultiPtr`` instead)'
            )
        pointer = self.load(base)
        if isinstance(pointer, ComptimeVal) and isinstance(pointer.obj, sval.ComptimePtr):
            # compile-time storage of the elements: the n-th element's own place,
            # which a compile-time index picks - there is no runtime address to
            # take.  The view of the pointer may name the array (``mptr(Array[T,
            # N])``, the old shape) or its element (``mptr(T)``, after a
            # ``ptr_cast``); both index the first element at offset 0
            cptr = pointer.obj
            elem = ptr_type.elem
            index_value = self._coerce(index_ev, self._usize_type())
            index_int = _comptime_int(index_value)
            if index_int is None:
                raise CompileError(
                    'cannot index compile-time storage with a runtime index: '
                    'it has to be a compile-time integer'
                )
            self._frames[-1].regs[ret] = ComptimeVal(sval.ComptimePtr(
                cptr.box, cptr.offset + index_int * sval.scalar_count(elem),
                sval.PointerType(elem, cptr.type.is_const, sval.PointerVariant.SINGLE),
            ))
            return PollResult.AGAIN
        if not isinstance(pointer, RuntimeVal):
            raise CompileError('cannot index a compile-time pointer')
        offset = self._to_runtime(self._coerce(index_ev, self._isize_type()))
        self._frames[-1].regs[ret] = RuntimeVal(
            self._emit(mir.Gep(pointer.value, offset)), ptr_type,
        )
        return PollResult.AGAIN

    def slice_object(self, inst: hir.Slice) -> InterpVal:
        """The ``std.slice`` object a slice subscript builds (see
        ``hir.Slice``): a compile-time aggregate of its bounds, of the type
        ``std.slice[usize]`` - a slice of a pointer counts its elements.  Every
        bound is an *option*: a bound the source left out (a null constant) is
        the *typed* absent value of ``Option[usize]`` (see ``sval.TypedNull``)
        and a present one an ``Option`` value (see ``sval.OptionValue``): an
        option has no runtime shape of its own, so this is how a field is held.
        Which meaning an absent bound has is decided where the slice is used
        (see ``slice_ptr``)."""
        usize = self._usize_type()
        start = self._slice_bound_field(self.operand(inst.lower), usize, 'the start of a slice')
        end = self._slice_bound_field(self.operand(inst.upper), usize, 'the end of a slice')
        step = self._slice_bound_field(self.operand(inst.step), usize, 'the step of a slice')
        slice_type = self._special_type.slice_type.specialize((usize,))
        return ComptimeVal(sval.AggregateValue((start, end, step), slice_type))

    def _slice_bound_field(self, ev: InterpVal, usize: sval.Type, what: str) -> sval.AnyValue:
        """One bound of the ``std.slice`` object a slice subscript builds, as an
        option value: the absent value when ``ev`` is a null constant, a present
        option holding it coerced to ``usize`` otherwise (see §5.3)."""
        if _comptime_bool(self._is_null(ev), what):
            return sval.TypedNull(usize)
        value = _to_comptime(self._coerce(ev, usize))
        assert value is not None
        return sval.OptionValue(value)

    def _slice_bound_value(self, obj: sval.AnyValue, what: str) -> InterpVal | None:
        """The value of a bound of the ``std.slice`` object a slice subscript
        built, or None when it is absent (see ``slice_object``)."""
        if isinstance(obj, sval.OptionValue):
            return ComptimeVal(obj.value)
        if isinstance(obj, (sval.Null, sval.TypedNull)):
            return None
        raise CompileError(f'{what} is not a slice bound')

    def slice_ptr(self, base: InterpVal, ptr_type: sval.PointerType, slice_obj: sval.AggregateValue) -> InterpVal:
        """The ``SlicePtr`` a slice subscript of the multi pointer ``ptr_type``
        builds: ``SlicePtr(mptr + a, b - a)``.  The result is the place of a
        compile-time aggregate - a subscript always yields a pointer - a const box
        when both its fields are compile-time and fresh memory otherwise: the
        pointer field holds the offset pointer (a runtime value when the base is a
        runtime one, the offset into the base's compile-time storage otherwise),
        the length field the number of elements.

        A missing lower bound is 0; a missing upper bound has no meaning for a
        bare pointer - it has no length to slice to the end - and a step has to
        be a compile-time 1: only the elements of a pointer, one after another,
        are a slice of it."""
        if ptr_type.variant != sval.PointerVariant.MULTI:
            raise CompileError(
                f'cannot slice {ptr_type}: only a multi pointer is sliceable'
            )
        usize = self._usize_type()
        start_raw, end_raw, step_raw = slice_obj.values
        start_ev = self._slice_bound_value(start_raw, 'the start of a slice')
        if start_ev is None:
            # a missing lower bound is 0
            start_ev = ComptimeVal(sval.Int(0, usize))
        end_ev = self._slice_bound_value(end_raw, 'the end of a slice')
        if end_ev is None:
            raise CompileError('a slice of a pointer needs an upper bound')
        # a slice of a pointer has no step: the elements follow one another, so
        # only a missing step (None) or a step of exactly 1 is allowed
        step_ev = self._slice_bound_value(step_raw, 'the step of a slice')
        if step_ev is not None and _comptime_int(step_ev) != 1:
            raise CompileError('a slice of a pointer has no step')
        elem = ptr_type.elem
        is_const = ptr_type.is_const
        base_ev = self.load(base)
        element_ptr: InterpVal
        if isinstance(base_ev, RuntimeVal):
            offset = self._to_runtime(self._coerce(start_ev, self._isize_type()))
            element_ptr = RuntimeVal(
                self._emit(mir.Gep(base_ev.value, offset)),
                sval.PointerType(elem, is_const, sval.PointerVariant.MULTI),
            )
        elif (
            isinstance(base_ev, ComptimeVal)
            and isinstance(base_ev.obj, sval.ComptimePtr)
            and isinstance(base_ev.obj.type.elem, sval.ArrayType)
        ):
            # compile-time storage of the elements: the slice names a range of the
            # cells it holds (a runtime bound has no place to name)
            start = _comptime_int(start_ev)
            end = _comptime_int(end_ev)
            if start is None or end is None:
                raise CompileError(
                    'a slice of compile-time storage needs constant bounds'
                )
            self._check_slice_range(start, end)
            cptr = base_ev.obj
            element_ptr = ComptimeVal(sval.ComptimePtr(
                cptr.box, cptr.offset + start * sval.scalar_count(elem),
                sval.PointerType(sval.ArrayType(elem, end - start), is_const, sval.PointerVariant.MULTI),
            ))
        else:
            raise CompileError('cannot slice a compile-time pointer')
        start_int = _comptime_int(start_ev)
        end_int = _comptime_int(end_ev)
        length: InterpVal
        if start_int is not None and end_int is not None:
            self._check_slice_range(start_int, end_int)
            length = ComptimeVal(sval.Int(end_int - start_int, usize))
        else:
            length = RuntimeVal(
                self._emit(mir.Arith(
                    '-',
                    self._to_runtime(self._coerce(end_ev, usize)),
                    self._to_runtime(self._coerce(start_ev, usize)),
                    mir.IntType(usize.bits, usize.signed),
                )),
                usize,
            )
        slice_type = self._special_type.slice_ptr_of(elem, is_const)
        element_obj = _to_comptime(element_ptr)
        length_obj = _to_comptime(length)
        if element_obj is not None and length_obj is not None:
            # a slice is a view, not a place to write the slice itself through: a
            # const box (see §4.2)
            box = self._new_box(slice_type, is_const=True)
            box.cells = _flatten_value(
                sval.AggregateValue((element_obj, length_obj), slice_type), slice_type,
            )
            return ComptimeVal(sval.ComptimePtr(box, 0, sval.PointerType(slice_type, is_const=True)))
        place = self._aggregate_place(slice_type, is_comptime=False)
        self.store(self.field_index_addr(place, _index_value(0)), element_ptr)
        self.store(self.field_index_addr(place, _index_value(1)), length)
        return place


    def _check_slice_range(self, start: int, end: int) -> None:
        if end < start:
            raise CompileError(
                f'a slice ends ({end}) before it starts ({start})'
            )

    def _pending_aggregate(self, slot: PendingSlot) -> _PendingAggregate | None:
        # the aggregate initialization a slot recorded, when one did: the places
        # an inline aggregate construction builds into (see
        # ``init_inline_aggregate``)
        for action in slot.stores:
            if isinstance(action.data, _PendingAggregate):
                return action.data
        return None

    def _pending_tuple(self, slot: PendingSlot) -> _PendingTuple | None:
        # the tuple initialization a slot recorded, when one did: the element
        # places an inline tuple construction builds into (see ``init_tuple``)
        for action in slot.stores:
            if isinstance(action.data, _PendingTuple):
                return action.data
        return None

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

        An aggregate construction is *deferred*: the fields are their own places,
        which the construction records on the slot for its commit to materialize
        into the aggregate - a ``Box`` (compile-time) or memory, decided then (see
        ``PendingSlot.use_comptime_storage``).  A storage that is already
        materialized wrote every field through its place before."""
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

        dest_ptr_type = _type_of(_shallow_normalize(dest))
        if (
            isinstance(dest_ptr_type, sval.PointerType)
            and isinstance(dest_ptr_type.elem, sval.TaggedUnionType)
            and not (isinstance(dest, PendingSlot) and dest.committed is None)
        ):
            # the construction builds the struct straight into a *materialized*
            # tagged union: the union takes the variant the struct is, and every
            # field goes into that variant's storage.  An uncommitted storage
            # (a declared ``Comptime`` union, or an inline field place) goes the
            # ordinary deferred way instead: the fields are recorded as a
            # ``_PendingAggregate``, which its commit delivers into the variant
            # (see ``_exec_pending_action``)
            self._finish_struct_into_tagged_union(
                _shallow_normalize(dest), dest_ptr_type.elem, struct_type, provided, fields,
            )
            return

        if isinstance(dest, PendingSlot) and dest.committed is None:
            # the fields are their own places (``field_index_addr`` with
            # ``is_aggregate_init`` gave each a fresh slot), which the construction
            # records on the slot for its commit to materialize into the aggregate
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
                # only the fields it leaves out are filled.  A second
                # construction of a *different* type has no peer type - the
                # storage cannot hold either (see §5.1, Q5)
                recorded = self._pending_aggregate(dest)
                assert recorded is not None
                if recorded.type != struct_type:
                    raise CompileError(
                        f'a slot is stored with incompatible types '
                        f'{recorded.type} and {struct_type}'
                    )
                for index in range(len(fields.by_id)):
                    if index not in provided:
                        self._default_field_place(dest, index, fields.get_by_id(index))
            return

        # an already materialized storage: its fields already are places, which
        # the provided arguments wrote through (see ``field_index_addr``)
        dest_ptr = _shallow_normalize(dest)
        for index, field0 in enumerate(fields.values()):
            if index not in provided:
                self._default_field_place(dest_ptr, index, field0)

    def _finish_struct_into_tagged_union(
        self,
        dest: InterpVal,
        union: sval.TaggedUnionType,
        struct_type: sval.Type,
        provided: dict[int, InterpVal],
        fields: IndexedMap[str, sval.StructField],
    ) -> None:
        """Close a struct construction whose storage is a tagged union: the union
        takes the variant the struct is (its tag is set, a compile-time storage
        gets the variant's payload place) and every field is written into that
        variant's storage."""
        index = self._tagged_union_index(union, struct_type)
        dest = self._write_tagged_union_tag(dest, union, index)
        variant_ptr = self._tagged_union_payload_ptr(dest, struct_type)
        for index0, place in provided.items():
            field_type = fields.get_by_id(index0).type
            if field_type.is_zst():
                if isinstance(place, PendingSlot) and place.committed is None:
                    self._commit_pending_slot(place, field_type)
                continue
            at = place.insertion if isinstance(place, PendingSlot) else None
            addr = self.field_index_addr(variant_ptr, _index_value(index0), at=at)
            if isinstance(addr, RuntimeVal):
                if isinstance(place, PendingSlot) and place.committed is None:
                    self._bind_slot(place, addr.value, field_type)
            else:
                # a compile-time variant place: the field's value is copied in
                if isinstance(place, PendingSlot) and place.committed is None:
                    self._commit_pending_slot(place, field_type)
                    self.store(addr, self.load(place))
                else:
                    self.store(addr, place)
        for index0, field0 in enumerate(fields.values()):
            if index0 not in provided:
                self._default_field_place(variant_ptr, index0, field0)
        # TODO: no more manual commits
        self._commit_place_tree(variant_ptr)

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

        An aggregate construction is deferred, exactly like a struct's (see
        ``finish_struct``): the elements are their own places, recorded on the
        slot for its commit to materialize into the array."""
        array_type = _array_construction_type(array, elements)
        if isinstance(array, PendingSlot) and array.committed is None:
            if self._pending_aggregate(array) is None:
                self._record_pending_action(array, _PendingAggregate(array_type, elements))
            return
        # an already materialized storage: the elements already wrote through
        # their places (see ``field_index_addr``)

    def _build_tuple_value(self, elems: tuple[ArgEntry[InterpVal], ...]) -> InterpVal:
        """The *value* of a ``tuple`` expression (see ``hir.Tuple``): a tuple is a
        real aggregate now (§7.1), so it is a compile-time ``AggregateValue`` when
        every element is compile-time and an ``InsertValue`` chain otherwise.  The
        elements share their peer type (an untyped literal takes the type of the
        runtime value beside it), like any other implicitly typed construction."""
        elem_types: list[sval.Type] = []
        for elem in elems:
            elem_type = _arg_type_of(elem)
            if elem_type is None:
                raise CompileError('a tuple element has no type')
            elem_types.append(elem_type)
        peer: sval.Type | None = elem_types[0] if len(elem_types) > 0 else None
        for elem_type in elem_types[1:]:
            if peer is None:
                break
            peer = peer.resolve_peer_type(elem_type)
        if peer is not None:
            elem_types = [peer] * len(elem_types)
        tuple_type = sval.TupleType(tuple(elem_types), False)
        values: list[InterpVal] = []
        all_comptime = True
        for elem, elem_type in zip(elems, elem_types):
            value = self._coerce(self._arg_value(elem), elem_type)
            values.append(value)
            if _to_comptime(value) is None:
                all_comptime = False
        if all_comptime:
            objs = tuple(_to_comptime(value) for value in values)
            assert all(obj is not None for obj in objs)
            return ComptimeVal(sval.AggregateValue(objs, tuple_type))
        mir_type = tuple_type.to_mir_type(self._mir_cache)
        if not isinstance(mir_type, mir.StructType):
            raise CompileError(f'a value of the tuple {tuple_type} has no runtime representation')
        result: mir.Value = mir.UndefValue(mir_type)
        for index, value in enumerate(values):
            result = self._emit(mir.InsertValue(result, self._to_runtime(value), index))
        return RuntimeVal(result, tuple_type)

    def init_tuple(self, location: InterpVal, length: int) -> None:
        # a tuple is initialized into a place of its own: a still uncommitted
        # location records a fresh tuple of element places as a pending action,
        # so that the slot's type - and the conflict another store into it would
        # be - is resolved at its commit (see ``_PendingTuple``); an already
        # materialized location needs nothing (its ``TuplePtrElement``s address
        # its elements directly, see ``tuple_ptr_element``).  A result location
        # that is already a *tree* of places (a declared tuple return type, see
        # ``_ret_spec_place``) needs nothing either - each element already has
        # its own place.
        if isinstance(location, tuple):
            if len(location) != length:
                raise CompileError(
                    f'cannot initialize a tuple of {length} element(s) in a '
                    f'location of {len(location)} place(s)'
                )
            return
        location = _shallow_normalize(location)
        if isinstance(location, PendingSlot) and location.committed is None:
            recorded = self._pending_tuple(location)
            if recorded is not None:
                # a second ``return a, b`` into the same result location (another
                # path): the element places are the same, so the values written on
                # this path go where the earlier path's did
                if len(recorded.places) != length:
                    raise CompileError(
                        f'cannot initialize a tuple of {length} element(s) in a '
                        f'location of {len(recorded.places)} place(s)'
                    )
                return
            self._record_pending_action(
                location,
                _PendingTuple(tuple(self.alloca() for _ in range(length))),
            )

    def tuple_ptr_element(self, location: InterpVal | tuple, index: int) -> InterpVal:
        # the place the index-th element of the tuple being initialized in
        # ``location`` is written through (see ``init_tuple``)
        if isinstance(location, tuple):
            if index < 0 or index >= len(location):
                raise CompileError(f'the tuple has no element at index {index}')
            return location[index]
        location = _shallow_normalize(location)
        if isinstance(location, PendingSlot) and location.committed is None:
            recorded = self._pending_tuple(location)
            if recorded is None:
                raise CompileError(f'cannot take an element of {location!r}')
            places = recorded.places
            if index < 0 or index >= len(places):
                raise CompileError(f'the tuple has no element at index {index}')
            return places[index]
        # an already materialized tuple place: the element's address in it
        return self.field_index_addr(location, _index_value(index))

    def _commit_place_tree(self, place: InterpVal) -> None:
        # a materialized place (a box, a runtime location) holds no unmaterialized
        # leaf places of its own any more: only a still uncommitted slot needs
        # committing
        place = _shallow_normalize(place)
        if isinstance(place, PendingSlot) and place.committed is None:
            self._commit_pending_slot(place)

    def _method_of(self, struct: sval.StructType, method_name: str) -> sval.AnyValue | None:
        """The value of the method ``method_name`` of the struct type
        ``struct``: its function value - bound with the struct's type-
        argument values when the struct is generic (see
        :class:`sval.BoundMethod`).  This is the ``typeof(a).m`` a method
        call ``a.m(...)`` resolves to; a class-name access (``Foo[i32].m(x)``)
        resolves the same way, through the specialization the class name
        denotes (see ``_call_class_method``)."""
        method = struct.get_method(method_name)
        if method is None:
            return None
        if isinstance(method, sval.Value):
            # the entry is already a spy value (a closure's ``__call__``, a method
            # attached by the compiler): use it as it is.  A method read off a
            # Python class body is a plain object and resolves through the host
            # instead.
            resolved = method
        else:
            resolved = self._analyser._resolver.resolve_global(method)
            if resolved is None:
                return None
        generic_var_values = frozendict(zip(struct.head.generic_args, struct.generic_args))
        if len(generic_var_values) == 0:
            return resolved
        return sval.BoundMethod(resolved, generic_var_values)

    def _enum_method_of(self, enum_type: sval.IntEnumType, method_name: str) -> sval.AnyValue | None:
        """The value of the method ``method_name`` of the enumeration
        ``enum_type``: its function value (resolved through the host context).
        An enumeration has no type arguments, so there is nothing to bind."""
        method = enum_type.get_method(method_name)
        if method is None or isinstance(method, sval.Value):
            return method
        return self._analyser._resolver.resolve_global(method)

    def _resolve_method(self, type: sval.Type, method_name: str) -> sval.AnyValue | None:
        match type:
            case sval.StructType():
                return self._method_of(type, method_name)
            case sval.IntEnumType():
                return self._enum_method_of(type, method_name)
            case _:
                return None

    def _class_name_base(self, ptr: InterpVal) -> Any | None:
        """The compile-time object a class-name method base denotes: a struct
        type (``Foo``, ``Foo[i32]``), a struct template head (``Foo`` without
        arguments, which a call has to specialize first), or a plain Python
        class used as a namespace of functions.  None when the base is an
        ordinary value, so the call is a method call on it (see
        ``call_method``)."""
        todo: list[InterpVal] = [_shallow_normalize(ptr)]
        while len(todo) > 0:
            ev = todo.pop()
            match ev:
                case PendingSlot() if ev.committed is not None:
                    todo.append(_shallow_normalize(ev.committed))
                case ComptimeVal(obj):
                    if isinstance(obj, sval.ComptimePtr):
                        # a class name boxed by ``_as_ref`` (a struct type kept in
                        # a compile-time box): read it back out of its box
                        todo.append(ComptimeVal(_load_box_cells(obj.box, obj.offset, obj.type.elem)))
                        continue
                    if isinstance(obj, (sval.Type, type)):
                        return obj
                    return None
                case _:
                    return None
        return None

    def _call_class_method(
        self,
        obj: Any,
        method_name: str,
        args: RawArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        on_return: Callable[[], None] | None,
    ) -> PollResult:
        """Call the function ``method_name`` of the compile-time object ``obj``
        - a struct type, a struct template head, or a plain Python class used
        as a namespace - with the given arguments and no implicit ``self``:
        every argument is passed through, ``self`` included."""
        if isinstance(obj, sval.StructType):
            method = self._method_of(obj, method_name)
            if method is None:
                raise CompileError(f'{obj} has no method named {method_name}')
            return self.call(ComptimeVal(method), args, ret, on_return)
        if isinstance(obj, sval.IntEnumType):
            method = self._enum_method_of(obj, method_name)
            if method is None:
                raise CompileError(f'{obj} has no method named {method_name}')
            return self.call(ComptimeVal(method), args, ret, on_return)
        if isinstance(obj, sval.StructTypeHead):
            raise CompileError(
                f'{obj} is a struct template: specialize it before calling its '
                f'methods (``{obj.name_base}[...].{method_name}(...)``)'
            )
        if isinstance(obj, type):
            # a plain Python class used as a namespace of functions: the
            # attribute names a function, called like any other
            fn = getattr(obj, method_name, None)
            if fn is None or not callable(fn):
                raise CompileError(f'{obj.__name__} has no function named {method_name}')
            target = self._analyser._resolver.resolve_global(fn)
            if target is None:
                raise CompileError(f'cannot call {obj.__name__}.{method_name}')
            return self.call(ComptimeVal(target), args, ret, on_return)
        raise CompileError(f'cannot call a method on the type {obj}')

    def call_method(self, ptr: InterpVal, method_name: str, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal, on_return: Callable[[], None] | None = None) -> PollResult:
        base = _shallow_normalize(ptr)
        # ``Foo.m(...)`` / ``Foo[i32].m(x)`` - a method accessed through the
        # class name (or a plain class used as a namespace) - resolves the
        # function from the compile-time type and calls it with the given
        # arguments and no implicit ``self`` (see ``_call_class_method``).
        class_base = self._class_name_base(base)
        if class_base is not None:
            return self._call_class_method(class_base, method_name, args, ret, on_return)
        ptr = self._auto_deref(base)
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f'cannot call a method on a {type} value')

        struct_type = type.elem
        method = self._resolve_method(struct_type, method_name)
        if method is None:
            raise CompileError(f'type {struct_type} has no method named {method_name}')

        # a ``@staticmethod`` takes no receiver: it is called with the given
        # arguments alone
        if isinstance(struct_type, (sval.StructType, sval.IntEnumType)) and struct_type.is_static_method(method_name):
            return self.call(ComptimeVal(method), args, ret, on_return)

        # a method's first parameter is the struct itself: it is passed by
        # reference (the base's address) unless the method declares
        # ``self`` as a pointer type, in which case the base's address is
        # already the pointer value the parameter expects
        fn = method.fn if isinstance(method, sval.BoundMethod) else method
        self_is_ref = True
        if isinstance(fn, FunctionValue):
            first = fn.hir.signature.positional.by_id[0]
            self_is_ref = (
                first.by_ref is TriState.TRUE
                or not isinstance(first.type, sval.PointerType)
            )

        return self.call(
            ComptimeVal(method),
            RawArgList((ArgEntry(ptr, self_is_ref),) + args.positional, args.kwargs),
            ret,
            on_return,
        )

    def eval_call_args(self, args: hir.CallArgs) -> RawArgList[ArgEntry[InterpVal]]:
        """Evaluate one call's HIR argument list (see ``hir.CallArgs``), flattening
        its ``*``/``**`` forwarding items into an already-evaluated ``RawArgList``
        (what ``call``/``call_method`` consume)."""
        positional: list[ArgEntry[InterpVal]] = []
        for item in args.positional:
            if isinstance(item, hir.Spread):
                positional.extend(self._spread_positional(self.operand(item.value)))
            else:
                positional.append(self.operand_arg(item))
        kwargs: dict[str, ArgEntry[InterpVal]] = {}
        for item in args.kwargs:
            if isinstance(item, hir.Spread):
                spread = self._spread_kwargs(self.operand(item.value))
                for key, value in spread.items():
                    if key in kwargs:
                        raise CompileError(f"got multiple values for argument '{key}'")
                    kwargs[key] = value
            else:
                key, entry = item
                if key in kwargs:
                    raise CompileError(f"got multiple values for argument '{key}'")
                kwargs[key] = self.operand_arg(entry)
        return RawArgList(tuple(positional), frozendict(kwargs))

    def _spread_positional(self, place: InterpVal) -> tuple[ArgEntry[InterpVal], ...]:
        """The positional arguments a ``*`` forwarding item contributes: the
        element places of the ``$args`` struct the item names (see §8.1), or of a
        tuple place."""
        place = _shallow_normalize(place)
        ptr_type = _type_of(place)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'cannot unpack {ptr_type} positionally')
        elem = ptr_type.elem
        if isinstance(elem, sval.TupleType):
            count = len(elem.types)
        elif isinstance(elem, sval.StructType) and elem.head.name_base == '$args':
            count = len(elem.fields().by_id)
        else:
            raise CompileError(f'cannot unpack {elem} positionally')
        return tuple(
            ArgEntry(self.field_index_addr(place, _index_value(index)), True)
            for index in range(count)
        )

    def _spread_kwargs(self, place: InterpVal) -> frozendict[str, ArgEntry[InterpVal]]:
        """The keyword arguments a ``**`` forwarding item contributes: the value
        places of the ``$kwargs`` struct the item names (see §8.1), or of a
        ``dict[str, T]`` place."""
        place = _shallow_normalize(place)
        ptr_type = _type_of(place)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f'cannot unpack {ptr_type} by keyword')
        elem = ptr_type.elem
        if isinstance(elem, sval.StrDictType):
            names = list(elem.values.keys())
        elif isinstance(elem, sval.StructType) and elem.head.name_base == '$kwargs':
            names = list(elem.fields().by_id.keys())
        else:
            raise CompileError(f'cannot unpack {elem} by keyword')
        return frozendict(
            (name, ArgEntry(self.field_index_addr(place, _index_value(index)), True))
            for index, name in enumerate(names)
        )

    def operand_arg(self, arg: ArgEntry[hir.Value]) -> ArgEntry[InterpVal]:
        return ArgEntry(self.operand(arg.value), arg.is_ref)

    def _type_value_arg(self, arg: ArgEntry[InterpVal], what: str) -> sval.Type:
        """The spy type a *type-valued* argument denotes - the argument of a
        ``type[X]`` parameter (see ``SignatureFormalArg.is_type_value``).  The
        argument is read as a compile-time value and has to be a spy type."""
        obj = _to_comptime(_shallow_normalize(self._arg_value(arg)))
        if not isinstance(obj, sval.Type):
            raise CompileError(f'{what} must be a spy type, got {obj!r}')
        return obj

    def _provided_types(
        self, sig: Signature, binded_args: ArgList[ArgEntry[InterpVal]]
    ) -> ArgList[ArgNode]:
        """The ``provided`` argument list of one call (see
        ``Signature.solve_param_types``): the :data:`ArgNode` of every argument -
        its value when it is compile-time (including the spy type a ``type[X]``
        argument denotes), its spy type when it is runtime, and the element tree
        when it is a tuple/``dict[str, T]``."""
        positional: list[ArgNode] = []
        for (name, param), arg in zip(sig.positional.items(), binded_args.positional):
            if param.is_type_value:
                positional.append(
                    self._type_value_arg(arg, f"the argument of parameter '{name}'")
                )
            else:
                positional.append(self._val_to_node(arg.value, arg.is_ref))
        varargs = tuple(self._val_to_node(arg.value, arg.is_ref) for arg in binded_args.varargs)
        kwargs = frozendict(
            (k, self._val_to_node(arg.value, arg.is_ref)) for k, arg in binded_args.kwargs.items()
        )
        return ArgList(tuple(positional), varargs, kwargs)

    def _val_to_node(self, val: InterpVal, is_ref: bool) -> ArgNode:
        """The :data:`ArgNode` an operand provides (see
        ``Signature.solve_param_types``), whether it is a value or - when
        ``is_ref`` - a *place*.

        A **value** provides a compile-time value as itself (a compile-time
        pointer by identity, see §4.2) or a runtime value as its spy type.  A
        **place** (``is_ref``) provides *what it holds*: a compile-time place the
        value it holds, a runtime place its element type (the caller emits the
        load, see ``_make_runtime_call``)."""
        ev = _shallow_normalize(val)
        # a tuple/``dict`` argument is provided element by element (its elements
        # are their own places/values, passed separately - see §7.4).  A *place*
        # of a tuple type (a tuple variable) is spread; a tuple *value* too.  A
        # pointer *value* to a tuple (``ref(t)``) is passed as the pointer itself
        # (a ``Ptr[tuple[...]]`` argument)
        val_type = _type_of(ev)
        if isinstance(val_type, sval.PointerType):
            container = val_type.elem
            if is_ref and isinstance(container, sval.TupleType):
                return self._tuple_val_to_nodes(ev, val_type, container)
            if is_ref and isinstance(container, sval.StrDictType):
                return self._dict_val_to_nodes(ev, val_type, container)
            if (
                is_ref and isinstance(container, sval.StructType)
                and container.head.name_base in ('$args', '$kwargs')
            ):
                # a ``tuple[...]``/``dict[str, T]`` parameter's own storage is a
                # synthesized struct (see §8.1): provide it element by element,
                # as the logical tuple/dict shape it stands for
                return self._vararg_struct_nodes(ev, container)
        else:
            if isinstance(val_type, sval.TupleType):
                return self._tuple_val_to_nodes(ev, val_type, val_type)
            if isinstance(val_type, sval.StrDictType):
                return self._dict_val_to_nodes(ev, val_type, val_type)
        if is_ref:
            # a place: what it provides is the content it holds
            if _is_comptime_val(ev):
                return self._val_to_node(self.load(ev), False)
            node_type = _arg_type_of(ArgEntry(ev, True))
            if node_type is None:
                raise CompileError(
                    'cannot determine the type of an argument passed to a spy function'
                )
            return RuntimeArgNode(node_type, False)
        match ev:
            case RuntimeVal(type=value_type):
                return RuntimeArgNode(value_type, False)
            case ComptimeVal(obj=obj):
                # a compile-time value, a compile-time pointer included (passed
                # by identity, see §4.2)
                return obj
            case _:
                node_type = _type_of(ev)
                if node_type is None:
                    raise CompileError(
                        'cannot determine the type of an argument passed to a spy function'
                    )
                return RuntimeArgNode(node_type, False)

    def _vararg_struct_nodes(self, place: InterpVal, struct: sval.StructType) -> ArgNode:
        """The provided node of a flattened ``*args``/``**kwargs`` parameter's
        synthesized struct (see §8.1): its fields are provided element by element
        - by position for ``$args``, by name for ``$kwargs`` - like the tuple/dict
        shape it stands for (a call forwarding such a parameter)."""
        fields = struct.fields()
        if struct.head.name_base == '$kwargs':
            return frozendict(
                (name, self._val_to_node(self.field_index_addr(place, _index_value(index)), True))
                for index, name in enumerate(fields.keys)
            )
        return tuple(
            self._val_to_node(self.field_index_addr(place, _index_value(index)), True)
            for index in range(len(fields.by_id))
        )

    def _tuple_val_to_nodes(self, ev: InterpVal, val_type: sval.Type | None, container: sval.TupleType) -> tuple[ArgNode, ...]:
        """One provided :data:`ArgNode` per element of a tuple argument (see
        ``_val_to_node``): the element places of a tuple place, the element values
        of a compile-time tuple, or - for a runtime tuple with no place of its own
        - its elements after materializing it."""
        if isinstance(val_type, sval.PointerType):
            return self._tuple_element_nodes(ev, container)
        aggregate = _as_aggregate(ev)
        if aggregate is not None:
            return tuple(self._val_to_node(ComptimeVal(value), False) for value in aggregate.values)
        slot = self.alloca(False)
        self._commit_pending_slot(slot, container)
        self.store(slot, ev)
        return self._tuple_element_nodes(slot, container)

    def _tuple_element_nodes(self, place: InterpVal, container: sval.TupleType) -> tuple[ArgNode, ...]:
        return tuple(
            self._val_to_node(self.field_index_addr(place, _index_value(index)), True)
            for index in range(len(container.types))
        )

    def _dict_val_to_nodes(self, ev: InterpVal, val_type: sval.Type | None, container: sval.StrDictType) -> frozendict[str, ArgNode]:
        """The named counterpart of ``_tuple_val_to_nodes`` (a ``dict[str, T]``)."""
        names = list(container.values.keys())
        if isinstance(val_type, sval.PointerType):
            return frozendict(
                (name, self._val_to_node(self.field_index_addr(ev, _index_value(index)), True))
                for index, name in enumerate(names)
            )
        aggregate = _as_aggregate(ev)
        if aggregate is not None:
            return frozendict(
                (name, self._val_to_node(ComptimeVal(value), False))
                for name, value in zip(names, aggregate.values)
            )
        slot = self.alloca(False)
        self._commit_pending_slot(slot, container)
        self.store(slot, ev)
        return frozendict(
            (name, self._val_to_node(self.field_index_addr(slot, _index_value(index)), True))
            for index, name in enumerate(names)
        )

    def _tuple_arg_entries(self, ev: ArgEntry[InterpVal]) -> tuple[ArgEntry[InterpVal], ...]:
        """One operand per element of a tuple argument, for a call that passes the
        elements separately (see ``ArgNode``): the element places of a tuple
        place, the element values of a compile-time tuple, or - for a runtime
        tuple value with no place of its own - its element places after
        materializing it."""
        value = _shallow_normalize(ev.value)
        ptr_type = _type_of(value)
        if isinstance(ptr_type, sval.PointerType):
            container = ptr_type.elem
            if isinstance(container, sval.TupleType):
                return tuple(
                    ArgEntry(self.field_index_addr(value, _index_value(index)), True)
                    for index in range(len(container.types))
                )
            if (
                isinstance(container, sval.StructType)
                and container.head.name_base == '$args'
            ):
                # a ``tuple[...]``/``*args`` parameter's own storage (see §8.1)
                return tuple(
                    ArgEntry(self.field_index_addr(value, _index_value(index)), True)
                    for index in range(len(container.fields().by_id))
                )
        aggregate = _as_aggregate(value)
        if aggregate is not None and isinstance(aggregate.type, sval.TupleType):
            return tuple(ArgEntry(ComptimeVal(element), False) for element in aggregate.values)
        if isinstance(ptr_type, sval.TupleType):
            slot = self.alloca(False)
            self._commit_pending_slot(slot, ptr_type)
            self.store(slot, value)
            return tuple(
                ArgEntry(self.field_index_addr(slot, _index_value(index)), True)
                for index in range(len(ptr_type.types))
            )
        raise CompileError(f'cannot take the elements of the argument value {ev!r}')

    def _dict_arg_entries(self, ev: ArgEntry[InterpVal]) -> dict[str, ArgEntry[InterpVal]]:
        """One operand per name of a ``dict[str, T]`` argument (see
        ``_tuple_arg_entries``)."""
        value = _shallow_normalize(ev.value)
        ptr_type = _type_of(value)
        if isinstance(ptr_type, sval.PointerType):
            container = ptr_type.elem
            if isinstance(container, sval.StrDictType):
                names = list(container.values.keys())
                return {
                    name: ArgEntry(self.field_index_addr(value, _index_value(index)), True)
                    for index, name in enumerate(names)
                }
            if (
                isinstance(container, sval.StructType)
                and container.head.name_base == '$kwargs'
            ):
                # a ``dict[str, T]``/``**kwargs`` parameter's own storage (see §8.1)
                fields = container.fields()
                return {
                    name: ArgEntry(self.field_index_addr(value, _index_value(index)), True)
                    for index, name in enumerate(fields.keys)
                }
        aggregate = _as_aggregate(value)
        if aggregate is not None and isinstance(aggregate.type, sval.StrDictType):
            names = list(aggregate.type.values.keys())
            return {name: ArgEntry(ComptimeVal(element), False) for name, element in zip(names, aggregate.values)}
        if isinstance(ptr_type, sval.StrDictType):
            slot = self.alloca(False)
            self._commit_pending_slot(slot, ptr_type)
            self.store(slot, value)
            return {
                name: ArgEntry(self.field_index_addr(slot, _index_value(index)), True)
                for index, name in enumerate(ptr_type.values.keys())
            }
        raise CompileError(f'cannot take the values of the argument value {ev!r}')

    def _call_function_entry(
        self,
        fn: FunctionValue,
        args: RawArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
        on_return: Callable[[], None] | None = None,
        catch_unwind: bool = False,
    ) -> PollResult:
        """A call of a registered spy function with the given (already
        evaluated) argument values - the common tail of an ordinary
        function call and of a method call (whose ``self`` the caller
        prepended to the arguments).  The call is specialized
        from the marshaled argument types (an annotated parameter fixes its
        type, an unannotated one is typed by its argument); a plain Python
        callee (``force_inline``) is inlined into the current stream
        instead of being compiled into a native specialization.

        ``on_return`` is invoked once the callee delivered its result into
        ``ret`` - directly for an inlined body (see ``_start_inline``), or
        after ``_make_runtime_call`` in the ``_resumer`` of a compiled one -
        so that a caller that has more to do with the result than store it
        (``subscript``) can do so; the callee's own result location is
        unaffected.

        ``generic_var_values`` are the type-argument values of the struct
        the callee is a method of (see :class:`sval.BoundMethod`): the
        method's signature names the struct's type parameters, and they are
        substituted into it before it is specialized.

        ``catch_unwind`` emits a may-panic call as a ``CatchUnwind`` instead
        of a ``CallMayPanic`` (see ``_builtin_catch_unwind``)."""
        sig = fn.hir.signature
        if generic_var_values:
            sig = sig.substitute_type_vars(dict(generic_var_values))
        binded_args = sig.bind_arg_pos(args, lambda e: ArgEntry(ComptimeVal(e), False))
        _check_comptime_args(sig, binded_args)
        if fn.force_inline:
            # an undecorated plain Python function or a forced-inline closure:
            # its body is inlined into the current stream (it has no native
            # specialization of its own).  Its declared return type still says
            # when it can never return a value, which its body must respect (see
            # ``_current_value_is_empty``).  Its own type parameters are solved
            # from the argument types here, since the body may name them in a
            # type expression (``syntax.MultiPtr[T]``, see ``astgen``): the frame
            # then resolves them like the type arguments of a compiled call (see
            # ``operand``)
            solved = sig.solve_param_types(self._provided_types(sig, binded_args))
            frame_values: dict[sval.TypeVar, sval.AnyValue] = dict(zip(sig.generic_args, solved))
            if generic_var_values:
                frame_values.update(generic_var_values)
            # the concrete declared type of every positional parameter: an inlined
            # argument is declared with it, so a bare literal takes the
            # annotation's type instead of its own (untyped) one - a compiled
            # call's specialization types its arguments the same way.  A
            # ``type[X]`` parameter's slot holds the spy type its argument
            # denotes, so it is declared with ``type`` (``sval.TYPE_TYPE``); the
            # formal's own ``type`` is the *bound* of ``X``, not the slot's type
            sub_sig = sig.substitute_type_vars(frame_values)
            param_types = tuple(
                None if formal is None
                else (sval.TYPE_TYPE if formal.is_type_value else formal.type)
                for formal in sub_sig.positional.values()
            )
            return self._start_inline(
                fn.hir.body, fn.hir.arg_is_ref, binded_args, ret,
                fn.name_base,
                frozendict(frame_values),
                value_is_empty=isinstance(sig.ret_type, sval.EmptyType),
                on_done=on_return,
                param_types=param_types,
                has_varargs=sig.varargs is not None,
                has_kwargs=sig.kwargs is not None,
            )
        provided = self._provided_types(sig, binded_args)
        call_sig, partial_ret_sig = sig.specialize(provided, self._mir_cache)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            instance = fn.specs.get(call_sig)
            res = self0._make_runtime_call(
                fn_mir, binded_args, ret, call_sig, ret_sig,
                catch_unwind=catch_unwind,
                comptime_values=None if instance is None else instance.ret_comptime_values,
            )
            if on_return is not None and not ret_sig.value_is_empty():
                # the callee delivered its value into ``ret``: finish what the
                # caller had to do with it (see ``subscript``)
                on_return()
            return res

        return self._request_function(fn, call_sig, partial_ret_sig, _resumer, generic_var_values)

    def _request_function(
        self,
        fn: FunctionValue,
        call_sig: CallSignature,
        ret_sig: PartialReturnSignature,
        resumer: Callable[[Self, mir.Value, ReturnSignature], PollResult],
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
    ) -> PollResult:
        """Request the compilation of the specialization ``call_sig`` of
        ``fn`` and resume this runner through ``resumer`` when it is typed: the
        resumer runs right away when the specialization is already compiled (or
        is the very function being typed, i.e. recursion), and otherwise is
        stored and runs when the callee's runner ends (``resume``, see
        ``Analyser._run``).  Returns ``AGAIN`` or ``SUSPEND``."""
        self._fn_req_resumer = resumer
        res = self._analyser._request_function(fn, call_sig, ret_sig, generic_var_values)
        if res is not None:
            fn_mir, actual_ret_sig = res
            return self.resume(fn_mir, actual_ret_sig)
        return PollResult.SUSPEND

    def resume(self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
        resumer = self._fn_req_resumer
        assert resumer is not None
        self._fn_req_resumer = None
        return resumer(self, fn_mir, ret_sig)

    def _make_runtime_call(self, callee: mir.Value, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal, call_sig: CallSignature, ret_sig: ReturnSignature, catch_unwind: bool = False, comptime_values: list[sval.AnyValue | None] | None = None) -> PollResult:
        """Emit the native call of an already-resolved callee and hand its
        result to the call's result location.  A callee that cannot return
        normally (no value and no error, or no value at all) ends the current
        path, which is unwound like any other ended one (see ``_cut``)."""
        mir_args: list[mir.Value] = []

        def convert_one(arg: ArgEntry[InterpVal], sig_arg: ArgNode) -> None:
            if not isinstance(sig_arg, RuntimeArgNode):
                # a compound argument: its elements are passed separately, in
                # their own order (see ``ArgNode``)
                if isinstance(sig_arg, tuple):
                    for child, sub in zip(sig_arg, self._tuple_arg_entries(arg)):
                        convert_one(sub, child)
                    return
                if isinstance(sig_arg, frozendict):
                    entries = self._dict_arg_entries(arg)
                    for key, child in sig_arg.items():
                        convert_one(entries[key], child)
                    return
                # a compile-time leaf (a value, a compile-time pointer, ...)
                # carries no runtime value and is never passed
                return
            if sig_arg.is_ref:
                if arg.is_ref:
                    ev = _shallow_normalize(arg.value)
                    if not (isinstance(ev, RuntimeVal) and isinstance(ev.type, sval.PointerType)):
                        raise CompileError('cannot pass a compile-time reference by pointer')
                    if ev.type.elem != sig_arg.type:
                        # the address is of another type than the parameter takes:
                        # handing it over would alias the argument as that type
                        # (the pointers of the MIR are untyped), and an aggregate
                        # is only ever its own type (see ``store``)
                        raise CompileError(
                            f'cannot convert a {ev.type.elem} value to {sig_arg.type}'
                        )
                    mir_args.append(ev.value)
                else:
                    slot = self.alloca(False)
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

        spec = ret_sig.ret_spec(self._mir_cache)
        callee_result = ret_sig.result_type()
        value_is_empty = ret_sig.value_is_empty()
        # a call that may panic while the enclosing regions still hold deferred
        # bodies a panic would run needs a landing pad: it is emitted as a
        # ``CallMayPanic`` whose unwind edge runs those bodies and reaches the
        # function-wide resume block.  ``catch_unwind`` instead catches the
        # panic itself (a ``CatchUnwind``), so it never takes this path.
        unwind_defers: tuple[mir.BasicBlock, ...] | None = None
        if call_sig.may_panic and not catch_unwind:
            defers = self._unwind_defers()
            if len(defers) > 0:
                unwind_defers = defers
        if len(ret_sig.exceptions) == 0:
            # the callee has no error part in its MIR: deliver its value alone
            # (and when that value is the empty type, nothing comes back at all)
            self._deliver_result(
                callee, mir_args, ret, ret_sig.ret_type_spec,
                noreturn=ret_sig.is_noreturn(),
                unwind_defers=unwind_defers, catch_unwind=catch_unwind,
                comptime_values=comptime_values,
            )
            if value_is_empty:
                return self._cut()
            return PollResult.AGAIN
        # the callee may raise: it delivers its normal result into ``ret`` and
        # its error code and payload through fresh places, dispatched below
        callee_exceptions = tuple(ret_sig.exceptions.values)
        use_ret_payload = self._use_ret_payload(callee_exceptions)
        code_place = self.alloca(False)
        if use_ret_payload:
            # no ``try`` catches the callee's errors: it writes them straight
            # into the function's own error payload (no copy on the way out)
            payload_place: InterpVal = self._function_result().payload
        else:
            payload_place = self.alloca(False)
        error_tuple: tuple[InterpVal, InterpVal] = (code_place, payload_place)
        self._deliver_result(
            callee, mir_args, ret, spec, error_tuple,
            unwind_defers=unwind_defers, catch_unwind=catch_unwind,
            comptime_values=comptime_values,
        )
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
            defer_blocks = self._collect_exit_defers(
                hir.DeferPath.RAISE, data, inclusive=True, current_frame_only=False,
            )
            incoming = self._union_variant_ptr(payload_place, exception)
            self._route_to_clause(data, clause, incoming, exception, defer_blocks)
        else:
            defer_blocks = self._collect_exit_defers(
                hir.DeferPath.RAISE, None, inclusive=False, current_frame_only=False,
            )
            self._add_function_exception(exception)
            self._defer_error_code_write(exception)
            if not use_ret_payload and not exception.is_zst():
                value = self.load(self._union_variant_ptr(payload_place, exception))
                self.store(self._payload_variant_ptr(exception), value)
            self._end_error_path(defer_blocks)

    def _deliver_result(
        self,
        callee: mir.Value,
        mir_args: list[mir.Value],
        ret_loc: InterpVal,
        spec: RetSpec,
        err_loc: tuple[InterpVal, InterpVal] | None = None,
        noreturn: bool = False,
        unwind_defers: tuple[mir.BasicBlock, ...] | None = None,
        catch_unwind: bool = False,
        comptime_values: list[sval.AnyValue | None] | None = None,
    ) -> None:
        """Emit the native call of a callee returning the value(s) of ``spec``
        and hand every result to its place.

        The results go either into the places the caller reserved - ``ret_loc``
        is the place of the value part (a single place, or a tuple of places
        nested exactly like the results, see ``_pair_places``), with the error
        code and payload union of the callee in ``err_loc`` (``None`` for a
        callee that raises nothing) - or, when a result (or a group of them) is
        given a single place of its own, into fresh places whose tuple is then
        stored in that place.  A tuple may be inlined, so such a place has to be
        able to hold its value inline (a ``Comptime`` variable); a plain slot
        rejects the tuple when it is committed (see ``_no_runtime_type``).

        A zero-sized result occupies no place of its own: its unit value is
        written into its place.  A ``noreturn`` call (``mir.NoReturn``) ends the
        block it is emitted into."""
        places: list[InterpVal] = []
        packed: list[tuple[InterpVal, InterpVal | tuple]] = []
        if err_loc is None:
            self._pair_places(spec, ArgEntry(ret_loc, True), places, packed)
        else:
            # the spec's root is the result-type group: its value, its error code
            # and its payload union are the three places the caller reserved
            code, payload = err_loc
            assert isinstance(spec, RetTuple) and isinstance(spec.type, sval.ResultType)
            for child, sub in zip(spec.values, (ret_loc, code, payload)):
                self._pair_places(child, ArgEntry(sub, True), places, packed)

        result_args: list[mir.Value] = []
        ret_type: mir.ReturnType = mir.VOID
        by_value: tuple[InterpVal, sval.Type] | None = None
        for leaf, place in zip(iter_ret_leaves(spec), places):
            if isinstance(leaf, RetComptime):
                # a fully compile-time result leaf: no MIR argument - the callee's
                # compile-time value (its per-``FunctionInstance`` list, see §7.3)
                # is stored into the reserved place
                if comptime_values is None or leaf.index >= len(comptime_values):
                    raise CompileError(
                        'the call delivered no compile-time result the callee returned'
                    )
                self.store(place, ComptimeVal(comptime_values[leaf.index]))
                continue
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
        if catch_unwind and not noreturn:
            # catch the panic the callee may raise (see ``mir.CatchUnwind``):
            # the unwind edge lands in a fresh catch block the interpreter fills
            # with the construction of the ``UnwindException`` and its dispatch,
            # and the normal edge continues in a fresh block of its own
            normal = mir.BasicBlock()
            catch_block = mir.BasicBlock()
            panic_type = self._panic_data_type()
            catch_data = self.alloca(False)
            self._commit_pending_slot(catch_data, sval.PointerType(panic_type, is_const=False))
            catch_data_ptr = self._to_runtime(_shallow_normalize(catch_data))
            call_inst = self._emit(mir.CatchUnwind(
                callee, (*mir_args, *result_args), ret_type,
                normal, catch_block, catch_data_ptr,
            ))
            self._cur_block = catch_block
            self._build_catch_path(catch_data)
            self._cur_block = normal
        elif unwind_defers is not None and not noreturn:
            # run the deferred bodies the panic would trigger on the unwind edge
            # and let the function-wide resume block carry the unwinding on
            normal = mir.BasicBlock()
            call_inst = self._emit(mir.CallMayPanic(
                callee, (*mir_args, *result_args), ret_type,
                normal, self._resume_block(), unwind_defers,
            ))
            self._cur_block = normal
        else:
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
        for target, tree in packed:
            self._deliver_packed_tuple(target, tree)

    def _deliver_packed_tuple(self, target: InterpVal, tree: InterpVal | tuple) -> None:
        """Deliver a multi-value result packed into the single place ``target``
        (a call the caller gave neither a place per result nor a tuple target, see
        ``_pair_places``).  A still uncommitted slot records the tree's leaf places
        as its own tuple storage (see ``_PendingTuple``); any other place receives
        the elements, one per field."""
        leaves = _flatten_result_tree(tree)
        if isinstance(target, PendingSlot) and target.committed is None:
            self._record_pending_action(target, _PendingTuple(leaves))
            return
        for index, place in enumerate(leaves):
            self.store(self.field_index_addr(target, _index_value(index)), self.load(place))

    def _pair_places(
        self,
        node: RetSpec,
        entry: ArgEntry[InterpVal],
        places: list[InterpVal],
        packed: list[tuple[InterpVal, InterpVal | tuple]],
    ) -> None:
        """Pair the results of ``node`` with the place tree the caller reserved: a
        single result takes the place ``entry`` denotes, a group takes a (nested)
        tuple of places nested exactly like the results (see ``astgen``), and the
        leaves are appended to ``places`` in depth-first order.  A group the caller
        gives a single non-tuple place is *packed*: fresh places are reserved and
        their (nested) tuple is stored in that place after the call (see
        ``_deliver_result``).  The root result-type group is unwrapped by the
        caller (``_deliver_result``), so ``node`` is never one here."""
        target = entry.value
        match node:
            case RetValue() | RetComptime():
                if isinstance(target, tuple):
                    raise CompileError('cannot unpack one value into a tuple target')
                if isinstance(node, RetValue) and not entry.is_ref:
                    raise CompileError(
                        'the target of a multi-value result must be addressable'
                    )
                places.append(target)
            case RetTuple(type=type, values=values):
                assert not isinstance(type, sval.ResultType)
                if isinstance(target, tuple):
                    if len(target) != len(values):
                        raise CompileError(
                            f'cannot unpack {len(values)} value(s) into '
                            f'{len(target)} target(s)'
                        )
                    for child, sub in zip(values, target):
                        self._pair_places(child, ArgEntry(sub, True), places, packed)
                    return
                tree = self._fresh_places(node, places)
                packed.append((target, tree))

    def _fresh_places(self, node: RetSpec, places: list[InterpVal]) -> InterpVal | tuple:
        """Fresh places for a group packed into a single location: a (nested)
        Python tuple of places mirroring the shape of the results, with the leaf
        places appended to ``places`` in depth-first order."""
        match node:
            case RetValue() | RetComptime():
                place = self.alloca()
                places.append(place)
                return place
            case RetTuple(values=values):
                return tuple(self._fresh_places(child, places) for child in values)
        raise CompileError(f'cannot reserve fresh places for {node!r}')

    def _inline_arg_place(self, arg: ArgEntry[InterpVal], declared: sval.Type | None = None) -> InterpVal:
        """The place an inlined callee's argument (a positional by-value
        argument, or a ``*args``/``**kwargs`` element) is bound to: a reference
        argument is the address it already carries, any other value is written
        into a fresh slot first (the copy a value parameter is, see
        ``_start_inline``).

        ``declared`` is the type the callee's parameter declares: the slot is
        declared with it, so an argument is typed by the annotation rather than
        by its own form (a bare literal takes the annotated type instead of the
        untyped one).  Whether the storage is a compile-time place or memory is
        left to the stores (see ``alloca``)."""
        if arg.is_ref:
            return arg.value
        if declared is None:
            slot = self.alloca()
        else:
            slot = self.alloca(declared=declared, commit_declared=False)
        self.store(slot, arg.value)
        self._commit_pending_slot(slot)
        return slot

    def _start_inline(
        self,
        body: tuple[hir.Inst, ...],
        arg_is_ref: tuple[bool, ...],
        args: ArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
        fn_name: str,
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
        value_is_empty: bool = False,
        on_done: Callable[[], None] | None = None,
        param_types: tuple[sval.Type | None, ...] | None = None,
        has_varargs: bool = False,
        has_kwargs: bool = False,
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
        arg_values: list[InterpVal] = []
        for index, (arg, by_ref) in enumerate(zip(args.positional, arg_is_ref)):
            # ``by_ref`` is the *callee's* HIR binding: a method's ``self`` is
            # bound directly to its argument (the address) even though the
            # caller passed it as a value; every other parameter is forwarded as
            # the caller passed it (by reference), while a by-value parameter the
            # caller passed by value is given a slot of its own (the copy a
            # value parameter is)
            if by_ref or arg.is_ref:
                arg_values.append(arg.value)
            else:
                declared = (
                    None if param_types is None or index >= len(param_types)
                    else param_types[index]
                )
                arg_values.append(self._inline_arg_place(arg, declared))
        # the varargs/kwargs the call bound come after every positional
        # parameter, as one tuple/dict place tree (see ``hir.Arg`` and
        # ``_init_args_from_signature``, which does the same for a compiled
        # callee)
        if has_varargs:
            arg_values.append(self._vararg_struct_from_entries(args.varargs))
        if has_kwargs:
            arg_values.append(self._vararg_struct_from_entries(args.kwargs))
        frame_values: dict[sval.TypeVar, InterpVal] = {}
        if generic_var_values is not None:
            frame_values = {tv: ComptimeVal(v) for tv, v in generic_var_values.items()}
        # the body delivers its result into the call's place and shares the
        # function proper's error places: an error of the body leaves the
        # function the same way
        error = self._function_result()
        frame = InlineFrame(
            frame_values, tuple(arg_values),
            ComptimeResultPtr(ret, error.code, error.payload),
            body, fn_name, value_is_empty=value_is_empty, on_done=on_done,
            compile_vars=self._inherit_compile_vars(),
        )
        self._frames.append(frame)
        return PollResult.AGAIN

    # -- finishing -------------------------------------------------------------

    def finish(self) -> None:
        """Called when the body of the function proper has been fully
        typed: end the last block (a body that fell off its end returns
        void), fix an inferred return convention, flatten the deferred
        insertion blocks away and expand the deferred bodies into explicit
        copies (see ``mir.instantiate_defers``)."""
        if not self._cur_block.is_finished:
            # the body fell off its end (the last block is not closed): it
            # returns void, after running the defers of its top-level region
            self._cur_block.emit(mir.Ret(None, _normal_defers(self._frames[0].body_defers)))
        self._finish_function()
        mir_fn = self._fn_instance.mir
        mir.normalize(mir_fn)
        # the deferred bodies were emitted as shared templates every triggering
        # transfer refers to; give the transfers their explicit copies now that
        # the CFG is complete (and the insertion placeholders are gone)
        mir.instantiate_defers(mir_fn)

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
            value_spec = sval.make_ret_spec(location.committed_type(), self._mir_cache)
            if isinstance(value_spec, sval.RetTuple):
                # a multi-value result: the body wrote its values into the element
                # places of the result slot (a tuple literal, a packed multi-value
                # call, or a tuple value); the result location becomes the place
                # tree those places make up, which ``_commit_ret_places`` then
                # gives its result pointers (see ``_result_place_tree``)
                ret_loc = self._current_result_loc()
                self._frames[-1].ret_loc = ComptimeResultPtr(
                    self._result_place_tree(location), ret_loc.code, ret_loc.payload,
                )
        exceptions = partial.exceptions if partial is not None else None
        callconv = partial.callconv if partial is not None else 'default'
        if exceptions is None:
            exceptions = ArraySet()
            for exception in self._error_types.values:
                exceptions.add(exception)
        self._materialize_ret_sig(ReturnSignature(value_spec, exceptions, callconv))
        sig = self.ret_sig
        assert sig is not None
        result_type = sig.result_type()
        self._write_deferred_error_codes(result_type)
        spec = sig.ret_spec(self._mir_cache)
        places = self._result_places(self._current_result_loc())
        if len(self._deferred_returns) > 0:
            # every return of this function deferred its ``mir.Ret`` (its return
            # convention was not fixed yet), so none of them went through
            # ``_emit_function_return``: record the compile-time result leaves
            # here, from the place the deferred return delivers - the last written
            # value wins, exactly like that return (see §7.3)
            for leaf, place in zip(iter_ret_leaves(spec), places):
                if not isinstance(leaf, RetComptime):
                    continue
                self._record_ret_comptime_value(leaf, place)
        index = ret_by_value_index(spec)
        for block, defer_blocks in self._deferred_returns:
            if index is None:
                block.insts.append(mir.Ret(None, defer_blocks))
            else:
                slot = _shallow_normalize(places[index])
                if isinstance(slot, RuntimeVal):
                    load = mir.Load(slot.value)
                    block.insts.append(load)
                    block.insts.append(mir.Ret(load, defer_blocks))
                elif isinstance(slot, ComptimeVal) and isinstance(slot.obj, sval.ComptimePtr):
                    # a compile-time result place (a box): its value is lowered into
                    # a constant the deferred return delivers
                    block.insts.append(mir.Ret(self._to_runtime(self.load(slot)), defer_blocks))
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
    def __init__(self, resolver: CompileContext, mir_lower_cache: sval.MirLowerCache) -> None:
        self._resolver = resolver
        # the host's MIR-mirror interning table, which every mirror a body
        # creates is made through (see ``sval.MirLowerCache``)
        self.mir_lower_cache = mir_lower_cache
        self._analyse_stack: list[HirRunner] = []
        # the byte-string globals the bodies compiled by this analyser share,
        # deduplicated by content (see ``HirRunner._string_global``): identical
        # byte strings lower to one global across every function of the analysis
        self._string_globals: dict[bytes, mir.GlobalStringValue] = {}
        self._symbol_table = CompileBatch(
            extern_anon_symbols={},
            newly_compiled=set(),
            mir_lower_cache=mir_lower_cache,
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
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
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
        instance.call_sig = call_sig
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

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

import math
import operator
import types as pytypes
from abc import abstractmethod
from annotationlib import Format
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Self, override

from . import hir, mir, sval
from .binop import BinaryOp, CompareOp, UnaryOp
from .errors import CoerceError, CompileError
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
    signature_of_fn_type,
)
from .hir import InlineMode
from .sval import (
    CompileContext,
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
class ComptimeBox(InterpVal):
    """A comptime-time writable box for non-aggregate values. Note that
    ``value`` does not have to be a comptime-time value: it also can be
    a runtime value :class:`RuntimeVal`. Supports non-aggregate values only,
    for aggregate values, use :class:`ComptimeAggregate` or
    :class:`ComptimeAggregatePtr` instead.

    A box may be ``is_const`` - a field of the ``SlicePtr`` a slice subscript
    builds is one (see ``HirRunner.slice_ptr``): the value it holds is read
    through it, but no store may write through it."""

    type: sval.Type
    value: InterpVal
    is_const: bool = False


class _PendingActionData:
    """One action recorded by a :class:`PendingSlot`: how it is delivered
    once the slot has an address is decided by the runner (see
    ``HirRunner._exec_pending_action``), the slot itself only asks for the
    type it contributes and whether it is inline."""

    @abstractmethod
    def info(self) -> tuple[sval.Type, bool]:
        """Returns (type, is_inline) of the value the action delivers."""
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

@dataclass(frozen=True, slots=True)
class ComptimeCastedPtr(InterpVal):
    """A compile-time pointer ``syntax.ptr_cast`` reinterpreted as another
    pointer type: the place it points at (a :class:`ComptimeBox` or a
    :class:`ComptimeAggregatePtr`) and the pointer type it is viewed as.  The
    place keeps its own storage type, so reading or writing *through* the cast
    pointer is not supported yet (the cast only changes what the pointer is
    typed as); a cast whose target the place converts to is materialized with
    ``_coerce`` instead and never produces one of these."""

    place: InterpVal
    type: sval.PointerType

@dataclass(frozen=True, slots=True)
class ComptimeOption(InterpVal):
    """The value form of an ``Option[T]``: a compile-time option, holding
    whether it is *absent* (``is_null``, a ``bool`` value) and the value it
    wraps (``value``, which may itself be a runtime value or a nested
    aggregate).  It is the option counterpart of :class:`ComptimeAggregate`; an
    option that is absent is still this form (its ``is_null`` is true), with the
    payload there only to be discarded.  The untyped ``sval.Null``/typed
    ``sval.TypedNull`` a Python expression evaluates to still occur as
    ``ComptimeVal``s, but any option *place* holds one of these."""

    is_null: InterpVal
    value: InterpVal


@dataclass
class ComptimeOptionPtr(InterpVal):
    """The compile-time storage of an ``Option[T]``: a pointer ``*Option[T]``
    whose tag is a *value* (``is_null``, a ``bool``) rather than an addressable
    place - the tag has no address of its own, only the payload does - and
    whose payload is a place (``payload_ptr``), of the child type.  It is the
    option counterpart of :class:`ComptimeAggregatePtr`: ``HirRunner.load``
    reads the option out of it, ``HirRunner.store`` splits a value into the tag
    and the payload, and ``HirRunner._option_payload_ptr`` hands the payload
    place over.  Both the tag and the payload may be runtime values."""

    is_null: InterpVal
    payload_ptr: InterpVal


@dataclass(frozen=True, slots=True)
class ComptimeTaggedUnionValue(InterpVal):
    """The value form of a tagged union: a compile-time tagged union holding
    the variant it is (``tag``, the position of the variant in the union) and
    that variant's value (``value``, which may itself be a runtime value or a
    nested aggregate).  It is the tagged-union counterpart of
    :class:`ComptimeAggregate`.

    The tag may be only known at runtime (a value read out of a runtime union);
    then ``value`` is not the variant but the (untagged) payload union storage
    of the type (``type.payload_type()``), and the tag picks the variant it
    holds - see ``_tagged_union_parts_to_runtime``."""

    type: sval.TaggedUnionType
    tag: InterpVal
    value: InterpVal


@dataclass
class ComptimeTaggedUnionPtr(InterpVal):
    """The compile-time storage of a tagged union: a pointer ``*Union`` whose
    tag is a *value* (``tag``, the position of the current variant) rather than
    an addressable place, and whose payload is a place (``payload_ptr``) of the
    current variant's type.  It is the tagged-union counterpart of
    :class:`ComptimeAggregatePtr`: ``HirRunner.load`` reads the union out of it,
    ``HirRunner.store`` splits a value into the tag and the payload, and
    ``HirRunner._tagged_union_payload_ptr`` hands the payload place over.  When
    the variant changes the payload place is rebuilt.

    When the tag is only known at runtime, ``payload_ptr`` points at the
    (untagged) payload union storage of the type rather than at a variant's own
    place, and a variant is read/written through it by reinterpreting its
    address (see ``HirRunner._store_comptime_tagged_union``)."""

    type: sval.TaggedUnionType
    tag: InterpVal
    payload_ptr: InterpVal


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
            case ComptimeOption():
                # likewise for an option value form (see ``ComptimeOption``)
                todo.append(val.is_null)
                todo.append(val.value)
            case ComptimeOptionPtr():
                # and its place form (see ``ComptimeOptionPtr``)
                todo.append(val.is_null)
                todo.append(val.payload_ptr)
            case ComptimeTaggedUnionValue():
                # a tagged union held compile-time is comptime when both its tag
                # and the variant value it holds are (see
                # ``ComptimeTaggedUnionValue``)
                todo.append(val.tag)
                todo.append(val.value)
            case ComptimeTaggedUnionPtr():
                # and its place form (see ``ComptimeTaggedUnionPtr``)
                todo.append(val.tag)
                todo.append(val.payload_ptr)
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
            case ComptimeCastedPtr():
                # a cast pointer is as compile-time as the place it wraps
                todo.append(val.place)
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

def _is_undefined_val(val: InterpVal) -> bool:
    """Whether the value is the undefined literal - the untyped
    ``sval.UntypedUndefined`` ``std.core.undefined`` evaluates to, or the typed
    ``sval.Undefined`` it becomes once it is coerced to a type: a store of one
    leaves its destination undefined (see ``HirRunner.store``)."""
    obj = _to_comptime(_shallow_normalize(val))
    return isinstance(obj, (sval.UntypedUndefined, sval.Undefined))

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


@dataclass(frozen=True, slots=True)
class _DeferEntry:
    """One ``with syntax.defer():`` region declared in an open region: its kind
    and the entry block of the deferred body the interpreter emitted (the
    template every transfer that triggers it refers to, see
    ``mir.EndDefer``/``Terminator.get_defer_blocks``)."""

    kind: hir.DeferKind
    entry: mir.BasicBlock


@dataclass
class DeferBlockData(BlockFrameData):
    """The state of the defer body currently being walked (a
    ``with syntax.defer():`` region).  ``variant`` is the kind of the defer,
    ``entry`` the entry block of the body (appended to the enclosing region's
    defer list), ``body_defers`` the defers declared inside the body itself -
    they are what its ``mir.EndDefer`` triggers - and ``saved_block`` the block
    the walk continues in after the body, which is emitted detached from the
    current one (see ``HirRunner._exec_defer``)."""

    variant: hir.DeferKind
    entry: mir.BasicBlock
    body_defers: list[_DeferEntry]
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
    def __init__(self, generic_var_values: dict[sval.TypeVar, InterpVal], arg_values: tuple[InterpVal, ...], ret_loc: ComptimeResult, insts: tuple[hir.Inst, ...], value_is_empty: bool = False, on_done: Callable[[], None] | None = None, compile_vars: CompileVars | None = None) -> None:
        self.generic_var_values = generic_var_values
        self.arg_values = arg_values
        # the compile-time configuration of this body, as a stack so a nested
        # scope can push an override and pop it again; the bottom entry is
        # inherited from the caller (a fresh default for the function proper)
        self.compile_vars_stack: list[CompileVars] = [CompileVars() if compile_vars is None else compile_vars]
        # the frame's result location: the place its result is delivered into and
        # the function proper's error places (an inlined plain body raises into
        # the enclosing function's error location, so it shares them)
        self.ret_loc: ComptimeResult = ret_loc
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


def _append_defers(out: list[mir.BasicBlock], defers: list[_DeferEntry], is_error: bool) -> None:
    """Append the entries of ``defers`` a transfer that exits with ``is_error``
    triggers, in the order they run - reverse declaration order (see
    ``hir.DeferKind``): ``defer`` always, ``okdefer`` only on a normal exit and
    ``errdefer`` only on an error one."""
    for entry in reversed(defers):
        if entry.kind is hir.DeferKind.DEFER:
            out.append(entry.entry)
        elif entry.kind is hir.DeferKind.OKDEFER:
            if not is_error:
                out.append(entry.entry)
        elif is_error:
            out.append(entry.entry)


def _normal_defers(defers: list[_DeferEntry]) -> tuple[mir.BasicBlock, ...]:
    """The entries of ``defers`` a *normal* exit of the region they were
    declared in triggers (``defer`` and ``okdefer``), in the order they run."""
    out: list[mir.BasicBlock] = []
    _append_defers(out, defers, False)
    return tuple(out)


def _sval_to_runtime(value: sval.AnyValue, cache: sval.MirLowerCache) -> mir.Value:
    match value:
        case bool():
            return mir.BoolValue(value)
        case sval.Int():
            return mir.Int(value.value, mir.IntType(value.type.bits, value.type.signed))
        case sval.Float():
            return mir.Float(value.value, mir.FloatType(value.type.bits))
        case sval.DeclareFunction():
            # an external function is a global symbol: a function pointer is a
            # reference to it (see ``mir.ExternSymbol`` and ``lower``)
            mir_type = value.get_type().to_mir_type(cache)
            assert isinstance(mir_type, mir.PointerType)
            return mir.ExternSymbol(value.linkname, mir_type)
        case sval.Undefined():
            # an undefined value has no defined content: it materializes as
            # LLVM's ``undef`` of its type
            mir_type = value.type.to_mir_type(cache)
            if mir_type is None:
                raise _no_runtime_type(value.type)
            return mir.UndefValue(mir_type)
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
            # a compile-time writable pointer: const when its box is (a field of
            # a slice, see ``ComptimeBox``)
            return sval.PointerType(ev.type, is_const=ev.is_const)
        case ComptimeOption(_, value):
            # an option held compile-time: the option of the type of the value
            # it wraps (see ``ComptimeOption``)
            child = _type_of(value)
            return None if child is None else sval.OptionType(child)
        case ComptimeOptionPtr(_, payload_ptr):
            # the compile-time place of an option: a pointer to it, its child
            # read off the payload place (see ``ComptimeOptionPtr``)
            payload_type = _type_of(payload_ptr)
            if not isinstance(payload_type, sval.PointerType):
                return None
            return sval.PointerType(sval.OptionType(payload_type.elem), is_const=False)
        case ComptimeTaggedUnionValue(type):
            # a tagged union held compile-time: the union type (see
            # ``ComptimeTaggedUnionValue``)
            return type
        case ComptimeTaggedUnionPtr(type):
            # the compile-time storage of a tagged union: a pointer to it
            return sval.PointerType(type, is_const=False)
        case ComptimeAggregate(type):
            # a struct value held compile-time: the struct type (see
            # ``ComptimeAggregate``)
            return type
        case ComptimeAggregatePtr(type):
            # the compile-time storage of a struct: a pointer to it
            return sval.PointerType(type, is_const=False)
        case ComptimeCastedPtr(_, type):
            # a pointer reinterpreted by ``ptr_cast``: the pointer type it is
            # viewed as (see ``ComptimeCastedPtr``)
            return type
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

def _comptime_int(ev: InterpVal) -> int | None:
    """The Python integer a compile-time integer value denotes, or None when
    the value is not one at all (a runtime value, or no integer)."""
    if isinstance(ev, ComptimeOption):
        # a present option holds its value (see ``ComptimeOption``)
        ev = ev.value
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
    """The Python integer a compile-time index denotes (a field or element of
    an aggregate held compile-time is always named by a compile-time index)."""
    if isinstance(index, ComptimeVal) and isinstance(index.obj, sval.Int):
        return index.obj.value
    raise CompileError(
        'cannot pick a field or element with a runtime index: it has to be a '
        'compile-time integer'
    )

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
        # location has been materialized, together with the deferred bodies the
        # return runs first
        self._deferred_returns: list[tuple[mir.Insertion, tuple[mir.BasicBlock, ...]]] = []
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
            loc = self._ret_spec_place(ret_sig.complete().ret_spec(self._mir_cache))
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
                    # ZSTs are never present in SpecializedRuntimeArg
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
        if leaf.type.classify() == sval.SpecialTypeKind.DST:
            # a dynamically-sized leaf is only ever delivered through the result
            # pointer: reaching here means a convention forced it by value
            raise CompileError(
                f'cannot return a value of the dynamically-sized type {leaf.type} by value'
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
        places = _result_places(self._frames[0].ret_loc)
        index = ret_by_value_index(spec)
        if index is None:
            self._cur_block.emit(mir.Ret(None, defer_blocks))
        else:
            self._cur_block.emit(mir.Ret(self._to_runtime(self.load(places[index])), defer_blocks))


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
        exception = _place_type(slot)
        if not isinstance(exception, sval.StructType):
            raise CompileError(f'cannot raise {exception!r}: an exception must be a struct')
        target = self._find_catching_clause(exception)
        if target is not None:
            data, index = target
            defer_blocks = self._collect_exit_defers(
                True, data, inclusive=True, current_frame_only=False,
            )
            if exception.is_zst():
                incoming = ComptimeVal(sval.Undefined(sval.PointerType(exception, is_const=False)))
            else:
                self._commit_pending_slot(slot, exception)
                incoming = _shallow_normalize(slot)
            self._route_to_clause(data, index, incoming, exception, defer_blocks)
        else:
            defer_blocks = self._collect_exit_defers(
                True, None, inclusive=False, current_frame_only=False,
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
        is_error: bool,
        stop_data: BlockFrameData | None,
        inclusive: bool,
        current_frame_only: bool,
    ) -> tuple[mir.BasicBlock, ...]:
        """The deferred bodies a transfer out of the currently walked regions
        triggers, in the order they run: from the innermost region outwards,
        every region's ``defer``/``okdefer`` (on a normal exit) or
        ``defer``/``errdefer`` (on an error exit), each in reverse declaration
        order.  The walk stops at the region of ``stop_data`` - included or not
        per ``inclusive`` - and, when ``current_frame_only`` is set, at the
        current inline frame's own body (an inlined ``return``); otherwise it
        continues into the caller's frames, so that an error escaping an inlined
        body runs the defers of the caller's regions too.  A transfer that would
        leave a defer body is rejected."""
        ret: list[mir.BasicBlock] = []
        for frame in reversed(self._frames):
            for bf in reversed(frame.block_stack):
                if bf.data is stop_data:
                    if inclusive:
                        _append_defers(ret, self._region_defers(bf.data), is_error)
                    return tuple(ret)
                if isinstance(bf.data, DeferBlockData):
                    raise CompileError('cannot jump out of a defer block')
                _append_defers(ret, self._region_defers(bf.data), is_error)
            _append_defers(ret, frame.body_defers, is_error)
            if current_frame_only:
                return tuple(ret)
        if stop_data is not None:
            raise CompileError('the target of a transfer is not an open region')
        return tuple(ret)

    def _run_defers_on_fallthrough(self, defers: list[_DeferEntry], is_error: bool) -> None:
        """A region ended by falling off its end into a continuation that has no
        transfer instruction of its own (a compile-time ``if`` whose chosen
        branch fell through): when its deferred bodies have to run, split the
        block - jump through them to a fresh block and continue there."""
        out: list[mir.BasicBlock] = []
        _append_defers(out, defers, is_error)
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
        restores the block the region was opened in (see ``_exec_end``).

        A compile-time (``syntax.unroll``) loop has no transfer between its
        unrolled iterations, so a defer declared inside one has no exit to be
        deferred to; it is rejected for now."""
        frame = self._frames[-1]
        for bf in reversed(frame.block_stack):
            if isinstance(bf.data, LoopBlockData) and bf.data.is_inline:
                raise CompileError('a defer block inside a compile-time loop is not supported yet')
        entry = frame.pc - 1
        p_else, _p_end = self._scan_block(entry)
        assert p_else is None, 'a defer body has no else marker'
        template = mir.BasicBlock()
        self._current_defers().append(_DeferEntry(inst.variant, template))
        frame.block_stack.append(
            BlockFrame(entry, DeferBlockData(inst.variant, template, [], self._cur_block))
        )
        self._cur_block = template

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
                                False, None, inclusive=False, current_frame_only=True,
                            ),
                        ))
                    return self._cut()
                # a ``return`` of the function proper runs the defers of every
                # region it leaves, up to the function body
                defer_blocks = self._collect_exit_defers(
                    False, None, inclusive=False, current_frame_only=False,
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
                    regs[inst] = ComptimeVal(sval.Undefined(sval.PointerType(exception, is_const=False)))
                else:
                    regs[inst] = RuntimeVal(phi, sval.PointerType(exception, is_const=False))
            case hir.AsBool():
                return self.as_bool(self.operand_arg(inst.value), inst)
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
            case hir.Not():
                return self._eval_not(self.operand(inst.value), inst)
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
                self._collect_exit_defers(False, data, inclusive=True, current_frame_only=False),
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
                self._collect_exit_defers(False, data, inclusive=True, current_frame_only=False),
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
        created on demand."""
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
        time condition folds: a ``false`` one does nothing, and any other taken
        break ends the path at the target block's exit block like
        ``break_loop``/``continue`` do (see ``_cut``).  A *runtime* condition
        splits the block being typed instead: the taken edge jumps to the exit
        and the walk continues in a fresh block for the rest of the body."""
        target_bf = self._find_block(inst.levels)
        target = target_bf.data
        assert isinstance(target, PlainBlockData)
        exit_block = self._block_exit(target)
        defer_blocks = self._collect_exit_defers(
            False, target, inclusive=True, current_frame_only=False,
        )
        if inst.cond is not None:
            cond = self.operand(inst.cond)
            if isinstance(cond, ComptimeVal):
                if not cond.obj:
                    return PollResult.AGAIN
            else:
                cont_block = mir.BasicBlock()
                self._cur_block.emit(
                    mir.Br(self._to_runtime(cond), exit_block, cont_block, if_true_defer_blocks=defer_blocks)
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
        self._run_defers_on_fallthrough(self._region_defers(data), False)
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
            self._run_defers_on_fallthrough(self._region_defers(data), False)
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
                return ComptimeVal(sval.as_value(obj, ctx=self._analyser._resolver))
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
            case ComptimeOptionPtr(is_null, payload_ptr):
                # an option is held by its tag and its payload place: loading
                # one is the value form of the two (see ``ComptimeOptionPtr``)
                return ComptimeOption(is_null, self.load(payload_ptr))
            case ComptimeTaggedUnionPtr(union_type, tag, payload_ptr):
                # a tagged union is held by its tag and its payload place:
                # loading one is the value form of the two (see
                # ``ComptimeTaggedUnionPtr``)
                return ComptimeTaggedUnionValue(union_type, tag, self.load(payload_ptr))
            case ComptimeVal(obj) if isinstance(obj, sval.ConstRef):
                # a reference to an immutable compile-time global behaves like
                # the value it refers to
                return ComptimeVal(obj.value)
            case ComptimeVal(obj) if isinstance(obj, sval.StrConstPtr):
                # ``*p`` of a string constant: the byte the cursor names
                byte = self._str_constant_byte(obj, obj.cursor, 'load')
                return ComptimeVal(sval.Int(byte, sval.IntType(8, False)))
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

        ``ptr`` is a *place* and ``value`` a *value* of its element type - a
        place *is* a pointer here (see ``_type_of``), so the two are different
        kinds of thing and the value is never read out of what it points at:
        whoever hands a place over where a value is needed reads it (``astgen``
        emits the ``Load`` of a name or a subscript, see ``_as_value``).

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
            self._add_function_exception(type)
            self._defer_error_code_write(type)
            self.store(self._error_payload_ptr(ptr, type), value)
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
                            value_type, self._split_runtime_aggregate(value),
                        ),
                    )
                else:
                    # a second value into the same storage (the branches of an
                    # ``if`` expression): its fields are places already, which
                    # the field values are written into
                    for index, field_value in enumerate(
                        self._runtime_aggregate_field_values(value)
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
        if isinstance(ptr, ComptimeVal) and isinstance(ptr.obj, sval.Undefined) and ptr_type.elem.is_zst():
            # a compile-time pointer with no storage at all: the address of a
            # zero-sized field or element (see ``field_index_addr``), or of a
            # zero-sized exception's variant (see ``_union_variant_ptr``).
            # Every value of the type it points at is the type's unit value, so
            # a store into it records nothing
            return

        if _is_undefined_val(value):
            # an undefined value leaves its destination undefined: a zero-sized
            # one has no storage, a runtime location is filled with ``undef``,
            # and the compile-time storage forms record it in their own way
            elem_unit = elem.get_unit_value()
            match ptr:
                case ComptimeBox():
                    if ptr.is_const:
                        raise CompileError(
                            f'cannot store through the const pointer {ptr.type}'
                        )
                    ptr.value = ComptimeVal(
                        elem_unit if elem_unit is not None else sval.Undefined(elem)
                    )
                case ComptimeAggregatePtr():
                    # an aggregate is its fields' (or elements') own places: each
                    # of them is left undefined in turn
                    for index in range(len(_aggregate_place_types(elem))):
                        self.store(self.field_index_addr(ptr, _index_value(index)), value)
                case ComptimeOptionPtr():
                    # the tag is what says whether the option is present; with no
                    # value, it is left undefined (the payload is never read)
                    ptr.is_null = ComptimeVal(sval.Undefined(sval.BoolType()))
                case ComptimeTaggedUnionPtr():
                    # likewise for a union: its variant tag is left undefined
                    ptr.tag = ComptimeVal(sval.Undefined(ptr.type.tag_type()))
                case RuntimeVal():
                    if elem_unit is not None:
                        # a zero-sized type has no storage to write
                        return
                    mir_type = elem.to_mir_type(self._mir_cache)
                    if mir_type is None:
                        raise _no_runtime_type(elem)
                    self._emit(mir.Store(ptr.value, mir.UndefValue(mir_type)))
                case _:
                    raise CompileError(f'cannot leave a {elem} undefined')
            return

        if _is_aggregate(elem):
            # an aggregate is held by its fields (or elements), never by a box: a
            # whole value is written place by place into a compile-time aggregate
            # and as a whole into memory (see ``ComptimeAggregatePtr``)
            match ptr:
                case ComptimeAggregatePtr():
                    aggregate = _as_aggregate(value)
                    if aggregate is None:
                        # a *runtime* aggregate value: its fields are read out of
                        # the value itself (see ``as_comptime_aggregate``)
                        aggregate = self.as_comptime_aggregate(value)
                    # an aggregate value is only ever its own type, exactly like a
                    # runtime one (see ``_convert_inst``): an aggregate of another
                    # type of the same shape is not a copy of it.  A zero-sized
                    # destination has no storage at all, so whatever is delivered
                    # is a no-op
                    if aggregate.type != elem and not elem.is_zst():
                        raise CompileError(f'cannot convert a {aggregate.type} value to {elem}')
                    for index, field_value in enumerate(aggregate.values):
                        self.store(self.field_index_addr(ptr, _index_value(index)), field_value)
                case RuntimeVal():
                    if elem.is_zst():
                        # a zero-sized aggregate has no storage to write into
                        return
                    if isinstance(value, RuntimeVal):
                        # a runtime aggregate value is written as a whole
                        self._emit(mir.Store(ptr.value, self._to_runtime(self._coerce(value, elem))))
                    else:
                        aggregate = _as_aggregate(value)
                        if aggregate is None or (aggregate.type != elem and not elem.is_zst()):
                            raise CompileError(f'cannot store {value!r} into {elem}')
                        self._emit(mir.Store(ptr.value, self._aggregate_to_runtime(aggregate)))
                case _:
                    raise CompileError(f'a compile-time box cannot hold the aggregate {elem}')
            return

        if isinstance(elem, sval.OptionType):
            # an option (and the ``T``/``Null`` a store delivers) is coerced to
            # its value form first, then written through the storage of ``elem``
            # (see ``ComptimeOption``)
            coerced = self._coerce(value, elem)
            match ptr:
                case ComptimeOptionPtr():
                    option = self._as_comptime_option(coerced, elem)
                    ptr.is_null = option.is_null
                    if _to_comptime(option.is_null) is not True:
                        # present (or unknown): write the payload value into its place
                        self.store(ptr.payload_ptr, option.value)
                case RuntimeVal():
                    self._emit(mir.Store(ptr.value, self._to_runtime(coerced)))
                case _:
                    raise CompileError(f'a compile-time box cannot hold the option {elem}')
            return

        if isinstance(elem, sval.TaggedUnionType):
            # a tagged union (and the variant value a store delivers) is coerced
            # to its value form first, then written through the storage of
            # ``elem`` (see ``ComptimeTaggedUnionValue``)
            coerced = self._coerce(value, elem)
            match ptr:
                case ComptimeTaggedUnionPtr():
                    if not isinstance(coerced, ComptimeTaggedUnionValue):
                        assert isinstance(coerced, RuntimeVal)
                        coerced = self._runtime_union_to_value_form(coerced, elem)
                    self._store_comptime_tagged_union(ptr, elem, coerced)
                case RuntimeVal():
                    self._emit(mir.Store(ptr.value, self._to_runtime(coerced)))
                case _:
                    raise CompileError(f'a compile-time box cannot hold the tagged union {elem}')
            return

        unit = elem.get_unit_value()
        match ptr:
            case ComptimeBox():
                if ptr.is_const:
                    raise CompileError(
                        f'cannot store through the const pointer {ptr.type}'
                    )
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

    def _extract_aggregate_value(self, value: InterpVal, index: int) -> InterpVal:
        """The ``index``-th field (or element) of the aggregate ``value``, in
        declaration (element) order: the place it holds for a compile-time
        aggregate, and the value read out of a *runtime* one with
        ``mir.ExtractValue`` - a zero-sized field holds its unit value, and the
        mirror of a struct of one stored field *is* that field (see
        ``StructType.mirror_is_a_field``), so it is the value itself.  A value
        that is not an aggregate at all (an ``Option``, say) is rejected."""
        value = _shallow_normalize(value)
        if isinstance(value, ComptimeAggregate):
            return value.values[index]
        type = _type_of(value)
        if not isinstance(type, (sval.StructType, sval.ArrayType)):
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
        if isinstance(type, sval.ArrayType):
            mir_index = index
        elif type.mirror_is_a_field(self._mir_cache):
            return RuntimeVal(value.value, field_type)
        else:
            mir_index = type.get_field_mir_indices(self._mir_cache)[index]
            assert mir_index is not None, 'a field with storage has a mirror position'
        return RuntimeVal(self._emit(mir.ExtractValue(value.value, mir_index)), field_type)

    def as_comptime_aggregate(self, ev: InterpVal) -> ComptimeAggregate:
        """The ``ComptimeAggregate`` form of the aggregate value ``ev`` (whose
        type it is read off): a compile-time aggregate is returned as it is, and
        a runtime one has every field (or element) read out of the value itself
        with ``_extract_aggregate_value`` (a struct and an array alike).  A
        value that is not an aggregate (an ``Option``, say) is rejected."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, ComptimeAggregate):
            return ev
        type = _type_of(ev)
        if not isinstance(type, (sval.StructType, sval.ArrayType)):
            raise CompileError(f'{ev!r} is not an aggregate')
        return ComptimeAggregate(
            type,
            tuple(
                self._extract_aggregate_value(ev, index)
                for index in range(sval.aggregate_type_length(type))
            ),
        )

    def _runtime_aggregate_field_values(self, value: InterpVal) -> list[InterpVal]:
        """The value of every field (or element) of the *runtime* aggregate
        ``value``, read out of the value itself (see
        ``as_comptime_aggregate``)."""
        return list(self.as_comptime_aggregate(value).values)

    def _split_runtime_aggregate(self, value: InterpVal) -> tuple[InterpVal, ...]:
        """One fresh place per field (or element) of the *runtime* aggregate
        ``value``, written with the field read out of it - the places a
        compile-time aggregate holds (see ``ComptimeAggregatePtr``).  A nested
        aggregate field is split the same way, recursively (its own place is a
        ``FULL`` slot holding a runtime aggregate, see ``store``)."""
        places: list[InterpVal] = []
        for field_value in self._runtime_aggregate_field_values(value):
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

    def _not_bool(self, value: InterpVal) -> mir.Value:
        """The negation of a boolean value as a MIR ``bool``: a compile-time one
        folds, a runtime one becomes a comparison against ``false``."""
        obj = _to_comptime(value)
        if isinstance(obj, bool):
            return mir.BoolValue(not obj)
        return self._emit(mir.Cmp('==', self._to_runtime(value), mir.BoolValue(False)))

    def _as_comptime_option(self, ev: InterpVal, option: sval.OptionType) -> ComptimeOption:
        """The value form of a value that already is of the option type
        ``option``: a compile-time option as it is, a runtime one read into the
        tag and the payload it holds (see ``_is_null``/``_option_payload``)."""
        if isinstance(ev, ComptimeOption):
            return ev
        assert isinstance(ev, RuntimeVal) and _type_of(ev) == option
        return ComptimeOption(self._is_null(ev), self._option_payload(ev, option))

    def _coerce_option_value(self, ev: InterpVal, option: sval.OptionType) -> InterpVal:
        """Materialize ``ev`` as a value form of the option type ``option``: a
        value that already is one passes through, the null value becomes the
        absent one and anything else a present value of the child type.

        The result is a :class:`ComptimeOption` whenever ``ev`` had to change -
        its tag a bool value and its payload the child value - so that no runtime
        representation is built here (see ``_option_parts_to_runtime``); a value
        that already is an option of the type is returned as it is."""
        ev = _shallow_normalize(ev)
        if _type_of(ev) == option:
            return ev
        child = option.child
        is_null = self._is_null(ev)
        if _to_comptime(is_null) is True:
            # absent: the payload is discarded, and the child type is only named
            # by the placeholder value the form carries
            return ComptimeOption(is_null, ComptimeVal(sval.Undefined(child)))
        payload = ev.value if isinstance(ev, ComptimeOption) else ev
        return ComptimeOption(is_null, self._coerce(payload, child))

    # -- tagged unions -------------------------------------------------------

    def _tagged_union_shape(self, type: sval.TaggedUnionType) -> str:
        """How the tagged union ``type`` is represented: ``'single'`` (a single
        variant, represented as the variant itself), ``'tag_only'`` (every
        variant is zero-sized, so only the tag is stored) or ``'tag_payload'``
        (a struct of the tag and the payload union)."""
        if len(type.types) == 1:
            return 'single'
        if type.payload_type().to_mir_type(self._mir_cache) is None:
            return 'tag_only'
        return 'tag_payload'

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

    def init_comptime_tagged_union(self, type: sval.TaggedUnionType) -> ComptimeTaggedUnionPtr:
        """Fresh compile-time storage for a tagged union, at variant 0 (see
        ``ComptimeTaggedUnionPtr``)."""
        variant = type.types[0]
        payload = self._fresh_place(variant, ComptimeVal(sval.Undefined(variant)))
        return ComptimeTaggedUnionPtr(type, ComptimeVal(sval.Int(0, type.tag_type())), payload)

    def _tagged_union_tag(self, ev: InterpVal) -> InterpVal:
        """The tag of the tagged union value ``ev``, as an interpreter int: the
        compile-time tag of a compile-time value, 0 for a single-variant union,
        the value itself when the payload is zero-sized (the union *is* the tag)
        and the first field of the struct otherwise."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, ComptimeTaggedUnionValue):
            return ev.tag
        type = _type_of(ev)
        assert isinstance(type, sval.TaggedUnionType)
        shape = self._tagged_union_shape(type)
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
            return ComptimeVal(sval.Undefined(sval.PointerType(variant_type, is_const=ptr_type.is_const)))
        if isinstance(place, ComptimeTaggedUnionPtr):
            if _to_comptime(place.tag) is None:
                # a tag only known at runtime: the payload place points at the
                # payload union storage, whose variant is read/written through
                # by reinterpreting its address (see ``_materialize_union``)
                return self._union_variant_ptr(place.payload_ptr, variant_type)
            payload_type = _type_of(place.payload_ptr)
            if isinstance(payload_type, sval.PointerType) and payload_type.elem == variant_type:
                return place.payload_ptr
            # the current variant differs (a payload is only read once the tag
            # matched, so such a place is never written through)
            return self._fresh_place(variant_type, ComptimeVal(sval.Undefined(variant_type)))
        if self._tagged_union_shape(type) == 'single':
            return RuntimeVal(self._to_runtime(place), sval.PointerType(variant_type, is_const=ptr_type.is_const))
        return self._union_variant_ptr(self._tagged_union_field_ptr(place), variant_type)

    def _write_tagged_union_tag(self, ptr: InterpVal, type: sval.TaggedUnionType, index: int) -> InterpVal:
        """Set the tag of the tagged union place ``ptr`` to the variant
        ``index`` (a compile-time storage also gets the payload place of that
        variant) and return the place to write the payload through."""
        place = _shallow_normalize(ptr)
        variant = type.types[index]
        if isinstance(place, ComptimeTaggedUnionPtr):
            place.tag = ComptimeVal(sval.Int(index, type.tag_type()))
            if variant.get_unit_value() is None:
                payload_type = _type_of(place.payload_ptr)
                if not (isinstance(payload_type, sval.PointerType) and payload_type.elem == variant):
                    place.payload_ptr = self._fresh_place(variant, ComptimeVal(sval.Undefined(variant)))
            return place
        assert isinstance(place, RuntimeVal)
        shape = self._tagged_union_shape(type)
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

    def _store_comptime_tagged_union(self, ptr: ComptimeTaggedUnionPtr, type: sval.TaggedUnionType, value: ComptimeTaggedUnionValue) -> None:
        """Write the tagged union ``value`` into the compile-time storage ``ptr``:
        the tag becomes the value's tag and the payload place holds the payload -
        the variant's own place, rebuilt when the variant changes, or the payload
        union storage in memory when the tag is only known at runtime."""
        tag = value.tag
        if _to_comptime(tag) is None:
            # a tag only known at runtime: the payload is the payload union
            # storage, which a variant is read/written through by reinterpreting
            # its address (see ``_tagged_union_payload_ptr``)
            ptr.tag = tag
            if type.payload_type().get_unit_value() is not None:
                # the payload holds no storage (every variant is zero-sized)
                ptr.payload_ptr = ComptimeVal(sval.Undefined(sval.PointerType(type.payload_type())))
            else:
                ptr.payload_ptr = self._materialize_union(value.value, type.payload_type())
            return
        index = self._tagged_union_int(tag)
        variant = type.types[index]
        ptr.tag = ComptimeVal(sval.Int(index, type.tag_type()))
        if variant.get_unit_value() is not None:
            ptr.payload_ptr = ComptimeVal(sval.Undefined(sval.PointerType(variant)))
            return
        payload_type = _type_of(ptr.payload_ptr)
        if not (isinstance(payload_type, sval.PointerType) and payload_type.elem == variant):
            ptr.payload_ptr = self._fresh_place(variant, ComptimeVal(sval.Undefined(variant)))
        self.store(ptr.payload_ptr, value.value)

    def _materialize_union(self, value: InterpVal, union_type: sval.UnionType) -> InterpVal:
        """The runtime pointer of fresh memory the union storage ``value`` is
        written into: a union has no compile-time place an address could be taken
        of, so a value of it whose variant has to be addressed is materialized
        into memory (see ``_tagged_union_payload_ptr``)."""
        slot = self.alloca(InlineMode.NONE)
        self._commit_pending_slot(slot, union_type)
        ptr = _shallow_normalize(slot)
        if isinstance(value, ComptimeVal) and isinstance(value.obj, sval.Undefined):
            mir_type = union_type.to_mir_type(self._mir_cache)
            assert mir_type is not None and isinstance(ptr, RuntimeVal)
            self._emit(mir.Store(ptr.value, mir.UndefValue(mir_type)))
        else:
            self.store(ptr, value)
        return ptr

    def _runtime_union_to_value_form(self, ev: RuntimeVal, type: sval.TaggedUnionType) -> ComptimeTaggedUnionValue:
        """The value form of the runtime tagged union ``ev`` (of the type
        ``type``): its tag and its payload - the variant value for a
        single-variant union, the payload union storage otherwise (see
        ``ComptimeTaggedUnionValue``)."""
        shape = self._tagged_union_shape(type)
        if shape == 'single':
            return ComptimeTaggedUnionValue(
                type, ComptimeVal(sval.Int(0, type.tag_type())),
                RuntimeVal(ev.value, type.types[0]),
            )
        tag = self._tagged_union_tag(ev)
        if shape == 'tag_only':
            unit = type.payload_type().get_unit_value()
            assert unit is not None
            return ComptimeTaggedUnionValue(type, tag, ComptimeVal(unit))
        return ComptimeTaggedUnionValue(
            type, tag, RuntimeVal(self._emit(mir.ExtractValue(ev.value, 1)), type.payload_type()),
        )

    def _tagged_union_parts_to_runtime(self, type: sval.TaggedUnionType, tag: InterpVal, value: InterpVal) -> mir.Value:
        """Build the runtime value of a tagged union from its tag ``tag`` (which
        may be a runtime value) and its payload ``value``: a compile-time tag
        makes the payload a variant value, joined into the payload union with an
        ``AsUnion``, while a runtime tag means the payload already *is* the
        payload union storage (see ``ComptimeTaggedUnionValue``).  A zero-sized
        payload keeps the tag alone, and a single-variant union is the variant
        itself."""
        shape = self._tagged_union_shape(type)
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
            if isinstance(value, ComptimeVal) and isinstance(value.obj, sval.Undefined):
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

    def _runtime_union_value_form(self, ev: RuntimeVal, from_type: sval.TaggedUnionType, to_type: sval.TaggedUnionType) -> ComptimeTaggedUnionValue:
        """The value form of the runtime tagged union ``ev`` (of the compatible
        union ``from_type``) converted to ``to_type``: the tag remapped, and the
        payload the variant value when the tag is compile-time (a single-variant
        union on either side) and the payload union storage otherwise (see
        ``ComptimeTaggedUnionValue``)."""
        shape = self._tagged_union_shape(to_type)
        from_shape = self._tagged_union_shape(from_type)
        if shape == 'single':
            # a subset of a single-variant union is that variant
            return ComptimeTaggedUnionValue(
                to_type, ComptimeVal(sval.Int(0, to_type.tag_type())),
                RuntimeVal(ev.value, to_type.types[0]),
            )
        if from_shape == 'single':
            variant = from_type.types[0]
            index = self._tagged_union_index(to_type, variant)
            return ComptimeTaggedUnionValue(
                to_type, ComptimeVal(sval.Int(index, to_type.tag_type())),
                RuntimeVal(ev.value, variant),
            )
        tag = self._tagged_union_reindex(self._tagged_union_tag(ev), from_type, to_type)
        if shape == 'tag_only':
            # the payload holds no storage: its value is the union's unit
            unit = to_type.payload_type().get_unit_value()
            assert unit is not None
            return ComptimeTaggedUnionValue(to_type, tag, ComptimeVal(unit))
        if from_shape == 'tag_payload':
            # the payload union storage reinterpreted as the target's
            payload = self._convert_union_storage(
                RuntimeVal(self._emit(mir.ExtractValue(ev.value, 1)), from_type.payload_type()),
                to_type.payload_type(),
            )
            return ComptimeTaggedUnionValue(to_type, tag, payload)
        # the source variants are all zero-sized: the active payload has no
        # storage, so the target's payload storage is undefined
        return ComptimeTaggedUnionValue(to_type, tag, ComptimeVal(sval.Undefined(to_type.payload_type())))

    def _convert_union_storage(self, value: InterpVal, to_union: sval.UnionType) -> InterpVal:
        """The union storage value ``value`` as a value of the union
        ``to_union``: the two share their storage (every variant lives at offset
        0), so the value is reinterpreted - a union without storage is the
        target's unit value, an undefined (or storage-less) one stays undefined,
        and any other runtime value is cast with ``mir.UnionCast``."""
        unit = to_union.get_unit_value()
        if unit is not None:
            return ComptimeVal(unit)
        if isinstance(value, ComptimeVal) and isinstance(value.obj, (sval.Undefined, sval.UnionValue)):
            return ComptimeVal(sval.Undefined(to_union))
        assert isinstance(value, RuntimeVal)
        if value.type == to_union:
            return value
        to_mir = to_union.to_mir_type(self._mir_cache)
        assert to_mir is not None
        return RuntimeVal(self._emit(mir.UnionCast(value.value, to_mir)), to_union)

    def _coerce_tagged_union_value(self, ev: InterpVal, type: sval.TaggedUnionType) -> InterpVal:
        """Materialize ``ev`` as a value form of the tagged union ``type``: a
        value of a compatible union is converted, anything else is the variant it
        belongs to, tagged.

        The result is a :class:`ComptimeTaggedUnionValue` whenever ``ev`` had to
        change - its tag the variant position (a compile-time value) and its
        value the variant, or - for a tag only known at runtime - the payload
        union storage (see ``ComptimeTaggedUnionValue``); a value that already is
        a union of the type is returned as it is."""
        ev = _shallow_normalize(ev)
        if _type_of(ev) == type:
            return ev
        if isinstance(ev, ComptimeTaggedUnionValue):
            if _to_comptime(ev.tag) is None:
                # a tag only known at runtime: the value already is the payload
                # union storage
                return ComptimeTaggedUnionValue(
                    type,
                    self._tagged_union_reindex(ev.tag, ev.type, type),
                    self._convert_union_storage(ev.value, type.payload_type()),
                )
            index = self._tagged_union_index(type, ev.type.types[self._tagged_union_int(ev.tag)])
            return ComptimeTaggedUnionValue(type, ComptimeVal(sval.Int(index, type.tag_type())), ev.value)
        ev_type = _type_of(ev)
        if isinstance(ev, RuntimeVal) and isinstance(ev_type, sval.TaggedUnionType):
            return self._runtime_union_value_form(ev, ev_type, type)
        index = self._tagged_union_variant_index(ev, type)
        variant_value = self._coerce(ev, type.types[index])
        return ComptimeTaggedUnionValue(type, ComptimeVal(sval.Int(index, type.tag_type())), variant_value)

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
                ptr = self._option_payload_ptr(ptr)
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
                ptr = self._option_payload_ptr(ptr)
                type = _type_of(ptr)
                assert isinstance(type, sval.PointerType)
                container_type = type.elem
                is_const = type.is_const

        if is_aggregate_init and isinstance(container_type, sval.TaggedUnionType):
            # a construction into a tagged union: which variant the storage holds
            # is only known from the struct being built (see ``finish_struct``),
            # so the field gets a pending place the closing ``FinishStruct`` binds
            # to the variant's field address
            return self.alloca(InlineMode.NONE)

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
            if (
                field_type.classify() == sval.SpecialTypeKind.DST
                and container_type.get_field_mir_indices(self._mir_cache)[index_int] is None
            ):
                # only the first dynamically-sized field (the struct's flexible
                # member) has a mirror position; every further one is
                # inaccessible (see ``sval.StructType.get_field_mir_indices``)
                raise CompileError(
                    f"only the first dynamically-sized field of {container_type} "
                    f"can be accessed"
                )
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
                elif isinstance(declared, sval.OptionType):
                    # a compile-time variable of an option type: the option has
                    # its own place form, a tag value and a payload place (see
                    # ``ComptimeOptionPtr``)
                    slot.committed = self.init_comptime_option(declared)
                elif isinstance(declared, sval.TaggedUnionType):
                    # a compile-time variable of a tagged union type: the union has
                    # its own place form, a tag value and a payload place (see
                    # ``ComptimeTaggedUnionPtr``)
                    slot.committed = self.init_comptime_tagged_union(declared)
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
        """Materialize a value of the spy type ``target``: a compile-time
        value is converted with ``sval.coerce_const``, a runtime value
        gets whatever numeric conversion the target needs - widening or
        narrowing, see ``_convert_inst``.  A committed slot is the address
        of the value it holds (``_shallow_normalize``), which is what an
        operation that takes a value without loading it (taking an address,
        ``ref``) hands over.

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
            # a tuple has no runtime representation to convert to: the
            # compile-time tuple itself is what a location of the type holds
            if not isinstance(ev, ComptimeTuple):
                raise CoerceError(f'cannot materialize a {target} from {ev!r}')
            return ev
        match ev:
            case ComptimeVal(obj) if isinstance(obj, sval.AggregateValue):
                # an aggregate held as one compile-time object (see
                # ``_as_aggregate``): it has no runtime representation of its
                # own, so it is materialized like an interpreter aggregate value
                aggregate = _as_aggregate(ev)
                assert aggregate is not None
                if not _is_aggregate(target):
                    raise CoerceError(f'cannot materialize a {target} from an aggregate')
                return self.load(self._materialize_aggregate(aggregate, target))
            case ComptimeVal(obj):
                return ComptimeVal(sval.coerce_const(obj, target))
            case RuntimeVal(value, type):
                return RuntimeVal(self._convert(value, type, target), target)
            case ComptimeAggregatePtr() | ComptimeOptionPtr() | ComptimeBox():
                # a compile-time aggregate as a value of a pointer type: it *is*
                # a pointer already (see ``_type_of``) - its fields are their own
                # places - so nothing is converted here.  Becoming an address of
                # real memory happens only where a MIR value is actually needed
                # (see ``_to_runtime``)
                # TODO: type check?
                if not isinstance(target, sval.PointerType):
                    raise CoerceError(
                        f'cannot materialize a {target} from a compile-time aggregate'
                    )
                return ev
            case ComptimeAggregate():
                # an aggregate has no runtime representation to convert to: it
                # is materialized into a temporary and read back as a runtime
                # value
                if not _is_aggregate(target):
                    raise CoerceError(f'cannot materialize a {target} from an aggregate')
                return self.load(self._materialize_aggregate(ev, target))
            case ComptimeCastedPtr():
                # a pointer reinterpreted by ``ptr_cast``: reading or writing
                # through it is not supported yet (see ``ComptimeCastedPtr``)
                raise CoerceError(
                    f'cannot materialize a {target} from the cast pointer {ev.type}'
                )
            case _:
                raise CoerceError('cannot materialize this value')

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
        materialized into.  An aggregate has no MIR representation of its own:
        the *place* of one (a ``ComptimeAggregatePtr``) yields the address of
        fresh memory it is copied into - what a pointer to it delivers - and an
        aggregate *value* (a ``ComptimeAggregate``) the value read back out of
        that memory, which is what a by-value use needs.  An uncommitted slot,
        a compile-time box and the aggregate of a zero-sized type (which has no
        runtime value at all) are rejected."""
        ev = _shallow_normalize(ev)
        match ev:
            case RuntimeVal():
                return ev.value
            case ComptimeVal():
                return _sval_to_runtime(ev.obj, self._mir_cache)
            case ComptimeAggregatePtr(aggregate_type, _):
                aggregate = self.load(ev)
                if not isinstance(aggregate, ComptimeAggregate):
                    # a zero-sized aggregate: its value *is* its unit value, which
                    # has no runtime representation at all
                    raise CompileError(
                        f'a value of the zero-sized {aggregate_type} has no runtime value'
                    )
                return self._to_runtime(self._materialize_aggregate(aggregate, aggregate_type))
            case ComptimeAggregate():
                return self._aggregate_to_runtime(ev)
            case ComptimeOption():
                option = _type_of(ev)
                assert isinstance(option, sval.OptionType)
                return self._option_parts_to_runtime(option, ev.is_null, ev.value)
            case ComptimeOptionPtr():
                # the compile-time *storage* of an option: what a value of its
                # pointer type delivers is the address of fresh memory the option
                # is written into (like a compile-time aggregate pointer), not
                # the option value itself
                ptr_type = _type_of(ev)
                assert isinstance(ptr_type, sval.PointerType) and isinstance(ptr_type.elem, sval.OptionType)
                option = ptr_type.elem
                mir_type = option.to_mir_type(self._mir_cache)
                assert mir_type is not None
                alloca = self._emit(mir.Alloca(mir_type))
                self._emit(mir.Store(alloca, self._to_runtime(self.load(ev))))
                return alloca
            case ComptimeTaggedUnionValue(type, tag, value):
                # the tagged union value form: the tag and the payload union are
                # joined into the struct representation (see
                # ``_tagged_union_parts_to_runtime``)
                return self._tagged_union_parts_to_runtime(type, tag, value)
            case ComptimeTaggedUnionPtr(type):
                # the compile-time *storage* of a tagged union: what a value of
                # its pointer type delivers is the address of fresh memory the
                # union is written into (like an option's storage)
                mir_type = type.to_mir_type(self._mir_cache)
                assert mir_type is not None
                alloca = self._emit(mir.Alloca(mir_type))
                self._emit(mir.Store(alloca, self._to_runtime(self.load(ev))))
                return alloca
            case ComptimeBox():
                raise CompileError('cannot use a compile-time box as a runtime value')
            case PendingSlot():
                raise CompileError('cannot use an uncommitted slot as a runtime value')
        raise CompileError('cannot return this value')

    def _aggregate_to_runtime(self, ev: ComptimeAggregate) -> mir.Value:
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
                        return self._to_runtime(ev.values[index])
                raise CompileError(f'{aggregate_type} has no stored field')
            result: mir.Value = mir.UndefValue(mir_type)
            for index, mir_index in enumerate(indices):
                if mir_index is None:
                    continue
                result = self._emit(mir.InsertValue(result, self._to_runtime(ev.values[index]), mir_index))
            return result
        # an array: the elements sit in element order
        result = mir.UndefValue(mir_type)
        for index, value in enumerate(ev.values):
            result = self._emit(mir.InsertValue(result, self._to_runtime(value), index))
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
        wrapped in a :class:`ComptimeCastedPtr` otherwise; recasting one of
        those drops the wrapper when the target converts the place it wraps."""
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
        if isinstance(ev, ComptimeCastedPtr):
            # a pointer already reinterpreted once: when the new target converts
            # the place it wraps, the place itself is the value (the wrapper is
            # dropped); otherwise the place is wrapped with the new type
            if self._try_coerce(ev.place, target_obj) is not None:
                return ev.place
            return ComptimeCastedPtr(ev.place, target_obj)
        coerced = self._try_coerce(ev, target_obj)
        if coerced is not None:
            return coerced
        return ComptimeCastedPtr(ev, target_obj)

    def exec_as_func_ptr(self, inst: hir.AsFuncPtr) -> PollResult:
        """``syntax.as_func_ptr(T, f)``: the runtime function pointer to the spy
        function ``f``, typed as ``ConstPtr[T]``.  The pointer is the address of
        the specialization of ``f`` for the function type ``T``, so the callee
        is compiled (if it is not already) and the instruction resumes with its
        ``mir`` value - exactly like a call, except that no call is emitted (see
        ``_request_function``).  A declared external function is already a
        pointer and is taken as it is."""
        fn_type = self.type_operand(inst.type, 'the type of a function pointer')
        if not isinstance(fn_type, sval.FunctionType):
            raise CompileError(f'as_func_ptr needs a function type, got {fn_type!r}')
        obj = self._operand_comptime_value(inst.obj)
        regs = self._frames[-1].regs
        if isinstance(obj, sval.DeclareFunction):
            # a declared external function already is a function pointer
            if obj.type != fn_type:
                raise CompileError(
                    f'as_func_ptr: {obj.linkname!r} has type {obj.type}, not {fn_type}'
                )
            regs[inst] = ComptimeVal(obj)
            return PollResult.AGAIN
        if not isinstance(obj, FunctionValue):
            raise CompileError(f'as_func_ptr expects a spy function, got {obj!r}')
        if obj.force_inline:
            raise CompileError(
                'cannot take the function pointer of an inlined Python function'
            )
        declared = obj.get_type()
        if declared != fn_type:
            raise CompileError(
                f'as_func_ptr: function {obj.hir.name} has type {declared}, not {fn_type}'
            )
        sig = obj.hir.signature
        provided = ArgList(
            tuple(arg.type for arg in sig.positional.by_id), (), frozendict(),
        )
        call_sig, partial_ret_sig = sig.specialize(provided, self._mir_cache)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            regs[inst] = RuntimeVal(fn_mir, sval.PointerType(fn_type, is_const=True))
            return PollResult.AGAIN

        return self._request_function(obj, call_sig, partial_ret_sig, _resumer)

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
            self.store(ret, ComptimeVal(_comptime_py_op(op, lv.obj, rv.obj)))
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
        """Compute ``lv op rv`` as ``target``: a compile-time pair folds in
        Python (and the result is re-tagged as ``target``), a runtime one is a
        ``mir.Arith`` over operands coerced to ``target``."""
        if _is_comptime_val(lv) and _is_comptime_val(rv):
            lobj = _to_comptime(lv)
            robj = _to_comptime(rv)
            assert lobj is not None and robj is not None
            obj = _comptime_py_op(op, lobj, robj)
            self.store(ret, ComptimeVal(sval.coerce_const(obj, target)))
            return
        lc = self._coerce(lv, target)
        rc = self._coerce(rv, target)
        mir_type = target.to_mir_type(self._mir_cache)
        assert mir_type is not None and not target.is_zst()
        value = self._emit(mir.Arith(op, self._to_runtime(lc), self._to_runtime(rc), mir_type))
        self.store(ret, RuntimeVal(value, target))

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
        slot = self.alloca(InlineMode.NON_AGGREGATE)
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
        slot = self.alloca(InlineMode.NON_AGGREGATE)
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

    def binary_assign(self, op: BinaryOp, left: InterpVal, right: ArgEntry[InterpVal]) -> PollResult:
        # ``x op= y`` calls the target's in-place magic method when its struct
        # declares one (``__iadd__``, ...); otherwise it is ``x = x op y``
        left_type = _place_type(left)
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
        # a runtime function pointer: a value of a pointer-to-function type,
        # called through the pointer the value carries (a function type alone
        # is dynamically sized, so only a pointer to one is a value)
        callee_type = _type_of(callee)
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
        if fn.name == 'type_info':
            if len(args.positional) != 1 or len(args.kwargs) > 0:
                raise CompileError('spy.type_info takes exactly one argument')
            obj = _to_comptime(_shallow_normalize(self._arg_value(args.positional[0])))
            if not isinstance(obj, sval.Type):
                raise CompileError(f'spy.type_info takes a type, got {obj!r}')
            self.store(ret, self._build_type_info(obj))
            return PollResult.AGAIN
        if fn.name == 'undefined':
            # ``std.core.undefined``: the undefined literal, the value of any
            # type (see ``sval.UndefinedType``); the store into the result
            # location coerces it to that location's type
            if len(args.positional) != 0 or len(args.kwargs) > 0:
                raise CompileError('std.core.undefined takes no arguments')
            self.store(ret, ComptimeVal(sval.UntypedUndefined()))
            return PollResult.AGAIN
        raise CompileError(f"cannot call the spy builtin {fn.name} inside a spy function")

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
        fields: tuple[InterpVal, ...]
        if isinstance(ty, sval.IntType):
            variant = 'IntType'
            fields = (ComptimeVal(ty.bits), ComptimeVal(ty.signed))
        elif isinstance(ty, sval.PointerType):
            variant = 'PointerType'
            fields = (ComptimeVal(ty.elem), ComptimeVal(ty.is_const))
        elif isinstance(ty, sval.ArrayType):
            variant = 'ArrayType'
            fields = (ComptimeVal(ty.elem), self._size_option(ty.length_int))
        elif isinstance(ty, sval.OptionType):
            variant = 'OptionType'
            fields = (ComptimeVal(ty.child),)
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
        else:
            raise CompileError(f'cannot reflect the type {ty}')
        union = self._reflect_type_info()
        struct_type = self._reflect_struct(variant)
        index = union.variant_index(struct_type)
        assert index is not None, f'{struct_type} is not a TypeInfo variant'
        return ComptimeTaggedUnionValue(
            union,
            ComptimeVal(sval.Int(index, union.tag_type())),
            ComptimeAggregate(struct_type, fields),
        )

    def _build_struct_field(self, field: sval.StructField) -> InterpVal:
        """The ``std.reflect.StructField`` value describing one field of a
        struct: its name as a ``ConstSlicePtr[u8]``, its type and its default
        value (absent when the field declares none)."""
        default: InterpVal
        if field.default is None:
            default = ComptimeOption(
                ComptimeVal(True), ComptimeVal(sval.Undefined(sval.AnyType()))
            )
        else:
            default = ComptimeOption(
                ComptimeVal(False),
                self._coerce(ComptimeVal(field.default), sval.AnyType()),
            )
        return ComptimeAggregate(self._reflect_struct('StructField'), (
            self._str_const_slice(field.name),
            ComptimeVal(field.type),
            default,
        ))

    def _head_option(self, ty: sval.StructType) -> InterpVal:
        """The ``Option[Any]`` a reflected struct's ``head`` is: the template
        head (:class:`sval.StructTypeHead`) of a struct that declares generic
        parameters, the absent option of a non-generic one."""
        if len(ty.head.generic_args) == 0:
            return ComptimeOption(
                ComptimeVal(True), ComptimeVal(sval.Undefined(sval.AnyType()))
            )
        return ComptimeOption(
            ComptimeVal(False), self._coerce(ComptimeVal(ty.head), sval.AnyType())
        )

    def _size_option(self, length: int | None) -> InterpVal:
        """The ``Option[int]`` a reflected array size is: an unsized array
        (``None``) as the absent option, a length as the present one."""
        if length is None:
            return ComptimeOption(
                ComptimeVal(True), ComptimeVal(sval.Undefined(sval.AnyIntType()))
            )
        return ComptimeOption(ComptimeVal(False), ComptimeVal(length))

    def _type_slice(self, types: tuple[sval.Type, ...]) -> InterpVal:
        """A ``ConstSlicePtr[type]`` of the compile-time types ``types``."""
        return self._const_slice(sval.TYPE_TYPE, tuple(ComptimeVal(type) for type in types))

    def _const_slice(self, elem: sval.Type, values: tuple[InterpVal, ...]) -> InterpVal:
        """A ``ConstSlicePtr[elem]`` compile-time value over ``values``: the
        values in a fresh array place of their own, the (const) pointer to it and
        the number of elements - the value ``slice_ptr`` builds for a slice of
        compile-time storage (see ``ComptimeAggregate``)."""
        array = self.init_inline_aggregate(sval.ArrayType(elem, len(values)))
        for index, value in enumerate(values):
            self.store(self.field_index_addr(array, _index_value(index)), value)
        slice_type = self._special_type.slice_ptr_of(elem, True)
        return ComptimeAggregate(slice_type, (
            array,
            ComptimeVal(sval.Int(len(values), self._usize_type())),
        ))

    def _str_const_slice(self, text: str) -> InterpVal:
        """A ``ConstSlicePtr[u8]`` compile-time value over the bytes of ``text``,
        backed by a :class:`sval.StrConstPtr` rather than by a box per byte."""
        elem = sval.IntType(8, False)
        data = text.encode()
        slice_type = self._special_type.slice_ptr_of(elem, True)
        return ComptimeAggregate(slice_type, (
            ComptimeVal(sval.StrConstPtr(data, 0)),
            ComptimeVal(sval.Int(len(data), self._usize_type())),
        ))

    def _str_constant_byte(self, ptr: sval.StrConstPtr, pos: int, what: str) -> int:
        """The byte at ``pos`` of the string constant ``ptr``; the constant is
        read-only and its positions unsigned, so a position outside the data is
        an error."""
        if not 0 <= pos < len(ptr.data):
            raise CompileError(f'the string constant is out of bounds ({what})')
        return ptr.data[pos]

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
        call_sig, partial_ret_sig = sig.specialize(binded_args.map(_arg_type_of), self._mir_cache)
        ret_sig = partial_ret_sig.complete()
        res = self._make_runtime_call(
            self._to_runtime(callee), binded_args, ret, call_sig, ret_sig,
        )
        if on_return is not None and not ret_sig.value_is_empty():
            on_return()
        return res

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
            elif isinstance(type, sval.OptionType):
                # a compile-time option: the tag is a value and the payload a
                # place of its own (see ``ComptimeOptionPtr``)
                val.committed = self.init_comptime_option(type)
            elif isinstance(type, sval.TaggedUnionType):
                # a compile-time tagged union: the tag is a value and the payload
                # a place of its own (see ``ComptimeTaggedUnionPtr``)
                val.committed = self.init_comptime_tagged_union(type)
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
            return ComptimeVal(sval.Undefined(sval.PointerType(child, type.is_const)))
        ptr = _shallow_normalize(ptr)
        if isinstance(ptr, ComptimeOptionPtr):
            if write_tag:
                ptr.is_null = ComptimeVal(False)
            return ptr.payload_ptr
        src = self._to_runtime(ptr)
        if sval.find_first_pointer_type_pos(child) is not None:
            return RuntimeVal(src, sval.PointerType(child, type.is_const))
        tag = self._emit(mir.Gep(src, 0))
        if write_tag:
            self._emit(mir.Store(tag, mir.BoolValue(True)))
        payload = self._emit(mir.Gep(src, 1))
        return RuntimeVal(payload, sval.PointerType(child, type.is_const))

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
        if isinstance(ev, ComptimeOption):
            # a compile-time option carries its own tag (see ``ComptimeOption``)
            return ev.is_null
        if isinstance(ev, ComptimeVal):
            return ComptimeVal(isinstance(ev.obj, (sval.Null, sval.TypedNull)))
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

        A compile-time present option (``ComptimeOption``) holds its payload,
        and any other compile-time value is the payload itself.  A runtime
        option value is read off its representation: a zero-sized child has no
        payload but its unit value, a child that still has a free pointer
        shares the option's representation (the value *is* the payload), and the
        ``(bool, T)`` representation holds the payload in its second field,
        which ``mir.ExtractValue`` reads out of the value.  An absent value has
        no payload and is rejected."""
        ev = _shallow_normalize(ev)
        child = option.child
        if isinstance(ev, ComptimeOption):
            if _to_comptime(ev.is_null) is True:
                raise CompileError(f'a present value is required here: {option} is absent')
            return ev.value
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
            self._frames[-1].regs[ret] = self._arg_value(value)
            return PollResult.AGAIN
        if isinstance(type, sval.StructType) and self._resolve_method(type, _BOOL_METHOD) is not None:
            slot = self.alloca(InlineMode.NON_AGGREGATE)
            regs = self._frames[-1].regs

            def on_return() -> None:
                self._commit_pending_slot(slot)
                regs[ret] = self.load(slot)

            return self.call_method(
                self._operand_place(value), _BOOL_METHOD, RawArgList((), frozendict()), slot, on_return,
            )

        raise CompileError(f'an if condition must be a bool value, got {type}')

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
        rather than a register (see ``_call_function_entry``)."""
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
            slot = self.alloca(InlineMode.FULL)
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

        if not (isinstance(base, ComptimeVal) and isinstance(base.obj, sval.ConstRef) and isinstance(base.obj.value, sval.StructTypeHead)):
            raise CompileError(
                f'cannot subscript {base!r}: a subscript is a place in an array, '
                f'an element of a pointer, or a specialization of a struct template'
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

    def _subscript_pointer(self, base: InterpVal, ptr_type: sval.PointerType, index: ArgEntry[InterpVal], ret: hir.Inst) -> PollResult:
        """The place a subscript of the pointer ``ptr_type`` - the value the
        place ``base`` holds - names: the ``SlicePtr`` of a slice, or the
        element of a multi pointer, an address ``p + i`` away (the element's own
        place when the pointer names compile-time storage, see
        ``ComptimeAggregatePtr``).  A single pointer has one place only, its
        pointee, and is dereferenced with ``p[...]``."""
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
        if isinstance(pointer, ComptimeVal) and isinstance(pointer.obj, sval.StrConstPtr):
            # a string constant: the element's own place, a read-only box holding
            # the byte at the compile-time index (a store through it is rejected,
            # the constant is const)
            index_value = self._coerce(index_ev, self._usize_type())
            index_int = _comptime_int(index_value)
            if index_int is None:
                raise CompileError('a string constant needs a constant index')
            byte = self._str_constant_byte(
                pointer.obj, pointer.obj.cursor + index_int, 'subscript'
            )
            self._frames[-1].regs[ret] = ComptimeBox(
                sval.IntType(8, False),
                ComptimeVal(sval.Int(byte, sval.IntType(8, False))),
                is_const=True,
            )
            return PollResult.AGAIN
        if isinstance(pointer, ComptimeAggregatePtr) and isinstance(pointer.type, sval.ArrayType):
            # compile-time storage of the elements: the n-th element's own place,
            # which a compile-time index picks - there is no runtime address to
            # take (see ``field_index_addr``)
            index_value = self._coerce(index_ev, self._usize_type())
            self._frames[-1].regs[ret] = self.field_index_addr(pointer, index_value)
            return PollResult.AGAIN
        if not isinstance(pointer, RuntimeVal):
            raise CompileError('cannot index a compile-time pointer')
        offset = self._to_runtime(self._coerce(index_ev, self._isize_type()))
        self._frames[-1].regs[ret] = RuntimeVal(
            self._emit(mir.Gep(pointer.value, offset)), ptr_type,
        )
        return PollResult.AGAIN

    def slice_object(self, inst: hir.Slice) -> ComptimeAggregate:
        """The ``std.slice`` object a slice subscript builds (see
        ``hir.Slice``): a compile-time aggregate of its bounds, of the type
        ``std.slice[usize]`` - a slice of a pointer counts its elements.  Every
        bound is an *option*: a bound the source left out (a null constant) is
        the *typed* absent value of ``Option[usize]`` (see ``sval.TypedNull``)
        and a present one an ``Option`` value (see ``ComptimeOption``): an option
        has no runtime shape of its own, so this is how a field is held.  Which
        meaning an absent bound has is decided where the slice is used (see
        ``slice_ptr``)."""
        usize = self._usize_type()
        start = self._slice_bound_field(self.operand(inst.lower), usize, 'the start of a slice')
        end = self._slice_bound_field(self.operand(inst.upper), usize, 'the end of a slice')
        step = self._slice_bound_field(self.operand(inst.step), usize, 'the step of a slice')
        return ComptimeAggregate(
            self._special_type.slice_type.specialize((usize,)), (start, end, step),
        )

    def _slice_bound_field(self, ev: InterpVal, usize: sval.Type, what: str) -> InterpVal:
        """One bound of the ``std.slice`` object a slice subscript builds, as an
        option value: the absent value when ``ev`` is a null constant, a present
        option holding it coerced to ``usize`` otherwise."""
        if _comptime_bool(self._is_null(ev), what):
            return ComptimeVal(sval.TypedNull(usize))
        return ComptimeOption(ComptimeVal(False), self._coerce(ev, usize))

    def _slice_bound_value(self, ev: InterpVal, what: str) -> InterpVal | None:
        """The value of a bound of the ``std.slice`` object a slice subscript
        built, or None when it is absent (see ``slice_object``)."""
        ev = _shallow_normalize(ev)
        if isinstance(ev, ComptimeOption):
            return None if _comptime_bool(ev.is_null, what) else ev.value
        if _comptime_bool(self._is_null(ev), what):
            return None
        return ev

    def slice_ptr(self, base: InterpVal, ptr_type: sval.PointerType, slice_obj: ComptimeAggregate) -> ComptimeAggregatePtr:
        """The ``SlicePtr`` a slice subscript of the multi pointer ``ptr_type``
        builds: ``SlicePtr(mptr + a, b - a)``.  The result is the place of a
        compile-time aggregate - a subscript always yields a pointer - whose
        fields are *const* boxes (a slice is a view, not a place to write the
        slice itself through, see ``ComptimeBox``): the pointer field holds the
        offset pointer (a runtime value when the base is a runtime one, the
        places the slice names when the base is compile-time storage), the
        length field the number of elements.

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
        elif isinstance(base_ev, ComptimeAggregatePtr) and isinstance(base_ev.type, sval.ArrayType):
            # compile-time storage of the elements: the slice names a range of
            # the places it holds (a runtime bound has no place to name)
            start = _comptime_int(start_ev)
            end = _comptime_int(end_ev)
            if start is None or end is None:
                raise CompileError(
                    'a slice of compile-time storage needs constant bounds'
                )
            self._check_slice_range(start, end)
            element_ptr = ComptimeAggregatePtr(
                sval.ArrayType(elem, end - start), base_ev.ptrs[start:end],
            )
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
        return ComptimeAggregatePtr(
            self._special_type.slice_ptr_of(elem, is_const),
            (
                ComptimeBox(
                    sval.PointerType(elem, is_const, sval.PointerVariant.MULTI),
                    element_ptr, is_const=True,
                ),
                ComptimeBox(usize, length, is_const=True),
            ),
        )


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
            unit = place_type.get_unit_value()
            initial = ComptimeVal(unit) if unit is not None else ComptimeVal(sval.Undefined(place_type))
            places.append(self._fresh_place(place_type, initial))
        return ComptimeAggregatePtr(type, tuple(places))

    def _fresh_place(self, type: sval.Type, initial: InterpVal) -> InterpVal:
        """A fresh compile-time place for a value of ``type``, used as the
        field/element/payload of a compile-time aggregate or option: a nested
        aggregate or option gets a pointer form of its own, anything else a
        box.  ``initial`` is the value the place holds before it is written
        (the type's unit value for a zero-sized one, undefined otherwise)."""
        if _is_aggregate(type):
            return self.init_inline_aggregate(type)
        if isinstance(type, sval.OptionType):
            return self.init_comptime_option(type)
        if isinstance(type, sval.TaggedUnionType):
            return self.init_comptime_tagged_union(type)
        return ComptimeBox(type, initial)

    def init_comptime_option(self, option: sval.OptionType) -> ComptimeOptionPtr:
        """Fresh compile-time storage for an option: the payload gets a place of
        its own and the tag starts absent (see ``ComptimeOptionPtr``)."""
        child = option.child
        if child.is_zst():
            payload: InterpVal = ComptimeVal(sval.Undefined(sval.PointerType(child, is_const=False)))
        else:
            payload = self._fresh_place(child, ComptimeVal(sval.Undefined(child)))
        return ComptimeOptionPtr(ComptimeVal(True), payload)

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

        dest_ptr_type = _type_of(_shallow_normalize(dest))
        if isinstance(dest_ptr_type, sval.PointerType) and isinstance(dest_ptr_type.elem, sval.TaggedUnionType):
            # the construction builds the struct straight into a tagged union: the
            # union takes the variant the struct is, and every field goes into
            # that variant's storage
            self._finish_struct_into_tagged_union(
                _shallow_normalize(dest), dest_ptr_type.elem, struct_type, provided, fields,
            )
            return

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
        generic_var_values = frozendict(zip(struct.head.generic_args, struct.generic_args))
        if len(generic_var_values) == 0:
            return resolved
        return sval.BoundMethod(resolved, generic_var_values)

    def _resolve_method(self, type: sval.Type, method_name: str) -> sval.AnyValue | None:
        match type:
            case sval.StructType():
                return self._method_of(type, method_name)
            case _:
                return None

    def call_method(self, ptr: InterpVal, method_name: str, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal, on_return: Callable[[], None] | None = None) -> PollResult:
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
            self_is_ref = (
                first.by_ref is TriState.TRUE
                or not isinstance(first.type, sval.PointerType)
            )

        return self.call(
            ComptimeVal(sval.ConstRef(method)),
            RawArgList((ArgEntry(ptr, self_is_ref),) + args.positional, args.kwargs),
            ret,
            on_return,
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
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
        on_return: Callable[[], None] | None = None,
    ) -> PollResult:
        """A call of a registered spy function with the given (already
        evaluated) argument values - the common tail of an ordinary
        function call and of a method call, whose ``self`` the caller
        prepended to the arguments.  The call is specialized from the
        marshaled argument types (an annotated parameter fixes its type,
        an unannotated one is typed by its argument); a plain Python
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
            # value, which its body must respect (see ``_current_value_is_empty``).
            # Its own type parameters are solved from the argument types here,
            # since the body may name them in a type expression
            # (``syntax.MultiPtr[T]``, see ``astgen``): the frame then resolves
            # them like the type arguments of a compiled call (see ``operand``)
            solved = sig.solve_param_types(binded_args.map(_arg_type_of))
            frame_values: dict[sval.TypeVar, sval.AnyValue] = dict(zip(sig.generic_args, solved))
            if generic_var_values:
                frame_values.update(generic_var_values)
            return self._start_inline(
                fn.hir.body, fn.hir.arg_is_ref, binded_args, ret,
                frozendict(frame_values),
                value_is_empty=isinstance(sig.ret_type, sval.EmptyType),
                on_done=on_return,
            )
        arg_types = binded_args.map(_arg_type_of)
        spec_sig = sig.specialize(arg_types, self._mir_cache)

        def _resumer(self0: Self, fn_mir: mir.Value, ret_sig: ReturnSignature) -> PollResult:
            res = self0._make_runtime_call(fn_mir, binded_args, ret, spec_sig[0], ret_sig)
            if on_return is not None and not ret_sig.value_is_empty():
                # the callee delivered its value into ``ret``: finish what the
                # caller had to do with it (see ``subscript``)
                on_return()
            return res

        return self._request_function(fn, spec_sig[0], spec_sig[1], _resumer, generic_var_values)

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

        spec = ret_sig.ret_spec(self._mir_cache)
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
            defer_blocks = self._collect_exit_defers(
                True, data, inclusive=True, current_frame_only=False,
            )
            incoming = self._union_variant_ptr(payload_place, exception)
            self._route_to_clause(data, clause, incoming, exception, defer_blocks)
        else:
            defer_blocks = self._collect_exit_defers(
                True, None, inclusive=False, current_frame_only=False,
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
        generic_var_values: frozendict[sval.TypeVar, sval.AnyValue] | None = None,
        value_is_empty: bool = False,
        on_done: Callable[[], None] | None = None,
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
        for (arg, by_ref) in zip(args.positional, arg_is_ref):
            # ``by_ref`` is the *callee's* HIR binding: a method's ``self`` is
            # bound directly to its argument (the address) even though the
            # caller passed it as a value; every other parameter is forwarded as
            # the caller passed it (by reference), while a by-value parameter the
            # caller passed by value is given a slot of its own (the copy a
            # value parameter is)
            if by_ref or arg.is_ref:
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
            body, value_is_empty=value_is_empty, on_done=on_done,
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
        places = _result_places(self._current_result_loc())
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
                elif isinstance(slot, ComptimeBox):
                    assert slot.value is not None
                    block.insts.append(mir.Ret(self._to_runtime(slot.value), defer_blocks))
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

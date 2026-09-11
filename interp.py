"""Compile-time interpretation of the HIR ("running" the HIR).

The interpreter executes the linear HIR instruction stream of a function
against the concrete argument types, emitting the typed MIR along the
way.  Every executed HIR instruction produces a value that is recorded in
a register table keyed by the instruction object itself, mirroring how
``symlat.jit.llvm`` registers work: operands of later instructions are
references to earlier instruction objects.

Values in the register table are either

* :class:`ComptimeVal` - a compile-time value of the ``spy`` domain
  (an ``sval.AnyValue``: a Python scalar, a spy type descriptor, a
  function to call/inline, ...).  "No value" is the unit value
  ``sval.Void()`` - the unique value of the zero-sized void type - never
  Python ``None``,
* :class:`RuntimeVal` - the object of an already emitted MIR
  instruction (a typed runtime value), or
* :class:`PendingSlot` - an executed ``Alloca`` whose typed MIR alloca
  is emitted by its first store (a slot whose content type is a
  zero-sized type never gets memory: it only records its unit value).

Instructions whose operands are all compile-time values are evaluated
eagerly in Python (the comptime semantics of the DSL); instructions
with runtime operands emit typed MIR.  A compile-time value flows into
runtime code only by being converted to a typed constant of the type
the runtime operation expects.  The interpreter types everything in the
``spy`` type system of ``type.py`` and *mirrors* the spy types into MIR
only when an instruction is emitted (``type.to_mir_type``): it never reads
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
arguments (resolved by ``hir.Arg`` leaves); the addressable parameter
slots themselves are the ``Alloca``/``Store`` prologue that ``astgen``
placed at the head of every function body.  The interpreter types an
``Alloca`` when its first store executes, so the untyped HIR needs no
type information of its own.
"""

import operator
import types as pytypes
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from . import hir, mir, sval
from .errors import CompileError
from .fn import (
    ArgEntry,
    ArgList,
    CompileBatch,
    FunctionInstance,
    FunctionResolver,
    FunctionValue,
    NativeFn,
    RawArgList,
    ReturnSignature,
    SpecializedCallSignature,
    SpecializedComptimeArg,
    SpecializedFormalArg,
    SpecializedRuntimeArg,
)
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

_ARITH_OPS = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'rem'}

_CMP_OPS = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}


class InterpVal:
    pass

@dataclass
class ComptimeVal(InterpVal):
    obj: sval.AnyValue


@dataclass
class RuntimeVal(InterpVal):
    """A value of the already emitted typed MIR.  The interpreter's own
    knowledge of the static type of the value lives here in the ``spy``
    type system (``type.py``) - the MIR type of the value is only ever
    *produced* from it (``type.to_mir_type``), never read back for a
decision."""

    value: mir.Value
    type: sval.Type


@dataclass
class ComptimeBox(InterpVal):
    """The materialization of a :class:`PendingSlot` whose stores are all
    compile-time: a writable pointer to a *compile-time* value.  Unlike a
    :class:`RuntimeVal` it owns no memory - ``value`` is the content
    itself (``None`` before the first commit)."""

    type: sval.Type
    value: InterpVal | None = None


@dataclass
class _MirInsertionBlock:
    """A list of MIR instructions to be spliced into the flat body at
    ``pos`` once the whole body has been generated (see
    ``HirRunner._splice_insertions``).  ``pos`` is an index into the body
    the interpreter emitted before splicing."""

    insts: list[mir.Inst]
    pos: int


@dataclass
class _PendingRetlocCall:
    """An RLS native call recorded by a :class:`PendingSlot` before the
    slot is materialized: the callee and the already emitted arguments.
    The hidden result pointer is only known (and appended) once the slot
    has an address."""

    callee: mir.Value
    prev_args: tuple[mir.Value, ...]


@dataclass
class _PendingStore:
    """One store point recorded by a :class:`PendingSlot`: the spy type of
    the stored value, whether it is compile-time, and how to deliver it
    once the slot's final type is known.  Exactly one of ``value`` (a
    plain store) and ``call`` (an RLS call writing into the slot) is set."""

    type: sval.Type
    is_comptime: bool
    insertion: _MirInsertionBlock
    value: InterpVal | None = None
    call: _PendingRetlocCall | None = None


@dataclass
class PendingSlot(InterpVal):
    """The value of an executed ``hir.Alloca`` before it is *committed*.
    In this phase a store (or an RLS call) into the slot only records a
    :class:`_PendingStore`; the slot acquires its final type (the pairwise
    ``resolve_peer_type`` of the store types) and its storage when
    ``hir.CommitSlot`` runs, which materializes it into a
    :class:`RuntimeVal` (a pointer to real memory) or a
    :class:`ComptimeBox` (see ``HirRunner._commit_pending_slot``).

    ``allow_inline`` marks the slots astgen allocates for an expression
    temporary (``Alloca(True)``): a temporary whose stores are all
    compile-time becomes a :class:`ComptimeBox` instead of memory."""

    mir_alloca_pos: int
    allow_inline: bool
    stores: list[_PendingStore] = field(default_factory=list)
    committed: InterpVal | None = None


@dataclass
class ComptimeTuple(InterpVal):
    value_ptrs: tuple[InterpVal, ...]

@dataclass
class ComptimeDict(InterpVal):
    value_ptrs: dict[str, InterpVal]

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
            case ComptimeBox():
                return True
            case ComptimeVal():
                return True
            case ComptimeTuple():
                todo.extend(val.value_ptrs)
            case ComptimeDict():
                todo.extend(val.value_ptrs.values())
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
    def __init__(self, arg_values: tuple[InterpVal, ...], ret_loc: InterpVal, insts: tuple[hir.Inst, ...]) -> None:
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
    """Materialize a value as a typed runtime value: runtime values must
    already have the target type, compile-time values adopt it (or,
    without a target, their Python type mapping).  Returns the typed MIR
    value and its spy type."""
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

def _arg_type_of(arg: ArgEntry[InterpVal]):
    """The spy type of the *value* an argument denotes: a reference
    argument carries the address of its value, so one pointer layer is
    stripped here."""
    type = _type_of(arg.value)
    if arg.is_ref:
        assert isinstance(type, sval.PointerType), f"pointer expected, got {type}"
        return type.elem
    return type

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

@dataclass
class ResumeInfo:
    args: ArgList[ArgEntry[InterpVal]]
    call_sig: SpecializedCallSignature
    ret_loc: InterpVal | None
    ret_reg: hir.Inst | None


class HirRunner:
    """Runs one function body (and everything it inlines) at compile
    time, filling the pre-created typed :class:`mir.Function` of one
    specialization.

    ``resolver`` is the compile-time host, typed as the
    :class:`FunctionResolver` interface it implements (``dsl._Context``
    in practice): it resolves a global object referenced inside a
    function body to its spy value (its function entry, or ``None`` for
    anything that is not a spy object).  The nested specializations a
    body calls are requested through ``Analyser._request_function``.
    """

    def __init__(self, analyser: Analyser, fn_instance: FunctionInstance) -> None:
        self._analyser = analyser
        # the frames of the function bodies under execution: the function
        # proper at the bottom, one frame per inlined plain function
        # above it (see ``_in_function_proper``; each frame carries the
        # IR of its body, see ``Frame``)
        self._frames: list[InlineFrame] = []
        # the function proper whose body is currently being typed (see
        # ``_bind_result_ptr``)
        self._fn_instance = fn_instance
        self._mir_insertion_blocks: list[_MirInsertionBlock] = []
        # the ``hir.Ret`` positions of the function proper whose return
        # convention is not fixed yet (an unannotated return type); their
        # ``mir.Ret`` is filled in by ``_finish_function`` once the result
        # location has been materialized
        self._deferred_returns: list[_MirInsertionBlock] = []
        self.return_sig: ReturnSignature | None = None

        self.resume_info: ResumeInfo | None = None

    # -- entry point ---------------------------------------------------------

    def run_function(
        self,
        body: tuple[hir.Inst, ...],
        sig: SpecializedCallSignature,
        ret_sig: ReturnSignature | None,
    ):
        # reset the per-specialization state; the result location of the
        # function proper is reserved first so its slot sits at a known
        # position in the body
        self.return_sig = None
        self.resume_info = None
        self._deferred_returns = []
        ret_loc = PendingSlot(self._reserve(), False)
        frame = InlineFrame((), ret_loc, body)
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
                type = node.type.to_mir_type()
                assert type is not None and not isinstance(type, mir.VoidType), f"void type has no runtime representation: {node.type}"
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

    def _init_args_from_signature(
        self,
        signature: SpecializedCallSignature,
        mir_args: list[mir.Type],
    ) -> tuple[InterpVal, ...]:
        arg_values: list[InterpVal] = []

        for arg in signature.positional:
            arg_values.append(self._init_one_arg(arg[1], mir_args))

        if signature.varargs:
            arg_values.append(ComptimeTuple(tuple(self._init_one_arg(a, mir_args) for a in signature.varargs)))
        if signature.kwargs:
            arg_values.append(ComptimeDict({k: self._init_one_arg(v, mir_args) for k, v in signature.kwargs.items()}))

        return tuple(arg_values)

    def _materialize_result_ptr(self, type: sval.Type, ret_by_ref: bool | None = None) -> None:
        """Fix the return convention of the function proper from the spy
        type its result location holds (its first return site, or its
        declared return annotation).  A result delivered through a result
        pointer appends the hidden result pointer formal to the lowered
        signature *after* every declared argument and makes the function
        return void; the location is then the memory of that pointer.  A
        direct return fixes the MIR return type and leaves the location
        recording its value.

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
                raise CompileError(f'type {type} has no runtime representation')
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
        proper ended (``DONE``) or an inlined/nested callee has to be
        typed first (``SUSPEND``).  There is no recursion: the walk is
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
        instruction only updates the register table."""

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
                    block = _MirInsertionBlock([], len(self._fn_instance.mir.insts))
                    self._mir_insertion_blocks.append(block)
                    self._deferred_returns.append(block)
                    return self._cut()
                location = self._current_result_loc()
                if self.return_sig.ret_by_ref or self.return_sig.ret_type.is_zst():
                    self._emit(mir.Ret(None))
                else:
                    self._emit(mir.Ret(_to_runtime(self.load(location))))
                return self._cut()
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
                regs[inst] = self.exec_field_name_addr(self.operand(inst.base), inst.name)
            case hir.CommitSlot():
                self._commit_pending_slot(self.operand(inst.slot))
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
        branches already continue at that shared continuation, and block
        scoping (a branch declaration never escapes its branch) keeps
        the state crossing the join in memory slots, where it needs no
        phi."""
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
                if not isinstance(obj, (int, float, str, bool, pytypes.NoneType)):
                    resolved = self._analyser._resolver.resolve_global(obj)
                    if resolved is not None:
                        return ComptimeVal(resolved)
                return ComptimeVal(sval.as_value(obj))
            case hir.ConstRef():
                # a reference to an immutable global.  At compile time a
                # reference to a global behaves exactly like the value it
                # refers to (its static type is a ``type.PointerType`` of
                # the referenced object - ``PointerType(typeof(expr),
                # True)`` - but nothing dereferences a compile-time
                # global at runtime yet, so the reference is only ever
                # consumed as an identity: the callee of a call).  The
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
        match ptr:
            case PendingSlot():
                if ptr.committed is None:
                    raise CompileError('cannot load from a slot before it is committed')
                return self.load(ptr.committed)
            case ComptimeBox():
                if ptr.value is None:
                    raise CompileError('cannot load from a compile-time box that was never written')
                return ptr.value
            case ComptimeVal(obj) if isinstance(obj, sval.ConstRef):
                # a reference to an immutable compile-time global behaves like
                # the value it refers to
                return ComptimeVal(obj.value)
        type = _type_of(ptr)
        if not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot load from a {type} value")
        unit_value = type.elem.get_unit_value()
        if unit_value is not None:
            return ComptimeVal(unit_value)
        match ptr:
            case RuntimeVal():
                return RuntimeVal(self._emit(mir.Load(ptr.value)), type.elem)
        raise CompileError('cannot load from a compile-time pointer')

    def store(self, ptr: InterpVal, value: InterpVal) -> None:
        """Write ``value`` into the slot or through the pointer ``ptr``.
        A store into a still uncommitted slot only records a store point
        (see :class:`PendingSlot`): the slot's final type is not known
        until it is committed, and the actual store is inserted then."""
        if isinstance(ptr, PendingSlot) and ptr.committed is None:
            value_type = _type_of(value)
            if value_type is None:
                raise CompileError('cannot store a value that has no spy type')
            insertion = _MirInsertionBlock([], len(self._fn_instance.mir.insts))
            self._mir_insertion_blocks.append(insertion)
            ptr.stores.append(
                _PendingStore(
                    type=value_type,
                    is_comptime=isinstance(value, ComptimeVal),
                    insertion=insertion,
                    value=value,
                )
            )
            return

        ptr = _shallow_normalize(ptr)
        match ptr:
            case ComptimeBox():
                # a compile-time writable pointer: the store is evaluated
                # on the spot
                ptr.value = value
                return
        ptr_type = _type_of(ptr)
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f"cannot store to a {ptr_type} value")
        elem = ptr_type.elem
        if elem.is_zst():
            return
        coerced = self._coerce(value, elem)
        match ptr:
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

    # -- struct values ---------------------------------------------------------

    def exec_field_name_addr(self, ptr: InterpVal, name: str) -> InterpVal:
        """Note: has auto deref  """
        ptr = self._auto_deref(_shallow_normalize(ptr))
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

        return self.field_index_addr(ptr, index)

    def field_index_addr(self, ptr: InterpVal, index: int) -> InterpVal:
        type = _type_of(_shallow_normalize(ptr))
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot take field address of {ptr}")
        container_type = type.elem
        if not isinstance(container_type, sval.StructType):
            raise CompileError(f"cannot take field address of {ptr}")

        field_type = container_type.fields[index].type
        mir_index = container_type.get_field_mir_indices()[index]
        if mir_index is None:
            return ComptimeVal(sval.Undefined(sval.PointerType(field_type, type.is_const)))

        match ptr:
            case ComptimeVal():
                raise CompileError(
                    'cannot take the address of a field of a compile-time value'
                )
            case RuntimeVal():
                return RuntimeVal(self._emit(mir.Gep(ptr.value, mir_index)), sval.PointerType(field_type, type.is_const))
            case _:
                raise CompileError(f"cannot take field address of {ptr}")

    def _emit(self, inst: mir.Inst, at: int | None = None) -> mir.Value:
        """Append one instruction to the flat body of the function
        being typed (``fn.insts``): the interpreter emits the whole MIR
        of a specialization into one list, delimited by the
        ``If``/``Else``/``End`` markers (there are no separate
        regions)."""
        fn = self._fn_instance.mir
        assert fn is not None
        if at is not None:
            fn.insts[at] = inst
        else:
            fn.insts.append(inst)
        return inst

    def _reserve(self) -> int:
        fn = self._fn_instance.mir
        assert fn is not None
        ret = len(fn.insts)
        fn.insts.append(mir.Nop())
        return ret

    def alloca(self, allow_comptime: bool = False):
        return PendingSlot(self._reserve(), allow_comptime)

    # -- helpers -------------------------------------------------------------

    def _coerce(self, ev: InterpVal, target: sval.Type) -> InterpVal:
        """Materialize a value of the spy type ``target``; numeric
        widening conversions (int -> float, float32 -> float64) are
        applied."""
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
        converted = self._convert_inst(value, from_type, to_type)
        if converted is None:
            return value
        return self._emit(converted)

    def _convert_inst(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Inst | None:
        """Build (but do not emit) the conversion of ``value`` from
        ``from_type`` to ``to_type``; returns ``value`` itself when no
        conversion is needed."""
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

    # -- operators ------------------------------------------------------------

    def _eval_binary(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
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
                mir.Arith(_ARITH_OPS[op], signed, _to_runtime(lc), _to_runtime(rc), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        elif isinstance(lhs_type, sval.StructType) or isinstance(rhs_type, sval.StructType):
            # call `__xxx__` methods
            raise NotImplementedError
        else:
            raise CompileError(f"unsupported operator '{op}' for {lhs_type} and {rhs_type}")

    def _eval_cmp(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
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
                mir.Cmp(_CMP_OPS[op], signed, kind, _to_runtime(lc), _to_runtime(rc))
            )
            self._frames[-1].regs[ret_reg] = RuntimeVal(value, sval.BoolType())
            return PollResult.AGAIN
        elif isinstance(lhs_type, sval.StructType) and isinstance(rhs_type, sval.StructType):
            # call `__xxx__` methods
            raise NotImplementedError
        else:
            raise CompileError(f'unsupported operand types: {lhs_type} and {rhs_type}')

    def _eval_boolop(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        lv = self._arg_value(lhs)
        rv = self._arg_value(rhs)
        if isinstance(lv, ComptimeVal) and isinstance(rv, ComptimeVal):
            result = (lv.obj and rv.obj) if op == 'and' else (lv.obj or rv.obj)
            self._frames[-1].regs[ret_reg] = ComptimeVal(bool(result))
            return PollResult.AGAIN
        raise CompileError(
            f"'{op}' between runtime values is not supported yet "
            '(only compile-time operands)'
        )

    def _eval_unary(self, op: str, operand: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        ev = self._arg_value(operand)
        if isinstance(ev, ComptimeVal):
            obj = ev.obj
            if op == 'not':
                result: sval.AnyValue = not obj
            elif op == 'neg':
                negated = sval.negate(obj)
                if negated is None:
                    raise CompileError(f'cannot negate {obj!r} at compile time')
                result = negated
            else:
                raise CompileError(f"unsupported unary operator '{op}'")
            self.store(ret, ComptimeVal(result))
            return PollResult.AGAIN
        type = _type_of(ev)
        if type is None:
            raise CompileError(f"cannot apply unary '{op}' to a compile-time object")
        coerced = self._coerce(ev, type)
        if op == 'not':
            if not isinstance(type, sval.BoolType):
                raise CompileError(f"cannot apply 'not' to a {type} value")
            value = self._emit(
                mir.Cmp('eq', False, 'int', _to_runtime(coerced), mir.BoolValue(False))
            )
            self.store(ret, RuntimeVal(value, sval.BoolType()))
            return PollResult.AGAIN
        if op == 'neg':
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
            value = self._emit(
                mir.Arith('sub', False, zero, _to_runtime(coerced), mir_type)
            )
            self.store(ret, RuntimeVal(value, type))
            return PollResult.AGAIN
        raise CompileError(f"unsupported unary operator '{op}'")

    # -- calls ----------------------------------------------------------------

    def call(self, callee: InterpVal, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """Resolve one call by its callee value and run it.  Spy
        functions compile to a native ``call`` producing a typed
        register, plain Python functions are inlined, and the spy
        builtins are evaluated at compile time.  The callee constant of a
        registered spy function already resolved to its entry when the
        callee operand was evaluated (see ``_operand``).

        Returns ``PollResult.AGAIN`` when the call completed here (the
        caller still hands its result to the result location), or
        ``PollResult.SUSPEND`` when an inlined callee's frame was pushed
        and its body is now running under the machine: the call's result
        is handed to the result location when the run ends."""
        if isinstance(callee, ComptimeVal):
            obj = callee.obj
            if not isinstance(obj, sval.ConstRef):
                raise CompileError("comptime values must be constant references")
            target = obj.value
            if isinstance(target, FunctionValue):
                return self._call_function_entry(target, args, ret)
            if isinstance(target, sval.BuiltinFn):
                return self._call_builtin(target, args, ret)
            if isinstance(target, sval.StructType):
                # a constructor ``Bar(...)``
                return self.call_constructor(target, args, ret)
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

    def _concretize(self, type: sval.Type) -> sval.Type:
        """Resolve a type that has no runtime representation of its own
        (an untyped integer literal) to a concrete default type."""
        if isinstance(type, sval.AnyIntType):
            return sval.IntType(sval.INT_DEFAULT_BITS, True)
        return type

    def _committed_type(self, slot: PendingSlot) -> sval.Type:
        """The final content type of a pending slot: the pairwise
        ``resolve_peer_type`` of its store-point types."""
        type: sval.Type | None = None
        for store in slot.stores:
            if type is None:
                type = store.type
            else:
                peer = type.resolve_peer_type(store.type)
                if peer is None:
                    raise CompileError(
                        f"a slot is stored with incompatible types {type} and {store.type}"
                    )
                type = peer
        if type is None:
            return sval.VoidType()
        return self._concretize(type)

    def _commit_pending_slot(
        self, val: InterpVal, type: sval.Type | None = None, ptr: mir.Value | None = None
    ) -> None:
        """Materialize a pending slot: it becomes a :class:`ComptimeBox`
        when it may inline values and every store point is compile-time,
        and a :class:`RuntimeVal` pointer otherwise.  The type is the
        pairwise ``resolve_peer_type`` of the store-point types (or the
        given one).  The recorded store points are delivered through
        ``_deliver_store``, which fills in the instructions that must be
        spliced at their original positions."""
        if not isinstance(val, PendingSlot):
            raise CompileError('can only commit a pending slot')
        if val.committed is not None:
            return
        if type is None:
            type = self._committed_type(val)
        if ptr is not None:
            self._bind_slot(val, ptr, type)
            return
        if val.allow_inline and len(val.stores) > 0 and all(s.is_comptime for s in val.stores):
            box = ComptimeBox(type)
            for store in val.stores:
                if isinstance(store.value, ComptimeVal):
                    box.value = ComptimeVal(sval.coerce_const(store.value.obj, type))
            val.committed = box
            return
        unit = type.get_unit_value()
        if unit is not None:
            val.committed = ComptimeBox(type, ComptimeVal(unit))
            return
        mir_type = type.to_mir_type()
        if mir_type is None or isinstance(mir_type, mir.VoidType):
            raise CompileError(f'cannot give a value of type {type} a runtime representation')
        alloca = self._emit(mir.Alloca(mir_type), val.mir_alloca_pos)
        self._bind_slot(val, alloca, type)

    def _bind_slot(self, slot: PendingSlot, ptr: mir.Value, type: sval.Type) -> None:
        slot.committed = RuntimeVal(ptr, sval.PointerType(type, is_const=False))
        for store in slot.stores:
            self._deliver_store(store, ptr, type)

    def _deliver_store(self, store: _PendingStore, ptr: mir.Value, type: sval.Type) -> None:
        if store.call is not None:
            call = store.call
            result_ptr = self._convert_result_ptr(ptr, type, store.type)
            store.insertion.insts.append(
                mir.Call(call.callee, (*call.prev_args, result_ptr), mir.VOID)
            )
            return
        value = store.value
        assert value is not None
        if store.is_comptime:
            assert isinstance(value, ComptimeVal)
            coerced = sval.coerce_const(value.obj, type)
            store.insertion.insts.append(mir.Store(ptr, _sval_to_runtime(coerced)))
        else:
            assert isinstance(value, RuntimeVal)
            converted = self._convert_inst(value.value, value.type, type)
            if converted is not None:
                store.insertion.insts.append(converted)
            else:
                converted = value.value
            store.insertion.insts.append(mir.Store(ptr, converted))

    def _convert_result_ptr(self, ptr: mir.Value, from_type: sval.Type, to_type: sval.Type) -> mir.Value:
        """The pointer a result-location call writes through, converted to
        the callee's result type.  Not implemented yet (the stub performs
        no conversion): it is only needed when the slot's final type and
        the call's return type differ."""
        return ptr

    def call_constructor(self, struct: sval.StructType, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        if '__init__' in struct.methods:
            init = self._analyser._resolver.resolve_global(struct.methods['__init__'])
            if init is not None and isinstance(init, FunctionValue):
                return self._call_function_entry(init, args, ret)

        # TODO: record pending slot!!!
        assert not isinstance(ret, PendingSlot) or ret.committed is not None
        for i, field_arg in enumerate(struct.bind_default_ctor_args(args)):
            if field_arg is None:
                # a zero-sized field occupies no storage and takes no value
                continue
            arg = field_arg.value
            if field_arg.is_ref:
                arg = self.load(arg)
            self.store(self.field_index_addr(ret, i), arg)
        return PollResult.AGAIN

    def _resolve_method(self, type: sval.Type, method_name: str):
        match type:
            case sval.StructType():
                if method_name in type.methods:
                    return self._analyser._resolver.resolve_global(type.methods[method_name])
                return None
            case _:
                return None

    def call_method(self, ptr: InterpVal, method_name: str, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        ptr = self._auto_deref(_shallow_normalize(ptr))
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
        self_is_ref = True
        if isinstance(method, FunctionValue):
            first = method.hir.signature.positional.by_id[0].type
            self_is_ref = not isinstance(first, sval.PointerType)

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
    ) -> PollResult:
        """A call of a registered spy function with the given (already
        evaluated) argument values - the common tail of an ordinary
        function call and of a method call, whose ``self`` the caller
        prepended to the arguments.  The call is specialized from the
        marshaled argument types (an annotated parameter fixes its type,
        an unannotated one is typed by its argument); a plain Python
        callee (``force_inline``) is inlined into the current stream
        instead of being compiled into a native specialization."""
        sig = fn.hir.signature
        binded_args = sig.bind_arg_pos(args, lambda e: ArgEntry(ComptimeVal(e), False))
        if fn.force_inline:
            # an undecorated plain Python function: its body is inlined into
            # the current stream (it has no native specialization of its own)
            return self._start_inline(fn.hir.body, binded_args, ret)
        arg_types = binded_args.map(_arg_type_of)
        spec_sig = sig.specialize(arg_types)
        self.resume_info = ResumeInfo(binded_args, spec_sig[0], ret, None)
        res = self._analyser._request_function(fn, spec_sig[0], spec_sig[1])
        if res is not None:
            fn_mir, ret_sig = res
            self.resume(fn_mir, ret_sig)
            return PollResult.AGAIN
        return PollResult.SUSPEND

    def resume(self, fn_mir: mir.Value, ret_sig: ReturnSignature):
        ri = self.resume_info
        assert ri is not None
        self.resume_info = None
        self._make_runtime_call(fn_mir, ri.args, ri.ret_loc, ri.ret_reg, ri.call_sig, ret_sig)

    def _make_runtime_call(self, callee: mir.Value, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal | None, ret_reg: hir.Inst | None, call_sig: SpecializedCallSignature, ret_sig: ReturnSignature) -> None:
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
            if ret is None:
                raise CompileError('a result-pointer call needs a result location')
            if isinstance(ret, PendingSlot) and ret.committed is None:
                # the result pointer of the slot is not known yet: record the
                # call and let the slot's commit supply it
                insertion = _MirInsertionBlock([], len(self._fn_instance.mir.insts))
                self._mir_insertion_blocks.append(insertion)
                ret.stores.append(
                    _PendingStore(
                        type=ret_sig.ret_type,
                        is_comptime=False,
                        insertion=insertion,
                        call=_PendingRetlocCall(callee, tuple(mir_args)),
                    )
                )
            else:
                ptr = _to_runtime(_shallow_normalize(ret))
                self._emit(mir.Call(callee, (*mir_args, ptr), mir.VOID))
        else:
            ret_type = ret_sig.ret_type.to_mir_type()
            if ret_type is None or isinstance(ret_type, mir.VoidType):
                self._emit(mir.Call(callee, tuple(mir_args), mir.VOID))
            else:
                call_inst = self._emit(mir.Call(callee, tuple(mir_args), ret_type))
                value = RuntimeVal(call_inst, ret_sig.ret_type)
                if ret is not None:
                    self.store(ret, value)
                elif ret_reg is not None:
                    self._frames[-1].regs[ret_reg] = value
                else:
                    raise CompileError('a call result has nowhere to go')

    def _start_inline(
        self,
        body: tuple[hir.Inst, ...],
        args: ArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
    ) -> PollResult:
        """Start the inlined body of a plain Python callee: convert its
        bound arguments into addressable values (the callee's ``hir.Arg``
        leaves denote its parameter slots) and push its frame under a
        fresh ``mir.Block`` (which its ``return`` statements leave with a
        ``mir.Break``).  The body now runs under the machine; its return
        statements write into ``ret`` directly (it is the body's result
        location), so no result is handed back here."""
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
        frame = InlineFrame(tuple(arg_values), ret, body)
        self._frames.append(frame)
        return PollResult.AGAIN

    # -- finishing -------------------------------------------------------------

    def finish(self) -> None:
        """Called when the body of the function proper has been fully
        typed: fix an inferred return convention and splice the deferred
        insertion blocks into the body."""
        self._finish_function()
        mir_fn = self._fn_instance.mir
        mir_fn.insts = self._splice_insertions(mir_fn.insts, self._mir_insertion_blocks)

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
                type = self._committed_type(location)
                self._materialize_result_ptr(type, None)
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

    def _splice_insertions(
        self, insts: list[mir.Inst], blocks: list[_MirInsertionBlock]
    ) -> list[mir.Inst]:
        """Build the final body by inserting every recorded block at its
        position (blocks are spliced in position order, so the positions
        - indices into the pre-splice body - stay valid)."""
        if not blocks:
            return insts
        result: list[mir.Inst] = []
        last = 0
        for block in sorted(blocks, key=lambda b: b.pos):
            result.extend(insts[last:block.pos])
            result.extend(block.insts)
            last = block.pos
        result.extend(insts[last:])
        return result

class Analyser:
    def __init__(self, resolver: FunctionResolver) -> None:
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

    def _request_function(self, fn_entry: FunctionValue, call_sig: SpecializedCallSignature, ret_sig: ReturnSignature | None) -> tuple[mir.Value, ReturnSignature] | None:
        """Make sure the specialization ``call_sig`` of ``fn_entry`` is
        compiled (into the module being built) and return its callee
        value and return signature - or ``None`` when the specialization
        was just started, in which case its runner has been pushed and the
        caller must suspend until it completes.

        A specialization that is already compiled is resolved to an
        external symbol (its definition lives in an earlier module); one
        that is still being compiled - a recursive reference - resolves
        to the in-module function being typed."""
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
        runner.run_function(fn_entry.hir.body, call_sig, ret_sig)
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

    def analyse_function(self, fn_entry: FunctionValue, call_sig: SpecializedCallSignature, ret_sig: ReturnSignature | None):
        """Type (and thereby compile) the specialization ``call_sig`` of
        ``fn_entry`` if it is not compiled yet."""
        if self._request_function(fn_entry, call_sig, ret_sig) is None:
            self._run()

    def finish(self) -> CompileBatch:
        return self._symbol_table

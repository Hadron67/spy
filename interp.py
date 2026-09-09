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
import typing
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any

from . import hir, mir, sval
from .errors import CompileError
from .fn import (
    ArgEntry,
    ArgList,
    FunctionInstance,
    FunctionValue,
    RawArgList,
    SpecializedComptimeArg,
    SpecializedFormalArg,
    SpecializedRuntimeArg,
    SpecializedSignature,
)
from .info import FunctionResolver
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
class PendingSlot(InterpVal):
    """The value of an executed ``hir.Alloca``: an addressable slot whose
    concrete type is fixed by its first store (the interpreter emits the
    typed MIR alloca at that moment).  A slot that receives the result of
    a *native* call (RLS) first only *records* the value: scalar and
    compile-time results are never given real memory - the matching
    ``Load`` hands the recorded value out directly - and memory is
    allocated only when the slot really must hold its value at a fixed
    address (a later plain ``Store``, a struct result, which the callee
    writes into the slot in place, or the result of an *inlined* call,
    whose per-path delivery stores into the slot immediately, see
    ``_deliver_inline_result``)."""

    # the spy type of the slot content (the type the slot is typed with
    # by its first store, in ``type.py``)
    mir_alloca_pos: int
    is_comptime: bool
    type: sval.Type | None = None
    runtime_ptr: mir.Value | None = None
    comptime_val: sval.AnyValue | None = None
    # This indicates the slot is a result pointer: write to this slot
    # would trigger ``_bind_result_loc``
    is_result_loc_ptr: bool = False

@dataclass
class ComptimeTuple(InterpVal):
    value_ptrs: tuple[InterpVal, ...]

@dataclass
class ComptimeDict(InterpVal):
    value_ptrs: dict[str, InterpVal]


@dataclass
class BlockFrame:
    """One open structured block of the flat instruction stream of the
    executing frame - an ``If`` whose branches are being typed (future
    block instructions will use the same frame).  The frame records the
    *entry* - the pc of the block-opening ``If`` in the frame's
    instruction list, from which the matching ``Else``/``End`` markers
    are found by a balanced scan - and the typing state of the ``if``:
    for a compile-time ``if`` the chosen branch (``chosen`` True: then,
    False: else), for a runtime ``if`` whether its then-region ended in
    a ``return`` (set once its typing leaves the then-region; the
    else-region is being typed once it is set, since both regions of a
    runtime ``if`` are always typed)."""

    entry: int
    chosen: bool | None = None
    then_returns: bool | None = None


class InlineFrame:
    """One function body being executed at compile time: the IR of the
    body it runs (``fn_ir``, which also fixes its by-value arguments -
    resolved by ``hir.Arg`` leaves: one ``InterpVal`` per argument, in
    declaration order - and its result location, the ``hir.ResultLoc``
    leaf its return statements write into, paired with the ``RetLocVal``
    holding the location's content, see ``RetLocVal``), together with the
    execution state of its body: the flat instruction list of the body
    (``insts``, its ``fn_ir.body``) with the pc of its next instruction
    (the walk of the list is driven by ``HirRunner``), the stack of its
    open blocks (see ``BlockFrame``) and, for an inlined callee, the
    pending call of the caller that the callee's value resumes when its
    run ends (``resume``; None for the function proper, whose run end
    just ends the machine) and the register table of its body
    (``regs``: the value of every executed instruction of the body,
    keyed by the instruction object - an instruction's register is only
    ever read by other instructions of the same body, so the table
    lives with the frame).

    Every frame carries its own ``fn_ir``, so the chain of frames above
    the function proper - one frame per inlined plain function - is
    also the chain of inlined bodies: the runner needs no separate
    inline stack (see ``HirRunner._start_inline``), and the number of
    inlined bodies under execution is ``len(frames) - 1``."""

    def __init__(self, arg_values: tuple[InterpVal, ...], ret_loc: InterpVal, insts: tuple[hir.Inst, ...]) -> None:
        self.arg_values: tuple[InterpVal, ...] = arg_values
        self.ret_loc = ret_loc
        self.insts = insts
        self.pc: int = 0
        self.block_stack: list[BlockFrame] = []
        self.regs: dict[hir.Inst, InterpVal] = {}

    def ret_levels(self):
        open_ifs = 0
        for bf in self.block_stack:
            if bf.chosen is None:
                open_ifs += 1
        return open_ifs + 1 if open_ifs > 0 else 0

# ---------------------------------------------------------------------------
# stateless helpers of the interpreter: pure functions over their arguments
# (argument/prototype construction, Python-literal constants, compile-time
# operators and operator error messages) - none of them uses instance state,
# so none of them is a method of :class:`HirRunner`
# ---------------------------------------------------------------------------

def _const_of_py(value: sval.AnyValue, type: sval.Type) -> sval.AnyValue:
    """Turn a Python literal into the typed MIR constant that mirrors the
    spy type ``type``."""
    match type:
        case sval.BoolType():
            if not isinstance(value, bool):
                raise CompileError(f"cannot use {value!r} as a bool constant")
            return value
        case sval.IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompileError(f"cannot use {value!r} as an integer constant")
            if type.signed:
                lo, hi = (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
            else:
                lo, hi = (0, 2 ** type.bits - 1)
            if not lo <= value <= hi:
                raise CompileError(
                    f"integer constant {value} is out of range for {sval.type_str(type)}"
                )
            return sval.Int(value, type)
        case sval.FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CompileError(f"cannot use {value!r} as a float constant")
            return sval.Float(float(value), type)
        case _:
            raise CompileError(
                f"cannot create a constant of type {sval.type_str(type)} from {value!r}"
            )


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
        case ComptimeVal():
            return _sval_to_runtime(ev.obj)
    raise CompileError('cannot return this value')

def _to_comptime(value: InterpVal) -> sval.AnyValue | None:
    match value:
        case ComptimeVal():
            return value.obj
        case _:
            return None

def _shallow_normalize(ev: InterpVal) -> InterpVal:
    """Shallow normalization, """
    if isinstance(ev, PendingSlot) and ev.type is not None:
        if ev.runtime_ptr is None:
            assert sval.to_mir_type(ev.type) is None
            return ComptimeVal(sval.Undefined(ev.type))
        else:
            return RuntimeVal(ev.runtime_ptr, sval.PointerType(ev.type, is_const=False))
    else:
        return ev

def _type_of(ev: InterpVal, allow_value_type: bool = False) -> sval.Type | None:
    """The spy type of the value ``ev`` denotes, or None when it has
    no spy representation (an un-typable compile-time object). Normalized values only"""
    match ev:
        case RuntimeVal(_, type):
            if isinstance(type, sval.ValueType) and not allow_value_type:
                return sval.type_of(type.value)
            return type
        case ComptimeVal(obj):
            return sval.type_of(obj) if not allow_value_type else sval.ValueType(obj)
        case _:
            return None

def _arg_type_of(arg: ArgEntry[InterpVal]):
    type = _type_of(arg.value)
    if arg.is_ref:
        assert isinstance(type, sval.PointerType)
        return type.elem
    return type

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
class FunctionInstanceRequest:
    hir: tuple[hir.Inst, ...]
    fn_entry: FunctionValue
    signature: SpecializedSignature

@dataclass
class ResumeInfo:
    args: ArgList[ArgEntry[InterpVal]]
    ret_loc: InterpVal | None
    ret_reg: hir.Inst | None

class HirRunner:
    """Runs one function body (and everything it inlines) at compile
    time, filling the pre-created typed :class:`mir.Function` of one
    specialization.

    ``resolver`` is the compile-time host, typed as the
    :class:`FunctionResolver` interface it implements (``dsl.JitContext``
    in practice) and provides:

    * ``hir_of(fn)``: the parsed (and cached) HIR of a Python function,
    * ``resolve_call(entry, arg_types)``: the callable value of one
      callee specialization (compiled into the module of the caller
      when it is still fresh, or a symbol of an earlier module),
    * ``resolve_global(obj)``: the entry of a global object that is a
      function registered in the host (or ``None``),
    * ``resolve_method(struct, name)``: the method ``name`` of a struct
      type (its registered entry, or the plain function to be
      inlined), or ``None`` when the struct has no such method.
    """

    def __init__(self, resolver: FunctionResolver, fn_instance: FunctionInstance) -> None:
        self._resolver = resolver
        # the frames of the function bodies under execution: the function
        # proper at the bottom, one frame per inlined plain function
        # above it (see ``_in_function_proper``; each frame carries the
        # IR of its body, see ``Frame``)
        self._frames: list[InlineFrame] = []
        # the function proper whose body is currently being typed (see
        # ``_bind_result_ptr``)
        self._fn_instance = fn_instance

        self.request: FunctionInstanceRequest | None = None
        self.resume_info: ResumeInfo | None = None

    # -- entry point ---------------------------------------------------------

    def run_function(
        self,
        hir: tuple[hir.Inst, ...],
        sig: SpecializedSignature,
    ):
        ret_loc = PendingSlot(self._reserve(), False, is_result_loc_ptr=True)
        mir_args: list[mir.Type] = []
        args = self._init_args_from_signature(sig, mir_args)
        for arg in mir_args:
            assert arg is not None
        self._fn_instance.mir.args = tuple(typing.cast(list[mir.Type], mir_args))
        if sig.ret_type is not None:
            self._materialize_result_ptr(sig.ret_type)

        frame = InlineFrame(args, ret_loc, hir)
        self._frames.append(frame)
        self._run_machine()

    def _init_one_arg(self, node: SpecializedFormalArg, mir_args: list[mir.Type]) -> InterpVal:
        match node:
            case SpecializedComptimeArg():
                return ComptimeVal(sval.ConstRef(node.value))
            case SpecializedRuntimeArg():
                type = sval.to_mir_type(node.type)
                assert not isinstance(type, mir.VoidType)
                index = len(mir_args)
                if node.is_ref:
                    type = mir.PointerType(type, True)
                    mir_args.append(type)
                    return RuntimeVal(mir.Param(index, type), sval.PointerType(node.type, True))
                else:
                    mir_args.append(type)
                    ret = self.alloca()
                    self.store(ret, RuntimeVal(mir.Param(index, type), node.type))
                    return ret
            case _:
                raise NotImplementedError

    def _init_args_from_signature(
        self,
        signature: SpecializedSignature,
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

    def _materialize_result_ptr(self, type: sval.Type) -> None:
        """ """
        raise NotImplementedError

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
        proper ended - its outcome recorded in ``self._flow`` by
        ``_frame_ended``.  There is no recursion: the walk is linear
        over one flat list per frame; the ``If``/``Else``/``End`` markers
        delimit the blocks, and every control state lives in the block
        stacks of the frames."""
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
            self._pop_frame()
            return PollResult.AGAIN
        inst = frame.insts[frame.pc]
        frame.pc += 1
        return self._exec_inst(inst)

    def _scan_block(self, entry: int) -> tuple[int | None, int]:
        """The positions of the ``Else`` (or None when the block has no
        else branch) and ``End`` markers that close the block opened at
        ``entry`` (an ``hir.If``) of the executing frame's flat
        instruction list, found by a balanced scan forward from the
        entry (nested blocks close their own markers first)."""
        insts = self._frames[-1].insts
        depth = 0
        p_else: int | None = None
        for i in range(entry + 1, len(insts)):
            inst = insts[i]
            if isinstance(inst, hir.If):
                depth += 1
            elif isinstance(inst, hir.End):
                if depth == 0:
                    return p_else, i
                depth -= 1
            elif isinstance(inst, hir.Else) and depth == 0:
                p_else = i
        assert False, 'unclosed block in the HIR'

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
                    self._cut()
                else:
                    ret_val = self.load(self._current_result_loc())
                    ret_type = _type_of(ret_val)
                    assert ret_type is not None
                    self._emit(mir.Ret(_to_runtime(ret_val) if not isinstance(ret_type, sval.VoidType) else None))
                    self._cut()
            case hir.If():
                self._exec_if(inst)
            case hir.Else():
                self._exec_else()
            case hir.End():
                self._exec_end()
            case hir.Load():
                regs[inst] = self.load(self.operand(inst.ptr))
            case hir.Alloca():
                regs[inst] = self.alloca()
            case hir.Store():
                self.store(self.operand(inst.ptr), self.operand(inst.value))
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
                frame.block_stack.append(BlockFrame(entry, chosen=True))
                return
            # the else branch is chosen: skip the (dead) then branch
            p_else, p_end = self._scan_block(entry)
            if p_else is None:
                # no else branch either: the whole ``if`` is dead
                frame.pc = p_end + 1
                return
            frame.block_stack.append(BlockFrame(entry, chosen=False))
            frame.pc = p_else + 1
            return
        self._exec_runtime_if(cond, entry)

    def _exec_else(self) -> None:
        """The walk fell off the end of the then branch and reached the
        ``Else`` marker of the innermost open block."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        if bf.chosen is None:
            # a runtime ``if``: its then-region fell off its end (it
            # does not return); the else-region is typed next
            self._rt_then_fell()
            return
        # a compile-time ``if`` whose chosen branch is the then branch,
        # which fell off its end: the (unchosen) else branch is dead -
        # skip it and close the block
        assert bf.chosen
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
        bf = frame.block_stack[-1]
        if bf.chosen is not None:
            # a compile-time ``if``: the chosen branch fell off its end
            frame.block_stack.pop()
            return
        self._rt_else_fell()

    # -- runtime ``if`` regions --------------------------------------------

    def _exec_runtime_if(
        self, cond: InterpVal, entry: int
    ) -> None:
        """A runtime ``if``: both branch bodies are typed and emitted
        (both survive at runtime): the walk continues into the
        then-region, then - once it ends, by falling off at its ``Else``
        marker or by a ``return`` (see ``_rt_then_fell`` and
        ``_rt_then_returned``) - into the else-region.  The MIR is
        emitted inline: a ``mir.If`` marker is opened here, closed by
        the ``mir.Else``/``mir.End`` markers emitted when the regions
        end.  A branch that falls off continues with the code after the
        ``End`` (both branches may fall through: the MIR's falling
        branches join there, see ``_rt_else_fell``).

        Inside an inlined body every return already stores its value
        into the call's result location on its own path (see the
        ``hir.Ret`` handling); a return inside a runtime branch
        additionally leaves the body with a ``mir.Break``."""
        if not isinstance(cond, RuntimeVal) or cond.type != sval.BoolType():
            raise CompileError('runtime if conditions must be boolean values')
        frame = self._frames[-1]
        self._emit(mir.If(cond.value))
        frame.block_stack.append(BlockFrame(entry))

    def _rt_then_fell(self) -> None:
        """The then-region of the runtime ``if`` on top of the executing
        frame's block stack fell off its end (its ``Else`` marker was
        reached): the else-region is typed next - the ``mir.Else``
        marker is emitted and the walk continues (its pc already points
        into the else-region)."""
        bf = self._frames[-1].block_stack[-1]
        assert bf.chosen is None and bf.then_returns is None
        bf.then_returns = False
        self._emit(mir.Else())

    def _rt_then_returned(self) -> None:
        """The then-region of the runtime ``if`` on top of the executing
        frame's block stack ended in a ``return`` (the path was cut):
        the else-region is typed next - or, when the ``if`` has no else
        branch, the ``if`` is complete (the empty else branch falls
        through) and the code after it is typed."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        assert bf.chosen is None and bf.then_returns is None
        bf.then_returns = True
        p_else, p_end = self._scan_block(bf.entry)
        if p_else is not None:
            self._emit(mir.Else())
            frame.pc = p_else + 1
            return
        self._emit(mir.End())
        frame.block_stack.pop()
        frame.pc = p_end + 1

    def _rt_else_fell(self) -> None:
        """The region of the runtime ``if`` on top of the executing
        frame's block stack that is currently being typed fell off its
        end (the ``End`` marker was reached): the ``if`` is complete and
        the code after the ``End`` is typed next.  A region that fell
        off continues at runtime with that code; when both branches fell
        through, both join it (a join - the MIR's falling branches
        already continue at the shared post-``End`` code, and block
        scoping keeps cross-join state in memory)."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        assert bf.chosen is None
        self._emit(mir.End())
        frame.block_stack.pop()

    def _rt_else_returned(self) -> bool:
        """The else-region of the runtime ``if`` on top of the executing
        frame's block stack ended in a ``return``: the ``if`` is
        complete.  Returns True when every path of the ``if`` returned
        (the code after it is dead and the enclosing path is cut too);
        otherwise the code after the ``if`` is typed next (it is the
        continuation of the then-region, which fell through)."""
        frame = self._frames[-1]
        bf = frame.block_stack[-1]
        assert bf.chosen is None
        then_returns = bf.then_returns
        assert then_returns is not None
        self._emit(mir.End())
        frame.block_stack.pop()
        if not then_returns:
            _, p_end = self._scan_block(bf.entry)
            frame.pc = p_end + 1
        return then_returns

    def _cut(self) -> None:
        """The current path of the executing frame ended - a ``return``
        was executed, or the path turned out dead (a runtime ``if``
        whose every branch returned): unwind the open blocks of the
        frame.  A compile-time ``if`` whose (chosen) branch the path ran
        through is just popped (the code after it is dead); a runtime
        ``if`` whose currently-typed region the path ended in continues
        with its sibling region, or - when its else-region ended - is
        complete: a single falling branch resumes the code after the
        ``if``, and when both branches returned the cut keeps unwinding.
        With no open block left, the run of the frame's body ended (see
        ``_frame_ended``)."""
        while True:
            frame = self._frames[-1]
            if not frame.block_stack:
                # the body run ended in a return: skip the dead code
                # after it
                self._pop_frame()
                return
            bf = frame.block_stack[-1]
            if bf.chosen is not None:
                # the path ran through the chosen branch of a
                # compile-time ``if`` and returned: dead code after it
                frame.block_stack.pop()
                continue
            if bf.then_returns is None:
                # the path ended inside the then-region: type the
                # else-region next (it is a fresh path)
                self._rt_then_returned()
                return
            # the path ended inside the else-region: the ``if`` is
            # complete; when every path returned the cut keeps unwinding
            if self._rt_else_returned():
                continue
            return

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
                    resolved = self._resolver.resolve_global(obj)
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
                resolved = self._resolver.resolve_global(obj)
                return ComptimeVal(sval.ConstRef(resolved))
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
        ptr = _shallow_normalize(ptr)
        type = _type_of(ptr)
        if not isinstance(type, sval.PointerType):
            raise CompileError(f"cannot load from a {type} value")
        unit_value = type.elem.get_unit_value()
        if unit_value is not None:
            return ComptimeVal(unit_value)
        match ptr:
            case PendingSlot():
                assert ptr.runtime_ptr is None and ptr.comptime_val is not None
                return ComptimeVal(ptr.comptime_val)
            case RuntimeVal():
                return RuntimeVal(
                    self._emit(mir.Load(ptr.value)), type.elem
                )
        raise CompileError('cannot load from a compile-time pointer')

    def store(self, ptr: InterpVal, value: InterpVal) -> None:
        ptr = _shallow_normalize(ptr)
        value = _shallow_normalize(value)
        ptr_type = _type_of(ptr)
        value_type = _type_of(value)
        if ptr_type is None:
            if value_type is None:
                raise CompileError('cannot store a value when both the pointer and value have no type')
            self.materialize_location(ptr, value_type)
            ptr = _shallow_normalize(ptr)
            ptr_type = _type_of(ptr)

        assert ptr_type is not None
        if not isinstance(ptr_type, sval.PointerType):
            raise CompileError(f"cannot store to a {ptr_type} value")
        value = self._coerce(value, ptr_type.elem)

        if isinstance(sval.to_mir_type(ptr_type.elem), mir.MayBeVoidType) or isinstance(value, sval.Undefined):
            # ZST
            return

        match ptr:
            case PendingSlot():
                assert ptr.is_comptime
                val = _to_comptime(value)
                if val is None:
                    raise CompileError('cannot store a non-comptime value to a comptime pointer')
                ptr.comptime_val = val
            case RuntimeVal():
                self._emit(mir.Store(ptr.value, _to_runtime(value)))
            case _:
                raise CompileError('cannot store through a compile-time pointer')

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
                raise NotImplementedError("TODO")
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
                return ComptimeVal(_const_of_py(obj, target))
            case RuntimeVal(value, type):
                return RuntimeVal(self._convert(value, type, target), target)
            case _:
                raise CompileError('cannot materialize this value')

    def _convert(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Value:
        if from_type == to_type:
            return value
        mir_to_type = sval.to_mir_type(to_type)
        assert mir_to_type is not None and not isinstance(mir_to_type, mir.MayBeVoidType)
        if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.IntType):
            if from_type.bits < to_type.bits:
                kind = 'sext' if from_type.signed else 'zext'
            else:
                kind = 'trunc'
            return self._emit(mir.Convert(kind, value, mir_to_type))
        if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.FloatType):
            kind = 'sitofp' if from_type.signed else 'uitofp'
            return self._emit(mir.Convert(kind, value, mir_to_type))
        if isinstance(from_type, sval.FloatType) and isinstance(to_type, sval.FloatType):
            kind = 'fpext' if from_type.bits < to_type.bits else 'fptrunc'
            return self._emit(mir.Convert(kind, value, mir_to_type))
        if isinstance(from_type, sval.PointerType) and isinstance(to_type, sval.PointerType):
            if from_type.is_const and not to_type.is_const:
                raise CompileError(
                    f"cannot convert a {from_type} value to {to_type}"
                )
            return value
        raise CompileError(
            f"cannot convert a {from_type} value to {to_type}"
        )

    # -- operators ------------------------------------------------------------

    def _eval_binary(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        raise NotImplementedError

    def _eval_cmp(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        raise NotImplementedError

    def _eval_boolop(self, op: str, lhs: ArgEntry[InterpVal], rhs: ArgEntry[InterpVal], ret_reg: hir.Inst) -> PollResult:
        raise NotImplementedError

    def _eval_unary(self, op: str, operand: ArgEntry[InterpVal], ret: InterpVal) -> PollResult:
        raise NotImplementedError

    # -- calls ----------------------------------------------------------------

    def call(self, callee: InterpVal, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """Resolve one call by its callee value and run it.  Spy
        functions compile to a native ``call`` producing a typed
        register, plain Python functions are inlined, and the spy
        builtins are evaluated at compile time.  The callee constant of a
        registered spy function already resolved to its entry when the
        callee operand was evaluated (see ``_operand``).

        Returns the value of the call when it completed here (the caller
        still hands it to the result location), or None when the call
        runs an inlined callee: its frame was pushed and its body is now
        running under the machine; the call's result is handed to the
        result location when the run ends (see ``_resume_call``)."""
        if isinstance(callee, ComptimeVal):
            obj = callee.obj
            if not isinstance(obj, sval.ConstRef):
                raise CompileError("comptime values must be constant references")
            obj = obj.value
            if isinstance(obj, FunctionValue):
                return self._call_function_entry(obj, args, ret)
            if isinstance(obj, sval.StructType):
                # a constructor ``Bar(...)``
                return self.call_constructor(obj, args, ret)
        raise CompileError(
            f"cannot compile a call to {callee!r}; only spy functions, plain Python "
            "functions and the spy builtins can be called"
        )

    # -- struct constructors and methods ----------------------------------------

    def materialize_location(self, val: InterpVal, type: sval.Type):
        """Initialize pending slot. Does not check type"""
        match val:
            case PendingSlot():
                if val.type is None:
                    assert val.type is None and val.comptime_val is None
                    val.type = type
                    assert val.type is None and val.runtime_ptr is None and val.comptime_val is None
                    if sval.is_comptime_only_type(type) and not val.is_comptime:
                        raise CompileError('cannot store a comptime-only value to a runtime pointer')
                    val.type = type
                    if not val.is_comptime:
                        mir_type = sval.to_mir_type(type)
                        assert mir_type is not None
                        if not isinstance(mir_type, mir.VoidType):
                            val.runtime_ptr = self._emit(mir.Alloca(mir_type), val.mir_alloca_pos)

    def call_constructor(self, desc: sval.StructType, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        """A struct constructor ``Bar(a, b)``: the result slot receives a
        new struct value.  With a user ``__init__`` the call is dispatched
        to it with ``self`` pointing at the result slot; otherwise every
        argument is written into the field of the same declaration index
        (the default constructor).  Either way the value is written into
        the result location in place - no value is handed back: returns
        ``InPlaceResult``, or None when a user ``__init__`` that is a
        plain Python function is being inlined (its run under the machine
        resumes the call, see ``_start_inline``)."""
        struct = desc
        self.materialize_location(ret, struct)
        if '__init__' in desc.methods:
            init = self._resolver.resolve_global(desc.methods['__init__'])
            if init is not None and isinstance(init, FunctionValue):
                return self._call_function_entry(init, args, ret)

        for i, field_arg in enumerate(desc.bind_default_ctor_args(args)):
            arg = field_arg.value
            if field_arg.is_ref:
                arg = self.load(arg)
            self.store(self.field_index_addr(ret, i), arg)
        return PollResult.AGAIN

    def _resolve_method(self, type: sval.Type, method_name: str):
        match type:
            case sval.StructType():
                if method_name in type.methods:
                    return self._resolver.resolve_global(type.methods[method_name])
                return None
            case _:
                return None

    def call_method(self, ptr: InterpVal, method_name: str, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult:
        ptr = self._auto_deref(_shallow_normalize(ptr))
        type = _type_of(ptr)
        if type is None or not isinstance(type, sval.PointerType):
            raise CompileError(f'cannot call a method on a {type} value')

        method = self._resolve_method(type, method_name)
        if method is None:
            raise CompileError(f'type {type} has no method named {method_name}')

        self_is_ref = not (isinstance(method, FunctionValue) and isinstance(method.hir.signature.positional.by_id[0].type, sval.PointerType))

        return self.call(ComptimeVal(sval.ConstRef(method)), RawArgList((ArgEntry(ptr, self_is_ref),) + args.positional, args.kwargs), ret)

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
        """A native call of a registered spy function (``@aot`` or
        ``@jit``) with the given (already evaluated) argument values -
        the common tail of an ordinary function call and of a method
        call, whose ``self`` the caller prepended to the arguments.  An
        aot function's signature is fixed by its entry's formals (a
        method's un-annotated ``self`` is typed there, see
        ``dsl._method_args``); a jit function solves the parameter types
        from the provided arguments."""
        sig = fn.hir.signature
        binded_args = sig.bind_arg_pos(args, lambda e: ArgEntry(ComptimeVal(e), False))
        arg_types = binded_args.map(_arg_type_of)
        spec_sig = sig.specialize(arg_types)

        if not fn.force_inline:
            self.request = FunctionInstanceRequest(fn.hir.body, fn, spec_sig)
            self.resume_info = ResumeInfo(binded_args, ret, None)
            if spec_sig in fn.specs:
                return self.resume()
            else:
                return PollResult.SUSPEND
        else:
            raise NotImplementedError

    def resume(self) -> PollResult:
        req = self.request
        ri = self.resume_info
        assert ri is not None and req is not None
        instance = req.fn_entry.specs[req.signature]
        self._make_runtime_call(instance.mir, ri.args, ri.ret_loc, ri.ret_reg, req.signature)
        return PollResult.AGAIN

    def _make_runtime_call(self, callee: mir.Value, args: ArgList[ArgEntry[InterpVal]], ret: InterpVal | None, ret_reg: hir.Inst | None, sig: SpecializedSignature) -> None:
        assert sig.ret_by_ref is not None and sig.ret_type is not None
        mir_args: list[mir.Value] = []

        def convert_one(arg: ArgEntry[InterpVal], sig_arg: SpecializedFormalArg):
            if isinstance(sig_arg, SpecializedRuntimeArg):
                if arg.is_ref and not sig_arg.is_ref:
                    mir_args.append(_to_runtime(self._coerce(self.load(arg.value), sig_arg.type)))
                elif not arg.is_ref and sig_arg.is_ref:
                    slot = self.alloca(False)
                    self.materialize_location(slot, sig_arg.type)
                    self.store(slot, arg.value)
                    mir_args.append(_to_runtime(slot))
                else:
                    mir_args.append(_to_runtime(self._coerce(arg.value, sig_arg.type)))

        for arg, (_, sig_arg) in zip(args.positional, sig.positional):
            convert_one(arg, sig_arg)

        if sig.varargs:
            for arg, sig_arg in zip(args.varargs, sig.varargs):
                convert_one(arg, sig_arg)

        if sig.kwargs:
            for name, arg in args.kwargs.items():
                convert_one(arg, sig.kwargs[name])

        if sig.ret_by_ref:
            assert ret is not None
            self.materialize_location(ret, sig.ret_type)
            mir_args.append(_to_runtime(ret))
            self._emit(mir.Call(callee, tuple(mir_args), mir.VoidType()))
        else:
            ret_val = RuntimeVal(self._emit(mir.Call(callee, tuple(mir_args), sval.to_mir_type(sig.ret_type))), sig.ret_type)
            if ret is not None:
                self.store(ret, ret_val)
            else:
                assert ret_reg is not None
                self._frames[-1].regs[ret_reg] = ret_val

    def _start_inline(
        self,
        hir: tuple[hir.Inst, ...],
        args: ArgList[ArgEntry[InterpVal]],
        ret: InterpVal,
    ) -> PollResult:
        if len(self._frames) - 1 >= _MAX_INLINE_DEPTH:
            raise CompileError(
                f'inline recursion or nesting exceeded '
                f'{_MAX_INLINE_DEPTH} levels'
            )
        raise NotImplementedError

class Analyser:
    def __init__(self) -> None:
        self._analyse_stack: list[HirRunner] = []

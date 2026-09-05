"""Compile-time interpretation of the HIR ("running" the HIR).

The interpreter executes the linear HIR instruction stream of a function
against the concrete argument types, emitting the typed MIR along the
way.  Every executed HIR instruction produces a value that is recorded in
a register table keyed by the instruction object itself, mirroring how
``symlat.jit.llvm`` registers work: operands of later instructions are
references to earlier instruction objects.

Values in the register table are either

* :class:`ComptimeVal` - a compile-time Python object (a literal, a spy
  type descriptor, a function to call/inline, ...),
* :class:`RuntimeVal` - the object of an already emitted MIR
  instruction (a typed runtime value), or
* :class:`PendingSlot` - an executed ``Alloca`` whose typed MIR alloca
  is emitted by its first store.

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

Both kinds of function calls push a *frame* holding the by-value
arguments (resolved by ``hir.Arg`` leaves); the addressable parameter
slots themselves are the ``Alloca``/``Store`` prologue that ``astgen``
placed at the head of every function body.  The interpreter types an
``Alloca`` when its first store executes, so the untyped HIR needs no
type information of its own.
"""

import operator
import types as pytypes
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, cast

from . import astgen, hir, mir, sval
from . import builtins as spy_builtins
from .errors import CompileError, TypeMismatchError
from .fn import FunctionEntry, FunctionValue
from .info import FunctionResolver

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
    obj: Any


@dataclass
class ComptimeRefVal(InterpVal):
    obj: Any


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
    a call (RLS) first only *records* the value: scalar and compile-time
    results are never given real memory - the matching ``Load`` hands the
    recorded value out directly - and memory is allocated only when the
    slot really must hold its value at a fixed address (a later plain
    ``Store``, or a struct result, which the callee writes into the slot
    in place)."""

    # the spy type of the slot content (the type the slot is typed with
    # by its first store, in ``type.py``)
    type: sval.Type | None = None
    ptr: mir.Value | None = None
    # the value an RLS call result recorded in the slot; ``Load`` returns
    # it while no memory has been allocated yet
    value: InterpVal | None = None


@dataclass
class InPlaceResult(InterpVal):
    """The result of a call that wrote its result into the result slot
    itself (a struct constructor, or a call whose result goes through a
    result pointer): the interpreter must not store it again."""


@dataclass
class RetLocVal(InterpVal):
    """The value of the result location of the function currently being
    typed (its ``hir.ResultLoc``): the location the return statements of
    the function write into, and whose content the terminating ``Ret``
    turns into the function's return.

    For a *direct-return* function the location only records the value
    of the path being typed (no memory - like the recorded result of an
    RLS scalar call); it is given real memory only when a writer needs
    an address (a constructor writing in place).  For a *result-pointer*
    function the location *is* the result pointer parameter of the
    function (``ptr`` is preset to it): a write is a store through it
    and the return is void."""

    # the value written by the last write of the current path: a typed
    # MIR value when the function proper is being typed, the raw
    # InterpVal of the return expression for an inlined body
    value: InterpVal | None = None
    # the memory of the location, when it has any (a direct-return
    # function whose result is written in place, or the result pointer
    # parameter of a result-pointer function); its static type is ``type``
    ptr: mir.Value | None = None
    # the spy type of the value the location holds (in ``type.py``)
    type: sval.Type | None = None


@dataclass
class Frame:
    """One function body being executed at compile time: its by-value
    arguments (resolved by ``hir.Arg`` leaves: each argument value with
    its spy type) together with its result location - the
    ``hir.ResultLoc`` leaf its return statements write into and the
    ``RetLocVal`` holding the location's content (see ``RetLocVal``)."""

    arg_values: tuple[tuple[mir.Value, sval.Type], ...]
    ret_loc: tuple[hir.ResultLoc, RetLocVal]


class Flow(IntEnum):
    """How executing a straight-line list of instructions ended: the
    list ``FALL`` off its end, or was cut short by a ``return``
    (``RET``) - the only case that carries a returned value."""

    FALL = auto()
    RET = auto()

# ---------------------------------------------------------------------------
# stateless helpers of the interpreter: pure functions over their arguments
# (argument/prototype construction, Python-literal constants, compile-time
# operators and operator error messages) - none of them uses instance state,
# so none of them is a method of :class:`HirRunner`
# ---------------------------------------------------------------------------


def _field_index(type: sval.StructType, name: str) -> int:
    """The declaration index of the field ``name`` of a struct type."""
    index = type.field_index(name)
    if index is None:
        raise CompileError(
            f"type {sval.type_str(type)} has no field named '{name}'"
        )
    return index


def _const_of_py(obj: Any, type: sval.Type) -> mir.Value:
    """Turn a Python literal into the typed MIR constant that mirrors the
    spy type ``type``."""
    match type:
        case sval.BoolType():
            if not isinstance(obj, bool):
                raise CompileError(f"cannot use {obj!r} as a bool constant")
            return mir.BoolValue(obj)
        case sval.IntType():
            if isinstance(obj, bool) or not isinstance(obj, int):
                raise CompileError(f"cannot use {obj!r} as an integer constant")
            if type.signed:
                lo, hi = (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
            else:
                lo, hi = (0, 2 ** type.bits - 1)
            if not lo <= obj <= hi:
                raise CompileError(
                    f"integer constant {obj} is out of range for {sval.type_str(type)}"
                )
            return mir.IntValue(obj, type.bits, type.signed)
        case sval.FloatType():
            if isinstance(obj, bool) or not isinstance(obj, (int, float)):
                raise CompileError(f"cannot use {obj!r} as a float constant")
            return mir.FloatValue(float(obj), type.bits)
        case _:
            raise CompileError(
                f"cannot create a constant of type {sval.type_str(type)} from {obj!r}"
            )


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


def _binary_type(lt: sval.Type, rt: sval.Type, what: str) -> sval.Type | None:
    """The spy type a binary operation on ``lt``/``rt`` is performed on,
    or None if the operand combination is not a number pair."""
    if isinstance(lt, sval.IntType) and isinstance(rt, sval.IntType):
        if lt != rt:
            raise CompileError(
                f"cannot {what} a {sval.type_str(lt)} value with a {sval.type_str(rt)} value "
                "(different integer types)"
            )
        return lt
    if isinstance(lt, sval.FloatType) and isinstance(rt, sval.FloatType):
        return sval.FloatType(max(lt.bits, rt.bits))
    if isinstance(lt, sval.IntType) and isinstance(rt, sval.FloatType):
        return rt
    if isinstance(lt, sval.FloatType) and isinstance(rt, sval.IntType):
        return lt
    return None


def _unsupported_type_error(op: str, type: sval.Type | None) -> CompileError:
    if isinstance(type, sval.PointerType) and isinstance(type.elem, sval.IntType) and type.elem.bits == 8:
        return CompileError(
            f"cannot apply '{op}' to string values "
            "(strings are compiled as arrays of u8)"
        )
    if isinstance(type, sval.PointerType):
        return CompileError(f"cannot apply '{op}' to pointer values")
    if type is None:
        return CompileError(f"cannot apply '{op}' to a compile-time object")
    return CompileError(f"cannot apply '{op}' to {sval.type_str(type)} values")



def _to_runtime(ev: InterpVal, target: sval.Type | None) -> tuple[mir.Value, sval.Type]:
    """Materialize a value as a typed runtime value: runtime values must
    already have the target type, compile-time values adopt it (or,
    without a target, their Python type mapping).  Returns the typed MIR
    value and its spy type."""
    if isinstance(ev, RuntimeVal):
        value = ev.value
        if target is not None and ev.type != target:
            raise CompileError(
                f"cannot return a {sval.type_str(ev.type)} value where {sval.type_str(target)} "
                "is expected"
            )
        return value, ev.type
    if isinstance(ev, ComptimeVal):
        if ev.obj is None:
            raise CompileError("cannot return None (functions must return a value)")
        if target is not None:
            return _const_of_py(ev.obj, target), target
        t = sval.value_type(ev.obj)
        if t is None:
            raise CompileError(f"cannot return the compile-time value {ev.obj!r}")
        return _const_of_py(ev.obj, t), t
    raise CompileError('cannot return this value')


def _to_slot(ev: InterpVal, type: sval.Type) -> mir.Value:
    """Materialize ``ev`` as a value of exactly ``type`` for a store
    into an already-typed slot (the strict sibling of ``_to_runtime``,
    whose messages talk about stores)."""
    match ev:
        case RuntimeVal(value, t):
            if t != type:
                raise CompileError(
                    f"cannot store a {sval.type_str(t)} value into a "
                    f"slot of type {sval.type_str(type)}"
                )
            return value
        case ComptimeVal(obj):
            return _const_of_py(obj, type)
        case _:
            raise CompileError(
                f"cannot store this value into a slot of type {sval.type_str(type)}"
            )


def _convert_evals(
    fn_ir: astgen.FunctionIR,
    evals: list[InterpVal],
    formal: tuple[sval.Type, ...],
) -> tuple[mir.Value, ...]:
    """Materialize the (possibly defaulted) arguments of one call as
    values of the given formal spy types."""
    values: list[mir.Value] = []
    for i, param in enumerate(fn_ir.params):
        if i < len(evals):
            ev = evals[i]
            if isinstance(ev, ComptimeVal):
                values.append(_const_of_py(ev.obj, formal[i]))
            elif isinstance(ev, RuntimeVal):
                if ev.type != formal[i]:
                    raise CompileError(
                        f"cannot pass a {sval.type_str(ev.type)} value as the "
                        f"'{param.name}' argument of function {fn_ir.name} "
                        f"(expected {sval.type_str(formal[i])})"
                    )
                values.append(ev.value)
            else:
                raise CompileError('cannot pass this value as an argument')
        else:
            assert param.has_default
            values.append(_const_of_py(param.default_value, formal[i]))
    return tuple(values)


def _materialize_arg(ev: InterpVal, target: sval.Type, what: str) -> mir.Value:
    """Materialize one argument value of exactly the spy type ``target``
    (used by constructors, whose parameters are the struct fields)."""
    match ev:
        case ComptimeVal(obj):
            return _const_of_py(obj, target)
        case RuntimeVal(value, type):
            if type != target:
                raise CompileError(
                    f"cannot pass a {sval.type_str(type)} value as {what} "
                    f"(expected {sval.type_str(target)})"
                )
            return value
        case _:
            raise CompileError(f'cannot pass this value as {what}')


def _type_of(ev: InterpVal) -> sval.Type | None:
    """The spy type of the value ``ev`` denotes, or None when it has
    no spy representation (an un-typable compile-time object)."""
    match ev:
        case RuntimeVal(_, type):
            return type
        case ComptimeVal(obj):
            return sval.value_type(obj)
        case ComptimeRefVal(obj):
            t = sval.value_type(obj)
            return sval.PointerType(t, is_const=True) if t is not None else None
        case _:
            return None

def _describe(ev: InterpVal) -> str:
    if isinstance(ev, ComptimeVal):
        return repr(ev.obj)
    t = _type_of(ev)
    if t is None:
        return 'an untyped value'
    return f'a {sval.type_str(t)} value'

def _bin_types(
    lhs: InterpVal, rhs: InterpVal
) -> tuple[sval.Type | None, sval.Type | None]:
    """The spy types of the two operands of a binary operation.  A
    compile-time integer constant adopts the type of a runtime
    integer operand (``x + 1`` with ``x: u64`` is a u64 addition,
    like an integer argument marshals to the annotated type at the
    Python boundary); compile-time floats keep the default mapping
    and mix with integers by promotion."""
    lt = _type_of(lhs)
    rt = _type_of(rhs)
    if (
        isinstance(rhs, ComptimeVal)
        and not isinstance(rhs.obj, bool)
        and isinstance(rhs.obj, int)
        and isinstance(lt, sval.IntType)
    ):
        rt = lt
    elif (
        isinstance(lhs, ComptimeVal)
        and not isinstance(lhs.obj, bool)
        and isinstance(lhs.obj, int)
        and isinstance(rt, sval.IntType)
    ):
        lt = rt
    return lt, rt

# ---------------------------------------------------------------------------
# the compile-time host interface
# ---------------------------------------------------------------------------


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

    def __init__(self, resolver: FunctionResolver) -> None:
        self._resolver = resolver
        # the frames of the function bodies under execution: the function
        # proper at the bottom, one frame per inlined plain function
        # above it (see ``_in_function_proper``)
        self._frames: list[Frame] = []
        self._regs: dict[hir.Inst, InterpVal] = {}
        self._inline_stack: list[astgen.FunctionIR] = []
        self._inline_depth = 0
        # the function proper whose body is currently being typed (see
        # ``_bind_result_ptr``)
        self._fn: mir.Function | None = None
        # the spy return type of the function proper, fixed by its
        # return sites (or its declared return annotation); every check
        # on it happens in the ``spy`` type system (``type.py``)
        self._ret_type: sval.Type | None = None
        # the declared spy return type of the function proper (its
        # annotation, or None when it is inferred from the body); the
        # target every return site is checked against
        self._ret_target: sval.Type | None = None
        # True when a path of the function proper ended in a void return
        # (a bare ``return``, or a fall-off of a void function)
        self._saw_void_return = False
        # the emission regions (see ``_emit``): a stack of lists whose
        # top is the region currently being typed
        self._regions: list[list[mir.Inst]] = []
        # the spy return type of the function proper declared by its
        # annotation (or None when it is inferred from the body); the
        # target every return site is checked against
        # the return mode of the function proper: 'ptr' when its return
        # type is delivered through a result pointer (the function then
        # has a trailing result pointer formal and returns void), 'value'
        # when it returns the value directly; None while the return type
        # is still unknown (a body that never returns a value is void)
        self._result_mode: str | None = None
        # the result pointer parameter of a result-pointer function
        # proper (its trailing formal; see ``_bind_result_ptr``)
        self._result_ptr: mir.Value | None = None
        # True when the return statement of the path currently being
        # typed has written its value into the result location
        self._ret_written = False

    # -- entry point ---------------------------------------------------------

    def run_function(
        self,
        fn: mir.Function,
        fn_ir: astgen.FunctionIR,
        arg_types: tuple[sval.Type, ...],
        ret_hint: sval.Type | None,
    ) -> sval.Type:
        """Type the body of ``fn_ir`` into ``fn``.

        ``fn`` is the MIR function the body is emitted into.  The host
        created it - fixing its name and, from the *spy* signature
        passed here, its lowered argument list and (when one is
declared) return type - and registered it *before* running the body,
        so a call the body makes to the function itself - recursion -
        resolves to ``fn``, whose signature is already fixed.  This
        fills ``fn.insts`` and fixes ``fn.ret_type``: the declared type,
        or the type inferred from the return sites when none is
declared.

        Returns the *logical spy return type* of the specialization -
        the type its callers see - which the host records for later
        callers (a ``FunctionCallInfo`` is derived from it, see
        ``type.function_call_info``).

        All type decisions happen on the spy types ``arg_types`` /
        ``ret_hint``; MIR types are only ever produced from them (the
        host's mirrors of the signature are never read back for a
decision).  The return convention of the function is decided here
        from its spy return type (see ``type.returns_via_result_ptr``)
        and lowered into the MIR signature of ``fn``: a function that
        returns through a result pointer gets a trailing result pointer
        formal appended to ``fn.args`` and a ``void`` return, and its
        return type is kept in ``fn.result_type``.
        """
        self._fn = fn
        self._ret_target = ret_hint
        self._result_mode = None
        self._result_ptr = None
        self._ret_written = False
        # the frame of the function proper is pushed before its
        # (possibly result-pointer) signature is lowered: its result
        # location must be in place when the result pointer is bound
        # (see ``_bind_result_ptr``); its argument values are filled in
        # once the signature is fixed
        frame = self._push_frame((), fn_ir.ret_loc)
        declared = ret_hint
        if declared is not None and sval.returns_via_result_ptr(declared):
            # the declared return type is delivered through a result
            # pointer: lower the signature before the body is typed, so
            # that recursive calls the body makes see the final form
            self._bind_result_ptr(fn, declared)
        param_values = tuple(
            (mir.Param(i, sval.to_mir_type(t), fn_ir.params[i].name), t)
            for i, t in enumerate(arg_types)
        )
        if self._result_mode == 'ptr':
            # the result pointer formal of a result-pointer function is a
            # parameter of the lowered signature like any other
            result_ptr = self._result_ptr
            assert result_ptr is not None
            result_type = ret_hint
            assert result_type is not None
            param_values += (
                (result_ptr, sval.PointerType(result_type, is_const=False)),
            )
        frame.arg_values = param_values
        self._ret_type = None
        self._saw_void_return = False
        self._regions = [fn.insts]

        flow, _ = self._run_list(fn_ir.body)
        if flow is not Flow.RET:
            # a path falls off the end of the body: allowed only for a
            # void function (declared, or inferred when the body never
            # returned a value) - the fall-off path ends in an implicit
            # ``ret void``; a value-returning function must end every
            # path with a ``return``
            if self._ret_type is not None:
                raise CompileError(
                    f"function {fn_ir.name} returns a value on some paths but "
                    'falls off its end (without a return) on others'
                )
            if declared is not None and declared != sval.VoidType():
                raise CompileError(
                    f"function {fn_ir.name} must end with a 'return' statement"
                )
            self._finish_void()
            ret_type = sval.VoidType()
        elif self._saw_void_return:
            if self._ret_type is not None:
                raise CompileError(
                    f"function {fn_ir.name} returns a value on some paths but "
                    'returns without a value on others'
                )
            ret_type = sval.VoidType()
        else:
            ret_type = self._ret_type
            assert ret_type is not None
            if declared is not None:
                # a declared return type is enforced at the return sites
                # (see ``_write_result``), so the two must agree
                assert declared == ret_type
        self._frames.pop()
        if self._result_mode == 'ptr':
            # a result-pointer function: its MIR signature was lowered to
            # a trailing result pointer formal and a void return when the
            # mode was bound (see ``_bind_result_ptr``)
            fn.ret_type = mir.VoidType()
        else:
            fn.ret_type = sval.to_mir_type(ret_type)
        return ret_type

    def _bind_result_ptr(self, fn: mir.Function, logical: sval.Type) -> None:
        """Lower the signature of the function proper to its result
        pointer form: append the trailing result pointer formal and fix
        the return type to void.  ``logical`` is the spy type of the
        value the function returns (kept in ``fn.result_type``, mirrored
        into MIR) - a type delivered through a result pointer is not
        necessarily a struct (arrays and other aggregates will use the
        same convention)."""
        index = len(fn.args)
        result_type = sval.to_mir_type(logical)
        formal = mir.FormalArg('$result', mir.PointerType(result_type))
        fn.args = fn.args + (formal,)
        fn.result_type = result_type
        param = mir.Param(index, formal.type, formal.name)
        self._result_ptr = param
        self._result_mode = 'ptr'
        # the result location of the function is its result pointer: return
        # values are written through it and the function returns void
        retloc = self._result_loc_of()
        retloc.ptr = param
        retloc.type = logical

    # -- return statements ------------------------------------------------

    def _result_loc_of(self) -> RetLocVal:
        assert len(self._frames) > 0, 'internal error: no function result location'
        return self._frames[-1].ret_loc[1]

    def _in_function_proper(self) -> bool:
        """Whether the instructions currently being executed are those
        of the function proper (whose ``return`` emits a typed return)
        rather than of an inlined plain function (whose ``return`` just
        yields a value to the caller): the frames stack holds the
        function proper at its bottom and one frame per inlined body
        above it, so the innermost body is the function proper exactly
        when it is the only frame."""
        return len(self._frames) == 1

    def _write_result(self, ev: InterpVal) -> None:
        """The value of one return expression of the function proper is
        written into its result location: a direct-return function
        records it (the terminating ``Ret`` turns it into the return
        value), a result-pointer function stores it through its result
        pointer.  This is where every return site is typed (against the
        declared return type) and cross-path consistency is checked."""
        retloc = self._result_loc_of()
        target = self._ret_target
        if target == sval.VoidType():
            raise CompileError(
                'cannot return a value from a void function (its return '
                'type is None)'
            )
        if isinstance(ev, ComptimeVal) and ev.obj is None and target is None:
            # the value of a void expression (e.g. a call of a void
            # function) in return position: the function is void
            return
        if isinstance(ev, ComptimeVal) and ev.obj is None:
            raise CompileError('cannot return None (functions must return a value)')
        value, t = _to_runtime(ev, target)
        if self._ret_type is not None and self._ret_type != t:
            raise CompileError(
                f"function returns values of conflicting types "
                f"{sval.type_str(self._ret_type)} and {sval.type_str(t)}"
            )
        self._ret_type = t
        if self._result_mode is None:
            # the return type is inferred from this site: decide the
            # return convention from it
            if sval.returns_via_result_ptr(t):
                # the signature is still being typed and nothing has
                # referenced the function yet (an inferred function can
                # never be recursive), so appending the formal is safe
                fn = self._fn
                assert fn is not None
                self._bind_result_ptr(fn, t)
            else:
                self._result_mode = 'value'
        if self._result_mode == 'ptr':
            assert self._result_ptr is not None
            self._emit(mir.Store(self._result_ptr, value))
            self._ret_written = True
            return
        # a direct-return function: the value of the path is recorded and
        # the terminating ``Ret`` returns it
        assert self._result_mode == 'value'
        retloc.value = RuntimeVal(value, t)
        self._ret_written = True

    def _note_inplace_ret(self, type: sval.Type) -> None:
        """A return-path write that happened in place (a constructor or
        a result-pointer callee wrote straight into the result location):
        the cross-path return-type bookkeeping, without a value."""
        if self._ret_type is not None and self._ret_type != type:
            raise CompileError(
                f"function returns values of conflicting types "
                f"{sval.type_str(self._ret_type)} and {sval.type_str(type)}"
            )
        self._ret_type = type
        self._ret_written = True

    def _finish_path(self) -> None:
        """One path of the function proper ends in its ``return``: emit
        the return of the path - the recorded value of a direct-return
        function, a ``ret void`` of a result-pointer function (whose
        result was stored through the result pointer) - or check the
        bare ``return`` of a void path."""
        written = self._ret_written
        self._ret_written = False
        if self._result_mode == 'ptr':
            if not written:
                # a result-pointer function always returns a value
                retloc = self._result_loc_of()
                logical = retloc.type
                assert logical is not None
                raise CompileError(
                    'cannot return without a value where '
                    f'{sval.type_str(logical)} is expected'
                )
            self._emit(mir.Ret(None))
            return
        if not written:
            self._finish_void()
            return
        retloc = self._result_loc_of()
        if retloc.ptr is not None:
            # the value was written in place (a constructor): load it
            # back to return it
            assert retloc.type is not None
            value = self._emit(mir.Load(retloc.ptr, sval.to_mir_type(retloc.type)))
            retloc.ptr = None
            retloc.type = None
            self._emit(mir.Ret(value))
            return
        value = retloc.value
        retloc.value = None
        assert isinstance(value, RuntimeVal)
        self._emit(mir.Ret(value.value))

    # -- running instruction lists -------------------------------------------

    def _run_list(self, insts: tuple[hir.Inst, ...]) -> tuple[Flow, InterpVal | None]:
        """Execute instructions in order; a ``Ret`` (executed directly in
        this list, i.e. not nested inside an inlined function) stops the
        list and reports the returned value."""
        for inst in insts:
            flow, value = self._exec_inst(inst)
            if flow is Flow.RET:
                return flow, value
        return Flow.FALL, None

    def _exec_inst(self, inst: hir.Inst) -> tuple[Flow, InterpVal | None]:
        match inst:
            case hir.Ret():
                if not self._in_function_proper():
                    # an inlined callee ends: yield the value its return
                    # statements wrote into its result location (the raw
                    # return expression value, or None for a void body)
                    retloc = self._result_loc_of()
                    if retloc.ptr is not None:
                        # the value was written in place (a constructor):
                        # load it back to yield it
                        assert retloc.type is not None
                        value: InterpVal | None = RuntimeVal(
                            self._emit(mir.Load(retloc.ptr, sval.to_mir_type(retloc.type))),
                            retloc.type,
                        )
                        retloc.ptr = None
                        retloc.type = None
                        return Flow.RET, value
                    value = retloc.value
                    retloc.value = None
                    return Flow.RET, value if value is not None else ComptimeVal(None)
                self._finish_path()
                return Flow.RET, None
            case hir.If():
                cond = self._operand(inst.cond)
                if isinstance(cond, ComptimeVal):
                    chosen = inst.then_body if cond.obj else inst.else_body
                    return self._run_list(chosen)
                return self._exec_runtime_if(inst, cond)
            case hir.Load():
                self._regs[inst] = self._load(self._operand(inst.ptr))
                return Flow.FALL, None
            case hir.Alloca():
                self._regs[inst] = PendingSlot()
                return Flow.FALL, None
            case hir.Store():
                self._store(self._operand(inst.ptr), self._operand(inst.value))
                return Flow.FALL, None
            case hir.Binary():
                self._regs[inst] = self._eval_binary(inst)
                return Flow.FALL, None
            case hir.Compare():
                self._regs[inst] = self._eval_cmp(inst)
                return Flow.FALL, None
            case hir.BoolOp():
                self._regs[inst] = self._eval_boolop(inst)
                return Flow.FALL, None
            case hir.Unary():
                self._regs[inst] = self._eval_unary(inst)
                return Flow.FALL, None
            case hir.CallInplace():
                self._exec_call_inplace(inst)
                return Flow.FALL, None
            case hir.CallMethodInplace():
                self._exec_call_method(inst)
                return Flow.FALL, None
            case hir.FieldAddr():
                self._regs[inst] = self._exec_field_addr(inst)
                return Flow.FALL, None
            case _:
                raise CompileError(f"unsupported instruction {type(inst).__name__}")

    def _operand(self, value: hir.Value) -> InterpVal:
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
                if not isinstance(obj, (int, float, str, bool, type(None))):
                    resolved = self._resolver.resolve_global(obj)
                    if resolved is not None:
                        return ComptimeVal(resolved)
                return ComptimeVal(obj)
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
                return ComptimeRefVal(resolved if resolved is not None else obj)
            case hir.Arg(index):
                if len(self._frames) == 0:
                    raise CompileError('internal error: Arg outside of any function frame')
                frame = self._frames[-1]
                if index >= len(frame.arg_values):
                    raise CompileError('internal error: Arg index out of range')
                arg_value, arg_type = frame.arg_values[index]
                return RuntimeVal(arg_value, arg_type)
            case hir.ResultLoc():
                # the result location of the innermost body being typed
                # whose leaf this is (its own, or - during an inlined
                # call - the callee's)
                for frame in reversed(self._frames):
                    leaf, retloc = frame.ret_loc
                    if leaf is value:
                        return retloc
                raise CompileError('internal error: result location outside of any function')
            case hir.Inst():
                reg = self._regs.get(value)
                if reg is None:
                    raise CompileError('internal error: register not evaluated')
                return reg
            case _:
                raise CompileError(f"unsupported operand {type(value).__name__}")

    # -- memory instructions -------------------------------------------------

    def _push_frame(
        self,
        arg_values: tuple[tuple[mir.Value, sval.Type], ...],
        ret_loc: hir.ResultLoc,
    ) -> Frame:
        """Push the frame of one function body (its by-value arguments
        and its result location, the ``hir.ResultLoc`` leaf its return
        statements write into, with a fresh ``RetLocVal``); returns the
        frame."""
        frame = Frame(arg_values, (ret_loc, RetLocVal()))
        self._frames.append(frame)
        return frame

    def _load(self, ptr: InterpVal) -> InterpVal:
        if isinstance(ptr, PendingSlot):
            if ptr.ptr is None:
                # an un-materialized RLS slot: the recorded value of the
                # ``CallInplace`` that initialized it is the load result
                if ptr.value is None:
                    raise CompileError(
                        'cannot load from a slot before any store to it executed'
                    )
                return ptr.value
            assert ptr.type is not None
            return RuntimeVal(
                self._emit(mir.Load(ptr.ptr, sval.to_mir_type(ptr.type))), ptr.type
            )
        if isinstance(ptr, RuntimeVal):
            ptype = ptr.type
            if not isinstance(ptype, sval.PointerType):
                raise CompileError(f"cannot load from a {sval.type_str(ptype)} value")
            return RuntimeVal(
                self._emit(mir.Load(ptr.value, sval.to_mir_type(ptype.elem))), ptype.elem
            )
        raise CompileError('cannot load from a compile-time pointer')

    def _store(self, ptr: InterpVal, value: InterpVal) -> None:
        if isinstance(ptr, RetLocVal):
            # a store into the function result location (the expression
            # of a ``return`` statement that is not a call): a
            # direct-return function records the value of the path, a
            # result-pointer function stores it through its result
            # pointer
            if self._in_function_proper():
                self._write_result(value)
            else:
                # an inlined body: the return expression is only recorded
                # and yielded at its ``Ret``
                ptr.value = value
            return
        if isinstance(ptr, PendingSlot):
            if ptr.ptr is None and ptr.value is not None:
                # the slot holds an RLS call result that was only
                # recorded: materialize it before overwriting the slot
                recorded = ptr.value
                ptr.value = None
                v0, t0 = _to_runtime(recorded, None)
                ptr.ptr = self._emit(mir.Alloca(mir.PointerType(sval.to_mir_type(t0))))
                ptr.type = t0
                self._emit(mir.Store(ptr.ptr, v0))
            # the first store types the slot
            v, t = _to_runtime(value, None)
            if ptr.ptr is None:
                assert ptr.type is None
                ptr.ptr = self._emit(mir.Alloca(mir.PointerType(sval.to_mir_type(t))))
                ptr.type = t
            elif ptr.type is None or ptr.type != t:
                raise CompileError(
                    f"cannot store a {sval.type_str(t)} value into a slot of a "
                    f"different type"
                )
            self._emit(mir.Store(ptr.ptr, v))
            return
        if isinstance(ptr, RuntimeVal):
            ptype = ptr.type
            if not isinstance(ptype, sval.PointerType):
                raise CompileError(f"cannot store through a {sval.type_str(ptype)} value")
            v, _ = _to_runtime(value, ptype.elem)
            self._emit(mir.Store(ptr.value, v))
            return
        raise CompileError('cannot store through a compile-time pointer')

    # -- struct values ---------------------------------------------------------

    def _struct_addr_of(self, ev: InterpVal) -> tuple[mir.Value, sval.StructType]:
        """The address of the struct value the base of a field/method
        access (``a.b``, ``a.h()``) denotes, and the spy struct type at
        that address.

        The base is the *storage* of the struct: the slot of a variable
        (an ``Alloca``), or the address of a nested field (a ``Gep``) -
        ``astgen`` generates it with ``_gen_ref``, so a base that is a
        pointer *variable* comes out as a pointer to the slot holding
        the pointer, i.e. a pointer to a pointer.  Storage may hold the
        struct itself or a chain of pointers to it (a ``self`` passed by
        pointer, a pointer local or field, ...); pointers are
        *auto-dereferenced* (loaded) until the struct is reached.  The
        returned address is a runtime pointer value whose element type
        is the MIR mirror of the struct type."""
        if isinstance(ev, PendingSlot):
            t = ev.type
            if t is None:
                raise CompileError('cannot access the fields of a variable that has not been assigned yet')
            if not isinstance(t, sval.PointerType):
                # the slot itself holds the struct value
                if isinstance(t, sval.StructType):
                    assert ev.ptr is not None
                    return ev.ptr, t
                raise CompileError(
                    f"cannot access fields of a {sval.type_str(t)} value: "
                    'only struct values have fields'
                )
            # the slot holds a pointer (a ``self`` passed by pointer, a
            # pointer local, ...): load the pointer stored in it and
            # follow it
            assert ev.ptr is not None
            ev = RuntimeVal(self._emit(mir.Load(ev.ptr, sval.to_mir_type(t))), t)
        if not isinstance(ev, RuntimeVal):
            raise CompileError('cannot access the fields of this value')
        value = ev.value
        type = ev.type
        if not isinstance(type, sval.PointerType):
            raise CompileError(
                f"cannot access fields of a {sval.type_str(type)} value: "
                'only struct values have fields'
            )
        elem = type.elem
        while isinstance(elem, sval.PointerType) and isinstance(elem.elem, sval.StructType):
            # the base is a pointer to a pointer to a struct (the address
            # of a pointer-valued field or variable): load the pointer
            # stored there before going on
            value = self._emit(mir.Load(value, sval.to_mir_type(elem)))
            elem = elem.elem
        if not isinstance(elem, sval.StructType):
            raise CompileError(
                f"cannot access fields of a {sval.type_str(type)} value: "
                'only struct values have fields'
            )
        return value, elem

    def _exec_field_addr(self, inst: hir.FieldAddr) -> InterpVal:
        """One step of an attribute chain on a struct value: the address
        of the field ``inst.name`` of the struct ``inst.base`` denotes
        (the base's pointer layers are auto-dereferenced here, see
        ``_struct_addr_of``)."""
        ptr, type = self._struct_addr_of(self._operand(inst.base))
        index = _field_index(type, inst.name)
        value = self._emit(mir.Gep(ptr, index))
        return RuntimeVal(value, sval.PointerType(type.fields[index].type, is_const=False))

    def _emit(self, inst: mir.Inst) -> mir.Value:
        self._regions[-1].append(inst)
        return inst

    # -- regions and runtime branches -----------------------------------------

    def _run_region(self, stmts: tuple[hir.Inst, ...]) -> tuple[list[mir.Inst], bool]:
        """Type and emit one region (a branch body of a runtime ``if``):
        a straight-line list that may itself contain runtime ``if``s.
        Returns the emitted instructions and whether the region returns
        on every path (i.e. never falls off its end)."""
        region: list[mir.Inst] = []
        self._regions.append(region)
        try:
            flow, _ = self._run_list(stmts)
        finally:
            self._regions.pop()
        return region, flow is Flow.RET

    def _exec_runtime_if(self, inst: hir.If, cond: InterpVal) -> tuple[Flow, InterpVal | None]:
        """A runtime ``if``: both branch bodies are typed and emitted as
        regions of a :class:`mir.If`.  A branch that returns ends its
        path; a branch that falls off continues with the code after the
        ``if``.  Both branches falling through (a join) is not supported
        yet, and neither are runtime branches inside inlined functions."""
        if not self._in_function_proper():
            raise CompileError(
                "runtime 'if' inside inlined functions is not supported yet"
            )
        if not isinstance(cond, RuntimeVal) or cond.type != sval.BoolType():
            raise CompileError('runtime if conditions must be boolean values')
        then_body, then_returns = self._run_region(inst.then_body)
        else_body, else_returns = self._run_region(inst.else_body)
        if not then_returns and not else_returns:
            raise CompileError(
                "runtime 'if' branches that both fall through are not supported yet"
            )
        self._emit(mir.If(cond.value, tuple(then_body), tuple(else_body)))
        if then_returns and else_returns:
            # every path returns: whatever follows in this region is dead
            return Flow.RET, None
        return Flow.FALL, None

    def _finish_void(self) -> None:
        """End one path of the function proper with a void return (a bare
        ``return``, or the implicit end of a void function body)."""
        target = self._ret_target
        if target is not None and target != sval.VoidType():
            raise CompileError(
                f"cannot return without a value where {sval.type_str(target)} is expected"
            )
        self._saw_void_return = True
        self._emit(mir.Ret(None))

    # -- helpers -------------------------------------------------------------

    def _coerce(self, ev: InterpVal, target: sval.Type) -> mir.Value:
        """Materialize a value of the spy type ``target``; numeric
        widening conversions (int -> float, float32 -> float64) are
        applied."""
        match ev:
            case ComptimeVal(obj):
                return _const_of_py(obj, target)
            case RuntimeVal(value, type):
                return self._convert(value, type, target)
            case _:
                raise CompileError('cannot materialize this value')

    def _convert(
        self, value: mir.Value, from_type: sval.Type, to_type: sval.Type
    ) -> mir.Value:
        if from_type == to_type:
            return value
        if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.IntType):
            if from_type.bits < to_type.bits:
                kind = 'sext' if from_type.signed else 'zext'
            else:
                kind = 'trunc'
            return self._emit(mir.Convert(kind, value, sval.to_mir_type(to_type)))
        if isinstance(from_type, sval.IntType) and isinstance(to_type, sval.FloatType):
            kind = 'sitofp' if from_type.signed else 'uitofp'
            return self._emit(mir.Convert(kind, value, sval.to_mir_type(to_type)))
        if isinstance(from_type, sval.FloatType) and isinstance(to_type, sval.FloatType):
            kind = 'fpext' if from_type.bits < to_type.bits else 'fptrunc'
            return self._emit(mir.Convert(kind, value, sval.to_mir_type(to_type)))
        raise CompileError(
            f"cannot convert a {sval.type_str(from_type)} value to {sval.type_str(to_type)}"
        )

    # -- operators ------------------------------------------------------------

    def _eval_binary(self, inst: hir.Binary) -> InterpVal:
        op = inst.op
        lhs = self._operand(inst.lhs)
        rhs = self._operand(inst.rhs)
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
            return ComptimeVal(_comptime_py_op(op, lhs.obj, rhs.obj))
        lt, rt = _bin_types(lhs, rhs)
        type = _binary_type(lt, rt, f"apply '{op}' to") if lt is not None and rt is not None else None
        if type is None:
            raise _unsupported_type_error(op, lt)
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

        lv = self._coerce(lhs, type)
        rv = self._coerce(rhs, type)
        signed = isinstance(type, sval.IntType) and type.signed
        value = self._emit(mir.Arith(_ARITH_OPS[op], signed, lv, rv, sval.to_mir_type(type)))
        return RuntimeVal(value, type)

    def _eval_cmp(self, inst: hir.Compare) -> InterpVal:
        op = inst.op
        lhs = self._operand(inst.lhs)
        rhs = self._operand(inst.rhs)
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
            return ComptimeVal(_comptime_py_op(op, lhs.obj, rhs.obj))
        lt, rt = _bin_types(lhs, rhs)
        type = _binary_type(lt, rt, 'compare') if lt is not None and rt is not None else None
        if type is None:
            raise _unsupported_type_error(op, lt)
        lv = self._coerce(lhs, type)
        rv = self._coerce(rhs, type)
        kind = 'int' if isinstance(type, sval.IntType) else 'float'
        signed = isinstance(type, sval.IntType) and type.signed
        value = self._emit(mir.Cmp(_CMP_OPS[op], signed, kind, lv, rv))
        return RuntimeVal(value, sval.BoolType())

    def _eval_boolop(self, inst: hir.BoolOp) -> InterpVal:
        lhs = self._operand(inst.lhs)
        rhs = self._operand(inst.rhs)
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
            if inst.op == 'and':
                return ComptimeVal(lhs.obj and rhs.obj)
            return ComptimeVal(lhs.obj or rhs.obj)
        raise CompileError(
            f"'{inst.op}' between runtime values is not supported yet "
            "(only compile-time operands)"
        )

    def _eval_unary(self, inst: hir.Unary) -> InterpVal:
        op = inst.op
        operand = self._operand(inst.operand)
        if isinstance(operand, ComptimeVal):
            obj = operand.obj
            if op == 'not':
                return ComptimeVal(not obj)
            if op == 'neg':
                try:
                    return ComptimeVal(-obj)
                except Exception as e:
                    raise CompileError(f"cannot negate {obj!r} at compile time: {e}") from e
            raise CompileError(f"unsupported unary operator '{op}'")
        type = _type_of(operand)
        if type is None:
            raise CompileError(f"cannot apply unary '{op}' to a compile-time object")
        value = self._coerce(operand, type)
        if op == 'not':
            if type != sval.BoolType():
                raise CompileError(f"cannot apply 'not' to a {sval.type_str(type)} value")
            one = mir.BoolValue(True)
            return RuntimeVal(
                self._emit(mir.Arith('xor', False, value, one, sval.to_mir_type(type))),
                sval.BoolType(),
            )
        if op == 'neg':
            if isinstance(type, sval.FloatType):
                zero = mir.FloatValue(0.0, type.bits)
                return RuntimeVal(
                    self._emit(mir.Arith('sub', False, zero, value, sval.to_mir_type(type))), type
                )
            if isinstance(type, sval.IntType):
                zero = mir.IntValue(0, type.bits, type.signed)
                return RuntimeVal(
                    self._emit(mir.Arith('sub', False, zero, value, sval.to_mir_type(type))), type
                )
            raise CompileError(f"cannot negate a {sval.type_str(type)} value")
        raise CompileError(f"unsupported unary operator '{op}'")

    # -- calls ----------------------------------------------------------------

    def _exec_call_inplace(self, inst: hir.CallInplace) -> None:
        """Run one call whose result is written into the result location
        ``inst.ret`` (RLS).  The call is dispatched like a by-value call
        and its result is handed to the slot: scalar and compile-time
        results are only *recorded* in the slot (no memory, no extra
        MIR - the matching ``Load`` passes the recorded value on); a
        slot that already has memory receives a real store; a struct
        value is materialized into the slot (its address may escape);
        and a constructor - or a call whose result goes through a result
        pointer - writes into the slot itself (in place), so no result
        is handed back at all."""
        callee = self._operand(inst.callee)
        ev = self._dispatch_call(callee, inst)
        self._store_result(ev, inst.ret)

    def _store_result(self, ev: InterpVal, ret: hir.Value) -> None:
        """The RLS tail of a call: hand the returned value of a call to
        its result location (see ``_exec_call_inplace``)."""
        slot = self._operand(ret)
        if isinstance(slot, RetLocVal):
            # the call is the expression of a ``return`` statement: its
            # result goes into the function result location
            if isinstance(ev, InPlaceResult):
                # the callee (a constructor, or a call whose result goes
                # through a result pointer) wrote into the location itself
                if self._in_function_proper() and slot.type is not None:
                    self._note_inplace_ret(slot.type)
                return
            if self._in_function_proper():
                self._write_result(ev)
            else:
                # an inlined body: the return value is only recorded and
                # yielded at its ``Ret``
                slot.value = ev
            return
        if isinstance(ev, InPlaceResult):
            # the callee (a constructor, or a call whose result goes
            # through a result pointer) wrote the result into the result
            # location itself
            return
        if isinstance(slot, PendingSlot):
            if slot.ptr is None:
                if isinstance(ev, RuntimeVal) and isinstance(ev.type, sval.StructType):
                    # a struct call result needs real memory (its address
                    # may escape: fields are accessed and values passed on)
                    struct = ev.type
                    ptr = self._materialize_location(slot, struct)
                    self._emit(mir.Store(ptr, ev.value))
                    return
                slot.value = ev
                return
            # the slot already has memory (its address escaped or it was
            # stored before): write the call result into it
            assert slot.type is not None
            v = _to_slot(ev, slot.type)
            self._emit(mir.Store(slot.ptr, v))
            return
        if isinstance(slot, RuntimeVal):
            ptype = slot.type
            if not isinstance(ptype, sval.PointerType):
                raise CompileError(
                    f"cannot write a call result through a {sval.type_str(ptype)} value"
                )
            v = _to_slot(ev, ptype.elem)
            self._emit(mir.Store(slot.value, v))
            return
        raise CompileError('cannot write a call result into a compile-time location')

    def _dispatch_call(self, callee: InterpVal, inst: hir.CallInplace) -> InterpVal:
        """Resolve one call by its callee value and run it, returning its
        value.  Spy functions compile to a native ``call`` producing a
        typed register, plain Python functions are inlined, and the spy
        builtins are evaluated at compile time.  The callee constant of a
        registered spy function already resolved to its entry when the
        callee operand was evaluated (see ``_operand``)."""
        assert not isinstance(callee, ComptimeVal), "values cannot be called at compile time"
        if not isinstance(callee, ComptimeRefVal):
            raise CompileError(
                "calls through runtime function values are not supported yet"
            )
        obj = callee.obj
        if obj is spy_builtins.spy_typeof:
            return self._call_builtin_type(inst)
        if obj is spy_builtins.spy_compile_log:
            return self._call_builtin_compile_log(inst)
        if obj is spy_builtins.spy_as:
            raise CompileError(
                "spy.as can only be used at the Python call boundary, not inside spy functions"
            )
        if isinstance(obj, FunctionEntry):
            return self._call_entry(obj, inst, [self._operand(a) for a in inst.args])
        if isinstance(obj, sval.StructType):
            # a constructor ``Bar(...)``
            return self._call_constructor(obj, inst)
        if isinstance(obj, pytypes.FunctionType):
            return self._inline_plain(obj, [self._operand(a) for a in inst.args], 'function')
        raise CompileError(
            f"cannot compile a call to {obj!r}; only spy functions, plain Python "
            "functions and the spy builtins can be called"
        )

    def _call_builtin_type(self, inst: hir.CallInplace) -> InterpVal:
        if len(inst.args) != 1:
            raise CompileError('spy.typeof takes exactly one argument')
        arg = self._operand(inst.args[0])
        match arg:
            case ComptimeVal(obj):
                type = sval.value_type(obj)
            case RuntimeVal(_, type):
                pass
            case _:
                type = None
        if type is None:
            raise CompileError(
                f"spy.typeof of the compile-time value {_describe(arg)} is not supported"
            )
        return ComptimeVal(type)

    def _call_builtin_compile_log(self, inst: hir.CallInplace) -> InterpVal:
        objs: list[Any] = []
        for arg in inst.args:
            ev = self._operand(arg)
            if not isinstance(ev, ComptimeVal):
                raise CompileError(
                    'spy.compile_log arguments must be compile-time constants'
                )
            objs.append(ev.obj)
        print(*objs)  # a compile-time log, exactly like spy.compile_log
        return ComptimeVal(None)

    # -- struct constructors and methods ----------------------------------------

    def _materialize_location(self, loc: InterpVal, type: sval.Type) -> mir.Value:
        """The address one call writes a result of spy type ``type``
        into - the result location of the enclosing statement: a slot
        that is given the memory of the type when it has none yet (a
        slot that already holds one is reused), or the result pointer of
        a result-pointer function.  The type is an aggregate (a struct
        today, arrays and others later); scalars never materialize."""
        if isinstance(loc, PendingSlot):
            if loc.ptr is None:
                if loc.value is not None:
                    # the slot only recorded a scalar call result that was
                    # never materialized: it is discarded by this assignment
                    loc.value = None
                loc.ptr = self._emit(mir.Alloca(mir.PointerType(sval.to_mir_type(type))))
                loc.type = type
            elif loc.type is None or loc.type != type:
                raise CompileError(
                    f'cannot write a {sval.type_str(type)} value into a slot that '
                    f'already holds a {sval.type_str(loc.type)} value'  # type: ignore[arg-type]
                )
            return loc.ptr
        if isinstance(loc, RetLocVal):
            if loc.ptr is None:
                if self._in_function_proper() and self._result_mode is None and sval.returns_via_result_ptr(type):
                    # the function proper turns out to return this value
                    # through a result pointer: return through it instead
                    # of an extra local copy
                    assert self._fn is not None
                    self._bind_result_ptr(self._fn, type)
                    assert loc.ptr is not None
                    return loc.ptr
                # a direct-return function (or an inlined body) returning
                # a value written in place: give the location memory
                loc.ptr = self._emit(mir.Alloca(mir.PointerType(sval.to_mir_type(type))))
                loc.type = type
                loc.value = None
            elif loc.type is None or loc.type != type:
                raise CompileError(
                    f'cannot write a {sval.type_str(type)} value into the result '
                    f'location that already holds a {sval.type_str(loc.type)} value'  # type: ignore[arg-type]
                )
            return loc.ptr
        raise CompileError(f'cannot write a {sval.type_str(type)} value into this location')

    def _call_result_addr(self, ret: hir.Value, type: sval.Type) -> mir.Value:
        """The address a call whose result is delivered through a result
        pointer (a function returning a large aggregate) writes into: its
        result location."""
        loc = self._operand(ret)
        if isinstance(loc, RuntimeVal):
            ptype = loc.type
            if not isinstance(ptype, sval.PointerType) or ptype.elem != type:
                raise CompileError(
                    f'cannot write a {sval.type_str(type)} value through a '
                    f'{sval.type_str(ptype)} pointer'
                )
            return loc.value
        return self._materialize_location(loc, type)

    def _call_constructor(self, desc: sval.StructType, inst: hir.CallInplace) -> InterpVal:
        """A struct constructor ``Bar(a, b)``: the result slot receives a
        new struct value.  With a user ``__init__`` the call is dispatched
        to it with ``self`` pointing at the result slot; otherwise every
        argument is written into the field of the same declaration index
        (the default constructor).  Either way the value is written into
        the result location in place - no value is handed back."""
        struct = desc
        ret = self._operand(inst.ret)
        if not isinstance(ret, (PendingSlot, RetLocVal)):
            raise CompileError('a constructor must write into a variable slot')
        ptr = self._materialize_location(ret, struct)
        init = self._resolver.resolve_method(desc, '__init__')
        if init is not None:
            # a user-provided ``__init__``: a method whose ``self`` is
            # always the address of the result location (a constructor
            # writes its fields in place) - it is called like any other
            # method, with that address prepended as the first argument
            target, _ = init
            self_ev = RuntimeVal(ptr, sval.PointerType(struct, is_const=False))
            evals: list[InterpVal] = [self_ev]
            evals.extend(self._operand(a) for a in inst.args)
            if isinstance(target, FunctionEntry):
                self._call_entry(target, inst, evals)
            else:
                self._inline_plain(target, evals, 'method')
            return InPlaceResult()
        fields = desc.fields
        if len(inst.args) != len(fields):
            raise CompileError(
                f"constructor {desc.name} takes {len(fields)} arguments "
                f"(one per field), got {len(inst.args)}"
            )
        for i, field in enumerate(fields):
            ev = self._operand(inst.args[i])
            value = _materialize_arg(
                ev, field.type, f"the '{field.name}' argument of {desc.name}"
            )
            field_ptr = self._emit(mir.Gep(ptr, i))
            self._emit(mir.Store(field_ptr, value))
        return InPlaceResult()

    def _exec_call_method(self, inst: hir.CallMethodInplace) -> None:
        """A method call ``x.h(...)``: a method is an ordinary function
        whose first parameter is the struct type of ``x`` (by value) or a
        pointer to it (a ``ptr_self`` method) - the call is run like any
        other call, with the base prepended as that first argument (its
        static type, compared with the callee's, decides whether the
        struct value or its address is passed)."""
        addr, struct = self._struct_addr_of(self._operand(inst.base))
        target = self._resolver.resolve_method(struct, inst.name)
        if target is None:
            raise CompileError(f"type {struct.name} has no method named '{inst.name}'")
        method, ptr_self = target
        self_type = self._self_type(method, ptr_self, struct)
        evals = [self._self_value(self_type, struct, addr)]
        evals.extend(self._operand(a) for a in inst.args)
        if isinstance(method, FunctionEntry):
            ev = self._call_entry(method, inst, evals)
        else:
            ev = self._inline_plain(method, evals, 'method')
        # the method's result lands in the result location like any call
        self._store_result(ev, inst.ret)

    def _self_type(
        self, method: Any, ptr_self: bool, struct: sval.StructType
    ) -> sval.Type:
        """The spy type of the first (``self``) parameter of the method
        ``method`` of ``struct``, as the callee declares it: an aot
        method's signature carries it (the struct itself, or a pointer to
        it for ``ptr_self``); a jit method and a plain (undecorated)
        method have no fixed signature, and their convention is the
        registered ``ptr_self`` flag (a plain method's ``self`` is always
        by value)."""
        if isinstance(method, FunctionValue):
            assert len(method.args) > 0, 'internal error: method entry has no self parameter'
            return method.args[0].type
        if ptr_self:
            return sval.PointerType(struct, is_const=False)
        return struct

    def _self_value(
        self, type: sval.Type, struct: sval.StructType, addr: mir.Value
    ) -> InterpVal:
        """The ``self`` argument of a method call, presented as the
        callee's first parameter declares it: by value (the parameter is
        the struct itself) it is the struct loaded from its address, by
        pointer it is the address itself."""
        if type == struct:
            value = self._emit(mir.Load(addr, sval.struct_mir_type(struct)))
            return RuntimeVal(value, struct)
        assert isinstance(type, sval.PointerType) and type.elem == struct
        return RuntimeVal(addr, type)

    def _call_entry(
        self,
        entry: FunctionEntry,
        inst: hir.CallInplace | hir.CallMethodInplace,
        evals: list[InterpVal],
    ) -> InterpVal:
        """A native call of a registered spy function (``@aot`` or
        ``@jit``) with the given (already evaluated) argument values -
        the common tail of an ordinary function call and of a method
        call, whose ``self`` the caller prepended to the arguments.  An
        aot function's signature is fixed by its entry's formals (a
        method's un-annotated ``self`` is typed there, see
        ``dsl._method_args``); a jit function solves the parameter types
        from the provided arguments."""
        if entry.context is not self._resolver:
            raise CompileError(
                f"cannot call function {entry.fn.__name__} from another JitContext"
            )
        fn_ir = entry.hir
        if isinstance(entry, FunctionValue):
            formal = tuple(a.type for a in entry.args)
            if len(evals) > len(formal):
                raise CompileError(
                    f'function {fn_ir.name} takes {len(formal)} arguments, '
                    f'got {len(evals)}'
                )
            values = _convert_evals(fn_ir, evals, formal)
        else:
            formal = self._solve_types(fn_ir, evals, 'jit')
            values = _convert_evals(fn_ir, evals, formal)
        callee, ret_type, info = self._resolver.resolve_call(entry, formal)
        return self._emit_native_call(callee, inst, ret_type, info, values)

    def _emit_native_call(
        self,
        callee: mir.Value,
        inst: hir.CallInplace | hir.CallMethodInplace,
        ret_type: sval.Type,
        info: sval.FunctionCallInfo,
        values: tuple[mir.Value, ...],
    ) -> InterpVal:
        """Emit one native call from its lowering plan - a
        :class:`FunctionCallInfo` the host derived from the callee's
        spy function type (see ``type.function_call_info``): the
        by-value arguments are placed onto their lowered MIR positions,
        and the result convention of the plan decides whether the call
        returns a value or writes the result into the result location
        through a trailing result pointer."""
        assert len(values) == len(info.args_map)
        # the lowered MIR argument list: every position is filled either
        # by a by-value argument or by the result-location pointer
        placed: list[mir.Value | None] = [None] * info.total_mir_args
        for i, mapped in enumerate(info.args_map):
            if mapped is not None:
                assert placed[mapped.index] is None
                placed[mapped.index] = values[i]
        if isinstance(info.return_info, sval.FunctionRetLocReturnInfo):
            # the callee writes the result into the result location,
            # whose address is passed as the trailing MIR argument
            placed[info.return_info.arg_index] = self._call_result_addr(inst.ret, ret_type)
        for arg in placed:
            assert arg is not None
        args = cast(tuple[mir.Value, ...], tuple(placed))
        if isinstance(info.return_info, sval.FunctionRetLocReturnInfo):
            self._emit(mir.Call(callee, args, mir.VoidType()))
            return InPlaceResult()
        assert isinstance(info.return_info, sval.FunctionValueReturnInfo)
        value = self._emit(mir.Call(callee, args, info.return_info.mir_type))
        if ret_type == sval.VoidType():
            # a void call produces no value: it only has effects
            return ComptimeVal(None)
        return RuntimeVal(value, ret_type)

    def _inline_plain(
        self, fn: Any, evals: list[InterpVal], what: str
    ) -> InterpVal:
        """A plain Python function - a helper, or the plain (undecorated)
        method of a struct (``what`` is 'function' or 'method', used in
        the error messages) - is inlined into the current stream like any
        plain Python function (its body may only use what inlining
        supports)."""
        fn_ir = self._resolver.hir_of_plain_fn(fn)
        if any(f.fn is fn for f in self._inline_stack):
            if what == 'method':
                raise CompileError(
                    f'a plain Python method cannot call itself recursively '
                    f'(method {fn_ir.name}) - declare it with @aot/@jit instead'
                )
            raise CompileError(
                f'a plain Python function cannot call itself recursively '
                f'(function {fn_ir.name}); plain functions are inlined, and a '
                'recursive inline would never finish compiling - declare it '
                'as a spy function instead'
            )
        if self._inline_depth >= _MAX_INLINE_DEPTH:
            raise CompileError('too deeply nested inlined functions')
        formal = self._solve_types(fn_ir, evals, 'jit')
        values = _convert_evals(fn_ir, evals, formal)
        return self._run_inline(fn_ir, values, formal)

    def _solve_types(
        self, fn_ir: astgen.FunctionIR, evals: list[InterpVal], mode: str
    ) -> tuple[sval.Type, ...]:
        """The concrete spy types of all formal parameters of one call,
        solved from the provided arguments (defaults included), plus the
        argument count check."""
        if len(evals) > len(fn_ir.params):
            raise CompileError(
                f"function {fn_ir.name} takes {len(fn_ir.params)} arguments, "
                f"got {len(evals)}"
            )
        provided: list[sval.Type | None] = [None] * len(fn_ir.params)
        for i, ev in enumerate(evals):
            match ev:
                case ComptimeVal(obj):
                    t = sval.value_type(obj)
                case RuntimeVal(_, type):
                    t = type
                case _:
                    t = None
            if t is None:
                raise CompileError(
                    f"cannot pass the compile-time value {_describe(ev)} "
                    f"as an argument of function {fn_ir.name}"
                )
            provided[i] = t
        try:
            param_types, _ = astgen.solve_call_types(fn_ir, mode, tuple(provided))
            return param_types
        except TypeMismatchError as e:
            raise CompileError(str(e)) from e

    def _run_inline(
        self,
        fn_ir: astgen.FunctionIR,
        values: tuple[mir.Value, ...],
        formal: tuple[sval.Type, ...],
    ) -> InterpVal:
        """Run the body of an inlined plain function with the given
        (already materialized) argument values, returning its result: the
        value of its ``return``, or ``None`` for a void body."""
        self._inline_stack.append(fn_ir)
        self._inline_depth += 1
        self._push_frame(tuple(zip(values, formal)), fn_ir.ret_loc)
        try:
            flow, value = self._run_list(fn_ir.body)
        finally:
            self._frames.pop()
            self._inline_depth -= 1
            self._inline_stack.pop()
        if flow is not Flow.RET:
            # the inlined body fell off its end: a void inline
            return ComptimeVal(None)
        assert isinstance(value, InterpVal)
        return value

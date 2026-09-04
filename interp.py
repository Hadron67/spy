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
the runtime operation expects.  Runtime values carry the static types
of the MIR itself (``mir``); the interpreter computes in the ``spy``
types of ``type.py`` and mirrors them into MIR types (``to_mir_type`` /
``to_spy_type``) at the points where runtime instructions are emitted
or read.

Calls are dispatched at compile time:

* calls to the ``spy`` builtins (``spy.type``, ``spy.compile_log``) are
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
from typing import Any

from . import astgen, hir, mir
from . import builtins as spy_builtins
from .errors import CompileError, TypeMismatchError
from .fn import FunctionEntry, FunctionValue
from .info import FunctionResolver
from .mir import (
    BoolType,
    BoolValue,
    FloatType,
    FloatValue,
    IntType,
    IntValue,
    PointerType,
    StructType,
    Type,
    VoidType,
    type_str,
)
from .type import (
    BoolType as SpyBoolType,
)
from .type import (
    FloatType as SpyFloatType,
)
from .type import (
    FormalArg as SpyFormalArg,
)
from .type import (
    FunctionType as SpyFunctionType,
)
from .type import (
    IntType as SpyIntType,
)
from .type import (
    PointerType as SpyPointerType,
)
from .type import (
    StructType as SpyStructType,
)
from .type import (
    Type as SpyType,
)
from .type import (
    VoidType as SpyVoidType,
)
from .type import (
    value_type,
)

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
    value: mir.Value


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

    type: Type | None = None
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
    type: Type | None = None


@dataclass
class Frame:
    """The by-value arguments of the function whose body is currently
    being executed (resolved by ``hir.Arg`` leaves)."""

    arg_values: tuple[mir.Value, ...]


class Flow(IntEnum):
    """How executing a straight-line list of instructions ended: the
    list ``FALL`` off its end, or was cut short by a ``return``
    (``RET``) - the only case that carries a returned value."""

    FALL = auto()
    RET = auto()


def typeof(value: mir.Value) -> Type:
    return value.type  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# mirrors between the spy types (``type.py``) and the MIR types (``mir``)
# ---------------------------------------------------------------------------


def to_mir_type(type: SpyType) -> Type:
    """The MIR mirror of a spy type: the static type the runtime register
    of a value of ``type`` has.  The mapping is one-to-one over the types
    that can cross into runtime code.  A spy struct type mirrors to one
    :class:`mir.StructType` object (created lazily and cached on the
    descriptor), so that all values of one struct share one identity."""
    match type:
        case SpyBoolType():
            return BoolType()
        case SpyIntType():
            return IntType(type.bits, type.signed)
        case SpyFloatType():
            return FloatType(type.bits)
        case SpyVoidType():
            return VoidType()
        case SpyStructType():
            return struct_mir_type(type)
        case SpyPointerType(elem, _):
            # const-ness is not tracked in the MIR
            return PointerType(to_mir_type(elem))
        case SpyFunctionType(args, ret):
            return mir.FunctionType(tuple(to_mir_type(a.type) for a in args), to_mir_type(ret))
        case _:
            raise CompileError(f"spy type {type!r} has no MIR representation")


def struct_mir_type(type: SpyStructType) -> StructType:
    """The (cached) MIR mirror of one spy struct type: fields in
    declaration order, mirroring the LLVM layout of the struct."""
    ret = type._mir
    if ret is not None:
        return ret
    fields: list[mir.FormalArg] = []
    for field in type.fields:
        fields.append(mir.FormalArg(field.name, to_mir_type(field.type)))
    ret = StructType(type, tuple(fields))
    ret.ctype = type._py_cls
    type._mir = ret
    return ret


def to_spy_type(type: Type) -> SpyType:
    """The spy type a MIR type mirrors (the inverse of
    :func:`to_mir_type` on its accepted types)."""
    match type:
        case BoolType():
            return SpyBoolType()
        case IntType():
            return SpyIntType(type.bits, type.signed)
        case FloatType():
            return SpyFloatType(type.bits)
        case VoidType():
            return SpyVoidType()
        case PointerType(elem):
            return SpyPointerType(to_spy_type(elem), is_const=False)
        case StructType():
            return type.spy_type
        case mir.FunctionType(args, ret):
            return SpyFunctionType(
                tuple(SpyFormalArg('', to_spy_type(a)) for a in args), to_spy_type(ret)
            )
        case _:
            raise CompileError(f"MIR type {type_str(type)} has no spy counterpart")


# ---------------------------------------------------------------------------
# stateless helpers of the interpreter: pure functions over their arguments
# (argument/prototype construction, Python-literal constants, compile-time
# operators and operator error messages) - none of them uses instance state,
# so none of them is a method of :class:`HirRunner`
# ---------------------------------------------------------------------------


def _field_index(type: StructType, name: str) -> int:
    """The declaration index of the field ``name`` of a struct type."""
    for i, field in enumerate(type.fields):
        if field.name == name:
            return i
    raise CompileError(
        f"type {type_str(type)} has no field named '{name}'"
    )


def _const_of_py(obj: Any, type: Type) -> mir.Value:
    """Turn a Python literal into a typed MIR constant."""
    match type:
        case BoolType():
            if not isinstance(obj, bool):
                raise CompileError(f"cannot use {obj!r} as a bool constant")
            return BoolValue(obj)
        case IntType():
            if isinstance(obj, bool) or not isinstance(obj, int):
                raise CompileError(f"cannot use {obj!r} as an integer constant")
            if type.signed:
                lo, hi = (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
            else:
                lo, hi = (0, 2 ** type.bits - 1)
            if not lo <= obj <= hi:
                raise CompileError(
                    f"integer constant {obj} is out of range for {type_str(type)}"
                )
            return IntValue(obj, type.bits, type.signed)
        case FloatType():
            if isinstance(obj, bool) or not isinstance(obj, (int, float)):
                raise CompileError(f"cannot use {obj!r} as a float constant")
            return FloatValue(float(obj), type.bits)
        case _:
            raise CompileError(
                f"cannot create a constant of type {type_str(type)} from {obj!r}"
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


def _binary_type(op: str, lt: Type, rt: Type, what: str) -> Type | None:
    """The type a binary operation on ``lt``/``rt`` is performed on, or
    None if the operand combination is not a number pair."""
    if isinstance(lt, IntType) and isinstance(rt, IntType):
        if lt != rt:
            raise CompileError(
                f"cannot {what} a {type_str(lt)} value with a {type_str(rt)} value "
                "(different integer types)"
            )
        return lt
    if isinstance(lt, FloatType) and isinstance(rt, FloatType):
        return FloatType(max(lt.bits, rt.bits))
    if isinstance(lt, IntType) and isinstance(rt, FloatType):
        return rt
    if isinstance(lt, FloatType) and isinstance(rt, IntType):
        return lt
    return None


def _unsupported_type_error(op: str, type: Type | None) -> CompileError:
    if isinstance(type, PointerType) and isinstance(type.elem, IntType) and type.elem.bits == 8:
        return CompileError(
            f"cannot apply '{op}' to string values "
            "(strings are compiled as arrays of u8)"
        )
    if isinstance(type, PointerType):
        return CompileError(f"cannot apply '{op}' to pointer values")
    if type is None:
        return CompileError(f"cannot apply '{op}' to a compile-time object")
    return CompileError(f"cannot apply '{op}' to {type_str(type)} values")



def _to_runtime(ev: InterpVal, target: Type | None) -> mir.Value:
    """Materialize a return value: runtime values must already have
    the target type, compile-time values adopt it (or, without a
    target, their Python type mapping)."""
    if isinstance(ev, RuntimeVal):
        value = ev.value
        t = typeof(value)
        if target is not None and t != target:
            raise CompileError(
                f"cannot return a {type_str(t)} value where {type_str(target)} "
                "is expected"
            )
        return value
    if isinstance(ev, ComptimeVal):
        if ev.obj is None:
            raise CompileError("cannot return None (functions must return a value)")
        if target is not None:
            return _const_of_py(ev.obj, target)
        t = value_type(ev.obj)
        if t is None:
            raise CompileError(f"cannot return the compile-time value {ev.obj!r}")
        return _const_of_py(ev.obj, to_mir_type(t))
    raise CompileError('cannot return this value')


def _to_slot(ev: InterpVal, type: Type) -> mir.Value:
    """Materialize ``ev`` as a value of exactly ``type`` for a store
    into an already-typed slot (the strict sibling of
    ``_to_runtime``, whose messages talk about stores)."""
    match ev:
        case RuntimeVal(value):
            if typeof(value) != type:
                raise CompileError(
                    f"cannot store a {type_str(typeof(value))} value into a "
                    f"slot of type {type_str(type)}"
                )
            return value
        case ComptimeVal(obj):
            return _const_of_py(obj, type)
        case _:
            raise CompileError(
                f"cannot store this value into a slot of type {type_str(type)}"
            )


def _convert_evals(
    fn_ir: astgen.FunctionIR,
    evals: list[InterpVal],
    formal: tuple[SpyType, ...],
) -> tuple[mir.Value, ...]:
    """Materialize the (possibly defaulted) arguments of one call as
    values of the given formal types."""
    formal_types = tuple(to_mir_type(t) for t in formal)
    values: list[mir.Value] = []
    for i, param in enumerate(fn_ir.params):
        if i < len(evals):
            ev = evals[i]
            if isinstance(ev, ComptimeVal):
                values.append(_const_of_py(ev.obj, formal_types[i]))
            elif isinstance(ev, RuntimeVal):
                value = ev.value
                if typeof(value) != formal_types[i]:
                    raise CompileError(
                        f"cannot pass a {type_str(typeof(value))} value as the "
                        f"'{param.name}' argument of function {fn_ir.name} "
                        f"(expected {type_str(formal_types[i])})"
                    )
                values.append(value)
            else:
                raise CompileError('cannot pass this value as an argument')
        else:
            assert param.has_default
            values.append(_const_of_py(param.default_value, formal_types[i]))
    return tuple(values)


def _materialize_arg(ev: InterpVal, target: Type, what: str) -> mir.Value:
    """Materialize one argument value of exactly the type ``target``
    (used by constructors, whose parameters are the struct fields)."""
    match ev:
        case ComptimeVal(obj):
            return _const_of_py(obj, target)
        case RuntimeVal(value):
            if typeof(value) != target:
                raise CompileError(
                    f"cannot pass a {type_str(typeof(value))} value as {what} "
                    f"(expected {type_str(target)})"
                )
            return value
        case _:
            raise CompileError(f'cannot pass this value as {what}')


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
        self._frames: list[Frame] = []
        self._regs: dict[hir.Inst, InterpVal] = {}
        self._inline_stack: list[astgen.FunctionIR] = []
        self._inline_depth = 0
        # the function proper whose body is currently being typed (see
        # ``_bind_result_ptr``)
        self._fn: mir.Function | None = None
        self._ret_type: Type | None = None
        self._ret_target: Type | None = None
        # True when a path of the function proper ended in a void return
        # (a bare ``return``, or a fall-off of a void function)
        self._saw_void_return = False
        # True while typing the body of the function proper (a ``Ret``
        # emits a typed return); False inside an inlined plain function,
        # whose return just yields a value to the caller
        self._ret_emit = True
        # the emission regions (see ``_emit``): a stack of lists whose
        # top is the region currently being typed
        self._regions: list[list[mir.Inst]] = []
        # the result locations of the function bodies being executed (see
        # ``RetLocVal``), keyed by their ``hir.ResultLoc`` leaf: the top
        # is the body whose instructions are currently being typed
        self._ret_locs: list[tuple[hir.ResultLoc, RetLocVal]] = []
        # the return type of the function proper declared by its
        # annotation (its logical MIR type, or None when it is inferred
        # from the body); the target every return site is checked against
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

    def run_function(self, fn: mir.Function, fn_ir: astgen.FunctionIR) -> None:
        """Type the body of ``fn_ir`` into ``fn``.

        The host created ``fn`` (fixing its name, arguments and, when
        one is declared, its return type) and registered it *before*
        running the body, so a call the body makes to the function
        itself - recursion - resolves to ``fn``, whose signature is
        already fixed.  This fills ``fn.insts`` and fixes ``fn.ret_type``:
        the declared type, or the type inferred from the return sites
        when none is declared.

        The return convention of the function is decided here, from its
        logical return type (see ``mir.via_result_ptr``), and
        lowered into the MIR signature of ``fn``: a function that
        returns through a result pointer gets a trailing result pointer
        formal appended to ``fn.args`` and a ``void`` return, and its
        return type is kept in ``fn.result_type``.
        """
        self._fn = fn
        # push the result location of the function's return statements
        self._ret_locs.append((fn_ir.ret_loc, RetLocVal()))
        self._ret_target = fn.ret_type
        self._result_mode = None
        self._result_ptr = None
        self._ret_written = False
        declared = fn.ret_type
        if declared is not None and mir.via_result_ptr(declared):
            # the declared return type is delivered through a result
            # pointer: lower the signature before the body is typed, so
            # that recursive calls the body makes see the final form
            assert isinstance(declared, StructType)
            self._bind_result_ptr(fn, declared)
        param_values = tuple(
            mir.Param(i, arg.type, arg.name) for i, arg in enumerate(fn.args)
        )
        self._push_frame(param_values)
        self._ret_type = None
        self._saw_void_return = False
        self._ret_emit = True
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
            if declared is not None and declared != VoidType():
                raise CompileError(
                    f"function {fn_ir.name} must end with a 'return' statement"
                )
            self._finish_void()
            ret_type = VoidType()
        elif self._saw_void_return:
            if self._ret_type is not None:
                raise CompileError(
                    f"function {fn_ir.name} returns a value on some paths but "
                    'returns without a value on others'
                )
            ret_type = VoidType()
        else:
            ret_type = self._ret_type
            assert ret_type is not None
            if declared is not None:
                # a declared return type is enforced at the return sites
                # (see ``_write_result``), so the two must agree
                assert declared == ret_type
        self._ret_locs.pop()
        if self._result_mode == 'ptr':
            # a result-pointer function: its MIR signature was lowered to
            # a trailing result pointer formal and a void return when the
            # mode was bound (see ``_bind_result_ptr``)
            fn.ret_type = VoidType()
            return
        fn.ret_type = ret_type

    def _bind_result_ptr(self, fn: mir.Function, logical: StructType) -> None:
        """Lower the signature of the function proper to its result
        pointer form: append the trailing result pointer formal and fix
        the return type to void.  ``logical`` is the type of the value
        the function returns (kept in ``fn.result_type``)."""
        index = len(fn.args)
        formal = mir.FormalArg('$result', PointerType(logical))
        fn.args = fn.args + (formal,)
        fn.result_type = logical
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
        assert len(self._ret_locs) > 0, 'internal error: no function result location'
        return self._ret_locs[-1][1]

    def _write_result(self, ev: InterpVal) -> None:
        """The value of one return expression of the function proper is
        written into its result location: a direct-return function
        records it (the terminating ``Ret`` turns it into the return
        value), a result-pointer function stores it through its result
        pointer.  This is where every return site is typed (against the
        declared return type) and cross-path consistency is checked."""
        retloc = self._result_loc_of()
        target = self._ret_target
        if target is VoidType():
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
        value = _to_runtime(ev, target)
        t = typeof(value)
        if self._ret_type is not None and self._ret_type != t:
            raise CompileError(
                f"function returns values of conflicting types "
                f"{type_str(self._ret_type)} and {type_str(t)}"
            )
        self._ret_type = t
        if self._result_mode is None:
            # the return type is inferred from this site: decide the
            # return convention from it
            if mir.via_result_ptr(t):
                assert isinstance(t, StructType)
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
        retloc.value = RuntimeVal(value)
        self._ret_written = True

    def _note_inplace_ret(self, type: Type) -> None:
        """A return-path write that happened in place (a constructor or
        a result-pointer callee wrote straight into the result location):
        the cross-path return-type bookkeeping, without a value."""
        if self._ret_type is not None and self._ret_type != type:
            raise CompileError(
                f"function returns values of conflicting types "
                f"{type_str(self._ret_type)} and {type_str(type)}"
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
                ptr = self._result_ptr
                assert ptr is not None
                logical = typeof(ptr)
                assert isinstance(logical, PointerType)
                raise CompileError(
                    'cannot return without a value where '
                    f'{type_str(logical.elem)} is expected'
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
            value = self._emit(mir.Load(retloc.ptr, retloc.type))
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
                if not self._ret_emit:
                    # an inlined callee ends: yield the value its return
                    # statements wrote into its result location (the raw
                    # return expression value, or None for a void body)
                    retloc = self._result_loc_of()
                    if retloc.ptr is not None:
                        # the value was written in place (a constructor):
                        # load it back to yield it
                        assert retloc.type is not None
                        value: InterpVal | None = RuntimeVal(
                            self._emit(mir.Load(retloc.ptr, retloc.type))
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
                return RuntimeVal(frame.arg_values[index])
            case hir.ResultLoc():
                # the result location of the innermost body being typed
                # whose leaf this is (its own, or - during an inlined
                # call - the callee's)
                for leaf, retloc in reversed(self._ret_locs):
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

    def _push_frame(self, arg_values: tuple[mir.Value, ...]) -> None:
        self._frames.append(Frame(arg_values))

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
            return RuntimeVal(self._emit(mir.Load(ptr.ptr, ptr.type)))
        if isinstance(ptr, RuntimeVal):
            ptype = typeof(ptr.value)
            if not isinstance(ptype, PointerType):
                raise CompileError(f"cannot load from a {type_str(ptype)} value")
            return RuntimeVal(self._emit(mir.Load(ptr.value, ptype.elem)))
        raise CompileError('cannot load from a compile-time pointer')

    def _store(self, ptr: InterpVal, value: InterpVal) -> None:
        if isinstance(ptr, RetLocVal):
            # a store into the function result location (the expression
            # of a ``return`` statement that is not a call): a
            # direct-return function records the value of the path, a
            # result-pointer function stores it through its result
            # pointer
            if self._ret_emit:
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
                v0 = _to_runtime(recorded, None)
                t0 = typeof(v0)
                ptr.ptr = self._emit(mir.Alloca(PointerType(t0)))
                ptr.type = t0
                self._emit(mir.Store(ptr.ptr, v0))
            # the first store types the slot
            v = _to_runtime(value, None)
            t = typeof(v)
            if ptr.ptr is None:
                assert ptr.type is None
                ptr.ptr = self._emit(mir.Alloca(PointerType(t)))
                ptr.type = t
            elif ptr.type is None or ptr.type != t:
                raise CompileError(
                    f"cannot store a {type_str(t)} value into a slot of a "
                    f"different type"
                )
            self._emit(mir.Store(ptr.ptr, v))
            return
        if isinstance(ptr, RuntimeVal):
            ptype = typeof(ptr.value)
            if not isinstance(ptype, PointerType):
                raise CompileError(f"cannot store through a {type_str(ptype)} value")
            v = _to_runtime(value, ptype.elem)
            self._emit(mir.Store(ptr.value, v))
            return
        raise CompileError('cannot store through a compile-time pointer')

    # -- struct values ---------------------------------------------------------

    def _struct_addr_of(self, ev: InterpVal) -> tuple[mir.Value, StructType]:
        """The address of the struct value ``ev`` denotes, and the static
        struct type at that address.  A struct value lives in memory: a
        variable whose slot holds the struct gives its slot, a slot that
        holds a *pointer* to a struct (a ``ptr_self`` parameter) is
        dereferenced first, and a runtime pointer value is used as it
        is.  The address is a runtime pointer value whose element type is
        the struct type."""
        match ev:
            case PendingSlot():
                t = ev.type
                if t is None:
                    raise CompileError('cannot access the fields of a variable that has not been assigned yet')
                if isinstance(t, StructType):
                    assert ev.ptr is not None
                    return ev.ptr, t
                if isinstance(t, PointerType) and isinstance(t.elem, StructType):
                    # the slot holds the address of the struct: dereference
                    assert ev.ptr is not None
                    ptr = self._emit(mir.Load(ev.ptr, t))
                    return ptr, t.elem
                raise CompileError(
                    f"cannot access fields of a {type_str(t)} value: "
                    'only struct values have fields'
                )
            case RuntimeVal(value):
                t = typeof(value)
                if isinstance(t, PointerType) and isinstance(t.elem, StructType):
                    return value, t.elem
                raise CompileError(
                    f"cannot access fields of a {type_str(t)} value: "
                    'only struct values have fields'
                )
            case _:
                raise CompileError('cannot access the fields of this value')

    def _exec_field_addr(self, inst: hir.FieldAddr) -> InterpVal:
        """One step of an attribute chain on a struct value: the address
        of the field ``inst.name`` of the struct ``inst.base`` denotes."""
        ptr, type = self._struct_addr_of(self._operand(inst.base))
        index = _field_index(type, inst.name)
        value = self._emit(mir.Gep(ptr, index))
        return RuntimeVal(value)

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
        if not self._ret_emit:
            raise CompileError(
                "runtime 'if' inside inlined functions is not supported yet"
            )
        if not isinstance(cond, RuntimeVal) or typeof(cond.value) != BoolType():
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
        if target is not None and target != VoidType():
            raise CompileError(
                f"cannot return without a value where {type_str(target)} is expected"
            )
        self._saw_void_return = True
        self._emit(mir.Ret(None))

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _type_of(ev: InterpVal) -> Type | None:
        match ev:
            case RuntimeVal(value):
                return typeof(value)
            case ComptimeVal(obj):
                t = value_type(obj)
                return to_mir_type(t) if t is not None else None
            case ComptimeRefVal(obj):
                t = value_type(obj)
                return PointerType(to_mir_type(t), True) if t is not None else None
            case _:
                return None

    def _describe(self, ev: InterpVal) -> str:
        if isinstance(ev, ComptimeVal):
            return repr(ev.obj)
        t = self._type_of(ev)
        if t is None:
            return 'an untyped value'
        return f'a {type_str(t)} value'

    def _coerce(self, ev: InterpVal, target: Type) -> mir.Value:
        """Materialize a value of type ``target``; numeric widening
        conversions (int -> float, float32 -> float64) are applied."""
        match ev:
            case ComptimeVal(obj):
                return _const_of_py(obj, target)
            case RuntimeVal(value):
                from_type = typeof(value)
                return self._convert(value, from_type, target)
            case _:
                raise CompileError('cannot materialize this value')

    def _convert(self, value: mir.Value, from_type: Type, to_type: Type) -> mir.Value:
        if from_type == to_type:
            return value
        if isinstance(from_type, IntType) and isinstance(to_type, IntType):
            if from_type.bits < to_type.bits:
                kind = 'sext' if from_type.signed else 'zext'
            else:
                kind = 'trunc'
            return self._emit(mir.Convert(kind, value, to_type))
        if isinstance(from_type, IntType) and isinstance(to_type, FloatType):
            kind = 'sitofp' if from_type.signed else 'uitofp'
            return self._emit(mir.Convert(kind, value, to_type))
        if isinstance(from_type, FloatType) and isinstance(to_type, FloatType):
            kind = 'fpext' if from_type.bits < to_type.bits else 'fptrunc'
            return self._emit(mir.Convert(kind, value, to_type))
        raise CompileError(
            f"cannot convert a {type_str(from_type)} value to {type_str(to_type)}"
        )

    # -- operators ------------------------------------------------------------

    def _bin_types(self, lhs: InterpVal, rhs: InterpVal) -> tuple[Type | None, Type | None]:
        """The static types of the two operands of a binary operation.  A
        compile-time integer constant adopts the type of a runtime
        integer operand (``x + 1`` with ``x: u64`` is a u64 addition,
        like an integer argument marshals to the annotated type at the
        Python boundary); compile-time floats keep the default mapping
        and mix with integers by promotion."""
        lt = self._type_of(lhs)
        rt = self._type_of(rhs)
        if (
            isinstance(rhs, ComptimeVal)
            and not isinstance(rhs.obj, bool)
            and isinstance(rhs.obj, int)
            and isinstance(lt, IntType)
        ):
            rt = lt
        elif (
            isinstance(lhs, ComptimeVal)
            and not isinstance(lhs.obj, bool)
            and isinstance(lhs.obj, int)
            and isinstance(rt, IntType)
        ):
            lt = rt
        return lt, rt

    def _eval_binary(self, inst: hir.Binary) -> InterpVal:
        op = inst.op
        lhs = self._operand(inst.lhs)
        rhs = self._operand(inst.rhs)
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
            return ComptimeVal(_comptime_py_op(op, lhs.obj, rhs.obj))
        lt, rt = self._bin_types(lhs, rhs)
        type = _binary_type(op, lt, rt, f"apply '{op}' to") if lt is not None and rt is not None else None
        if type is None:
            raise _unsupported_type_error(op, lt)
        if isinstance(type, IntType):
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
        signed = isinstance(type, IntType) and type.signed
        value = self._emit(mir.Arith(_ARITH_OPS[op], signed, lv, rv, type))
        return RuntimeVal(value)

    def _eval_cmp(self, inst: hir.Compare) -> InterpVal:
        op = inst.op
        lhs = self._operand(inst.lhs)
        rhs = self._operand(inst.rhs)
        if isinstance(lhs, ComptimeVal) and isinstance(rhs, ComptimeVal):
            return ComptimeVal(_comptime_py_op(op, lhs.obj, rhs.obj))
        lt, rt = self._bin_types(lhs, rhs)
        type = _binary_type(op, lt, rt, 'compare') if lt is not None and rt is not None else None
        if type is None:
            raise _unsupported_type_error(op, lt)
        lv = self._coerce(lhs, type)
        rv = self._coerce(rhs, type)
        kind = 'int' if isinstance(type, IntType) else 'float'
        signed = isinstance(type, IntType) and type.signed
        value = self._emit(mir.Cmp(_CMP_OPS[op], signed, kind, lv, rv))
        return RuntimeVal(value)

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
        type = self._type_of(operand)
        if type is None:
            raise CompileError(f"cannot apply unary '{op}' to a compile-time object")
        value = self._coerce(operand, type)
        if op == 'not':
            if type != BoolType():
                raise CompileError(f"cannot apply 'not' to a {type_str(type)} value")
            one = BoolValue(True)
            return RuntimeVal(self._emit(mir.Arith('xor', False, value, one, type)))
        if op == 'neg':
            if isinstance(type, FloatType):
                zero = FloatValue(0.0, type.bits)
                return RuntimeVal(self._emit(mir.Arith('sub', False, zero, value, type)))
            if isinstance(type, IntType):
                zero = IntValue(0, type.bits, type.signed)
                return RuntimeVal(self._emit(mir.Arith('sub', False, zero, value, type)))
            raise CompileError(f"cannot negate a {type_str(type)} value")
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
                if self._ret_emit and slot.type is not None:
                    self._note_inplace_ret(slot.type)
                return
            if self._ret_emit:
                self._write_result(ev)
            else:
                # an inlined body: the return value is only recorded and
                # yielded at its ``Ret``
                slot.value = ev
            return
        if isinstance(ev, InPlaceResult):
            # the callee (a struct constructor) already wrote the result
            # into the result location itself
            return
        if isinstance(slot, PendingSlot):
            if slot.ptr is None:
                if isinstance(ev, RuntimeVal) and isinstance(typeof(ev.value), StructType):
                    # a struct call result needs real memory (its address
                    # may escape: fields are accessed and values passed on)
                    struct = typeof(ev.value)
                    assert isinstance(struct, StructType)
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
            ptype = typeof(slot.value)
            if not isinstance(ptype, PointerType):
                raise CompileError(
                    f"cannot write a call result through a {type_str(ptype)} value"
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
        if obj is spy_builtins.spy_type:
            return self._call_builtin_type(inst)
        if obj is spy_builtins.spy_compile_log:
            return self._call_builtin_compile_log(inst)
        if obj is spy_builtins.spy_as:
            raise CompileError(
                "spy.as can only be used at the Python call boundary, not inside spy functions"
            )
        if isinstance(obj, FunctionEntry):
            return self._call_spy_function(obj, inst)
        if isinstance(obj, SpyStructType):
            # a constructor ``Bar(...)``
            return self._call_constructor(obj, inst)
        if isinstance(obj, pytypes.FunctionType):
            return self._call_plain_function(obj, inst)
        raise CompileError(
            f"cannot compile a call to {obj!r}; only spy functions, plain Python "
            "functions and the spy builtins can be called"
        )

    def _call_builtin_type(self, inst: hir.CallInplace) -> InterpVal:
        if len(inst.args) != 1:
            raise CompileError('spy.type takes exactly one argument')
        arg = self._operand(inst.args[0])
        match arg:
            case ComptimeVal(obj):
                type = value_type(obj)
            case RuntimeVal(value):
                type = to_spy_type(typeof(value))
            case _:
                type = None
        if type is None:
            raise CompileError(
                f"spy.type of the compile-time value {self._describe(arg)} is not supported"
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

    def _materialize_location(self, loc: InterpVal, struct: StructType) -> mir.Value:
        """The address one call writes a struct result of type ``struct``
        into - the result location of the enclosing statement: a slot that
        is given the memory of the struct when it has none yet (a slot
        that already holds a struct is reused), or the result pointer of
        a result-pointer function."""
        if isinstance(loc, PendingSlot):
            if loc.ptr is None:
                if loc.value is not None:
                    # the slot only recorded a scalar call result that was
                    # never materialized: it is discarded by this assignment
                    loc.value = None
                loc.ptr = self._emit(mir.Alloca(PointerType(struct)))
                loc.type = struct
            elif loc.type is None or loc.type != struct:
                raise CompileError(
                    f'cannot write a {type_str(struct)} value into a slot that '
                    f'already holds a {type_str(loc.type)} value'  # type: ignore[arg-type]
                )
            return loc.ptr
        if isinstance(loc, RetLocVal):
            if loc.ptr is None:
                if self._ret_emit and self._result_mode is None and mir.via_result_ptr(struct):
                    # the function proper turns out to return this struct
                    # through a result pointer: return through it instead
                    # of an extra local copy
                    assert self._fn is not None
                    self._bind_result_ptr(self._fn, struct)
                    assert loc.ptr is not None
                    return loc.ptr
                # a direct-return function (or an inlined body) returning
                # a value written in place: give the location memory
                loc.ptr = self._emit(mir.Alloca(PointerType(struct)))
                loc.type = struct
                loc.value = None
            elif loc.type is None or loc.type != struct:
                raise CompileError(
                    f'cannot write a {type_str(struct)} value into the result '
                    f'location that already holds a {type_str(loc.type)} value'  # type: ignore[arg-type]
                )
            return loc.ptr
        raise CompileError('cannot write a struct value into this location')

    def _call_result_addr(self, ret: hir.Value, struct: StructType) -> mir.Value:
        """The address a call whose result is delivered through a result
        pointer (a function returning a large struct) writes into: its
        result location."""
        loc = self._operand(ret)
        if isinstance(loc, RuntimeVal):
            ptype = typeof(loc.value)
            if not isinstance(ptype, PointerType) or ptype.elem != struct:
                raise CompileError(
                    f'cannot write a {type_str(struct)} value through a '
                    f'{type_str(ptype)} pointer'
                )
            return loc.value
        return self._materialize_location(loc, struct)

    def _call_constructor(self, desc: SpyStructType, inst: hir.CallInplace) -> InterpVal:
        """A struct constructor ``Bar(a, b)``: the result slot receives a
        new struct value.  With a user ``__init__`` the call is dispatched
        to it with ``self`` pointing at the result slot; otherwise every
        argument is written into the field of the same declaration index
        (the default constructor).  Either way the value is written into
        the result location in place - no value is handed back."""
        struct = struct_mir_type(desc)
        ret = self._operand(inst.ret)
        if not isinstance(ret, (PendingSlot, RetLocVal)):
            raise CompileError('a constructor must write into a variable slot')
        ptr = self._materialize_location(ret, struct)
        init = self._resolver.resolve_method(desc, '__init__')
        if init is not None:
            # a user-provided ``__init__``: call it with ``self`` bound to
            # the address of the result (constructors always write
            # through the result pointer)
            target, _ = init
            evals = [self._operand(a) for a in inst.args]
            if isinstance(target, FunctionEntry):
                self._call_entry_with_self(target, True, inst, struct, ptr, evals)
            else:
                self._inline_method_with_self(target, True, struct, ptr, evals)
            return InPlaceResult()
        fields = struct.fields
        if len(inst.args) != len(fields):
            raise CompileError(
                f"constructor {desc.name} takes {len(fields)} arguments "
                f"(one per field), got {len(inst.args)}"
            )
        for i, (field, arg) in enumerate(zip(fields, inst.args)):
            ev = self._operand(arg)
            value = _materialize_arg(
                ev, field.type, f"the '{field.name}' argument of {desc.name}"
            )
            field_ptr = self._emit(mir.Gep(ptr, i))
            self._emit(mir.Store(field_ptr, value))
        return InPlaceResult()

    def _exec_call_method(self, inst: hir.CallMethodInplace) -> None:
        """A method call ``x.h(...)``: the method is resolved from the
        static type of the struct ``x`` denotes and called with the
        ``self`` argument injected (by value, or - for a ``ptr_self``
        method - as the address of the struct, so that the method can
        modify it in place)."""
        addr, struct = self._struct_addr_of(self._operand(inst.base))
        desc: SpyStructType = struct.spy_type
        target = self._resolver.resolve_method(desc, inst.name)
        if target is None:
            raise CompileError(f"type {desc.name} has no method named '{inst.name}'")
        method, ptr_self = target
        evals = [self._operand(a) for a in inst.args]
        if isinstance(method, FunctionEntry):
            ev = self._call_entry_with_self(method, ptr_self, inst, struct, addr, evals)
        else:
            ev = self._inline_method_with_self(method, ptr_self, struct, addr, evals)
        # the method's result lands in the result location like any call
        self._store_result(ev, inst.ret)

    def _self_value(
        self, ptr_self: bool, struct: StructType, addr: mir.Value
    ) -> tuple[InterpVal, SpyType]:
        """The ``self`` argument of a method call: by value it is the
        struct loaded from its address, by pointer it is the address
        itself."""
        if ptr_self:
            return RuntimeVal(addr), SpyPointerType(struct.spy_type, is_const=False)
        value = self._emit(mir.Load(addr, struct))
        return RuntimeVal(value), struct.spy_type

    def _call_entry_with_self(
        self,
        entry: FunctionEntry,
        ptr_self: bool,
        inst: hir.CallMethodInplace | hir.CallInplace,
        struct: StructType,
        addr: mir.Value,
        evals: list[InterpVal],
    ) -> InterpVal:
        """A native call of a registered spy method (``@aot`` or
        ``@jit``), with its first parameter bound to ``self``."""
        if entry.context is not self._resolver:
            raise CompileError(
                f"cannot call method {entry.fn.__name__} from another JitContext"
            )
        fn_ir = entry.hir
        self_ev, _ = self._self_value(ptr_self, struct, addr)
        evals2 = [self_ev, *evals]
        if isinstance(entry, FunctionValue):
            formal = tuple(a.type for a in entry.args)
            if len(evals2) > len(fn_ir.params):
                raise CompileError(
                    f'method {fn_ir.name} takes {len(fn_ir.params) - 1} '
                    f'arguments, got {len(evals)}'
                )
            values = _convert_evals(fn_ir, evals2, formal)
        else:
            formal = self._solve_types(fn_ir, evals2, 'jit')
            values = _convert_evals(fn_ir, evals2, formal)
        callee, ret_type = self._resolver.resolve_call(entry, formal)
        if ret_type is not None and mir.via_result_ptr(ret_type):
            # the method returns a large struct: it writes into the
            # result location, whose address is passed as its trailing
            # result pointer argument, and returns void
            assert isinstance(ret_type, StructType)
            dest = self._call_result_addr(inst.ret, ret_type)
            self._emit(mir.Call(callee, values + (dest,), VoidType()))
            return InPlaceResult()
        value = self._emit(mir.Call(callee, values, ret_type))
        if ret_type == VoidType():
            return ComptimeVal(None)
        return RuntimeVal(value)

    def _inline_method_with_self(
        self,
        fn: Any,
        ptr_self: bool,
        struct: StructType,
        addr: mir.Value,
        evals: list[InterpVal],
    ) -> InterpVal:
        """A plain (undecorated) method: it is inlined into the current
        stream like any plain Python function (its body may only use what
        inlining supports)."""
        fn_ir = self._resolver.hir_of_plain_fn(fn)
        if any(f.fn is fn for f in self._inline_stack):
            raise CompileError(
                f'a plain Python method cannot call itself recursively '
                f'(method {fn_ir.name}) - declare it with @aot/@jit instead'
            )
        if self._inline_depth >= _MAX_INLINE_DEPTH:
            raise CompileError('too deeply nested inlined functions')
        self_ev, _ = self._self_value(ptr_self, struct, addr)
        evals2 = [self_ev, *evals]
        formal = self._solve_types(fn_ir, evals2, 'jit')
        values = _convert_evals(fn_ir, evals2, formal)
        return self._run_inline(fn_ir, values)

    def _solve_types(
        self, fn_ir: astgen.FunctionIR, evals: list[InterpVal], mode: str
    ) -> tuple[SpyType, ...]:
        """The concrete spy types of all formal parameters of one call,
        solved from the provided arguments (defaults included), plus the
        argument count check."""
        if len(evals) > len(fn_ir.params):
            raise CompileError(
                f"function {fn_ir.name} takes {len(fn_ir.params)} arguments, "
                f"got {len(evals)}"
            )
        provided: list[SpyType | None] = [None] * len(fn_ir.params)
        for i, ev in enumerate(evals):
            match ev:
                case ComptimeVal(obj):
                    t = value_type(obj)
                case RuntimeVal(value):
                    t = to_spy_type(typeof(value))
                case _:
                    t = None
            if t is None:
                raise CompileError(
                    f"cannot pass the compile-time value {self._describe(ev)} "
                    f"as an argument of function {fn_ir.name}"
                )
            provided[i] = t
        try:
            return astgen.solve_call_types(fn_ir, mode, tuple(provided))
        except TypeMismatchError as e:
            raise CompileError(str(e)) from e

    def _arg_values_of(
        self, fn_ir: astgen.FunctionIR, args: tuple[hir.Value, ...], mode: str
    ) -> tuple[tuple[mir.Value, ...], tuple[SpyType, ...]]:
        """Resolve the arguments of one call against the callee signature.

        Returns the marshaled argument values (defaults included) and the
        concrete spy types of all formal parameters (the ``spy`` side
        drives the call resolution).
        """
        evals = [self._operand(a) for a in args]
        formal = self._solve_types(fn_ir, evals, mode)
        values = _convert_evals(fn_ir, evals, formal)
        return values, formal

    def _call_spy_function(self, entry: FunctionEntry, inst: hir.CallInplace) -> InterpVal:
        if entry.context is not self._resolver:
            raise CompileError(
                f"cannot call function {entry.fn.__name__} from another JitContext"
            )
        fn_ir = entry.hir
        values, formal = self._arg_values_of(fn_ir, inst.args, entry.kind)
        callee, ret_type = self._resolver.resolve_call(entry, formal)
        if ret_type is not None and mir.via_result_ptr(ret_type):
            # the callee returns a large struct: it writes into the
            # result location, whose address is passed as its trailing
            # result pointer argument, and returns void
            assert isinstance(ret_type, StructType)
            dest = self._call_result_addr(inst.ret, ret_type)
            self._emit(mir.Call(callee, values + (dest,), VoidType()))
            return InPlaceResult()
        value = self._emit(mir.Call(callee, values, ret_type))
        if ret_type == VoidType():
            # a void call produces no value: it only has effects
            return ComptimeVal(None)
        return RuntimeVal(value)

    def _run_inline(
        self, fn_ir: astgen.FunctionIR, values: tuple[mir.Value, ...]
    ) -> InterpVal:
        """Run the body of an inlined plain function with the given
        (already materialized) argument values, returning its result: the
        value of its ``return``, or ``None`` for a void body."""
        self._inline_stack.append(fn_ir)
        self._inline_depth += 1
        self._ret_locs.append((fn_ir.ret_loc, RetLocVal()))
        self._push_frame(values)
        saved_ret_emit = self._ret_emit
        self._ret_emit = False
        try:
            flow, value = self._run_list(fn_ir.body)
        finally:
            self._ret_emit = saved_ret_emit
            self._ret_locs.pop()
            self._frames.pop()
            self._inline_depth -= 1
            self._inline_stack.pop()
        if flow is not Flow.RET:
            # the inlined body fell off its end: a void inline
            return ComptimeVal(None)
        assert isinstance(value, InterpVal)
        return value

    def _call_plain_function(self, fn: Any, inst: hir.CallInplace) -> InterpVal:
        fn_ir = self._resolver.hir_of_plain_fn(fn)
        if any(f.fn is fn for f in self._inline_stack):
            raise CompileError(
                f"a plain Python function cannot call itself recursively "
                f"(function {fn_ir.name}); plain functions are inlined, and a "
                'recursive inline would never finish compiling - declare it '
                'as a spy function instead'
            )
        if self._inline_depth >= _MAX_INLINE_DEPTH:
            raise CompileError('too deeply nested inlined functions')
        evals = [self._operand(a) for a in inst.args]
        if len(evals) > len(fn_ir.params):
            raise CompileError(
                f"function {fn_ir.name} takes {len(fn_ir.params)} arguments, "
                f"got {len(evals)}"
            )
        formal = self._solve_types(fn_ir, evals, 'jit')
        values = _convert_evals(fn_ir, evals, formal)
        return self._run_inline(fn_ir, values)

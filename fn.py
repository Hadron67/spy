
import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FunctionType as PyFunctionType
from typing import Any, TypeAlias, TypeVar, override

from . import hir, mir
from .type import (
    AnyFunction,
    FormalArg,
    FunctionCallInfo,
    FunctionType,
    Type,
    Value,
)


@dataclass(frozen=True)
class ParamDef:
    name: str
    # The evaluated annotation from ``fn.__annotations__``: either a concrete
    # spy type or one of the function's PEP 695 type parameter objects.
    annotation: Any | None = None
    has_default: bool = False
    default_value: Any | None = None


@dataclass
class FunctionIR:
    fn: Callable
    name: str
    # The declared PEP 695 type parameter objects (``[T]``), kept so that
    # annotation values can be recognized as type parameters by identity.
    type_params: tuple[TypeVar, ...]
    params: tuple[ParamDef, ...]
    # The evaluated return annotation from ``fn.__annotations__``.
    ret_annotation: Any | None
    body: tuple[hir.Inst, ...]
    # The result location the return statements of the body write into
    # (see ``hir.ResultLoc``)
    ret_loc: 'hir.ResultLoc' = None  # type: ignore[assignment]

@dataclass
class NativeFn:
    """A compiled native function of one specialization.

    ``arg_types``/``ret_type`` are the *lowered* signature (see
    ``mir.returns_via_result_ptr``): a function that returns through a
    result pointer carries its trailing result pointer formal in
    ``arg_types``, a void ``ret_type`` and its logical return type in
    ``result_type``.  The Python-facing ``_entry`` is pointer-ABI form
    (see ``lower.compile_module``)."""

    name: str
    arg_types: tuple[mir.Type, ...]
    ret_type: mir.Type
    lines: list[str] = field(default_factory=list)
    # the return type of a result-pointer function (``ret_type`` is then
    # void and ``arg_types`` carries the trailing result pointer formal);
    # None for a direct-return function
    result_type: mir.Type | None = None
    _engine: object = None  # type: ignore[assignment]
    _addr: int = 0
    _entry: Any = None

    def call(self, *values) -> object:
        logical = self.result_type if self.result_type is not None else self.ret_type
        if isinstance(logical, mir.StructType):
            # the Python-facing entry writes the result into an out buffer
            # (see ``lower.compile_module``): allocate the instance, pass
            # its address as the trailing argument and return it
            out = logical.ctype()
            self._entry(*values, ctypes.addressof(out))
            return out
        return self._entry(*values)

    @property
    def addr(self) -> int:
        return self._addr

    def print_all(self) -> list[str]:
        return self.lines


@dataclass(frozen=True)
class LazyJitFunctionInstance:
    """The compiled artifact of one ``@jit`` specialization: its native
    function (what a Python-side call invokes, see :class:`NativeFn`)
    and the call lowering plan a spy function body follows when it
    calls the specialization (see ``type.function_call_info``)."""

    native_fn: NativeFn
    call_info: FunctionCallInfo


class LazyJitFunction(Value):
    """The function value of a ``@jit`` function: only compiled - and
    thereby typed - when a call specializes it, so as a value its type
    is the untyped :class:`AnyFunction`.

    Like :class:`FunctionValue` the value doubles as the per-function
    entry of its host context (function values are identity objects: two
    are equal only if they are the same object).  The call logic itself
    lives in the interpreter and the host, not here.
    """

    kind = 'jit'

    def __init__(self, fn: PyFunctionType, hir: FunctionIR) -> None:
        self.fn = fn
        # the hosting JitContext (set when the value is registered),
        # only used when checking whether the function is called within
        # the same context
        self.context: Any = None
        # the context-unique base name of the native symbols
        self.name_base = ''
        # the parsed HIR of the function (see ``JitContext.hir_of``)
        self.hir = hir
        # spy argument types -> the typed MIR function of the
        # specialization.  The function is registered here - with an
        # empty body - before its body is typed, so a recursive call
        # made by the body resolves to it.  A function whose module
        # build aborted stays cached until the next build that
        # references it (its spec is only registered when the module it
        # was lowered into finishes).
        self.mir_cache: dict[tuple[Type, ...], mir.Function] = {}
        # spy argument types -> the compiled artifacts of the
        # specialization (see ``LazyJitFunctionInstance``)
        self.specs: dict[tuple[Type, ...], LazyJitFunctionInstance] = {}
        # spy argument types -> error message of a failed compilation
        self.failed: dict[tuple[Type, ...], str] = {}

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def type(self) -> Type:
        return AnyFunction()


class FunctionValue(Value):
    """The function value of a ``@aot`` function: compiled from its type
    annotations at its first use, so the value carries the concrete
    signature (``args`` and ``ret``) and the compiled
    :class:`mir.Function` (calling it emits a ``mir.Call`` of that
    function).

    An aot function has exactly one specialization (its signature is
    fixed by the annotations), so unlike :class:`LazyJitFunction` it
    needs no per-argument-type registries: ``mir_fn`` is the single
    typed MIR function - it is set before the body is typed (so a
    recursive call the body makes resolves to it) and filled in by the
    typing.  Function values are identity objects: two are equal only
    if they are the same object.  The call logic itself lives in the
    interpreter and the host, not here.
    """

    kind = 'aot'

    def __init__(
        self,
        fn: PyFunctionType,
        hir: FunctionIR,
        args: tuple[FormalArg, ...],
        ret: Type | None,
        mir_fn: mir.Function | None = None,
    ) -> None:
        self.fn = fn
        # the hosting JitContext (set when the value is registered),
        # only used when checking whether the function is called within
        # the same context
        self.context: Any = None
        # the context-unique base name of the native symbols
        self.name_base = ''
        # the parsed HIR of the function (see ``JitContext.hir_of``)
        self.hir = hir
        self.args = args
        # the declared return type, or None when it is inferred from the
        # body (an ``aot`` method without a return annotation)
        self.ret: Type | None = ret
        self.mir_fn = mir_fn

        self.native_fn: NativeFn | None = None

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def type(self) -> FunctionType:
        """The spy type of the function *value*: its signature - a
        function type.  A function type is a runtime DST (dynamically
        sized type: it has no runtime representation of its own), so a
        function value is never a legal runtime value by itself; it can
        only be *referenced* - a ``hir.ConstRef`` of it, whose type is a
        const pointer to this function type (a function pointer)."""
        ret = self.ret
        assert ret is not None, 'the function is still being typed'
        return FunctionType(self.args, ret)


# A registered spy function of either kind: the per-function entry of
# its host context.  jit and aot functions share no base class; the
# union only types the code that works with entries of both kinds
# (``dsl``/``interp``).
FunctionEntry: TypeAlias = LazyJitFunction | FunctionValue

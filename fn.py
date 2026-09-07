import ctypes
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FunctionType as PyFunctionType
from typing import Any, TypeAlias, override

from spy.util import IndexedMap

from . import hir, mir
from .errors import TypeMismatchError
from .sval import (
    AnyFunction,
    AnyValue,
    FormalArg,
    FunctionType,
    Type,
    TypeVar,
    TypeVarSolver,
    Value,
    returns_via_result_ptr,
    to_mir_type,
)


@dataclass(frozen=True)
class ParamDef:
    name: str
    # The evaluated annotation of the parameter, in the spy domain (see
    # ``Signature``): a concrete spy type, a generic type parameter of
    # the signature (a ``TypeVar`` of ``Signature.generic_args``), or
    # None when the parameter is unannotated.
    type: Type | None
    # The evaluated default value of the parameter, in the spy domain
    # (see ``Signature``); None when the parameter has no default.
    default_value: AnyValue | None


@dataclass(frozen=True)
class Signature:
    """The complete signature of a spy function, read off its Python
    definition: the declared generic type parameters (PEP 695 ``[T]``,
    as the spy-domain ``TypeVar`` the annotations name them by), the
    formal parameters by declaration position (``positional``) and - for
    the calls the parser allows - the ``*args``/``**kwargs`` parameters,
    and the return annotation.  Every annotation and default value is
    stored in the *spy domain*: it has been converted with
    ``sval.as_value``, so a generic parameter annotation is the
    signature's own ``TypeVar`` and a default value is an
    :class:`~spy.sval.AnyValue` (a plain ``None`` default is the unit
    value of the void type, ``sval.Void()``)."""

    # the declared generic type parameters, by name: ``[T]`` declares
    # one entry ``T -> TypeVar('T')``
    generic_args: IndexedMap[str, TypeVar]
    # the formal parameters, by declaration position
    positional: IndexedMap[int, ParamDef]
    # the ``*args``/``**kwargs`` parameters (always None for now: spy
    # function definitions do not accept them yet, but the signature
    # model - and ``bind_args`` - already does)
    varargs: ParamDef | None
    kwargs: ParamDef | None
    # the evaluated return annotation, in the spy domain (a concrete spy
    # type, a type parameter, or the void type for an explicit
    # ``-> None``); None when no return annotation is written and the
    # return type is inferred from the body
    ret_type: Type | None

    def bind_args[T](
        self,
        positional: tuple[T, ...],
        kwargs: dict[str, T],
        default_converter: Callable[[AnyValue], T],
    ) -> tuple[tuple[T, ...], tuple[T, ...], dict[str, T]]:
        """Bind the arguments of one call to the function's formal
        parameters.  Returns ``(positional, varargs, kwargs)`` where

        * ``positional`` holds the value bound to every *positional*
          formal parameter, in declaration order: the argument the call
          provides for it - positionally or by name - or, for a
          parameter the call leaves out, its default value converted by
          ``default_converter``;
        * ``varargs`` holds the excess positional arguments (bound to
          the ``*args`` formal, when the function has one);
        * ``kwargs`` holds the keyword arguments that name no formal
          parameter (bound to the ``**kwargs`` formal, when the
          function has one).

        Binding errors - too many positional arguments, an unknown or
        duplicate keyword argument, a missing required argument - are
        raised as :class:`TypeError`."""
        pos_params = self.positional.values()
        n = len(pos_params)
        param_index = {p.name: i for i, p in enumerate(pos_params)}
        values: dict[int, T] = {}
        varargs_out: list[T] = []
        # the positional arguments bind the leading parameters in order;
        # the excess bind the ``*args`` formal
        for i, value in enumerate(positional):
            if i < n:
                values[i] = value
            elif self.varargs is not None:
                varargs_out.append(value)
            else:
                raise TypeError(
                    f'takes {n} positional arguments but {len(positional)} were given'
                )
        # the keyword arguments bind the remaining parameters by name
        kwargs_out: dict[str, T] = {}
        for key, value in kwargs.items():
            idx = param_index.get(key)
            if idx is not None:
                if idx in values:
                    raise TypeError(f"got multiple values for argument '{key}'")
                values[idx] = value
            elif self.kwargs is not None:
                kwargs_out[key] = value
            else:
                raise TypeError(f"got an unexpected keyword argument '{key}'")
        # the parameters the call leaves out take their default values
        bound: list[T] = []
        for i, param in enumerate(pos_params):
            if i in values:
                bound.append(values[i])
            else:
                default = param.default_value
                if default is None:
                    raise TypeError(f"missing required argument '{param.name}'")
                bound.append(default_converter(default))
        return tuple(bound), tuple(varargs_out), kwargs_out

    def solve_param_types(
        self, provided: tuple[Type | None, ...]
    ) -> tuple[Type, ...]:
        """The concrete spy type of every formal parameter of one call,
        given ``provided``: the marshaled type of each argument the call
        provides, and ``None`` for a parameter whose default value
        applies (no argument was provided for it).

        This is how a jit function is typed: a provided argument always
        types the parameter it is provided for (parameter annotations,
        concrete ones included, do not constrain a jit call), but
        parameters annotated with the same type parameter must all be
        provided arguments marshaling to one type, which the type
        parameter unifies.  A parameter that no argument covers takes
        the type its parameter was unified to, the type of its default
        value, or raises when it has neither."""
        params = self.positional.values()
        assert len(provided) == len(params), 'argument count mismatch'
        # unify the type parameters over the provided arguments: two
        # arguments of parameters annotated with the same type parameter
        # must marshal to the same type
        solver = TypeVarSolver()
        for param, cand in zip(params, provided):
            if cand is not None and isinstance(param.type, TypeVar):
                solver.add_constraint(param.type, cand)
        solver.finish()
        solved: dict[TypeVar, Type] = {
            k: v
            for k, v in solver.get_solved().items()
            if isinstance(v, Type)
        }
        types: list[Type] = []
        for param, cand in zip(params, provided):
            if cand is not None:
                # a provided argument types the parameter it is provided
                # for
                types.append(cand)
            elif isinstance(param.type, TypeVar):
                bound = solved.get(param.type)
                if bound is not None:
                    types.append(bound)
                elif param.default_value is not None:
                    types.append(_default_value_type(param))
                else:
                    raise TypeMismatchError(f"missing argument '{param.name}'")
            elif param.default_value is not None:
                types.append(_default_value_type(param))
            else:
                raise TypeMismatchError(f"missing argument '{param.name}'")
        return tuple(types)

    def solve_return_type(self, arg_types: tuple[Type, ...]) -> Type | None:
        """The declared return type of one call whose formal parameter
        types are ``arg_types``, or ``None`` when the function declares
        none (the return type is then inferred from the body).  A return
        annotation that names a type parameter resolves to the type the
        parameter is bound to - the type of the first parameter
        annotated with the same type parameter."""
        ret = self.ret_type
        if ret is None:
            return None
        if isinstance(ret, TypeVar):
            for param, t in zip(self.positional.values(), arg_types):
                if param.type is ret:
                    return t
            return None
        return ret

    def is_generic(self) -> bool:
        if len(self.generic_args.by_id) > 0 or self.varargs is not None or self.kwargs is not None:
            return True
        if self.ret_type is None:
            return True
        for arg in self.positional.by_id:
            if arg.type is None:
                return True
        return False

    def as_non_generic_fn_type(self) -> FunctionType | None:
        if self.is_generic():
            return None
        assert self.ret_type is not None
        formal: list[FormalArg] = []
        for arg in self.positional.by_id:
            assert arg.type is not None
            formal.append(FormalArg(arg.name, arg.type, arg.default_value))
        return FunctionType(tuple(formal), self.ret_type)


@dataclass
class FunctionIR:
    name: str
    signature: Signature
    body: tuple[hir.Inst, ...]
    # The result location the return statements of the body write into
    # (see ``hir.ResultLoc``)

    # -- the aot signature (``mode == 'aot'``) ---------------------------

    def aot_param_type(self, param: ParamDef) -> Type:
        """The concrete spy type the annotation of the ``aot`` parameter
        ``param`` declares.  Raises :class:`TypeMismatchError` when the
        parameter is unannotated, annotated with a type parameter, or
        annotated with a value that is not a spy type."""
        t = param.type
        if t is None:
            raise TypeMismatchError(
                f"parameter '{param.name}' of function {self.name} requires a "
                'type annotation'
            )
        if isinstance(t, TypeVar):
            raise TypeMismatchError(
                f'type parameter {t.name} is not allowed in aot function {self.name}'
            )
        if not isinstance(t, Type):
            raise TypeMismatchError(
                f"annotation of parameter '{param.name}' of function {self.name} "
                f'is not a spy type: {t!r}'
            )
        return t

    def aot_return_type(self) -> Type:
        """The declared return type of an ``aot`` function - mandatory:
        unlike a method, a plain ``aot`` function must declare its
        return type."""
        ret = self.signature.ret_type
        if ret is None:
            raise TypeMismatchError(
                f'function {self.name} requires a return type annotation'
            )
        if isinstance(ret, TypeVar):
            raise TypeMismatchError(
                f'type parameter {ret.name} is not allowed in aot function {self.name}'
            )
        if not isinstance(ret, Type):
            raise TypeMismatchError(
                f'return annotation of function {self.name} is not a spy type: {ret!r}'
            )
        return ret

@dataclass
class NativeFn:
    """A compiled native function of one specialization.

    ``arg_types``/``ret_type`` are the *lowered* signature (see
    ``mir.returns_via_result_ptr``): a function that returns through a
    result pointer carries its trailing result pointer formal in
    ``arg_types``, a ``None`` (void) ``ret_type`` and its logical return
    type in ``result_type``.  The Python-facing ``_entry`` is
    pointer-ABI form (see ``lower.compile_module``)."""

    name: str
    arg_types: tuple[mir.Type, ...]
    ret_type: mir.Type | None
    lines: list[str] = field(default_factory=list)
    # the return type of a result-pointer function (``ret_type`` is then
    # ``None`` and ``arg_types`` carries the trailing result pointer
    # formal); None for a direct-return function
    result_type: mir.Type | None = None
    _engine: object = None  # type: ignore[assignment]
    _addr: int = 0
    _entry: Any = None

    def call(self, *values) -> object:
        logical = self.result_type if self.result_type is not None else self.ret_type
        if isinstance(logical, mir.StructType):
            # the Python-facing entry writes the result into an out buffer
            # (see ``lower.compile_module``): allocate the instance, pass
            # its address as the trailing argument and return it.  The
            # class of the buffer is the ctypes view of the struct's MIR
            # layout (``lower.struct_ctype``), materialized when the
            # function was compiled
            out = logical.ctype()
            self._entry(*values, ctypes.addressof(out))
            return out
        return self._entry(*values)

    @property
    def addr(self) -> int:
        return self._addr

    def print_all(self) -> list[str]:
        return self.lines


class ArgSignatureNode:
    pass

@dataclass(frozen=True)
class ComptimeValSignatureNode(ArgSignatureNode):
    value: AnyValue

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class RuntimeValSignatureNode(ArgSignatureNode):
    type: Type
    by_ref: bool
    mir_arg_index: int

    def __str__(self) -> str:
        return f"<{self.type}>"

@dataclass(frozen=True)
class FnInstanceSignature:
    """
        A function signature.

        Serves both as the cache key of the specialization and how to call the MIR function
    """
    positional: tuple[ArgSignatureNode, ...]
    varargs: tuple[ArgSignatureNode, ...]
    kwargs: frozenset[tuple[str, ArgSignatureNode]]

    def __str__(self) -> str:
        positional = ", ".join(str(a) for a in self.positional)
        varargs = ", ".join(str(a) for a in self.varargs)
        kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs)
        return f"({positional}{", " if varargs else ""}({varargs}){", " if kwargs else ""}{{{kwargs}}})"

    def mir_arg_count(self):
        count = 0
        for arg in self.positional + self.varargs + tuple(a[1] for a in self.kwargs):
            if isinstance(arg, RuntimeValSignatureNode):
                count += 1
        return count

class FunctionReturnInfo:
    pass

@dataclass(frozen=True)
class FunctionValueReturnInfo(FunctionReturnInfo):
    """Return by value: No result location pointer, mir function returns `mir_type`."""
    mir_type: mir.MayBeVoidType

@dataclass(frozen=True)
class FunctionRetLocReturnInfo(FunctionReturnInfo):
    """Return by result location: the result pointer is the `arg_index`-th argument, mir function returns void."""
    arg_index: int

def function_call_info(type: FunctionType) -> tuple[FnInstanceSignature, FunctionReturnInfo, mir.FunctionType]:
    positional: list[ArgSignatureNode] = []
    mir_args: list[mir.Type] = []
    for arg in type.args:
        mir_type = to_mir_type(arg.type)
        if isinstance(mir_type, mir.VoidType):
            val = arg.type.get_unit_value()
            assert val is not None
            positional.append(ComptimeValSignatureNode(val))
        else:
            positional.append(RuntimeValSignatureNode(arg.type, False, len(mir_args)))
            mir_args.append(mir_type)
    return_info = None
    if returns_via_result_ptr(type.return_type):
        # the result is written into a caller-provided location whose
        # address is appended as the trailing MIR argument; the lowered
        # function returns void
        result_type = to_mir_type(type.return_type)
        assert not isinstance(result_type, mir.VoidType)
        return_info = FunctionRetLocReturnInfo(arg_index=len(mir_args))
        mir_args.append(mir.PointerType(result_type, False))
    else:
        return_info = FunctionValueReturnInfo(mir_type=to_mir_type(type.return_type))
    signature = FnInstanceSignature(
        positional=tuple(positional),
        varargs=(),
        kwargs=frozenset(),
    )
    return (
        signature,
        return_info,
        mir.FunctionType(tuple(mir_args), to_mir_type(type.return_type)),
    )

@dataclass(frozen=True)
class FunctionInstance:
    """The compiled artifact of one ``@jit`` specialization: its native
    function (what a Python-side call invokes, see :class:`NativeFn`)
    and the lowering result its spy function type yields - the call
    lowering plan a spy function body follows when it calls the
    specialization, together with its lowered MIR signature (see
    ``type.function_call_info``)."""

    mir: mir.Function
    native_fn: NativeFn | None = None
    mir_type: mir.FunctionType | None = None
    complete: bool = False

class HirCompiler:
    @abstractmethod
    def compile(self, signature: FnInstanceSignature) -> FunctionInstance:
        ...

class FunctionValue(Value):
    """The function value of a ``@jit`` function: only compiled - and
    thereby typed - when a call specializes it, so as a value its type
    is the untyped :class:`AnyFunction`.

    Like :class:`FunctionValue` the value doubles as the per-function
    entry of its host context (function values are identity objects: two
    are equal only if they are the same object).  The call logic itself
    lives in the interpreter and the host, not here.
    """

    def __init__(self, name_base: str, hir: FunctionIR) -> None:
        # the context-unique base name of the native symbols
        self.name_base = name_base
        # the parsed HIR of the function (see ``JitContext.hir_of``)
        self.hir = hir
        # spy argument types -> the compiled artifacts of the
        # specialization (see ``LazyJitFunctionInstance``)
        self.specs: dict[FnInstanceSignature, FunctionInstance] = {}
        # spy argument types -> error message of a failed compilation
        self.failed: dict[FnInstanceSignature, str] = {}

        self._non_generic_instance: FunctionInstance | None = None

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> Type:
        return self.hir.signature.as_non_generic_fn_type() or AnyFunction()


class SymbolTable:
    def __init__(self) -> None:
        self._symbols: dict[str, FunctionInstance] = {}

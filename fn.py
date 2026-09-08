import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, override

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
)


@dataclass(frozen=True)
class SignatureFormalArg:
    # The evaluated annotation of the parameter, in the spy domain (see
    # ``Signature``): a concrete spy type, a generic type parameter of
    # the signature (a ``TypeVar`` of ``Signature.generic_args``), or
    # None when the parameter is unannotated.
    type: Type | None
    is_comptime: bool
    # whether pass this parameter by reference (i.e. as a constant pointer)
    by_ref: bool
    # The evaluated default value of the parameter, in the spy domain
    # (see ``Signature``); None when the parameter has no default.
    default_value: AnyValue | None

    def map_type(self, f: Callable[[Type], Type]) -> 'SignatureFormalArg':
        return SignatureFormalArg(
            None if self.type is None else f(self.type),
            self.is_comptime,
            self.by_ref,
            self.default_value,
        )

@dataclass(frozen=True)
class ArgEntry[T]:
    value: T
    is_ref: bool


@dataclass(frozen=True)
class RawArgList[T]:
    positional: tuple[T, ...]
    kwargs: frozenset[tuple[str, T]]

    def map[K](self, f: Callable[[T], K]) -> 'RawArgList[K]':
        return RawArgList(
            tuple(f(p) for p in self.positional),
            frozenset((k, f(v)) for k, v in self.kwargs),
        )

@dataclass(frozen=True)
class ArgList[T]:
    positional: tuple[T, ...]
    varargs: tuple[T, ...]
    kwargs: frozenset[tuple[str, T]]

    def map[K](self, f: Callable[[T], K]) -> 'ArgList[K]':
        return ArgList(
            tuple(f(p) for p in self.positional),
            tuple(f(v) for v in self.varargs),
            frozenset((k, f(v)) for k, v in self.kwargs),
        )

class SpecializedFormalArg:
    pass

@dataclass(frozen=True)
class SpecializedRuntimeArg(SpecializedFormalArg):
    type: Type
    is_ref: bool

    def __str__(self) -> str:
        return f"<{'&' if self.is_ref else ''}{self.type}>"

@dataclass(frozen=True)
class SpecializedComptimeArg(SpecializedFormalArg):
    value: AnyValue

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class SpecializedSignature:
    generic_args: tuple[Value, ...]
    positional: tuple[tuple[str, SpecializedFormalArg]]
    varargs: tuple[SpecializedFormalArg, ...] | None
    kwargs: frozenset[tuple[str, SpecializedFormalArg]] | None
    ret_by_ref: bool | None
    ret_type: Type | None

    def __str__(self) -> str:
        """Note: return type not included"""
        generic = ", ".join(str(a) for a in self.generic_args)
        parts: list[str] = []
        parts.extend(str(a[1] for a in self.positional))
        if self.varargs is not None:
            s = ", ".join(str(a) for a in self.varargs)
            parts.append(f"*({s})")
        if self.kwargs is not None:
            s = ", ".join(str(a) for a in self.kwargs)
            parts.append(f"**{{{s}}}")
        return f"[{generic}]({', '.join(parts)})"

    def with_ret_type(self, ret_type: Type, ret_by_ret: bool) -> 'SpecializedSignature':
        return SpecializedSignature(
            generic_args=self.generic_args,
            positional=self.positional,
            varargs=self.varargs,
            kwargs=self.kwargs,
            ret_type=ret_type,
            ret_by_ref=ret_by_ret,
        )

@dataclass
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
    generic_args: tuple[TypeVar, ...]
    # the formal parameters, by declaration position
    positional: IndexedMap[str, SignatureFormalArg]
    # the ``*args``/``**kwargs`` parameters (always None for now: spy
    # function definitions do not accept them yet, but the signature
    # model - and ``bind_args`` - already does)
    varargs: SignatureFormalArg | None
    kwargs: SignatureFormalArg | None
    # the evaluated return annotation, in the spy domain (a concrete spy
    # type, a type parameter, or the void type for an explicit
    # ``-> None``); None when no return annotation is written and the
    # return type is inferred from the body
    ret_type: Type | None
    ret_by_ref: bool | None

    def bind_arg_pos[T](
        self,
        args: RawArgList[T],
        default_converter: Callable[[AnyValue], T],
    ) -> ArgList[T]:
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
        n = len(self.positional.by_id)
        positional = args.positional
        kwargs = args.kwargs

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
        for key, value in kwargs:
            idx = self.positional.by_key.get(key)
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
        for i, (name, param) in enumerate(self.positional.items()):
            if i in values:
                bound.append(values[i])
            else:
                default = param.default_value
                if default is None:
                    raise TypeError(f"missing required argument '{name}'")
                bound.append(default_converter(default))
        return ArgList(tuple(bound), tuple(varargs_out), frozenset(kwargs_out.items()))

    def solve_param_types(
        self, provided: ArgList[Type | None]
    ) -> tuple[Value, ...]:
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
        assert len(provided.positional) == len(self.positional.by_id), 'argument count mismatch'
        # unify the type parameters over the provided arguments: two
        # arguments of parameters annotated with the same type parameter
        # must marshal to the same type
        solver = TypeVarSolver()
        for param, cand in zip(self.positional.by_id, provided.positional):
            if cand is not None and param.type is not None:
                solver.add_constraint(cand, param.type, True)
        if self.varargs is not None and self.varargs.type is not None:
            for cand in provided.varargs:
                if cand is not None:
                    solver.add_constraint(cand, self.varargs.type, True)
        if self.kwargs is not None and self.kwargs.type is not None:
            for _, type in provided.kwargs:
                if type is not None:
                    solver.add_constraint(type, self.kwargs.type, True)
        solver.finish()
        solved = solver.get_solved()
        ret: list[Value] = []
        for type_var in self.generic_args:
            if type_var not in solved:
                raise TypeMismatchError(f"type variable {type_var.name} not solved")
            ret.append(solved[type_var])
        return tuple(ret)

    def is_generic(self) -> bool:
        if len(self.generic_args) > 0 or self.varargs is not None or self.kwargs is not None:
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
        for name, arg in self.positional.items():
            assert arg.type is not None
            formal.append(FormalArg(name, arg.type, arg.default_value))
        return FunctionType(tuple(formal), self.ret_type)

    def specialize(self, provided: ArgList[Type | None]) -> SpecializedSignature:
        type_vars = self.solve_param_types(provided)
        # TODO: substitute solved type vars, fill remaining
        raise NotImplementedError


@dataclass
class FunctionIR:
    name: str
    signature: Signature
    body: tuple[hir.Inst, ...]
    # The result location the return statements of the body write into
    # (see ``hir.ResultLoc``)

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
    complete: bool = False

class FunctionValue(Value):
    """The function value of a ``@jit`` function: only compiled - and
    thereby typed - when a call specializes it, so as a value its type
    is the untyped :class:`AnyFunction`.

    Like :class:`FunctionValue` the value doubles as the per-function
    entry of its host context (function values are identity objects: two
    are equal only if they are the same object).  The call logic itself
    lives in the interpreter and the host, not here.
    """

    def __init__(self, name_base: str, hir: FunctionIR, force_inline: bool = False) -> None:
        # the context-unique base name of the native symbols
        self.name_base = name_base
        # the parsed HIR of the function (see ``JitContext.hir_of``)
        self.hir = hir
        self.force_inline = force_inline
        # spy argument types -> the compiled artifacts of the
        # specialization (see ``LazyJitFunctionInstance``)
        self.specs: dict[SpecializedSignature, FunctionInstance] = {}
        # spy argument types -> error message of a failed compilation
        self.failed: dict[SpecializedSignature, str] = {}

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> Type:
        return self.hir.signature.as_non_generic_fn_type() or AnyFunction()

@dataclass
class FnSymbol:
    fn: FunctionValue
    sig: SpecializedSignature

class SymbolTable:
    def __init__(self) -> None:
        self._symbols: dict[str, FunctionInstance] = {}

from __future__ import annotations

import ctypes
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, override

from . import hir, mir, opt
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
    pass_by_ref,
    replace_type_vars_type,
    returns_via_result_ptr,
    type_of,
)
from .util import IndexedMap, frozendict


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

    def map_type(self, f: Callable[[Type], Type]) -> SignatureFormalArg:
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
    kwargs: frozendict[str, T]

    def map[K](self, f: Callable[[T], K]) -> RawArgList[K]:
        return RawArgList(
            tuple(f(p) for p in self.positional),
            frozendict((k, f(v)) for k, v in self.kwargs.items()),
        )

@dataclass(frozen=True)
class ArgList[T]:
    positional: tuple[T, ...]
    varargs: tuple[T, ...]
    kwargs: frozendict[str, T]

    def map[K](self, f: Callable[[T], K]) -> ArgList[K]:
        return ArgList(
            tuple(f(p) for p in self.positional),
            tuple(f(v) for v in self.varargs),
            frozendict((k, f(v)) for k, v in self.kwargs.items()),
        )

    def values(self) -> Iterable[T]:
        yield from self.positional
        yield from self.varargs
        yield from self.kwargs.values()

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
    value: Value

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class ReturnSignature:
    ret_by_ref: bool
    ret_type: Type

@dataclass(frozen=True)
class SpecializedCallSignature:
    generic_args: tuple[Value, ...]
    positional: tuple[tuple[str, SpecializedFormalArg], ...]
    varargs: tuple[SpecializedFormalArg, ...] | None
    kwargs: frozendict[str, SpecializedFormalArg] | None

    def __str__(self) -> str:
        """Note: return type not included"""
        generic = ", ".join(str(a) for a in self.generic_args)
        parts: list[str] = []
        parts.extend(str(a[1]) for a in self.positional)
        if self.varargs is not None:
            s = ", ".join(str(a) for a in self.varargs)
            parts.append(f"*({s})")
        if self.kwargs is not None:
            s = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
            parts.append(f"**{{{s}}}")
        return f"[{generic}]({', '.join(parts)})"


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
        for key, value in kwargs.items():
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
        return ArgList(tuple(bound), tuple(varargs_out), frozendict(kwargs_out))

    def solve_param_types(
        self, provided: ArgList[Type | None]
    ) -> tuple[Value, ...]:
        """The concrete value of every declared generic type parameter
        of one call, given ``provided``: the marshaled type of each
        argument the call provides, and ``None`` for a parameter whose
        default value applies (no argument was provided for it).

        A provided argument always types the parameter it is provided
        for (parameter annotations, concrete ones included, do not
        constrain a jit call - a call always resolves to the exact type
        of its arguments), but parameters annotated with the same type
        parameter must all be provided arguments whose types unify,
        which the type parameter solves to.  A missing argument can
        still solve a type parameter when its default value has a spy
        type."""
        assert len(provided.positional) == len(self.positional.by_id), 'argument count mismatch'
        # unify the type parameters over the provided arguments: two
        # arguments of parameters annotated with the same type parameter
        # must marshal to the same type
        solver = TypeVarSolver()
        for param, cand in zip(self.positional.by_id, provided.positional):
            declared = param.type
            if not isinstance(declared, TypeVar):
                continue
            if cand is not None:
                solver.add_constraint(cand, declared, True)
            elif param.default_value is not None:
                default_type = type_of(param.default_value)
                if default_type is not None:
                    solver.add_constraint(default_type, declared, True)
        if self.varargs is not None and isinstance(self.varargs.type, TypeVar):
            for cand in provided.varargs:
                if cand is not None:
                    solver.add_constraint(cand, self.varargs.type, True)
        if self.kwargs is not None and isinstance(self.kwargs.type, TypeVar):
            for cand in provided.kwargs.values():
                if cand is not None:
                    solver.add_constraint(cand, self.kwargs.type, True)
        solver.finish()
        solved = solver.get_solved()
        ret: list[Value] = []
        for type_var in self.generic_args:
            if type_var not in solved:
                raise TypeMismatchError(f"type variable {type_var.name} not solved")
            value = solved[type_var]
            ret.append(value)
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

    def specialize(self, provided: ArgList[Type | None]) -> tuple[SpecializedCallSignature, ReturnSignature | None]:
        """Specialize one call of this signature: the concrete typing of
        its arguments and (when the signature declares a return type)
        of its result.

        The declared generic type parameters are solved from ``provided``
        (see :meth:`solve_param_types`) and substituted into every
        annotation.  A parameter's type is then, in order of precedence:
        its (substituted) annotation, the marshaled type of the argument
        the call provides for it, or the spy type of its default value.
        The compiler decides the calling convention here too: a parameter
        whose type is a large aggregate, or whose formal declares it as a
        reference, is passed by reference (see ``sval.pass_by_ref``).  A
        zero-sized parameter is dropped from the runtime signature - it
        carries its unit value as a compile-time argument.

        Returns the specialized call signature - also the cache key of
        the specialization - and the return signature, or ``None`` when
        the signature declares no return type (the interpreter infers
        it from the body)."""
        type_var_values = self.solve_param_types(provided)
        reps: dict[TypeVar, Value] = dict(zip(self.generic_args, type_var_values))

        def substitute(type: Type) -> Type:
            replaced = replace_type_vars_type(type, reps)
            if isinstance(replaced, TypeVar):
                raise TypeMismatchError(f"type variable {replaced.name} is not solved")
            return replaced

        def resolve(name: str, param: SignatureFormalArg, cand: Type | None) -> SpecializedFormalArg:
            resolved: Type | None = None
            if param.type is not None:
                resolved = substitute(param.type)
            if resolved is None:
                resolved = cand
            if resolved is None and param.default_value is not None:
                resolved = type_of(param.default_value)
            if resolved is None:
                raise TypeMismatchError(
                    f"cannot determine the type of parameter '{name}'"
                )
            unit = resolved.get_unit_value()
            if unit is not None:
                assert isinstance(unit, Value)
                return SpecializedComptimeArg(unit)
            if param.is_comptime:
                raise TypeMismatchError(
                    f"compile-time parameter '{name}' must have a zero-sized type"
                )
            return SpecializedRuntimeArg(resolved, param.by_ref or pass_by_ref(resolved))

        positional = tuple(
            (name, resolve(name, param, cand))
            for (name, param), cand in zip(self.positional.items(), provided.positional)
        )

        varargs: tuple[SpecializedFormalArg, ...] | None = None
        if self.varargs is not None:
            formal = self.varargs
            varargs = tuple(
                resolve('*args', formal, cand) for cand in provided.varargs
            )

        kwargs: frozendict[str, SpecializedFormalArg] | None = None
        if self.kwargs is not None:
            kw = self.kwargs
            kwargs = frozendict(
                (name, resolve(name, kw, cand))
                for name, cand in provided.kwargs.items()
            )

        call_sig = SpecializedCallSignature(
            tuple(type_var_values), positional, varargs, kwargs
        )

        if self.ret_type is None:
            return call_sig, None
        ret_type = substitute(self.ret_type)
        ret_by_ref = self.ret_by_ref
        if ret_by_ref is None:
            ret_by_ref = returns_via_result_ptr(ret_type)
        return call_sig, ReturnSignature(ret_by_ref, ret_type)


@dataclass
class FunctionIR:
    name: str
    signature: Signature
    body: tuple[hir.Inst, ...]

@dataclass(eq=False)
class NativeFn:
    @abstractmethod
    def call(self, *values: ctypes._CDataType) -> ctypes._CDataType | None:
        ...

    @abstractmethod
    def print_all(self) -> list[str]:
        ...

@dataclass(eq=False)
class FunctionInstance:
    """The compiled artifact of one ``@jit`` specialization: its native
    function (what a Python-side call invokes, see :class:`NativeFn`)
    and the lowering result its spy function type yields - the call
    lowering plan a spy function body follows when it calls the
    specialization, together with its lowered MIR signature (see
    ``type.function_call_info``)."""

    mir: mir.Function
    ret_sig: ReturnSignature | None = None
    native_fn: NativeFn | None = None

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
        # the parsed HIR of the function (see ``astgen.parse_function``)
        self.hir = hir
        self.force_inline = force_inline
        # spy argument types -> the compiled artifacts of the
        # specialization (see ``LazyJitFunctionInstance``)
        self.specs: dict[SpecializedCallSignature, FunctionInstance] = {}
        # spy argument types -> error message of a failed compilation
        self.failed: dict[SpecializedCallSignature, str] = {}

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> Type:
        return self.hir.signature.as_non_generic_fn_type() or AnyFunction()

@dataclass
class SymbolTable:
    extern_anon_symbols: dict[NativeFn, mir.ExternAnonSymbol]
    newly_compiled: set[FunctionInstance]

    def add_to_mir(self, mir_mod: mir.Module) -> None:
        for anon_sym in self.extern_anon_symbols.values():
            mir_mod.add_recursively([anon_sym])
        for fn in self.newly_compiled:
            mir_mod.add_recursively([fn.mir])

    def compile(self, backend: Backend):
        mir_mod = mir.Module()
        self.add_to_mir(mir_mod)

        for instance in self.newly_compiled:
            # fold the trivial store/load slots of the freshly typed body
            # back into registers before it is lowered
            opt.simplify(instance.mir)

        native_fns = backend.compile(mir_mod) if self.newly_compiled else {}
        for instance in self.newly_compiled:
            instance.native_fn = native_fns[instance.mir]

class FunctionResolver:
    @abstractmethod
    def resolve_global(self, value: Any) -> AnyValue | None:
        """The spy value a global object referenced inside a function
        body resolves to.  A function registered in this host - reached
        as the raw function object or through the callable view its
        decorated name binds to - resolves to its function entry
        (creating the entry of an aot function that is not used yet);
        any other object is not a spy value of this host and returns
        ``None`` (the object stays a plain compile-time Python value)."""
        ...

class Backend:
    @abstractmethod
    def compile(self, module: mir.Module) -> dict[mir.GlobalValue, NativeFn]:
        ...

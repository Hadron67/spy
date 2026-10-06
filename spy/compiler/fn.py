from __future__ import annotations

import ctypes
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, override

from . import hir, mir, opt
from .errors import CompileError, TypeMismatchError
from .sval import (
    AnyFunction,
    AnyValue,
    ClosureType,
    FormalArg,
    FunctionType,
    MirLowerCache,
    ResultType,
    RetSpec,
    RetTuple,
    RetValue,
    Type,
    TypeVar,
    TypeVarSolver,
    Value,
    iter_ret_leaves,
    make_ret_spec,
    pass_by_ref,
    replace_type_vars_type,
    ret_by_value_index,
    ret_spec_value_is_empty,
    type_of,
)
from .target import TargetInfo
from .util import (
    ArraySet,
    FrozenArraySet,
    IndexedMap,
    StrBiMap,
    TriState,
    frozendict,
    sanitize_name,
)


@dataclass(frozen=True, slots=True)
class SignatureFormalArg:
    # The evaluated annotation of the parameter, in the spy domain (see
    # ``Signature``): a concrete spy type, a generic type parameter of
    # the signature (a ``TypeVar`` of ``Signature.generic_args``), or
    # None when the parameter is unannotated.
    type: Type | None
    is_comptime: bool
    # The evaluated default value of the parameter, in the spy domain
    # (see ``Signature``); None when the parameter has no default.
    default_value: AnyValue | None
    # Whether the parameter is passed by reference (i.e. as a constant
    # pointer): what the definition says about it (``sval.pass_by_ref`` of
    # the declared type), which is ``UNKNOWN`` when the annotation names a
    # type parameter - its layout is not known before a call substitutes it,
    # so the specialization decides (see ``Signature.specialize``).
    by_ref: TriState = TriState.UNKNOWN
    # Whether the annotation is ``type[X]``: the parameter takes a *type value*
    # (the ``type`` above is then ``X``, what the parameter solves to).  The
    # argument is the spy type the call passes, held as a compile-time value -
    # there is no runtime argument (see ``Signature.solve_param_types``).
    is_type_value: bool = False

    def map_type(self, f: Callable[[Type], Type]) -> SignatureFormalArg:
        return SignatureFormalArg(
            None if self.type is None else f(self.type),
            self.is_comptime,
            self.default_value,
            self.by_ref,
            self.is_type_value,
        )

@dataclass(frozen=True, slots=True)
class ArgEntry[T]:
    value: T
    is_ref: bool


# one argument of a call as ``Signature.solve_param_types``/``specialize`` see it:
# the spy type of the value, and - only for a ``type[X]`` parameter - the spy
# type its argument denotes (None otherwise)
type ProvidedArg = tuple[Type | None, Value | None]

def plain_provided_arg(type: Type | None) -> ProvidedArg:
    """A :data:`ProvidedArg` for an argument that is not a type value: the
    argument's spy type, with no denoted type."""
    return (type, None)


@dataclass(frozen=True, slots=True)
class RawArgList[T]:
    positional: tuple[T, ...]
    kwargs: frozendict[str, T]

    def map[K](self, f: Callable[[T], K]) -> RawArgList[K]:
        return RawArgList(
            tuple(f(p) for p in self.positional),
            frozendict((k, f(v)) for k, v in self.kwargs.items()),
        )

@dataclass(frozen=True, slots=True)
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

type ArgNode = Value | RuntimeArgNode | tuple[ArgNode, ...] | frozendict[str, ArgNode]

@dataclass(frozen=True, slots=True)
class RuntimeArgNode:
    type: Type
    by_ref: bool

class SpecializedFormalArg:
    pass

@dataclass(frozen=True, slots=True)
class SpecializedRuntimeArg(SpecializedFormalArg):
    type: Type
    is_ref: bool

    def __str__(self) -> str:
        return f"<{'&' if self.is_ref else ''}{self.type}>"

@dataclass(frozen=True, slots=True)
class SpecializedComptimeArg(SpecializedFormalArg):
    value: Value

    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True, slots=True)
class PartialReturnSignature:
    """The return convention of one specialization as far as the function's
    definition declares it: the return spec of its return annotation
    (``ret_type_spec``) and the exceptions it may raise (``exceptions``, in
    error-code order).  Either part is ``None`` when it is left to be inferred
    from the body (see ``HirRunner._finish_function``)."""

    ret_type_spec: RetSpec | None
    exceptions: ArraySet[Type] | None
    # the calling convention the function is declared with (see
    # ``Signature.callconv``): a non-default one forces the result by value
    # and forbids raising
    callconv: str = 'default'

    def is_complete(self) -> bool:
        """Whether both parts are declared - the convention can then be fixed
        before the body is typed (see ``HirRunner.run_function``)."""
        return self.ret_type_spec is not None and self.exceptions is not None

    def complete(self) -> ReturnSignature:
        """The complete return signature of this one, which requires both of
        its parts to be declared (see :meth:`is_complete`)."""
        ret_type_spec = self.ret_type_spec
        exceptions = self.exceptions
        assert ret_type_spec is not None and exceptions is not None, (
            'the return signature is not complete yet'
        )
        return ReturnSignature(ret_type_spec, exceptions, self.callconv)

@dataclass(frozen=True, slots=True)
class ReturnSignature:
    """The complete return convention of one specialization, *before* the error
    part is added: the return spec of the declared return type
    (``ret_type_spec``) and the exceptions the function may raise
    (``exceptions``, in error-code order; empty for a function that raises
    nothing).

    The *effective* spec - the one the lowered function and its callers work
    with, in which the error union is spread into its error code and its
    payload (see ``sval.make_ret_spec``) - is :meth:`ret_spec`.  A signature
    that still leaves a part to be inferred is a
    :class:`PartialReturnSignature` instead."""

    ret_type_spec: RetSpec
    exceptions: ArraySet[Type]
    # the calling convention the function is declared with (see
    # ``Signature.callconv``): a non-default one forces the result by value
    callconv: str = 'default'

    def ret_spec(self, cache: MirLowerCache) -> RetSpec:
        """The effective return spec: the declared return spec with the value's
        result type - its error code and its payload union - spread into the
        leaves.  The error part is always present - a function that raises
        nothing has the empty ``ResultType``, whose code and payload are
        zero-sized and so never reach the MIR.  ``cache`` is the MIR-mirror
        cache of the host the function is compiled for: which leaf is returned
        by value is decided by its layout (see ``sval.make_ret_spec``)."""
        return make_ret_spec(
            self.result_type(),
            cache,
            force_by_value=self.callconv != 'default',
        )

    def result_type(self) -> ResultType:
        """The result type of this function: the value it returns normally and
        the exceptions it may raise, which is what decides how its error codes
        are encoded (see :class:`sval.ResultType`)."""
        return ResultType(self.ret_type_spec.type, FrozenArraySet(self.exceptions.values))

    def value_is_empty(self) -> bool:
        """Whether the function has no value to return at all: its only value
        is of the empty type (a body that never delivers a result), so no
        ``return`` path can exist and its error codes carry no "no error"
        one."""
        return ret_spec_value_is_empty(self.ret_type_spec)

    def is_noreturn(self) -> bool:
        """Whether the function can never return: it has no value to return and
        raises nothing either, so its lowered form is a ``mir.NoReturn``
        function."""
        return self.value_is_empty() and len(self.exceptions) == 0

    def is_single_value(self) -> bool:
        """Whether the function returns exactly one value - the trivial case
        whose lowered signature is one plain result rather than a result
        pointer or a regrouped tuple.  It is decided by the *declared* return
        spec: an error part never makes a single value a group."""
        return isinstance(self.ret_type_spec, RetValue)

@dataclass(frozen=True, slots=True)
class CallSignature:
    generic_args: tuple[AnyValue, ...]
    positional: tuple[tuple[str, SpecializedFormalArg], ...]
    varargs: tuple[SpecializedFormalArg, ...] | None
    kwargs: frozendict[str, SpecializedFormalArg] | None
    # the hidden capture parameters of a closure (see ``ClosureFunction``):
    # independent of ``positional``/``varargs``/``kwargs``, so the declared
    # argument layout can grow ``*args``/``**kwargs`` without disturbing them
    captures: tuple[SpecializedFormalArg, ...] = ()
    # whether the callee may panic (see ``Signature.may_panic``): a call of it
    # may unwind through the enclosing deferred bodies (see ``interp``)
    may_panic: bool = True

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
        if len(self.captures) > 0:
            s = ", ".join(str(a) for a in self.captures)
            parts.append(f"@({s})")
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
    :class:`~spy.sval.AnyValue` (a plain ``None`` default is the null
    value, ``sval.Null()``, which is the absent value of an option)."""

    # the declared generic type parameters, by name: ``[T]`` declares
    # one entry ``T -> TypeVar('T')``
    generic_args: tuple[TypeVar, ...]
    # the formal parameters, by declaration position
    positional: IndexedMap[str, SignatureFormalArg]
    # the ``*args``/``**kwargs`` parameters (always None for now: spy
    # function definitions do not accept them yet, but the signature
    # model - and ``bind_arg_pos`` - already does)
    varargs: SignatureFormalArg | None
    kwargs: SignatureFormalArg | None
    # the evaluated return annotation, in the spy domain: the spy type of the
    # value the function returns, or a ``sval.TupleType`` when it returns
    # several (a ``tuple[...]``); None when no return annotation is written
    # and the return type is inferred from the body.  The return convention -
    # one ``sval.RetSpec`` tree - is resolved from it when a call is
    # specialized (see ``sval.make_ret_spec``)
    ret_type: Type | None
    # the exceptions the function may raise, in error-code order (tag ``i+1``
    # corresponds to the i-th one); an empty set for a function that raises
    # nothing, and None when the set is inferred from the body
    exceptions: ArraySet[Type] | None
    # the calling convention: ``'default'`` is the spy one; any other value
    # names a C one, in which every argument is passed by value, the result is
    # returned by value, and the function may not raise (see
    # ``sval.FunctionType``)
    callconv: str = 'default'
    # whether the function may panic (a call of one may unwind through the
    # enclosing deferred bodies); a function may panic by default
    may_panic: bool = True

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
        self, provided: ArgList[ProvidedArg]
    ) -> tuple[AnyValue, ...]:
        """The concrete value of every declared generic type parameter of
        one call.  ``provided`` carries one :data:`ProvidedArg` per argument
        the call provides - the marshal type of its value and, for a
        ``type[X]`` parameter, the spy type its argument denotes - and ``None``
        for a parameter the call leaves out (its default value applies).

        Every provided argument records a subtype constraint on the type
        parameter of the parameter it is provided for; ``finish`` solves
        each parameter to the peer type of those bounds.  A ``type[X]``
        parameter constrains ``X`` with the type its argument denotes
        (``denoted <: X``).  A parameter annotated with a concrete type
        constrains nothing here - it keeps its annotation, and the call
        specialized for it converts the argument to that type (see
        :meth:`specialize`).  A missing
        argument can still solve a type parameter when its default value
        has a spy type.  A parameter that stands for something other than
        a type - the length of an ``Array[T, N]`` - is solved to the value
        itself (the Python integer)."""
        assert len(provided.positional) == len(self.positional.by_id), 'argument count mismatch'
        # unify the type parameters over the provided arguments:
        # arguments of parameters annotated with the same type parameter
        # are recorded as subtype bounds, which the solver binds the
        # parameter to the peer type of
        solver = TypeVarSolver()
        for param, entry in zip(self.positional.by_id, provided.positional):
            declared = param.type
            if param.is_type_value:
                # ``type[X]``: the argument denotes a spy type, which is the
                # bound of ``X`` (usually a type parameter of the signature)
                if declared is not None and entry is not None and entry[1] is not None:
                    solver.add_constraint(entry[1], declared, True)
                continue
            cand = entry[0] if entry is not None else None
            # only an annotation that names a type parameter of this signature
            # constrains one - directly (``b: T``), or inside a generic type
            # (``p: Pair[T]``); any other annotation is just the type the
            # argument is converted to
            if declared is None or not any(declared.contains(tv) for tv in self.generic_args):
                continue
            if cand is not None:
                solver.add_constraint(cand, declared, True)
            elif param.default_value is not None:
                default_type = type_of(param.default_value)
                if default_type is not None:
                    solver.add_constraint(default_type, declared, True)
        if self.varargs is not None and isinstance(self.varargs.type, TypeVar):
            for entry in provided.varargs:
                cand = entry[0] if entry is not None else None
                if cand is not None:
                    solver.add_constraint(cand, self.varargs.type, True)
        if self.kwargs is not None and isinstance(self.kwargs.type, TypeVar):
            for entry in provided.kwargs.values():
                cand = entry[0] if entry is not None else None
                if cand is not None:
                    solver.add_constraint(cand, self.kwargs.type, True)
        solver.finish()
        solved = solver.get_solved()
        ret: list[AnyValue] = []
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
        exceptions: FrozenArraySet[Type] = (
            FrozenArraySet() if self.exceptions is None
            else FrozenArraySet(self.exceptions.values)
        )
        return FunctionType(
            tuple(formal), self.ret_type, exceptions, self.callconv, self.may_panic,
        )

    def substitute_type_vars(self, reps: dict[TypeVar, AnyValue]) -> Signature:
        """A copy of this signature with every type parameter of ``reps``
        replaced by its value.  A method of a generic struct names the
        struct's type parameters in its annotations (its ``self`` is typed as
        the struct template); a call binds them from the struct
        specialization the method was resolved on (see ``interp``), and this
        substitutes them before the signature is specialized."""
        if len(reps) == 0:
            return self

        def substitute(type: Type) -> Type:
            return replace_type_vars_type(type, reps)

        return replace(
            self,
            positional=self.positional.map(lambda arg: arg.map_type(substitute)),
            varargs=None if self.varargs is None else self.varargs.map_type(substitute),
            kwargs=None if self.kwargs is None else self.kwargs.map_type(substitute),
            ret_type=None if self.ret_type is None else substitute(self.ret_type),
            exceptions=None if self.exceptions is None else _substitute_exceptions(self.exceptions, substitute),
        )

    def specialize(self, provided: ArgList[ProvidedArg], cache: MirLowerCache) -> tuple[CallSignature, PartialReturnSignature]:
        """Specialize one call of this signature: the concrete typing of
        its arguments and the return convention this signature declares.

        The declared generic type parameters are solved from ``provided``
        (see :meth:`solve_param_types`) and substituted into every
        annotation.  A parameter's type is then, in order of precedence:
        A parameter's type is then, in order of precedence: its (substituted)
        annotation, the marshaled type of the argument the call provides for
        it, or the spy type of its default value.  Whether it is passed by
        reference is settled here from the type the call substitutes - what
        the formal says (``SignatureFormalArg.by_ref``, unknown while the
        type is still a type parameter) and what ``sval.pass_by_ref`` says of
        the substituted type, combined (see :meth:`util.TriState.or_`).  A
        zero-sized parameter is dropped from the runtime signature - it
        carries its unit value as a compile-time argument.  A signature that
        does not use the default calling convention (``callconv``) forces
        every argument by value instead.

        Returns the specialized call signature - also the cache key of
        the specialization - and the return convention as far as the
        definition declares it: a part it leaves out is missing (``None``),
        and the interpreter infers it from the body (see
        ``PartialReturnSignature``).  ``cache`` is the MIR-mirror cache of
        the host the call is compiled for: the return convention needs the
        layout a mirror carries (see ``sval.make_ret_spec``)."""
        is_c = self.callconv != 'default'
        type_var_values = self.solve_param_types(provided)
        reps: dict[TypeVar, AnyValue] = dict(zip(self.generic_args, type_var_values))

        def substitute(type: Type) -> Type:
            replaced = replace_type_vars_type(type, reps)
            if isinstance(replaced, TypeVar):
                raise TypeMismatchError(f"type variable {replaced.name} is not solved")
            return replaced

        def resolve(name: str, param: SignatureFormalArg, entry: ProvidedArg | None) -> SpecializedFormalArg:
            cand = entry[0] if entry is not None else None
            if param.is_type_value:
                # ``type[X]``: the parameter holds the spy type its argument
                # denotes, as a compile-time value - there is no runtime argument
                den = entry[1] if entry is not None else None
                if not isinstance(den, Type):
                    raise TypeMismatchError(
                        f"the argument of parameter '{name}' must be a spy type"
                    )
                return SpecializedComptimeArg(den)
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
            # the convention the formal declares (unknown while its type was a
            # type parameter) and the one the substituted type asks for: by
            # reference unless one of them says otherwise - and never for a C
            # convention, which passes every argument by value
            if is_c:
                by_ref = TriState.FALSE
            else:
                by_ref = TriState.or_(param.by_ref, pass_by_ref(resolved, cache))
            return SpecializedRuntimeArg(resolved, by_ref is not TriState.FALSE)

        positional = tuple(
            (name, resolve(name, param, entry))
            for (name, param), entry in zip(self.positional.items(), provided.positional)
        )

        varargs: tuple[SpecializedFormalArg, ...] | None = None
        if self.varargs is not None:
            formal = self.varargs
            varargs = tuple(
                resolve('*args', formal, entry) for entry in provided.varargs
            )

        kwargs: frozendict[str, SpecializedFormalArg] | None = None
        if self.kwargs is not None:
            kw = self.kwargs
            kwargs = frozendict(
                (name, resolve(name, kw, entry))
                for name, entry in provided.kwargs.items()
            )

        call_sig = CallSignature(
            tuple(type_var_values), positional, varargs, kwargs,
            may_panic=self.may_panic,
        )

        # the parts the definition declares; the interpreter infers the ones it
        # leaves out from the body
        if is_c:
            # a C function may not raise: a declared exception set has to be
            # empty, one left to be inferred is checked once the body is typed
            # (see ``HirRunner._materialize_ret_sig``)
            if self.exceptions is not None and len(self.exceptions) > 0:
                raise CompileError(
                    'a non-default-callconv function may not declare exceptions'
                )
            ret_type_spec = (
                None if self.ret_type is None
                else make_ret_spec(substitute(self.ret_type), cache, force_by_value=True)
            )
            exceptions = self.exceptions
        else:
            ret_type_spec = None if self.ret_type is None else make_ret_spec(substitute(self.ret_type), cache)
            exceptions = None if self.exceptions is None else _substitute_exceptions(self.exceptions, substitute)
        return call_sig, PartialReturnSignature(ret_type_spec, exceptions, self.callconv)


def _substitute_exceptions(exceptions: ArraySet[Type], substitute: Callable[[Type], Type]) -> ArraySet[Type]:
    """A copy of an exception set with every type substituted, keeping the
    error-code order."""
    ret: ArraySet[Type] = ArraySet()
    for exception in exceptions.values:
        ret.add(substitute(exception))
    return ret


def signature_of_fn_type(fn_type: FunctionType) -> Signature:
    """The :class:`Signature` a function-pointer type denotes: its formal
    parameters (by name, with their defaults) and its return convention.  A
    function type is always concrete, so the signature has no generic type
    parameters and no ``*args``/``**kwargs``; ``callconv`` and ``may_panic``
    are carried over.  The call logic rebuilds the :class:`CallSignature` and
    :class:`ReturnSignature` of one call through it (see ``interp``)."""
    positional: IndexedMap[str, SignatureFormalArg] = IndexedMap()
    for arg in fn_type.args:
        positional.add(
            arg.name,
            SignatureFormalArg(arg.type, False, arg.default_value, TriState.UNKNOWN),
        )
    exceptions: ArraySet[Type] = ArraySet()
    for exception in fn_type.exceptions:
        exceptions.add(exception)
    return Signature(
        (), positional, None, None, fn_type.return_type, exceptions,
        fn_type.callconv, fn_type.may_panic,
    )


@dataclass
class FunctionIR:
    name: str
    signature: Signature
    # for each positional parameter, whether the HIR binds its argument directly
    # as an address.  It only matters where the parameter is not already passed
    # by reference (``Signature.is_ref``): then True binds ``hir.Arg(i)`` to the
    # argument - the address of the value - instead of materializing the value
    # into a fresh slot.  A method's ``self`` (``self_by_value=False``) is such a
    # parameter: its signature type is ``Ptr[Self]`` and the HIR reads the
    # receiver out of the pointer (see ``interp``)
    arg_is_ref: tuple[bool, ...]
    body: tuple[hir.Inst, ...]

class NativeFn:
    @abstractmethod
    def print_all(self) -> list[str]:
        ...

    @abstractmethod
    def call(self, *args: ctypes._CDataType) -> ctypes._CDataType | None:
        ...

@dataclass(eq=False)
class FunctionInstance:
    """The compiled artifact of one specialization of a registered
    function: the lowered MIR function it was compiled into (``mir``),
    its return convention (``ret_sig``) and the native functions a
    Python-side call invokes - ``native_fn`` (the value form, called by
    spy-to-spy calls) and ``wrapper_fn`` (the always-generated
    Python-entry thunk, see ``_make_thunk``).

    ``call_sig`` is the specialized signature the instance was compiled
    for; ``thunk_call_sig``/``thunk_ret`` describe, in the spy domain, how
    the thunk is called and how it returns - what the Python-side marshaling
    (``glue``) reads (see ``_make_thunk``)."""

    mir: mir.Function
    ret_sig: ReturnSignature | None = None
    wrapper_fn: NativeFn | None = None
    native_fn: NativeFn | None = None
    # the specialized signature: set when the instance is created (see
    # ``interp.Analyser._request_function``)
    call_sig: CallSignature | None = None
    # the Python-entry calling convention: ``thunk_call_sig`` mirrors
    # ``call_sig`` with the runtime arguments whose aggregate form is passed
    # as a pointer there, and ``thunk_ret`` walks the result leaves with every
    # storage leaf delivered through a result pointer (the thunk returns void)
    thunk_call_sig: CallSignature | None = None
    thunk_ret: RetSpec | None = None

class FunctionValue(Value):
    """The function value of a registered function: only compiled - and
    thereby typed - when a call specializes it, so its type is the
    signature's :class:`~spy.sval.FunctionType` when that signature is
    complete and the untyped :class:`~spy.sval.AnyFunction` otherwise.

    Like :class:`~spy.sval.TypeVar` and :class:`~spy.sval.StructType`,
    the value doubles as the per-function entry of its host context (it
    is an identity object: two function values are equal only if they
    are the same object).  The call logic itself lives in the interpreter
    and the host, not here.
    """

    def __init__(self, name_base: str, hir: FunctionIR, force_inline: bool = False) -> None:
        # the context-unique base name of the native symbols
        self.name_base = name_base
        # the parsed HIR of the function (see ``astgen.parse_function``)
        self.hir = hir
        self.force_inline = force_inline
        # specialized call signatures -> the compiled artifacts of the
        # specialization
        self.specs: dict[CallSignature, FunctionInstance] = {}
        # specialized call signatures -> error message of a failed
        # compilation
        self.failed: dict[CallSignature, str] = {}

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def get_type(self) -> Type:
        return self.hir.signature.as_non_generic_fn_type() or AnyFunction()

@dataclass
class ClosureFunction:
    """The parsed body of one nested ``def``/``lambda`` (see
    :class:`~spy.compiler.hir.MakeClosure`).  It is shared by every closure
    value the statement creates.  The declared parameters and the captures are
    kept separate: ``arg_is_ref`` describes the declared ones only, while the
    captures are named by :class:`~spy.compiler.hir.Closure` and passed as the
    ``captures`` of a :class:`CallSignature`.

    It carries only a *local* name (``name``/``local_name``); the native symbol
    of a compiled closure is prefixed at creation with the name of the function
    the closure is created in (the specialization's name, which is only known
    when that function is compiled - see ``ClosureValue``)."""

    # the bare name of the nested function (``add``, ``<lambda>``), for error
    # messages
    name: str
    # a name unique among the closures of the builder (``add#1``), prefixed
    # with the creating frame's function name to form the native symbol
    local_name: str
    # the parsed body (see ``astgen``): its declared parameters are
    # ``hir.Arg(i)`` and its captures ``hir.Closure(i)``
    body: tuple[hir.Inst, ...]
    # the declared parameter names, in position (parallel to the annotation
    # and default operands of the ``hir.MakeClosure`` that creates it)
    param_names: tuple[str, ...]
    # whether each declared parameter is a compile-time one (``Comptime``)
    is_comptime: tuple[bool, ...]
    # whether each declared parameter's annotation is ``type[X]`` (a
    # type-valued parameter, see ``SignatureFormalArg.is_type_value``)
    is_type_value: tuple[bool, ...]
    # the ``*args``/``**kwargs`` names (None when not declared); their element/
    # value annotations are the ``vararg_annotation``/``kwarg_annotation``
    # operands of the ``hir.MakeClosure`` that creates the closure (evaluated in
    # the enclosing frame, like the positional ones)
    vararg_name: str | None
    kwarg_name: str | None
    # whether ``*args``/``**kwargs`` carry the ``Comptime`` marker
    vararg_is_comptime: bool
    kwarg_is_comptime: bool
    # for each declared positional parameter, whether the HIR binds its
    # argument directly as an address (see ``FunctionIR.arg_is_ref``)
    arg_is_ref: tuple[bool, ...]
    # the type parameters the closure itself declares (``[T]``), which a call
    # solves; a capture is never one of these
    generic_args: tuple[TypeVar, ...]
    # whether the closure is forced to be inlined at its call sites
    force_inline: bool
    # the declared exception set (None: inferred from the body)
    exceptions: ArraySet[Type] | None
    callconv: str = 'default'
    may_panic: bool = True

class ClosureValue(FunctionValue):
    """One closure value: the parsed :class:`ClosureFunction` together with
    the capture places the creating statement resolved and the concrete
    signature its annotations evaluated to (an annotation may name a type
    argument of the enclosing function, which is concrete per creation).

    It is a :class:`FunctionValue` so the call/compile machinery treats it
    like any other function entry; a call dispatches through
    ``interp._call_closure``, which inlines a forced-inline closure and
    compiles the others.  ``name_base`` is built by the ``hir.MakeClosure``
    that creates it, from the *creating frame's* function name (the
    specialization's name, with its signature) and the closure's local name -
    so a closure compiled inside ``foo(i32)`` is named ``foo(i32).add#1(...)``.
    Every creation caches its own specializations: two creations of one ``def``
    may have different concrete signatures (an annotation naming an enclosing
    type argument), which a shared cache keyed by the call signature alone
    could not tell apart."""

    def __init__(self, fn: ClosureFunction, hir_ir: FunctionIR, captures: tuple[Any, ...], name_base: str) -> None:
        super().__init__(name_base, hir_ir, force_inline=fn.force_inline)
        self.closure_fn = fn
        # the capture places, in the order of the closure's ``hir.Closure``
        # indices (interpreter values, kept untyped here - ``fn`` cannot name
        # ``interp``'s ``InterpVal``)
        self.captures = captures

    @override
    def get_type(self) -> Type:
        return ClosureType(self.hir.signature.as_non_generic_fn_type())

    def __repr__(self) -> str:
        return f'ClosureValue({self.hir.name})'

class SymbolTable:
    """The link names of every compiled function of the process, and the
    native function each name refers to.  A later compilation resolves the
    functions it imports from here."""

    def __init__(self):
        self._symbols: StrBiMap[NativeFn] = StrBiMap()

    def _assign_names(self, globals: set[mir.GlobalValue], extern_anon_symbols: dict[NativeFn, mir.ExternAnonSymbol]):
        ret: dict[mir.GlobalValue, str] = {}
        used: set[str] = set()

        ext_anon_sym_to_native_fn = {v: k for k, v in extern_anon_symbols.items()}

        # scan unrenamable globals first, so that if a renamable global
        # shares the same name as an unrenamable global, it can be properly renamed.
        for g in globals:
            name, can_rename = g.get_name()
            if not can_rename:
                if name in used or self._symbols.has_key(name):
                    raise CompileError(f"Duplicate global name: {name}")
                ret[g] = name
                used.add(name)

        for g in globals:
            name, can_rename = g.get_name()
            if can_rename:
                if isinstance(g, mir.ExternAnonSymbol):
                    ret[g] = self._symbols.get_key(ext_anon_sym_to_native_fn[g])
                else:
                    name = sanitize_name('__spy_' + name)
                    name = self._symbols.next_unique_name(name, extra_set=used)
                    ret[g] = name
                    used.add(name)
        return ret

    def _add(self, name: str, fn: NativeFn):
        self._symbols.add(name, fn)

def _must_pass_by_ref(type: mir.ReturnType) -> bool:
    """Whether a value of ``type`` crosses the native boundary as a
    pointer: ctypes cannot carry an aggregate (a struct, an array or a
    union) by value, so those are passed by pointer there."""
    return isinstance(type, (mir.StructType, mir.ArrayType, mir.UnionType))

def _storage_leaves(spec: RetSpec) -> list[RetValue]:
    """The leaves of ``spec`` that have storage, in depth-first declaration
    order: a zero-sized leaf is delivered as its unit value and needs no result
    location of its own."""
    return [leaf for leaf in iter_ret_leaves(spec) if leaf.type.get_unit_value() is None]


def _make_thunk(fn: mir.Function, ret_spec: RetSpec, cache: MirLowerCache) -> mir.Function:
    """The always-generated Python-facing entry of this function: a MIR
    function that adapts its value form to the ABI ctypes can call.  A by-value
    aggregate argument is taken as a pointer and loaded; every *storage* result
    leaf - the one the value form returns by value included - is delivered
    through a trailing out pointer, in depth-first declaration order (the thunk
    finally returns void).  Spy-to-spy calls never go through it: they call the
    value form directly.

    A future revision catches a panic here (see ``may_panic``), which is why
    every function has one."""
    # the value form's arguments are the declared ones followed by the result
    # pointers it was given for the leaves it delivers through one (see
    # ``interp._ret_leaf_ptr``)
    all_leaves = list(iter_ret_leaves(ret_spec))
    storage_leaves = _storage_leaves(ret_spec)
    result_ptr_count = sum(
        1 for leaf in all_leaves
        if leaf.type.get_unit_value() is None and leaf.via_result_ptr
    )
    n_declared = len(fn.args) - result_ptr_count
    assert n_declared >= 0

    arg_types: list[mir.Type] = []
    arg_names: list[str | None] = []
    by_refs: list[bool] = []
    for arg in fn.args[:n_declared]:
        br = _must_pass_by_ref(arg)
        by_refs.append(br)
        arg_types.append(mir.PointerType(arg) if br else arg)
        # the interpreter does not name the formals of a specialization
        arg_names.append(None)

    # one out pointer per storage leaf, in depth-first declaration order
    for leaf in storage_leaves:
        mir_type = leaf.type.to_mir_type(cache)
        assert mir_type is not None and not isinstance(mir_type, (mir.VoidType, mir.NoReturn)), \
            'a storage result leaf has a MIR type'
        arg_types.append(mir.PointerType(mir_type))
        arg_names.append('$result')

    thunk = mir.Function(
        f'{fn.name_base}.thunk', arg_types, arg_names, mir.VOID,
        is_complete=True,
    )

    # the call's arguments: the declared ones (an aggregate is loaded out of
    # the pointer the thunk took) followed by the value form's result pointers,
    # each fed from the thunk parameter of the leaf it delivers
    call_args: list[mir.Value] = []
    for i, (arg_type, by_ref) in enumerate(zip(arg_types[:n_declared], by_refs)):
        if by_ref:
            value = mir.Load(mir.Param(i, arg_type))
            thunk.entry.emit(value)
            call_args.append(value)
        else:
            call_args.append(mir.Param(i, arg_type))
    for position, leaf in enumerate(storage_leaves):
        if leaf.via_result_ptr:
            index = n_declared + position
            call_args.append(mir.Param(index, arg_types[index]))

    if isinstance(fn.ret_type, mir.NoReturn):
        # the callee never returns, so neither does the thunk: its entry ends
        # with the call (and a Python-side call of the function never returns
        # either)
        thunk.entry.emit(mir.Call(fn, tuple(call_args), mir.NORETURN))
        return thunk

    if isinstance(fn.ret_type, mir.VoidType):
        thunk.entry.emit(mir.Call(fn, tuple(call_args), mir.VOID))
    else:
        # the value form's one by-value leaf is stored into its own out pointer
        value = mir.Call(fn, tuple(call_args), fn.ret_type)
        thunk.entry.emit(value)
        by_value_index = ret_by_value_index(ret_spec)
        assert by_value_index is not None
        by_value_leaf = all_leaves[by_value_index]
        position = next(
            position for position, leaf in enumerate(storage_leaves)
            if leaf is by_value_leaf
        )
        index = n_declared + position
        thunk.entry.emit(mir.Store(mir.Param(index, arg_types[index]), value))
    thunk.entry.emit(mir.Ret(None))
    return thunk


def _thunk_ret(ret_spec: RetSpec) -> RetSpec:
    """The return convention the Python-side marshaling (``glue``) works with:
    the leaves of ``ret_spec`` with every storage leaf marked ``via_result_ptr``
    (the thunk always delivers it through an out pointer; a zero-sized leaf
    keeps its unit value and is not passed at all)."""
    match ret_spec:
        case RetValue():
            return RetValue(ret_spec.type, ret_spec.type.get_unit_value() is None)
        case RetTuple():
            return RetTuple(
                ret_spec.type, tuple(_thunk_ret(value) for value in ret_spec.values),
            )


def _thunk_call_sig(call_sig: CallSignature, fn: mir.Function) -> CallSignature:
    """The specialized call signature of the Python-entry thunk: ``call_sig``
    with every runtime argument whose aggregate form crosses the boundary as a
    pointer marked ``is_ref`` (its value is then the address)."""

    def convert(arg: SpecializedFormalArg, index: int) -> tuple[SpecializedFormalArg, int]:
        match arg:
            case SpecializedComptimeArg():
                return arg, index
            case SpecializedRuntimeArg():
                is_ref = arg.is_ref or _must_pass_by_ref(fn.args[index])
                return SpecializedRuntimeArg(arg.type, is_ref), index + 1
        raise AssertionError(f'unsupported specialized argument {arg!r}')

    positional: list[tuple[str, SpecializedFormalArg]] = []
    index = 0
    for name, arg in call_sig.positional:
        arg, index = convert(arg, index)
        positional.append((name, arg))
    varargs: tuple[SpecializedFormalArg, ...] | None = None
    if call_sig.varargs is not None:
        varargs_values: list[SpecializedFormalArg] = []
        for arg in call_sig.varargs:
            arg, index = convert(arg, index)
            varargs_values.append(arg)
        varargs = tuple(varargs_values)
    kwargs: frozendict[str, SpecializedFormalArg] | None = None
    if call_sig.kwargs is not None:
        kwargs_values: dict[str, SpecializedFormalArg] = {}
        for name, arg in call_sig.kwargs.items():
            kwargs_values[name], index = convert(arg, index)
        kwargs = frozendict(kwargs_values)
    return replace(call_sig, positional=tuple(positional), varargs=varargs, kwargs=kwargs)

@dataclass
class CompileBatch:
    extern_anon_symbols: dict[NativeFn, mir.ExternAnonSymbol]
    newly_compiled: set[FunctionInstance]
    # the MIR-mirror interning table of the analysis (see ``Analyser``): the
    # thunks are built from the layout a mirror carries
    mir_lower_cache: MirLowerCache

    def collect_symbols(self, extra: Iterable[mir.Function] = ()) -> set[mir.GlobalValue | mir.StructType]:
        entry: list[mir.GlobalValue] = list(self.extern_anon_symbols.values())
        for fn in self.newly_compiled:
            entry.append(fn.mir)
        entry.extend(extra)
        return mir.collect_symbols(entry)

    def compile(self, symbol_table: SymbolTable, backend: Backend, target: TargetInfo):
        thunks: dict[mir.Function, mir.Function] = {}
        cache = self.mir_lower_cache
        for instance in self.newly_compiled:
            ret_sig = instance.ret_sig
            assert ret_sig is not None
            ret_spec = ret_sig.ret_spec(cache)
            thunks[instance.mir] = _make_thunk(instance.mir, ret_spec, cache)
            instance.thunk_ret = _thunk_ret(ret_spec)
            assert instance.call_sig is not None
            instance.thunk_call_sig = _thunk_call_sig(instance.call_sig, instance.mir)

        # a Python-entry thunk is a function of the module like any other
        # (the host calls it through the symbol table), so it is collected
        # and lowered with the freshly typed functions
        mir_symbols = self.collect_symbols(thunks.values())

        for instance in self.newly_compiled:
            # fold the trivial store/load slots of the freshly typed body
            # back into registers before it is lowered
            opt.simplify(instance.mir)

        names = symbol_table._assign_names({a for a in mir_symbols if isinstance(a, mir.GlobalValue)}, self.extern_anon_symbols)

        structs: set[mir.StructType] = set()
        globals: StrBiMap[mir.GlobalValue] = StrBiMap()
        for sym in mir_symbols:
            if isinstance(sym, mir.StructType):
                structs.add(sym)
            elif isinstance(sym, mir.GlobalValue):
                globals.add(names[sym], sym)

        native_fns = backend.compile(structs, globals, target) if self.newly_compiled else {}
        for instance in self.newly_compiled:
            native_fn = native_fns[instance.mir]
            instance.native_fn = native_fn
            symbol_table._add(names[instance.mir], native_fn)
            instance.wrapper_fn = native_fns[thunks[instance.mir]]


class Backend:
    @abstractmethod
    def compile(self, structs: set[mir.StructType], globals: StrBiMap[mir.GlobalValue], target: TargetInfo) -> dict[mir.GlobalValue, NativeFn]:
        ...

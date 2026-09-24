from __future__ import annotations

import ctypes
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import override

from . import hir, mir, opt
from .errors import CompileError, TypeMismatchError
from .sval import (
    AnyFunction,
    AnyValue,
    FormalArg,
    FunctionType,
    RetSpec,
    RetValue,
    Type,
    TypeVar,
    TypeVarSolver,
    Value,
    iter_ret_leaves,
    make_ret_spec,
    pass_by_ref,
    replace_type_vars_type,
    type_of,
)
from .util import IndexedMap, StrBiMap, frozendict, sanitize_name


@dataclass(frozen=True, slots=True)
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

@dataclass(frozen=True, slots=True)
class ArgEntry[T]:
    value: T
    is_ref: bool


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
class ReturnSignature:
    """The return convention of one specialization: the whole return type as
    one :class:`sval.RetSpec` tree - a leaf (:class:`sval.RetValue`) carries
    the spy type of one value and whether it is delivered through a hidden
    result pointer, and a group (:class:`sval.RetTuple`) the ``tuple[...]`` a
    function (or a nested element of its annotation) returns, whose values the
    caller regroups.  At most one leaf is returned by value (see
    ``sval.make_ret_spec``)."""

    ret_spec: RetSpec

    def returned_type(self) -> Type | None:
        """The spy type of the value the lowered function returns directly
        (the by-value result), or ``None`` when it returns void - every
        value then goes through a result pointer, or is zero-sized."""
        for leaf in iter_ret_leaves(self.ret_spec):
            if not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
                return leaf.type
        return None

    def by_value_index(self) -> int | None:
        """The position of the one leaf returned by value (its value has
        storage and fits in registers) among all the leaves, in depth-first
        declaration order, or ``None`` when the lowered function returns
        void."""
        for index, leaf in enumerate(iter_ret_leaves(self.ret_spec)):
            if not leaf.via_result_ptr and leaf.type.get_unit_value() is None:
                return index
        return None

    def is_single_value(self) -> bool:
        """Whether the function returns exactly one value - the trivial case
        whose lowered signature is one plain result rather than a result
        pointer or a regrouped tuple."""
        return isinstance(self.ret_spec, RetValue)

@dataclass(frozen=True, slots=True)
class CallSignature:
    generic_args: tuple[AnyValue, ...]
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
    # the evaluated return annotation, in the spy domain: the whole return
    # type as one ``RetSpec`` tree (a ``RetValue`` for a single value, a
    # ``RetTuple`` - possibly nested - when it is a ``tuple[...]``); None when
    # no return annotation is written and the return type is inferred from
    # the body (see ``sval.make_ret_spec``)
    ret_spec: RetSpec | None

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
    ) -> tuple[AnyValue, ...]:
        """The concrete value of every declared generic type parameter of
        one call.  ``provided`` carries the marshaled type of each
        argument the call provides and ``None`` for a parameter the call
        leaves out (its default value applies).

        Every provided argument records a subtype constraint on the type
        parameter of the parameter it is provided for; ``finish`` solves
        each parameter to the peer type of those bounds.  A parameter
        annotated with a concrete type constrains nothing here - it keeps
        its annotation, and the call specialized for it converts the
        argument to that type (see :meth:`specialize`).  A missing
        argument can still solve a type parameter when its default value
        has a spy type.  A parameter that stands for something other than
        a type - the constness of a pointer (``Ptr[T, C]``) - is solved to
        the value itself (a ``bool``)."""
        assert len(provided.positional) == len(self.positional.by_id), 'argument count mismatch'
        # unify the type parameters over the provided arguments:
        # arguments of parameters annotated with the same type parameter
        # are recorded as subtype bounds, which the solver binds the
        # parameter to the peer type of
        solver = TypeVarSolver()
        for param, cand in zip(self.positional.by_id, provided.positional):
            declared = param.type
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
            for cand in provided.varargs:
                if cand is not None:
                    solver.add_constraint(cand, self.varargs.type, True)
        if self.kwargs is not None and isinstance(self.kwargs.type, TypeVar):
            for cand in provided.kwargs.values():
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
        if self.ret_spec is None:
            return True
        for arg in self.positional.by_id:
            if arg.type is None:
                return True
        return False

    def as_non_generic_fn_type(self) -> FunctionType | None:
        if self.is_generic():
            return None
        assert self.ret_spec is not None
        formal: list[FormalArg] = []
        for name, arg in self.positional.items():
            assert arg.type is not None
            formal.append(FormalArg(name, arg.type, arg.default_value))
        return FunctionType(tuple(formal), self.ret_spec.type)

    def substitute_type_vars(self, reps: dict[TypeVar, Value]) -> Signature:
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
            ret_spec=self.map_ret_spec(substitute) if self.ret_spec is not None else None,
        )

    def map_ret_spec(self, f: Callable[[Type], Type]) -> RetSpec:
        """The return spec of this signature with every declared result type
        replaced by ``f`` of it, re-resolving the by-value/result-pointer
        convention for the substituted types.  ``f`` is applied to the whole
        return type (a nested ``tuple[...]`` included), so it must substitute
        inside it (see ``replace_type_vars_type``)."""
        assert self.ret_spec is not None
        return make_ret_spec(f(self.ret_spec.type))

    def specialize(self, provided: ArgList[Type | None]) -> tuple[CallSignature, ReturnSignature | None]:
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
        reps: dict[TypeVar, AnyValue] = dict(zip(self.generic_args, type_var_values))

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

        call_sig = CallSignature(
            tuple(type_var_values), positional, varargs, kwargs
        )

        if self.ret_spec is None:
            return call_sig, None
        return call_sig, ReturnSignature(self.map_ret_spec(substitute))


@dataclass
class FunctionIR:
    name: str
    signature: Signature
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
    Python-side call invokes - ``native_fn``, and ``wrapper_fn`` (the
    Python-entry thunk) when the value form cannot be called through
    ctypes (see ``_needs_thunk``/:class:`NativeFn`)."""

    mir: mir.Function
    ret_sig: ReturnSignature | None = None
    wrapper_fn: NativeFn | None = None
    native_fn: NativeFn | None = None

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

def _must_pass_by_ref(type: mir.MayBeVoidType) -> bool:
    """Whether a value of ``type`` crosses the native boundary as a
    pointer: ctypes cannot carry an aggregate (a struct or an array) by
    value, so those are passed by pointer there."""
    return isinstance(type, (mir.StructType, mir.ArrayType))

def _needs_thunk(fn: mir.Function) -> bool:
    """Whether this function's value form cannot be called through
    ctypes directly, so that a Python-entry thunk is needed."""
    return _must_pass_by_ref(fn.ret_type) or any(
        _must_pass_by_ref(a) for a in fn.args
    )

def _make_thunk(fn: mir.Function) -> mir.Function:
    """The Python-facing entry of this function: a MIR function that
    adapts its value form to the ABI ctypes can call - a by-value
    aggregate argument is taken as a pointer and loaded, and a
    by-value aggregate result is written through a trailing out
    pointer (the thunk then returns void).  Spy-to-spy calls never go
    through it: they call this function directly."""
    arg_types: list[mir.Type] = []
    arg_names: list[str | None] = []
    by_refs: list[bool] = []
    for arg in fn.args:
        br = _must_pass_by_ref(arg)
        by_refs.append(br)
        arg_types.append(mir.PointerType(arg) if br else arg)
        # the interpreter does not name the formals of a specialization
        arg_names.append(None)

    out_arg: mir.Param | None = None
    ret_type: mir.MayBeVoidType = fn.ret_type
    if not isinstance(ret_type, mir.VoidType) and _must_pass_by_ref(ret_type):
        out_arg = mir.Param(len(arg_types), mir.PointerType(ret_type))
        arg_types.append(out_arg.type)
        arg_names.append('$result')
        ret_type = mir.VOID

    thunk = mir.Function(
        f'{fn.name_base}.thunk', arg_types, arg_names, ret_type,
        is_complete=True,
    )

    call_args: list[mir.Value] = []
    for i, (arg_type, by_ref) in enumerate(zip(arg_types, by_refs)):
        if by_ref:
            value = mir.Load(mir.Param(i, arg_type))
            thunk.entry.emit(value)
            call_args.append(value)
        else:
            call_args.append(mir.Param(i, arg_type))

    if out_arg is not None:
        assert not isinstance(fn.ret_type, mir.VoidType)
        value = mir.Call(fn, tuple(call_args), fn.ret_type)
        thunk.entry.emit(value)
        thunk.entry.emit(mir.Store(out_arg, value))
        thunk.entry.emit(mir.Ret(None))
    elif isinstance(fn.ret_type, mir.VoidType):
        thunk.entry.emit(mir.Call(fn, tuple(call_args), mir.VOID))
        thunk.entry.emit(mir.Ret(None))
    else:
        value = mir.Call(fn, tuple(call_args), fn.ret_type)
        thunk.entry.emit(value)
        thunk.entry.emit(mir.Ret(value))

    return thunk

@dataclass
class CompileBatch:
    extern_anon_symbols: dict[NativeFn, mir.ExternAnonSymbol]
    newly_compiled: set[FunctionInstance]

    def collect_symbols(self, extra: Iterable[mir.Function] = ()) -> set[mir.GlobalValue | mir.StructType]:
        entry: list[mir.GlobalValue] = list(self.extern_anon_symbols.values())
        for fn in self.newly_compiled:
            entry.append(fn.mir)
        entry.extend(extra)
        return mir.collect_symbols(entry)

    def compile(self, symbol_table: SymbolTable, backend: Backend):
        thunks: dict[mir.Function, mir.Function] = {}
        for instance in self.newly_compiled:
            if _needs_thunk(instance.mir):
                thunk = _make_thunk(instance.mir)
                thunks[instance.mir] = thunk

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

        native_fns = backend.compile(structs, globals) if self.newly_compiled else {}
        for instance in self.newly_compiled:
            native_fn = native_fns[instance.mir]
            instance.native_fn = native_fn
            symbol_table._add(names[instance.mir], native_fn)
            thunk = thunks.get(instance.mir)
            instance.wrapper_fn = native_fns[thunk] if thunk is not None else None


class Backend:
    @abstractmethod
    def compile(self, structs: set[mir.StructType], globals: StrBiMap[mir.GlobalValue]) -> dict[mir.GlobalValue, NativeFn]:
        ...

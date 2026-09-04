"""The user-facing DSL: ``JitContext`` with the ``jit`` and ``aot``
decorators.

A decorated function is *registered* in its context - its entry is
recorded and a thin callable view (``_FunctionView``) is bound to the
decorated name - and parsed (astgen) only when it is first used.  A
Python-side call goes through the view, which at call time

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. solves the concrete spy types of the parameters (in jit mode from
   the marshaled types of the arguments plus type-parameter unification,
   in aot mode from the annotations),
3. marshals every argument to the parameter types,
4. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see
``interp``); calling it is compiled to a native ``call``.
"""

from types import FunctionType
from typing import Any

from typing_extensions import override

from . import astgen, mir
from .builtins import AsValue
from .errors import CompileError, SpyError, TypeMismatchError
from .fn import FunctionEntry, FunctionValue, LazyJitFunction
from .interp import FunctionResolver, HirRunner, to_mir_type
from .lower import NativeFn, compile_module
from .mir import FormalArg as MirFormalArg
from .mir import Function as MirFunction
from .mir import FunctionType as MirFunctionType
from .mir import Symbol as MirSymbol
from .type import (
    BoolType,
    FloatType,
    FormalArg,
    IntType,
    PointerType,
    Type,
    Value,
    int_range,
    type_str,
    value_type,
)


def _sanitize(name: str) -> str:
    return ''.join(c if (c.isalnum() or c in '_.') else '_' for c in name)


def symbol_of(fn_name: str, arg_types: tuple[Type, ...]) -> str:
    """The native symbol name of one specialization.  ``fn_name`` must
    be the context-unique name allocated at registration (see
    ``JitContext._allocate_name``); the suffix of ``type_str`` strings
    distinguishes the specializations of one function."""
    parts = ['spy', _sanitize(fn_name)]
    parts.extend(_sanitize(type_str(t)) for t in arg_types)
    return '.'.join(parts)


def _types_str(arg_types: tuple[Type, ...]) -> str:
    return ', '.join(type_str(t) for t in arg_types)


def _candidate_type(fn_name: str, param_name: str, value: object) -> Type:
    """The spy type a provided Python argument marshals to."""
    if isinstance(value, AsValue):
        return value.type
    t = value_type(value)
    if t is None:
        raise TypeMismatchError(
            f"cannot pass a {type(value).__name__} value as the '{param_name}' argument "
            f"of {fn_name} (not a spy value)"
        )
    return t


def _marshal(fn_name: str, param_name: str, value: object, target: Type) -> object:
    """Convert one Python argument to the native calling convention of
    ``target`` (returns the value handed to the ctypes entry point)."""
    if isinstance(value, AsValue):
        if value.type != target:
            raise TypeMismatchError(
                f"type mismatch: spy.as(..., {type_str(value.type)}) cannot be passed "
                f"as the '{param_name}' argument of {fn_name} (expected "
                f"{type_str(target)})"
            )
        value = value.value

    def fail() -> TypeMismatchError:
        kind = 'a bool' if isinstance(value, bool) else f'a {type(value).__name__}'
        return TypeMismatchError(
            f"type mismatch: {kind} value cannot be passed as the '{param_name}' "
            f"argument of {fn_name} (expected {type_str(target)})"
        )

    match target:
        case BoolType():
            if not isinstance(value, bool):
                raise fail()
            return bool(value)
        case IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise fail()
            lo, hi = int_range(target)
            if not lo <= value <= hi:
                raise TypeMismatchError(
                    f"integer {value} is out of range for type {type_str(target)} "
                    f"in argument '{param_name}' of {fn_name}"
                )
            return int(value)
        case FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise fail()
            try:
                return float(value)
            except OverflowError:
                raise TypeMismatchError(
                    f"cannot convert {value} to {type_str(target)} in argument "
                    f"'{param_name}' of {fn_name}"
                ) from None
        case PointerType():
            if not isinstance(value, str):
                raise fail()
            return value.encode('utf-8')
        case _:
            raise TypeMismatchError(
                f"type {type_str(target)} is not supported for arguments of {fn_name} yet"
            )


class _FunctionView:
    """A callable view of a registered function.  The decorated name in
    a module (or an enclosing factory) scope binds to this view, so it
    is also the object a spy body references when it uses the function.
    Registration itself never parses the function: the concrete entry
    is looked up - and for an ``aot`` function created - on first use
    through the context (see ``JitContext._entry_of``)."""

    __slots__ = ('_context', '_fn', '_kind')

    def __init__(self, context: 'JitContext', fn: FunctionType, kind: str) -> None:
        self._context = context
        self._fn = fn
        self._kind = kind

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._context._dispatch(self._entry, args, kwargs)

    @property
    def _entry(self) -> FunctionEntry:
        """The entry of the registered function (created on first use
        for an aot function)."""
        return self._context._entry_of(self._fn)

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def __repr__(self) -> str:
        return f'<spy {self._kind} function {self._fn.__name__}>'


class JitContext(FunctionResolver):
    """A cache of compiled spy functions; functions decorated by the
    same context may call each other (as native calls)."""

    def __init__(self) -> None:
        self._entries: dict[FunctionType, FunctionEntry] = {}
        # registered ``aot`` functions that have not been used yet:
        # fn -> the preallocated base name of their native symbols.  An
        # aot entry fixes its signature from the annotations, which
        # needs a parse, so registration only records the function; the
        # entry is created when the function is first used (see
        # ``_entry_of``)
        self._pending_aot: dict[FunctionType, str] = {}
        # parsed HIR of plain Python functions that are only ever
        # inlined (inlined functions are not registered);
        # a registered function caches its HIR on the function
        # value instead
        self._inline_fn_hir_cache: dict[FunctionType, astgen.FunctionIR] = {}
        # allocated base names (``spy.<name>.<types>``) -> their owner
        # (the raw function object); keeps the native symbols of
        # different functions apart even when they happen to share a
        # ``__name__``
        self._name_owners: dict[str, FunctionType] = {}
        # the MIR functions defined by the module under construction
        # (see ``ensure_spec``), keyed like the per-function
        # ``mir_cache``; None outside a build
        self._module: dict[tuple[FunctionEntry, tuple[Type, ...]], MirFunction] | None = None

    # -- decorators ----------------------------------------------------------

    def jit(self, fn: FunctionType | None = None):
        """``@cache.jit()``: compile lazily at the first call, using the
        marshaled types of the arguments (annotations have no effect on
        the chosen argument types; type parameters like ``T`` unify the
        parameters, and a declared return annotation fixes the return
        type of the specialization)."""
        if fn is not None:
            return self._register(fn, 'jit')
        return lambda f: self._register(f, 'jit')

    def aot(self, fn: FunctionType | None = None):
        """``@cache.aot()``: compile lazily at the first use from the
        (concrete) type annotations, which are required for every
        parameter and for the return type.  Unlike a ``jit`` function
        an ``aot`` function has exactly one specialization, fixed by its
        annotations (a plain Python argument is still marshaled to the
        annotated type)."""
        if fn is not None:
            return self._register(fn, 'aot')
        return lambda f: self._register(f, 'aot')

    # -- Python-side calls ---------------------------------------------------

    def _dispatch(self, entry: FunctionEntry, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Bind Python arguments to the formal parameters of ``entry``
        and call the (possibly just compiled) specialization; this is
        what the callable view of a function value forwards to."""
        fn_ir = self.hir_of(entry.fn)
        name = entry.fn.__name__
        params = fn_ir.params
        param_names = [p.name for p in params]

        if len(args) > len(params):
            raise TypeError(
                f"{name}() takes {len(params)} positional arguments but {len(args)} were given"
            )
        present: dict[str, Any] = {}
        for i, value in enumerate(args):
            present[param_names[i]] = value
        for key, value in kwargs.items():
            if key not in param_names:
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
            if key in present:
                raise TypeError(f"{name}() got multiple values for argument '{key}'")
            present[key] = value

        provided: list[Type | None] = []
        for param in params:
            if param.name in present:
                provided.append(_candidate_type(name, param.name, present[param.name]))
            else:
                provided.append(None)
        formal = astgen.solve_call_types(fn_ir, entry.kind, tuple(provided))

        marshaled: list[Any] = []
        for i, param in enumerate(params):
            if param.name in present:
                value = present[param.name]
            else:
                if not param.has_default:
                    raise TypeError(f"{name}() missing required argument '{param.name}'")
                value = param.default_value
            marshaled.append(_marshal(name, param.name, value, formal[i]))

        spec = self.ensure_spec(entry, formal)
        return spec.call(*marshaled)

    # -- registry ------------------------------------------------------------

    def hir_of(self, fn: FunctionType) -> astgen.FunctionIR:
        # a registered function caches its HIR on its function value; a
        # plain function (which is only ever inlined) is cached by the
        # context
        entry = self._entries.get(fn)
        if entry is not None:
            ir = entry.hir
            if ir is None:
                ir = astgen.parse_function(fn)
                entry.hir = ir
            return ir
        ir = self._inline_fn_hir_cache.get(fn)
        if ir is None:
            ir = astgen.parse_function(fn)
            self._inline_fn_hir_cache[fn] = ir
        return ir

    def _native_spec(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> NativeFn | None:
        """The compiled native function of one specialization, or None
        when it is not (yet) compiled.

        An aot function has exactly one specialization, fixed by its
        annotations and lowered at its first use: its native function is
        stored on the value (the value also keeps the MIR, in
        ``mir_fn``)."""
        if isinstance(entry, FunctionValue):
            return entry.native_fn
        return entry.specs.get(arg_types)

    def ensure_spec(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> NativeFn:
        """Compile (or look up) the specialization of ``entry`` for
        ``arg_types`` and return its native function.

        Compilation happens module-at-a-time: a fresh function and every
        function it depends on that is not compiled yet are lowered
        together into one LLVM module (``define``s); specializations
        compiled in earlier modules are referenced as external symbols.
        """
        native = self._native_spec(entry, arg_types)
        if native is not None:
            return native
        message: str | None = None
        if not isinstance(entry, FunctionValue):
            # a jit function may be retried through its wrapper, so its
            # failures are cached; an aot function simply recompiles on
            # its next call
            message = entry.failed.get(arg_types)
        if message is not None:
            raise CompileError(message)
        if self._module is not None:
            raise CompileError('internal error: nested module compilation')

        self._module = {}
        try:
            self._compile_mir(entry, arg_types)
            return self._compile_module()[(entry, arg_types)]
        except SpyError as e:
            if not isinstance(entry, FunctionValue):
                entry.failed[arg_types] = str(e)
            raise CompileError(
                f"error while compiling {entry.fn.__name__}({_types_str(arg_types)}): {e}"
            ) from e
        finally:
            self._module = None

    def _compile_mir(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> MirFunction:
        """Produce (and cache) the typed MIR of one specialization.

        The function is created - with an empty body - and registered
        (in the cache of its entry and in the module under
        construction) *before* its body is run, so a call the body
        makes to it - recursion, direct or mutual - resolves to the
        very function being typed, whose signature is already fixed;
        running the body fills it in.  Calls to other functions may
        compile (MIR-wise) further functions, which are scheduled into
        the module under construction as well (see :meth:`ensure_spec`);
        the whole module is lowered at once.
        """
        key = (entry, arg_types)
        if isinstance(entry, FunctionValue):
            # an aot function has exactly one specialization: ``mir_fn``
            # is its cache slot
            fn = entry.mir_fn
        else:
            fn = entry.mir_cache.get(arg_types)
        if fn is not None:
            # the MIR is already being (or has been) typed, but no native
            # function is registered yet - the module build that produces
            # it is the one under construction (a recursive call, or a
            # build that never finished); the current build references
            # the function, so it is defined here too
            module = self._module
            if module is None:
                raise CompileError(
                    f'internal error: cached MIR of '
                    f'{entry.fn.__name__}({_types_str(arg_types)}) is reused '
                    'outside of a module build'
                )
            module[key] = fn
            return fn

        fn_ir = self.hir_of(entry.fn)
        assert len(arg_types) == len(fn_ir.params)
        # a declared return type - concrete, or naming a type parameter
        # bound by the parameters - fixes the return type of the
        # specialization (a recursive function needs it: its calls are
        # typed while its body is still being compiled); without one the
        # return type is inferred from the return sites
        ret_hint = self._declared_ret_type(entry, arg_types)
        args = tuple(
            MirFormalArg(fn_ir.params[i].name, to_mir_type(arg_types[i]))
            for i in range(len(arg_types))
        )
        fn = MirFunction(
            symbol_of(entry.name_base, arg_types),
            args,
            to_mir_type(ret_hint) if ret_hint is not None else None,
            [],
        )
        module = self._module
        if module is None:
            raise CompileError(
                f"internal error: {entry.fn.__name__}({_types_str(arg_types)}) "
                'is compiled outside of a module build'
            )
        if isinstance(entry, FunctionValue):
            entry.mir_fn = fn
        else:
            entry.mir_cache[arg_types] = fn
        module[key] = fn
        try:
            runner = HirRunner(self)
            runner.run_function(fn, fn_ir)
            return fn
        except BaseException:
            # the body typing failed: drop the partially-filled function
            # so that a retry starts from scratch (functions completed
            # earlier in this build stay cached)
            if isinstance(entry, FunctionValue):
                entry.mir_fn = None
            else:
                entry.mir_cache.pop(arg_types, None)
            raise

    def _compile_module(self) -> dict[tuple[FunctionEntry, tuple[Type, ...]], NativeFn]:
        """Lower the module under construction (see :meth:`ensure_spec`)
        into one native module and register its specializations."""
        assert self._module is not None
        natives = compile_module(list(self._module.values()), self._extern_symbols())
        by_name = {native.name: native for native in natives}
        result: dict[tuple[FunctionEntry, tuple[Type, ...]], NativeFn] = {}
        for key, mir_fn in self._module.items():
            native = by_name[mir_fn.name]
            entry, arg_types = key
            if isinstance(entry, FunctionValue):
                # an aot function has exactly one specialization: its
                # single native function is stored on the value (see
                # ``_native_spec``)
                entry.native_fn = native
            else:
                entry.specs[arg_types] = native
            result[key] = native
        return result

    @override
    def resolve_call(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> tuple[mir.Value, mir.Type]:
        """The callable value of one callee specialization as seen from
        inside a compiled function: a :class:`mir.Function` of the
        module under construction - already compiled, or still being
        compiled when the call is a recursive one - or a
        :class:`mir.Symbol` of a specialization compiled in an earlier
        module."""
        native = self._native_spec(entry, arg_types)
        if native is not None:
            fn_type = MirFunctionType(native.arg_types, native.ret_type)
            return MirSymbol(native.name, fn_type), native.ret_type
        fn = self._compile_mir(entry, arg_types)
        ret = fn.ret_type
        if ret is None:
            # the callee is still being typed (a recursive call) and has
            # no declared return type: the call cannot be typed
            raise CompileError(self._recursion_ret_type_error(entry, arg_types))
        return fn, ret

    @override
    def resolve_global(self, value: Any) -> Value | None:
        """The spy value a global object referenced inside a function
        body resolves to (see :class:`FunctionResolver`): a function
        registered in this context - reached as the raw function object
        or through the callable view its decorated name binds to -
        becomes its function entry, and anything else returns ``None``
        (the object stays a plain compile-time Python value)."""
        if isinstance(value, _FunctionView):
            # a view always belongs to the context that registered its
            # function
            if value._context is not self:
                return value._context.resolve_global(value._fn)
            value = value._fn
        if not isinstance(value, FunctionType):
            return None
        entry = self._entries.get(value)
        if entry is not None:
            return entry
        if value in self._pending_aot:
            # an aot function referenced before its first use: create
            # its entry (fixing the signature from the annotations)
            return self._entry_of(value)
        return None

    def _declared_ret_type(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> Type | None:
        """The concrete spy return type of one specialization declared by
        its annotations, or ``None`` when none is declared.

        An ``aot`` function fixes its signature at registration.  A
        ``jit`` function with a return annotation uses it as the return
        type of the specialization: the annotation is either a concrete
        spy type or a type parameter bound by the parameters (which
        binds it to the argument types of the specialization).
        """
        if isinstance(entry, FunctionValue):
            return entry.ret
        fn_ir = self.hir_of(entry.fn)
        ret_ann = fn_ir.ret_annotation
        if ret_ann is None:
            return None
        for type_param in fn_ir.type_params:
            if ret_ann is type_param:
                # the return type parameter is bound by the (concrete)
                # arguments of the parameters annotated with it
                for i, param in enumerate(fn_ir.params):
                    if param.annotation is ret_ann:
                        return arg_types[i]
                return None
        if not isinstance(ret_ann, Type):
            return None
        return ret_ann

    def _recursion_ret_type_error(self, entry: FunctionEntry, arg_types: tuple[Type, ...]) -> str:
        """Explain why the return type of a recursive specialization
        cannot be determined from its annotations."""
        fn_ir = self.hir_of(entry.fn)
        ret_ann = fn_ir.ret_annotation
        name = entry.fn.__name__
        if ret_ann is None:
            return (
                f"recursive function {name} requires a return type annotation: "
                'the return type of a recursive call must be known while '
                'the body is being compiled'
            )
        for type_param in fn_ir.type_params:
            if ret_ann is type_param:
                return (
                    f"cannot determine the return type of the recursive function {name}: "
                    f"type parameter {type_param.__name__} appears only in the return "
                    'annotation, not on any parameter'
                )
        return (
            f"cannot determine the return type of the recursive function {name}: "
            f'the return annotation {ret_ann!r} is not a spy type'
        )

    # -- internals -----------------------------------------------------------

    def _entry_of(self, fn: FunctionType) -> FunctionEntry:
        """The entry of the registered function ``fn``, creating it on
        first use.

        A jit entry needs no parse and exists from registration on.  An
        aot entry fixes its signature from the annotations, which needs
        a parse, so it is only created here - or in
        :meth:`resolve_global` when a spy body references the function -
        the first time the function is used.
        """
        entry = self._entries.get(fn)
        if entry is not None:
            return entry
        name = self._pending_aot.get(fn)
        if name is None:
            raise CompileError(f'internal error: {fn.__name__} is not a registered function')
        # resolve the fixed signature from the annotations
        fn_ir = astgen.parse_function(fn)
        params = fn_ir.params
        formal = astgen.solve_call_types(fn_ir, 'aot', (None,) * len(params))
        ret = astgen.return_annotation_type(fn_ir)
        args = tuple(FormalArg(params[i].name, formal[i]) for i in range(len(params)))
        entry = FunctionValue(fn, args, ret)
        entry.context = self
        entry.name_base = name
        # reuse the parse that was done for the signature above
        entry.hir = fn_ir
        self._entries[fn] = entry
        del self._pending_aot[fn]
        return entry

    def _register(self, fn: FunctionType, kind: str) -> _FunctionView:
        if not isinstance(fn, FunctionType):
            raise TypeError('spy decorators can only be applied to plain Python functions')
        if fn in self._entries or fn in self._pending_aot:
            raise ValueError(f'function {fn.__name__} is already registered in this JitContext')
        # registration never parses (astgen happens only when the
        # function is used): a jit entry needs no parse and is created
        # right away; an aot entry fixes its signature from the
        # annotations, which needs a parse, so only its native-name base
        # is recorded here (see ``_entry_of``)
        name = self._allocate_name(fn)
        if kind == 'aot':
            self._pending_aot[fn] = name
        else:
            entry = LazyJitFunction(fn)
            entry.context = self
            entry.name_base = name
            self._entries[fn] = entry
        return _FunctionView(self, fn, kind)

    def _allocate_name(self, fn: FunctionType) -> str:
        """The context-unique base name of the native symbols of ``fn``.

        Two different functions registered in the same context may share
        a ``__name__`` (e.g. same-named functions of two modules, or two
        instances of a nested function); their symbols must still not
        collide, because extern calls are linked by name.  The plain
        ``__name__`` is kept when it is free; otherwise the
        module-qualified name is used, then numbered suffixes."""
        candidates: list[str] = [_sanitize(fn.__name__)]
        qualified = _sanitize(f'{fn.__module__}.{fn.__qualname__}')
        if qualified not in candidates:
            candidates.append(qualified)
        name: str | None = None
        for candidate in candidates:
            if candidate not in self._name_owners:
                name = candidate
                break
        if name is None:
            n = 2
            while True:
                candidate = f'{qualified}.{n}'
                if candidate not in self._name_owners:
                    name = candidate
                    break
                n += 1
        assert name is not None
        self._name_owners[name] = fn
        return name

    def _extern_symbols(self) -> dict[str, NativeFn]:
        """The addresses of the native functions compiled in *earlier*
        modules, keyed by their symbol names: everything a module under
        construction may reference from outside (see
        :meth:`_compile_module`).  Every registered function of the
        context contributes its compiled specializations."""
        externs: dict[str, NativeFn] = {}
        for entry in self._entries.values():
            if isinstance(entry, FunctionValue):
                native = entry.native_fn
                if native is not None:
                    externs[native.name] = native
            else:
                for native in entry.specs.values():
                    externs[native.name] = native
        return externs

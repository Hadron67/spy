"""The user-facing DSL: the ``func`` and ``struct`` decorators.

A function decorated with ``func`` is *registered* in its host context - the
registration binds a callable handle (``_RegisteredFn``) to the decorated
name - and parsed (astgen) only when it is first used.  A Python-side call
goes through the handle, which at call time

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. specializes the signature: the declared generic type parameters are
   solved from the marshaled argument types, and each parameter then
   takes its (substituted) annotation, the marshaled type of the
   argument provided for it, or the spy type of its default value, in
   that order,
3. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

A *class* decorated with ``struct`` declares a spy struct type of the same
name: the annotated attributes of its body are the fields, in declaration
order, and the functions of its body are its methods (a decorated one is a
registered function, compiled into a native call, and a plain one is inlined
at its call sites).  The decorated name stands for that struct: a spy body
annotates with it, constructs it (``Foo(a, b)`` - the arguments fill the
fields in place, positional ones in declaration order and keyword ones by
name; a custom ``__init__`` is not supported) and calls its methods on a
value of it (``x.m()``, the object passed as the method's ``self``) or
through the class name (``Foo.m(x, ...)``, every argument passed explicitly,
``self`` included; a ``@staticmethod`` takes none).  Its methods include the
ones a plain class or ``Protocol`` base contributes (``std.mem.Allocator``),
which a subclass such as ``DynamicAllocator`` therefore inherits.  A class
with type parameters (``class Foo[T]``) declares a struct *template*:
``Foo[i32]`` names one specialization of it, a construction of the bare
template (``Foo(...)``) takes the arguments of the specialization from the
type of the location it is built into, and a method call carries the
specialization's type arguments into the method (``x.m()`` behaves like
``typeof(x).m(x)``, see ``interp``).  A struct is a compile-time type only:
Python-side construction is not supported yet.

A decorated function used from inside another spy function body is
resolved to its function entry when the reference runs (see ``interp``);
calling it is compiled to a native ``call``.  An *undecorated* plain
Python function reached the same way is inlined instead.  The ``spy.*``
builtins (``spy.compile_log``) are evaluated at compile time; ``spy.typeof``
is a ``syntax`` marker, lowered by the parser to a type probe.
"""

import inspect
import types as pytypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NoDefault,
    TypeVar,
    cast,
    dataclass_transform,
    override,
)

from . import astgen, glue, sval
from .builtins import spy_as, spy_compile_log
from .errors import CompileError
from .fn import (
    AnyValue,
    Backend,
    CallSignature,
    FunctionValue,
    PartialReturnSignature,
    RawArgList,
    SymbolTable,
)
from .interp import Analyser
from .lower import LLVMBackend
from .sval import CompileContext, MirLowerCache, StructDecl
from .target import TargetInfo
from .util import FrozenArraySet, IndexedMap, frozendict


def _builtin_signature(fn: Callable[..., Any]) -> sval.BuiltinSignature:
    """The :class:`sval.BuiltinSignature` of a builtin Python function, read
    off its ``def``: the positional parameters in declaration order (each as
    ``(name, default_value)``, or ``None`` for one with no default), whether it
    takes ``*args`` and whether it takes ``**kwargs``.  A keyword-only
    parameter is unsupported."""
    positional: IndexedMap[str, tuple[str, AnyValue | None]] = IndexedMap()
    varargs = False
    kwargs = False
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            varargs = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            default = None if param.default is inspect.Parameter.empty else param.default
            positional.add(name, (name, default))
        else:
            raise CompileError(
                f"the builtin {fn.__name__} may not take a keyword-only parameter"
            )
    return sval.BuiltinSignature(positional, varargs, kwargs)


# the ``spy.*`` builtins, by the name the interpreter knows them by
_BUILTINS: dict[Any, sval.BuiltinFn] = {
    spy_compile_log: sval.BuiltinFn('compile_log', _builtin_signature(spy_compile_log)),
    spy_as: sval.BuiltinFn('as', _builtin_signature(spy_as)),
}


def builtin_func[T](fn: T) -> T:
    """Register a Python function as a spy builtin: the decorated name is bound
    to a :class:`sval.BuiltinFn` (not to the function itself), which the
    compile-time interpreter evaluates by name while running the HIR (see
    ``interp._call_builtin``).  Unlike the ``spy.*`` builtins of
    :mod:`spy.compiler.builtins`, which the host recognizes by object identity,
    a ``@builtin_func`` builtin lives in a ``std`` module and is dispatched on
    its own ``__name__``.  The builtin's ``def`` is also read into a
    :class:`sval.BuiltinSignature` (see ``_builtin_signature``), which the
    interpreter binds every call's arguments with before evaluating it."""
    return cast(T, sval.BuiltinFn(cast(Any, fn).__name__, _builtin_signature(cast(Any, fn))))


@dataclass(frozen=True)
class FnMetadata:
    sfv: bool  # self by value
    extern: bool
    linkname: str | None
    # an undecorated struct method: it is inlined at its call sites like a
    # plain Python function instead of being compiled into a native call
    inline: bool = False
    # the exceptions the function may raise: ``None`` (the default) means it
    # raises nothing, ``"infer"`` that they are inferred from the body, and a
    # tuple of spy struct classes the exceptions it may raise (in error-code
    # order)
    exceptions: tuple[type, ...] | Literal["infer"] | None = None
    # the calling convention (see ``sval.FunctionType.callconv``): ``'default'``
    # is the spy one, any other value names a C one
    callconv: str = 'default'
    # whether the function may panic (a call of one may unwind through the
    # enclosing deferred bodies); a function may panic by default
    may_panic: bool = True


@dataclass(frozen=True, slots=True)
class FuncTypeMetadata:
    """The ``@func_type(...)`` declaration of a spy function *type* (see
    ``_FuncTypeDecl``)."""

    callconv: str
    may_panic: bool
    exceptions: tuple[type, ...] | Literal["infer"] | None


@dataclass(frozen=True)
class StructMetadata:
    extern_c: bool


# the metadata of an undecorated method (see ``_RegisteredClass.get_entry``)
_INLINE_META = FnMetadata(sfv=False, extern=False, linkname=None, inline=True)


def _normalize_exceptions(
    exceptions: type | tuple[type, ...] | Literal["infer"] | None,
) -> tuple[type, ...] | Literal["infer"] | None:
    """The ``exceptions`` declaration of ``@func``/``@func_type``: a single
    exception type is taken as the one-element tuple, so
    ``@func(exceptions=ErrorA)`` and ``@func(exceptions=(ErrorA,))`` mean the
    same thing; ``None`` and ``"infer"`` are kept as they are."""
    if exceptions is None:
        return None
    if isinstance(exceptions, str):
        return 'infer'
    if isinstance(exceptions, (tuple, list)):
        return tuple(exceptions)
    return (exceptions,)


class _RegisteredFn:
    def __init__(self, fn, cls, meta: FnMetadata, context: _Context) -> None:
        self.fn = fn
        self.cls = cls
        self.meta = meta
        self.entry: FunctionValue | None = None
        self.context = context

        # The type parameters of an enclosing context, such as a generic
        # class: the Python type parameter object an annotation evaluates to
        # -> its spy value (see ``astgen.parse_function``)
        self.context_type_vars: dict[TypeVar, sval.Value] = {}

    def with_context(self, context: _Context) -> _RegisteredFn:
        # the handle of the same Python function in ``context``: its compiled
        # artifacts (and its native symbols) belong to that context, so a
        # handle resolved from another context must be re-bound there (see
        # ``_Context.resolve_global``).  The enclosing context's type
        # parameters carry over, so that a method still resolves the struct's
        # annotations in the scope it was written in.
        clone = _RegisteredFn(self.fn, self.cls, self.meta, context)
        clone.context_type_vars = dict(self.context_type_vars)
        return clone

    def __call__(self, *args, **kwds):
        entry = self.get_entry()
        arglist = entry.hir.signature.bind_arg_pos(
            RawArgList(
                tuple(glue.boundary_arg(a, self.context) for a in args),
                frozendict((k, glue.boundary_arg(v, self.context)) for k, v in kwds.items()),
            ),
            lambda e: e,
        )
        arg_types = glue.provided_arglist(entry.hir.signature, arglist)
        call_sig, ret_sig = entry.hir.signature.specialize(
            arg_types, self.context.mir_lower_cache,
        )

        analyser = Analyser(self.context, self.context.mir_lower_cache)
        analyser.analyse_function(entry, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self.context._symbol_table, self.context.backend, self.context.target_info())

        return glue.invoke(entry.specs[call_sig], call_sig, arglist, self.context)

    def get_entry(self):
        if self.entry is None:
            hir = astgen.parse_function(
                self.fn, self.context, self.cls,
                self.meta.sfv, self.context_type_vars, self.meta.exceptions,
                self.meta.callconv, self.meta.may_panic,
            )
            self.entry = FunctionValue(self.fn.__qualname__, hir, force_inline=self.meta.inline)
        return self.entry

class _RegisteredClass(StructDecl):
    """One class decorated with ``@struct()``, bound to its name in place of
    the class itself: the declaration of one spy struct.  The struct type the
    class names is built from the class body - the annotated class attributes
    are the fields, in declaration order, and the functions of the class body
    are its methods - and is cached on the handle (``get_entry``)."""

    def __init__(self, cls: type, context: _Context, meta: StructMetadata) -> None:
        self.cls = cls
        self.context = context
        self.meta = meta
        self.entry: sval.StructTypeHead | None = None
        # the declared generic type parameters of the class, by the Python
        # type parameter object their annotations evaluate to (see
        # ``get_entry``)
        self.class_type_vars: dict[TypeVar, sval.Value] = {}

    def get_entry(self) -> sval.StructTypeHead:
        if self.entry is None:
            if '__init__' in self.cls.__dict__:
                raise CompileError(
                    f'struct {self.cls.__name__} cannot declare __init__: custom '
                    f'constructors are not supported; a construction initializes '
                    f'the fields directly (a keyword argument names one)'
                )
            # the declared generic type parameters (PEP 695 ``[T]``): one spy
            # ``sval.TypeVar`` per parameter, which the annotations of the
            # class and of its methods name
            generic_args: list[sval.TypeVar] = []
            # the declared default of every type parameter (``[C: bool =
            # Literal[False]]``): a use that leaves the trailing arguments out
            # takes them (see ``sval.StructTypeHead.specialize``)
            generic_defaults: list[sval.AnyValue | None] = []
            for type_param in getattr(self.cls, '__type_params__', ()):
                if not isinstance(type_param, TypeVar):
                    raise CompileError(
                        f'unsupported type parameter {type_param!r} of struct '
                        f'{self.cls.__name__}'
                    )
                spy_type = sval.TypeVar(type_param.__name__)
                generic_args.append(spy_type)
                self.class_type_vars[type_param] = spy_type
                declared_default = getattr(type_param, '__default__', NoDefault)
                generic_defaults.append(
                    None if declared_default is NoDefault
                    else sval.as_value(declared_default, self.context)
                )
            head = sval.StructTypeHead(
                self.cls.__name__,
                tuple(generic_args),
                modifiers=sval.StructModifiers(extern_c=self.meta.extern_c),
                generic_defaults=tuple(generic_defaults),
            )
            # the head is bound before the class body is read: a field may
            # name the struct itself, or one of its methods ``self``
            self.entry = head
            # the annotations are evaluated lazily by Python, in the
            # annotation scope of the class: a subscripted struct template in
            # one of them (``inner: Pair[T]``) evaluates to an application
            # that resolves against the class' type parameters here
            for name, annotation in self.cls.__annotations__.items():
                type = sval.as_value(annotation, self.context, self.class_type_vars)
                if type is None or not isinstance(type, sval.Type):
                    raise CompileError(f'cannot convert annotation {annotation!r} to a value')
                # the value the class body assigned to the annotated attribute
                # is the field's default, which a construction leaves it at when
                # it is not given (``None`` - no value written - is the null
                # value, so it marks a default like any other; a field with no
                # default has no entry in the class body at all)
                default: sval.AnyValue | None = None
                if name in self.cls.__dict__:
                    default = sval.as_value(
                        self.cls.__dict__[name], self.context, self.class_type_vars,
                    )
                head.add_field(name, type, default)
            # the ``self`` of every method is the struct *template*: a
            # specialization whose type arguments are the struct's own
            # parameters, so that a call substitutes the arguments of the
            # specialization the method was resolved on into it (see
            # ``interp``)
            template = head.specialize(tuple(generic_args))
            # the methods, collected from the whole MRO (base classes first, so
            # that an override wins): a plain class or ``Protocol`` base - such
            # as ``std.mem.Allocator`` - contributes its functions as methods
            # (``DynamicAllocator`` inherits its ``new``/``deinit``/... from
            # ``Allocator``), while a ``@staticmethod`` takes no receiver.  The
            # machinery of ``typing``/``object`` and their ``__init__`` are not
            # methods and are skipped.
            for base in reversed(self.cls.__mro__):
                if base is object or base.__module__ in ('typing', 'builtins'):
                    continue
                for name, value in base.__dict__.items():
                    if name == '__init__':
                        continue
                    is_static = isinstance(value, staticmethod)
                    if is_static:
                        value = value.__func__
                    if isinstance(value, _RegisteredFn):
                        # a registered method: ``self`` is the struct it
                        # belongs to (the method is parsed with that type, see
                        # astgen).  A handle registered in another context is
                        # re-bound here, so that the method is parsed and
                        # compiled with this context's struct type and type
                        # parameters
                        if value.context is not self.context:
                            value = value.with_context(self.context)
                        value.cls = None if is_static else template
                        value.context_type_vars = self.class_type_vars
                        head.methods[name] = value
                    elif isinstance(value, pytypes.FunctionType):
                        # an undecorated method: inlined at its call sites like
                        # a plain function.  It is wrapped like a registered one
                        # so that it is parsed lazily with the struct as its
                        # ``self`` type and the struct's type parameters in
                        # scope (a static one takes no ``self``)
                        method = _RegisteredFn(
                            value, None if is_static else template, _INLINE_META, self.context,
                        )
                        method.context_type_vars = self.class_type_vars
                        head.methods[name] = method
                    else:
                        continue
                    if is_static:
                        head.static_methods.add(name)
                    else:
                        head.static_methods.discard(name)
        return self.entry

    def __getitem__(self, key: Any) -> glue.SpecializedStruct:
        """``Foo[i32]``: an application of the struct template to generic
        arguments, which is also callable (a construction).  Python evaluates an
        annotation lazily, in the annotation scope of the annotated function or
        class, so this is what a subscripted struct *annotation* evaluates to;
        the arguments name the type parameters of that scope, which
        ``__getitem__`` does not see - ``sval.as_value`` turns the application
        into the struct specialization once it is given the scope (see
        :class:`sval.StructTypeApplication`).  A ``Foo[i32]`` used as an
        expression inside a body is the HIR's ``hir.Subscript`` instead,
        resolved by the interpreter.

        A *class-name method access* (``Foo[i32].m(x)``) resolves the same
        specialization through this path (see ``interp``)."""
        return glue.specialized_struct(self, key, self.context)

    def as_spy_value(self) -> sval.AnyValue:
        """The spy value of this class: the struct type it declares, built in
        the context that resolves it (see ``_Context.resolve_global``).  The
        name of a *generic* struct stands for its template, which a call has to
        specialize."""
        entry = self.get_entry()
        if len(entry.generic_args) == 0:
            return entry.specialize(())
        return entry

    def __getattr__(self, name: str) -> Any:
        """``Foo.m``: the method ``m`` of this struct, called from Python with
        no implicit ``self`` (``Foo.m(x, ...)`` passes every argument, ``self``
        included)."""
        method = self.get_entry().methods.get(name)
        if method is None:
            raise AttributeError(f'{self.cls.__name__} has no method {name!r}')
        return method

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """``Foo(...)``: construct a spy struct value from Python.  The
        specialization of a generic struct is inferred from the provided field
        values (or written explicitly as ``Foo[i32](...)``)."""
        return glue.construct(self.get_entry(), None, args, kwds, self.context)


class _FuncTypeDecl:
    """One class decorated with ``@func_type()``: the declaration of a spy
    function *type*.

    Like a ``@struct()`` class, the decorated name is a handle the context it
    is resolved in turns into an actual spy value - here a
    ``sval.FunctionType`` built from the annotations of the class' ``__call__``
    (the receiver is dropped).  The declaration is resolved lazily because its
    annotations name objects of the resolving context (a ``std`` struct, whose
    type is the context's own copy).

    A function type is dynamically sized: a parameter may use it directly (a
    DST argument is passed by reference), while a local or a struct field has
    to name a pointer to it (``ConstPtr[...]``).
    """

    def __init__(self, cls: type, meta: FuncTypeMetadata, context: _Context) -> None:
        self.cls = cls
        self.meta = meta
        self.context = context
        self.entry: sval.FunctionType | None = None

    def with_context(self, context: _Context) -> _FuncTypeDecl:
        # this context's copy of the declaration (see ``_Context._local_func_type``)
        return _FuncTypeDecl(self.cls, self.meta, context)

    def as_spy_value(self) -> sval.FunctionType:
        """The function type this declaration names, built in this handle's
        context."""
        if self.entry is not None:
            return self.entry
        call = self.cls.__call__
        if getattr(call, '__code__', None) is None:
            raise CompileError(
                f'function type {self.cls.__name__} must declare a Python __call__'
            )
        self.entry = _build_fn_type(
            call, self.context, drop_receiver=True,
            callconv=self.meta.callconv, may_panic=self.meta.may_panic,
            exceptions=self.meta.exceptions,
            what=f'function type {self.cls.__name__}',
        )
        return self.entry

    def __or__(self, other: Any) -> sval.TaggedUnionApplication:
        return sval.union_application(self, other)

    def __ror__(self, other: Any) -> sval.TaggedUnionApplication:
        return sval.union_application(other, self)


def _build_fn_type(
    fn: Callable[..., Any],
    context: _Context,
    drop_receiver: bool,
    callconv: str,
    may_panic: bool,
    exceptions: tuple[type, ...] | Literal["infer"] | None,
    what: str,
) -> sval.FunctionType:
    """The ``sval.FunctionType`` the annotations of ``fn`` describe: every
    parameter (the receiver is dropped when ``drop_receiver``, for a protocol's
    ``__call__``) and the return.  The annotations resolve in ``context``, so a
    ``std`` struct they name is that context's copy.  ``exceptions`` is the
    declared exception set (``"infer"`` is rejected: a declared function type
    has no body to infer from); ``callconv``/``may_panic`` are carried over.
    ``what`` names the declaration in the errors.

    A trailing ``*args`` (unannotated) declares a C-variadic function, and is
    only accepted for a non-default ``callconv`` - a ``...`` in the signature,
    whose extra arguments are passed by value with their own types (see
    ``sval.FunctionType.varargs``).  ``**kwargs`` has no C meaning and is
    rejected."""
    code = getattr(fn, '__code__', None)
    if code is None:
        raise CompileError(f'{what} must declare a Python function')
    count = code.co_argcount
    names = code.co_varnames[:count]
    annotations = fn.__annotations__
    has_varargs = bool(code.co_flags & inspect.CO_VARARGS)
    if code.co_flags & inspect.CO_VARKEYWORDS:
        raise CompileError(f'{what} may not declare **kwargs')
    if has_varargs:
        if callconv == 'default':
            raise CompileError(
                f'a default-callconv {what} may not declare *args'
            )
        vararg_name = code.co_varnames[count]
        if vararg_name in annotations:
            raise CompileError(
                f'the *args of {what} may not be annotated: a C-variadic '
                f'function passes every extra argument with its own type'
            )
    defaults = fn.__defaults__ if fn.__defaults__ is not None else ()
    offset = count - len(defaults)
    args: list[sval.FormalArg] = []
    for i, name in enumerate(names):
        if drop_receiver and i == 0:
            # the receiver of a protocol method is not a parameter of the
            # function type
            continue
        if name not in annotations:
            raise CompileError(f'every parameter of {what} must be annotated')
        arg_type = sval.as_value(annotations[name], context)
        if not isinstance(arg_type, sval.Type):
            raise CompileError(
                f'cannot use {annotations[name]!r} as the type of parameter '
                f'{name!r} of {what}'
            )
        default: sval.AnyValue | None = None
        if i >= offset:
            default = sval.as_value(defaults[i - offset], context)
        args.append(sval.FormalArg(name, arg_type, default))
    if 'return' not in annotations or annotations['return'] is None:
        ret: sval.AnyValue = sval.VoidType()
    else:
        ret = sval.as_value(annotations['return'], context)
    if not isinstance(ret, sval.Type):
        raise CompileError(
            f'cannot use {annotations.get("return")!r} as the return type of {what}'
        )
    if exceptions == 'infer':
        raise CompileError(f'the exceptions of {what} cannot be inferred')
    exception_types: list[sval.Type] = []
    if exceptions is not None:
        for exception in exceptions:
            value = sval.as_value(exception, context)
            if not isinstance(value, sval.StructType):
                raise CompileError(
                    f'cannot use {exception!r} as an exception of {what}: '
                    f'an exception must be a spy struct'
                )
            exception_types.append(value)
    if callconv != 'default' and len(exception_types) > 0:
        raise CompileError(f'a non-default-callconv {what} may not declare exceptions')
    return sval.FunctionType(
        tuple(args), ret, FrozenArraySet(exception_types), callconv, may_panic,
        has_varargs,
    )


@dataclass(frozen=True, slots=True)
class DeclFuncMetadata:
    """The ``@decl_func(...)`` declaration of an external function (see
    ``_DeclFuncDecl``)."""

    linkname: str
    callconv: str
    may_panic: bool
    exceptions: tuple[type, ...] | Literal["infer"] | None


class _DeclFuncDecl:
    """One function decorated with ``@decl_func(linkname)``: the declaration of
    an external function.

    The decorated name is a handle the context it is resolved in turns into a
    ``sval.DeclareFunction`` - the signature read off the annotations, the link
    name the one declared - which lowers to a ``mir.ExternSymbol`` (see
    ``interp``).  The declaration is resolved lazily because its annotations
    name objects of the resolving context."""

    def __init__(self, fn: Any, meta: DeclFuncMetadata, context: _Context) -> None:
        self.fn = fn
        self.meta = meta
        self.context = context
        self.entry: sval.DeclareFunction | None = None

    def with_context(self, context: _Context) -> _DeclFuncDecl:
        # this context's copy of the declaration (see ``_Context._local_decl_func``)
        return _DeclFuncDecl(self.fn, self.meta, context)

    def as_spy_value(self) -> sval.DeclareFunction:
        """The declared function this handle names, built in its context."""
        if self.entry is not None:
            return self.entry
        fn_type = _build_fn_type(
            self.fn, self.context, drop_receiver=False,
            callconv=self.meta.callconv, may_panic=self.meta.may_panic,
            exceptions=self.meta.exceptions,
            what=f'declared function {self.fn.__name__!r}',
        )
        self.entry = sval.DeclareFunction(fn_type, self.meta.linkname)
        return self.entry


class _Context(CompileContext):
    def __init__(self, backend: Backend, target: TargetInfo | None = None) -> None:
        self.backend = backend
        # the parameters of the compile target this context compiles for (the
        # pointer size, which the layout of every type depends on)
        target_info = TargetInfo() if target is None else target
        self._fn_anotation_cache: dict[Any, _RegisteredFn] = {}
        self._cls_annotation_cache: dict[type, _RegisteredClass] = {}
        # the ``@func_type()`` declarations, by the class that carries them
        self._func_type_cache: dict[type, _FuncTypeDecl] = {}
        # the ``@decl_func()`` declarations, by the function that carries them
        self._decl_func_cache: dict[Any, _DeclFuncDecl] = {}
        # the inline entries of the undecorated Python functions reached
        # from a spy body, by function object
        self._inline_cache: dict[Any, FunctionValue] = {}
        self._symbol_table = SymbolTable()
        # the MIR-mirror interning table of this context, shared by every
        # analysis it runs and bound to the target it compiles for (see
        # ``sval.MirLowerCache``)
        self.mir_lower_cache = sval.MirLowerCache(target_info)
        # the ``std`` types the type rules need, resolved lazily on the first
        # request (see ``sval.SpecialTypes``)
        self._special_types: sval.SpecialTypes | None = None

    def _local_fn(self, handle: _RegisteredFn) -> _RegisteredFn:
        # this context's handle of the Python function ``handle`` names: a
        # context registers one handle per Python function, so a handle that
        # belongs to another context is re-bound here and cached, so that the
        # function is compiled (and its native symbols named) in this context
        existing = self._fn_anotation_cache.get(handle.fn)
        if existing is None:
            existing = handle.with_context(self)
            self._fn_anotation_cache[handle.fn] = existing
        return existing

    def _local_class(self, handle: _RegisteredClass) -> _RegisteredClass:
        # likewise for a struct class: the struct type this context declares
        # for it, with this context's own method handles (see
        # ``_RegisteredClass.get_entry``)
        existing = self._cls_annotation_cache.get(handle.cls)
        if existing is None:
            existing = _RegisteredClass(handle.cls, self, handle.meta)
            self._cls_annotation_cache[handle.cls] = existing
        return existing

    def _local_func_type(self, handle: _FuncTypeDecl) -> _FuncTypeDecl:
        # likewise for a function-type declaration: its annotations resolve in
        # this context, so that a ``std`` struct it names is this context's copy
        existing = self._func_type_cache.get(handle.cls)
        if existing is None:
            existing = handle.with_context(self)
            self._func_type_cache[handle.cls] = existing
        return existing

    def _local_decl_func(self, handle: _DeclFuncDecl) -> _DeclFuncDecl:
        # likewise for a declared function (see ``_DeclFuncDecl``)
        existing = self._decl_func_cache.get(handle.fn)
        if existing is None:
            existing = handle.with_context(self)
            self._decl_func_cache[handle.fn] = existing
        return existing

    @override
    def resolve_global(self, value: Any) -> AnyValue | None:
        match value:
            case _RegisteredFn():
                # a handle of another context is re-bound here, so that a spy
                # body only ever reaches this context's functions and structs
                handle = value if value.context is self else self._local_fn(value)
                return handle.get_entry()
            case _RegisteredClass():
                handle = value if value.context is self else self._local_class(value)
                return handle.as_spy_value()
            case _FuncTypeDecl():
                handle = value if value.context is self else self._local_func_type(value)
                return handle.as_spy_value()
            case _DeclFuncDecl():
                handle = value if value.context is self else self._local_decl_func(value)
                return handle.as_spy_value()
            case pytypes.FunctionType():
                builtin = _BUILTINS.get(value)
                if builtin is not None:
                    return builtin
                # an undecorated plain Python function: it is inlined where
                # it is called (it contributes no native specialization)
                entry = self._inline_cache.get(value)
                if entry is None:
                    hir = astgen.parse_function(value, self)
                    entry = FunctionValue(value.__qualname__, hir, force_inline=True)
                    self._inline_cache[value] = entry
                return entry
            case _:
                # any other object stays a plain compile-time Python value
                return None

    @override
    def target_info(self) -> TargetInfo:
        return self.mir_lower_cache.target

    def _struct_head(self, handle: Any) -> sval.StructTypeHead:
        # the head this context declares for a ``std`` struct: the handle is
        # re-bound to this context (see ``resolve_global``), so the struct type
        # rules reach is this context's own
        head = self.resolve_global(handle)
        assert isinstance(head, sval.StructTypeHead)
        return head

    @override
    def special_types(self) -> sval.SpecialTypes:
        if self._special_types is None:
            from ..std import core
            self._special_types = sval.SpecialTypes(
                self._struct_head(core.slice),
                self._struct_head(core.SlicePtr),
                self._struct_head(core.ConstSlicePtr),
            )
        return self._special_types

    @override
    def mir_cache(self) -> MirLowerCache:
        return self.mir_lower_cache

    def _resolve_call(
        self, fn: FunctionValue, call_sig: CallSignature, ret_sig: PartialReturnSignature
    ):
        analyser = Analyser(self, self.mir_lower_cache)
        analyser.analyse_function(fn, call_sig, ret_sig)
        sym = analyser.finish()
        sym.compile(self._symbol_table, self.backend, self.target_info())

    def func(self, sfv: bool = False, extern: bool = False, linkname: str | None = None, exceptions: type | tuple[type, ...] | Literal["infer"] | None = None, callconv: str = 'default', may_panic: bool = True):
        meta = FnMetadata(
            sfv=sfv, extern=extern, linkname=linkname,
            exceptions=_normalize_exceptions(exceptions),
            callconv=callconv, may_panic=may_panic,
        )

        def wrapper[T](fn: T) -> T:
            if fn in self._fn_anotation_cache:
                return self._fn_anotation_cache[fn]
            result = _RegisteredFn(fn, None, meta, self)
            self._fn_anotation_cache[fn] = result
            return cast(T, result)
        return wrapper

    def func_type(self, callconv: str = 'default', may_panic: bool = True, exceptions: type | tuple[type, ...] | Literal["infer"] | None = None):
        """Declare a spy function *type*: the decorated ``Protocol`` names the
        signature through its ``__call__`` (whose receiver is dropped), and
        the decorated name is the handle that resolves to a
        ``sval.FunctionType`` (see ``_FuncTypeDecl``)."""
        meta = FuncTypeMetadata(
            callconv=callconv, may_panic=may_panic,
            exceptions=_normalize_exceptions(exceptions),
        )

        def wrapper[T](cls: type[T]) -> type[T]:
            if cls in self._func_type_cache:
                return cast(type[T], self._func_type_cache[cls])
            result = _FuncTypeDecl(cls, meta, self)
            self._func_type_cache[cls] = result
            return cast(type[T], result)
        return wrapper

    def decl_func(self, linkname: str | None = None, exceptions: type | tuple[type, ...] | Literal["infer"] | None = None, callconv: str = 'c', may_panic: bool = True):
        """Declare an external function of the link name ``linkname``: the
        decorated function names the signature through its annotations, and the
        decorated name is the handle that resolves to a
        ``sval.DeclareFunction`` (see ``_DeclFuncDecl``).  Its arguments are the
        ones of ``@func_type``, with the link name first and ``callconv``
        defaulting to the C one."""

        def wrapper[T: pytypes.FunctionType](fn: T) -> T:
            name = linkname
            if name is None:
                name = cast(str, fn.__name__)
            meta = DeclFuncMetadata(
                linkname=name, callconv=callconv, may_panic=may_panic,
                exceptions=_normalize_exceptions(exceptions),
            )
            if fn in self._decl_func_cache:
                return cast(T, self._decl_func_cache[fn])
            result = _DeclFuncDecl(fn, meta, self)
            self._decl_func_cache[fn] = result
            return cast(T, result)
        return wrapper

    @dataclass_transform()
    def struct(self, extern_c: bool = False):
        meta = StructMetadata(extern_c=extern_c)

        def wrapper[T](cls: type[T]) -> type[T]:
            if cls in self._cls_annotation_cache:
                return self._cls_annotation_cache[cls]
            result = _RegisteredClass(cls, self, meta)
            self._cls_annotation_cache[cls] = result
            return cast(type[T], result)
        return wrapper

_GLOBAL_CONTEXT = _Context(LLVMBackend())

func = _GLOBAL_CONTEXT.func
struct = _GLOBAL_CONTEXT.struct
func_type = _GLOBAL_CONTEXT.func_type
decl_func = _GLOBAL_CONTEXT.decl_func

"""Lowering of Python source to the untyped HIR (``hir``).

``parse_function`` turns the source of a Python function into a
:class:`FunctionIR`: its signature - the declared generic type
parameters (PEP 695 ``[T]``), the formal parameters and the return
annotation - and the translated body.  The parameter annotations,
default values and return annotation are read off the function object
itself (``fn.__annotations__``/``fn.__defaults__``/...), where Python
has already evaluated them at definition time, so the source
expressions are never re-evaluated; like every compile-time object they
are stored in the spy domain (``sval.as_value``, with the declared
type parameters converted to ``sval.TypeVar``s).  The signature analysis
itself - typing one call from its arguments - lives with
:class:`Signature`/:class:`FunctionIR` in ``fn``, not here.

Like ``symlat.jit.llvm`` the body is one *linear* list of instructions;
expression evaluation appends temporary instructions to the list and
returns the instruction object whose register holds the value - or,
with a result location (RLS, see ``_Builder._gen_expr``), writes the
value into a caller-provided slot and returns nothing.  The result
location of a function itself (``hir.ResultLoc``) is the target of its
``return`` statements: ``return expr`` evaluates ``expr`` with the
function's result location, so a call in return position writes its
result straight into the location the function returns through.

``astgen`` performs *almost* all name resolution.  Since every parameter
is addressable, the translated body starts with an
``Alloca``/``Store`` prologue per parameter (storing the by-value
``Arg(i)``), and a read of a parameter becomes a ``Load`` of its Alloca.
Local variables are addressable the same way: ``name = expr`` declares a
block-local variable - a fresh ``Alloca`` - when ``name`` is not yet
bound in the current block, and stores into the existing slot
otherwise.  Every ``if`` body is a lexical block of its own (a child of
the enclosing block): a declaration inside it shadows outer bindings
within the block and is invisible after it.  Global names - everything
that is not a variable in scope - are resolved here to their Python
objects.  Every global is an *immutable value*: in a value context a
read embeds the object as a ``hir.Const`` leaf; in a reference context
(``is_ref``, e.g. the callee of a call) it embeds a ``hir.ConstRef`` -
a const reference to the value (see ``_gen_name``).  A name captured
from an enclosing Python scope (a spy function may be defined inside a
factory) is read from its closure cell the same way.  Attributes on
such compile-time objects (``spy.typeof``, ``spy.u64``, ...) are
evaluated here as well.  Whether a function object denotes a registered
spy function - and which function value it stands for - is decided by
the interpreter when a call runs: a function body may be parsed before
its callees, or even itself (an aot function parses its own body while
it is being registered), are registered.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, TypeVar, cast

from . import hir, sval
from .errors import CompileError
from .fn import ArgEntry, FunctionIR, RawArgList, Signature, SignatureFormalArg
from .sval import (
    AnyValue,
    Type,
    Value,
    Void,
    VoidType,
    as_value,
)
from .sval import (
    TypeVar as SpyTypeVar,
)
from .util import IndexedMap, frozendict

_BIN_OPS = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
}

_BOOL_OPS = {ast.And: 'and', ast.Or: 'or'}

_UNARY_OPS = {ast.USub: 'neg', ast.Not: 'not'}

_CMP_OPS = {
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
}


class _Scope:
    """One lexical block of a spy function: the variable bindings of the
    block (name -> the Alloca of its slot), chained to the enclosing
    block.  A *read* resolves through the chain; an assignment binds in
    the current block: into the slot of a name the block already holds,
    or - the first ``=`` on a name - into a fresh block-local slot that
    shadows any outer binding of the same name.  A declaration is never
    visible outside its block."""

    __slots__ = ('bindings', 'parent')

    def __init__(self, parent: _Scope | None) -> None:
        self.parent = parent
        self.bindings: dict[str, hir.Value] = {}

    def lookup(self, name: str) -> hir.Value | None:
        """The Alloca of the nearest binding of ``name``, or None when
        the name is not bound in this or any enclosing block."""
        scope = self
        while scope is not None:
            slot = scope.bindings.get(name)
            if slot is not None:
                return slot
            scope = scope.parent
        return None


class _Builder:
    """Translates the AST of one function body into one linear
    instruction list of the untyped HIR.

    Each builder translates one *block* - the function body, or the
    body of one ``if`` branch - and carries the lexical scope of that
    block: a child of the enclosing block's scope whose bindings (the
    parameters, for the function body; local declarations, in every
    block) are added as the block is translated.  Expression builders
    of nested blocks look up names through the chain.
    """

    def __init__(self, fn: Any, fn_ir: FunctionIR, scope: _Scope) -> None:
        self.fn = fn
        self._fn_ir = fn_ir
        self._scope = scope
        self.insts: list[hir.Inst] = []

    def add(self, inst: hir.Inst) -> hir.Inst:
        self.insts.append(inst)
        return inst

    # -- statements -----------------------------------------------------------

    def _gen_stmt(self, node: ast.stmt) -> None:
        fn_name = self._fn_ir.name
        match node:
            case ast.Return():
                # the return expression is generated into the function's
                # result location (result-location semantics): a call in
                # return position writes its result straight into the
                # location the function returns through, instead of
                # materializing a temporary value first
                if node.value is not None:
                    self._gen_result_loc(node.value, hir.ResultLoc())
                self.add(hir.Ret())
            case ast.Pass():
                pass
            case ast.Expr():
                self._gen_value(node.value)
            case ast.Assign():
                if len(node.targets) != 1:
                    raise CompileError(
                        f"chained assignments are not supported yet in spy function {fn_name}"
                    )
                self._gen_assign(node.targets[0], node.value)
            case ast.AugAssign():
                self._gen_augassign(node)
            case ast.If():
                # the branches are generated into the same flat list,
                # delimited by the ``Else``/``End`` markers (WASM-style)
                cond = self._gen_value(node.test)
                self.add(hir.If(cond))
                self._gen_branch(node.body)
                if len(node.orelse) > 0:
                    self.add(hir.Else())
                    self._gen_branch(node.orelse)
                self.add(hir.End())
            case _:
                raise CompileError(
                    f"unsupported statement {type(node).__name__} in spy function {fn_name}"
                )

    def _gen_branch(self, stmts: list[ast.stmt]) -> None:
        """Translate one branch body of an ``if``, appending its
        instructions to this builder's list (between the ``If``/``Else``
        and ``End`` markers).  A branch is a lexical scope of its own -
        a child of the enclosing scope - so declarations inside it
        shadow outer bindings and are not visible after the block."""
        sub = _Builder(self.fn, self._fn_ir, _Scope(self._scope))
        for stmt in stmts:
            sub._gen_stmt(stmt)
        self.insts.extend(sub.insts)

    # -- variables ------------------------------------------------------------

    def _gen_assign(self, target: ast.expr, value: ast.expr) -> None:
        """One ``target = expr`` statement.  The first ``=`` on a name
        declares a block-local variable (a fresh slot, shadowing any
        outer binding); later assignments in the block only store into
        its slot.  A call on the right hand side writes its result
        straight into the target slot (result-location semantics): a
        constructor ``x = Bar(...)`` fills the fields of the slot in
        place, and a scalar call result is only recorded in it."""
        emit_commit = False
        if isinstance(target, ast.Name):
            slot = self._scope.bindings.get(target.id)
            if slot is None:
                # the slot is bound before the initializer is generated, so
                # a self-referencing declaration (``y = y + 1``) reads the
                # not-yet-stored slot - a compile error when it runs, like
                # an unbound local - instead of silently reading an outer
                # ``y``
                slot = self.add(hir.Alloca())
                self._scope.bindings[target.id] = slot
                emit_commit = True
        lhs = self._gen_ref(target)
        self.add(hir.Store(lhs, self._gen_value(value)))
        if emit_commit:
            self.add(hir.CommitSlot(lhs))

    def _gen_augassign(self, node: ast.AugAssign) -> None:
        """One ``name += expr`` statement: read the value, add ``expr``
        and store the result back.  The target is a variable slot or the
        address of a field of a runtime struct value (``self.h += e``);
        ``+=`` never declares: it requires the name to be declared."""
        fn_name = self._fn_ir.name
        if not isinstance(node.op, ast.Add):
            raise CompileError(f"only '+=' is supported yet in spy function {fn_name}")

        lhs = self._gen_ref(node.target)
        rhs = self._gen_arg(node.value)
        self.add(hir.Binary('+', ArgEntry(lhs, True), rhs, lhs))

    # -- expressions ----------------------------------------------------------

    def _resolve_closure(self, name: str) -> Any | None:
        """The raw object of the name ``name`` captured from an
        enclosing Python scope (a spy function may be defined inside a
        factory, e.g. ``def make(k): @cache.jit() def f(x): return x *
        k``), or None when the name is not a free variable.  A captured
        variable behaves like a global: the value of its closure cell at
        parse time is embedded as a compile-time constant."""
        fn = self.fn
        closure = fn.__closure__
        if closure is not None:
            for i, free_var in enumerate(fn.__code__.co_freevars):
                if free_var == name:
                    try:
                        return closure[i].cell_contents
                    except ValueError:
                        if name == self._fn_ir.name:
                            # the function refers to its own name while
                            # it is being registered (an aot function
                            # decorated in an enclosing scope is parsed
                            # before the decorator has bound the name):
                            # the name then holds the raw function
                            # object, which the interpreter resolves to
                            # the function value when a call runs
                            return fn
                        raise CompileError(
                            f"captured variable '{name}' is not bound yet in the "
                            f"scope of function {self._fn_ir.name}"
                        ) from None
        return None

    def _resolve_global(self, name: str) -> Any:
        """The raw Python object the global name ``name`` resolves to:
        the value of its closure cell, or of its module global.  The
        object is embedded by ``_gen_name`` as a ``hir.Const`` (value
        context) or a ``hir.ConstRef`` (reference context)."""
        closure = self._resolve_closure(name)
        if closure is not None:
            return closure
        fn = self.fn
        globals = fn.__globals__
        if name in globals:
            return globals[name]
        raise CompileError(
            f"name '{name}' is not defined in the scope of function {self._fn_ir.name}"
        )

    def _gen_expr(
        self,
        node: ast.expr,
        is_ref: bool = False,
        result_loc: hir.Value | None = None,
    ) -> hir.Value | None:
        """Translate one expression, with result-location semantics (RLS).

        Of the four flag combinations only three are meaningful
        (``is_ref=True`` together with ``result_loc`` is an error):

        * ``is_ref=False, result_loc=None`` (the default): produce the
          expression value in a register and return it;
        * ``is_ref=False, result_loc=<pointer>``: write the expression's
          result into the pointer and return ``None`` - the caller
          already has a slot for it (a local variable, the result slot
          of an enclosing statement, ...);
        * ``is_ref=True, result_loc=None``: produce a *reference* to the
          result - a pointer, not a value.  Addressable names yield
          their slot; a global is an immutable value and yields a
          ``hir.ConstRef`` (a const pointer) to it - the callee of a
          call is generated this way, as a callee must be a reference -
          and any other expression is evaluated into a fresh slot whose
          pointer is returned.

        A call in value context therefore allocates a temporary slot,
        emits a :class:`hir.CallInplace` writing into it and loads the
        value back; with a ``result_loc`` the call writes straight into
        it.
        """
        assert not (is_ref and result_loc is not None)
        if is_ref:
            return self._gen_ref(node)
        if result_loc is not None:
            self._gen_result_loc(node, result_loc)
            return None
        return self._gen_value(node)

    def _gen_ref(self, node: ast.expr) -> hir.Value:
        """A reference to the value of ``node`` (see ``_gen_expr``):
        addressable names give their slot, the fields of a runtime
        struct value give their address (a :class:`hir.FieldAddr` chain
        rooted at the storage of the base), globals - immutable values -
        give a :class:`hir.ConstRef` to them, and everything else gives
        a pointer to a freshly allocated slot holding its value."""
        match node:
            case ast.Name():
                return self._gen_name(node.id, True)
            case ast.Attribute():
                base = self._gen_ref(node.value)
                if isinstance(base, hir.ConstRef):
                    if hasattr(base.value, node.attr):
                        return hir.ConstRef(getattr(base.value, node.attr))
                    raise AttributeError(f"Attribute '{node.attr}' not found on {base.value}")
                return self.add(hir.FieldAddr(self._gen_ref(node.value), node.attr))
            case _:
                loc = self.add(hir.Alloca(True))
                self._gen_result_loc(node, loc)
                self.add(hir.CommitSlot(loc))
                return loc

    # -- struct values ---------------------------------------------------------

    def _gen_result_loc(self, node: ast.expr, result_loc: hir.Value) -> None:
        """Evaluate ``node`` writing its result into ``result_loc``
        (result-location semantics); no value register is produced."""
        fn_name = self._fn_ir.name
        match node:
            case ast.Call():
                if len(node.keywords) > 0:
                    raise CompileError(
                        f"calls with keyword arguments inside spy functions are not supported yet "
                        f"(function {fn_name})"
                    )
                self._gen_call(node, result_loc)
            case ast.UnaryOp():
                op = _UNARY_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported unary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                self.add(hir.Unary(op, self._gen_arg(node.operand), result_loc))
            case ast.BinOp():
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported binary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_arg(node.left)
                rhs = self._gen_arg(node.right)
                self.add(hir.Binary(op, lhs, rhs, result_loc))
            case _:
                # every other expression computes a value first; only the
                # call (and, later, the ``if`` expression) can write
                # through a result location without materializing a value
                value = self._gen_value(node)
                self.add(hir.Store(result_loc, value))

    def _gen_arg(self, node: ast.expr):
        match node:
            case ast.Name() | ast.Call() | ast.Attribute():
                return ArgEntry(self._gen_ref(node), True)
            case _:
                return ArgEntry(self._gen_value(node), False)

    def _gen_arglist(self, args: list[ast.expr], keywords: list[ast.keyword]) -> RawArgList[ArgEntry[hir.Value]]:
        positional = tuple(self._gen_arg(a) for a in args)
        kwargs: dict[str, ArgEntry[hir.Value]] = {}
        for kw in keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = self._gen_arg(kw.value)
        return RawArgList(positional, frozendict(kwargs.items()))

    def _gen_call(self, node: ast.Call, result_loc: hir.Value) -> None:
        """One call whose result is written into ``result_loc``: a method
        call ``x.h(...)`` on a runtime struct value, or an ordinary call
        (a spy function, a constructor ``Foo(...)``, an inlined plain
        function or a spy builtin)."""
        if isinstance(node.func, ast.Attribute):
            # a method of the struct ``base``: the method and its self
            # parameter are resolved by the interpreter from the static
            # type of the base; only the base's address is carried here
            base = self._gen_ref(node.func.value)
            self.add(hir.CallMethodInplace(base, node.func.attr, self._gen_arglist(node.args, node.keywords), result_loc))
            return
        # the callee must be addressable (a reference), the arguments are
        # by-value values
        callee = self._gen_ref(node.func)
        self.add(hir.CallInplace(callee, self._gen_arglist(node.args, node.keywords), result_loc))

    def _gen_name(self, name: str, is_ref: bool) -> hir.Value:
        """One reference to the name ``name``.  A variable (a parameter
        or a block-local) is addressable: its slot *is* the reference,
        and a value context reads it back with a :class:`hir.Load`.  A
        global is an immutable *value*: in a value context the name is
        embedded as a :class:`hir.Const` of the resolved object; in a
        reference context it becomes a :class:`hir.ConstRef` - a const
        reference (pointer) to the global.  A function value, whose type
        is a runtime DST, is only legal behind such a reference (a
        function pointer); it is an error to use it as a plain value."""
        slot = self._scope.lookup(name)
        if slot is not None:
            # reading a variable (parameter or local): load its slot
            return slot if is_ref else self.add(hir.Load(slot))
        obj = self._resolve_global(name)
        # a global: its resolved object is the immutable value of the
        # name; a reference to it is a ``ConstRef`` of that object
        return hir.ConstRef(obj) if is_ref else hir.Const(obj)

    def _make_load(self, value: hir.Value):
        if isinstance(value, hir.ConstRef):
            return hir.Const(value.value)
        return self.add(hir.Load(value))

    def _gen_value(self, node: ast.expr) -> hir.Value:
        """Evaluate ``node`` producing its value in a register (the plain
        by-value context)."""
        fn_name = self._fn_ir.name
        match node:
            case ast.Constant():
                if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                    return hir.Const(node.value)
                raise CompileError(f"unsupported constant {node.value!r} in spy function {fn_name}")
            case ast.Name():
                return self._gen_name(node.id, False)
            case ast.BoolOp():
                op = _BOOL_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported boolean operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                if len(node.values) != 2:
                    raise CompileError(
                        f"chained boolean operators are not supported yet in spy function {fn_name}"
                    )
                lhs = self._gen_arg(node.values[0])
                rhs = self._gen_arg(node.values[1])
                return self.add(hir.BoolOp(op, lhs, rhs))
            case ast.Compare():
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise CompileError(
                        f"chained comparisons are not supported yet in spy function {fn_name}"
                    )
                op = _CMP_OPS.get(type(node.ops[0]))
                if op is None:
                    raise CompileError(
                        f"unsupported comparison {type(node.ops[0]).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_arg(node.left)
                rhs = self._gen_arg(node.comparators[0])
                return self.add(hir.Compare(op, lhs, rhs))
            case ast.Call() | ast.BinOp() | ast.UnaryOp():
                loc = self.add(hir.Alloca(True))
                self._gen_result_loc(node, loc)
                self.add(hir.CommitSlot(loc))
                return self._make_load(loc)
            case ast.Attribute():
                return self._make_load(self._gen_ref(node))
            case _:
                raise CompileError(
                    f"unsupported expression {type(node).__name__} in spy function {fn_name}"
                )


def parse_function(fn: Callable, self_type: Type | None = None) -> FunctionIR:
    """Parse ``fn`` (a plain Python function) into a :class:`FunctionIR`.

    ``mode`` is how the function will be compiled and typed when it is
    called (see ``FunctionIR.mode``): ``'jit'`` (the marshaled argument
    types solve each specialization) or ``'aot'`` (the concrete
    annotations fix its single signature).  A plain function that is
    only ever inlined is parsed in ``'jit'`` mode.
    """
    try:
        source = inspect.getsource(fn)
    except OSError as e:
        raise CompileError(
            f"cannot obtain the source of function {fn.__name__}; "
            "spy functions must be defined in a source file"
        ) from e
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise CompileError(
            f"cannot parse function {fn.__name__}: expected a single function definition"
        )
    node = tree.body[0]
    if node.name != fn.__name__:
        raise CompileError(f"function name mismatch: expected {node.name}, got {fn.__name__}")

    if node.args.vararg is not None or node.args.kwarg is not None:
        raise CompileError(f"*args/**kwargs are not supported in spy function {node.name}")
    if len(node.args.posonlyargs) > 0:
        raise CompileError(f"positional-only arguments are not supported in spy function {node.name}")
    if len(node.args.kwonlyargs) > 0:
        raise CompileError(f"keyword-only arguments are not supported in spy function {node.name}")

    # Read the signature metadata off the function object instead of
    # re-evaluating the source: Python already evaluated the annotations
    # (PEP 695 annotations may evaluate lazily on access) and the default
    # values when it created the function.
    # ``fn.__annotations__`` holds the evaluated annotations; the return
    # annotation is normalized here: ``None`` (no ``->`` written) stays
    # ``None``, and an explicit ``-> None`` becomes the spy ``VoidType``
    # (so that the two can be told apart - the first one lets the return
    # type be inferred from the body, the second declares a void
    # function).
    try:
        annotations = fn.__annotations__
        defaults = fn.__defaults__ if fn.__defaults__ is not None else ()
    except Exception as e:
        raise CompileError(
            f"cannot read the annotations of function {node.name}: {e}"
        ) from e
    if 'return' not in annotations:
        ret_annotation: Any = None
    elif annotations['return'] is None:
        ret_annotation = VoidType()
    else:
        ret_annotation = annotations['return']

    # ``fn.__type_params__`` exposes the declared PEP 695 type parameters
    # (Python 3.13+); the AST ``[T]`` syntax may parse on 3.12, but the
    # annotation values of a generic function are only accessible there
    # through ``__type_params__``.  Each PEP 695 type parameter object is
    # converted into a spy-domain ``sval.TypeVar`` of its own, which the
    # annotations that name the parameter refer to by identity (see
    # ``sval.as_value``).
    declared_type_params = getattr(fn, '__type_params__', ())
    if len(node.type_params) > 0 and len(declared_type_params) == 0:
        raise CompileError(
            f"generic spy functions require Python 3.13 or newer (function {node.name})"
        )
    generic_args: list[SpyTypeVar] = []
    type_vars: dict[TypeVar, Value] = {}
    for type_param in declared_type_params:
        if not isinstance(type_param, TypeVar):
            raise CompileError(
                f"unsupported type parameter {type_param!r} in function {node.name}"
            )
        spy = SpyTypeVar(type_param.__name__)
        generic_args.append(spy)
        type_vars[type_param] = spy

    def convert(annotation: Any, what: str) -> AnyValue | None:
        """The spy-domain value of one evaluated annotation or default
        (see ``sval.as_value``); ``None`` (no annotation written, no
        default) stays ``None``."""
        if annotation is None:
            return None
        try:
            return as_value(annotation, type_vars)
        except Exception as e:
            raise CompileError(
                f"cannot use {annotation!r} as {what} of function {node.name}: {e}"
            ) from e

    def annotation_of(annotation: Any) -> Type | None:
        # an annotation is a spy type or a type parameter in practice; a
        # value that is neither is kept as-is and rejected by the aot
        # discipline (``FunctionIR.aot_param_type``) when the function
        # is used
        return cast(Type | None, convert(annotation, 'the annotation'))

    def default_of(value: Any) -> AnyValue | None:
        # a default value of ``None`` is the unit value of the void type
        # (see ``sval.as_value``): ``default_value`` being ``None`` means
        # the parameter has no default
        if value is None:
            return Void()
        return convert(value, 'a default value')

    # the signature: the formal parameters, by declaration position
    all_args = list(node.args.args)
    offset = len(all_args) - len(defaults)
    positional = IndexedMap[str, SignatureFormalArg]()
    for i, arg in enumerate(all_args):
        has_default = i >= offset
        default_value = default_of(defaults[i - offset]) if has_default else None
        arg_type = annotation_of(annotations.get(arg.arg))
        if i == 0 and self_type is not None:
            arg_type = self_type
        positional.add(
            arg.arg, SignatureFormalArg(arg_type, False, False, default_value)
        )

    # ``*args``/``**kwargs`` are rejected above (a spy function definition
    # may not declare them yet), so the ``varargs``/``kwargs`` slots of the
    # signature are always None; the signature model and ``bind_arg_pos``
    # already support them for the calls the parser will allow later.
    signature = Signature(
        tuple(generic_args), positional, None, None, annotation_of(ret_annotation), None,
    )

    ir = FunctionIR(node.name, signature, ())

    # At HIR level, parameters are passed by ref (pointer)
    scope = _Scope(None)
    for i, name in enumerate(positional.keys):
        scope.bindings[name] = hir.Arg(i)

    builder = _Builder(fn, ir, scope)
    for stmt in node.body:
        builder._gen_stmt(stmt)
    builder.add(hir.Store(hir.ResultLoc(), hir.Const(sval.Void())))
    ir.body = tuple(builder.insts)
    return ir

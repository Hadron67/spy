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

Like ``llvm`` the body is one *linear* list of instructions;
expression evaluation appends temporary instructions to the list and
returns the instruction object whose register holds the value - or,
with a result location (RLS, see ``_Builder._gen_expr``), writes the
value into a caller-provided slot and returns nothing.  The result
location of a function itself (``hir.ResultLoc``) is the target of its
``return`` statements: ``return expr`` evaluates ``expr`` with the
function's result location, so a call in return position writes its
result straight into the location the function returns through.

``astgen`` performs *almost* all name resolution.  At HIR level a
parameter is passed by reference (its address), so the translated body
binds the name of a parameter directly to its ``hir.Arg(i)`` leaf and a
read of it becomes a ``Load`` of that address; the interpreter
materializes the parameter's storage (a MIR alloca holding
``mir.Param``) when the first store runs.
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
its callees, or even itself (a registered function defined in an
enclosing scope refers to its own name before the decorator has bound
it, see ``_resolve_closure``).
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, TypeVar, cast

from . import hir
from .errors import CompileError
from .fn import ArgEntry, FunctionIR, RawArgList, Signature, SignatureFormalArg
from .sval import (
    AnyValue,
    StructDecl,
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

_BIN_OPS: dict[type[ast.AST], hir.BinaryOp] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
}

_BOOL_OPS: dict[type[ast.AST], hir.BoolOpType] = {ast.And: 'and', ast.Or: 'or'}

_UNARY_OPS: dict[type[ast.AST], hir.UnaryOp] = {ast.USub: '-', ast.Not: 'not'}

_CMP_OPS: dict[type[ast.AST], hir.CompareOp] = {
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
}


def _is_struct_class(obj: Any) -> bool:
    """Whether the raw global object ``obj`` is a ``@struct()`` class handle
    (see :class:`sval.StructDecl`): the parser recognizes a construction by
    its callee at parse time (see ``_Builder._struct_operand``)."""
    return isinstance(obj, StructDecl)


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

    def __init__(self, fn: Any, fn_ir: FunctionIR, scope: _Scope, generic_names: dict[str, Value]) -> None:
        self.fn = fn
        self._fn_ir = fn_ir
        self._scope = scope
        # the type parameters of the function (and of the struct a method
        # belongs to) by name: a name that denotes one is a compile-time
        # value, see ``_gen_expr``
        self._generic_names = generic_names
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
                else:
                    self.add(hir.StoreVoidRetloc())
                self.add(hir.Ret())
            case ast.Pass():
                pass
            case ast.Expr():
                self._gen_expr(node.value)
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
                cond = self._gen_expr(node.test)
                self.add(hir.If(self.add(hir.AsBool(cond))))
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
        sub = _Builder(self.fn, self._fn_ir, _Scope(self._scope), self._generic_names)
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
        if isinstance(target, ast.Tuple):
            new_slots: list[hir.Value] = []
            ptrs = self._gen_target_tuple(target, new_slots)
            self.add(hir.Store(ptrs, self._as_value(self._gen_expr(value))))
            for slot in new_slots:
                self.add(hir.CommitSlot(slot))
            return
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
        lhs = self._gen_expr(target, False)
        if not lhs.is_ref:
            raise CompileError(f"target of augmented assignment must be a variable, got {target}")
        self._gen_result_loc(value, lhs.value)
        if emit_commit:
            self.add(hir.CommitSlot(lhs.value))

    def _gen_target_tuple(self, target: ast.Tuple, new_slots: list[hir.Value]) -> hir.Value:
        """The tuple of addresses a destructuring target denotes: a plain
        target contributes the address of its slot (or field), a nested
        tuple target contributes its own tuple of addresses."""
        elems: list[ArgEntry[hir.Value]] = []
        for elt in target.elts:
            if isinstance(elt, ast.Tuple):
                elems.append(ArgEntry(self._gen_target_tuple(elt, new_slots), False))
                continue
            if isinstance(elt, ast.Name) and elt.id not in self._scope.bindings:
                slot = self.add(hir.Alloca())
                self._scope.bindings[elt.id] = slot
                new_slots.append(slot)
            ref = self._gen_expr(elt, False)
            if not ref.is_ref:
                raise CompileError(
                    f"target of a destructuring assignment must be addressable, got {elt}"
                )
            elems.append(ref)
        return self.add(hir.Tuple(tuple(elems)))

    def _gen_augassign(self, node: ast.AugAssign) -> None:
        """One ``name += expr`` statement: read the value, add ``expr``
        and store the result back.  The target is a variable slot or the
        address of a field of a runtime struct value (``self.h += e``);
        ``+=`` never declares: it requires the name to be declared."""
        fn_name = self._fn_ir.name
        if not isinstance(node.op, ast.Add):
            raise CompileError(f"only '+=' is supported yet in spy function {fn_name}")

        lhs = self._gen_expr(node.target, False)
        if not lhs.is_ref:
            raise CompileError(f"target of augmented assignment must be a variable, got {node.target}")
        rhs = self._gen_expr(node.value)
        self.add(hir.BinaryAssign(_BIN_OPS[type(node.op)], lhs.value, rhs))

    # -- expressions ----------------------------------------------------------

    def _resolve_closure(self, name: str) -> Any | None:
        """The raw object of the name ``name`` captured from an
        enclosing Python scope (a spy function may be defined inside a
        factory, e.g. ``def make(k): @func() def f(x): return x *
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
                            # it is being registered (a registered
                            # function decorated in an enclosing scope
                            # is parsed before the decorator has bound
                            # the name): the name then holds the raw
                            # function object, which the interpreter
                            # resolves to the function value when a call
                            # runs
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

    def _as_ref(self, node: ArgEntry[hir.Value]):
        if node.is_ref:
            return node.value
        loc = self.add(hir.Alloca(True))
        self.add(hir.Store(loc, node.value))
        self.add(hir.CommitSlot(loc))
        return loc

    def _as_value(self, node: ArgEntry[hir.Value]) -> hir.Value:
        if node.is_ref:
            return self._make_load(node.value)
        return node.value

    def _gen_expr(self, node: ast.expr, allow_retloc: bool = True) -> ArgEntry[hir.Value]:
        """A reference to the value of ``node``: addressable names give
        their slot, the fields of a runtime struct value give their
        address (a :class:`hir.FieldAddr` chain rooted at the storage of
        the base), globals - immutable values - give a
        :class:`hir.ConstRef` to them, and everything else gives a
        pointer to a freshly allocated slot holding its value."""
        match node:
            case ast.Name():
                generic = self._generic_names.get(node.id)
                if generic is not None:
                    # a type parameter of the function (or of the struct a
                    # method belongs to) used as a value: the interpreter
                    # resolves it to the type the call solved it to, from
                    # the frame it runs in (see ``interp.operand``)
                    return ArgEntry(hir.Const(generic), False)
                return ArgEntry(self._gen_name(node.id), True)
            case ast.Attribute():
                base = self._as_ref(self._gen_expr(node.value))
                if isinstance(base, hir.ConstRef):
                    if hasattr(base.value, node.attr):
                        return ArgEntry(hir.ConstRef(getattr(base.value, node.attr)), True)
                    raise AttributeError(f"Attribute '{node.attr}' not found on {base.value}")
                return ArgEntry(self.add(hir.FieldAddr(base, node.attr)), True)
            case ast.Constant():
                if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                    return ArgEntry(hir.Const(node.value), False)
                raise CompileError(f"unsupported constant {node.value!r}")
            case ast.BoolOp():
                op = _BOOL_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported boolean operator {type(node.op).__name__}"
                    )
                if len(node.values) != 2:
                    raise CompileError(
                        "chained boolean operators are not supported yet"
                    )
                lhs = self._gen_expr(node.values[0])
                rhs = self._gen_expr(node.values[1])
                return ArgEntry(self.add(hir.BoolOp(op, lhs, rhs)), False)
            case ast.Compare():
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise CompileError(
                        "chained comparisons are not supported yet"
                    )
                op = _CMP_OPS.get(type(node.ops[0]))
                if op is None:
                    raise CompileError(
                        f"unsupported comparison {type(node.ops[0]).__name__}"
                    )
                lhs = self._gen_expr(node.left)
                rhs = self._gen_expr(node.comparators[0])
                return ArgEntry(self.add(hir.Compare(op, lhs, rhs)), False)
            case ast.Tuple():
                values = tuple(self._gen_expr(elt) for elt in node.elts)
                return ArgEntry(self.add(hir.Tuple(values)), False)
            case ast.Subscript():
                base = self._gen_expr(node.value)
                if not base.is_ref:
                    raise CompileError(f"subscript of non-reference {base}")
                index = self._gen_expr(node.slice)
                return ArgEntry(self.add(hir.Subscript(base.value, index)), False)
            case _:
                if not allow_retloc:
                    raise CompileError(f"unexpected expression {node}")
                loc = self.add(hir.Alloca(True))
                self._gen_result_loc(node, loc, False)
                self.add(hir.CommitSlot(loc))
                return ArgEntry(loc, True)

    # -- struct values ---------------------------------------------------------

    def _gen_result_loc(self, node: ast.expr, result_loc: hir.Value, allow_fall_back: bool = True) -> None:
        """Evaluate ``node`` writing its result into ``result_loc``
        (result-location semantics); no value register is produced."""
        fn_name = self._fn_ir.name
        match node:
            case ast.Call():
                self._gen_call(node, result_loc)
            case ast.UnaryOp():
                op = _UNARY_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported unary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                self.add(hir.Unary(op, self._gen_expr(node.operand), result_loc))
            case ast.BinOp():
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported binary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_expr(node.left)
                rhs = self._gen_expr(node.right)
                self.add(hir.Binary(op, lhs, rhs, result_loc))
            case _:
                # every other expression computes its value first and
                # stores it into the result location; only a call, a
                # unary and a binary operation (the cases above) write
                # through the location without materializing a value
                if not allow_fall_back:
                    raise CompileError(f"unsupported expression {node}")
                value = self._as_value(self._gen_expr(node))
                self.add(hir.Store(result_loc, value))

    def _gen_arglist(self, args: list[ast.expr], keywords: list[ast.keyword]) -> RawArgList[ArgEntry[hir.Value]]:
        positional = tuple(self._gen_expr(a) for a in args)
        kwargs: dict[str, ArgEntry[hir.Value]] = {}
        for kw in keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = self._gen_expr(kw.value)
        return RawArgList(positional, frozendict(kwargs.items()))

    def _struct_operand(self, node: ast.expr) -> hir.Value | None:
        """When the callee expression ``node`` denotes a struct, the operand
        the construction names it by: a ``hir.ConstRef`` of the class for a
        plain name, and the ``hir.Subscript`` that specializes it for
        ``Foo[i32]``.  ``None`` when the callee is not a struct, in which
        case the call compiles as an ordinary one.

        The name of a class is a global (a *struct* is only ever named by
        its decoration binding): a variable of the same name shadows it,
        and so does a type parameter."""
        match node:
            case ast.Name():
                if self._scope.lookup(node.id) is not None or node.id in self._generic_names:
                    return None
                obj = self._resolve_global(node.id)
                if not _is_struct_class(obj):
                    return None
                return hir.ConstRef(obj)
            case ast.Subscript():
                # ``Foo[i32]``: the specialization the base names, subscripted
                # by the generic arguments - the interpreter resolves it while
                # running the ``hir.Subscript``
                base = self._struct_operand(node.value)
                if base is None:
                    return None
                return self.add(hir.Subscript(base, self._gen_expr(node.slice)))
            case _:
                return None

    def _gen_struct_ctor(self, struct: hir.Value, args: list[ast.expr], keywords: list[ast.keyword], result_loc: hir.Value) -> None:
        """One construction ``Foo(a1, a2, k=v)``: an ``hir.InitStruct`` opens
        the struct in the result location, every argument is generated with
        result-location semantics straight into the address of the field it
        initializes - a nested construction fills the field in place, with no
        copy - and ``hir.FinishStruct`` closes the construction, filling the
        fields that were left out with their defaults."""
        inst = self.add(hir.InitStruct(struct, result_loc))
        indices: set[int] = set()
        for i, arg in enumerate(args):
            self._gen_result_loc(arg, self.add(hir.FieldIndexAddr(inst, i)))
            indices.add(i)
        names: set[str] = set()
        for kw in keywords:
            if kw.arg is None:
                raise CompileError(
                    f"**kwargs are not supported in spy function {self._fn_ir.name}"
                )
            self._gen_result_loc(kw.value, self.add(hir.FieldAddr(inst, kw.arg)))
            names.add(kw.arg)
        self.add(hir.FinishStruct(inst, frozenset(indices), frozenset(names)))

    def _gen_call(self, node: ast.Call, result_loc: hir.Value) -> None:
        """One call whose result is written into ``result_loc``: a
        construction ``Foo(...)``, a method call ``x.h(...)`` on a runtime
        struct value, or an ordinary call (a spy function, an inlined plain
        function or a spy builtin)."""
        if isinstance(node.func, ast.Attribute):
            # a method of the struct ``base``: the method and its self
            # parameter are resolved by the interpreter from the static
            # type of the base; only the base's address is carried here
            base = self._as_ref(self._gen_expr(node.func.value))
            self.add(hir.CallMethodInplace(base, node.func.attr, self._gen_arglist(node.args, node.keywords), result_loc))
            return
        struct = self._struct_operand(node.func)
        if struct is not None:
            # a construction: it writes the fields of the struct in place
            # instead of producing a value the call site would copy
            self._gen_struct_ctor(struct, node.args, node.keywords, result_loc)
            return
        # the callee must be addressable (a reference), the arguments are
        # by-value values
        callee = self._as_ref(self._gen_expr(node.func))
        self.add(hir.CallInplace(callee, self._gen_arglist(node.args, node.keywords), result_loc))

    def _gen_name(self, name: str) -> hir.Value:
        """Always returns a reference to the name ``name``."""
        slot = self._scope.lookup(name)
        if slot is not None:
            # reading a variable (parameter or local): load its slot
            return slot
        obj = self._resolve_global(name)
        # a global: its resolved object is the immutable value of the
        # name; a reference to it is a ``ConstRef`` of that object
        return hir.ConstRef(obj)

    def _make_load(self, value: hir.Value):
        if isinstance(value, hir.ConstRef):
            return hir.Const(value.value)
        return self.add(hir.Load(value))

def parse_function(
    fn: Callable,
    self_type: Type | None = None,
    self_by_value: bool = False,
    context_type_vars: dict[TypeVar, Value] | None = None,
) -> FunctionIR:
    """Parse ``fn`` (a plain Python function) into a :class:`FunctionIR`.

    ``self_type`` is the struct a *method* belongs to: the first parameter
    is then typed as that struct itself and passed by reference (its
    address), unless ``self_by_value`` asks for the object's value.

    ``context_type_vars`` are the type parameters of an enclosing context a
    method may name in its annotations and its body - the generic type
    parameters of the struct the method belongs to, which Python only makes
    visible inside the method's annotation scope.  They are keyed by the
    Python type parameter object the annotations evaluate to.
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

    # ``fn.__type_params__`` exposes the declared PEP 695 type parameters
    # (Python 3.13+); the AST ``[T]`` syntax may parse on 3.12, but the
    # annotation values of a generic function are only accessible there
    # through ``__type_params__``.  Each PEP 695 type parameter object is
    # converted into a spy-domain ``sval.TypeVar`` of its own, which the
    # annotations that name the parameter refer to by identity (see
    # ``sval.as_value``).  The type parameters are collected *before* the
    # annotations are read: reading them evaluates the annotations, which
    # may name the type parameters in a subscripted struct template.
    declared_type_params = getattr(fn, '__type_params__', ())
    if len(node.type_params) > 0 and len(declared_type_params) == 0:
        raise CompileError(
            f"generic spy functions require Python 3.13 or newer (function {node.name})"
        )
    generic_args: list[SpyTypeVar] = []
    type_vars: dict[TypeVar, Value] = dict(context_type_vars) if context_type_vars is not None else {}
    for type_param in declared_type_params:
        if not isinstance(type_param, TypeVar):
            raise CompileError(
                f"unsupported type parameter {type_param!r} in function {node.name}"
            )
        spy = SpyTypeVar(type_param.__name__)
        generic_args.append(spy)
        type_vars[type_param] = spy

    # Read the signature metadata off the function object instead of
    # re-evaluating the source: Python already evaluated the annotations
    # (PEP 695 annotations may evaluate lazily on access) and the default
    # values when it created the function.
    # ``fn.__annotations__`` holds the evaluated annotations; the return
    # annotation is normalized here: ``None`` (no ``->`` written) stays
    # ``None``, and an explicit ``-> None`` becomes the spy ``VoidType``
    # (so that the two can be told apart - the first one lets the return
    # type be inferred from the body, the second declares a void
    # function).  An annotation that subscripts a struct template evaluates
    # to a ``sval.StructTypeApplication``; ``convert`` resolves it against
    # the type parameters (see ``sval.as_value``).
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
        # an annotation is a spy type or a type parameter in practice;
        # anything else is kept as-is and rejected when the call is
        # specialized (``fn.Signature.specialize``)
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
        by_ref = False
        if i == 0 and self_type is not None:
            # the ``self`` of a method: the object is passed by reference
            # (its address), which is what makes a method able to write
            # through ``self``; ``self_by_value`` passes its value instead
            arg_type = self_type
            by_ref = not self_by_value
        positional.add(
            arg.arg, SignatureFormalArg(arg_type, False, by_ref, default_value)
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

    # a name that denotes a type parameter (the function's own, or one of the
    # struct a method belongs to) refers to the compile-time value the call
    # solved it to; the function's own parameters are added last, so they
    # shadow a struct's parameter of the same name, like Python scoping
    generic_names: dict[str, Value] = {tp.__name__: v for tp, v in type_vars.items()}

    builder = _Builder(fn, ir, scope, generic_names)
    for stmt in node.body:
        builder._gen_stmt(stmt)
    builder.add(hir.StoreVoidRetloc())
    ir.body = tuple(builder.insts)
    return ir

"""Lowering of Python source to the untyped HIR (``hir``), plus the shared
signature analysis used both by the call boundary (``dsl``) and by
compile-time calls inside a function body (``interp``).

``parse_function`` turns the source of a Python function into a
:class:`FunctionIR`: the declared type parameters (PEP 695 ``[T]``
syntax), the formal parameters and the translated body.  The parameter
annotations, default values and return annotation are read off the
function object itself (``fn.__annotations__``/``fn.__defaults__``/...),
where Python has already evaluated them at definition time, so the
source expressions are never re-evaluated.

Like ``symlat.jit.llvm`` the body is one *linear* list of instructions;
expression evaluation appends temporary instructions to the list and
returns the instruction object whose register holds the value - or,
with a result location (RLS, see ``_Builder._gen_expr``), writes the
value into a caller-provided slot and returns nothing.

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
such compile-time objects (``spy.type``, ``spy.u64``, ...) are
evaluated here as well.  Whether a function object denotes a registered
spy function - and which function value it stands for - is decided by
the interpreter when a call runs: a function body may be parsed before
its callees, or even itself (an aot function parses its own body while
it is being registered), are registered.

``solve_call_types`` computes the concrete spy types of all formal
parameters of one call:

* in *jit* mode every provided argument contributes the type it marshals
  to, and parameters annotated with the same type parameter ``T`` must
  all marshal to the same type (``T`` is unified);
* in *aot* mode the parameter types are simply the (concrete,
  non-generic) annotations.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, TypeVar

from . import hir
from .errors import CompileError, TypeMismatchError
from .fn import FunctionIR, ParamDef
from .type import Type, VoidType, type_str, value_type

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

    def __init__(self, parent: '_Scope | None') -> None:
        self.parent = parent
        self.bindings: dict[str, hir.Inst] = {}

    def lookup(self, name: str) -> hir.Inst | None:
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

    def __init__(self, fn_ir: FunctionIR, scope: _Scope) -> None:
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
                value = None if node.value is None else self._gen_value(node.value)
                self.add(hir.Ret(value))
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
                cond = self._gen_value(node.test)
                then_body = self._gen_body(node.body)
                else_body = self._gen_body(node.orelse)
                self.add(hir.If(cond, tuple(then_body), tuple(else_body)))
            case _:
                raise CompileError(
                    f"unsupported statement {type(node).__name__} in spy function {fn_name}"
                )

    def _gen_body(self, stmts: list[ast.stmt]) -> list[hir.Inst]:
        """Translate a statement block into its own linear instruction
        list.  A block is a lexical scope of its own - a child of the
        enclosing block - so declarations inside it shadow outer
        bindings and are not visible after the block."""
        sub = _Builder(self._fn_ir, _Scope(self._scope))
        for stmt in stmts:
            sub._gen_stmt(stmt)
        return sub.insts

    # -- variables ------------------------------------------------------------

    def _gen_assign(self, target: ast.expr, value: ast.expr) -> None:
        """One ``target = expr`` statement.  The first ``=`` on a name
        declares a block-local variable (a fresh slot, shadowing any
        outer binding); later assignments in the block only store into
        its slot.  A call on the right hand side writes its result
        straight into the target slot (result-location semantics): a
        constructor ``x = Bar(...)`` fills the fields of the slot in
        place, and a scalar call result is only recorded in it."""
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
            if isinstance(value, ast.Call):
                self._gen_call(value, slot)
                return
            self.add(hir.Store(slot, self._gen_value(value)))
        elif isinstance(target, ast.Attribute):
            # assignment to a field of a runtime struct value (``x.h =
            # e``): store through the address of the field (globals are
            # immutable: their fields cannot be assigned)
            if self._attr_runtime(target):
                self.add(hir.Store(self._gen_ref(target), self._gen_value(value)))
                return
            raise CompileError(
                f"cannot assign to the attribute of a compile-time value "
                f"(function {self._fn_ir.name})"
            )
        else:
            raise CompileError(
                f"unsupported assignment target in spy function {self._fn_ir.name}"
            )

    def _gen_augassign(self, node: ast.AugAssign) -> None:
        """One ``name += expr`` statement: read the value, add ``expr``
        and store the result back.  The target is a variable slot or the
        address of a field of a runtime struct value (``self.h += e``);
        ``+=`` never declares: it requires the name to be declared."""
        fn_name = self._fn_ir.name
        if not isinstance(node.op, ast.Add):
            raise CompileError(f"only '+=' is supported yet in spy function {fn_name}")

        lhs = self._gen_ref(node.target)
        rhs = self._gen_value(node.value)
        self.add(hir.Store(lhs, self.add(hir.Binary('+', self.add(hir.Load(lhs)), rhs))))

    # -- expressions ----------------------------------------------------------

    def _resolve_closure(self, name: str) -> Any | None:
        """The raw object of the name ``name`` captured from an
        enclosing Python scope (a spy function may be defined inside a
        factory, e.g. ``def make(k): @cache.jit() def f(x): return x *
        k``), or None when the name is not a free variable.  A captured
        variable behaves like a global: the value of its closure cell at
        parse time is embedded as a compile-time constant."""
        fn = self._fn_ir.fn
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
        fn = self._fn_ir.fn
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
                if self._attr_runtime(node):
                    # the fields of a runtime struct value: the address
                    # of the innermost field is the FieldAddr chain
                    # rooted at the storage of the base value
                    return self.add(hir.FieldAddr(self._gen_ref(node.value), node.attr))
                # attribute chains on compile-time objects fold to a
                # ``hir.Const`` leaf, which is its own reference (the
                # folded value is immutable)
                return self._gen_value(node)
            case _:
                loc = self.add(hir.Alloca())
                self._gen_result_loc(node, loc)
                return loc

    # -- struct values ---------------------------------------------------------

    @staticmethod
    def _attr_root(node: ast.Attribute) -> ast.Name | None:
        """The Name an attribute chain is rooted at: ``x.f.g`` returns
        ``x``, ``spy.type`` returns ``spy``."""
        while isinstance(node, ast.Attribute):
            node = node.value  # type: ignore[assignment]
        if isinstance(node, ast.Name):
            return node
        return None

    def _attr_runtime(self, node: ast.expr) -> bool:
        """Whether ``node`` is an attribute chain rooted at a *variable*
        (a parameter or a local): only then is the attribute a field of a
        runtime struct value (chains rooted at globals - ``spy.u64``,
        ``spy.type``, ... - fold at compile time instead)."""
        root = self._attr_root(node) if isinstance(node, ast.Attribute) else None
        return root is not None and self._scope.lookup(root.id) is not None

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
            case _:
                # every other expression computes a value first; only the
                # call (and, later, the ``if`` expression) can write
                # through a result location without materializing a value
                value = self._gen_value(node)
                self.add(hir.Store(result_loc, value))

    def _gen_call(self, node: ast.Call, result_loc: hir.Value) -> None:
        """One call whose result is written into ``result_loc``: a method
        call ``x.h(...)`` on a runtime struct value, or an ordinary call
        (a spy function, a constructor ``Foo(...)``, an inlined plain
        function or a spy builtin)."""
        if isinstance(node.func, ast.Attribute) and self._attr_runtime(node.func):
            # a method of the struct ``base``: the method and its self
            # parameter are resolved by the interpreter from the static
            # type of the base; only the base's address is carried here
            base = self._gen_ref(node.func.value)
            args = tuple(self._gen_value(a) for a in node.args)
            self.add(hir.CallMethodInplace(base, node.func.attr, args, result_loc))
            return
        # the callee must be addressable (a reference), the arguments are
        # by-value values
        callee = self._gen_ref(node.func)
        args = tuple(self._gen_value(a) for a in node.args)
        self.add(hir.CallInplace(callee, args, result_loc))

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
            case ast.Attribute():
                if self._attr_runtime(node):
                    # a field of a runtime struct value: read it through
                    # the address of the (innermost) field - the chain is
                    # addressable, so no intermediate struct value is
                    # materialized
                    return self.add(hir.Load(self._gen_ref(node)))
                base = self._gen_value(node.value)
                if isinstance(base, hir.Const) and not isinstance(
                    base.value, (int, float, str, bool, type(None))
                ):
                    # attribute of a compile-time object (e.g. spy.type)
                    try:
                        obj = getattr(base.value, node.attr)
                    except AttributeError as e:
                        raise CompileError(
                            f"compile-time value {base.value!r} has no attribute {node.attr} "
                            f"in spy function {fn_name}"
                        ) from e
                    return hir.Const(obj)
                raise CompileError(
                    f"attribute access on values is not supported yet in spy function {fn_name}"
                )
            case ast.BinOp():
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported binary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_value(node.left)
                rhs = self._gen_value(node.right)
                return self.add(hir.Binary(op, lhs, rhs))
            case ast.BoolOp():
                op = _BOOL_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(f"unsupported boolean operator in spy function {fn_name}")
                values = [self._gen_value(v) for v in node.values]
                result: hir.Value = values[0]
                for v in values[1:]:
                    result = self.add(hir.BoolOp(op, result, v))
                return result
            case ast.UnaryOp():
                if isinstance(node.op, ast.UAdd):
                    return self._gen_value(node.operand)
                op = _UNARY_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported unary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                return self.add(hir.Unary(op, self._gen_value(node.operand)))
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
                lhs = self._gen_value(node.left)
                rhs = self._gen_value(node.comparators[0])
                return self.add(hir.Compare(op, lhs, rhs))
            case ast.Call():
                loc = self.add(hir.Alloca())
                self._gen_result_loc(node, loc)
                return self.add(hir.Load(loc))
            case _:
                raise CompileError(
                    f"unsupported expression {type(node).__name__} in spy function {fn_name}"
                )


def parse_function(fn: Callable) -> FunctionIR:
    """Parse ``fn`` (a plain Python function) into a :class:`FunctionIR`."""
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
    # through ``__type_params__``.
    declared_type_params = getattr(fn, '__type_params__', ())
    if len(node.type_params) > 0 and len(declared_type_params) == 0:
        raise CompileError(
            f"generic spy functions require Python 3.13 or newer (function {node.name})"
        )
    type_params: list[TypeVar] = []
    for type_param in declared_type_params:
        if isinstance(type_param, TypeVar):
            type_params.append(type_param)
        else:
            raise CompileError(
                f"unsupported type parameter {type_param!r} in function {node.name}"
            )

    all_args = list(node.args.args)
    offset = len(all_args) - len(defaults)
    params: list[ParamDef] = []
    for i, arg in enumerate(all_args):
        has_default = i >= offset
        default_value = defaults[i - offset] if has_default else None
        params.append(ParamDef(arg.arg, annotations.get(arg.arg), has_default, default_value))

    ir = FunctionIR(fn, node.name, tuple(type_params), tuple(params), ret_annotation, ())

    # prologue: every parameter is addressable, so allocate one slot per
    # parameter and store its by-value argument into it.  The function
    # body is the outermost block: its scope is pre-populated with the
    # parameter slots, so an assignment to a parameter name at the top
    # level stores into the parameter slot.  The interpreter types an
    # Alloca when its first store runs.
    scope = _Scope(None)
    prologue: list[hir.Inst] = []
    for i, param in enumerate(params):
        alloca = hir.Alloca()
        prologue.append(alloca)
        prologue.append(hir.Store(alloca, hir.Arg(i)))
        scope.bindings[param.name] = alloca

    builder = _Builder(ir, scope)
    for stmt in node.body:
        builder._gen_stmt(stmt)
    ir.body = tuple(prologue + builder.insts)
    return ir


# ---------------------------------------------------------------------------
# signature analysis
# ---------------------------------------------------------------------------


def _type_param_of(fn_ir: FunctionIR, param: ParamDef) -> str | None:
    """If the parameter annotation names a type parameter, return its name."""
    ann = param.annotation
    if ann is None:
        return None
    for type_param in fn_ir.type_params:
        if ann is type_param:
            return type_param.__name__
    return None


def annotation_type(fn_ir: FunctionIR, param: ParamDef) -> Type:
    """The (concrete) spy type of the annotation of an aot parameter."""
    if param.annotation is None:
        raise TypeMismatchError(
            f"parameter '{param.name}' of function {fn_ir.name} requires a type annotation"
        )
    tp = _type_param_of(fn_ir, param)
    if tp is not None:
        raise TypeMismatchError(
            f"type parameter {tp} is not allowed in aot function {fn_ir.name}"
        )
    if not isinstance(param.annotation, Type):
        raise TypeMismatchError(
            f"annotation of parameter '{param.name}' of function {fn_ir.name} "
            f"is not a spy type: {param.annotation!r}"
        )
    return param.annotation


def return_annotation_type(fn_ir: FunctionIR) -> Type:
    """The spy type of the return annotation of an aot function."""
    ret_ann = fn_ir.ret_annotation
    if ret_ann is None:
        raise TypeMismatchError(f"function {fn_ir.name} requires a return type annotation")
    if not isinstance(ret_ann, Type):
        raise TypeMismatchError(
            f"return annotation of function {fn_ir.name} is not a spy type: {ret_ann!r}"
        )
    return ret_ann


def _param_missing_error(fn_ir: FunctionIR, param: ParamDef) -> TypeMismatchError:
    if param.has_default:
        raise TypeMismatchError(
            f"cannot determine the type of the default value of parameter "
            f"'{param.name}' of function {fn_ir.name}"
        )
    raise TypeMismatchError(f"missing argument '{param.name}' of function {fn_ir.name}")


def solve_call_types(
    fn_ir: FunctionIR, mode: str, provided: tuple[Type | None, ...]
) -> tuple[Type, ...]:
    """Compute the spy type of every formal parameter of a call.

    ``provided`` holds the marshaled type of every provided argument, in
    parameter order, and ``None`` for parameters whose default value
    applies.
    """
    assert len(provided) == len(fn_ir.params), 'argument count mismatch'
    if mode == 'aot':
        return tuple(annotation_type(fn_ir, p) for p in fn_ir.params)

    if mode != 'jit':
        raise ValueError(f"unknown call mode {mode!r}")

    # unify the type parameters over the provided arguments
    bound: dict[str, Type] = {}
    for param, cand in zip(fn_ir.params, provided):
        if cand is None:
            continue
        tp = _type_param_of(fn_ir, param)
        if tp is None:
            continue
        if tp in bound:
            if bound[tp] != cand:
                raise TypeMismatchError(
                    f"type parameter {tp} of function {fn_ir.name} got conflicting types "
                    f"{type_str(bound[tp])} and {type_str(cand)}"
                )
        else:
            bound[tp] = cand

    ret: list[Type] = []
    for param, cand in zip(fn_ir.params, provided):
        tp = _type_param_of(fn_ir, param)
        if tp is not None:
            if cand is not None:
                assert tp in bound and bound[tp] == cand
                ret.append(cand)
            elif tp in bound:
                ret.append(bound[tp])
            elif param.has_default:
                t = value_type(param.default_value)
                if t is None:
                    raise _param_missing_error(fn_ir, param)
                ret.append(t)
            else:
                raise _param_missing_error(fn_ir, param)
        else:
            if cand is not None:
                ret.append(cand)
            elif param.has_default:
                t = value_type(param.default_value)
                if t is None:
                    raise _param_missing_error(fn_ir, param)
                ret.append(t)
            else:
                raise _param_missing_error(fn_ir, param)
    return tuple(ret)

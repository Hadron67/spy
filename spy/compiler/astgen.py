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
Local variables are addressable the same way: ``name = expr`` stores into
the slot ``name`` is already bound to - in this block, or in an enclosing
one - and *declares* the variable - a fresh ``Alloca`` - only when the
name is bound nowhere.  Every ``if`` body is a lexical block of its own
(a child of the enclosing block): a declaration inside it is invisible
after it, while an assignment there writes the variable it sees (the
outer slot is memory, so the write survives the join).  Global names -
everything that is not a variable in scope - are resolved here to their
Python objects.  Every global is an *immutable value*: in a value context a
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
import builtins
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast, get_args, get_origin

from . import hir, syntax
from .binop import BoolOp
from .errors import CompileError
from .fn import (
    ArgEntry,
    ClosureFunction,
    FunctionIR,
    Signature,
    SignatureFormalArg,
)
from .sval import (
    AnyValue,
    BytesType,
    CompileContext,
    Null,
    PointerType,
    StructDecl,
    StructType,
    Type,
    Value,
    VoidType,
    as_value,
    unwrap_comptime,
)
from .sval import (
    TypeVar as SpyTypeVar,
)
from .util import ArraySet, IndexedMap, TriState, frozendict

_BIN_OPS: dict[type[ast.AST], hir.BinaryOp] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.BitOr: '|',
    ast.BitAnd: '&',
    ast.BitXor: '^',
    ast.LShift: '<<',
    ast.RShift: '>>',
}

_BOOL_OPS: dict[type[ast.AST], BoolOp] = {ast.And: 'and', ast.Or: 'or'}


def _is_none_literal(node: ast.expr) -> bool:
    """Whether ``node`` is the Python literal ``None`` - the absent value an
    ``expr is None`` tests against (see ``_Builder._gen_is_none``)."""
    return isinstance(node, ast.Constant) and node.value is None

_UNARY_OPS: dict[type[ast.AST], hir.UnaryOp] = {ast.USub: '-', ast.Invert: '~'}

_CMP_OPS: dict[type[ast.AST], hir.CompareOp] = {
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
}

# the ``syntax.*`` markers that start a deferred region with a fixed set of
# triggering exits when they are the context manager of a ``with`` statement
# (see ``_Builder._gen_defer``); ``syntax.defer(flags)`` carries the flags
# itself and is handled separately
_DEFER_CALLS: dict[Any, hir.DeferPath] = {
    syntax.okdefer: hir.DeferPath.OK,
    syntax.errdefer: hir.DeferPath.ERR,
}

# the ``syntax.*`` markers that are *calls* in the source and are recognized by
# identity (unlike ``syntax.array`` and ``syntax.Comptime``, which are resolved
# through ``_gen_call``/``_split_comptime``)
_SYNTAX_CALLS = (syntax.ref, syntax.unroll, syntax.comptime, syntax.ptr_cast, syntax.as_func_ptr, syntax.typeof)

# the ``syntax.*`` classes that name a pointer type, by the (is_const, is_multi)
# of the pointer each one is; ``Array``/``Option`` are handled alongside them in
# ``_Builder._gen_type_ctor``
_POINTER_MARKERS: tuple[tuple[Any, bool, bool], ...] = (
    (syntax.Ptr, False, False),
    (syntax.ConstPtr, True, False),
    (syntax.MultiPtr, False, True),
    (syntax.ConstMultiPtr, True, True),
)


def _is_struct_class(obj: Any) -> bool:
    """Whether the raw global object ``obj`` is a ``@struct()`` class handle
    (see :class:`sval.StructDecl`): the parser recognizes a construction by
    its callee at parse time (see ``_Builder._gen_expr``)."""
    return isinstance(obj, StructDecl)

class _Scope:
    """One lexical block of a spy function: the variable bindings of the
    block (name -> the Alloca of its slot).  A read and an assignment
    both resolve through the enclosing blocks: an assignment stores into
    the slot the name is already bound to - the nearest enclosing
    binding, so an assignment inside a branch writes the variable it
    sees - and only a name that is bound *nowhere* is declared: it gets a
    fresh block-local slot, which is not visible outside its block.  The
    chain of enclosing blocks is the scope stack of the :class:`_Builder`
    translating the function (see ``_Builder._lookup``).

    ``pending`` names the slots the body's pre-scan declared but whose
    declaration statement has not run yet (see ``_Builder._predeclare``):
    such a slot is committed by the first statement that initializes it."""

    __slots__ = ('pending', 'vars')

    def __init__(self) -> None:
        # every binding names a *pointer to the referenced object*: a
        # variable's slot, the option a ``:=`` binds, or the payload of one an
        # unwrap binds (see ``_Builder._gen_walrus``/``_gen_unwrap``)
        self.vars: dict[str, hir.Value] = {}
        self.pending: set[str] = set()

class _ClosureScope(_Scope):
    """The root scope of one closure body: like a :class:`_Scope`, plus the
    captures the body resolved through the enclosing scopes (see
    ``_Builder._tunnel_through_closures``).  ``captures`` holds each captured
    place in the order the body's ``hir.Closure`` indices name it, and
    ``capture_index`` maps a captured name to its index so the same variable is
    captured once."""

    __slots__ = ('capture_index', 'captures', 'closure_fn')

    def __init__(self, closure_fn: ClosureFunction) -> None:
        super().__init__()
        self.captures: list[hir.Value] = []
        self.capture_index: dict[str, int] = {}
        self.closure_fn = closure_fn

class _Pragma:
    pass

@dataclass(frozen=True, slots=True)
class _Unroll(_Pragma):
    pass

@dataclass(frozen=True, slots=True)
class _Comptime(_Pragma):
    pass

_UNROLL = _Unroll()
_COMPTIME = _Comptime()

class _Builder:
    """Translates the AST of one function body into one linear
    instruction list of the untyped HIR.

    One builder translates the whole function; the lexical blocks it
    enters (the function body, an ``if`` branch, a loop body, an
    ``except`` clause) are a stack of :class:`_Scope` bindings, whose top
    is the block being translated.  The bindings (the parameters, for the
    function body; local declarations, in every block) are added as each
    block is translated.  An expression of a nested block looks up names
    through the stack.
    """

    def __init__(self, fn: Any, fn_ir: FunctionIR, type_vars: dict[TypeVar, Value], resolver: CompileContext) -> None:
        self.fn = fn
        self._fn_ir = fn_ir
        # the host the annotations/declarations are resolved in (see
        # ``parse_function``); a closure's ``@syntax.closure(exceptions=...)``
        # is resolved through it
        self._resolver = resolver
        # the open lexical blocks, innermost last: names resolve through
        # this stack (see ``_lookup``) and a declaration goes into its top
        # (see ``_declare``)
        self._scopes: list[_Scope] = [_Scope()]
        # the type parameters of the function (and of the struct a method
        # belongs to), keyed by the Python type parameter object their
        # annotations evaluate to: the names are the compile-time type values
        # a name in the body denotes (see ``_gen_expr``), and the Python
        # parameters themselves let a local annotation - which Python leaves
        # unevaluated - be resolved (see ``_gen_ann_assign``)
        self._type_vars = type_vars
        self._generic_names: dict[str, Value] = {tp.__name__: v for tp, v in type_vars.items()}
        self._pragmas: set[_Pragma] = set()
        self.insts: list[hir.Inst] = []
        # the type parameters of the closure body currently being generated, or
        # None outside one: a closure body may only name its own type
        # parameters, not the enclosing function's (see ``_gen_expr``)
        self._own_generic_names: set[str] | None = None
        # a counter that keeps the native names of the closures of one function
        # apart
        self._closure_counter: int = 0

    def _lookup(self, name: str) -> hir.Value | None:
        """The pointer the nearest binding of ``name`` holds, or None when the
        name is not bound in any open block.  A binding that lies outside the
        innermost closure body the name is read in is threaded through every
        closure between it and the reader (see ``_tunnel_through_closures``),
        so the returned value is a ``hir.Closure`` leaf of the reader's
        frame."""
        for i in range(len(self._scopes) - 1, -1, -1):
            slot = self._scopes[i].vars.get(name)
            if slot is not None:
                return self._tunnel_through_closures(i, name, slot)
        return None

    def _lookup_within_function(self, name: str) -> hir.Value | None:
        """The nearest binding of ``name`` in the *current* function body,
        searching every open block down to (and including) the innermost
        closure scope - but never crossing it.  It is what decides whether an
        assignment declares a fresh variable: a name assigned in a closure body
        is local to that closure, whatever the enclosing function binds."""
        for i in range(len(self._scopes) - 1, -1, -1):
            scope = self._scopes[i]
            slot = scope.vars.get(name)
            if slot is not None:
                return slot
            if isinstance(scope, _ClosureScope):
                break
        return None

    def _tunnel_through_closures(self, found: int, name: str, slot: hir.Value) -> hir.Value:
        """Bring the binding at scope index ``found`` to the innermost scope:
        for every closure boundary between it and the reader, capture the
        current value (the enclosing slot, or an enclosing closure's
        ``hir.Closure``) in that closure scope and replace the binding with the
        ``hir.Closure`` that names it.  A name is captured at most once per
        closure (see ``_ClosureScope.capture_index``)."""
        for i in range(found + 1, len(self._scopes)):
            scope = self._scopes[i]
            if isinstance(scope, _ClosureScope):
                index = scope.capture_index.get(name)
                if index is None:
                    index = len(scope.captures)
                    scope.captures.append(slot)
                    scope.capture_index[name] = index
                slot = hir.Closure(index)
        return slot

    def _declare(self, name: str, slot: hir.Value) -> None:
        """Bind ``name`` to ``slot`` in the innermost open block."""
        self._scopes[-1].vars[name] = slot

    def add(self, inst: hir.Inst) -> hir.Inst:
        self.insts.append(inst)
        return inst

    # -- statements -----------------------------------------------------------

    def _as_marker(self, node: ast.stmt) -> _Pragma | None:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            try:
                fn = self._try_resolve_object(node.value.func)
            except CompileError:
                # a name that does not resolve (yet) cannot be a marker: a
                # ``def`` name is only declared where its sub-block begins, so a
                # reference to it from an earlier sub-block is not a marker (its
                # own error surfaces when the statement is generated)
                return None
            if fn is syntax.unroll:
                return _UNROLL
            if fn is syntax.comptime:
                return _COMPTIME
        return None

    def _gen_body(self, stmts: list[ast.stmt]) -> None:
        """Generate a whole statement list, one *sub-block* at a time.  A
        sub-block begins at the block's start or at a nested ``def`` that
        follows a non-``def`` statement; every declaration of the sub-block - an
        assignment target, an annotated declaration, a ``def`` - is
        *pre-declared* just before the sub-block is generated (see
        ``_predeclare``), so a nested closure written before a variable is
        assigned can still capture it, and a name assigned in a closure body is
        local to that closure.  Declaring a ``def``'s slot where the sub-block
        begins (rather than at the block's head) keeps its ``Alloca`` in the
        same runtime block as the closure value it stores, so a sub-block that
        follows runtime control flow still compiles (see
        ``pending-problems.md`` #5).  A marker that no loop or declaration
        follows at its end is rejected."""
        index = 0
        while index < len(stmts):
            end = self._predeclare(stmts, index)
            for stmt in stmts[index:end]:
                self._gen_stmt(stmt)
            index = end
        if self._pragmas:
            raise CompileError('unused pragmas')

    def _predeclare(self, stmts: list[ast.stmt], start: int) -> int:
        """Declare every name one *sub-block* of ``stmts`` - the run beginning
        at ``start``, up to the next ``def`` that follows a non-``def`` - will
        bind: an assignment target, an annotated declaration (its annotation is
        evaluated here, so its type is fixed before the closure bodies below can
        capture it) and a ``def`` name (a full compile-time slot the closure
        value is written into).  A name bound already - by a parameter, an
        enclosing block, or a preceding marker - is left alone.  Compound
        statements are not descended into: their own bodies pre-declare the
        names they bind.  Returns the index the next sub-block begins at, where
        *its* declarations are pre-declared in turn (see ``_gen_body``)."""
        composable = False
        index = start
        while index < len(stmts):
            stmt = stmts[index]
            if (
                index > start
                and isinstance(stmt, ast.FunctionDef)
                and not isinstance(stmts[index - 1], ast.FunctionDef)
            ):
                # a ``def`` following a non-``def`` starts the next sub-block:
                # its slots are allocated there, in that sub-block's own runtime
                # block
                break
            marker = self._as_marker(stmt)
            if marker is _COMPTIME:
                composable = True
                index += 1
                continue
            if isinstance(stmt, ast.FunctionDef):
                self._predeclare_name(stmt.name, hir.InlineMode.FULL)
            elif isinstance(stmt, ast.Assign):
                # a ``syntax.comptime()`` marker makes the assignment declare a
                # compile-time variable of no declared type (``a = e`` is
                # ``a: Comptime = e``, see ``_gen_assign``): like the annotated
                # case, such a slot is committed by its initializing store and
                # declared where it is written - pre-declaring it here would put
                # its ``Alloca`` in the wrong runtime block, before a later
                # call's continuation (see ``_predeclare_ann`` and
                # ``pending-problems.md`` #5)
                if not composable and len(stmt.targets) > 0:
                    self._predeclare_target(stmt.targets[0], hir.InlineMode.NONE)
            elif isinstance(stmt, ast.AnnAssign):
                self._predeclare_ann(stmt, composable)
            composable = False
            index += 1
        return index

    def _predeclare_name(self, name: str, mode: hir.InlineMode) -> None:
        if self._lookup_within_function(name) is not None:
            return
        slot = self.add(hir.Alloca(mode))
        self._declare(name, slot)
        self._scopes[-1].pending.add(name)

    def _predeclare_target(self, target: ast.expr, mode: hir.InlineMode) -> None:
        if isinstance(target, ast.Name):
            self._predeclare_name(target.id, mode)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._predeclare_target(elt, mode)

    def _predeclare_ann(self, node: ast.AnnAssign, marker_comptime: bool) -> None:
        target = node.target
        if not isinstance(target, ast.Name):
            return
        if self._lookup_within_function(target.id) is not None:
            return
        is_comptime, type_node = self._split_comptime(node.annotation)
        if is_comptime and marker_comptime:
            raise CompileError(
                f"'{target.id}' is already declared compile-time by its "
                f"annotation; drop the syntax.comptime() marker before it"
            )
        is_comptime = is_comptime or marker_comptime
        if is_comptime and type_node is None:
            # a compile-time slot with no declared type is committed by its
            # initializing store, not at the pre-scan: declaring it here would
            # put its ``Alloca`` in the wrong runtime block (see
            # ``pending-problems.md`` #5).  It is declared where it is written
            # instead, like any ordinary declaration.
            return
        declared = None if type_node is None else self._as_value(self._gen_expr(type_node)[0])
        slot = self.add(hir.Alloca(
            hir.InlineMode.FULL if is_comptime else hir.InlineMode.NONE, declared
        ))
        self._declare(target.id, slot)
        self._scopes[-1].pending.add(target.id)

    def _consume_pending(self, name: str) -> bool:
        """Whether ``name`` is a slot this body pre-declared and whose
        declaration statement has not run yet; a consuming caller emits the
        ``CommitSlot`` that materializes it."""
        pending = self._scopes[-1].pending
        if name in pending:
            pending.discard(name)
            return True
        return False

    def _gen_stmt(self, node: ast.stmt) -> None:
        if (marker := self._as_marker(node)) is not None:
            self._pragmas.add(marker)
            return
        is_unroll = _UNROLL in self._pragmas
        is_comptime = _COMPTIME in self._pragmas
        self._pragmas.clear()

        fn_name = self._fn_ir.name
        match node:
            case ast.While():
                if is_comptime:
                    raise CompileError('comptime not allowed here')
                self._gen_while(node, is_unroll)
                return
            case ast.For():
                if is_comptime:
                    raise CompileError('comptime not allowed here')
                self._gen_for(node, is_unroll)
                return
            case ast.AnnAssign():
                if is_unroll:
                    raise CompileError('unroll not allowed here')
                self._gen_ann_assign(node, is_comptime)
                return
            case ast.Assign():
                if is_unroll:
                    raise CompileError('unroll not allowed here')
                self._gen_assign(node.targets[0], node.value, is_comptime)
                return
        if is_unroll or is_comptime:
            raise CompileError('comptime/unroll not allowed here')
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
            case ast.FunctionDef():
                self._gen_function_def(node)
            case ast.Raise():
                # the exception is built into a slot of its own (result-location
                # semantics); ``hir.Raise`` then tags and dispatches it (see
                # ``interp``)
                if node.exc is None:
                    raise CompileError(
                        f'a bare raise is not supported yet in spy function {fn_name}'
                    )
                slot = self.add(hir.Alloca(hir.InlineMode.NON_AGGREGATE))
                self._gen_result_loc(node.exc, slot)
                self.add(hir.Raise(slot))
            case ast.Pass():
                pass
            case ast.Expr():
                self._gen_expr(node.value)[0]
            case ast.AugAssign():
                self._gen_augassign(node)
            case ast.If():
                self._gen_if(node)
            case ast.Break():
                self.add(hir.BreakLoop())
            case ast.Continue():
                self.add(hir.Continue())
            case ast.Try():
                self._gen_try(node)
            case ast.With():
                self._gen_defer(node)
            case ast.Match():
                self._gen_match(node)
            case _:
                raise CompileError(
                    f"unsupported statement {type(node).__name__} in spy function {fn_name}"
                )

    def _gen_block(self, stmts: list[ast.stmt]) -> None:
        """Translate one lexical block - a branch body, a loop body, the
        ``else`` clause of a loop - appending its instructions to this
        builder's list.  The block is a lexical scope of its own, a child of
        the scope enclosing it: its declarations go into a fresh scope that
        is dropped when the block ends, so a name it *declares* is not
        visible after the block (an assignment to a name it sees writes that
        variable, see ``_gen_assign``)."""
        self._scopes.append(_Scope())
        self._gen_body(stmts)
        self._scopes.pop()

    def _gen_if(self, node: ast.If) -> None:
        """Translate one ``if``/``elif``/``else``.  A plain condition becomes a
        WASM-style :class:`hir.If` (its branches follow in the same flat list,
        delimited by ``hir.Else``/``hir.End``).  An ``and`` chain is lowered
        with a :class:`hir.Block` instead (see ``_gen_and_condition``), its body
        sitting inside the block: the operands' ``:=`` bindings then *dominate*
        the body, which the ``if``'s branches (reached from both sides of every
        short-circuit) would not give them.

        The condition opens a scope of its own: a ``:=`` declared in it is
        visible in the then branch (a child scope of the condition's), but
        neither in the else branch nor after the ``if``."""
        has_else = len(node.orelse) > 0
        self._scopes.append(_Scope())
        if self._is_and_chain(node.test):
            assert isinstance(node.test, ast.BoolOp)
            self.add(hir.Block())
            if has_else:
                self.add(hir.Block())
            self._gen_and_condition(node.test)
            self._gen_block(node.body)
            if has_else:
                # the body succeeded: leave both blocks, skipping the else
                self.add(hir.BreakIf(None, 2))
                self.add(hir.End())
                self._scopes.pop()
                self._gen_block(node.orelse)
            else:
                self._scopes.pop()
            self.add(hir.End())
            return
        cond = self._gen_expr(node.test)[0]
        self.add(hir.If(self.add(hir.AsBool(cond))))
        self._gen_block(node.body)
        self._scopes.pop()
        if has_else:
            self.add(hir.Else())
            self._gen_block(node.orelse)
        self.add(hir.End())

    def _is_and_chain(self, node: ast.expr) -> bool:
        """Whether ``node`` is an ``and`` chain (a ``BoolOp`` of ``and``): the
        only condition the ``if``/``while`` lowering inlines into a
        ``hir.Block`` (see ``_gen_if``/``_gen_and_condition``)."""
        return isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)

    def _gen_and_condition(self, node: ast.BoolOp) -> None:
        """Lower an ``and`` chain used as an ``if``/``while`` condition into
        ``break_if``s: every operand is tested and a false one leaves the
        enclosing :class:`hir.Block` (the code after it - the body or the else
        clause - is skipped).  The operands are *not* scoped apart: their
        ``:=`` bindings are what the block's body reads."""
        for value in node.values:
            cond = self.add(hir.AsBool(self._gen_expr(value)[0]))
            self.add(hir.BreakIf(self.add(hir.Not(cond)), 1))

    def _gen_while(self, node: ast.While, is_inline: bool) -> None:
        """Translate one ``while``/``else`` statement into a dead ``loop``:

        .. code-block:: text

            loop
                %1 = <cond>
                %2 = not %1
                if %2
                    <else_clause>
                    break
                else
                    <body>
                end
            end

        The condition is evaluated at the head of every iteration; when it
        turns false the ``else`` clause runs and the implicit ``break``
        leaves the loop, while otherwise the body runs and its falling end
        loops back.  Only a natural exit (the condition turning false)
        reaches the else clause - a ``break`` in the body skips it, like
        Python - and a ``continue`` jumps back to the head, re-evaluating
        the condition.  Both the body and the else clause are lexical
        blocks of their own (children of the enclosing block).

        An ``and`` chain is lowered with a :class:`hir.Block` whose body holds
        the loop body, so the operands' ``:=`` bindings dominate it (see
        ``_gen_if``); the body ends with an explicit ``continue`` back to the
        loop head, and a false operand leaves the block onto the else clause.

        The ``syntax.unroll()`` marker immediately before the ``while`` makes it
        a *compile-time* loop: ``is_inline`` marks the ``Loop`` and the
        interpreter unrolls the body once per compile-time iteration instead of
        emitting a back edge (see ``interp``)."""
        self.add(hir.Loop(is_inline=is_inline))
        # the condition opens a scope of its own: a ``:=`` declared in it is
        # visible in the body, but not in the ``else`` clause (which runs when
        # the condition is false, so the walrus may not have run) nor after the
        # loop.
        cond_scope = _Scope()
        self._scopes.append(cond_scope)
        if self._is_and_chain(node.test) and not is_inline:
            assert isinstance(node.test, ast.BoolOp)
            self.add(hir.Block())
            self._gen_and_condition(node.test)
            self._gen_block(node.body)
            # the body fell off its end: loop back (the else clause is reached
            # only by a false operand leaving the block)
            self.add(hir.Continue())
            self.add(hir.End())
            self._scopes.pop()
            self._gen_block(node.orelse)
            self.add(hir.BreakLoop())
            self.add(hir.End())
            return
        # ``%2 = not %1``: negating a boolean is a value -> value instruction
        # (``hir.Not``), so the ``if`` sees a boolean value whether the
        # condition is compile-time or not
        cond = self.add(hir.AsBool(self._gen_expr(node.test)[0]))
        self.add(hir.If(self.add(hir.Not(cond))))
        self._scopes.pop()
        self._gen_block(node.orelse)
        self._scopes.append(cond_scope)
        self.add(hir.BreakLoop())
        self.add(hir.Else())
        self._gen_block(node.body)
        self._scopes.pop()
        self.add(hir.End())
        self.add(hir.End())

    def _gen_for(self, node: ast.For, is_inline: bool) -> None:
        """Translate one ``for exprs in iter: body`` (with an optional
        ``else``) into an explicit iterator loop:

        .. code-block:: text

            %it = alloca
            %it = iter.__iter__()
            commit %it
            loop
                try
                    <exprs> = %it.__next__()
                    commit <exprs>
                    <body>
                except StopIteration
                    <else>
                    break
                end
            end

        The iterator is taken once, before the loop; every iteration asks it for
        the next element (with result-location semantics straight into the
        place ``exprs`` denotes, so a destructuring target unpacks it in place)
        and runs the body.  The loop ends when ``__next__`` raises
        ``StopIteration``, which the ``try`` catches - when the ``else`` clause
        runs before the ``break``.  A ``break`` in the body leaves the loop
        directly (skipping the else clause), like Python, and a ``continue``
        starts the next iteration.  The ``syntax.unroll()`` marker immediately
        before the ``for`` opens the loop as a compile-time one (see
        ``interp``): the iterable and the loop variable(s) of a compile-time
        loop are *compile-time values*, so both are built into inline slots -
        a compile-time iterable is an aggregate with no runtime representation
        of its own, which an ordinary expression temporary may not hold."""
        from ..std.core import StopIteration

        # the iterator: ``__iter__`` once, before the loop (a fresh iterator per
        # iteration would restart the iteration)
        it = self.add(hir.Alloca(hir.InlineMode.FULL if is_inline else hir.InlineMode.NON_AGGREGATE))
        if is_inline:
            # a compile-time iterable is evaluated into an inline slot of its
            # own (result-location semantics), so that it never goes through an
            # expression temporary
            iter_slot = self.add(hir.Alloca(hir.InlineMode.FULL))
            self._gen_result_loc(node.iter, iter_slot)
            self.add(hir.CommitSlot(iter_slot))
            base = iter_slot
        else:
            base = self._as_ref(self._gen_expr(node.iter)[0])
        self.add(hir.CallMethodInplace(base, '__iter__', hir.CallArgs((), ()), it))
        self.add(hir.CommitSlot(it))
        # the loop body, in a block of its own: the loop variable(s) are
        # declared there and are not visible after the loop.  Its scope also
        # encloses the ``else`` clause, which reads the loop variable.
        self._scopes.append(_Scope())
        self.add(hir.Loop(is_inline=is_inline))
        self.add(hir.Try((None,), (hir.Const(StopIteration),)))
        new_slots: list[hir.Value] = []
        place = self._gen_lhs(
            node.target, new_slots,
            hir.InlineMode.FULL if is_inline else hir.InlineMode.NONE,
        )
        self.add(hir.CallMethodInplace(it, '__next__', hir.CallArgs((), ()), place))
        for slot in new_slots:
            self.add(hir.CommitSlot(slot))
        self._gen_body(node.body)
        self.add(hir.Except(0))
        self._gen_body(node.orelse)
        self.add(hir.BreakLoop())
        self.add(hir.End())
        self.add(hir.End())
        self._scopes.pop()

    def _gen_try(self, node: ast.Try) -> None:
        """Translate one ``try``/``except`` statement: the try body, then one
        ``hir.Except`` marker (and clause body) per handler, closed by an
        ``hir.End``.  Every clause's type expression is evaluated as a value
        *before* the ``Try`` opens and carried by it (``hir.Try.except_types``):
        an error is dispatched to a clause while the try body is typed, so the
        clause types have to be known by then.  A clause that binds its
        exception (``as e``) opens with an :class:`hir.ExceptBind`, whose value
        - the address the caught exception was delivered through - the clause's
        ``as`` name is bound to (see ``interp``).  ``else``/``finally`` are not
        supported yet."""
        fn_name = self._fn_ir.name
        if len(node.orelse) > 0:
            raise CompileError(f'try-else is not supported in spy function {fn_name}')
        if len(node.finalbody) > 0:
            raise CompileError(f'try-finally is not supported in spy function {fn_name}')
        if len(node.handlers) == 0:
            raise CompileError(f'a try must have an except clause in spy function {fn_name}')
        binds: list[hir.ExceptBind | None] = []
        for handler in node.handlers:
            if handler.name is not None:
                binds.append(hir.ExceptBind())
            else:
                binds.append(None)
        # the clause types are evaluated here, before the ``Try``: their
        # instructions run before any instruction of the try body can dispatch
        # an error (see ``interp._find_catching_clause``)
        except_types: list[hir.Value | None] = []
        for handler in node.handlers:
            except_types.append(
                None if handler.type is None
                else self._as_value(self._gen_expr(handler.type)[0])
            )
        self.add(hir.Try(tuple(binds), tuple(except_types)))
        self._gen_body(node.body)
        for index, handler in enumerate(node.handlers):
            self.add(hir.Except(index))
            bind = binds[index]
            if bind is not None:
                # the bind reads the clause's payload pointer: it is the first
                # instruction of the clause body
                self.add(bind)
            self._scopes.append(_Scope())
            if handler.name is not None:
                assert bind is not None
                self._declare(handler.name, bind)
            self._gen_body(handler.body)
            self._scopes.pop()
        self.add(hir.End())

    def _gen_defer(self, node: ast.With) -> None:
        """Translate one ``with syntax.defer(...): body`` (or ``okdefer``/
        ``errdefer``) into a deferred region: a :class:`hir.Defer` marker carrying
        the exits that trigger it (``syntax.defer``'s bitflags; ``okdefer``/
        ``errdefer`` are the ``OK``/``ERR`` spellings), the body in a lexical block
        of its own, and the matching ``hir.End``.  The body is not run where it is
        written - it is deferred to the exit of the enclosing region (see
        ``interp``) - and it may not jump out of itself (``return``/``break``/
        ``continue``/``raise`` that would leave it are rejected by the
        interpreter).

        Only a single, unbound context manager is accepted, and it has to be a
        ``syntax`` marker (see ``_gen_defer_flags``)."""
        fn_name = self._fn_ir.name
        if len(node.items) != 1:
            raise CompileError(
                f'a defer statement takes exactly one context manager in spy '
                f'function {fn_name}'
            )
        item = node.items[0]
        if item.optional_vars is not None:
            raise CompileError(
                f'a defer statement binds no name in spy function {fn_name}'
            )
        flags = self._gen_defer_flags(item.context_expr, fn_name)
        self.add(hir.Defer(flags))
        self._gen_block(node.body)
        self.add(hir.End())

    def _gen_defer_flags(self, call: ast.expr, fn_name: str) -> hir.Value:
        """The compile-time value the exits a ``with syntax.defer(...):`` region
        triggers are read off, from the context-manager call: ``defer(flags)``
        takes any compile-time integer expression of the ``syntax`` bitflags (no
        argument means ``ALL``), while ``okdefer``/``errdefer`` are the fixed
        ``OK``/``ERR`` spellings (see ``_DEFER_CALLS``)."""
        if not isinstance(call, ast.Call):
            raise CompileError(self._defer_what(fn_name))
        target = self._try_resolve_object(call.func)
        if target is syntax.defer:
            if len(call.args) == 0 and len(call.keywords) == 0:
                return hir.Const(int(hir.DeferPath.ALL))
            if len(call.args) == 1 and len(call.keywords) == 0:
                value = call.args[0]
            elif (
                len(call.args) == 0
                and len(call.keywords) == 1
                and call.keywords[0].arg == 'flags'
            ):
                value = call.keywords[0].value
            else:
                raise CompileError(
                    f'syntax.defer takes at most one flags argument in spy '
                    f'function {fn_name}'
                )
            # the flags are an ordinary compile-time integer expression (the
            # interpreter folds it when the region is opened, see
            # ``interp._exec_defer``)
            return self._as_value(self._gen_expr(value)[0])
        flags = _DEFER_CALLS.get(target)
        if flags is None:
            raise CompileError(self._defer_what(fn_name))
        if len(call.args) > 0 or len(call.keywords) > 0:
            raise CompileError(
                f'syntax.okdefer()/syntax.errdefer() take no argument in spy '
                f'function {fn_name}'
            )
        return hir.Const(int(flags))

    def _defer_what(self, fn_name: str) -> str:
        return (
            f'expected syntax.defer(...)/syntax.okdefer()/syntax.errdefer() '
            f'in spy function {fn_name}'
        )

    # -- variables ------------------------------------------------------------

    def _gen_lhs(self, target: ast.expr, new_slots: list[hir.Value], inline_mode: hir.InlineMode = hir.InlineMode.NONE) -> hir.Value:
        """The place - or, for a destructuring target, the ``hir.Tuple`` of
        places - a target denotes, declaring every name it binds nowhere (their
        fresh slots are appended to ``new_slots``).  It is the first half of an
        assignment, shared by ``_gen_assign`` and the ``for`` desugaring (see
        ``_gen_for``): the caller generates the value into the returned place
        with result-location semantics and then commits ``new_slots``.

        ``inline_mode`` is how much of the value the fresh slots may keep inline
        (see ``hir.Alloca``): the loop variable of a compile-time ``for`` is a
        compile-time value, so it is bound to an inline slot."""
        if isinstance(target, ast.Tuple):
            # a destructuring target is a tuple of addresses, one per element:
            # the right-hand side is generated straight into them (result-
            # location semantics), so no intermediate tuple value is built
            return self._gen_target_tuple(target, new_slots, inline_mode)
        if isinstance(target, ast.Name):
            slot = self._lookup_within_function(target.id)
            if slot is None:
                # the name is bound nowhere: declare it here, in the current block
                slot = self.add(hir.Alloca(inline_mode))
                self._declare(target.id, slot)
                new_slots.append(slot)
                return slot
            if self._consume_pending(target.id):
                # a slot the body pre-declared: commit it once initialized
                new_slots.append(slot)
            return slot
        lhs = self._gen_expr(target, False)[0]
        if not lhs.is_ref:
            raise CompileError(f"the target of an assignment must be a variable, got {target}")
        return lhs.value

    def _gen_assign(self, target: ast.expr, value: ast.expr, by_marker: bool = False) -> None:
        """One ``target = expr`` statement.  An assignment to a name that
        is already bound - in this block, or in an enclosing one, a
        parameter included - stores into that slot, and reads that same
        variable on the right hand side (``x = x + 1`` uses the value it
        already has); only a name bound *nowhere* is declared, as a fresh
        block-local slot that shadows nothing and is invisible after the
        block.  A declaration is bound before its initializer is
        generated, so a self-referencing declaration (``y = y + 1``) reads
        the not-yet-stored slot - a compile error when it runs, like an
        unbound local.  A call on the right hand side writes its result
        straight into the target slot (result-location semantics): a
        constructor ``x = Bar(...)`` fills the fields of the slot in
        place, and a scalar call result is only recorded in it.

        ``by_marker`` says a ``syntax.comptime()`` marker precedes the
        statement: the name it declares is a compile-time variable - a box
        rather than memory, exactly like ``name: Comptime`` (see
        ``_gen_ann_assign``)."""
        if by_marker and not self._declares_a_fresh_name(target):
            raise CompileError(
                'syntax.comptime() marks a declaration: it must be followed by an '
                'assignment that declares a fresh variable'
            )
        new_slots: list[hir.Value] = []
        place = self._gen_lhs(
            target, new_slots,
            hir.InlineMode.FULL if by_marker else hir.InlineMode.NONE,
        )
        self._gen_result_loc(value, place)
        for slot in new_slots:
            self.add(hir.CommitSlot(slot))

    def _declares_a_fresh_name(self, target: ast.expr) -> bool:
        """Whether an assignment to ``target`` declares a name that is bound
        nowhere - a declaration, which is what a ``syntax.comptime()`` marker
        marks (see ``_gen_assign`` and ``_gen_lhs``, whose declaration rule this
        mirrors).  A slot the body pre-declared but has not initialized yet is
        a declaration all the same."""
        match target:
            case ast.Name():
                return self._lookup_within_function(target.id) is None or self._is_pending(target.id)
            case ast.Tuple():
                return all(self._declares_a_fresh_name(elt) for elt in target.elts)
            case _:
                return False

    def _is_pending(self, name: str) -> bool:
        return name in self._scopes[-1].pending

    def _gen_ann_assign(self, node: ast.AnnAssign, by_marker: bool = False) -> None:
        """One annotated declaration ``name: T`` or ``name: T = expr``.  The
        annotation declares the type of the variable: ``name: T`` declares the
        type ``T``, ``name: Comptime`` a compile-time variable whose type its
        value determines, and ``name: Comptime[T]`` a compile-time variable of
        the declared type ``T`` (see ``hir.Alloca``).  A ``syntax.comptime()``
        marker before the declaration (``by_marker``) says the same: the
        variable is a compile-time one, of the annotated type - or of the type
        its value determines when it declares none.  The variable gets a fresh
        slot - the annotation's type is the compile-time value ``_gen_expr``
        produces for it, which the interpreter resolves against the call's type
        arguments - and, when a value is written, is initialized with it
        (result-location semantics, like a plain declaration; see
        ``_gen_assign``)."""
        fn_name = self._fn_ir.name
        target = node.target
        if not isinstance(target, ast.Name):
            raise CompileError(
                f"only a name can be annotated in spy function {fn_name}, "
                f"got {ast.unparse(target)!r}"
            )
        if self._is_pending(target.id):
            # the body pre-declared it (its annotation was evaluated there):
            # this statement only initializes the slot
            slot = self._scopes[-1].vars[target.id]
            self._consume_pending(target.id)
            if node.value is not None:
                self._gen_result_loc(node.value, slot)
            self.add(hir.CommitSlot(slot))
            return
        if self._lookup(target.id) is not None:
            raise CompileError(
                f"'{target.id}' is already bound in spy function {fn_name}; "
                f"an annotated declaration introduces a new variable"
            )
        is_comptime, type_node = self._split_comptime(node.annotation)
        if is_comptime and by_marker:
            raise CompileError(
                f"'{target.id}' is already declared compile-time by its "
                f"annotation; drop the syntax.comptime() marker before it"
            )
        is_comptime = is_comptime or by_marker
        declared = None if type_node is None else self._as_value(self._gen_expr(type_node)[0])
        slot = self.add(hir.Alloca(
            hir.InlineMode.FULL if is_comptime else hir.InlineMode.NONE, declared
        ))
        self._declare(target.id, slot)
        if node.value is not None:
            self._gen_result_loc(node.value, slot)
        self.add(hir.CommitSlot(slot))

    def _split_comptime(self, node: ast.expr) -> tuple[bool, ast.expr | None]:
        """Split a *local* variable's annotation into its ``Comptime`` marker
        and the type it wraps (the annotation counterpart of the parameter
        path, see ``sval.unwrap_comptime``): the bare ``Comptime`` splits to
        ``(True, None)`` and ``Comptime[T]`` to ``(True, T)``, anything else to
        ``(False, node)``.  The marker is recognized by the global name it is
        written as, so a variable of the same name shadows it like any
        global."""
        if self._try_resolve_object(node) is syntax.Comptime:
            return True, None
        if isinstance(node, ast.Subscript) and self._try_resolve_object(node.value) is syntax.Comptime:
            if isinstance(node.slice, ast.Tuple):
                raise CompileError('Comptime takes exactly one type argument')
            return True, node.slice
        return False, node

    def _split_type_value(self, node: ast.expr) -> tuple[bool, ast.expr]:
        """Split a parameter annotation into its ``type[X]`` marker and the
        type it wraps: ``type[X]`` splits to ``(True, X)`` and anything else to
        ``(False, node)``.  The marker is recognized by the global name it is
        written as (the builtin ``type``), so a variable of the same name
        shadows it like any global."""
        if isinstance(node, ast.Subscript) and self._try_resolve_object(node.value) is type:
            if isinstance(node.slice, ast.Tuple):
                raise CompileError('type[...] takes exactly one type argument')
            return True, node.slice
        return False, node

    def _gen_target_tuple(self, target: ast.Tuple, new_slots: list[hir.Value], inline_mode: hir.InlineMode = hir.InlineMode.NONE) -> hir.Value:
        """The tuple of addresses a destructuring target denotes (``hir.TuplePtr``):
        a plain target contributes the address of its slot (or field), a nested
        tuple target contributes its own tuple of addresses.  Only a name
        that is bound nowhere is declared (see ``_gen_assign``); its fresh slot
        keeps values inline as ``inline_mode`` says (see ``_gen_lhs``)."""
        elems: list[hir.Value] = []
        for elt in target.elts:
            if isinstance(elt, ast.Tuple):
                elems.append(self._gen_target_tuple(elt, new_slots, inline_mode))
                continue
            if isinstance(elt, ast.Name):
                slot = self._lookup_within_function(elt.id)
                if slot is None:
                    slot = self.add(hir.Alloca(inline_mode))
                    self._declare(elt.id, slot)
                    new_slots.append(slot)
                elif self._consume_pending(elt.id):
                    new_slots.append(slot)
            ref = self._gen_expr(elt, False)[0]
            if not ref.is_ref:
                raise CompileError(
                    f"target of a destructuring assignment must be addressable, got {elt}"
                )
            elems.append(ref.value)
        return self.add(hir.TuplePtr(tuple(elems)))

    def _gen_augassign(self, node: ast.AugAssign) -> None:
        """One ``name += expr`` statement: read the value, add ``expr``
        and store the result back.  The target is a variable slot or the
        address of a field of a runtime struct value (``self.h += e``);
        ``+=`` never declares: it requires the name to be declared."""
        fn_name = self._fn_ir.name
        # ``x op= y`` is ``x = x op y`` (or the in-place magic method, see
        # ``interp``): every binary operator the source spells is accepted here
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CompileError(
                f"unsupported augmented assignment operator "
                f"{type(node.op).__name__} in spy function {fn_name}"
            )
        lhs = self._gen_expr(node.target, False)[0]
        if not lhs.is_ref:
            raise CompileError(f"target of augmented assignment must be a variable, got {node.target}")
        rhs = self._gen_expr(node.value)[0]
        self.add(hir.BinaryAssign(op, lhs.value, rhs))

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
        context) or a ``hir.ConstRef`` (reference context).

        A builtin the compiler gives a spy meaning resolves to the ``std``
        struct it stands for: ``range`` is the iterable struct and
        ``StopIteration`` the exception its ``__next__`` raises - the
        desugarings the compiler writes refer to them by these names."""
        closure = self._resolve_closure(name)
        if closure is not None:
            return closure
        fn = self.fn
        globals = fn.__globals__
        if name in globals:
            return globals[name]
        builtin = getattr(builtins, name, None)
        if builtin is range or builtin is StopIteration:
            from ..std import core
            return core.range if builtin is range else core.StopIteration
        if builtin is bytes:
            # ``bytes`` names the compile-time byte-string type: its spy value is
            # the type itself, so ``Comptime[bytes]`` and friends resolve
            return BytesType()
        if builtin is str:
            raise CompileError(
                'the str type is not available in spy: use bytes (a string '
                'literal is encoded to bytes automatically)'
            )
        if builtin is ord:
            # ``ord(x)`` is lowered to ``hir.Ord`` by the call parser (see
            # ``_gen_call``)
            return builtin
        if builtin is len:
            # ``len(x)`` is lowered to ``hir.Len`` by the call parser (see
            # ``_gen_call``)
            return builtin
        if builtin is isinstance:
            # ``isinstance(value, T)`` against a tagged union: the parser lowers
            # it to the tag test itself (see ``_gen_isinstance``)
            return builtin
        if builtin is type:
            # ``type[X]`` annotates a type-valued parameter (see
            # ``_split_type_value``): the marker is recognized by identity
            return builtin
        raise CompileError(
            f"name '{name}' is not defined in the scope of function {self._fn_ir.name}"
        )

    def _as_ref(self, node: ArgEntry[hir.Value]):
        if node.is_ref:
            return node.value
        loc = self.add(hir.Alloca(hir.InlineMode.NON_AGGREGATE))
        self.add(hir.Store(loc, node.value))
        self.add(hir.CommitSlot(loc))
        return loc

    def _as_value(self, node: ArgEntry[hir.Value]) -> hir.Value:
        if node.is_ref:
            return self._make_load(node.value)
        return node.value

    def _try_resolve_object(self, node: ast.expr) -> Any | None:
        """The raw Python object a *global* expression denotes, or None when
        it denotes no global (a variable, a type parameter, a call, ...): a
        name bound to a global, or an attribute of one (``syntax.ref``)."""
        match node:
            case ast.Name():
                if self._lookup(node.id) is not None or node.id in self._generic_names:
                    return None
                return self._resolve_global(node.id)
            case ast.Attribute():
                base = self._try_resolve_object(node.value)
                return None if base is None else getattr(base, node.attr, None)
            case _:
                return None

    def _is_syntax_call(self, callee: ast.expr) -> bool:
        fn = self._try_resolve_object(callee)
        if fn is None:
            return False
        return fn in _SYNTAX_CALLS

    def _gen_expr(self, node: ast.expr, allow_retloc: bool = True) -> tuple[ArgEntry[hir.Value], bool]:
        """A reference to the value of ``node``: addressable names give
        their slot, the fields of a runtime struct value give their
        address (a :class:`hir.FieldAddr` chain rooted at the storage of
        the base), ``ref(a)`` gives the address of ``a`` and ``p[...]`` the
        address the pointer value ``p`` holds (C's ``&`` and ``*``), globals
        - immutable values - give a :class:`hir.ConstRef` to them, and
        everything else gives a pointer to a freshly allocated slot holding
        its value.

        The flag tells whether the expression denotes a *struct*: the name
        of a ``@struct()`` class, or a specialization of one (``Foo[i32]``).
        Only the callee of a call reads it, where ``Foo(...)`` becomes a
        construction (see ``_gen_call``); every other consumer takes the
        first element of the pair and drops the flag."""
        match node:
            case ast.Name():
                generic = self._generic_names.get(node.id)
                if generic is not None:
                    # a type parameter of the function (or of the struct a
                    # method belongs to) used as a value: the interpreter
                    # resolves it to the type the call solved it to, from
                    # the frame it runs in (see ``interp.operand``).  A closure
                    # body may only name its *own* type parameters: the
                    # enclosing function's are not available where it runs
                    if self._own_generic_names is not None and node.id not in self._own_generic_names:
                        raise CompileError(
                            f"a closure may not reference the enclosing function's "
                            f"type parameter '{node.id}' in spy function {self._fn_ir.name}"
                        )
                    return ArgEntry(hir.Const(generic), False), False
                ref = self._gen_name(node.id)
                # the name of a class denotes a struct: calling it constructs
                # one.  A class is a global (a variable of the same name
                # shadows it), so the reference to it is a ``ConstRef``
                is_struct = isinstance(ref, hir.ConstRef) and _is_struct_class(ref.value)
                return ArgEntry(ref, True), is_struct
            case ast.Attribute():
                base = self._as_ref(self._gen_expr(node.value)[0])
                if isinstance(base, hir.ConstRef):
                    if hasattr(base.value, node.attr):
                        return ArgEntry(hir.ConstRef(getattr(base.value, node.attr)), True), False
                    raise AttributeError(f"Attribute '{node.attr}' not found on {base.value}")
                return ArgEntry(self.add(hir.FieldAddr(base, node.attr)), True), False
            case ast.Constant():
                value = node.value
                if isinstance(value, str):
                    # spy has no ``str`` type: a string literal is a byte string,
                    # encoded at parse time (``b'...'`` is already ``bytes``)
                    value = value.encode()
                if isinstance(value, (int, float, complex, bytes, bool)) or value is None:
                    return ArgEntry(hir.Const(value), False), False
                raise CompileError(f"unsupported constant {node.value!r}")
            case ast.UnaryOp(op=ast.Not()):
                # ``not`` is value -> value (``hir.Not``), so it needs no result
                # location: ``not expr`` is ``Not(AsBool(expr))``
                return ArgEntry(self.add(hir.Not(self.add(hir.AsBool(self._gen_expr(node.operand)[0])))), False), False
            case ast.Compare():
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise CompileError(
                        "chained comparisons are not supported yet"
                    )
                op_type = type(node.ops[0])
                if op_type in (ast.Is, ast.IsNot):
                    return self._gen_is_none(node, op_type is ast.IsNot)
                op = _CMP_OPS.get(op_type)
                if op is None:
                    raise CompileError(
                        f"unsupported comparison {op_type.__name__}"
                    )
                lhs = self._gen_expr(node.left)[0]
                rhs = self._gen_expr(node.comparators[0])[0]
                return ArgEntry(self.add(hir.Compare(op, lhs, rhs)), False), False
            case ast.NamedExpr():
                return self._gen_walrus(node), False
            case ast.Tuple():
                values = tuple(self._gen_expr(elt)[0] for elt in node.elts)
                return ArgEntry(self.add(hir.Tuple(values)), False), False
            case ast.Lambda():
                # a lambda is a forced-inline closure (see ``_gen_lambda``)
                return ArgEntry(self._gen_lambda(node), False), False
            case ast.Call() if self._is_syntax_call(node.func):
                callee = self._try_resolve_object(node.func)
                assert callee is not None
                return self._gen_syntax_call(callee, node.args)
            case ast.Subscript():
                if isinstance(node.slice, ast.Constant) and node.slice.value is Ellipsis:
                    # ``expr[...]`` is C's ``*expr``: the place the pointer value
                    # points at.  It denotes a *reference* like a name does, so a
                    # store may target it (``p[...] = v``)
                    return ArgEntry(self._as_value(self._gen_expr(node.value)[0]), True), False
                marker = self._try_resolve_object(node.value)
                if marker is Literal:
                    # ``Literal[X]``: the compile-time value ``X`` (the very value
                    # a ``typing.Literal`` annotation denotes at the Python level,
                    # see ``sval.as_value``), used where a *value* is what a type
                    # argument stands for - e.g. the length of
                    # ``Array[T, Literal[N]]``
                    return ArgEntry(self._gen_literal(node.slice), False), False
                if marker is not None:
                    # ``Ptr[T]``/``Array[T, N]``/``Option[T]``/...: a ``syntax``
                    # type marker used as a value.  The type is built by the
                    # interpreter from the executing frame's type parameters
                    # (see ``hir.PointerType`` and friends).
                    built = self._gen_type_ctor(marker, node.slice)
                    if built is not None:
                        return ArgEntry(built, False), False
                base, base_is_struct = self._gen_expr(node.value)
                if not base.is_ref:
                    raise CompileError(f"subscript of non-reference {base}")
                if isinstance(node.slice, ast.Slice):
                    # ``p[a:b]`` / ``p[a:b:c]``: the slice *object* the subscript
                    # turns into a ``SlicePtr`` when the base is a multi pointer
                    # (see ``hir.Slice`` and ``interp``).  Every bound is optional
                    # - a bound the source left out is a null constant - and it is
                    # the consumer that gives a missing one its meaning: a slice
                    # of a pointer takes a missing lower bound as 0 and requires
                    # an upper one, and a step is checked where the slice is used
                    lower = (
                        hir.Const(None)
                        if node.slice.lower is None
                        else self._as_value(self._gen_expr(node.slice.lower)[0])
                    )
                    upper = (
                        hir.Const(None)
                        if node.slice.upper is None
                        else self._as_value(self._gen_expr(node.slice.upper)[0])
                    )
                    step = (
                        hir.Const(None)
                        if node.slice.step is None
                        else self._as_value(self._gen_expr(node.slice.step)[0])
                    )
                    slice_obj = self.add(hir.Slice(lower, upper, step))
                    sub = self.add(hir.Subscript(base.value, ArgEntry(slice_obj, False)))
                    return ArgEntry(sub, True), False
                index = self._gen_expr(node.slice)[0]
                sub = self.add(hir.Subscript(base.value, index))
                if isinstance(base.value, hir.ConstRef):
                    # a subscript of a *global* names a type: a specialization of
                    # a struct template (``Foo[i32]``, which constructs a struct
                    # when it is called) or a ``syntax`` type marker such as
                    # ``Array[i32, 3]``
                    return ArgEntry(sub, False), base_is_struct
                # the i-th element of the array the base is: a *place*, written
                # and read through like a field of a struct
                return ArgEntry(sub, True), False
            case _:
                if not allow_retloc:
                    raise CompileError(f"unexpected expression {node}")
                loc = self.add(hir.Alloca(hir.InlineMode.NON_AGGREGATE))
                self._gen_result_loc(node, loc, False)
                self.add(hir.CommitSlot(loc))
                return ArgEntry(loc, True), False

    def _gen_is_none(self, node: ast.Compare, negated: bool) -> tuple[ArgEntry[hir.Value], bool]:
        """``expr is None`` / ``expr is not None``: whether the option ``expr`` is
        absent (the ``is not None`` negates the test).  The operand has to be an
        ``Option[T]`` - ``hir.IsNull`` rejects anything else.

        ``(name := expr) is not None`` is the *unwrap* form: it is handled as a
        whole (see ``_gen_unwrap``), binding ``name`` to the option's *payload*
        pointer rather than to the option pointer."""
        left, right = node.left, node.comparators[0]
        if not _is_none_literal(left) and not _is_none_literal(right):
            raise CompileError(
                "``is``/``is not`` can only be compared against None"
            )
        if (
            negated
            and isinstance(left, ast.NamedExpr)
            and _is_none_literal(right)
        ):
            return self._gen_unwrap(left)
        operand_node = right if _is_none_literal(left) else left
        is_null = self.add(hir.IsNull(self._gen_expr(operand_node)[0]))
        if negated:
            return ArgEntry(self.add(hir.Not(is_null)), False), False
        return ArgEntry(is_null, False), False

    def _gen_walrus(self, node: ast.NamedExpr) -> ArgEntry[hir.Value]:
        """The general ``(name := expr)``: ``name`` is bound to the *option*
        pointer - an alias of the place ``expr`` denotes, of type ``Option[T]`` -
        and the expression itself is that reference.  The ``(name := expr) is not
        None`` form binds the payload pointer instead (see ``_gen_unwrap``)."""
        if not isinstance(node.target, ast.Name):
            raise CompileError('the target of ``:=`` has to be a name')
        opt_ptr = self._as_ref(self._gen_expr(node.value)[0])
        self._declare_walrus(node.target.id, opt_ptr)
        return ArgEntry(opt_ptr, True)

    def _gen_unwrap(self, node: ast.NamedExpr) -> tuple[ArgEntry[hir.Value], bool]:
        """The ``(name := expr) is not None`` form: ``name`` is bound to the
        option's *payload* pointer - writing through it writes the payload - and
        the expression itself is the ``is not None`` test."""
        if not isinstance(node.target, ast.Name):
            raise CompileError('the target of ``:=`` has to be a name')
        opt_ptr = self._as_ref(self._gen_expr(node.value)[0])
        payload = self.add(hir.OptionPayloadPtr(opt_ptr))
        self._declare_walrus(node.target.id, payload)
        is_null = self.add(hir.IsNull(ArgEntry(opt_ptr, True)))
        return ArgEntry(self.add(hir.Not(is_null)), False), False

    def _declare_walrus(self, name: str, value: hir.Value) -> None:
        """Bind the name a ``:=`` declares in the current scope.  A duplicate is
        rejected - the target of a walrus introduces a new variable, exactly like
        an annotated declaration (see ``_gen_ann_assign``)."""
        if self._lookup(name) is not None:
            raise CompileError(
                f"'{name}' is already bound in spy function {self._fn_ir.name}; "
                f"``:=`` introduces a new variable"
            )
        self._declare(name, value)

    def _gen_isinstance(self, node: ast.Call, result_loc: hir.Value) -> None:
        """``isinstance(value, T)`` against a tagged union: the boolean the test
        yields, and - in the ``isinstance(e := value, T)`` form - the unwrap that
        binds ``e`` to the payload of the variant ``T`` (like the option's
        ``(e := opt) is not None``).  The interpreter checks that ``value``
        really is a tagged union and that ``T`` is one of its variants."""
        if len(node.args) != 2 or len(node.keywords) > 0:
            raise CompileError('isinstance takes exactly two positional arguments')
        value_node = node.args[0]
        type_value = self._as_value(self._gen_expr(node.args[1])[0])
        place: hir.Value
        if isinstance(value_node, ast.NamedExpr):
            if not isinstance(value_node.target, ast.Name):
                raise CompileError('the target of ``:=`` has to be a name')
            place = self._as_ref(self._gen_expr(value_node.value)[0])
            payload = self.add(hir.TaggedUnionPayloadPtr(place, type_value))
            self._declare_walrus(value_node.target.id, payload)
        else:
            place = self._as_ref(self._gen_expr(value_node)[0])
        test = self.add(hir.IsInstance(ArgEntry(place, True), type_value))
        self.add(hir.Store(result_loc, test))

    def _gen_match(self, node: ast.Match) -> None:
        """``match union: case T1() as e: ... case T2(): ...`` over a tagged
        union: each ``case Ti()`` is the tag test ``isinstance(union, Ti)`` and
        ``case Ti() as e`` also binds the body's ``e`` to the payload of ``Ti``;
        ``case _:`` is the fallback, which has to come last.  The subject is
        tested but binds nothing, so the name it names keeps naming the union
        inside the cases.  The bound name is only visible in the case that binds
        it - it is not declared outside the cases."""
        place = self._as_ref(self._gen_expr(node.subject)[0])
        if_count = 0
        wildcard_seen = False
        for case in node.cases:
            if case.guard is not None:
                raise CompileError('a ``match`` case guard is not supported yet')
            type_node, bind_name = self._match_case_pattern(case)
            if type_node is None:
                # the wildcard fallback: everything after it would be dead
                wildcard_seen = True
                self._scopes.append(_Scope())
                self._gen_block(case.body)
                self._scopes.pop()
                continue
            if wildcard_seen:
                raise CompileError('the wildcard ``case _:`` must come last')
            type_value = self._as_value(self._gen_expr(type_node)[0])
            test = self.add(hir.IsInstance(ArgEntry(place, True), type_value))
            self.add(hir.If(self.add(hir.AsBool(ArgEntry(test, False)))))
            self._scopes.append(_Scope())
            if bind_name is not None:
                payload = self.add(hir.TaggedUnionPayloadPtr(place, type_value))
                self._declare(bind_name, payload)
            self._gen_block(case.body)
            self._scopes.pop()
            self.add(hir.Else())
            if_count += 1
        for _ in range(if_count):
            self.add(hir.End())

    def _match_case_pattern(self, case: ast.match_case) -> tuple[ast.expr | None, str | None]:
        """The variant type a ``match`` case names and the name its payload is
        bound to (``case T() as e:``); ``(None, None)`` for the wildcard
        ``case _:``."""
        pattern = case.pattern
        bind_name: str | None = None
        if isinstance(pattern, ast.MatchAs):
            if pattern.name is None:
                # the wildcard ``case _:``
                if pattern.pattern is not None:
                    raise CompileError(f'unsupported ``match`` pattern {ast.unparse(case.pattern)!r}')
                return None, None
            bind_name = pattern.name
            pattern = pattern.pattern
        if isinstance(pattern, ast.MatchClass):
            if len(pattern.patterns) > 0 or len(pattern.kwd_attrs) > 0 or len(pattern.kwd_patterns) > 0:
                raise CompileError('a ``match`` case takes no arguments')
            return pattern.cls, bind_name
        raise CompileError(f'unsupported ``match`` pattern {ast.unparse(case.pattern)!r}')

    def _gen_syntax_call(self, callee: Any, args: list[ast.expr]) -> tuple[ArgEntry[hir.Value], bool]:
        if callee is syntax.ref:
            if len(args) != 1:
                raise CompileError('ref takes exactly one argument')
            return ArgEntry(self._as_ref(self._gen_expr(args[0])[0]), False), False

        if callee is syntax.ptr_cast:
            # ``ptr_cast(ptr, T)``: the pointer value ``ptr`` reinterpreted as the
            # pointer type ``T`` names (see ``hir.PtrCast`` and ``interp``)
            if len(args) != 2:
                raise CompileError('ptr_cast takes exactly two arguments')
            value = self._as_value(self._gen_expr(args[0])[0])
            target = self._as_value(self._gen_expr(args[1])[0])
            return ArgEntry(self.add(hir.PtrCast(value, target)), False), False

        if callee is syntax.as_func_ptr:
            # ``as_func_ptr(T, f)``: the runtime pointer to the spy function ``f``
            # of the function type ``T`` (see ``hir.AsFuncPtr`` and ``interp``)
            if len(args) != 2:
                raise CompileError('as_func_ptr takes exactly two arguments')
            type = self._as_value(self._gen_expr(args[0])[0])
            obj = self._as_value(self._gen_expr(args[1])[0])
            return ArgEntry(self.add(hir.AsFuncPtr(type, obj)), False), False

        if callee is syntax.typeof:
            # ``typeof(expr)``: a *type probe* (see ``hir.TypeOfBegin``).  The
            # argument's instructions follow, but the interpreter runs them into
            # a detached block - they are only typed (compiling whatever
            # functions they name) and never reach the MIR - so the probe yields
            # ``expr``'s static type with no runtime effect.
            if len(args) != 1:
                raise CompileError('typeof takes exactly one argument')
            self.add(hir.TypeOfBegin())
            entry = self._gen_expr(args[0])[0]
            end = self.add(hir.TypeOfEnd(entry.value, entry.is_ref))
            return ArgEntry(end, False), False

        if callee is syntax.unroll:
            # it is a *statement* marker placed before a loop (see
            # ``_gen_stmt``), not a value
            raise CompileError(
                'syntax.unroll() is a statement marker and must be followed by a loop'
            )

        if callee is syntax.comptime:
            # it is a *statement* marker placed before a variable declaration
            # (see ``_gen_stmt``), not a value
            raise CompileError(
                'syntax.comptime() is a statement marker and must be followed by '
                'a variable declaration'
            )

        raise CompileError(f'unsupported syntax call {callee}')

    def _gen_type_ctor(self, marker: Any, slice_node: ast.expr) -> hir.Inst | None:
        """The type value a ``syntax`` type marker used as an expression builds
        (``Ptr[T]``, ``Array[T, N]``, ``Option[T]``, ...), or None when
        ``marker`` names no such type.  The result is a compile-time type value
        the interpreter builds from the executing frame's type parameters (see
        (``hir.PointerType``/``hir.ArrayType``/``hir.OptionType``)."""
        for cls, is_const, is_multi in _POINTER_MARKERS:
            if marker is cls:
                elem_node = self._type_args(slice_node, marker, 1)[0]
                elem = self._as_value(self._gen_expr(elem_node)[0])
                return self.add(hir.PointerType(elem, is_const, is_multi))
        if marker is syntax.Array:
            elem_node, length_node = self._type_args(slice_node, marker, 2)
            elem = self._as_value(self._gen_expr(elem_node)[0])
            if isinstance(length_node, ast.Constant) and length_node.value is None:
                # ``Array[T, None]``: an array of unknown length (a DST)
                return self.add(hir.ArrayType(elem, None))
            length = self._as_value(self._gen_expr(length_node)[0])
            return self.add(hir.ArrayType(elem, length))
        if marker is syntax.Option:
            child_node = self._type_args(slice_node, marker, 1)[0]
            child = self._as_value(self._gen_expr(child_node)[0])
            return self.add(hir.OptionType(child))
        return None

    def _type_args(self, slice_node: ast.expr, marker: Any, count: int) -> list[ast.expr]:
        """The arguments of a subscripted ``syntax`` type marker: the elements of
        a tuple subscript, or the single subscript itself."""
        args = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
        if len(args) != count:
            name = getattr(marker, '__name__', marker)
            raise CompileError(f'{name} takes exactly {count} type argument(s)')
        return args

    def _gen_literal(self, slice_node: ast.expr) -> hir.Value:
        """The compile-time value a ``Literal[X]`` subscript denotes: ``X``
        itself, the value a ``typing.Literal`` annotation stands for (see
        ``sval.as_value``).  ``X`` has to be a single literal constant - a
        ``bool``, an ``int`` or a ``bytes`` - since that is what a value used as
        a type argument (the length of an ``Array``) can be."""
        if isinstance(slice_node, ast.Tuple):
            raise CompileError('Literal[...] takes exactly one value')
        value = self._as_value(self._gen_expr(slice_node)[0])
        if not (isinstance(value, hir.Const) and isinstance(value.value, (bool, int, bytes))):
            raise CompileError(
                'Literal[...] must name a single compile-time bool, integer or '
                'byte string'
            )
        return value

    # -- struct values ---------------------------------------------------------

    def _gen_result_loc(self, node: ast.expr, result_loc: hir.Value, allow_fall_back: bool = True) -> None:
        """Evaluate ``node`` writing its result into ``result_loc``
        (result-location semantics); no value register is produced."""
        fn_name = self._fn_ir.name
        match node:
            case ast.Call() if not self._is_syntax_call(node.func):
                self._gen_call(node, result_loc)
            case ast.UnaryOp() if not isinstance(node.op, ast.Not):
                op = _UNARY_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported unary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                self.add(hir.Unary(op, self._gen_expr(node.operand)[0], result_loc))
            case ast.BinOp():
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported binary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_expr(node.left)[0]
                rhs = self._gen_expr(node.right)[0]
                self.add(hir.Binary(op, lhs, rhs, result_loc))
            case ast.BoolOp():
                self._gen_boolop(node, result_loc)
            case ast.Tuple():
                self.add(hir.InitTuple(result_loc, len(node.elts)))
                for i, elt in enumerate(node.elts):
                    self._gen_result_loc(elt, self.add(hir.TuplePtrElement(result_loc, i)))
            case ast.IfExp():
                # each arm opens a scope of its own: a ``:=`` in an arm is not
                # visible outside it (the other arm may have run instead)
                cond = self.add(hir.AsBool(self._gen_expr(node.test)[0]))
                self.add(hir.If(cond))
                self._scopes.append(_Scope())
                self._gen_result_loc(node.body, result_loc)
                self._scopes.pop()
                self.add(hir.Else())
                self._scopes.append(_Scope())
                self._gen_result_loc(node.orelse, result_loc)
                self._scopes.pop()
                self.add(hir.End())
            case _:
                # every other expression computes its value first and
                # stores it into the result location; only a call, a unary
                # ``-``, a binary operation, an if-expression and a boolean
                # operator (the cases above) write through the location
                # without materializing a value
                if not allow_fall_back:
                    raise CompileError(f"unsupported expression {node}")
                value = self._as_value(self._gen_expr(node)[0])
                self.add(hir.Store(result_loc, value))

    def _gen_boolop(self, node: ast.BoolOp, result_loc: hir.Value) -> None:
        """Lower ``a and b``/``a or b`` (any length of chain) into a
        :class:`hir.Block` whose operands short-circuit by ``break_if``s,
        writing the result into ``result_loc``.  With the operands ``a1, a2,
        ..., aN``:

        .. code-block:: text

            block
                <a1>; <result_loc> = a1
                break_if <not a1>           # ``or`` breaks on ``a1`` instead
                <a2>; <result_loc> = a2
                break_if <not a2>
                ...
                <aN> -> <result_loc>        # the last operand, no copy (RLS)
            end

        Every operand but the last is stored into ``result_loc`` before the
        break that tests it, so an operand that short-circuits is the result;
        the last operand is generated with result-location semantics (its value
        is built straight into ``result_loc``).  A compile-time operand's
        ``break_if`` folds: a short-circuiting one leaves the block, so the rest
        of the chain (and the block) is dead.

        Every operand opens a scope of its own: an operand is only evaluated
        when the earlier ones did not short-circuit, so a ``:=`` in it is not
        guaranteed to have run outside of it."""
        op = _BOOL_OPS.get(type(node.op))
        if op is None:
            raise CompileError(f"unsupported boolean operator {type(node.op).__name__}")
        values = node.values
        self.add(hir.Block())
        for value in values[:-1]:
            self._scopes.append(_Scope())
            operand = self._gen_expr(value)[0]
            self._scopes.pop()
            self.add(hir.Store(result_loc, self._as_value(operand)))
            cond = self.add(hir.AsBool(operand))
            if op == 'and':
                # ``and`` short-circuits on a *false* operand: break on it
                cond = self.add(hir.Not(cond))
            self.add(hir.BreakIf(cond, 1))
        # the last operand is the result (no copy)
        self._scopes.append(_Scope())
        self._gen_result_loc(values[-1], result_loc)
        self._scopes.pop()
        self.add(hir.End())

    def _gen_arglist(self, args: list[ast.expr], keywords: list[ast.keyword]) -> hir.CallArgs:
        positional = tuple(self._gen_expr(a)[0] for a in args)
        kwargs: list[tuple[str, ArgEntry[hir.Value]]] = []
        for kw in keywords:
            if kw.arg is not None:
                kwargs.append((kw.arg, self._gen_expr(kw.value)[0]))
        return hir.CallArgs(positional, tuple(kwargs))

    def _gen_struct_ctor(self, struct: hir.Value, args: list[ast.expr], keywords: list[ast.keyword], result_loc: hir.Value) -> None:
        """One construction ``Foo(a1, a2, k=v)``: every argument is generated
        with result-location semantics straight into the address of the field
        it initializes - a nested construction fills the field in place, with
        no copy - and ``hir.FinishStruct`` closes the construction, resolving
        the struct type, binding the field addresses and filling the fields
        that were left out with their defaults."""
        indices: list[hir.Value] = []
        for i, arg in enumerate(args):
            field = self.add(hir.FieldIndexAddr(result_loc, i, is_aggregate_init=True))
            self._gen_result_loc(arg, field)
            indices.append(field)
        names: dict[str, hir.Value] = {}
        for kw in keywords:
            if kw.arg is None:
                raise CompileError(
                    f"**kwargs are not supported in spy function {self._fn_ir.name}"
                )
            field = self.add(hir.FieldAddr(result_loc, kw.arg, is_aggregate_init=True))
            self._gen_result_loc(kw.value, field)
            names[kw.arg] = field
        self.add(hir.FinishStruct(struct, result_loc, tuple(indices), frozendict(names)))

    def _gen_call(self, node: ast.Call, result_loc: hir.Value) -> None:
        """One call whose result is written into ``result_loc``: a
        construction ``Foo(...)`` or ``array(...)``, a method call ``x.h(...)``
        on a runtime struct value, or an ordinary call (a spy function, an
        inlined plain function or a spy builtin)."""
        fn_global = self._try_resolve_object(node.func)
        if fn_global is not None:
            if fn_global is cast:
                # ``cast(T, value)`` is a no-op: the type is only there for the
                # Python type checker (``std.arr_slice`` names its result type this
                # way), and the first argument is never evaluated
                if len(node.args) != 2 or len(node.keywords) > 0:
                    raise CompileError('cast takes exactly two positional arguments')
                self._gen_result_loc(node.args[1], result_loc)
                return
            if fn_global is syntax.array:
                # an array construction: like a struct one, the elements are
                # generated straight into the array's storage
                self._gen_array_ctor(node.args, node.keywords, result_loc)
                return
            if fn_global is isinstance:
                # ``isinstance(value, T)`` against a tagged union (see
                # ``_gen_isinstance``)
                self._gen_isinstance(node, result_loc)
                return
            if fn_global is ord:
                # ``ord(x)``: the encoding of the byte the compile-time byte
                # string ``x`` holds (see ``hir.Ord``)
                if len(node.args) != 1 or len(node.keywords) > 0:
                    raise CompileError('ord takes exactly one argument')
                operand = self._as_value(self._gen_expr(node.args[0])[0])
                self.add(hir.Ord(operand, result_loc))
                return
            if fn_global is len:
                # ``len(x)``: the number of elements of a tuple, or a struct's
                # own ``__len__`` (see ``hir.Len``)
                if len(node.args) != 1 or len(node.keywords) > 0:
                    raise CompileError('len takes exactly one argument')
                operand = self._gen_expr(node.args[0])[0]
                self.add(hir.Store(result_loc, self.add(hir.Len(operand))))
                return
        if isinstance(node.func, ast.Attribute):
            # a method of the struct ``base``: the method and its self
            # parameter are resolved by the interpreter from the static
            # type of the base; only the base's address is carried here
            base = self._as_ref(self._gen_expr(node.func.value)[0])
            self.add(hir.CallMethodInplace(base, node.func.attr, self._gen_arglist(node.args, node.keywords), result_loc))
            return
        callee, is_struct = self._gen_expr(node.func)
        if is_struct:
            # a construction: it writes the fields of the struct in place
            # instead of producing a value the call site would copy
            self._gen_struct_ctor(callee.value, node.args, node.keywords, result_loc)
            return
        # the callee must be addressable (a reference), the arguments are
        # by-value values
        self.add(hir.CallInplace(self._as_ref(callee), self._gen_arglist(node.args, node.keywords), result_loc))

    def _gen_array_ctor(self, args: list[ast.expr], keywords: list[ast.keyword], result_loc: hir.Value) -> None:
        """One construction ``array(a1, a2, ...)``: every element is generated
        with result-location semantics straight into the place of the element
        it initializes, and ``hir.FinishArray`` closes the construction, which
        is where the length of the array (the number of elements) and its
        element type (the common type of the elements) are resolved.

        The signature of ``syntax.array`` takes a ``length`` keyword because
        the Python type system cannot tell the length from the elements; the
        compiler takes it from the number of elements either way and ignores
        the keyword."""
        elements: list[hir.Value] = []
        for i, arg in enumerate(args):
            element = self.add(hir.FieldIndexAddr(result_loc, i, is_aggregate_init=True))
            self._gen_result_loc(arg, element)
            elements.append(element)
        for kw in keywords:
            if kw.arg != 'length':
                raise CompileError(
                    f"array takes no keyword argument {kw.arg!r} in spy function {self._fn_ir.name}"
                )
        self.add(hir.FinishArray(result_loc, tuple(elements)))

    # -- closures -------------------------------------------------------------

    def _gen_function_def(self, node: ast.FunctionDef) -> None:
        """Translate a nested ``def`` into a closure value.  The body's
        pre-scan already declared the name (a full compile-time slot), so this
        only writes the closure into that slot.  A ``@syntax.closure(...)``
        decorator stands in for the ``@func`` decorator a closure cannot carry
        (see ``_parse_closure_decorators``)."""
        if len(node.decorator_list) == 0:
            inline = True
            exceptions: ArraySet[Type] | None = ArraySet()
            callconv = 'default'
            may_panic = True
        else:
            inline, exceptions, callconv, may_panic = self._parse_closure_decorators(node)

        def gen_body() -> None:
            self._gen_body(node.body)
            self.add(hir.StoreVoidRetloc())

        closure = self._gen_closure(
            node.name, node.type_params, node.args, node.returns, gen_body,
            force_inline=inline, exceptions=exceptions,
            callconv=callconv, may_panic=may_panic,
        )
        slot = self._lookup(node.name)
        assert slot is not None, 'the pre-scan declares every def name'
        self.add(hir.Store(slot, closure))
        self.add(hir.CommitSlot(slot))

    def _gen_lambda(self, node: ast.Lambda) -> hir.MakeClosure:
        """A ``lambda`` is a closure that is forced to be inlined: its body is a
        single expression, returned from the closure's result location."""
        def gen_body() -> None:
            self._gen_result_loc(node.body, hir.ResultLoc())
            self.add(hir.Ret())

        return self._gen_closure(
            '<lambda>', (), node.args, None, gen_body,
            force_inline=True, exceptions=ArraySet(),
            callconv='default', may_panic=True,
        )

    def _parse_closure_decorators(self, node: ast.FunctionDef) -> tuple[bool, ArraySet[Type] | None, str, bool]:
        """The declaration a ``@syntax.closure(...)`` decorator gives: whether
        the closure is inlined, its exception set (as ``@func``), and its
        calling convention.  Any other decorator is rejected."""
        inline = True
        exceptions: ArraySet[Type] | None = ArraySet()
        callconv = 'default'
        may_panic = True
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else ast.Call(dec, [], [])
            if self._try_resolve_object(call.func) is not syntax.closure:
                raise CompileError(
                    f"only @syntax.closure(...) may decorate the nested def "
                    f"'{node.name}' in spy function {self._fn_ir.name}"
                )
            for kw in call.keywords:
                if kw.arg is None:
                    raise CompileError('@syntax.closure() takes no **kwargs')
                if kw.arg == 'inline':
                    inline = self._bool_constant(kw.value, 'inline')
                elif kw.arg == 'exceptions':
                    exceptions = self._closure_exception_set(kw.value)
                elif kw.arg == 'callconv':
                    callconv = self._str_constant(kw.value, 'callconv')
                elif kw.arg == 'may_panic':
                    may_panic = self._bool_constant(kw.value, 'may_panic')
                else:
                    raise CompileError(
                        f'@syntax.closure() got an unexpected keyword argument {kw.arg!r}'
                    )
        return inline, exceptions, callconv, may_panic

    def _bool_constant(self, node: ast.expr, what: str) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise CompileError(f'@syntax.closure({what}=...) needs a bool constant')

    def _str_constant(self, node: ast.expr, what: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise CompileError(f'@syntax.closure({what}=...) needs a string constant')

    def _closure_exception_set(self, node: ast.expr) -> ArraySet[Type] | None:
        """The exception set of a closure from its ``exceptions=`` declaration:
        ``None`` raises nothing, ``"infer"`` is inferred from the body, and a
        struct class (or a tuple of them, in error-code order) is the set of
        exceptions it may raise."""
        if isinstance(node, ast.Constant) and node.value is None:
            return ArraySet()
        if isinstance(node, ast.Constant) and node.value == 'infer':
            return None
        names = node.elts if isinstance(node, ast.Tuple) else [node]
        ret: ArraySet[Type] = ArraySet()
        for name in names:
            obj = self._try_resolve_object(name)
            if obj is None:
                raise CompileError(f'cannot resolve the exception {ast.unparse(name)!r}')
            value = as_value(obj, self._resolver, self._type_vars)
            if not isinstance(value, StructType):
                raise CompileError(f'cannot use {obj!r} as an exception: it must be a spy struct')
            ret.add(value)
        return ret

    def _gen_closure_vararg(
        self, arg: ast.arg | None, name: str,
    ) -> tuple[str | None, bool, hir.Value | None]:
        """Parse the ``*args``/``**kwargs`` parameter of a closure (``arg`` is
        None when it is not declared): its name, whether it carries the
        ``Comptime`` marker, and its *element/value* annotation as an operand
        evaluated in the enclosing frame (None when unannotated).  A
        ``type[...]`` annotation is rejected."""
        if arg is None:
            return None, False, None
        is_comptime = False
        annotation: hir.Value | None = None
        if arg.annotation is not None:
            is_comptime, type_node = self._split_comptime(arg.annotation)
            if type_node is not None:
                is_type_value, type_node = self._split_type_value(type_node)
                if is_type_value:
                    raise CompileError(
                        f'*args/**kwargs cannot be a type[...] parameter in spy '
                        f'closure {name}'
                    )
                annotation = self._as_value(self._gen_expr(type_node)[0])
        return arg.arg, is_comptime, annotation

    def _gen_closure(
        self,
        name: str,
        type_params: Any,
        args: ast.arguments,
        ret_node: ast.expr | None,
        body_gen: Callable[[], None],
        *,
        force_inline: bool,
        exceptions: ArraySet[Type] | None,
        callconv: str,
        may_panic: bool,
    ) -> hir.MakeClosure:
        """Parse one closure (a nested ``def`` or a ``lambda``) into a
        :class:`~spy.compiler.fn.ClosureFunction` and emit the ``hir.MakeClosure``
        that creates it.  The parameter annotations, defaults and return
        annotation are lowered into the *enclosing* stream (they are evaluated
        where the closure is created); the body is generated into an instruction
        list of its own, under a :class:`_ClosureScope` that records the captures
        it resolves through the enclosing scopes."""
        if len(args.posonlyargs) > 0:
            raise CompileError(f'positional-only arguments are not supported in spy closure {name}')
        if len(args.kwonlyargs) > 0:
            raise CompileError(f'keyword-only arguments are not supported in spy closure {name}')

        own: dict[str, Value] = {}
        generic_args: list[SpyTypeVar] = []
        for tp in type_params:
            tpname = getattr(tp, 'name', None)
            if tpname is None:
                raise CompileError(f'unsupported type parameter in spy closure {name}')
            var = SpyTypeVar(tpname)
            own[tpname] = var
            generic_args.append(var)

        saved_generics = self._generic_names
        self._generic_names = {**saved_generics, **own}
        try:
            param_names = [a.arg for a in args.args]
            n = len(param_names)
            annotations: list[hir.Value | None] = [None] * n
            is_comptime = [False] * n
            is_type_value = [False] * n
            for i, a in enumerate(args.args):
                if a.annotation is not None:
                    ct, type_node = self._split_comptime(a.annotation)
                    is_comptime[i] = ct
                    if type_node is not None:
                        tv, type_node = self._split_type_value(type_node)
                        is_type_value[i] = tv
                        annotations[i] = self._as_value(self._gen_expr(type_node)[0])
            vararg_name, vararg_is_comptime, vararg_annotation = self._gen_closure_vararg(
                args.vararg, name,
            )
            kwarg_name, kwarg_is_comptime, kwarg_annotation = self._gen_closure_vararg(
                args.kwarg, name,
            )
            defaults: list[hir.Value | None] = [None] * n
            offset = n - len(args.defaults)
            for i, default in enumerate(args.defaults):
                defaults[offset + i] = self._as_value(self._gen_expr(default)[0])
            if ret_node is None:
                ret_operand: hir.Value | None = None
            elif isinstance(ret_node, ast.Constant) and ret_node.value is None:
                # an explicit ``-> None`` declares a void function (an absent
                # annotation lets the return type be inferred)
                ret_operand = hir.Const(VoidType())
            else:
                ret_operand = self._as_value(self._gen_expr(ret_node)[0])

            self._closure_counter += 1
            closure = ClosureFunction(
                name=name,
                local_name=f'{name}#{self._closure_counter}',
                body=(), param_names=tuple(param_names), is_comptime=tuple(is_comptime),
                is_type_value=tuple(is_type_value),
                vararg_name=vararg_name, kwarg_name=kwarg_name,
                vararg_is_comptime=vararg_is_comptime, kwarg_is_comptime=kwarg_is_comptime,
                arg_is_ref=tuple([False] * n),
                generic_args=tuple(generic_args), force_inline=force_inline,
                exceptions=exceptions, callconv=callconv, may_panic=may_panic,
            )
            scope = _ClosureScope(closure)
            self._scopes.append(scope)
            try:
                for i, pname in enumerate(param_names):
                    scope.vars[pname] = hir.Arg(i)
                next_arg = n
                if vararg_name is not None:
                    scope.vars[vararg_name] = hir.Arg(next_arg)
                    next_arg += 1
                if kwarg_name is not None:
                    scope.vars[kwarg_name] = hir.Arg(next_arg)
                saved_insts = self.insts
                saved_own = self._own_generic_names
                saved_pragmas = self._pragmas
                self.insts = []
                self._pragmas = set()
                self._own_generic_names = set(own)
                try:
                    body_gen()
                    closure.body = tuple(self.insts)
                finally:
                    self.insts = saved_insts
                    self._pragmas = saved_pragmas
                    self._own_generic_names = saved_own
            finally:
                self._scopes.pop()
            return cast(hir.MakeClosure, self.add(hir.MakeClosure(
                closure, tuple(annotations), tuple(defaults), ret_operand,
                vararg_annotation, kwarg_annotation, tuple(scope.captures),
            )))
        finally:
            self._generic_names = saved_generics

    def _gen_name(self, name: str) -> hir.Value:
        """Always returns a reference to the name ``name``: the pointer its
        binding holds (a variable's slot, the option a ``:=`` bound, or the
        payload of one an unwrap bound)."""
        slot = self._lookup(name)
        if slot is not None:
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
    resolver: CompileContext,
    self_type: Type | None = None,
    self_arg: Literal["const ptr", "ptr", "value"] = "ptr",
    context_type_vars: dict[TypeVar, Value] | None = None,
    exceptions: tuple[Any, ...] | Literal["infer"] | None = None,
    callconv: str = 'default',
    may_panic: bool = True,
) -> FunctionIR:
    """Parse ``fn`` (a plain Python function) into a :class:`FunctionIR`.

    ``self_type`` is the struct a *method* belongs to: the first parameter is
    then typed from ``self_arg`` - ``"ptr"``/``"const ptr"`` make it a
    mutable/const pointer to the struct itself, passed by reference (its
    address), and ``"value"`` passes the object's value itself.

    ``context_type_vars`` are the type parameters of an enclosing context a
    method may name in its annotations and its body - the generic type
    parameters of the struct the method belongs to, which Python only makes
    visible inside the method's annotation scope.  They are keyed by the
    Python type parameter object the annotations evaluate to.

    ``exceptions`` is the ``@func(exceptions=...)`` declaration: ``None`` (the
    default) declares that the function raises nothing, ``"infer"`` that the
    exceptions are inferred from the body, and a tuple of spy struct classes the
    exceptions it may raise (in error-code order; the decorator normalizes a
    single type to a one-element tuple, see ``dsl._normalize_exceptions``).

    ``callconv`` is the ``@func(callconv=...)`` calling convention:
    ``'default'`` is the spy one; any other value names a C one, in which
    every argument is passed by value, the result is returned by value, and
    the function may not raise.  ``may_panic`` is carried through only.

    ``resolver`` is the host the annotations (and the declared exceptions) are
    resolved in: a struct class or function handle the annotation names is
    resolved to *that host's* object, so that a function parses against the
    structs and functions of the context it is compiled in (see
    ``sval.as_value``); it is required, since a function is always parsed *for*
    a context.

    ``mir_lower_cache`` is that host's MIR-mirror cache: which parameters are
    passed by reference is decided here, from the layout a mirror carries
    (see ``sval.pass_by_ref``).
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
    # function); an explicit ``-> Never`` becomes the spy ``EmptyType``
    # (a function that never returns a value, see ``sval.EmptyType``).  An
    # annotation that subscripts a struct template evaluates
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
            return as_value(annotation, resolver, type_vars)
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
        # a default value of ``None`` is the null value: the absent value of
        # an option (see ``sval.as_value``).  ``default_value`` being ``None``
        # means the parameter has no default, so it cannot be the value itself
        if value is None:
            return Null()
        return convert(value, 'a default value')

    def exception_set() -> ArraySet[Type] | None:
        """The declared exception set of the function: ``None`` when it is to
        be inferred, an empty set when the function raises nothing, and the
        spy struct types of the declared exceptions otherwise (in
        declaration order, which is the error-code order)."""
        if exceptions is None:
            return ArraySet()
        if exceptions == 'infer':
            return None
        ret: ArraySet[Type] = ArraySet()
        for exception in exceptions:
            value = as_value(exception, resolver, type_vars)
            if not isinstance(value, StructType):
                raise CompileError(
                    f'cannot use {exception!r} as an exception of function '
                    f'{node.name}: an exception must be a spy struct'
                )
            ret.add(value)
        return ret

    # the signature: the formal parameters, by declaration position
    all_args = list(node.args.args)
    offset = len(all_args) - len(defaults)
    positional = IndexedMap[str, SignatureFormalArg]()
    arg_is_ref: list[bool] = []
    for i, arg in enumerate(all_args):
        has_default = i >= offset
        default_value = default_of(defaults[i - offset]) if has_default else None
        # ``Comptime``/``Comptime[T]`` annotate a compile-time parameter (see
        # ``SignatureFormalArg``): the marker is split off the annotation and
        # the type it wraps is the parameter's declared type
        is_comptime, annotated = unwrap_comptime(annotations.get(arg.arg))
        # ``type[X]`` annotates a *type-valued* parameter: the argument is the
        # spy type the call passes and ``X`` (usually a type parameter of this
        # signature) is what it solves to (see ``SignatureFormalArg``)
        is_type_value = get_origin(annotated) is type
        if is_type_value:
            inner = get_args(annotated)
            if len(inner) != 1:
                raise CompileError(
                    f"type[...] of parameter '{arg.arg}' takes exactly one type argument"
                )
            annotated = inner[0]
        arg_type = annotation_of(annotated)
        if i == 0 and self_type is not None:
            # the ``self`` of a method: ``"ptr"``/``"const ptr"`` type it as a
            # mutable/const pointer to the struct itself (the HIR reads the
            # receiver through it, see ``FunctionIR.arg_is_ref`` and ``interp``),
            # ``"value"`` passes the object's value instead
            if self_arg == "value":
                arg_type = self_type
            else:
                arg_type = PointerType(self_type, is_const=self_arg == "const ptr")
        positional.add(
            arg.arg, SignatureFormalArg(arg_type, is_comptime, default_value, TriState.UNKNOWN, is_type_value)
        )
        # a method's ``self`` is bound directly to its argument (the receiver's
        # address); every other parameter is passed as the signature says
        arg_is_ref.append(i == 0 and self_type is not None and self_arg != "value")

    # ``*args``/``**kwargs``: the excess positional/keyword arguments a call
    # passes are bound to them (see ``Signature.bind_arg_pos``).  The formal
    # carries the *element* (varargs) or *value* (kwargs) annotation; an
    # unannotated one is inferred from the provided arguments (see
    # ``Signature.specialize``).  ``type[...]`` is rejected: a type-valued
    # ``*args``/``**kwargs`` element is not supported
    def vararg_formal(arg: ast.arg | None) -> SignatureFormalArg | None:
        if arg is None:
            return None
        is_comptime, annotated = unwrap_comptime(annotations.get(arg.arg))
        if get_origin(annotated) is type:
            raise CompileError(
                f"*args/**kwargs of function {node.name} cannot be a type[...] parameter"
            )
        return SignatureFormalArg(annotation_of(annotated), is_comptime, None, TriState.UNKNOWN, False)

    varargs = vararg_formal(node.args.vararg)
    kwargs = vararg_formal(node.args.kwarg)

    if callconv != 'default' and (varargs is not None or kwargs is not None):
        # a C-variadic function *declaration* is supported (see
        # ``dsl._build_fn_type``), but defining one - a body that reads its
        # arguments with ``va_arg`` - is not implemented yet
        raise CompileError(
            f'a non-default-callconv function {node.name} may not declare *args/**kwargs'
        )

    signature = Signature(
        tuple(generic_args), positional, varargs, kwargs, annotation_of(ret_annotation), exception_set(),
        callconv, may_panic,
    )

    ir = FunctionIR(node.name, signature, tuple(arg_is_ref), ())

    # a name that denotes a type parameter (the function's own, or one of the
    # struct a method belongs to) refers to the compile-time value the call
    # solved it to; the function's own parameters are added last, so they
    # shadow a struct's parameter of the same name, like Python scoping
    builder = _Builder(fn, ir, type_vars, resolver)
    # At HIR level, parameters are passed by ref (pointer); they are the
    # function body's block, the bottom scope of the builder's stack.  The
    # varargs/kwargs parameters come after every positional one, matching the
    # frame entries ``_init_args_from_signature`` builds
    for i, name in enumerate(positional.keys):
        builder._declare(name, hir.Arg(i))
    next_arg = len(positional.keys)
    if node.args.vararg is not None:
        builder._declare(node.args.vararg.arg, hir.Arg(next_arg))
        next_arg += 1
    if node.args.kwarg is not None:
        builder._declare(node.args.kwarg.arg, hir.Arg(next_arg))
    builder._gen_body(node.body)
    builder.add(hir.StoreVoidRetloc())
    ir.body = tuple(builder.insts)
    return ir

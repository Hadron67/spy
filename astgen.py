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
returns the instruction object whose register holds the value.

``astgen`` performs *almost* all name resolution.  Since every parameter
is addressable, the translated body starts with an
``Alloca``/``Store`` prologue per parameter (storing the by-value
``Arg(i)``), and a read of a parameter becomes a ``Load`` of its Alloca.
Global names - everything that is not a parameter - are resolved here to
their Python objects and embedded as ``hir.Const`` leaves; a name
captured from an enclosing Python scope (a spy function may be defined
inside a factory) is read from its closure cell and embedded the same
way.  Attributes on such compile-time objects (``spy.type``,
``spy.u64``, ...) are evaluated here as well.  Whether a function object
denotes a registered spy function - and which function value it stands
for - is decided by the interpreter when a call runs: a function body
may be parsed before its callees, or even itself (an aot function parses
its own body while it is being registered), are registered.

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
from typing import TypeVar

from . import hir
from .errors import CompileError, TypeMismatchError
from .fn import FunctionIR, ParamDef
from .type import Type, type_str, value_type

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


class _Builder:
    """Translates the AST of one function body into one linear
    instruction list of the untyped HIR.

    All builders of one function (including the sub-lists of ``if``
    branches) share the same ``env``: a static map from variable names
    to their parameter ``Alloca`` instructions, built by
    ``parse_function`` before translation.
    """

    def __init__(self, fn_ir: FunctionIR, env: dict[str, hir.Inst]) -> None:
        self._fn_ir = fn_ir
        self._env = env
        self.insts: list[hir.Inst] = []

    def add(self, inst: hir.Inst) -> hir.Inst:
        self.insts.append(inst)
        return inst

    # -- statements -----------------------------------------------------------

    def _gen_stmt(self, node: ast.stmt) -> None:
        fn_name = self._fn_ir.name
        match node:
            case ast.Return():
                if node.value is None:
                    raise CompileError(f"bare 'return' is not supported yet in spy function {fn_name}")
                self.add(hir.Ret(self._gen_expr(node.value)))
            case ast.Pass():
                pass
            case ast.Expr():
                self._gen_expr(node.value)
            case ast.Assign():
                raise CompileError(
                    f"local variables are not supported yet in spy function {fn_name}"
                )
            case ast.If():
                cond = self._gen_expr(node.test)
                then_body = self._gen_body(node.body)
                else_body = self._gen_body(node.orelse)
                self.add(hir.If(cond, tuple(then_body), tuple(else_body)))
            case _:
                raise CompileError(
                    f"unsupported statement {type(node).__name__} in spy function {fn_name}"
                )

    def _gen_body(self, stmts: list[ast.stmt]) -> list[hir.Inst]:
        """Translate a statement block into its own linear instruction list
        (the sub-builder shares the variable environment of the enclosing
        function, so parameter reads resolve to the same Allocas)."""
        sub = _Builder(self._fn_ir, self._env)
        for stmt in stmts:
            sub._gen_stmt(stmt)
        return sub.insts

    # -- expressions ----------------------------------------------------------

    def _resolve_global(self, name: str) -> hir.Const:
        fn = self._fn_ir.fn
        # a variable captured from an enclosing Python scope (a spy
        # function may be defined inside a factory, e.g. ``def make(k):
        # @cache.jit() def f(x): return x * k``): it behaves like a
        # global - the value of its closure cell at parse time is
        # embedded as a compile-time constant
        closure = fn.__closure__
        if closure is not None:
            for i, free_var in enumerate(fn.__code__.co_freevars):
                if free_var == name:
                    try:
                        value = closure[i].cell_contents
                    except ValueError:
                        if name == self._fn_ir.name:
                            # the function refers to its own name while
                            # it is being registered (an aot function
                            # decorated in an enclosing scope is parsed
                            # before the decorator has bound the name):
                            # the name then holds the raw function
                            # object, which the interpreter resolves to
                            # the function value when a call runs
                            return hir.Const(fn)
                        raise CompileError(
                            f"captured variable '{name}' is not bound yet in the "
                            f"scope of function {self._fn_ir.name}"
                        ) from None
                    return hir.Const(value)
        globals = fn.__globals__
        if name in globals:
            return hir.Const(globals[name])
        raise CompileError(
            f"name '{name}' is not defined in the scope of function {self._fn_ir.name}"
        )

    def _gen_expr(self, node: ast.expr) -> hir.Value:
        fn_name = self._fn_ir.name
        match node:
            case ast.Constant():
                if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                    return hir.Const(node.value)
                raise CompileError(f"unsupported constant {node.value!r} in spy function {fn_name}")
            case ast.Name():
                alloca = self._env.get(node.id)
                if alloca is not None:
                    # reading a parameter: load its slot
                    return self.add(hir.Load(alloca))
                return self._resolve_global(node.id)
            case ast.Attribute():
                base = self._gen_expr(node.value)
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
            case ast.Call():
                if len(node.keywords) > 0:
                    raise CompileError(
                        f"calls with keyword arguments inside spy functions are not supported yet "
                        f"(function {fn_name})"
                    )
                callee = self._gen_expr(node.func)
                args = tuple(self._gen_expr(a) for a in node.args)
                return self.add(hir.Call(callee, args))
            case ast.BinOp():
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported binary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                lhs = self._gen_expr(node.left)
                rhs = self._gen_expr(node.right)
                return self.add(hir.Binary(op, lhs, rhs))
            case ast.BoolOp():
                op = _BOOL_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(f"unsupported boolean operator in spy function {fn_name}")
                result: hir.Value = self._gen_expr(node.values[0])
                for v in node.values[1:]:
                    rhs = self._gen_expr(v)
                    result = self.add(hir.BoolOp(op, result, rhs))
                return result
            case ast.UnaryOp():
                if isinstance(node.op, ast.UAdd):
                    return self._gen_expr(node.operand)
                op = _UNARY_OPS.get(type(node.op))
                if op is None:
                    raise CompileError(
                        f"unsupported unary operator {type(node.op).__name__} in spy function {fn_name}"
                    )
                return self.add(hir.Unary(op, self._gen_expr(node.operand)))
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
                lhs = self._gen_expr(node.left)
                rhs = self._gen_expr(node.comparators[0])
                return self.add(hir.Compare(op, lhs, rhs))
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
    try:
        annotations = fn.__annotations__
        defaults = fn.__defaults__ if fn.__defaults__ is not None else ()
    except Exception as e:
        raise CompileError(
            f"cannot read the annotations of function {node.name}: {e}"
        ) from e

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

    ir = FunctionIR(fn, node.name, tuple(type_params), tuple(params), annotations.get('return'), ())

    # prologue: every parameter is addressable, so allocate one slot per
    # parameter and store its by-value argument into it.  ``env`` maps
    # each parameter name to its Alloca; the interpreter types an Alloca
    # when its first store runs.
    env: dict[str, hir.Inst] = {}
    prologue: list[hir.Inst] = []
    for i, param in enumerate(params):
        alloca = hir.Alloca()
        prologue.append(alloca)
        prologue.append(hir.Store(alloca, hir.Arg(i)))
        env[param.name] = alloca

    builder = _Builder(ir, env)
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

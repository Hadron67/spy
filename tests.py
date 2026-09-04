"""Integration tests for the spy JIT (``symlat.spy``).

The functions under test are defined inside :func:`make_samples`, which
registers them into a :class:`JitContext` the way a user would, with
the ordinary ``@cache.jit()``/``@cache.aot()`` decorators.  Bodies that
call other spy functions (``foo`` calling ``add_aot``/``add_inline``,
``is_even`` and ``is_odd`` calling each other, ...) resolve those names
through the enclosing factory scope, exactly like closures in user
code.  Every test builds a fresh context in ``setUp``, so no module
globals are touched.
"""

# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false
# pyright: reportReturnType=false
# The sample functions are DSL source compiled by spy, not ordinary
# Python that a type checker could understand (e.g. ``a + b`` with ``a,
# b: T``).

import io
from contextlib import redirect_stdout
from typing import Any
from unittest import TestCase

from . import (
    CompileError,
    JitContext,
    TypeMismatchError,
    astgen,
    compile_log,
    f64,
    hir,
    i32,
    u64,
)
from . import as_ as spy_as
from . import bool as spy_bool
from . import type as spy_type
from .fn import LazyJitFunction

# ---------------------------------------------------------------------------
# functions under test (the example of spy/instructions.md)
# ---------------------------------------------------------------------------


def make_samples(cache: JitContext) -> dict[str, object]:
    """Define and register the sample spy functions of the test suite
    into ``cache``, returning the wrappers by name.  ``add_inline`` is
    deliberately *not* registered: it stays the plain function that
    ``foo`` inlines."""

    @cache.jit()
    def add[T](a: T, b: T) -> T:
        return a + b

    @cache.aot()
    def add_aot(a: u64, b: u64) -> u64:
        return a + b

    @cache.jit()
    def add_default[T](a: T, b: T = 0) -> T:
        return a + b

    def add_inline[T](a: T, b: T) -> T:
        compile_log("add_inline was compiled")
        return a + b

    @cache.jit()
    def foo(a, b):
        if spy_type(a) == u64 and spy_type(b) == u64:
            return add_aot(a, b)
        else:
            return add_inline(a, b)

    @cache.jit()
    def is_i32(a):
        return spy_type(a) == i32

    @cache.jit()
    def call_add[T](a: T, b: T) -> T:
        # calling another jit function with fresh types triggers a nested
        # compilation during the compile of call_add
        return add(a, b)

    @cache.jit()
    def use_default[T](a: T) -> T:
        # call a spy function whose default parameter applies
        return add_default(a)

    # -- runtime ``if`` ----------------------------------------------------

    @cache.jit()
    def sign(n):
        # a runtime if whose branches both return
        if n > 0:
            return 1
        else:
            return -1

    @cache.jit()
    def clamped(n):
        # the then-branch returns; the else falls through to the code after
        # the if (the trailing return)
        if n > 100:
            return 100
        return n

    @cache.jit()
    def classify(n):
        # two consecutive partially-returning runtime ifs
        if n > 100:
            return 1
        if n > 10:
            return 2
        return 3

    @cache.jit()
    def nested_if(n):
        # a runtime if whose branches contain further runtime ifs
        if n > 0:
            if n > 100:
                return 2
            else:
                return 1
        else:
            return 0

    @cache.jit()
    def bad_join(n):
        # both branches fall through (a join): not supported yet
        if n > 0:
            pass
        else:
            pass
        return n

    # -- recursion ---------------------------------------------------------

    @cache.jit()
    def fact(n) -> i32:
        # direct recursion; the return annotation fixes the return type
        if n <= 1:
            return 1
        return n * fact(n - 1)

    @cache.aot()
    def fact_aot(n: i32) -> i32:
        if n <= 1:
            return 1
        return n * fact_aot(n - 1)

    @cache.jit()
    def gcd[T](a: T, b: T) -> T:
        # recursion whose return annotation is the type parameter ``T``
        if b == 0:
            return a
        return gcd(b, a % b)

    @cache.jit()
    def pow2[T](n: T) -> T:
        # a generic recursion that is specialized to several types
        if n <= 0:
            return 1
        return 2 * pow2(n - 1)

    @cache.jit()
    def fib(n) -> i32:
        # recursion inside the branches of a runtime if (and a recursive
        # call used twice on one path)
        if n <= 1:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    @cache.jit()
    def is_even(n) -> spy_bool:
        # mutual recursion with is_odd
        if n == 0:
            return True
        return is_odd(n - 1)

    @cache.jit()
    def is_odd(n) -> spy_bool:
        if n == 0:
            return False
        return is_even(n - 1)

    @cache.jit()
    def fact_untyped(n):
        # recursion needs the return type while the body is being typed
        if n <= 1:
            return 1
        return n * fact_untyped(n - 1)

    @cache.jit()
    def v_fn(a) -> i32:
        return a + 1

    @cache.jit()
    def abort_after_call(n):
        # compiles v_fn first (nested, MIR-cached by the module build),
        # then fails on its own unannotated recursion - the whole build
        # aborts
        v_fn(n)
        return abort_after_call(n - 1)

    return {
        'add': add,
        'add_aot': add_aot,
        'add_default': add_default,
        'foo': foo,
        'is_i32': is_i32,
        'call_add': call_add,
        'use_default': use_default,
        'sign': sign,
        'clamped': clamped,
        'classify': classify,
        'nested_if': nested_if,
        'bad_join': bad_join,
        'fact': fact,
        'fact_aot': fact_aot,
        'gcd': gcd,
        'pow2': pow2,
        'fib': fib,
        'is_even': is_even,
        'is_odd': is_odd,
        'fact_untyped': fact_untyped,
        'v_fn': v_fn,
        'abort_after_call': abort_after_call,
    }


def bad_aot(a: u64, b: u64) -> u64:
    # body computes a float; the return type annotation does not match;
    # only ever registered in the test that expects the failure
    return a + b + 1.5


def _make_twin():
    """A factory whose results all share one ``__name__``; registered
    into the same JitContext they must still get distinct native
    symbols (see ``test_same_name_functions_get_distinct_symbols``)."""

    def twin(a, b):
        return a + b

    return twin


# -- closure variables -------------------------------------------------------


def _make_scale(k):
    """Factory: the returned ``scale(x)`` computes ``x * k`` with the
    captured ``k`` acting as a compile-time constant of that function."""

    def scale(x):
        return x * k

    return scale


def _make_is_type(ty):
    """Factory: the returned function compares ``spy_type(x)`` with the
    captured type descriptor ``ty`` (a compile-time comparison)."""

    def is_ty(x):
        return spy_type(x) == ty

    return is_ty


def _make_call_pair(cache):
    """Factory: the returned spy function captures and calls the sibling
    spy function defined in the same factory."""

    @cache.jit()
    def add_pair(a, b):
        return a + b

    @cache.jit()
    def twice_pair(x, y):
        return add_pair(x, y) * 2

    return twice_pair


def _make_offset_aot(cache, k):
    """Factory: an ``aot`` function capturing a factory argument."""

    @cache.aot()
    def offset_aot(x: i32) -> i32:
        return x + k

    return offset_aot


threshold = 1000  # module global shadowed by the factory parameter below


def _make_cap(cache, threshold):
    """Factory: the captured ``threshold`` parameter shadows the module
    global of the same name."""

    @cache.jit()
    def cap(x):
        if x > threshold:
            return threshold
        return x

    return cap


class SpyExampleTest(TestCase):
    # the registered sample wrappers of this test's context (built by
    # setUp from make_samples)
    add: Any
    add_aot: Any
    add_default: Any
    foo: Any
    is_i32: Any
    call_add: Any
    use_default: Any
    sign: Any
    clamped: Any
    classify: Any
    nested_if: Any
    bad_join: Any
    fact: Any
    fact_aot: Any
    gcd: Any
    pow2: Any
    fib: Any
    is_even: Any
    is_odd: Any
    fact_untyped: Any
    v_fn: Any
    abort_after_call: Any

    def setUp(self) -> None:
        self.cache = JitContext()
        for name, value in make_samples(self.cache).items():
            setattr(self, name, value)

    def _spec_lines(self, wrapper, arg_types) -> list[str]:
        entry = wrapper._entry
        spec = entry.specs[arg_types]
        return spec.lines

    # -- the example of instructions.md --------------------------------------

    def test_jit_int(self) -> None:
        self.assertEqual(self.add(1, 2), 3)
        self.assertEqual(self.add(7, 8), 15)

    def test_jit_float(self) -> None:
        self.assertEqual(self.add(1.0, 2.0), 3.0)
        self.assertEqual(self.add(0.5, 0.25), 0.75)

    def test_jit_strings_fail_to_compile(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            self.add('', '')
        message = str(ctx.exception)
        self.assertIn("'+'", message)
        self.assertIn('strings are compiled as arrays of u8', message)
        # compilation failure is deterministic (and cached)
        with self.assertRaises(CompileError):
            self.add('x', 'y')

    def test_jit_conflicting_types(self) -> None:
        with self.assertRaises(TypeMismatchError) as ctx:
            self.add(1, 2.0)
        self.assertIn('conflicting types', str(ctx.exception))

    def test_aot(self) -> None:
        self.assertEqual(self.add_aot(1, 2), 3)
        # large unsigned values round-trip
        self.assertEqual(self.add_aot(2**63 - 1, 2), 2**63 + 1)

    def test_aot_type_mismatch(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self.add_aot(1.0, 2.0)
        self.assertIn('type mismatch', str(ctx.exception))

    def test_aot_out_of_range(self) -> None:
        with self.assertRaises(TypeMismatchError) as ctx:
            self.add_aot(-1, 2)
        self.assertIn('out of range', str(ctx.exception))

    def test_aot_as_mismatch(self) -> None:
        with self.assertRaises(TypeMismatchError):
            self.add_aot(spy_as(1, i32), spy_as(2, u64))

    def test_aot_return_type_mismatch_at_first_use(self) -> None:
        # the bad body type-checks when the function is first used: aot
        # functions are compiled lazily, like jit ones
        cache = JitContext()
        add = cache.aot()(bad_aot)
        with self.assertRaises(CompileError):
            add(1, 2)

    def test_defaults(self) -> None:
        self.assertEqual(self.add_default(1, 2), 3)
        self.assertEqual(self.add_default(1), 1)
        # keyword arguments may come in any order
        self.assertEqual(self.add_default(b=45, a=12), 57)
        self.assertEqual(self.add_default(a=3, b=4), 7)
        # default arguments are marshaled to the resolved type parameter
        self.assertEqual(self.add_default(spy_as(1, u64)), 1)

    def test_missing_argument(self) -> None:
        with self.assertRaises(TypeError):
            self.add_default()

    def test_unknown_kwarg(self) -> None:
        with self.assertRaises(TypeError):
            self.add_default(a=1, c=2)

    def test_foo_u64_branch_calls_add_aot(self) -> None:
        # the compile-time branch picks add_aot: no log, and the compiled
        # function contains a native call to the aot specialization
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = self.foo(spy_as(1, u64), spy_as(2, u64))
        self.assertEqual(result, 3)
        self.assertEqual(buf.getvalue(), '')
        lines = self._spec_lines(self.foo, (u64, u64))
        self.assertTrue(any('call' in line and 'spy.add_aot.u64.u64' in line for line in lines))

    def test_foo_else_branch_inlines(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(self.foo(1, 2), 3)
            # the specialization is cached: no second compile log
            self.assertEqual(self.foo(3, 4), 7)
        self.assertEqual(buf.getvalue(), 'add_inline was compiled\n')
        lines = self._spec_lines(self.foo, (i32, i32))
        self.assertFalse(any('spy.add_aot' in line for line in lines))

    def test_comptime_type_query(self) -> None:
        self.assertTrue(self.is_i32(1))
        self.assertFalse(self.is_i32(1.0))
        self.assertTrue(self.is_i32(spy_as(3, i32)))

    def test_nested_jit_compilation(self) -> None:
        # call_add compiles a fresh specialization of add on the fly
        self.assertEqual(self.call_add(2, 3), 5)
        # add's i32 specialization now exists and is reused by add itself
        self.assertEqual(self.add(10, 20), 30)

    def test_defaults_inside_function_body(self) -> None:
        self.assertEqual(self.use_default(41), 41)
        self.assertEqual(self.use_default(1.5), 1.5)

    def test_same_name_functions_get_distinct_symbols(self) -> None:
        # two different functions that share a ``__name__`` must not
        # collide in the context-wide native symbol table (calls between
        # spy functions are linked by symbol name)
        cache = JitContext()
        first = cache.jit()(_make_twin())
        second = cache.jit()(_make_twin())
        self.assertEqual(first(1, 2), 3)
        self.assertEqual(second(1, 2), 3)
        self.assertEqual(first(3, 4), 7)
        self.assertEqual(second(5, 6), 11)
        names = []
        for entry in (first._entry, second._entry):
            assert isinstance(entry, LazyJitFunction)
            for spec in entry.specs.values():
                names.append(spec.name)
        names.sort()
        self.assertEqual(len(names), 2, names)
        # the first registration keeps the plain name, the second one
        # gets a distinct one instead of overwriting it
        self.assertIn('spy.twin.i32.i32', names)
        self.assertEqual(len(set(names)), 2, names)

    # -- runtime ``if`` ------------------------------------------------------

    def test_runtime_if_both_branches_return(self) -> None:
        self.assertEqual(self.sign(5), 1)
        self.assertEqual(self.sign(-3), -1)

    def test_runtime_if_partial_return_falls_through(self) -> None:
        self.assertEqual(self.clamped(50), 50)
        self.assertEqual(self.clamped(500), 100)
        self.assertEqual(self.clamped(-1), -1)

    def test_runtime_if_sequence(self) -> None:
        self.assertEqual(self.classify(1), 3)
        self.assertEqual(self.classify(50), 2)
        self.assertEqual(self.classify(500), 1)

    def test_runtime_if_nested(self) -> None:
        self.assertEqual(self.nested_if(-1), 0)
        self.assertEqual(self.nested_if(5), 1)
        self.assertEqual(self.nested_if(500), 2)

    def test_runtime_if_join_error(self) -> None:
        # a runtime if whose branches both fall through needs a join,
        # which the structured MIR does not support yet
        with self.assertRaises(CompileError) as ctx:
            self.bad_join(1)
        self.assertIn('both fall through', str(ctx.exception))

    # -- recursion -----------------------------------------------------------

    def test_recursion_factorial(self) -> None:
        self.assertEqual(self.fact(1), 1)
        self.assertEqual(self.fact(5), 120)
        self.assertEqual(self.fact(10), 3628800)
        # the second call reuses the compiled specialization
        self.assertEqual(self.fact(10), 3628800)

    def test_recursion_is_a_native_recursive_call(self) -> None:
        self.assertEqual(self.fact(3), 6)
        lines = self._spec_lines(self.fact, (i32,))
        self.assertTrue(
            any('call' in line and 'spy.fact.i32' in line for line in lines), lines
        )

    def test_recursion_aot(self) -> None:
        self.assertEqual(self.fact_aot(5), 120)
        self.assertEqual(self.fact_aot(10), 3628800)

    def test_recursion_generic_over_types(self) -> None:
        # one generic recursion specialized to two types; each
        # specialization recurses to itself
        self.assertEqual(self.pow2(5), 32)
        self.assertEqual(self.pow2(3.0), 8.0)
        names = sorted(spec.name for spec in self.pow2._entry.specs.values())
        self.assertEqual(names, ['spy.pow2.f64', 'spy.pow2.i32'], names)

    def test_recursion_with_type_parameter_return(self) -> None:
        # the return annotation names the type parameter ``T``
        self.assertEqual(self.gcd(48, 36), 12)
        self.assertEqual(self.gcd(17, 5), 1)
        self.assertEqual(self.gcd(2**31 - 1, 3), 1)

    def test_recursion_inside_if_branches(self) -> None:
        # the recursive calls sit in the branch of a runtime if, and one
        # path calls the function twice
        self.assertEqual(self.fib(0), 0)
        self.assertEqual(self.fib(1), 1)
        self.assertEqual(self.fib(10), 55)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(self.is_even(10))
        self.assertFalse(self.is_even(7))
        self.assertTrue(self.is_odd(7))
        self.assertFalse(self.is_odd(10))

    def test_recursion_requires_return_annotation(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            self.fact_untyped(5)
        self.assertIn('requires a return type annotation', str(ctx.exception))
        # compilation failures are deterministic (and cached)
        with self.assertRaises(CompileError):
            self.fact_untyped(5)

    def test_aborted_build_reuses_cached_callees(self) -> None:
        # the failing build of abort_after_call compiled v_fn into its
        # module first (the MIR survives, the spec is never registered);
        # a later direct call of v_fn must still compile and run
        with self.assertRaises(CompileError):
            self.abort_after_call(3)
        self.assertEqual(self.v_fn(41), 42)
        self.assertEqual(self.v_fn(1), 2)

    # -- local variables and block scope --------------------------------------

    def test_local_variables(self) -> None:
        @self.cache.jit()
        def local(x, y):
            s = x + y
            return s
        self.assertEqual(local(1, 2), 3)
        self.assertEqual(local(0.5, 0.25), 0.75)

    def test_reassignment_in_the_same_block(self) -> None:
        @self.cache.jit()
        def reuse(x):
            y = x
            y = y * 2
            return y
        self.assertEqual(reuse(4), 8)
        self.assertEqual(reuse(1.5), 3.0)

    def test_augmented_assignment(self) -> None:
        @self.cache.jit()
        def aug(x):
            y = x
            y += 1
            y += 2
            return y
        self.assertEqual(aug(4), 7)
        self.assertEqual(aug(1.5), 4.5)

    def test_parameter_reassignment_at_function_scope(self) -> None:
        # the function body is the outermost block: its scope already
        # holds the parameter slots, so a top-level assignment to a
        # parameter name stores into the parameter slot
        @self.cache.jit()
        def bump(x):
            x = x + 1
            x += 1
            return x
        self.assertEqual(bump(4), 6)

    def test_block_scope_shadowing(self) -> None:
        # an ``if`` body is a block of its own: the first ``=`` on a
        # name inside it declares a block-local variable shadowing the
        # outer one (the outer binding is untouched after the block)
        @self.cache.jit()
        def pick(x):
            s = 1
            if spy_type(x) == i32:
                s = x
                s += 100
                return s
            return s
        self.assertEqual(pick(5), 105)
        # float specialization: the branch is not taken and the outer s
        # still holds its initial value
        self.assertEqual(pick(2.0), 1)

    def test_branch_declared_variable_is_visible_in_nested_blocks(self) -> None:
        @self.cache.jit()
        def nested(x):
            if x > 0:
                s = x
                if x > 10:
                    s2 = s + 5
                    return s2
                return s
            return 0
        self.assertEqual(nested(20), 25)
        self.assertEqual(nested(5), 5)
        self.assertEqual(nested(-1), 0)

    def test_local_variables_in_runtime_if_branches(self) -> None:
        @self.cache.jit()
        def rt(x):
            if x > 0:
                y = x * 2
                y += 1
                return y
            return 0
        self.assertEqual(rt(3), 7)
        self.assertEqual(rt(-1), 0)

    def test_self_referencing_declaration_is_unbound(self) -> None:
        # the first ``=`` declares the variable, so reading it in its own
        # initializer sees an uninitialized slot (like an unbound local)
        @self.cache.jit()
        def bad(x):
            y = y + 1
            return y
        with self.assertRaises(CompileError) as ctx:
            bad(1)
        self.assertIn('before any store', str(ctx.exception))

    def test_nested_block(self) -> None:
        @self.cache.jit()
        def foo(x):
            s = x
            if x > 0:
                s += 1
                return s
            return s
        self.assertEqual(foo(0), 0)
        self.assertEqual(foo(1), 2)

    def test_augassign_requires_a_prior_declaration(self) -> None:
        @self.cache.jit()
        def bad(x):
            y += 1
            return y
        with self.assertRaises(CompileError) as ctx:
            bad(1)
        self.assertIn("name 'y' is not defined in the scope", str(ctx.exception))

    def test_only_plus_equals_augmentation(self) -> None:
        @self.cache.jit()
        def bad(x):
            y = x
            y -= 1
            return y
        with self.assertRaises(CompileError) as ctx:
            bad(1)
        self.assertIn("only '+='", str(ctx.exception))

    def test_chained_assignment_rejected(self) -> None:
        @self.cache.jit()
        def bad(x):
            a = b = x
            return a + b
        with self.assertRaises(CompileError) as ctx:
            bad(1)
        self.assertIn('chained assignments', str(ctx.exception))

    # -- closure variables ---------------------------------------------------

    def test_closure_value_is_compile_time_constant(self) -> None:
        scale = self.cache.jit()(_make_scale(3))
        self.assertEqual(scale(2), 6)
        # a second specialization of the same function still sees k = 3
        self.assertEqual(scale(2.0), 6.0)
        self.assertEqual(scale(5), 15)
        # each factory call captures its own k
        other = self.cache.jit()(_make_scale(10))
        self.assertEqual(other(2), 20)
        self.assertEqual(scale(4), 12)

    def test_closure_captures_spy_type(self) -> None:
        # a captured type descriptor usable in a compile-time comparison
        is_u64 = self.cache.jit()(_make_is_type(u64))
        self.assertFalse(is_u64(1))
        self.assertTrue(is_u64(spy_as(1, u64)))
        is_f64 = self.cache.jit()(_make_is_type(f64))
        self.assertTrue(is_f64(1.0))
        self.assertFalse(is_f64(1))

    def test_closure_calls_captured_spy_function(self) -> None:
        # the body captures and calls the sibling spy function of the
        # same factory; both are compiled into one module
        twice = _make_call_pair(self.cache)
        self.assertEqual(twice(1, 2), 6)
        self.assertEqual(twice(3, 4), 14)

    def test_closure_aot(self) -> None:
        # an aot function captures the factory argument; it is compiled
        # lazily at its first call, inside the factory's scope
        off = _make_offset_aot(self.cache, 100)
        self.assertEqual(off(5), 105)
        self.assertEqual(off(-100), 0)

    def test_closure_aot_captures_later_assignment(self) -> None:
        # the aot body is only parsed at its first use, so a capture
        # that is assigned after the decoration is bound by then
        cache = JitContext()

        @cache.aot()
        def off(x: i32) -> i32:
            return x + late

        late = 100
        self.assertEqual(off(5), 105)
        self.assertEqual(off(-100), 0)

    def test_closure_shadows_module_global(self) -> None:
        cap = _make_cap(self.cache, 50)
        self.assertEqual(cap(5), 5)
        # 50 (the captured factory argument), not the module global 1000
        self.assertEqual(cap(5000), 50)


def _make_rls_pair():
    """Factory: the returned caller calls the sibling ``inner_add`` of the
    same factory scope (captured, so the callee resolves like a spy
    function callee would)."""

    def inner_add(a, b):
        return a + b

    def outer(x):
        return inner_add(x, 1)

    return outer


def _make_type_caller():
    """Factory: the returned caller uses the compile-time builtin
    ``spy.type`` (a module global of this test module) in a value
    context."""

    def outer(x):
        return spy_type(x) == i32

    return outer


class RlsHirTest(TestCase):
    """The HIR that ``astgen`` produces under result-location semantics
    (RLS): a call writes its result into the slot of a
    ``hir.CallInplace``, and the value context allocates the temporary
    slot up front and loads the value back right after the call."""

    def _single_call(self, fn) -> tuple[astgen.FunctionIR, hir.CallInplace]:
        ir = astgen.parse_function(fn)
        calls = [i for i in ir.body if isinstance(i, hir.CallInplace)]
        self.assertEqual(len(calls), 1)
        return ir, calls[0]

    def test_value_context_call_shape(self) -> None:
        outer = _make_rls_pair()
        ir, call = self._single_call(outer)
        # the callee is generated as a reference: the compile-time object
        # of the captured sibling function
        callee = call.callee
        assert isinstance(callee, hir.Const)
        self.assertEqual(callee.value.__name__, 'inner_add')
        # the arguments are by-value values: a load of the parameter slot
        # and the literal 1
        self.assertEqual(len(call.args), 2)
        self.assertIsInstance(call.args[0], hir.Load)
        arg = call.args[1]
        assert isinstance(arg, hir.Const)
        self.assertEqual(arg.value, 1)
        # the result location is a temporary slot; the instructions right
        # after the call load it back and return that value
        self.assertIsInstance(call.ret, hir.Alloca)
        idx = list(ir.body).index(call)
        load = ir.body[idx + 1]
        ret = ir.body[idx + 2]
        assert isinstance(load, hir.Load)
        assert isinstance(ret, hir.Ret)
        self.assertIs(load.ptr, call.ret)
        self.assertIs(ret.value, load)

    def test_comptime_call_through_result_slot(self) -> None:
        # a compile-time builtin (spy.type) goes through the same
        # temporary-slot + load shape: the interpreter records its
        # compile-time result in the slot and the matching Load passes it
        # on without giving the slot memory
        outer = _make_type_caller()
        ir, call = self._single_call(outer)
        callee = call.callee
        assert isinstance(callee, hir.Const)
        self.assertIs(callee.value, spy_type)
        self.assertEqual(len(call.args), 1)
        self.assertIsInstance(call.args[0], hir.Load)
        self.assertIsInstance(call.ret, hir.Alloca)
        idx = list(ir.body).index(call)
        load = ir.body[idx + 1]
        cmp = ir.body[idx + 2]
        assert isinstance(load, hir.Load)
        assert isinstance(cmp, hir.Compare)
        self.assertIs(load.ptr, call.ret)
        self.assertIs(cmp.lhs, load)


all_tests = [SpyExampleTest, RlsHirTest]

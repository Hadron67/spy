from typing import Protocol
from unittest import TestCase

from ..compiler import (
    CompileError,
    hir,
    i32,
    interp,
    sval,
    syntax,
    u0,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, func, func_type, struct
from ..compiler.fn import (
    ArgEntry,
    ArgList,
    ArgNode,
    ClosureValue,
    CompoundArgNode,
    ComptimePtrArg,
    FunctionIR,
    FunctionValue,
    IndexedMap,
    RuntimeArgNode,
    Signature,
    SignatureFormalArg,
    frozendict,
)
from ..compiler.interp import Analyser, InterpVal, RuntimeClosurePlan
from ..compiler.syntax import Option, as_func_ptr
from ..compiler.util import ArraySet
from ..std.core import as_runtime_closure

# ---------------------------------------------------------------------------
# closures: a nested ``def`` (or ``lambda``) inside a spy function captures
# the variables of the enclosing function.  A closure only exists at compile
# time - it can be called inside the function (inlined, or compiled into a
# runtime function whose captures are hidden pointer parameters) and passed to
# an inline function, but never handed to a runtime function.
# ---------------------------------------------------------------------------


@func()
def closure_capture_param(n: i32) -> i32:
    # the default closure is inlined; ``n`` is captured by reference
    def add(x: i32) -> i32:
        return x + n
    return add(3)


def apply_inline(f, x: i32) -> i32:
    # a plain Python function: inlined at its call site, so it may take a
    # closure as a parameter
    return f(x)


@func()
def closure_passed_to_inline(a: i32) -> i32:
    def add(x: i32) -> i32:
        return x + a
    return apply_inline(add, 10)


@func()
def closure_lambda_arg(a: i32) -> i32:
    # a lambda is a forced-inline closure and may be written directly as an
    # argument
    return apply_inline(lambda x: x + a, 10)


@func()
def closure_forward_declaration() -> i32:
    # the closure is written before the variable it captures is assigned: the
    # body's pre-scan declares ``x`` up front
    def f() -> i32:
        return x
    x: i32 = 7
    return f()


@func()
def closure_nested(a: i32) -> i32:
    def outer() -> i32:
        def inner() -> i32:
            return a + 1
        return inner() + 1
    return outer()


@struct()
class Box:
    v: i32


@func()
def closure_mutates_captured_field(a: i32) -> i32:
    # a captured struct is written back through the pointer the closure holds
    b = Box(a)
    def bump() -> None:
        b.v += 1
    bump()
    bump()
    return b.v


@func()
def closure_compiled(n: i32) -> i32:
    # a non-inlined closure is compiled into a runtime function that takes the
    # address of ``n`` as a hidden parameter
    @syntax.closure(inline=False)
    def add(x: i32) -> i32:
        return x + n
    return add(20)


@func()
def closure_compiled_calls(n: i32) -> i32:
    @syntax.closure(inline=False)
    def double(x: i32) -> i32:
        return twice(x)
    return double(n)


@func()
def twice(x: i32) -> i32:
    return x * 2


@func()
def closure_comptime() -> i32:
    # an inlined closure captures a compile-time variable directly
    n: syntax.Comptime = 5
    def f() -> i32:
        return n + 1
    return f()


@func()
def closure_compiled_comptime() -> i32:
    # a compile-time capture is used when the closure is compiled and does not
    # reach the runtime signature
    n: syntax.Comptime = 5
    @syntax.closure(inline=False)
    def f() -> i32:
        return n + 1
    return f()


@struct()
class ClosureError(Exception):
    code: i32


@func(exceptions=ClosureError)
def raise_closure_error(n: i32) -> i32:
    if n < 0:
        raise ClosureError(1)
    return n


@func()
def closure_compiled_raises(n: i32) -> i32:
    # a compiled closure declares the exceptions it may raise; the error it
    # propagates is dispatched to the caller's ``try``
    @syntax.closure(inline=False, exceptions=ClosureError)
    def f(x: i32) -> i32:
        return raise_closure_error(x)
    try:
        return f(n)
    except ClosureError:
        return -1


@func()
def plus_one(x: i32) -> i32:
    return x + 1

@func_type()
class PlusOneFn(Protocol):
    def __call__(self, x: i32) -> i32: ...

@func()
def closure_as_func_ptr() -> i32:
    # a capture-free, non-inlined closure is an ordinary runtime function
    @syntax.closure(inline=False)
    def inc(x: i32) -> i32:
        return x + 1
    p = as_func_ptr(PlusOneFn, inc)
    return p[...](5)


@func()
def closure_with_generic_params[T](x: T) -> T:
    # a closure may declare its own type parameters
    def identity(y: T) -> T:
        return y
    return identity(x)


@func()
def closure_in_a_loop(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        captured = i
        def add(x: i32) -> i32:
            return x + captured  # noqa: B023
        total += add(1)
    return total


@func()
def closure_naming(n: i32) -> i32:
    # a compiled closure's native name is prefixed with the function it is
    # created in (the enclosing specialization, its signature included)
    @syntax.closure(inline=False)
    def mid(x: i32) -> i32:
        @syntax.closure(inline=False)
        def inner(y: i32) -> i32:
            return y + 1
        return inner(x) + n
    return mid(2)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@func()
def closure_capturing_as_func_ptr(n: i32) -> i32:
    @syntax.closure(inline=False)
    def f(x: i32) -> i32:
        return x + n
    p = as_func_ptr(PlusOneFn, f)
    return p[...](1)


@func()
def closure_inlined_as_func_ptr() -> i32:
    def inc(x: i32) -> i32:
        return x + 1
    p = as_func_ptr(PlusOneFn, inc)
    return p[...](1)


@func()
def closure_to_runtime_fn(n: i32) -> i32:
    # a closure may not be handed to a non-inline (runtime) function
    def add(x: i32) -> i32:
        return x + n
    return twice(add)  # pyright: ignore


@func()
def closure_enclosing_type_param[T](x: T) -> T:
    def f() -> i32:
        if spy_typeof(x) == T:
            return 1
        return 0
    return x


@func()
def closure_rejected_decorator() -> i32:
    @func()
    def f() -> i32:
        return 1
    return f()


# ---------------------------------------------------------------------------
# runtime closures: ``std.core.as_runtime_closure`` turns a compile-time closure
# into an ordinary spy struct value whose fields hold its runtime captures and
# whose ``__call__`` rebuilds them and calls the closure, so the value can be
# stored, passed, returned and called like any other (see the design doc).
# ---------------------------------------------------------------------------

# the multi-return type of the closure below, behind a name: a closure's return
# annotation is generated in the enclosing frame, where a bare ``tuple[...]``
# subscript is not resolvable
type PairTwo = tuple[i32, i32]


@struct()
class RtBox:
    v: i32


@struct()
class RtTagA:
    x: i32


@struct()
class RtTagB:
    y: i32


@struct()
class RtHolder[T]:
    # a generic holder: the anonymous runtime-closure type can only reach a
    # struct field through a type parameter (it cannot be written in a source
    # annotation)
    f: T


@func()
def rtc_capture_param(n: i32) -> i32:
    # the value can be stored in a variable and called like a function
    def add(x: i32) -> i32:
        return x + n
    f = as_runtime_closure(add)
    return f(3)


@func()
def rtc_compiled(n: i32) -> i32:
    # a non-inline closure converts the same way (the ``__call__`` compiles the
    # closure into a runtime function)
    @syntax.closure(inline=False)
    def add(x: i32) -> i32:
        return x + n
    f = as_runtime_closure(add)
    return f(20)


@func()
def rtc_no_capture() -> i32:
    # no runtime capture: the struct is zero-sized
    def inc(x: i32) -> i32:
        return x + 1
    f = as_runtime_closure(inc)
    return f(5)


@func()
def rtc_write_back(start: i32) -> i32:
    # ``as_copy=False`` (the default): the field holds a pointer, so the closure
    # writes the captured variable back
    b = RtBox(start)
    def bump() -> None:
        b.v += 1
    f = as_runtime_closure(bump)
    f()
    f()
    return b.v


@func()
def rtc_copies(start: i32) -> i32:
    # ``as_copy=True``: the captured value is copied into the struct, so the
    # closure does not change the original variable
    b = RtBox(start)
    def bump() -> None:
        b.v += 1
    f = as_runtime_closure(bump, True)
    f()
    f()
    return b.v


@func()
def rtc_copy_keyword(start: i32) -> i32:
    # ``as_copy`` is accepted as a keyword too
    b = RtBox(start)
    def bump() -> None:
        b.v += 1
    f = as_runtime_closure(bump, as_copy=True)
    f()
    f()
    return b.v


@func()
def rtc_copies_survive(start: i32) -> i32:
    # the copy lives in the struct: a second call sees the first one's write
    b = RtBox(start)
    def bump() -> i32:
        b.v += 1
        return b.v
    f = as_runtime_closure(bump, as_copy=True)
    f()
    return f()


@func()
def rtc_comptime() -> i32:
    # a compile-time capture is baked into the generated ``__call__``
    n: syntax.Comptime = 5
    @syntax.closure(inline=False)
    def f() -> i32:
        return n + 1
    g = as_runtime_closure(f)
    return g()


@func()
def rtc_capture_aggregate(x: i32) -> i32:
    # a captured struct
    p = RtBox(x)
    def get() -> i32:
        return p.v + 1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_option(x: i32) -> i32:
    # a captured option
    o: Option[i32] = x
    def get() -> i32:
        if (v := o) is not None:
            return v
        return -1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_zst_option(n: i32) -> i32:
    # a captured compile-time option over a zero-sized child (``u0``): only the
    # tag has storage, the payload has none, so the capture contributes no field
    o: syntax.Comptime[Option[u0]] = 0
    def get() -> i32:
        if o is None:
            return n - 1
        return n + 1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_zst_option_absent(n: i32) -> i32:
    # the absent value of the same zero-sized option: only the tag differs
    o: syntax.Comptime[Option[u0]] = None
    def get() -> i32:
        if o is None:
            return n - 1
        return n + 1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_union(x: i32) -> i32:
    # a captured tagged union
    u: RtTagA | RtTagB = RtTagA(x)
    def get() -> i32:
        if isinstance(v := u, RtTagA):
            return v.x
        return -1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_comptime_union() -> i32:
    # a captured compile-time tagged union
    u: syntax.Comptime[RtTagA | RtTagB] = RtTagA(2)
    def get() -> i32:
        if isinstance(v := u, RtTagA):
            return v.x
        return -1
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_capture_tuple(n: i32) -> i32:
    # a captured tuple: its elements are their own fields
    t: syntax.Comptime = (n, n + 1)
    def get() -> i32:
        return t[0] + t[1]
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_two_captures(a: i32, b: i32) -> i32:
    # two runtime captures, one field each
    def add(x: i32) -> i32:
        return x + a + b
    f = as_runtime_closure(add)
    return f(1)


@func()
def rtc_generic[T](x: T) -> T:
    # the generated ``__call__`` mirrors the closure's type parameters
    def identity(y: T) -> T:
        return y
    f = as_runtime_closure(identity)
    return f(x)


@func()
def rtc_multi(n: i32) -> i32:
    # a closure returning several values
    def pair2(x: i32) -> PairTwo:
        return x, x + 1
    f = as_runtime_closure(pair2)
    a, b = f(n)
    return a * 100 + b


@struct()
class RtError(Exception):
    code: i32


@func(exceptions=RtError)
def raise_rtc_error(n: i32) -> i32:
    if n < 0:
        raise RtError(1)
    return n


@func()
def rtc_raises(n: i32) -> i32:
    # an exception the closure raises propagates through ``__call__``
    @syntax.closure(inline=False, exceptions=RtError)
    def f(x: i32) -> i32:
        return raise_rtc_error(x)
    g = as_runtime_closure(f)
    try:
        return g(n)
    except RtError:
        return -1


@func()
def rtc_varargs(n: i32) -> i32:
    # ``*args`` is forwarded through the ``__call__``
    def add(*args: i32) -> i32:
        i: syntax.Comptime = 0
        total: i32 = 0
        syntax.unroll()
        while i < len(args):
            total = total + args[i]
            i = i + 1
        return total
    f = as_runtime_closure(add)
    return f(1, 2, 3) + n


@func()
def rtc_kwargs(n: i32) -> i32:
    # ``*args``/``**kwargs`` are forwarded together
    def combine(*args: i32, **kwargs: i32) -> i32:
        return args[0] + kwargs['x']
    f = as_runtime_closure(combine)
    return f(1, x=2) + n


def call_a_runtime_closure(f, x: i32) -> i32:
    # a plain Python function passed a runtime closure value: it calls it
    return f(x)


@func()
def rtc_passed_inline(n: i32) -> i32:
    def add(x: i32) -> i32:
        return x + n
    f = as_runtime_closure(add)
    return call_a_runtime_closure(f, 7)


@func()
def rtc_in_struct_field(n: i32) -> i32:
    # the value is built straight into a struct field (no temporary struct) and
    # read back out to be called
    def add(x: i32) -> i32:
        return x + n
    h = RtHolder(as_runtime_closure(add))
    g = h.f
    return g(3)


@func()
def rtc_in_struct_field_copies(start: i32) -> i32:
    # ``as_copy=True``: the copy lives in the field, and a read-back call sees
    # the previous call's write
    b = RtBox(start)
    def bump() -> i32:
        b.v += 1
        return b.v
    h = RtHolder(as_runtime_closure(bump, True))
    g = h.f
    g()
    return g()


@func()
def rtc_kwargs_only(n: i32) -> i32:
    # only ``**kwargs`` (no ``*args``): its forwarding slot sits right after the
    # positional ones
    def combine(**kwargs: i32) -> i32:
        return kwargs['a'] + kwargs['b']
    f = as_runtime_closure(combine)
    return f(a=1, b=2) + n


@func()
def rtc_varargs_compiled(n: i32) -> i32:
    # a non-inline closure with ``*args``: the forwarding is the same
    @syntax.closure(inline=False)
    def add(*args: i32) -> i32:
        i: syntax.Comptime = 0
        total: i32 = 0
        syntax.unroll()
        while i < len(args):
            total = total + args[i]
            i = i + 1
        return total
    f = as_runtime_closure(add)
    return f(1, 2, 3) + n


@func()
def rtc_capture_scalar_local(n: i32) -> i32:
    # a runtime scalar local captured by reference: a write to the local after
    # the conversion is seen through the capture
    x: i32 = n
    def get() -> i32:
        return x
    f = as_runtime_closure(get)
    x = x + 1
    return f()


@func()
def rtc_capture_scalar_local_copies(n: i32) -> i32:
    # the same scalar captured by value: the copy is made when the closure is
    # converted, so the later write is not seen
    x: i32 = n
    def get() -> i32:
        return x
    f = as_runtime_closure(get, True)
    x = x + 1
    return f()


@func()
def rtc_single_leaf(n: i32) -> i32:
    # exactly one runtime capture: the struct's MIR mirror is its own field
    def add(x: i32) -> i32:
        return x + n
    f = as_runtime_closure(add)
    return f(3)


@func()
def rtc_round_trip(n: i32) -> i32:
    # one capture of every shape, so a test can check that building the struct
    # and rebuilding the capture round-trips the ``ArgNode``
    s: i32 = n + 100
    p = RtBox(n)
    o: Option[i32] = n
    u: RtTagA | RtTagB = RtTagA(n)
    t: syntax.Comptime = (n, n + 1)
    k: syntax.Comptime = 3
    def get() -> i32:
        total: i32 = s
        total = total + p.v
        if (v := o) is not None:
            total = total + v
        if isinstance(w := u, RtTagA):
            total = total + w.x
        total = total + t[0] + t[1] + k
        return total
    f = as_runtime_closure(get)
    return f()


@func()
def rtc_option_contrast(n: i32) -> i32:
    # a runtime option (one runtime pointer) next to a compile-time one (a
    # compound tag + payload), so a test can tell their captures apart
    o: Option[i32] = n
    c: syntax.Comptime[Option[u0]] = 0
    def get() -> i32:
        total: i32 = 0
        if (v := o) is not None:
            total = total + v
        if c is None:
            return -1
        return total
    f = as_runtime_closure(get)
    return f()


def _a_builtin_call_with_captures() -> FunctionValue:
    """A hand-built body whose call carries captures to a builtin - a call the
    generated ``__call__`` never makes (its callee is a closure)."""
    pos: IndexedMap[str, SignatureFormalArg] = IndexedMap()
    sig = Signature((), pos, None, None, sval.VoidType(), ArraySet(), 'default', True)
    body = (
        hir.CallInplace(
            callee=hir.Const(sval.BuiltinFn('as_runtime_closure')),
            args=hir.CallArgs((ArgEntry(hir.Const(0), False),), ()),
            ret=hir.ResultLoc(),
            captures=(hir.Const(0),),
        ),
        hir.Ret(),
    )
    return FunctionValue(
        'a_builtin_call_with_captures',
        FunctionIR('a_builtin_call_with_captures', sig, (), body),
    )


# ---------------------------------------------------------------------------
# runtime closures: errors
# ---------------------------------------------------------------------------


@func()
def rtc_as_copy_not_bool() -> i32:
    def f() -> i32:
        return 1
    g = as_runtime_closure(f, as_copy=1)  # pyright: ignore
    return g()


@func()
def rtc_not_a_closure() -> i32:
    return as_runtime_closure(1)  # pyright: ignore


@func()
def rtc_unexpected_keyword() -> i32:
    def f() -> i32:
        return 1
    g = as_runtime_closure(f, nope=True)  # pyright: ignore
    return g()


class SpyClosureTest(TestCase):
    """Nested ``def``/``lambda`` closures: capture by reference, inline and
    compiled forms, capture-free function pointers, and the rejected cases."""

    def test_capture_a_parameter(self) -> None:
        self.assertEqual(closure_capture_param(10), 13)

    def test_pass_a_closure_to_an_inline_function(self) -> None:
        self.assertEqual(closure_passed_to_inline(5), 15)

    def test_lambda_as_an_argument(self) -> None:
        self.assertEqual(closure_lambda_arg(5), 15)

    def test_forward_declaration(self) -> None:
        self.assertEqual(closure_forward_declaration(), 7)

    def test_nested_closures(self) -> None:
        self.assertEqual(closure_nested(10), 12)

    def test_mutate_a_captured_field(self) -> None:
        self.assertEqual(closure_mutates_captured_field(1), 3)

    def test_compiled_closure(self) -> None:
        self.assertEqual(closure_compiled(10), 30)
        # the specialization is shared; each creation passes its own captures
        self.assertEqual(closure_compiled(1), 21)

    def test_compiled_closure_calls_a_function(self) -> None:
        self.assertEqual(closure_compiled_calls(21), 42)

    def test_compiled_closure_with_a_comptime_capture(self) -> None:
        self.assertEqual(closure_compiled_comptime(), 6)

    def test_inlined_closure_with_a_comptime_capture(self) -> None:
        self.assertEqual(closure_comptime(), 6)

    def test_compiled_closure_declares_its_exceptions(self) -> None:
        self.assertEqual(closure_compiled_raises(5), 5)
        self.assertEqual(closure_compiled_raises(-5), -1)

    def test_capture_free_closure_as_a_function_pointer(self) -> None:
        self.assertEqual(closure_as_func_ptr(), 6)

    def test_a_closure_with_its_own_generic_parameters(self) -> None:
        self.assertEqual(closure_with_generic_params(4), 4)

    def test_a_closure_in_a_loop(self) -> None:
        self.assertEqual(closure_in_a_loop(3), 6)

    def test_a_compiled_closures_name_is_qualified(self) -> None:
        self.assertEqual(closure_naming(10), 13)
        names = list(_GLOBAL_CONTEXT._symbol_table._symbols.keys())
        nested = [n for n in names if 'mid' in n and 'inner' in n]
        self.assertTrue(
            any('closure_naming' in n and n.index('mid') < n.index('inner') for n in nested),
            names,
        )

    def test_capturing_closure_cannot_become_a_function_pointer(self) -> None:
        with self.assertRaises(CompileError):
            closure_capturing_as_func_ptr(1)

    def test_an_inlined_closure_cannot_become_a_function_pointer(self) -> None:
        with self.assertRaises(CompileError):
            closure_inlined_as_func_ptr()

    def test_an_enclosing_type_parameter_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            closure_enclosing_type_param(1)

    def test_a_closure_cannot_be_passed_to_a_runtime_function(self) -> None:
        with self.assertRaises(CompileError):
            closure_to_runtime_fn(1)

    def test_a_non_closure_decorator_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            closure_rejected_decorator()

    # -- runtime closures (``std.core.as_runtime_closure``) --------------------

    def test_runtime_closure_captures_a_parameter(self) -> None:
        self.assertEqual(rtc_capture_param(5), 8)

    def test_runtime_closure_of_a_compiled_closure(self) -> None:
        self.assertEqual(rtc_compiled(5), 25)

    def test_runtime_closure_without_captures(self) -> None:
        self.assertEqual(rtc_no_capture(), 6)

    def test_runtime_closure_writes_back_by_reference(self) -> None:
        self.assertEqual(rtc_write_back(5), 7)

    def test_runtime_closure_copies_by_value(self) -> None:
        # the original variable is unchanged...
        self.assertEqual(rtc_copies(5), 5)
        # ...whether ``as_copy`` is positional or by keyword
        self.assertEqual(rtc_copy_keyword(5), 5)

    def test_a_runtime_closure_copy_survives_between_calls(self) -> None:
        self.assertEqual(rtc_copies_survive(5), 7)

    def test_runtime_closure_with_a_comptime_capture(self) -> None:
        self.assertEqual(rtc_comptime(), 6)

    def test_runtime_closure_captures_an_aggregate(self) -> None:
        self.assertEqual(rtc_capture_aggregate(5), 6)

    def test_runtime_closure_captures_an_option(self) -> None:
        self.assertEqual(rtc_capture_option(5), 5)

    def test_runtime_closure_captures_a_zst_option(self) -> None:
        # a zero-sized option child has no payload storage: the capture carries
        # only the tag (its presence)
        self.assertEqual(rtc_capture_zst_option(5), 6)
        self.assertEqual(rtc_capture_zst_option_absent(5), 4)

    def test_runtime_closure_forwards_kwargs_only(self) -> None:
        # only ``**kwargs``: its forwarding slot sits right after the positional
        self.assertEqual(rtc_kwargs_only(10), 13)

    def test_runtime_closure_forwards_compiled_varargs(self) -> None:
        self.assertEqual(rtc_varargs_compiled(0), 6)

    def test_runtime_closure_captures_a_runtime_scalar_local(self) -> None:
        # by reference the write-back is visible, by value it is not
        self.assertEqual(rtc_capture_scalar_local(5), 6)
        self.assertEqual(rtc_capture_scalar_local_copies(5), 5)

    def test_a_capture_round_trips_through_the_struct(self) -> None:
        # building the struct and running ``_rebuild_capture`` back through
        # ``_provided_node`` must give the capture's own ``ArgNode``
        pairs: list[tuple[ArgNode, ArgNode]] = []
        orig = interp.HirRunner._rebuild_capture

        def patched(self: interp.HirRunner, node: ArgNode, as_copy: bool, field_base: int, base: InterpVal) -> InterpVal:
            place = orig(self, node, as_copy, field_base, base)
            pairs.append((node, self._provided_node(ArgEntry(place, False))))
            return place

        interp.HirRunner._rebuild_capture = patched
        try:
            self.assertEqual(rtc_round_trip(5), 134)
        finally:
            interp.HirRunner._rebuild_capture = orig
        self.assertEqual(len(pairs), 6)
        for node, rebuilt in pairs:
            self.assertEqual(rebuilt, node, (node, rebuilt))

    def test_a_single_runtime_leaf_mirrors_as_its_field(self) -> None:
        plans: list[RuntimeClosurePlan] = []
        orig = interp.HirRunner._build_runtime_closure

        def patched(self: interp.HirRunner, closure: ClosureValue, capture_places: tuple[InterpVal, ...], as_copy: bool) -> RuntimeClosurePlan:
            plan = orig(self, closure, capture_places, as_copy)
            plans.append(plan)
            return plan

        interp.HirRunner._build_runtime_closure = patched
        try:
            self.assertEqual(rtc_single_leaf(5), 8)
        finally:
            interp.HirRunner._build_runtime_closure = orig
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(plans[0].field_types), 1)
        self.assertTrue(plans[0].struct_type.mirror_is_a_field(_GLOBAL_CONTEXT.mir_lower_cache))

    def test_a_runtime_tag_option_is_a_single_pointer(self) -> None:
        # contrast: a runtime option is one runtime pointer (its tag is only
        # known at runtime, so it is not transportable as compile-time content),
        # while a compile-time option is a compound tag + payload
        plans: list[RuntimeClosurePlan] = []
        orig = interp.HirRunner._build_runtime_closure

        def patched(self: interp.HirRunner, closure: ClosureValue, capture_places: tuple[InterpVal, ...], as_copy: bool) -> RuntimeClosurePlan:
            plan = orig(self, closure, capture_places, as_copy)
            plans.append(plan)
            return plan

        interp.HirRunner._build_runtime_closure = patched
        try:
            self.assertEqual(rtc_option_contrast(5), 5)
        finally:
            interp.HirRunner._build_runtime_closure = orig
        captures = plans[0].captures
        self.assertTrue(any(
            isinstance(c, RuntimeArgNode)
            and isinstance(c.type, sval.PointerType)
            and isinstance(c.type.elem, sval.OptionType)
            for c in captures
        ))
        self.assertTrue(any(
            isinstance(c, ComptimePtrArg) and isinstance(c.content, CompoundArgNode)
            and isinstance(c.content.container_type, sval.OptionType)
            for c in captures
        ))

    def test_a_non_closure_callee_cannot_take_captures(self) -> None:
        # captures are only ever produced by the generated ``__call__`` (whose
        # callee is a closure); handing them to a builtin is rejected
        entry = _a_builtin_call_with_captures()
        call_sig, ret_sig = entry.hir.signature.specialize(
            ArgList((), (), frozendict()), _GLOBAL_CONTEXT.mir_lower_cache,
        )
        analyser = Analyser(_GLOBAL_CONTEXT, _GLOBAL_CONTEXT.mir_lower_cache)
        with self.assertRaises(CompileError) as ctx:
            analyser.analyse_function(entry, call_sig, ret_sig)
        self.assertIn('captures', str(ctx.exception))

    def test_runtime_closure_captures_a_tagged_union(self) -> None:
        self.assertEqual(rtc_capture_union(5), 5)

    def test_runtime_closure_captures_a_comptime_tagged_union(self) -> None:
        self.assertEqual(rtc_capture_comptime_union(), 2)

    def test_runtime_closure_of_a_generic_closure(self) -> None:
        self.assertEqual(rtc_generic(4), 4)

    def test_runtime_closure_captures_a_tuple(self) -> None:
        self.assertEqual(rtc_capture_tuple(5), 11)

    def test_runtime_closure_with_two_captures(self) -> None:
        self.assertEqual(rtc_two_captures(2, 3), 6)

    def test_runtime_closure_with_multiple_returns(self) -> None:
        self.assertEqual(rtc_multi(5), 506)

    def test_runtime_closure_propagates_an_exception(self) -> None:
        self.assertEqual(rtc_raises(5), 5)
        self.assertEqual(rtc_raises(-5), -1)

    def test_runtime_closure_forwards_varargs(self) -> None:
        self.assertEqual(rtc_varargs(10), 16)

    def test_runtime_closure_forwards_kwargs(self) -> None:
        self.assertEqual(rtc_kwargs(10), 13)

    def test_a_runtime_closure_value_is_passed_to_an_inline_function(self) -> None:
        self.assertEqual(rtc_passed_inline(5), 12)

    def test_a_runtime_closure_value_can_live_in_a_struct_field(self) -> None:
        self.assertEqual(rtc_in_struct_field(5), 8)
        self.assertEqual(rtc_in_struct_field_copies(5), 7)

    def test_as_copy_must_be_a_comptime_bool(self) -> None:
        with self.assertRaises(CompileError):
            rtc_as_copy_not_bool()

    def test_as_runtime_closure_needs_a_closure(self) -> None:
        with self.assertRaises(CompileError):
            rtc_not_a_closure()

    def test_as_runtime_closure_rejects_an_unknown_keyword(self) -> None:
        with self.assertRaises(CompileError):
            rtc_unexpected_keyword()


all_tests = [
    SpyClosureTest,
]

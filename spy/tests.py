"""Integration tests for the spy JIT (``spy``).

The functions under test are defined at module level and registered with
the ordinary ``@spy.func()`` decorator; a function body may call the
other registered functions by name (they are module globals, resolved by
the compile-time interpreter) exactly like a user would.  The
undecorated ``add_inline`` is deliberately left unregistered: it stays a
plain Python function and is inlined at its call sites.

Function calls are exercised by ``SpyFunctionCallTest`` below; the struct
features (declaration, layout, construction and methods) by
``SpyStructTest``/``SpyStructMirrorTest``, and generic structs by
``SpyGenericStructTest``.
The global host context caches specializations, so the tests share the
compiled functions; a test that needs a fresh compilation calls a
function no earlier test has compiled.
"""

import ctypes
import io
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any, Literal, Never, Protocol, cast
from unittest import TestCase

from .compiler import (
    CompileError,
    SpyError,
    TypeMismatchError,
    compile_log,
    f32,
    f64,
    i8,
    i32,
    i64,
    mir,
    sval,
    syntax,
    u0,
    u64,
    usize,
    void,
)
from .compiler import as_ as spy_as
from .compiler import bool as spy_bool
from .compiler import typeof as spy_typeof
from .compiler.dsl import _GLOBAL_CONTEXT, _Context, decl_func, func, func_type, struct
from .compiler.lower import LLVMBackend
from .compiler.syntax import (
    Array,
    Comptime,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
    Opaque,
    Option,
    Ptr,
    array,
    as_func_ptr,
    defer,
    errdefer,
    okdefer,
    ptr_cast,
    ref,
)
from .compiler.util import FrozenArraySet, StrBiMap, TriState
from .std import ConstSlicePtr, Numeric, SlicePtr, arr_slice, const_arr_slice, undefined
from .std.mem import DynamicAllocator, align_of, layout_of, size_of
from .std.reflect import (
    ArrayType,
    IntType,
    OptionType,
    PointerType,
    StructType,
    TaggedUnionType,
    type_info,
)

# ---------------------------------------------------------------------------
# functions under test
# ---------------------------------------------------------------------------


@func()
def smoke_test(a: i32, b: i32) -> i32:
    return a + b


@func()
def add[T: Numeric](a: T, b: T) -> T:
    return a + b


@func()
def add_u64(a: u64, b: u64) -> u64:
    return a + b


@func()
def sub(a: i32, b: i32) -> i32:
    return a - b


@func()
def mul(a: i32, b: i32) -> i32:
    return a * b


@func()
def negative_literals(n: i32) -> i32:
    # a negative literal is a ``-`` over an untyped literal, not a bare constant
    # like ``3``: it is evaluated into an expression temporary, which is a
    # compile-time box of the literal's own (untyped) type (see
    # ``sval.coerce_const``)
    if n < -10:
        return -1
    return n * -2 + 3


@func()
def mod(a: i32, b: i32) -> i32:
    return a % b


@func()
def scale(a: f64, b: f64) -> f64:
    return a * b + 1.0


@func()
def add_default[T: Numeric](a: T, b: T = 0) -> T:
    return a + b


def add_inline[T: Numeric](a: T, b: T) -> T:
    compile_log("add_inline was compiled")
    return a + b


def abs_inline(n: i32) -> i32:
    if n < 0:
        return -n
    else:
        return n


def clamp_inline(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def call_abs(n: i32) -> i32:
    return abs_inline(n)


@func()
def call_clamp(n: i32) -> i32:
    return clamp_inline(n)


@func()
def call_abs_plus_one(n: i32) -> i32:
    return abs_inline(n) + 1


@func()
def call_add[T: Numeric](a: T, b: T) -> T:
    return add(a, b)


@func()
def call_inline[T: Numeric](a: T, b: T) -> T:
    return add_inline(a, b)


@func()
def call_inline_log(a: i32, b: i32) -> i32:
    return add_inline(a, b)


@func()
def use_default[T: Numeric](a: T) -> T:
    return add_default(a) # pyright: ignore[reportReturnType]


@func()
def accumulate(a: i32, b: i32) -> i32:
    a += b
    return a


@func()
def sign(n: i32) -> i32:
    if n > 0:
        return 1
    else:
        return -1


@func()
def clamped(n: i32) -> i32:
    if n > 100:
        return 100
    return n


@func()
def assign_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    return x


@func()
def assign_in_both_branches(c: spy_bool, a: i32, b: i32) -> i32:
    x = a
    if c:
        x = b
    else:
        x = b + 1
    return x


@func()
def assign_parameter_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # a parameter is bound in the function body's scope: a branch writes it
    if c:
        a = b
    return a


@func()
def assign_tuple_in_branch(c: spy_bool, a: i32, b: i32) -> i32:
    # ``x`` is bound outside (written), ``y`` is bound nowhere (declared here)
    x = a
    if c:
        x, y = b, a
        return x + y
    return x


@func()
def assign_from_choose(c: spy_bool, d: spy_bool, a: i32, b: i32) -> i32:
    # the branches of an if-expression write the variable the branch sees
    x = a
    if d:
        x = a if c else b
    return x


@func()
def branch_declaration(c: spy_bool, a: i32) -> i32:
    # a name bound nowhere is declared in the branch it is assigned in
    if c:
        y = a
        y = y + 1
        return y
    return a


@func()
def use_branch_declaration(c: spy_bool, a: i32) -> i32:
    if c:
        y = a
    # ``y`` was declared inside the branch, so it is not bound here: a spy
    # compile error (pyright cannot tell, hence the ignore)
    return y  # pyright: ignore


@func()
def le(a: i32, b: i32) -> spy_bool:
    return a <= b


@func()
def eq(a: i32, b: i32) -> spy_bool:
    return a == b


@func()
def is_i32(a) -> spy_bool:
    return spy_typeof(a) == i32


@func()
def is_u64(a) -> spy_bool:
    return spy_typeof(a) == u64


@func()
def nothing(_: i32) -> None:
    pass


@func()
def fact(n: i32) -> i32:
    if n <= 1:
        return 1
    return n * fact(n - 1)


@func()
def is_even(n: i32) -> spy_bool:
    if n == 0:
        return True
    return is_odd(n - 1)


@func()
def is_odd(n: i32) -> spy_bool:
    if n == 0:
        return False
    return is_even(n - 1)


@func()
def inc(a: i32) -> i32:
    return a + 1


@func()
def twice_inc(a: i32) -> i32:
    return inc(inc(a))


@func()
def id_f32(a: f32) -> f32:
    return a


# ---------------------------------------------------------------------------
# loops: ``while`` compiles into a dead ``loop`` block whose head re-evaluates
# the condition at every iteration; ``break`` leaves the loop, ``continue``
# jumps back to the head and the ``else`` clause runs only on a natural exit.
# A loop-carried variable is an ordinary block-local slot (memory), so no phi
# is needed across the back edge.
# ---------------------------------------------------------------------------


@func()
def sum_to(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


@func()
def count_up(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        i = i + 1
    return i


@func()
def sum_calls(n: i32) -> i32:
    # a spy call inside the loop body
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + inc(i)
        i = i + 1
    return total


@func()
def find_divisor(n: i32) -> i32:
    # leaves the loop at the first divisor; when there is none the condition
    # fails and the loop exits normally (with ``i == n``)
    i: i32 = 2
    while i < n:
        if n % i == 0:
            break
        i = i + 1
    return i


@func()
def sum_odd(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
        total = total + i
    return total


@func()
def skip_and_stop(n: i32) -> i32:
    # both ``continue`` and ``break`` in the body, each under a runtime ``if``
    i: i32 = 0
    total: i32 = 0
    while i < n:
        i = i + 1
        if i == 2:
            continue
        if i == 5:
            break
        total = total + i
    return total


@func()
def while_else_natural(n: i32) -> i32:
    # the else clause runs when the condition turns false
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        result = 100
    return result + i


@func()
def while_else_break(n: i32) -> i32:
    # ... and is skipped by a ``break``
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
        if i == 2:
            break
    else:
        result = 100
    return result + i


@func()
def while_else_continue(n: i32) -> i32:
    # a ``continue`` does not skip the else clause for good: it runs when the
    # condition finally turns false
    i: i32 = 0
    result: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        result = 100
    return result + i


@func()
def nested_loop_sum(n: i32, m: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n:
        j: i32 = 0
        while j < m:
            total = total + i * j
            j = j + 1
        i = i + 1
    return total


@func()
def loop_return(n: i32) -> i32:
    # ``while True`` whose only exit is a ``return``: the code after the loop is
    # dead and the loop never needs an exit block
    steps: i32 = 0
    while True:
        if n == 0:
            return steps
        n = n - 1
        steps = steps + 1


@func()
def loop_void(n: i32) -> None:
    i: i32 = 0
    while i < n:
        i = i + 1


@func()
def assign_outside_from_loop(n: i32) -> i32:
    # a variable declared outside the loop is written in the body: it lives in
    # memory, so the write survives the back edge
    acc: i32 = 0
    i: i32 = 0
    while i < n:
        acc += i
        i = i + 1
    return acc


@func()
def comptime_false_loop(n: i32) -> i32:
    # a compile-time false condition never runs the body and runs the else
    # clause once (only the chosen branch of the head ``if`` is emitted)
    total: i32 = 0
    while False:
        total = total + n
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 1
    return total


@func()
def break_in_try(n: i32) -> i32:
    # the try body may raise (``raise_a`` may), so the handler is live; a
    # ``break`` in the body still leaves the loop directly, without going
    # through the clause
    i: i32 = 0
    while i < 100:
        i = i + 1
        try:
            raise_a(1)
            if i == n:
                break
        except ErrorA:
            return -1
    return i


@func()
def continue_in_try(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < 10:
        i = i + 1
        try:
            raise_a(1)
            if i == n:
                continue
            total = total + i
        except ErrorA:
            return -1
    return total


@func()
def bounded_loop(n: i32) -> i32:
    # ``while True`` whose only exit is an explicit ``break``: the exit block
    # comes from that break alone (the implicit one of the lowering is
    # compile-time dead), and the code after the loop is reachable through it
    i: i32 = 0
    while True:
        if i >= n:
            break
        i = i + 1
    return i + 100


def sum_inline(n: i32) -> i32:
    # an undecorated plain function: its body - loop included - is inlined at
    # the call site, in a frame of its own
    i: i32 = 0
    total: i32 = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


@func()
def call_sum_inline(n: i32) -> i32:
    return sum_inline(n)


# ---------------------------------------------------------------------------
# for loops: ``for x in iter`` desugars into an explicit iterator loop (see
# ``astgen._gen_for``); ``range`` (the builtin name) names the ``std.range``
# struct and the loop ends when ``__next__`` raises ``std.StopIteration``.
# ---------------------------------------------------------------------------


@func()
def for_sum(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + i
    return total


@func()
def for_else(n: i32) -> i32:
    # the else clause runs when the sequence is exhausted
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + i
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total


@func()
def for_break(n: i32) -> i32:
    # ... and is skipped by a ``break``
    total: i32 = 0
    for i in range(n, 0, 1):
        if i == 3:
            break
        total = total + i
    else:
        total = total + 100
    return total


@func()
def for_continue(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        if i == 2:
            continue
        total = total + i
    return total


@func()
def for_step(n: i32) -> i32:
    # start and step are given explicitly
    total: i32 = 0
    for i in range(n, 1, 2):
        total = total + i
    return total


@func()
def for_defaults(n: i32) -> i32:
    # ``start`` and ``step`` are left out: the field defaults fill them
    total: i32 = 0
    for i in range(n):
        total = total + i
    return total


@func()
def for_nested(n: i32) -> i32:
    total: i32 = 0
    for i in range(n, 0, 1):
        for j in range(i, 0, 1):
            total = total + j
    return total


@func()
def for_call(n: i32) -> i32:
    # a spy call in the body of a for loop
    total: i32 = 0
    for i in range(n, 0, 1):
        total = total + inc(i)
    return total


@func()
def for_leaks(n: i32) -> i32:
    # the loop variable is only visible inside the loop
    for i in range(n, 0, 1):
        pass
    return i  # pyright: ignore


@func()
def for_iter_is_a_copy(n: i32) -> i32:
    # ``__iter__`` returns the range by value (a copy), so writing through the
    # returned iterator does not touch the original range
    r = range(n, 0, 1)
    it = r.__iter__()
    it.start = 42  # pyright: ignore[reportAttributeAccessIssue]
    return r.start  # pyright: ignore[reportAttributeAccessIssue]


@struct()
class Addressable:
    x: i32

    def addr(self) -> Ptr[Addressable]:
        # ``self`` is bound directly to the incoming pointer, so ``ref(self)``
        # is a ``Ptr[Self]``
        return ref(self)


@func()
def write_through_self_ref(x: i32) -> i32:
    p = Addressable(x)
    q = p.addr()
    q[...].x = 99
    return p.x


# ---------------------------------------------------------------------------
# compile-time loops: a loop preceded by the statement ``syntax.unroll()``
# unrolls its body once per compile-time iteration (the condition is a
# compile-time value that the body advances); ``break`` leaves the whole
# unrolled sequence and ``continue`` jumps to the next unrolled body.
# ---------------------------------------------------------------------------


@func()
def inline_sum() -> i32:
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 4:
        total = total + i
        i = i + 1
    return total


@func()
def inline_false() -> i32:
    # a compile-time false condition unrolls no body and runs the else clause
    total: i32 = 0
    syntax.unroll()
    while False:
        total = total + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 2
    return total


@func()
def inline_break(n: i32) -> i32:
    # a runtime-conditional break leaves all the remaining unrolled bodies
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 10:
        i = i + 1
        if i == n:
            break
        total = total + i
    return total


@func()
def inline_continue(n: i32) -> i32:
    # a runtime-conditional continue jumps to the next unrolled body
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 6:
        i = i + 1
        if i == n:
            continue
        total = total + i
    return total


@func()
def inline_else() -> i32:
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 3:
        total = total + i
        i = i + 1
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total + i


@func()
def inline_nested() -> i32:
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 3:
        j: Comptime = 0
        syntax.unroll()
        while j < 2:
            total = total + 1
            j = j + 1
        i = i + 1
    return total


@func()
def inline_continue_always() -> i32:
    # an unconditional continue: the body has no falling end, so the next body
    # is unrolled from the block the continue jumped to
    i: Comptime = 0
    syntax.unroll()
    while i < 4:
        i = i + 1
        continue
    return i


@func()
def runtime_loop_with_inline(n: i32) -> i32:
    # an inline loop inside a runtime loop
    k: i32 = 0
    total: i32 = 0
    while k < n:
        i: Comptime = 0
        syntax.unroll()
        while i < 3:
            total = total + 1
            i = i + 1
        k = k + 1
    return total


@func()
def inline_continue_or_return(n: i32) -> i32:
    # a ``continue`` whose sibling region of the enclosing if returns: the next
    # body is unrolled from the continue's jump even though nothing falls
    # through
    i: Comptime = 0
    total: i32 = 0
    syntax.unroll()
    while i < 5:
        i = i + 1
        if i == n:
            return total
        if i == 3:
            continue
        total = total + i
    return total + 100


@func()
def inline_bad_cond(n: i32) -> i32:
    # a runtime condition cannot be unrolled: the cap is what reports it
    syntax.unroll()
    while n > 0:
        n = n - 1
    return n


@func()
def inline_loop_misuse(n: i32) -> i32:
    # the marker must be followed by a loop
    syntax.unroll()
    return n


# ---------------------------------------------------------------------------
# compile-time ``for`` loops: the iterable (and so the loop variable) of a loop
# marked with ``syntax.unroll()`` is a compile-time value, so the desugared
# iterator loop unrolls at compile time - ``__next__`` raises ``StopIteration``
# while the interpreter runs, and the ``break`` its ``except`` clause runs ends
# the unrolled sequence (see ``astgen._gen_for``)
# ---------------------------------------------------------------------------


@func()
def unroll_for_sum() -> i32:
    # the literals have no runtime type of their own: the whole loop is
    # compile-time
    total: i32 = 0
    syntax.unroll()
    for i in range(4, 0, 1):
        total = total + i
    return total


@func()
def unroll_for_defaults() -> i32:
    # ``start``/``step`` are left out: the field defaults fill them
    total: i32 = 0
    syntax.unroll()
    for i in range(4):
        total = total + i
    return total


@func()
def unroll_for_else() -> i32:
    total: i32 = 0
    syntax.unroll()
    for i in range(3, 0, 1):
        total = total + i
    else:  # noqa: PLW0120 - the else clause is the point of the fixture
        total = total + 100
    return total


@func()
def unroll_for_break(n: i32) -> i32:
    # a runtime-conditional break leaves every remaining unrolled body (and the
    # else clause, like Python)
    total: i32 = 0
    syntax.unroll()
    for i in range(10, 0, 1):
        if i == n:
            break
        total = total + i
    else:
        total = total + 100
    return total


@func()
def unroll_for_continue() -> i32:
    total: i32 = 0
    syntax.unroll()
    for i in range(5, 0, 1):
        if i == 2:
            continue
        total = total + i
    return total


@func()
def unroll_for_nested() -> i32:
    total: i32 = 0
    syntax.unroll()
    for i in range(3, 0, 1):
        syntax.unroll()
        for j in range(2, 0, 1):
            total = total + i * 10 + j
    return total


@func()
def unroll_for_typed_var() -> i32:
    # a compile-time iterator *variable*, of an explicit element type with a
    # runtime representation (``range`` names ``std.range``, so it is
    # subscriptable in spy but not in the Python type system)
    r: Comptime[range[i32]] = range(3, 0, 1)  # pyright: ignore
    total: i32 = 0
    syntax.unroll()
    for i in r:
        total = total + i
    return total


@func()
def unroll_for_runtime_iter(n: i32) -> i32:
    # a runtime iterator cannot be unrolled: the unroll cap reports it
    total: i32 = 0
    syntax.unroll()
    for i in range(n, 0, 1):
        total = total + i
    return total


# ---------------------------------------------------------------------------
# struct values: a struct is declared by decorating a class with ``@struct()``
# - its annotated class attributes are the fields, in declaration order, and
# the functions of its body are its methods - and is laid out by the mirror
# rules of ``sval.StructType._calculate_mir``.  A construction fills the
# fields of the location it is built in (a pending default constructor, see
# ``interp``) and a small struct (up to the by-value limit) is returned by
# value, a larger one through a result pointer (``sval.returns_via_result_ptr``).
#
# The structs of this module are named in the spy source like any other global:
# as an annotation (``s: Small``), as a constructor (``Small(...)``) and as the
# type a method is called on.  The tests read the struct *type* off the handle
# with ``struct_type``; there is no Python-level constructor yet.
# ---------------------------------------------------------------------------


def struct_type(handle: Any) -> sval.StructType:
    """The struct type a ``@struct()`` class declares."""
    type = handle.as_spy_value()
    assert isinstance(type, sval.StructType)
    return type


def spy_type(annotation: Any) -> sval.Type:
    """The spy type a Python annotation evaluates to (see ``sval.as_value``)."""
    type = sval.as_value(annotation, _GLOBAL_CONTEXT)
    assert isinstance(type, sval.Type)
    return type


# the MIR-mirror interning table of the global host context: the MIR types the
# tests compare against are the ones it created (see ``sval.MirLowerCache``)
MIR_CACHE = _GLOBAL_CONTEXT.mir_lower_cache


def struct_mirror(handle: Any) -> mir.Type:
    """The MIR mirror of the struct type a handle declares (see
    ``sval.StructType.get_mir_type``); the structs a test takes the mirror of
    have storage, so it has one."""
    mirror = struct_type(handle).get_mir_type(MIR_CACHE)
    assert mirror is not None
    return mirror


@struct()
class Small:
    a: i32
    b: i32

    def total(self) -> i32:
        return self.a + self.b


@struct()
class Large:
    a: i64
    b: i64
    c: i64
    d: i64


@struct()
class Counter:
    """A struct with both kinds of method (a registered ``bump``, compiled
    into a native call, and a plain ``double``, inlined)."""

    n: i32

    @func()
    def bump(self, k: i32) -> i32:
        self.n = self.n + k
        return self.n

    def double(self) -> i32:
        return self.n * 2


# a struct of one field mirrors to that field's own type, and the fields of
# a struct of several fields are ordered by alignment (the least-aligned
# first) - unless the struct is ``extern_c``, which keeps the C layout
@struct()
class One:
    a: i32

    def get(self) -> i32:
        return self.a


@struct()
class Nested:
    inner: One


@struct()
class Mixed:
    wide: i64
    narrow: i8


@struct(extern_c=True)
class ExternOne:
    a: i32


@struct(extern_c=True)
class ExternMixed:
    wide: i64
    narrow: i8


# a zero-sized field occupies no storage: it has no mirror position and its
# value is the unit value of its type
@struct()
class Holder:
    v: void
    n: i32


# a struct whose fields are all zero-sized is zero-sized itself: it holds no
# storage at all, and returning one is a call that returns nothing
@struct()
class Nothing:
    v: void
    z: u0


# a second zero-sized struct, a different type than ``Nothing``
@struct()
class Blank:
    v: void


# a struct whose fields declare defaults: a construction may leave them out, and
# the field takes the value the class body assigned to it (see
# ``sval.StructField``/``interp.finish_struct``)
@struct()
class Defaulted:
    x: i32
    y: i32 = 7
    z: i32 = 9


# a struct whose zero-sized field declares a default too: leaving it out stores
# nothing, since every value of a zero-sized type is its unit value
@struct()
class DefaultedZst:
    v: void = None
    n: i32 = 3


# a generic struct with a defaulted field: the default is coerced to the type
# the field got from the specialization (``cast`` keeps the Python type checker
# happy about a type parameter's default, like ``std.range``'s fields)
@struct()
class GenericDefault[T]:
    value: T
    other: T = cast(T, 0)


# a struct whose void methods are called for their effect alone, one
# registered (``reset``, compiled into a native call) and one plain
# (``clear``, inlined)
@struct()
class Sink:
    n: i32

    @func()
    def reset(self) -> None:
        self.n = 0

    def clear(self) -> None:
        self.n = 0


# a declaration built by hand: a pointer field has no annotation spelling yet
# (a pointer is word-aligned, like an integer of that width)
WithPointer: Any = sval.StructTypeHead('WithPointer')
WithPointer.add_field('p', sval.PointerType(i32))  # pyright: ignore[reportArgumentType]
WithPointer.add_field('n', i8)


@func()
def struct_defaults(x: i32) -> i32:
    # every field left out is filled from its default
    p = Defaulted(x)
    return p.x + p.y + p.z


@func()
def struct_defaults_partial(x: i32) -> i32:
    p = Defaulted(x, 100)
    return p.x + p.y + p.z


@func()
def struct_defaults_comptime(x: i32) -> i32:
    # the defaults fill an inline (compile-time) aggregate too
    p: Comptime = Defaulted(x, 1)
    return p.x + p.y + p.z


@func()
def struct_defaults_zst() -> i32:
    p = DefaultedZst()
    return p.n


@func()
def struct_defaults_generic(x: i32) -> i32:
    g = GenericDefault[i32](x)
    return g.value + g.other


@func()
def struct_missing_a_value() -> i32:
    # a field with no default may not be left out
    p = Defaulted()  # pyright: ignore[reportCallIssue]
    return p.x


@func()
def struct_local(x: i32) -> i32:
    s = Small(x, 1)
    return s.a + s.b


@func()
def struct_all_comptime() -> i32:
    s = Small(1, 2)
    return s.a + s.b


@func()
def make_small(x: i32) -> Small:
    return Small(x, 2)


@func()
def use_small(x: i32) -> i32:
    s = make_small(x)
    return s.a + s.b


@func()
def sum_small(s: Small) -> i32:
    return s.a + s.b


@func()
def use_sum_small(x: i32) -> i32:
    return sum_small(Small(x, 3))


@func()
def total_small(x: i32) -> i32:
    s = Small(x, 1)
    return s.total()


@func()
def make_large(x: i64, c: spy_bool) -> Large:
    if c:
        return Large(x, 1, 2, 3)
    return Large(x, 4, 5, 6)


@func()
def use_large(x: i64, c: spy_bool) -> i64:
    return make_large(x, c).a + make_large(x, c).d


@func()
def bump_counter(n: i32, k: i32) -> i32:
    c = Counter(n)
    c.bump(k)
    return c.double()


@func()
def one_field_local(x: i32) -> i32:
    s = One(x)
    return s.a


@func()
def mixed_fields_local(w: i64, n: i8) -> i64:
    m = Mixed(w, n)
    return m.wide + m.narrow


@func()
def extern_one_field_local(x: i32) -> i32:
    s = ExternOne(x)
    return s.a


@func()
def zst_field_local(n: i32) -> i32:
    h = Holder(None, n)
    return h.n


@func()
def nested_field_local(x: i32) -> i32:
    n = Nested(One(x))
    return n.inner.a


@func()
def nested_method_local(x: i32) -> i32:
    n = Nested(One(x))
    return n.inner.get()


# ``nothing`` returns no value: as an expression statement the call is dropped
# (its result location is a temporary of no further use), and bound to a
# variable it types that variable as ``void``
@func()
def discard_void_call(x: i32) -> i32:
    nothing(x)
    return x


@func()
def void_call_type(x: i32) -> spy_bool:
    y = nothing(x)
    return spy_typeof(y) == void


@func()
def zero_bits(_: i32) -> u0:
    return 0


@func()
def discard_zero_bits_call(x: i32) -> i32:
    zero_bits(x)
    return x


@func()
def zero_bits_call_type(x: i32) -> spy_bool:
    y = zero_bits(x)
    return spy_typeof(y) == u0


@func()
def call_registered_void_method(x: i32) -> i32:
    s = Sink(x)
    s.reset()
    return s.n


@func()
def call_inline_void_method(x: i32) -> i32:
    s = Sink(x)
    s.clear()
    return s.n


@func()
def make_nothing(x: i32) -> Nothing:
    return Nothing(None, 0)


# a call that returns a zero-sized struct delivers no value, but its result
# location takes the struct type: the variable bound to the call is a
# compile-time box of the type's unit value, and it is typed as the struct
@func()
def use_nothing(x: i32) -> i32:
    n = make_nothing(x)
    return x if spy_typeof(n) == Nothing else x + 1


# the same struct built where it is declared: the construction writes nothing,
# so the slot takes its type from the construction itself
@func()
def construct_nothing(x: i32) -> i32:
    n = Nothing(None, 0)
    return x if spy_typeof(n) == Nothing else x + 1


# a construction takes its type from the construction itself: ``Blank``'s only
# field is zero-sized, and its value is given all the same (a field may only be
# left out when it has a default, which is not implemented yet)
@func()
def construct_blank(x: i32) -> i32:
    b = Blank(None)
    return x if spy_typeof(b) == Blank else x + 1


# a construction delivers its value like any other store point, so the slot's
# type is still the peer type of everything stored into it: two *different*
# zero-sized structs have no peer type (rather than the first one silently
# pinning the slot)
@func()
def choose_two_zst_structs(c: spy_bool) -> spy_bool:
    s = Blank(None) if c else Nothing(None, 0)
    return spy_typeof(s) == Blank


@func()
def untyped_local(n: i32) -> i32:
    x = 1
    x = 2
    return x + n


@func()
def untyped_return():
    return 1


@func()
def untyped_param(x) -> i32:
    return x


@func()
def call_untyped_param() -> i32:
    return untyped_param(1)


# ---------------------------------------------------------------------------
# annotated local variables: ``name: T = expr`` declares the type of a fresh
# variable (its slot is materialized right away, and the value is coerced to
# the declared type), and ``name: Comptime``/``name: Comptime[T]`` mark it
# compile-time - a box rather than memory, which is what lets it hold a
# compile-time-only value such as a type
# ---------------------------------------------------------------------------


@func()
def annotated_local(x: i32) -> i32:
    y: i32 = x + 1
    return y * 2


@func()
def annotated_local_coerced(x: i32) -> i64:
    # the declared type governs: the value written into the slot is widened
    y: i64 = x
    return y + 1


@func()
def annotated_struct_local(x: i32) -> i32:
    p: Pair[i32] = Pair(x, 2)
    return p.total()


@func()
def annotated_plain_struct_local(x: i32) -> i32:
    s: Small = Small(x, 2)
    return s.total()


@func()
def annotated_generic_local[T](x: T) -> T:
    # the declared type is a type parameter: the call solves it
    y: T = x
    return y


@func()
def call_annotated_generic_local(x: i32) -> i32:
    return annotated_generic_local(x)


@func()
def comptime_local_holds_a_type(x: i32) -> spy_bool:
    # a compile-time variable may hold a type, a compile-time-only value
    t: Comptime = spy_typeof(x)
    return t == i32


@func()
def comptime_typed_local(x: i32) -> i32:
    c: Comptime[i32] = x
    return c + 1


@func()
def annotated_comptime_param(x: Comptime[void]) -> spy_bool:
    # a zero-sized parameter can be marked compile-time (a runtime parameter
    # cannot be passed at compile time yet)
    return spy_typeof(x) == void


@func()
def comptime_local_holds_a_tuple(x: i32) -> i32:
    # a ``Comptime`` variable may hold anything that is *not* a runtime
    # value, a tuple included: the tuple is a compile-time structure of the
    # places of its elements, even though the elements themselves are runtime
    # values here
    a = x
    b = x + 1
    t: Comptime = a, b
    p, q = t
    return p * 100 + q


@func()
def passes_comptime_value(x: i32) -> spy_bool:
    # the null value is a compile-time value
    return annotated_comptime_param(None)


# ---------------------------------------------------------------------------
# ``syntax.comptime()``: the statement marker that declares the variable of the
# declaration immediately after it as a compile-time one, the way
# ``syntax.unroll()`` marks the loop that follows it - ``syntax.comptime()``
# followed by ``a: T = e`` is ``a: Comptime[T] = e``, and followed by ``a = e``
# (no annotation) it is ``a: Comptime = e`` (see ``astgen._gen_stmt``)
# ---------------------------------------------------------------------------


@struct()
class MarkerState:
    on: spy_bool
    n: i32


@func()
def comptime_marker_typed() -> i32:
    # the marker plus a typed declaration is ``i: Comptime[i32] = 0``: the loop
    # condition is a compile-time value, so the loop unrolls
    syntax.comptime()
    i: i32 = 0
    total: i32 = 0
    syntax.unroll()
    while i < 4:
        total = total + i
        i = i + 1
    return total


@func()
def comptime_marker_untyped() -> i32:
    # no annotation: the value determines the type (a bare ``Comptime``), and an
    # untyped literal has no runtime representation at all
    syntax.comptime()
    n = 4
    total: i32 = 0
    syntax.unroll()
    while n > 0:
        total = total + n
        n = n - 1
    return total


@func()
def comptime_marker_holds_a_type(x: i32) -> spy_bool:
    # a compile-time variable is what holds a type (a plain local cannot)
    syntax.comptime()
    t = spy_typeof(x)
    return t == i32


@func()
def comptime_marker_struct(x: i32) -> i32:
    # a compile-time struct declared with the marker: its field conditions a
    # compile-time loop
    syntax.comptime()
    s: MarkerState = MarkerState(True, 0)
    total: i32 = 0
    syntax.unroll()
    while s.on:
        s.on = False
        total = total + x
    return total


@func()
def comptime_marker_holds_a_runtime_value(x: i32) -> i32:
    # a compile-time variable of a runtime type holds the runtime value
    syntax.comptime()
    v: i32 = x
    return v + 1


@func()
def comptime_marker_without_a_value() -> i32:
    # the marker also marks a declaration that writes no value
    syntax.comptime()
    n: i32
    n = 5
    return n + 1


@func()
def comptime_marker_destructuring() -> i32:
    # the marker marks every name a destructuring declaration declares
    syntax.comptime()
    a, b = 1, 2
    total: i32 = 0
    syntax.unroll()
    while a > 0:
        total = total + a * 10 + b
        a = a - 1
    return total


@func()
def comptime_marker_in_a_branch(c: spy_bool) -> i32:
    # the marker marks a declaration of the block it sits in
    total: i32 = 0
    if c:
        syntax.comptime()
        k = 3
        total = total + k
    return total


@func()
def comptime_marker_before_a_non_declaration() -> i32:
    syntax.comptime()
    return 1


@func()
def comptime_marker_at_the_end():
    syntax.comptime()


@func()
def comptime_marker_redundant() -> i32:
    # the annotation already declares the variable compile-time
    syntax.comptime()
    a: Comptime = 1
    return a


@func()
def comptime_marker_on_an_assignment() -> i32:
    # the marker marks a *declaration*: this statement assigns to a variable
    # that is bound already
    a: i32 = 1
    syntax.comptime()
    a = 2
    return a


@func()
def comptime_marker_as_a_value() -> i32:
    return syntax.comptime()  # pyright: ignore[reportReturnType]


@func()
def comptime_marker_before_a_loop() -> i32:
    # the marker must come immediately before the declaration it marks
    syntax.comptime()
    syntax.unroll()
    while False:
        pass
    return 1


@func()
def passes_runtime_value_to_comptime(x: i32) -> spy_bool:
    # a runtime value would be dropped by the callee, which reads its
    # compile-time parameter as a compile-time value whatever it is given
    return annotated_comptime_param(x)  # pyright: ignore[reportArgumentType]


@func()
def bad_redeclare(x: i32) -> i32:
    y: i32 = x  # pyright: ignore[reportRedeclaration]
    y: i32 = x + 1
    return y


@func()
def tuple_declare(x: i32) -> i32:
    a, b = x, x + 1
    return a * 10 + b


@func()
def tuple_swap(a: i32, b: i32) -> i32:
    # known design flaw: a destructuring is written into its targets in order,
    # so ``a, b = b, a`` reads ``a`` after it has been overwritten (see
    # ``_gen_assign``): both variables end up holding ``b``
    a, b = b, a
    return a * 10 + b


@func()
def tuple_nested(x: i32) -> i32:
    a, (b, c) = x, (x + 1, x + 2)
    return a * 100 + b * 10 + c


@func()
def tuple_of_calls(x: i32) -> i32:
    # the right-hand side of a destructuring is generated straight into the
    # targets: each call writes its result into the target it is bound to,
    # with no intermediate tuple value (see ``_gen_assign``)
    a, b = inc(x), inc(x + 1)
    return a * 10 + b


@func()
def bad_comptime_tuple_branch(cond: spy_bool) -> i32:
    # one branch of the if-expression is a tuple and the other a single value:
    # the slot is stored with incompatible types (see ``init_tuple``)
    t: Comptime = (1, 2) if cond else inc(3)
    p, q = t
    return p * 10 + q


@struct()
class Slice[T]:
    ptr: T
    len: u64


# ---------------------------------------------------------------------------
# generic structs: ``class Foo[T]`` declares a struct *template*, and
# ``Foo[i32]`` names one specialization of it.  Every method resolves through
# the specialization it is called on - ``a.m()`` behaves like
# ``typeof(a).m(a)`` - so its ``self`` is the struct template and a call
# substitutes the specialization's type arguments into the method's
# signature.  A type parameter is also usable as a value inside a body.
# ---------------------------------------------------------------------------


@struct()
class Pair[T]:
    a: T
    b: T

    @func()
    def total(self) -> T:
        return self.a + self.b  # pyright: ignore[reportOperatorIssue]

    def doubled_a(self) -> T:
        return self.a + self.a  # pyright: ignore[reportOperatorIssue]


@struct()
class Box[T]:
    """A generic struct with a registered method that returns its type
    parameter."""

    v: T

    @func()
    def get(self) -> T:
        return self.v


@struct()
class PlainBox[T]:
    """A generic struct whose method is an inlined plain method."""

    v: T

    def get(self) -> T:
        return self.v


@struct()
class Shadow[T]:
    """A generic struct with a method that declares a type parameter of its
    own, shadowing the struct's parameter of the same name."""

    a: T

    @func()
    def pick[T](self, b: T) -> T:  # pyright: ignore[reportGeneralTypeIssues]
        return b

    @func()
    def own(self) -> T:
        return self.a


@struct()
class NestedGeneric[T]:
    """A generic struct with a generic field: the field's type is
    substituted recursively."""

    inner: Pair[T]
    n: T


@struct()
class Maker[T]:
    """A generic struct whose method builds and returns a specialization of
    another generic struct from its own type parameter."""

    a: T

    @func()
    def pair(self) -> Pair[T]:
        return Pair[T](self.a, self.a)

    @func()
    def is_own_type(self) -> spy_bool:
        return spy_typeof(self.a) == T

    def plain_is_own_type(self) -> spy_bool:
        return spy_typeof(self.a) == T


@struct()
class StructHolder:
    """A non-generic struct with a generic field type."""

    p: Pair[i32]
    extra: i32


@func()
def make_pair[T](a: T, b: T) -> Pair[T]:
    return Pair[T](a, b)


@func()
def first[T](p: Pair[T]) -> T:
    return p.a


@func()
def generic_typeof[T](a: T) -> spy_bool:  # pyright: ignore[reportInvalidTypeVarUse]
    return spy_typeof(a) == T


@func()
def generic_pair_total(x: i32) -> i32:
    return Pair[i32](x, 3).total()


@func()
def generic_pair_plain_method(x: i32) -> i32:
    return Pair[i32](x, 4).doubled_a()


@func()
def generic_box_get(x: i32) -> i32:
    return Box[i32](x).get()


@func()
def generic_plain_box_get(x: i32) -> i32:
    return PlainBox[i32](x).get()


@func()
def generic_method_with_own_type_param(x: i32) -> i32:
    return Shadow[i64](0).pick(x)


@func()
def generic_method_uses_the_struct_type(x: i32) -> i64:
    return Shadow[i64](x).own()


@func()
def generic_struct_returned(x: i32) -> i32:
    return make_pair(x, 4).total()


@func()
def generic_struct_argument(x: i32) -> i32:
    return first(Pair[i32](x, x))


@func()
def generic_nested_field(x: i32) -> i32:
    n = NestedGeneric[i32](Pair[i32](x, 1), 2)
    return n.inner.total() + n.n


@func()
def generic_method_returns_a_struct(x: i32) -> i32:
    return Maker[i32](x).pair().total()


@func()
def generic_struct_typeof_dispatch(x: i32) -> spy_bool:
    return Maker[i32](x).is_own_type()


@func()
def generic_struct_typeof_inline(x: i32) -> spy_bool:
    return Maker[i32](x).plain_is_own_type()


@func()
def is_pair_i32(p: Pair[i32]) -> spy_bool:
    return spy_typeof(p) == Pair[i32]


@func()
def generic_struct_type_compared(x: i32) -> spy_bool:
    return is_pair_i32(Pair[i32](x, x))


@func()
def generic_field_of_nongeneric(x: i32) -> i32:
    h = StructHolder(Pair[i32](x, 1), 5)
    return h.p.total() + h.extra


# a generic struct whose type parameter is used by no field, built into a
# fresh local slot whose type is not known: a construction of the bare
# template cannot tell which specialization to build
@struct()
class Phantom[T]:
    v: i32

    def get(self) -> i32:
        return self.v


# the generic arguments of a construction that names the bare template are
# taken from the type of the location it is built into - here the result
# location of ``make_pair_inferred``, whose declared return type is known
@func()
def make_pair_inferred[T](a: T, b: T) -> Pair[T]:
    return Pair(a, b)


@func()
def inferred_pair_total(x: i32) -> i32:
    return make_pair_inferred(x, 3).total()


@func()
def inferred_pair_total_f64(x: f64) -> f64:
    return make_pair_inferred(x, 2.5).total()


@func()
def explicit_generic_construction(x: i32) -> i32:
    return Pair[i32](x, 3).total()


@func()
def keyword_construction(x: i32) -> i32:
    return Pair[i32](b=x, a=3).total()


@func()
def mixed_construction(x: i32) -> i32:
    return Pair[i32](x, b=x).total()


@func()
def inferred_from_field_values(x: i32) -> i32:
    # ``Pair(...)`` built in a fresh local slot whose type is not known: the
    # generic argument is inferred from the values written into the fields
    p = Pair(x, x)
    return p.total()


@func()
def uninferable_construction(x: i32) -> i32:
    return Phantom(x).get()


@func()
def wrong_generic_arguments(x: i32) -> i32:
    return Pair[i32, i64](x, 3).total()  # pyright: ignore


# a struct of one struct field whose own mirror is a struct type (two fields of
# the same width): the field still sits at the address of the value itself
@struct()
class TwoI64:
    a: i64
    b: i64


@struct()
class OuterTwo:
    inner: TwoI64


@func()
def nested_struct_field(x: i64) -> i64:
    o = OuterTwo(TwoI64(x, 1))
    return o.inner.a


# ---------------------------------------------------------------------------
# pointers: ``syntax.Ptr`` is C's pointer type - ``ref(a)`` takes the address
# of ``a`` (C's ``&a``) and ``p[...]`` denotes the place the pointer value
# ``p`` points at (C's ``*p``)
# ---------------------------------------------------------------------------


@func()
def deref_local(x: i32) -> i32:
    p = ref(x)
    return p[...]


@func()
def write_through_ptr(x: i32, v: i32) -> i32:
    p = ref(x)
    p[...] = v
    return x


@func()
def add_through_ptr(x: i32) -> i32:
    p = ref(x)
    p[...] += 1
    return x


@func()
def deref_module_qualified(x: i32) -> i32:
    # ``syntax.ref`` names the same function as a plain import of ``ref``
    p = syntax.ref(x)
    return p[...]


@func()
def incr_ptr(p: Ptr[i32]) -> i32:
    p[...] = p[...] + 1
    return p[...]


@func()
def call_incr_ptr(x: i32) -> i32:
    v = x
    r = incr_ptr(ref(v))
    return v * 100 + r


@func()
def deref_generic[T](p: Ptr[T]) -> T:
    # the pointee type is solved from the pointer argument
    return p[...]


@func()
def call_deref_generic(x: i32) -> i32:
    return deref_generic(ref(x))


@func()
def ptr_identity[T](p: Ptr[T]) -> Ptr[T]:
    # the pointee type is solved from the pointer argument
    return p


@func()
def call_ptr_identity(x: i32) -> i32:
    p = ref(x)
    q = ptr_identity(p)
    return q[...]


@func()
def read_const_ptr(p: ConstPtr[i32]) -> i32:
    return p[...]


@func()
def call_read_const_ptr(x: i32) -> i32:
    # a mutable pointer is accepted where a const one is expected (a spy rule;
    # pyright cannot express the conversion)
    return read_const_ptr(ref(x))  # pyright: ignore[reportArgumentType]


@struct()
class PtrHolder:
    p: Ptr[i32]


@func()
def field_read(x: i32) -> i32:
    h = PtrHolder(ref(x))
    return h.p[...]


@func()
def field_write(x: i32, v: i32) -> i32:
    h = PtrHolder(ref(x))
    h.p[...] = v
    return x


@func()
def field_through_ptr(a: i32, b: i32) -> i32:
    # a pointer to a struct: a field of the pointee is written and read
    # through the pointer
    s = Small(a, b)
    p = ref(s)
    p[...].a = p[...].a + 1
    return p[...].total()


# ---------------------------------------------------------------------------
# multi pointers and slices: a pointer is either a *single* one (``Ptr``: one
# place, dereferenced with ``p[...]``) or a *multi* one (``MultiPtr``: the
# places of the elements that follow one another, indexed with ``p[i]`` and
# offset with ``p + n``).  A pointer to an array converts to the multi pointer of
# its elements, and ``std.arr_slice``/``std.const_arr_slice`` turn it into the
# ``std.SlicePtr``/``std.ConstSlicePtr`` of the whole array; a *slice* of a
# multi pointer builds one directly: ``p[a:b]`` is ``SlicePtr(p + a, b - a)``
# ---------------------------------------------------------------------------


@func()
def multi_ptr_index(x: i32) -> i32:
    # ``s.ptr`` is the multi pointer of the array's elements
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.ptr[2]


@func()
def multi_ptr_write(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s.ptr[1] = 99
    return a[1]


@func()
def multi_ptr_offset(x: i32) -> i32:
    # ``p + n`` is the address ``n`` elements after the one ``p`` carries
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = s.ptr
    q = p + 2
    return q[...]


@func()
def multi_ptr_offset_assign(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = s.ptr
    p += 2
    return p[...]


@func()
def ref_of_index_is_the_offset(x: i32) -> i32:
    # ``ref(p[n])`` is ``p + n``
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = s.ptr
    return ref(p[1])[...]


@func()
def single_ptr_offset(x: i32) -> i32:
    p = ref(x)
    q = p + 1  # pyright: ignore[reportOperatorIssue]
    return q[...]


@func()
def single_ptr_index(x: i32) -> i32:
    p = ref(x)
    return p[0]  # pyright: ignore[reportArgumentType]


@func()
def read_multi(p: MultiPtr[i32]) -> i32:
    # the annotated multi pointer type: an array pointer converts to it
    return p[1]


@func()
def read_const_multi(p: ConstMultiPtr[i32]) -> i32:
    return p[1]


@func()
def read_single(p: Ptr[i32]) -> i32:
    return p[...]


@func()
def multi_ptr_parameter(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return read_multi(s.ptr)


@func()
def multi_ptr_to_const_parameter(x: i32) -> i32:
    # a multi pointer converts to a const one
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return read_const_multi(s.ptr)


@func()
def multi_ptr_to_single_parameter(x: i32) -> i32:
    # a multi pointer converts to a single one (not the other way around)
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return read_single(s.ptr)


@func()
def single_ptr_is_not_multi(x: i32) -> i32:
    return read_multi(ref(x))  # pyright: ignore[reportArgumentType]


@func()
def slice_length_of(s: SlicePtr[i32]) -> u64:
    return s.length


@func()
def slice_length(x: i32) -> u64:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.length


@func()
def generic_slice_length[T](s: SlicePtr[T]) -> u64:
    # the element type is solved from the slice argument
    return s.length


@func()
def const_slice_length_of(s: ConstSlicePtr[i64]) -> u64:
    return s.length


@func()
def slice_argument(x: i32) -> u64:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return slice_length_of(s)


@func()
def slice_of_an_array_argument(x: i32) -> u64:
    # ``arr_slice`` turns a pointer to an array into the slice of the whole array
    a = array(x, x + 1, x + 2, x + 3)
    return generic_slice_length(arr_slice(ref(a)))


@func()
def slice_of_a_const_array(p: Array[i64, Literal[4]]) -> u64:
    # an array parameter is addressed by the callee (32 bytes: past the by-value
    # limit), and that address is const: the slice of it is a const one
    return const_slice_length_of(const_arr_slice(ref(p)))


@func()
def call_slice_of_a_const_array(x: i64) -> u64:
    return slice_of_a_const_array(array(x, x + 1, x + 2, x + 3, length=4))


@func()
def slice_variable_is_writable(x: i32) -> u64:
    # a slice *variable* is ordinary storage: the fields of it are writable
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s.length = 2
    return s.length


@func()
def slice_of_a_temporary_is_const(x: i32) -> u64:
    # the slice a subscript *builds* is a view, not storage: writing through it
    # is rejected (its pointer is const)
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s.ptr[1:3].length = 5
    return s.length


@func()
def slice_of_a_slice(x: i32) -> i32:
    # ``p[1:3]`` is ``SlicePtr(p + 1, 3 - 1)``: the elements 1 and 2
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    t = s.ptr[1:3]
    return t.ptr[0] + t.ptr[1]


@func()
def slice_of_a_slice_length(x: i32) -> u64:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.ptr[1:3].length


@func()
def slice_of_a_slice_with_a_step_of_one(x: i32) -> u64:
    # a step is part of the subscript syntax but a slice of a *pointer* has no
    # step: the elements follow one another (only a step of 1 says so)
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.ptr[1:3:1].length


@func()
def slice_of_a_slice_with_a_step(x: i32) -> u64:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.ptr[0:2:2].length


@func()
def slice_of_a_slice_without_a_lower(x: i32) -> i32:
    # a bound the source left out is absent; a slice of a *pointer* takes a
    # missing lower bound as 0
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    t = s.ptr[:2]
    return t.ptr[0] + t.ptr[1]


@func()
def slice_of_a_slice_without_an_upper(x: i32) -> u64:
    # a slice of a *pointer* has no length to slice to the end, so an upper
    # bound is required
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s.ptr[1:].length


@func()
def slice_subscript_element(x: i32) -> i32:
    # ``s[i]`` goes through ``SlicePtr.__spy_getitemptr__``, the place of the
    # i-th element (see ``interp.HirRunner.subscript``)
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    return s[2]


@func()
def slice_subscript_write(x: i32) -> i32:
    # the place the method returns is written through, landing in the array
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s[1] = 99
    return a[1]


@func()
def const_slice_subscript_element(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a)).as_const()
    return s[1]


@func()
def slice_method_slice(x: i32) -> i32:
    # ``slice(begin, end)`` is a sub-view: elements 1 and 2
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    t = s.slice(1, 3)
    return t[0] + t[1]


@func()
def slice_method_open_bounds(x: i32) -> i32:
    # an absent bound is 0 (begin) or the length (end)
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    t = s.slice(None, None)
    return t[0] + t[3]


@func()
def const_slice_method_slice(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a)).as_const()
    t = s.slice(1, 4)
    return t[0] + t[1] + t[2]


@func()
def slice_out_of_bounds_calls_the_stub(x: i32) -> i32:
    # an out-of-bounds slice calls ``_out_of_bounds`` (a no-op stub for now) and
    # carries on - the result is never read
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s.slice(0, 5)
    return 7


@func()
def slice_subscript_out_of_bounds_calls_the_stub(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    s[s.length]
    return 7


# ---------------------------------------------------------------------------
# subscript overloading: a struct value is subscripted through its own
# ``__spy_getitemptr__`` method, which gives the *place* of the element
# (see ``interp.HirRunner.subscript``)
# ---------------------------------------------------------------------------


@struct()
class Indexable:
    """A struct that overloads the subscript with an inlined plain method."""

    ptr: MultiPtr[i32]

    def __spy_getitemptr__(self, index: i64) -> MultiPtr[i32]:
        return self.ptr + index

    if TYPE_CHECKING:
        # only so that the Python type checker accepts the ``p[i]`` spellings
        # below: the subscript compiles to ``hir.Subscript`` and never reaches
        # these (see ``interp.HirRunner.subscript``), so they are not struct
        # methods
        def __getitem__(self, index: int) -> i32: ...
        def __setitem__(self, index: int, value: i32) -> None: ...


@struct()
class RegisteredIndexable:
    """Likewise, with a registered (compiled) method."""

    ptr: MultiPtr[i32]

    @func()
    def __spy_getitemptr__(self, index: i64) -> MultiPtr[i32]:
        return self.ptr + index

    if TYPE_CHECKING:
        def __getitem__(self, index: int) -> i32: ...
        def __setitem__(self, index: int, value: i32) -> None: ...


@struct()
class BadIndexable:
    """A struct whose ``__spy_getitemptr__`` returns a value, not a place."""

    ptr: MultiPtr[i32]

    def __spy_getitemptr__(self, index: i64) -> i32:
        return self.ptr[index]

    if TYPE_CHECKING:
        def __getitem__(self, index: int) -> i32: ...


@func()
def subscript_overload_read(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = Indexable(s.ptr)
    return p[2]


@func()
def subscript_overload_write(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = Indexable(s.ptr)
    p[1] = 99
    return a[1]


@func()
def registered_subscript_overload(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    p = RegisteredIndexable(s.ptr)
    return p[0] + p[3]


@func()
def bad_subscript_overload(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    b = BadIndexable(s.ptr)
    return b[1]


# ---------------------------------------------------------------------------
# iterating a slice: ``__iter__`` of ``SlicePtr``/``ConstSlicePtr`` yields the
# elements by value, ``refs()`` a pointer to each of them
# ---------------------------------------------------------------------------


@func()
def iterate_slice_sum(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    total: i32 = 0
    for v in s:
        total += v
    return total


@func()
def iterate_slice_refs_write(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    s = arr_slice(ref(a))
    for p in s.refs():
        p[...] = 99
    return a[0] + a[1] + a[2] + a[3]


@func()
def iterate_const_slice_sum(p: Array[i32, Literal[4]]) -> i32:
    s = const_arr_slice(ref(p))
    total: i32 = 0
    for v in s:
        total += v
    return total


@func()
def call_iterate_const_slice_sum(x: i32) -> i32:
    return iterate_const_slice_sum(array(x, x + 1, x + 2, x + 3, length=4))


@func()
def iterate_const_slice_refs_sum(p: Array[i32, Literal[4]]) -> i32:
    s = const_arr_slice(ref(p))
    total: i32 = 0
    for q in s.refs():
        total += q[...]
    return total


@func()
def call_iterate_const_slice_refs_sum(x: i32) -> i32:
    return iterate_const_slice_refs_sum(array(x, x + 1, x + 2, x + 3, length=4))


@func()
def comptime_slice_element() -> i32:
    # the array *is* compile-time: its elements are their own places, the slice
    # names them, and indexing it is folded in Python
    syntax.comptime()
    v = array(Small(1, 2), Small(3, 4))
    s: Comptime[SlicePtr[Small]] = arr_slice(ref(v))
    return s.ptr[1].b


@func()
def comptime_slice_length() -> u64:
    syntax.comptime()
    v = array(Small(1, 2), Small(3, 4))
    s: Comptime[SlicePtr[Small]] = arr_slice(ref(v))
    return s.length


@func()
def runtime_index_into_comptime_storage(x: i32) -> i32:
    # a runtime index cannot pick a compile-time place
    syntax.comptime()
    v = array(Small(1, 2), Small(3, 4))
    syntax.comptime()
    s = arr_slice(ref(v))
    return s.ptr[x].a


# ---------------------------------------------------------------------------
# type expressions as values: a ``syntax`` type marker (``Ptr[T]``,
# ``Array[T, N]``, ``Option[T]``, ...) written in a body builds the type value
# (``hir.PointerType``/``hir.ArrayType``/``hir.OptionType``), which a
# ``spy.typeof`` comparison can name; ``syntax.ptr_cast`` reinterprets a
# pointer as another pointer type (``hir.PtrCast``)
# ---------------------------------------------------------------------------


@func()
def ptr_type_is_a_value(x: i32) -> i32:
    # the pointer type marker written in the body builds ``Ptr[i32]``
    return 1 if spy_typeof(ref(x)) == Ptr[i32] else 0


@func()
def multi_ptr_type_is_a_value(p: MultiPtr[i32]) -> i32:
    return 1 if spy_typeof(p) == MultiPtr[i32] else 0


@func()
def const_ptr_type_is_a_value(p: ConstPtr[i32]) -> i32:
    return 1 if spy_typeof(p) == ConstPtr[i32] else 0


@func()
def array_type_is_a_value(a: Array[i32, Literal[2]]) -> i32:
    # the length of an ``Array`` written as a value is a plain integer
    return 1 if spy_typeof(a) == Array[i32, 2] else 0


@func()
def option_type_is_a_value(o: Option[i32]) -> i32:
    return 1 if spy_typeof(o) == Option[i32] else 0


@func()
def call_const_ptr_type_is_a_value(x: i32) -> i32:
    # a ``Ptr`` implicitly converts to a ``ConstPtr``
    return const_ptr_type_is_a_value(ref(x))


@func()
def call_multi_ptr_type_is_a_value(x: i32) -> i32:
    a = array(x, x + 1)
    return multi_ptr_type_is_a_value(arr_slice(ref(a)).ptr)


@func()
def call_array_type_is_a_value(x: i32) -> i32:
    return array_type_is_a_value(array(x, x + 1, length=2))


@func()
def call_option_type_is_a_value(x: i32) -> i32:
    return option_type_is_a_value(x)


@func()
def ptr_cast_of_the_same_type(x: i32) -> i32:
    # ``ptr_cast`` to the very same pointer type is a no-op
    p = syntax.ptr_cast(ref(x), Ptr[i32])
    p[...] = p[...] + 1
    return x


@func()
def ptr_cast_to_another_pointee(x: i32) -> i32:
    # ``ptr_cast`` reinterprets the address: a ``Ptr[i32]`` viewed as the
    # ``MultiPtr[i32]`` of its element reads the same storage
    p = syntax.ptr_cast(ref(x), MultiPtr[i32])
    return p[0]


@func()
def ptr_cast_of_a_multi_pointer(p: MultiPtr[i32]) -> i32:
    # casting a multi pointer to a single one is allowed (the reverse is not an
    # implicit conversion, but ``ptr_cast`` is a reinterpretation)
    q = syntax.ptr_cast(p, Ptr[i32])
    return q[...]


@func()
def call_ptr_cast_of_a_multi_pointer(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    return ptr_cast_of_a_multi_pointer(arr_slice(ref(a)).ptr)


# ---------------------------------------------------------------------------
# aggregate arguments: an aggregate is only ever its *own* type - a struct and
# an array are a subtype of themselves and of nothing else - and neither
# calling convention (by value, or by reference for one past the by-value
# limit) may hide a mismatch: an address of one type handed over for another
# would alias the argument, the MIR pointers being untyped
# ---------------------------------------------------------------------------


@struct()
class FourI32:
    a: i32
    b: i32
    c: i32
    d: i32


@struct()
class FourI64:
    a: i64
    b: i64
    c: i64
    d: i64


@func()
def take_two_i64(t: TwoI64) -> i64:
    return t.a + t.b


@func()
def pass_two_i64(x: i32) -> i64:
    return take_two_i64(TwoI64(x, x + 1))


@func()
def pass_a_small_for_two_i64(x: i32) -> i64:
    # 8 bytes where 16 are taken: the value is read out and does not convert
    return take_two_i64(Small(x, x + 1))  # pyright: ignore[reportArgumentType]


@func()
def take_four_i64(f: FourI64) -> i64:
    return f.a + f.b + f.c + f.d


@func()
def pass_four_i64(x: i32) -> i64:
    return take_four_i64(FourI64(x, x + 1, x + 2, x + 3))


@func()
def pass_four_i64_through_a_name(x: i32) -> i64:
    # a name is an address already: it is passed as the address the parameter
    # takes (no copy)
    f = FourI64(x, x + 1, x + 2, x + 3)
    return take_four_i64(f)


@func()
def pass_four_i32_for_four_i64(x: i32) -> i64:
    # 32 bytes: the parameter is passed by reference and the argument is an
    # in-place construction, so the address of a *FourI32* would be handed over
    # for a *FourI64*
    return take_four_i64(FourI32(x, x + 1, x + 2, x + 3))  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# arrays: ``syntax.array(a1, a2, ...)`` builds an array of the elements it is
# given - in place, like a struct construction: the length of the array is the
# number of elements and its element type either the one the place it is built
# in declares or the common type of the elements - and ``a[i]`` is the *place*
# of the i-th element, read and written through like a field.  The spy type of
# an array is ``syntax.Array[T, N]``: an annotation writes the length out
# (``array`` cannot, see ``syntax``), the compiler takes it from the arguments.
# ---------------------------------------------------------------------------


@struct()
class Vec2:
    """A struct that holds an array: ``data`` is a field of the struct, and
    the elements of the array are elements of that field."""

    data: Array[i32, Literal[2]]

    def total(self) -> i32:
        return self.data[0] + self.data[1]


@func()
def read_element(x: i32) -> i32:
    a = array(x, 2)
    return a[0]


@func()
def sum_elements(x: i32) -> i32:
    a = array(x, x + 1)
    return a[0] + a[1]


@func()
def write_element(x: i32, v: i32) -> i32:
    a = array(x, x)
    a[1] = v
    return a[1]


@func()
def add_to_element(x: i32) -> i32:
    a = array(x, x + 1)
    a[0] += 10
    return a[0] * 100 + a[1]


@func()
def array_of_structs(x: i32) -> i32:
    a = array(Small(x, 1), Small(x, 2))
    return a[0].a + a[1].b


@func()
def write_struct_element(x: i32) -> i32:
    a = array(Small(x, 1), Small(x, 2))
    a[1].b = 7
    return a[1].b


@func()
def nested_array(x: i32) -> i32:
    # an array of arrays: each inner construction fills an element of the
    # outer array in place, and an element of an element is a place like any
    # other
    a = array(array(x, x + 1), array(x + 2, x + 3))
    return a[0][1] + a[1][0]


@func()
def array_through_ptr(x: i32) -> i32:
    # the address of an array is a pointer: ``p[...][i]`` is the i-th element
    # of the array ``p`` points at
    a = array(x, x + 1)
    p = ref(a)
    return p[...][0] + p[...][1]


@func()
def copy_array(x: i32) -> i32:
    a = array(x, x + 1)
    b = a
    return b[0] + b[1]


@func()
def pass_array(a: Array[i32, Literal[2]], i: i32) -> i32:
    return a[i]


@func()
def call_pass_array(x: i32) -> i32:
    # ``length`` is what the Python type checker needs to see the length of the
    # array (``array`` cannot tell it from the elements); the compiler takes it
    # from the number of elements either way
    return pass_array(array(x, x + 1, length=2), 1)


@func()
def call_pass_array_runtime_index(x: i32, i: i32) -> i32:
    # a runtime index is a runtime address (the array's own type is what the
    # address arithmetic uses)
    return pass_array(array(x, x + 1, length=2), i)


@func()
def return_array(x: i32) -> Array[i32, Literal[2]]:
    return array(x, 3, length=2)


@func()
def use_returned_array(x: i32) -> i32:
    b = return_array(x)
    return b[0] + b[1]


@func()
def wide_array(x: i64) -> i64:
    # 32 bytes: an array past the by-value limit, so one is returned (and
    # passed) through a result pointer
    a = array(x, x + 1, x + 2, x + 3)
    return a[0] + a[3]


@func()
def array_in_struct(x: i32) -> i32:
    v = Vec2(array(x, x + 1, length=2))
    return v.total() + v.data[0]


@func()
def write_struct_array(x: i32) -> i32:
    v = Vec2(array(x, x + 1, length=2))
    v.data[1] = x
    return v.total()


@func()
def length_keyword(x: i32) -> i32:
    # ``length`` is what the *Python* type checker needs (``array`` cannot tell
    # the length of the array from its elements); the compiler takes it from
    # the number of elements and ignores the keyword
    a = array(x, x + 1, length=2)
    return a[0] + a[1]


@func()
def discard_array(x: i32) -> i32:
    # an expression statement: the array is built into a temporary nobody reads
    array(x, x + 1)
    return x


@func()
def zero_sized_array(x: i32) -> i32:
    # an array of zero-sized elements holds no storage: it is written like any
    # other (the writes go nowhere), and every element equals the unit value of
    # the element type
    a = array(Blank(None), Blank(None))
    a[1] = Blank(None)
    return x if spy_typeof(a[0]) == Blank else x + 1


@func()
def empty_array_result() -> Array[i32, Literal[0]]:
    # a zero-length array is zero-sized whatever its element type is
    return array()


@func()
def pass_empty_array(a: Array[i32, Literal[0]]) -> i32:
    return 4


@func()
def use_empty_array() -> i32:
    return pass_empty_array(empty_array_result())


@func()
def element_out_of_bounds(x: i32) -> i32:
    a = array(x, x)
    return a[2]


@func()
def wrong_element_count(x: i32) -> Array[i32, Literal[2]]:
    # two elements are declared, one is given: the compiler counts the
    # arguments, not the keyword
    return array(x, length=2)


@func()
def untyped_elements(x: i32) -> i32:
    a = array(1, 2)
    return a[0] + x


@func()
def unknown_array_keyword(x: i32) -> i32:
    a = array(x, nope=1)  # pyright: ignore
    return a[0] + x


# ---------------------------------------------------------------------------
# if-expressions: ``a if c else b`` writes the branch it takes into the
# result location of the expression (a runtime branch of the MIR, or nothing
# at all when the condition is a compile-time value)
# ---------------------------------------------------------------------------


@func()
def choose_value(c: spy_bool, a: i32, b: i32) -> i32:
    return a if c else b


@func()
def choose_local(c: spy_bool, a: i32, b: i32) -> i32:
    x = a if c else b
    return x


@func()
def choose_nested(c1: spy_bool, c2: spy_bool, a: i32, b: i32, d: i32) -> i32:
    return a if c1 else (b if c2 else d)


@func()
def choose_comparison(a: i32, b: i32) -> i32:
    # a comparison as the condition
    return a if a > 0 else b


@func()
def choose_argument(c: spy_bool, a: i32, b: i32) -> i32:
    return inc(a if c else b)


@func()
def choose_arithmetic(c: spy_bool, a: i32, b: i32) -> i32:
    return (a if c else b) + 10


@func()
def choose_call_test(a: i32, b: i32, x: i32) -> i32:
    # the condition is a call of another spy function
    return a if le(x, 0) else b


@func()
def choose_comptime(a: i32, b: i32, c: spy_bool) -> i32:
    # a compile-time condition folds the expression: the dead else branch (a
    # nested if-expression) is never typed
    return a if spy_typeof(a) == i32 else (b if c else a)


@func()
def choose_comptime_else(a: i32, b: i32, c: spy_bool) -> i32:
    # ... and the branch it chooses may be the else one
    return a if spy_typeof(a) == i64 else (b if c else a)


def pick_inline(c: bool, a: i32, b: i32) -> i32:
    # an undecorated plain function: its body (with the if-expression) is
    # inlined at its call sites
    return a if c else b


@func()
def call_pick_inline(c: spy_bool, a: i32, b: i32) -> i32:
    return pick_inline(c, a, b)


@func()
def choose_struct(c: spy_bool, a: i32, b: i32) -> i32:
    # both branches construct a struct in place, in the same slot
    s = Small(a, b) if c else Small(b, b)
    return s.total()


@func()
def choose_large(c: spy_bool, x: i64) -> Large:
    # a large struct is returned through a result pointer: the branches write
    # into the caller's result location
    return Large(x, 1, 2, 3) if c else Large(x, 4, 5, 6)


@func()
def use_choose_large(c: spy_bool, x: i64) -> i64:
    return choose_large(c, x).a + choose_large(c, x).d


@func()
def choose_pointer(c: spy_bool, a: i32, b: i32) -> i32:
    p = ref(a) if c else ref(b)
    return p[...]


@func()
def choose_deref_write(c: spy_bool, a: i32, b: i32) -> i32:
    v = a
    p = ref(v)
    p[...] = b if c else a
    return v


@func()
def choose_non_bool_condition(a: i32, b: i32) -> i32:
    # spy has no truthiness: the condition has to be a bool
    return a if a else b


# ---------------------------------------------------------------------------
# compile-time structs: a struct built in an inline slot - an expression
# temporary or a ``Comptime`` variable - is an aggregate whose fields are their
# own compile-time places, so a field is read and written at compile time: a
# field assignment is folded in Python, a field read may condition a
# compile-time loop marked with ``syntax.unroll()``, and a whole aggregate is
# copied field by field (see ``interp.ComptimeAggregatePtr``)
# ---------------------------------------------------------------------------


@struct()
class Toggle:
    on: bool
    n: i32


@func()
def comptime_struct_field_read(x: i32) -> i32:
    s: Comptime = Small(1, 2)
    s.b = s.a
    return s.b + x


@func()
def comptime_struct_with_a_runtime_field(x: i32) -> i32:
    # a field that holds a runtime value is a runtime place of its own: the
    # other field is still compile-time
    s: Comptime = Small(x, 2)
    return s.a + s.b


@func()
def comptime_struct_declared_type() -> i32:
    # a declared compile-time type builds the aggregate in place
    s: Comptime[Small] = Small(4, 5)
    return s.b


@func()
def comptime_struct_reassigned() -> i32:
    # a second construction writes the fields of the aggregate already there
    s: Comptime[Small] = Small(4, 5)
    s = Small(6, 7)
    return s.b


@func()
def comptime_struct_copied() -> i32:
    # an assignment copies the value: writing the copy leaves the source alone
    a: Comptime = Toggle(True, 1)
    b: Comptime = a
    b.on = False
    return 1 if a.on else 0


@func()
def comptime_struct_zst_field() -> i32:
    h: Comptime = Holder(None, 7)
    return h.n


@func()
def comptime_struct_nested() -> i32:
    # a field of struct type is an aggregate of its own, built in place
    h: Comptime = StructHolder(Pair[i32](1, 2), 3)
    a: i32 = h.p.a
    b: i32 = h.p.b
    c: i32 = h.extra
    return a * 100 + b * 10 + c


@func()
def comptime_struct_ref_aliases() -> i32:
    # the compile-time variable holds the aggregate itself, so the pointer is
    # the aggregate: writing through it writes the aggregate
    s: Comptime = Small(1, 2)
    p: Comptime = ref(s)
    p[...].b = 5
    return s.b


@func()
def comptime_struct_ref_in_memory() -> i32:
    # ... a plain local has no compile-time storage for the pointer: the
    # aggregate is materialized and the local points at that copy (an aggregate
    # has no address of its own)
    s: Comptime = Small(1, 2)
    p = ref(s)
    p[...].b = 5
    return s.b


@func()
def comptime_struct_method() -> i32:
    # a native method takes the aggregate's address: the value is materialized
    return Pair[i32](1, 2).total()


@func()
def comptime_struct_argument() -> i32:
    return sum_small(Small(1, 2))


@func()
def comptime_struct_choose(c: spy_bool) -> i32:
    # both branches build into one storage: the second reuses the field places
    # the first recorded (see ``finish_struct``)
    s: Comptime = Small(1, 1) if c else Small(2, 2)
    return s.b


@func()
def comptime_struct_unroll() -> i32:
    # the field is a compile-time value, so it may condition a compile-time
    # loop: the body runs once, then the condition turns false
    s: Comptime = Toggle(True, 0)
    total: i32 = 0
    syntax.unroll()
    while s.on:
        s.on = False
        total = total + 1
    return total


@func()
def runtime_struct_unroll() -> i32:
    # ... a runtime struct's field is a runtime value, so it cannot: the unroll
    # cap reports it instead of unrolling forever (see ``inline_bad_cond``)
    s = Toggle(True, 0)

    syntax.unroll()
    while s.on:
        s.on = False
    return s.n


@func()
def comptime_struct_nested_copy() -> i32:
    # a whole-aggregate copy of a struct with a struct field: the field place is
    # an aggregate of its own (see ``init_inline_aggregate``)
    h: Comptime = StructHolder(Pair[i32](1, 2), 3)
    b: Comptime = h
    return b.p.a * 100 + b.p.b * 10 + b.extra


@func()
def comptime_struct_from_a_runtime_value(x: i32) -> i32:
    # a whole *runtime* struct value assigned to a compile-time variable: such a
    # variable holds its fields as places of their own (see
    # ``ComptimeAggregatePtr``), so the value is split into one store per field -
    # the runtime field lands in memory, a compile-time one in a box
    s = Small(x, 2)
    c: Comptime = s
    return c.a + c.b


@func()
def comptime_struct_declared_from_a_runtime_value(x: i32) -> i32:
    # ... the same into a declared compile-time variable, whose field places
    # already exist
    s = Small(x, 2)
    c: Comptime[Small] = s
    return c.a + c.b


@func()
def comptime_struct_reassigned_from_a_runtime_value(x: i32) -> i32:
    # ... and a whole runtime value assigned after a construction wrote the
    # field places
    c: Comptime = Small(x, 1)
    s = Small(x, 3)
    c = s
    return c.a + c.b


@func()
def comptime_struct_of_one_field_from_a_runtime_value(x: i64) -> i64:
    # a struct whose mirror *is* its one stored field (see ``mirror_is_a_field``):
    # a whole runtime value of it is still split into one place per field, and
    # that field is the value itself
    o = OuterTwo(TwoI64(x, 1))
    c: Comptime = o
    return c.inner.b


@func()
def comptime_struct_runtime_field_by_ref(x: i32) -> i32:
    # the runtime field of a compile-time aggregate is a runtime place of its
    # own, so its address can be handed to a native function (which writes
    # through it)
    s: Comptime = Small(x, 2)
    incr_ptr(ref(s.a))
    return s.a


@func()
def comptime_struct_field_written_in_branches(c: spy_bool, x: i32) -> i32:
    # the runtime field of a compile-time aggregate written in both branches of a
    # runtime ``if``: both runtime paths have to write the same place, so it is
    # memory - a compile-time box would keep only the value the walk wrote last
    s: Comptime = Small(x, 1)
    if c:
        s.a = 5
    else:
        s.a = 9
    return s.a + s.b


@func()
def comptime_struct_from_a_nested_runtime_value(x: i32) -> i32:
    # a nested aggregate field holding a runtime struct is split the same way,
    # recursively
    inner = Pair[i32](x, 2)
    h: Comptime = StructHolder(inner, 3)
    return h.p.a + h.p.b + h.extra


@func()
def comptime_struct_from_a_runtime_choice(c: spy_bool, x: i32) -> i32:
    # both branches assign a whole runtime value into one compile-time variable:
    # the field places are the ones the first store recorded, and each runtime
    # branch writes its own value into them (unlike the compile-time values of
    # ``comptime_struct_choose``, whose last write wins)
    a = Small(x, 2)
    b = Small(x, 3)
    s: Comptime = a if c else b
    return s.a + s.b


@func()
def comptime_struct_argument_from_a_variable(x: i32) -> i32:
    # a compile-time aggregate with a runtime field handed to a native function:
    # its fields are materialized into memory (see ``_materialize_aggregate``)
    s: Comptime = Small(x, 2)
    return sum_small(s)


@func()
def comptime_struct_field_comparison() -> spy_bool:
    # a field read is a compile-time constant, so an operator on it is folded
    # in Python
    s: Comptime = Small(1, 2)
    return s.a == 1


@func()
def comptime_struct_zst_in_a_comptime_local() -> spy_bool:
    # a zero-sized aggregate is an aggregate too: it is held by its places, not
    # by a compile-time box
    b: Comptime = Blank(None)
    return spy_typeof(b) == Blank


# ---------------------------------------------------------------------------
# compile-time arrays: an array built in an inline slot - an expression
# temporary or a ``Comptime`` variable - is an aggregate (like a struct) whose
# elements are their own compile-time places (see ``interp.ComptimeAggregatePtr``)
# ---------------------------------------------------------------------------


@func()
def comptime_array_element() -> i32:
    a: Comptime = array(1, 2)
    return a[1]


@func()
def comptime_array_copy() -> i32:
    a: Comptime = array(1, 2)
    b: Comptime = a
    return b[1]


@func()
def comptime_array_of_structures() -> i32:
    a: Comptime = array(Small(1, 2), Small(3, 4))
    b: Comptime = a
    return b[1].a * 10 + b[0].b


@func()
def comptime_nested_array() -> i32:
    a: Comptime = array(array(1, 2), array(3, 4))
    return a[1][0]


@func()
def comptime_array_with_a_runtime_element(x: i32) -> i32:
    # a runtime element is a runtime place of its own: the other element is
    # still compile-time
    a: Comptime = array(x, 2)
    return a[0] + a[1]


@func()
def comptime_array_from_a_runtime_value(x: i32) -> i32:
    # a whole runtime *array* value assigned to a compile-time variable: its
    # elements are read out of the value itself (``mir.ExtractValue``)
    a = array(x, x + 1)
    c: Comptime = a
    return c[1]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SpyFunctionCallTest(TestCase):
    """Basic function calls: arithmetic, generics, calls between spy
    functions, inlining of plain functions, compile-time dispatch and
    recursion."""

    def test_smoke(self) -> None:
        self.assertEqual(smoke_test(2, 3), 5)
        self.assertEqual(smoke_test(-4, 1), -3)

    def test_arithmetic(self) -> None:
        self.assertEqual(sub(7, 12), -5)
        self.assertEqual(mul(6, 7), 42)
        self.assertEqual(mod(17, 5), 2)

    def test_negative_literals(self) -> None:
        # a negative literal is a ``-`` applied to an untyped literal, not a
        # bare constant like ``3``: it is evaluated into an expression
        # temporary, which is a compile-time box holding the literal's own
        # (untyped) type - a value position, a comparison and an operand alike
        self.assertEqual(negative_literals(0), 3)
        self.assertEqual(negative_literals(5), -7)
        self.assertEqual(negative_literals(-20), -1)

    def test_generic_int(self) -> None:
        # a plain Python int marshals to the default signed 64-bit type
        # (see ``dsl._INT_LITERAL_BITS``)
        self.assertEqual(add(2, 3), 5)

    def test_generic_float(self) -> None:
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)

    def test_float_promotion(self) -> None:
        self.assertAlmostEqual(scale(3.0, 0.5), 2.5)

    def test_non_default_integer_type(self) -> None:
        # ``spy.as_`` binds an argument to an explicit spy type
        self.assertEqual(add_u64(spy_as(2**63 - 1, u64), spy_as(2, u64)), 2**63 + 1)

    def test_default_argument(self) -> None:
        self.assertEqual(add_default(41), 41)
        self.assertEqual(add_default(40, 2), 42)

    def test_call_spy_function(self) -> None:
        self.assertEqual(call_add(20, 22), 42)

    def test_inline_plain_function(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline(20, 22), 42)

    def test_inline_with_runtime_branches(self) -> None:
        # the inlined body has a runtime ``if`` whose branches both return
        # (the ``Block``/``Break`` shape of an inlined return)
        self.assertEqual(call_abs(-5), 5)
        self.assertEqual(call_abs(5), 5)
        # a branch returns, the other falls through to the trailing return
        self.assertEqual(call_clamp(42), 42)
        self.assertEqual(call_clamp(101), 100)
        # the inlined result is consumed by the caller's continuation
        self.assertEqual(call_abs_plus_one(-5), 6)

    def test_call_with_default(self) -> None:
        self.assertEqual(use_default(7), 7)

    def test_augmented_assignment(self) -> None:
        self.assertEqual(accumulate(5, 6), 11)

    def test_runtime_if_both_return(self) -> None:
        self.assertEqual(sign(5), 1)
        self.assertEqual(sign(-5), -1)

    def test_runtime_if_fallthrough(self) -> None:
        self.assertEqual(clamped(42), 42)
        self.assertEqual(clamped(101), 100)

    def test_runtime_if_assignment(self) -> None:
        # an assignment in a branch writes the variable the branch sees
        self.assertEqual(assign_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_in_both_branches(True, 1, 2), 2)
        self.assertEqual(assign_in_both_branches(False, 1, 2), 3)
        # ... a parameter included
        self.assertEqual(assign_parameter_in_branch(True, 1, 2), 2)
        self.assertEqual(assign_parameter_in_branch(False, 1, 2), 1)
        # a destructuring target written outside, one declared in the branch
        self.assertEqual(assign_tuple_in_branch(True, 1, 2), 3)
        self.assertEqual(assign_tuple_in_branch(False, 1, 2), 1)
        self.assertEqual(assign_from_choose(False, True, 1, 2), 2)
        self.assertEqual(assign_from_choose(True, True, 1, 2), 1)
        self.assertEqual(assign_from_choose(True, False, 1, 2), 1)

    def test_branch_declaration(self) -> None:
        # a name bound nowhere is declared in the branch it is assigned in,
        # and is invisible after it
        self.assertEqual(branch_declaration(True, 5), 6)
        self.assertEqual(branch_declaration(False, 5), 5)
        with self.assertRaises(CompileError):
            use_branch_declaration(True, 5)

    def test_comparisons(self) -> None:
        self.assertTrue(le(1, 2))
        self.assertFalse(le(2, 1))
        self.assertTrue(eq(3, 3))
        self.assertFalse(eq(3, 4))

    def test_comptime_typeof(self) -> None:
        # a plain Python int marshals to i64 (see ``dsl._INT_LITERAL_BITS``)
        self.assertFalse(is_i32(1))
        self.assertTrue(is_i32(spy_as(1, i32)))
        self.assertFalse(is_i32(1.0))
        self.assertTrue(is_u64(spy_as(1, u64)))

    def test_void_function(self) -> None:
        self.assertIsNone(nothing(3))

    def test_recursion(self) -> None:
        self.assertEqual(fact(5), 120)
        self.assertEqual(fact(0), 1)
        self.assertEqual(fact(10), 3628800)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(is_even(10))
        self.assertFalse(is_even(7))
        self.assertTrue(is_odd(7))

    def test_cross_module_call(self) -> None:
        # ``inc`` is compiled on its own first, so the compilation of
        # ``twice_inc`` references it as an external symbol of an earlier
        # module (the modular-compilation path)
        self.assertEqual(inc(10), 11)
        self.assertEqual(twice_inc(10), 12)

    def test_f32(self) -> None:
        self.assertAlmostEqual(id_f32(spy_as(0.5, f32)), 0.5)

    def test_specialization_is_cached(self) -> None:
        # repeated calls reuse the compiled specialization
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(smoke_test(1, 1), 2)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)

    def test_compile_error_on_unsupported_expression(self) -> None:
        with self.assertRaises(CompileError):
            div(1, 2)

    def test_untyped_integer_slot_is_rejected(self) -> None:
        # an untyped integer literal has no runtime type of its own: a
        # slot that has to live in memory must declare its type
        with self.assertRaises(CompileError):
            untyped_local(5)
        with self.assertRaises(CompileError):
            untyped_return()
        # the same holds for the slot of an unannotated parameter, which
        # is typed by the call (here: by an untyped literal argument)
        with self.assertRaises(CompileError):
            call_untyped_param()


@func()
def div(a: i32, b: i32) -> i32:
    return a / b # pyright: ignore[reportReturnType]


class SpyWhileTest(TestCase):
    """``while`` loops: the condition is re-evaluated at the head of every
    iteration, ``break`` leaves the loop, ``continue`` starts the next
    iteration and the ``else`` clause runs only on a natural exit."""

    def test_counting_loop(self) -> None:
        self.assertEqual(sum_to(0), 0)
        self.assertEqual(sum_to(1), 0)
        self.assertEqual(sum_to(5), 10)
        self.assertEqual(count_up(4), 4)

    def test_call_inside_the_body(self) -> None:
        self.assertEqual(sum_calls(4), 10)
        self.assertEqual(sum_calls(0), 0)

    def test_break(self) -> None:
        self.assertEqual(find_divisor(15), 3)
        self.assertEqual(find_divisor(9), 3)
        # a prime has no divisor below it: the loop exits normally with i == n
        self.assertEqual(find_divisor(7), 7)
        self.assertEqual(skip_and_stop(10), 8)
        self.assertEqual(skip_and_stop(1), 1)

    def test_continue(self) -> None:
        self.assertEqual(sum_odd(5), 9)
        self.assertEqual(sum_odd(0), 0)

    def test_while_else(self) -> None:
        # the else clause runs on a natural exit ...
        self.assertEqual(while_else_natural(3), 103)
        # ... and is skipped by a ``break``
        self.assertEqual(while_else_break(5), 2)
        self.assertEqual(while_else_break(1), 101)
        # ... while a ``continue`` only delays it
        self.assertEqual(while_else_continue(4), 104)

    def test_nested_loops(self) -> None:
        self.assertEqual(nested_loop_sum(3, 4), 18)
        self.assertEqual(nested_loop_sum(0, 4), 0)

    def test_comptime_condition(self) -> None:
        self.assertEqual(comptime_false_loop(5), 1)

    def test_loop_whose_only_exit_is_return(self) -> None:
        self.assertEqual(loop_return(4), 4)
        self.assertEqual(loop_return(0), 0)

    def test_loop_in_a_void_function(self) -> None:
        self.assertIsNone(loop_void(3))

    def test_assignment_across_iterations(self) -> None:
        self.assertEqual(assign_outside_from_loop(5), 10)

    def test_break_and_continue_in_try(self) -> None:
        # a ``break``/``continue`` in a try body leaves the loop directly; the
        # clause is only reachable when the body raises first
        self.assertEqual(break_in_try(3), 3)
        self.assertEqual(continue_in_try(3), 52)

    def test_break_out_of_while_true(self) -> None:
        self.assertEqual(bounded_loop(3), 103)
        self.assertEqual(bounded_loop(0), 100)

    def test_loop_in_an_inlined_body(self) -> None:
        self.assertEqual(call_sum_inline(4), 6)


class SpyInlineLoopTest(TestCase):
    """Compile-time loops: a loop preceded by ``syntax.unroll()`` unrolls its
    body once per compile-time iteration, ``break`` leaves the whole unrolled
    sequence and ``continue`` jumps to the next unrolled body."""

    def test_unrolled(self) -> None:
        # the body is emitted once per iteration, not looped over at runtime
        self.assertEqual(inline_sum(), 6)

    def test_comptime_false(self) -> None:
        # no body is unrolled; the else clause runs once
        self.assertEqual(inline_false(), 2)

    def test_break_skips_the_remaining_bodies(self) -> None:
        # i == n under a runtime condition: the break leaves every remaining
        # unrolled body, so only 1 + ... + (n - 1) is added
        self.assertEqual(inline_break(3), 3)
        # a break never taken unwinds the whole loop: 1 + ... + 10
        self.assertEqual(inline_break(0), 55)

    def test_continue_jumps_to_the_next_body(self) -> None:
        # 1 + ... + 6, minus the skipped n
        self.assertEqual(inline_continue(3), 18)
        self.assertEqual(inline_continue(0), 21)

    def test_while_else(self) -> None:
        self.assertEqual(inline_else(), 106)

    def test_nested_unrolled_loops(self) -> None:
        self.assertEqual(inline_nested(), 6)

    def test_unconditional_continue(self) -> None:
        # the unroll runs through the continue's jump, with no falling end
        self.assertEqual(inline_continue_always(), 4)

    def test_continue_with_a_return_sibling(self) -> None:
        # the continue's if has a returning sibling, so the whole iteration ends
        # without falling through: the unroll still runs from the continue
        self.assertEqual(inline_continue_or_return(2), 1)
        self.assertEqual(inline_continue_or_return(0), 112)

    def test_inline_loop_inside_a_runtime_loop(self) -> None:
        self.assertEqual(runtime_loop_with_inline(2), 6)
        self.assertEqual(runtime_loop_with_inline(0), 0)

    def test_runtime_condition_is_reported(self) -> None:
        # the condition is not a compile-time value, so the loop cannot be
        # unrolled: the unroll cap reports it instead of looping forever
        with self.assertRaises(CompileError):
            inline_bad_cond(3)

    def test_marker_outside_a_while_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            inline_loop_misuse(3)


class SpyUnrollForTest(TestCase):
    """Compile-time ``for`` loops: the iterable of a loop marked with
    ``syntax.unroll()`` is a compile-time value (the loop variable may have no
    runtime representation at all), so the desugared iterator loop unrolls at
    compile time; ``break`` leaves the whole unrolled sequence and ``continue``
    jumps to the next unrolled body."""

    def test_unrolled(self) -> None:
        self.assertEqual(unroll_for_sum(), 0 + 1 + 2 + 3)

    def test_defaulted_fields(self) -> None:
        self.assertEqual(unroll_for_defaults(), 0 + 1 + 2 + 3)

    def test_else(self) -> None:
        self.assertEqual(unroll_for_else(), 0 + 1 + 2 + 100)

    def test_break_skips_the_remaining_bodies_and_the_else(self) -> None:
        self.assertEqual(unroll_for_break(2), 0 + 1)
        # a break never taken runs the whole loop and the else
        self.assertEqual(unroll_for_break(100), 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 100)

    def test_continue_jumps_to_the_next_body(self) -> None:
        self.assertEqual(unroll_for_continue(), 0 + 1 + 3 + 4)

    def test_nested_unrolled_loops(self) -> None:
        self.assertEqual(unroll_for_nested(), 1 + 21 + 41)

    def test_a_compile_time_iterator_variable(self) -> None:
        self.assertEqual(unroll_for_typed_var(), 0 + 1 + 2)

    def test_a_runtime_iterator_is_reported(self) -> None:
        # the loop only ends when a ``StopIteration`` reaches the except clause
        # at compile time, which a runtime iterator never does: the unroll cap
        # reports it instead of unrolling forever
        with self.assertRaises(CompileError):
            unroll_for_runtime_iter(3)


class SpyForTest(TestCase):
    """``for`` loops: the iterable is iterated with ``__iter__``/``__next__``
    (``range`` names ``std.range``), a ``break`` leaves the loop, a
    ``continue`` starts the next iteration and the ``else`` clause runs only
    when the sequence is exhausted."""

    def test_sum(self) -> None:
        self.assertEqual(for_sum(0), 0)
        self.assertEqual(for_sum(5), 10)

    def test_else_runs_when_exhausted(self) -> None:
        self.assertEqual(for_else(3), 3 + 100)

    def test_break_skips_the_else(self) -> None:
        self.assertEqual(for_break(6), 1 + 2)
        # a break never taken runs the whole loop and the else
        self.assertEqual(for_break(2), 1 + 100)

    def test_continue(self) -> None:
        self.assertEqual(for_continue(5), 0 + 1 + 3 + 4)

    def test_start_and_step(self) -> None:
        self.assertEqual(for_step(7), 1 + 3 + 5)

    def test_defaulted_start_and_step(self) -> None:
        # ``range(n)`` leaves ``start`` and ``step`` to their defaults
        self.assertEqual(for_defaults(4), 0 + 1 + 2 + 3)
        self.assertEqual(for_defaults(0), 0)

    def test_nested(self) -> None:
        # the inner loop runs ``range(i, 0, 1)`` = ``0 .. i - 1``
        self.assertEqual(for_nested(4), 0 + 0 + (0 + 1) + (0 + 1 + 2))

    def test_call_in_the_body(self) -> None:
        self.assertEqual(for_call(4), (0 + 1) + (1 + 1) + (2 + 1) + (3 + 1))

    def test_loop_variable_is_not_visible_after_the_loop(self) -> None:
        with self.assertRaises(CompileError):
            for_leaks(3)

    def test_iter_returns_a_copy(self) -> None:
        # ``__iter__`` returns the range by value, so the loop iterates a copy
        self.assertEqual(for_iter_is_a_copy(3), 0)


class SpyMethodSelfTest(TestCase):
    """A method's ``self`` is bound directly to the incoming pointer (its
    ``FunctionIR.arg_is_ref``), so reading ``self`` implicitly loads the
    receiver and ``ref(self)`` is a ``Ptr[Self]``."""

    def test_ref_of_self_is_a_pointer_to_the_receiver(self) -> None:
        self.assertEqual(write_through_self_ref(1), 99)


class SpyAnnotationTest(TestCase):
    """Annotated local variables: ``name: T`` declares the type of a fresh
    variable and its slot is materialized right away, while ``name:
    Comptime``/``name: Comptime[T]`` declare it compile-time - a box rather
    than memory, which is what lets it hold a compile-time-only value such as
    a type."""

    def test_typed_local(self) -> None:
        self.assertEqual(annotated_local(3), 8)

    def test_declared_type_coerces_the_value(self) -> None:
        # the slot has the declared type, so the value written into it is
        # converted to that type
        self.assertEqual(annotated_local_coerced(3), 4)

    def test_typed_struct_locals(self) -> None:
        self.assertEqual(annotated_struct_local(3), 5)
        self.assertEqual(annotated_plain_struct_local(3), 5)

    def test_type_parameter_locals(self) -> None:
        # a local may be typed by a type parameter of the function: the
        # call solves it
        self.assertEqual(call_annotated_generic_local(7), 7)

    def test_comptime_local_holds_a_type(self) -> None:
        self.assertTrue(comptime_local_holds_a_type(1))

    def test_comptime_typed_local(self) -> None:
        self.assertEqual(comptime_typed_local(3), 4)

    def test_comptime_local_holds_a_tuple(self) -> None:
        self.assertEqual(comptime_local_holds_a_tuple(3), 304)

    def test_compile_time_parameter(self) -> None:
        self.assertTrue(annotated_comptime_param(None))

    def test_compile_time_parameter_from_spy_code(self) -> None:
        self.assertTrue(passes_comptime_value(3))

    def test_runtime_argument_to_a_compile_time_parameter_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            passes_runtime_value_to_comptime(3)

    def test_redeclaration_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_redeclare(3)


class SpyComptimeMarkerTest(TestCase):
    """``syntax.comptime()``: the statement marker that declares the variable
    of the declaration right after it as a compile-time one, the way
    ``syntax.unroll()`` marks the loop that follows it.  It stands for the
    ``Comptime`` annotation: ``syntax.comptime()`` + ``a: T = e`` is
    ``a: Comptime[T] = e``, and with no annotation ``a: Comptime = e``."""

    def test_a_typed_declaration(self) -> None:
        # ``syntax.comptime()`` + ``i: i32 = 0`` is ``i: Comptime[i32] = 0``
        self.assertEqual(comptime_marker_typed(), 0 + 1 + 2 + 3)

    def test_an_unannotated_declaration(self) -> None:
        # ... and with no annotation it is ``n: Comptime = 4``
        self.assertEqual(comptime_marker_untyped(), 4 + 3 + 2 + 1)

    def test_it_holds_a_type(self) -> None:
        self.assertTrue(comptime_marker_holds_a_type(1))

    def test_a_compile_time_struct(self) -> None:
        self.assertEqual(comptime_marker_struct(5), 5)

    def test_it_may_hold_a_runtime_value(self) -> None:
        self.assertEqual(comptime_marker_holds_a_runtime_value(5), 6)

    def test_a_declaration_without_a_value(self) -> None:
        self.assertEqual(comptime_marker_without_a_value(), 6)

    def test_a_destructuring_declaration(self) -> None:
        self.assertEqual(comptime_marker_destructuring(), 1 * 10 + 2)

    def test_a_declaration_in_a_branch(self) -> None:
        self.assertEqual(comptime_marker_in_a_branch(True), 3)
        self.assertEqual(comptime_marker_in_a_branch(False), 0)

    def test_it_must_be_followed_by_a_declaration(self) -> None:
        with self.assertRaises(CompileError):
            comptime_marker_before_a_non_declaration()
        with self.assertRaises(CompileError):
            comptime_marker_at_the_end()

    def test_a_redundant_annotation_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_redundant()
        self.assertIn('already declared compile-time', str(ctx.exception))

    def test_it_marks_a_declaration_not_an_assignment(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_on_an_assignment()
        self.assertIn('declares a fresh variable', str(ctx.exception))

    def test_it_is_not_a_value(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            comptime_marker_as_a_value()
        self.assertIn('statement marker', str(ctx.exception))

    def test_it_must_come_immediately_before_the_declaration(self) -> None:
        with self.assertRaises(CompileError):
            comptime_marker_before_a_loop()


class SpyStructTest(TestCase):
    """Struct values: a construction fills the fields in place - the
    arguments bind the fields by declaration order and by name - the
    annotations of a spy function name a struct like any other type, and its
    methods are called on the object."""

    def test_fields_of_a_local(self) -> None:
        self.assertEqual(struct_local(5), 6)

    def test_all_fields_comptime(self) -> None:
        self.assertEqual(struct_all_comptime(), 3)

    def test_by_value_return(self) -> None:
        self.assertEqual(use_small(5), 7)

    def test_struct_annotation(self) -> None:
        self.assertEqual(use_sum_small(5), 8)

    def test_result_pointer_return(self) -> None:
        self.assertEqual(use_large(5, True), 8)
        self.assertEqual(use_large(5, False), 11)

    def test_plain_method(self) -> None:
        self.assertEqual(total_small(5), 6)

    def test_construction_and_methods(self) -> None:
        # ``Counter(1)`` fills the field ``n`` and ``bump`` mutates the
        # object through ``self``: 1 + 2 = 3, doubled is 6
        self.assertEqual(bump_counter(1, 2), 6)

    def test_one_field_mirror(self) -> None:
        # the field of such a struct is the struct itself: the slot of the
        # local is the storage of the field, with no address arithmetic
        self.assertEqual(one_field_local(5), 5)

    def test_reordered_fields(self) -> None:
        self.assertEqual(mixed_fields_local(5, 3), 8)

    def test_extern_c_field(self) -> None:
        self.assertEqual(extern_one_field_local(5), 5)

    def test_zero_sized_field(self) -> None:
        self.assertEqual(zst_field_local(5), 5)

    def test_nested_field(self) -> None:
        self.assertEqual(nested_field_local(5), 5)

    def test_method_on_a_field(self) -> None:
        self.assertEqual(nested_method_local(5), 5)

    def test_nested_struct_field(self) -> None:
        # a struct of one struct field whose mirror is itself a struct type:
        # the field is still at the address of the value itself
        self.assertEqual(nested_struct_field(5), 5)


class SpyStructDefaultsTest(TestCase):
    """Struct field defaults: a construction may leave out a field the class
    body gave a value, and the field takes it (coerced to its type), exactly
    like a provided argument.  A field with no default still has to be given a
    value."""

    def test_a_left_out_field_takes_its_default(self) -> None:
        self.assertEqual(struct_defaults(1), 1 + 7 + 9)

    def test_a_provided_field_overrides_its_default(self) -> None:
        self.assertEqual(struct_defaults_partial(1), 1 + 100 + 9)

    def test_defaults_in_a_compile_time_aggregate(self) -> None:
        self.assertEqual(struct_defaults_comptime(1), 1 + 1 + 9)

    def test_a_zero_sized_default_stores_nothing(self) -> None:
        self.assertEqual(struct_defaults_zst(), 3)

    def test_a_default_is_coerced_to_the_field_type(self) -> None:
        # the default is the untyped literal ``0``, written into an ``i32`` field
        self.assertEqual(struct_defaults_generic(5), 5)

    def test_a_field_without_a_default_may_not_be_left_out(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            struct_missing_a_value()
        self.assertIn("missing a value for field 'x'", str(ctx.exception))


class SpyComptimeStructTest(TestCase):
    """Compile-time structs: a struct built in an inline slot - an expression
    temporary or a ``Comptime`` variable - is an aggregate whose fields are
    their own compile-time places, so a field read or write is folded in
    Python (a field value may condition a compile-time loop marked with
    ``syntax.unroll()``)."""

    def test_field_read(self) -> None:
        self.assertEqual(comptime_struct_field_read(0), 1)

    def test_runtime_field(self) -> None:
        self.assertEqual(comptime_struct_with_a_runtime_field(5), 7)

    def test_declared_comptime_type(self) -> None:
        self.assertEqual(comptime_struct_declared_type(), 5)

    def test_reassigned(self) -> None:
        self.assertEqual(comptime_struct_reassigned(), 7)

    def test_copy_is_a_value(self) -> None:
        self.assertEqual(comptime_struct_copied(), 1)

    def test_zero_sized_field(self) -> None:
        self.assertEqual(comptime_struct_zst_field(), 7)

    def test_nested_aggregate(self) -> None:
        self.assertEqual(comptime_struct_nested(), 123)

    def test_native_method(self) -> None:
        self.assertEqual(comptime_struct_method(), 3)

    def test_a_pointer_to_an_aggregate_aliases_it(self) -> None:
        self.assertEqual(comptime_struct_ref_aliases(), 5)

    def test_a_pointer_in_memory_points_at_a_copy(self) -> None:
        self.assertEqual(comptime_struct_ref_in_memory(), 2)

    def test_native_argument(self) -> None:
        self.assertEqual(comptime_struct_argument(), 3)

    def test_field_conditions_a_compile_time_loop(self) -> None:
        self.assertEqual(comptime_struct_unroll(), 1)

    def test_a_runtime_branch_writes_the_aggregate(self) -> None:
        # both branches of a runtime ``if`` are typed, so the construction
        # writes the same places twice and the last write wins - exactly like a
        # scalar compile-time local (``x: Comptime = 1 if c else 2`` is always
        # 2): a compile-time location holds one value, and no store knows the
        # runtime condition
        self.assertEqual(comptime_struct_choose(True), 2)
        self.assertEqual(comptime_struct_choose(False), 2)

    def test_nested_aggregate_copy(self) -> None:
        self.assertEqual(comptime_struct_nested_copy(), 123)

    def test_a_whole_runtime_value_into_a_comptime_variable(self) -> None:
        # the value is split into one store per field, so the variable is still
        # an aggregate of places rather than a struct in memory
        self.assertEqual(comptime_struct_from_a_runtime_value(5), 7)

    def test_a_whole_runtime_value_into_a_declared_comptime_variable(self) -> None:
        self.assertEqual(comptime_struct_declared_from_a_runtime_value(5), 7)

    def test_a_whole_runtime_value_after_a_construction(self) -> None:
        self.assertEqual(comptime_struct_reassigned_from_a_runtime_value(5), 8)

    def test_a_runtime_field_of_such_a_variable_is_addressable(self) -> None:
        # the runtime field is memory, not a box: a native callee writes it
        self.assertEqual(comptime_struct_runtime_field_by_ref(5), 6)

    def test_a_runtime_field_written_in_branches(self) -> None:
        # both runtime paths write the same memory place, so each keeps its own
        # value (a box would have kept the one the walk wrote last)
        self.assertEqual(comptime_struct_field_written_in_branches(True, 3), 6)
        self.assertEqual(comptime_struct_field_written_in_branches(False, 3), 10)

    def test_a_nested_runtime_aggregate_field(self) -> None:
        self.assertEqual(comptime_struct_from_a_nested_runtime_value(5), 10)

    def test_a_runtime_choice_of_whole_values(self) -> None:
        # each branch writes its own whole value into the recorded field places,
        # so the choice is a runtime one here (the values are runtime values)
        self.assertEqual(comptime_struct_from_a_runtime_choice(True, 5), 7)
        self.assertEqual(comptime_struct_from_a_runtime_choice(False, 5), 8)

    def test_a_native_argument_from_a_comptime_variable(self) -> None:
        self.assertEqual(comptime_struct_argument_from_a_variable(5), 7)

    def test_field_comparison(self) -> None:
        self.assertTrue(comptime_struct_field_comparison())

    def test_zero_sized_aggregate_in_a_comptime_local(self) -> None:
        self.assertTrue(comptime_struct_zst_in_a_comptime_local())

    def test_a_struct_whose_mirror_is_its_field(self) -> None:
        # a struct of one stored field mirrors to that field itself: splitting a
        # whole runtime value of it still reads the field out of the value
        self.assertEqual(comptime_struct_of_one_field_from_a_runtime_value(4), 1)


class SpyComptimeArrayTest(TestCase):
    """Compile-time arrays: an array built in an inline slot - an expression
    temporary or a ``Comptime`` variable - is an aggregate whose elements are
    their own compile-time places, like a struct's fields."""

    def test_element_read(self) -> None:
        self.assertEqual(comptime_array_element(), 2)

    def test_copy_is_a_value(self) -> None:
        self.assertEqual(comptime_array_copy(), 2)

    def test_array_of_structures(self) -> None:
        self.assertEqual(comptime_array_of_structures(), 32)

    def test_nested_array(self) -> None:
        self.assertEqual(comptime_nested_array(), 3)

    def test_runtime_element(self) -> None:
        self.assertEqual(comptime_array_with_a_runtime_element(5), 7)

    def test_a_whole_runtime_value(self) -> None:
        # a whole runtime array value into a compile-time variable: the elements
        # are read out of the value itself
        self.assertEqual(comptime_array_from_a_runtime_value(3), 4)

    def test_a_runtime_struct_cannot_unroll(self) -> None:
        with self.assertRaises(CompileError):
            runtime_struct_unroll()


class SpyGenericStructTest(TestCase):
    """Generic structs: ``class Foo[T]`` declares a template, ``Foo[i32]``
    names a specialization, and a method call carries that specialization's
    type arguments into the method (``a.m()`` is ``typeof(a).m(a)``)."""

    def test_construction_and_method(self) -> None:
        self.assertEqual(generic_pair_total(2), 5)

    def test_inlined_method(self) -> None:
        self.assertEqual(generic_pair_plain_method(2), 4)

    def test_registered_method(self) -> None:
        self.assertEqual(generic_box_get(2), 2)

    def test_plain_method(self) -> None:
        self.assertEqual(generic_plain_box_get(2), 2)

    def test_method_with_its_own_type_param(self) -> None:
        # the method's own ``T`` shadows the struct's ``T``
        self.assertEqual(generic_method_with_own_type_param(5), 5)

    def test_method_uses_the_struct_type_param(self) -> None:
        # ``own`` returns the struct's ``T``: the result is an i64
        self.assertEqual(generic_method_uses_the_struct_type(5), 5)

    def test_generic_struct_is_returned(self) -> None:
        self.assertEqual(generic_struct_returned(2), 6)

    def test_generic_struct_argument(self) -> None:
        # the type parameter is solved from the generic struct argument
        self.assertEqual(generic_struct_argument(7), 7)

    def test_nested_generic_field(self) -> None:
        self.assertEqual(generic_nested_field(2), 5)

    def test_method_returns_a_generic_struct(self) -> None:
        self.assertEqual(generic_method_returns_a_struct(3), 6)

    def test_type_param_is_a_value(self) -> None:
        # ``T`` used as a value in a body is the type the call solved it to,
        # in a registered method, an inlined one and a generic function
        self.assertTrue(generic_struct_typeof_dispatch(3))
        self.assertTrue(generic_struct_typeof_inline(3))
        self.assertTrue(generic_typeof(3))

    def test_struct_type_is_compared(self) -> None:
        # a specialization is also nameable as a value inside a body
        self.assertTrue(generic_struct_type_compared(3))

    def test_generic_field_of_a_non_generic_struct(self) -> None:
        self.assertEqual(generic_field_of_nongeneric(2), 8)

    def test_construction_with_explicit_arguments(self) -> None:
        # a construction of a generic struct at a site whose specialization
        # is not known names it explicitly, positionally or by keyword
        self.assertEqual(explicit_generic_construction(2), 5)
        self.assertEqual(keyword_construction(2), 5)
        self.assertEqual(mixed_construction(2), 4)

    def test_result_location_inference(self) -> None:
        # ``make_pair_inferred`` returns ``Pair(a, b)`` without naming the
        # specialization: the return type of the function is declared, so the
        # result location's type decides it
        self.assertEqual(inferred_pair_total(2), 5)
        self.assertAlmostEqual(inferred_pair_total_f64(1.5), 4.0)

    def test_inference_from_field_values(self) -> None:
        # ``Pair(x, x)`` built in a fresh local slot whose type is not known:
        # the generic argument comes from the values written into the fields
        self.assertEqual(inferred_from_field_values(3), 6)

    def test_uninferable_construction(self) -> None:
        # a construction that names the bare template and whose
        # construction site has no type cannot pick a specialization
        with self.assertRaises(CompileError):
            uninferable_construction(1)

    def test_wrong_generic_arguments(self) -> None:
        with self.assertRaises(CompileError):
            wrong_generic_arguments(1)


class SpyPointerTest(TestCase):
    """Pointers: ``syntax.Ptr`` annotates a pointer type, ``ref`` takes the
    address of a value and ``p[...]`` dereferences a pointer value."""

    def test_deref_a_local(self) -> None:
        self.assertEqual(deref_local(5), 5)

    def test_write_through_a_pointer(self) -> None:
        self.assertEqual(write_through_ptr(1, 9), 9)
        self.assertEqual(add_through_ptr(1), 2)

    def test_module_qualified_ref(self) -> None:
        self.assertEqual(deref_module_qualified(6), 6)

    def test_pointer_parameter(self) -> None:
        # ``incr_ptr(ref(v))`` mutates the caller's local through the pointer:
        # both the local and the returned pointee value are 6
        self.assertEqual(call_incr_ptr(5), 606)

    def test_pointee_type_is_solved(self) -> None:
        self.assertEqual(call_deref_generic(3), 3)

    def test_pointee_type_identity(self) -> None:
        self.assertEqual(call_ptr_identity(4), 4)

    def test_a_mutable_pointer_converts_to_a_const_one(self) -> None:
        # ``read_const_ptr`` takes a ``ConstPtr``; a ``Ptr`` value converts to it
        self.assertEqual(call_read_const_ptr(8), 8)

    def test_pointer_field(self) -> None:
        self.assertEqual(field_read(3), 3)
        self.assertEqual(field_write(3, 7), 7)

    def test_field_through_a_pointer(self) -> None:
        self.assertEqual(field_through_ptr(1, 2), 4)


class SpyMultiPointerTest(TestCase):
    """Multi pointers: ``MultiPtr[T]`` is the address of ``T`` and of the
    elements that follow it, indexed with ``p[i]`` (the place ``p + i``) and
    offset with ``p + n``/``p += n``, while a single pointer (``Ptr``) names
    one place only and is dereferenced with ``p[...]``.  One converts to the
    other in one direction only, and only a multi pointer may be indexed and
    offset."""

    def test_index_reads_and_writes_an_element(self) -> None:
        self.assertEqual(multi_ptr_index(10), 12)
        self.assertEqual(multi_ptr_write(10), 99)

    def test_offset(self) -> None:
        self.assertEqual(multi_ptr_offset(10), 12)
        self.assertEqual(multi_ptr_offset_assign(10), 12)

    def test_ref_of_an_index_is_the_offset(self) -> None:
        self.assertEqual(ref_of_index_is_the_offset(10), 11)

    def test_the_annotated_type_accepts_a_pointer_to_an_array(self) -> None:
        # ``MultiPtr[i32]`` is what an array pointer's elements are addressed by
        self.assertEqual(multi_ptr_parameter(10), 11)

    def test_a_multi_pointer_accepts_a_const_one(self) -> None:
        self.assertEqual(multi_ptr_to_const_parameter(10), 11)

    def test_a_multi_pointer_converts_to_a_single_one(self) -> None:
        self.assertEqual(multi_ptr_to_single_parameter(10), 10)

    def test_a_single_pointer_is_not_indexed(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            single_ptr_index(1)
        self.assertIn('single pointer', str(ctx.exception))

    def test_a_single_pointer_is_not_offset(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            single_ptr_offset(1)
        self.assertIn('multi pointer', str(ctx.exception))

    def test_a_single_pointer_is_not_a_multi_one(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            single_ptr_is_not_multi(1)
        self.assertIn('cannot convert', str(ctx.exception))

    def test_a_pointer_reinterpreted_by_ptr_cast_indexes_by_its_new_pointee(self) -> None:
        # ``ptr_cast`` re-tags the pointer without an instruction (the LLVM
        # pointers are untyped): the index still strides by the *new* pointee
        # (the element), not by the array the address came from
        self.assertEqual(ptr_cast_index(10), 12)


class SpySliceIteratorTest(TestCase):
    """Iterating a ``std.SlicePtr``/``std.ConstSlicePtr``: ``__iter__`` yields
    the elements by value and ``refs()`` a pointer to each of them (see
    ``std._ConstSliceIterator``/``_SliceRefIterator``)."""

    def test_iterating_a_slice_yields_its_elements(self) -> None:
        self.assertEqual(iterate_slice_sum(10), 10 + 11 + 12 + 13)

    def test_refs_yields_each_element_place(self) -> None:
        # the pointers name the array's own elements, so a write lands in it
        self.assertEqual(iterate_slice_refs_write(10), 99 * 4)

    def test_iterating_a_const_slice(self) -> None:
        self.assertEqual(call_iterate_const_slice_sum(10), 10 + 11 + 12 + 13)

    def test_refs_of_a_const_slice(self) -> None:
        self.assertEqual(call_iterate_const_slice_refs_sum(10), 10 + 11 + 12 + 13)


class SpySlicePtrTest(TestCase):
    """``std.SlicePtr[T]``/``std.ConstSlicePtr[T]``: a pointer and the number of
    elements it carries.  ``arr_slice``/``const_arr_slice`` turn a pointer to an
    array into the slice of the whole array (the constness of the pointer chooses
    which), and a slice of a multi pointer builds one: ``p[a:b]`` is
    ``SlicePtr(p + a, b - a)``.  The slice a subscript builds is a compile-time
    aggregate of const places - a *view* - so it cannot be written through, while
    a slice *variable* is storage like any other."""

    def test_fields_of_a_slice(self) -> None:
        # the pointer names the array's elements and the length is its length
        self.assertEqual(multi_ptr_index(10), 12)
        self.assertEqual(slice_length(10), 4)

    def test_a_slice_names_the_array(self) -> None:
        # a write through the slice's pointer lands in the array itself
        self.assertEqual(multi_ptr_write(10), 99)

    def test_a_slice_of_a_slice(self) -> None:
        self.assertEqual(slice_of_a_slice(10), 11 + 12)
        self.assertEqual(slice_of_a_slice_length(10), 2)

    def test_a_step_of_one_is_the_same_slice(self) -> None:
        self.assertEqual(slice_of_a_slice_with_a_step_of_one(10), 2)

    def test_a_step_of_a_pointer_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            slice_of_a_slice_with_a_step(10)
        self.assertIn('has no step', str(ctx.exception))

    def test_a_missing_lower_bound_is_zero(self) -> None:
        # a bound the source left out is absent; the slice of a pointer takes a
        # missing lower bound as 0
        self.assertEqual(slice_of_a_slice_without_a_lower(10), 10 + 11)

    def test_a_slice_of_a_pointer_needs_an_upper_bound(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            slice_of_a_slice_without_an_upper(10)
        self.assertIn('needs an upper bound', str(ctx.exception))

    def test_a_slice_argument(self) -> None:
        self.assertEqual(slice_argument(10), 4)

    def test_arr_slice_of_a_pointer_to_an_array(self) -> None:
        # the conversion, and the element type the signature solves from it
        self.assertEqual(slice_of_an_array_argument(10), 4)

    def test_the_constness_of_the_pointer_is_the_slices(self) -> None:
        # an array the callee addresses (a by-reference parameter) is const, so
        # ``const_arr_slice`` of it is a ``ConstSlicePtr``
        self.assertEqual(call_slice_of_a_const_array(10), 4)

    def test_a_slice_variable_is_storage(self) -> None:
        self.assertEqual(slice_variable_is_writable(10), 2)

    def test_the_slice_of_a_subscript_is_const(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            slice_of_a_temporary_is_const(10)
        self.assertIn('const pointer', str(ctx.exception))

    def test_a_slice_of_compile_time_storage(self) -> None:
        # the elements are their own compile-time places, so indexing it is
        # folded (``Small(3, 4).b``)
        self.assertEqual(comptime_slice_element(), 4)
        self.assertEqual(comptime_slice_length(), 2)

    def test_a_runtime_index_of_compile_time_storage_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            runtime_index_into_comptime_storage(1)
        self.assertIn('compile-time integer', str(ctx.exception))

    def test_a_slice_is_subscripted_through_getitemptr(self) -> None:
        # ``s[i]`` is the place ``(Const)SlicePtr.__spy_getitemptr__`` returns
        self.assertEqual(slice_subscript_element(10), 12)
        self.assertEqual(const_slice_subscript_element(10), 11)

    def test_a_slice_subscript_write_lands_in_the_array(self) -> None:
        self.assertEqual(slice_subscript_write(10), 99)

    def test_the_slice_method_returns_a_sub_view(self) -> None:
        self.assertEqual(slice_method_slice(10), 11 + 12)
        self.assertEqual(const_slice_method_slice(10), 11 + 12 + 13)

    def test_an_absent_slice_bound_is_open(self) -> None:
        self.assertEqual(slice_method_open_bounds(10), 10 + 13)

    def test_an_out_of_bounds_call_reaches_the_stub(self) -> None:
        # the bounds check calls ``_out_of_bounds``, a no-op stub for now
        self.assertEqual(slice_out_of_bounds_calls_the_stub(10), 7)
        self.assertEqual(slice_subscript_out_of_bounds_calls_the_stub(10), 7)


class SpySubscriptOverloadTest(TestCase):
    """A struct value overloads the subscript with ``__spy_getitemptr__``: the
    place the method returns is what ``x[i]`` denotes, whether the method is
    inlined or compiled into a specialization of its own (see
    ``interp.HirRunner.subscript``)."""

    def test_a_subscript_reads_through_the_method(self) -> None:
        self.assertEqual(subscript_overload_read(10), 12)

    def test_a_subscript_writes_through_the_method(self) -> None:
        # the place the method returned is written through, landing in the array
        self.assertEqual(subscript_overload_write(10), 99)

    def test_a_registered_method_overloads_the_subscript(self) -> None:
        self.assertEqual(registered_subscript_overload(10), 10 + 13)

    def test_a_method_that_returns_a_value_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            bad_subscript_overload(10)
        self.assertIn('must return a pointer', str(ctx.exception))


class SpyTypeValueTest(TestCase):
    """A ``syntax`` type marker written in a body builds its spy type as a value
    (``hir.PointerType``/``hir.ArrayType``/``hir.OptionType``), and
    ``syntax.ptr_cast`` reinterprets a pointer as another pointer type
    (``hir.PtrCast``)."""

    def test_a_pointer_type_is_a_value(self) -> None:
        self.assertEqual(ptr_type_is_a_value(3), 1)

    def test_a_const_pointer_type_is_a_value(self) -> None:
        self.assertEqual(call_const_ptr_type_is_a_value(3), 1)

    def test_a_multi_pointer_type_is_a_value(self) -> None:
        self.assertEqual(call_multi_ptr_type_is_a_value(3), 1)

    def test_an_array_type_is_a_value(self) -> None:
        self.assertEqual(call_array_type_is_a_value(3), 1)

    def test_an_option_type_is_a_value(self) -> None:
        self.assertEqual(call_option_type_is_a_value(3), 1)

    def test_ptr_cast_of_the_same_type_is_a_noop(self) -> None:
        self.assertEqual(ptr_cast_of_the_same_type(3), 4)

    def test_ptr_cast_reinterprets_the_pointee(self) -> None:
        self.assertEqual(ptr_cast_to_another_pointee(5), 5)

    def test_ptr_cast_of_a_multi_pointer(self) -> None:
        self.assertEqual(call_ptr_cast_of_a_multi_pointer(7), 7)


class SpyAggregateArgumentTest(TestCase):
    """An aggregate argument is passed as its own type: a struct (or an array)
    is a subtype of itself and of nothing else, so a same-shaped aggregate of
    another type does not convert - neither by value (the value is read out and
    refused) nor by reference (the address would alias it, the MIR pointers
    being untyped)."""

    def test_an_argument_of_the_type_itself_is_taken(self) -> None:
        self.assertEqual(pass_two_i64(10), 21)
        self.assertEqual(pass_four_i64(10), 10 + 11 + 12 + 13)
        self.assertEqual(pass_four_i64_through_a_name(10), 46)

    def test_a_by_value_aggregate_of_another_type_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            pass_a_small_for_two_i64(10)
        self.assertIn('cannot convert', str(ctx.exception))

    def test_a_by_reference_aggregate_of_another_type_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            pass_four_i32_for_four_i64(10)
        self.assertIn('cannot convert', str(ctx.exception))


class SpyIfExprTest(TestCase):
    """If-expressions: ``a if c else b`` evaluates one of its branches into
    the result location of the expression - a runtime branch in the MIR, or
    just the chosen branch when the condition is a compile-time value."""

    def test_value(self) -> None:
        self.assertEqual(choose_value(True, 3, 4), 3)
        self.assertEqual(choose_value(False, 3, 4), 4)

    def test_into_a_fresh_local(self) -> None:
        # the expression is evaluated into the slot of the local it declares
        self.assertEqual(choose_local(True, 3, 4), 3)
        self.assertEqual(choose_local(False, 3, 4), 4)

    def test_nested(self) -> None:
        self.assertEqual(choose_nested(False, False, 1, 2, 3), 3)
        self.assertEqual(choose_nested(False, True, 1, 2, 3), 2)
        self.assertEqual(choose_nested(True, False, 1, 2, 3), 1)

    def test_condition(self) -> None:
        # a comparison, a call of another spy function, ...
        self.assertEqual(choose_comparison(3, 4), 3)
        self.assertEqual(choose_comparison(-3, 4), 4)
        self.assertEqual(choose_call_test(3, 4, -1), 3)
        self.assertEqual(choose_call_test(3, 4, 1), 4)

    def test_operand_and_argument(self) -> None:
        # the value of the expression feeds an operator and a call argument
        self.assertEqual(choose_arithmetic(True, 3, 4), 13)
        self.assertEqual(choose_arithmetic(False, 3, 4), 14)
        self.assertEqual(choose_argument(True, 3, 4), 4)
        self.assertEqual(choose_argument(False, 3, 4), 5)

    def test_comptime_condition(self) -> None:
        # only the chosen branch is typed, so the dead one may contain another
        # if-expression (and disagree with the result type)
        self.assertEqual(choose_comptime(3, 4, True), 3)
        self.assertEqual(choose_comptime(3, 4, False), 3)
        self.assertEqual(choose_comptime_else(3, 4, True), 4)
        self.assertEqual(choose_comptime_else(3, 4, False), 3)

    def test_inlined_body(self) -> None:
        self.assertEqual(call_pick_inline(True, 3, 4), 3)
        self.assertEqual(call_pick_inline(False, 3, 4), 4)

    def test_struct_branches(self) -> None:
        # each branch constructs its struct in place, in the same slot
        self.assertEqual(choose_struct(True, 1, 2), 3)
        self.assertEqual(choose_struct(False, 1, 2), 4)

    def test_result_pointer_branches(self) -> None:
        # a large struct is delivered through the result pointer of the
        # function: the branches write into the caller's location
        self.assertEqual(use_choose_large(True, 5), 8)
        self.assertEqual(use_choose_large(False, 5), 11)

    def test_with_pointers(self) -> None:
        self.assertEqual(choose_pointer(True, 3, 4), 3)
        self.assertEqual(choose_pointer(False, 3, 4), 4)
        self.assertEqual(choose_deref_write(True, 3, 4), 4)
        self.assertEqual(choose_deref_write(False, 3, 4), 3)

    def test_non_bool_condition(self) -> None:
        with self.assertRaises(CompileError):
            choose_non_bool_condition(1, 2)


class SpyStructMirrorTest(TestCase):
    """How a struct lowers to MIR (``sval.StructType._calculate_mir``): a spy
    struct is laid out by the compiler, an ``extern_c`` one for the C ABI."""

    def test_single_field_mirrors_to_the_field(self) -> None:
        one = struct_type(One)
        self.assertEqual(one.get_mir_type(MIR_CACHE), mir.IntType(32, True))
        self.assertEqual(one.get_field_mir_indices(MIR_CACHE), (0,))
        # ... and so does a struct of one struct field, whose own mirror is
        # the mirror of the field it holds
        self.assertEqual(struct_type(Nested).get_mir_type(MIR_CACHE), mir.IntType(32, True))

    def test_single_struct_field_mirror(self) -> None:
        # the mirror of a struct of one struct field is the mirror of the
        # field it holds - which may itself be a struct type, and the field
        # still sits at the address of the value itself
        outer = struct_type(OuterTwo)
        self.assertIsInstance(outer.get_mir_type(MIR_CACHE), mir.StructType)
        self.assertTrue(outer.mirror_is_a_field(MIR_CACHE))
        self.assertEqual(outer.get_field_mir_indices(MIR_CACHE), (0,))

    def test_fields_are_ordered_by_alignment(self) -> None:
        mixed = struct_type(Mixed)
        mirror = mixed.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['narrow', 'wide'])
        self.assertEqual(mixed.get_field_mir_indices(MIR_CACHE), (1, 0))

        # a pointer is word-aligned, like an integer of that width
        with_pointer = WithPointer.specialize(())
        mirror = with_pointer.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['n', 'p'])

    def test_extern_c_keeps_the_declaration_order(self) -> None:
        extern_mixed = struct_type(ExternMixed)
        mirror = extern_mixed.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['wide', 'narrow'])
        self.assertEqual(extern_mixed.get_field_mir_indices(MIR_CACHE), (0, 1))
        # an ``extern_c`` struct of one field keeps its wrapper struct, so
        # that the C layout is the one the declaration asks for
        self.assertIsInstance(struct_type(ExternOne).get_mir_type(MIR_CACHE), mir.StructType)


class SpyTupleTest(TestCase):
    """Compile-time tuples: destructuring assignment unpacks a tuple of
    values into a tuple of target addresses (which may nest)."""

    def test_declaring_destructuring(self) -> None:
        self.assertEqual(tuple_declare(4), 45)

    def test_swap_destructuring_is_a_known_flaw(self) -> None:
        # the right-hand side is written into its targets in order, so the
        # second element reads its source after the first has overwritten it:
        # ``a, b = b, a`` gives ``b, b`` (a known design flaw)
        self.assertEqual(tuple_swap(1, 2), 22)

    def test_nested_destructuring(self) -> None:
        self.assertEqual(tuple_nested(4), 456)

    def test_destructuring_from_calls(self) -> None:
        self.assertEqual(tuple_of_calls(3), 45)

    def test_tuple_into_a_comptime_variable_conflict(self) -> None:
        # a slot holding a tuple cannot also be stored a single value, and the
        # conflict is reported when the slot is committed
        with self.assertRaises(CompileError):
            bad_comptime_tuple_branch(True)


class SpyZeroSizedResultTest(TestCase):
    """A call whose result is a zero-sized type.  The callee returns no value
    (a zero-sized type has no runtime representation, so the call is a MIR call
    of the void type), but the call still writes the result type's *unit value*
    into its result location: an expression statement then commits that value
    into the temporary it drops - the location is left untyped otherwise, and
    the type is what makes its slot a compile-time box - and a variable bound
    to the call is a box of the unit value, typed as the result type."""

    def test_void_call_as_a_statement(self) -> None:
        self.assertEqual(discard_void_call(7), 7)

    def test_void_call_bound_to_a_variable(self) -> None:
        self.assertTrue(void_call_type(7))

    def test_zero_bits_call_as_a_statement(self) -> None:
        self.assertEqual(discard_zero_bits_call(7), 7)

    def test_zero_bits_call_bound_to_a_variable(self) -> None:
        self.assertTrue(zero_bits_call_type(7))

    def test_registered_void_method_as_a_statement(self) -> None:
        self.assertEqual(call_registered_void_method(7), 0)

    def test_inline_void_method_as_a_statement(self) -> None:
        self.assertEqual(call_inline_void_method(7), 0)

    def test_zero_sized_struct_result(self) -> None:
        # every field of ``Nothing`` is zero-sized, so the struct itself is: it
        # has no layout (the size and alignment estimates of a layoutless type,
        # ``mir.estimated_size_of``, decide how one is returned) and a call
        # returning one delivers no value, only the type's unit value
        self.assertEqual(use_nothing(7), 7)

    def test_zero_sized_struct_local(self) -> None:
        self.assertEqual(construct_nothing(7), 7)
        self.assertEqual(construct_blank(7), 7)

    def test_zero_sized_construction_participates_in_peer_resolution(self) -> None:
        # the construction of a zero-sized struct delivers its value as an
        # ordinary store point, so the slot's type is the peer type of all its
        # stores - and two unrelated structs have none (the first construction
        # must not pin the slot's type by itself)
        with self.assertRaises(CompileError):
            choose_two_zst_structs(True)


class SpyArrayTest(TestCase):
    """Arrays: ``array(a1, a2, ...)`` builds an array in place - the length is
    the number of elements, the element type the one the place it is built in
    declares or the common type of the elements - and ``a[i]`` is the place of
    the i-th element, read and written through like a field."""

    def test_read_and_sum_elements(self) -> None:
        self.assertEqual(read_element(5), 5)
        self.assertEqual(sum_elements(5), 11)

    def test_write_element(self) -> None:
        self.assertEqual(write_element(5, 9), 9)
        self.assertEqual(add_to_element(5), 1506)

    def test_elements_are_structs(self) -> None:
        self.assertEqual(array_of_structs(5), 7)
        self.assertEqual(write_struct_element(5), 7)

    def test_nested_array(self) -> None:
        self.assertEqual(nested_array(5), 13)

    def test_array_through_pointer(self) -> None:
        self.assertEqual(array_through_ptr(5), 11)

    def test_array_value_is_copied(self) -> None:
        self.assertEqual(copy_array(5), 11)

    def test_array_argument_and_result(self) -> None:
        self.assertEqual(call_pass_array(5), 6)
        self.assertEqual(call_pass_array_runtime_index(5, 0), 5)
        self.assertEqual(call_pass_array_runtime_index(5, 1), 6)
        self.assertEqual(use_returned_array(5), 8)

    def test_wide_array(self) -> None:
        # 32 bytes: the array is a by-reference aggregate at the call boundary
        self.assertEqual(wide_array(5), 13)

    def test_array_field(self) -> None:
        self.assertEqual(array_in_struct(5), 16)
        self.assertEqual(write_struct_array(5), 10)

    def test_length_keyword_is_ignored(self) -> None:
        self.assertEqual(length_keyword(5), 11)

    def test_discarded_array(self) -> None:
        self.assertEqual(discard_array(5), 5)

    def test_zero_sized_array(self) -> None:
        self.assertEqual(zero_sized_array(5), 5)

    def test_empty_array(self) -> None:
        self.assertEqual(use_empty_array(), 4)

    def test_element_out_of_bounds(self) -> None:
        with self.assertRaises(CompileError):
            element_out_of_bounds(5)

    def test_wrong_element_count(self) -> None:
        with self.assertRaises(CompileError):
            wrong_element_count(5)

    def test_untyped_elements_are_rejected(self) -> None:
        # an untyped integer literal has no runtime type of its own, so an
        # array of them has no element type either
        with self.assertRaises(CompileError):
            untyped_elements(5)

    def test_unknown_keyword(self) -> None:
        with self.assertRaises(CompileError):
            unknown_array_keyword(5)


class SpyCompileLogTest(TestCase):
    def test_compile_log_prints_at_compile_time(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(call_inline_log(1, 2), 3)
        self.assertIn('add_inline was compiled', out.getvalue())


# ---------------------------------------------------------------------------
# options: ``Option[T]`` holds a ``T`` or the null value, which is what the
# Python literal ``None`` evaluates to.  Both a ``T`` and the null value
# convert to ``Option[T]``, so a slot (or a branch) that receives both takes
# the option type.  The representation is chosen by the child type (see
# ``sval.OptionType.to_mir_type``): a zero-sized child is a ``bool``, a child
# that holds a pointer uses its first pointer as the absent tag, and any other
# child a struct of a tag and the value.
# ---------------------------------------------------------------------------

# the spy types the tests compare against, as module globals: ``spy.typeof``
# of an option-typed place is the one ``Option[spy.i32]`` of the compiler
NULL_TYPE = sval.NullType()
OPT_I32 = sval.OptionType(spy_type(i32))
OPT_PTR_I32 = sval.OptionType(spy_type(Ptr[i32]))


@struct()
class OptHolder:
    """A struct with an option field: an option is an ordinary field."""

    o: Option[i32]
    n: i32


@func()
def maybe_add(x: i32, y: i32, c: spy_bool) -> Option[i32]:
    # one path returns a value of the child type, the other the null value:
    # the result location is the peer type of the two, the option
    if c:
        return x + y
    return None


@func()
def opt_width(o: Option[i32]) -> i32:
    return 1


@func()
def use_maybe(x: i32, c: spy_bool) -> i32:
    o = maybe_add(x, 1, c)
    return x + opt_width(o)


@func()
def null_argument(x: i32) -> i32:
    # the null value as an argument: it converts to the option of the
    # parameter's type
    return x + opt_width(None)


@func()
def maybe_large(x: i64, c: spy_bool) -> Option[Large]:
    # ``Large`` holds no pointer and is too big for the by-value limit: the
    # option is returned through a result pointer, and the construction of
    # ``Large`` is delivered into the payload of the option (the tag is set)
    if c:
        return Large(x, 1, 2, 3)
    return None


@func()
def take_large(o: Option[Large]) -> i64:
    return 2


@func()
def use_maybe_large(x: i64, c: spy_bool) -> i64:
    o = maybe_large(x, c)
    return x + take_large(o)


@func()
def maybe_ptr(p: Ptr[i32], c: spy_bool) -> Option[Ptr[i32]]:
    # a child that holds a pointer: the option *is* the pointer, a null one
    # being the absent value
    if c:
        return p
    return None


@func()
def take_ptr(o: Option[Ptr[i32]]) -> i32:
    return 1


@func()
def use_maybe_ptr(x: i32, c: spy_bool) -> i32:
    y = x
    o = maybe_ptr(ref(y), c)
    return y + take_ptr(o)


@struct()
class PtrMixed:
    """A struct whose first pointer is not the first field of its mirror: the
    fields are ordered by alignment (the ``i8`` before the pointer), while
    ``find_first_pointer_type_pos`` names the declaration order."""

    p: Ptr[i32]
    b: i8


@func()
def maybe_ptr_mixed(p: Ptr[i32], b: i8, c: spy_bool) -> Option[PtrMixed]:
    # the option shares the representation of its child, whose first pointer
    # (in mirror order) is the tag: a constructed value sets it
    if c:
        return PtrMixed(p, b)
    return None


@func()
def take_ptr_mixed(o: Option[PtrMixed]) -> i32:
    return 4


@func()
def use_maybe_ptr_mixed(x: i32, c: spy_bool) -> i32:
    y = x
    o = maybe_ptr_mixed(ref(y), 1, c)
    return y + take_ptr_mixed(o)


@func()
def option_field(x: i32, c: spy_bool) -> i32:
    # the two branches of the ``if`` write a value and the null value into
    # the field place: the field is the option the two peers to
    h = OptHolder(x if c else None, 1)
    return h.n + x


@func()
def option_as_null() -> spy_bool:
    return spy_typeof(None) == NULL_TYPE


@func()
def option_field_is_an_option(x: i32) -> spy_bool:
    h = OptHolder(x, 1)
    return spy_typeof(h.o) == OPT_I32


@func()
def option_pointer_is_an_option(x: i32, c: spy_bool) -> spy_bool:
    y = x
    o = maybe_ptr(ref(y), c)
    return spy_typeof(o) == OPT_PTR_I32


@func()
def bad_option_argument(x: i32) -> i32:
    # a struct is neither the child type nor the null value: it does not
    # convert to the option
    return opt_width(Small(x, 1))  # pyright: ignore[reportArgumentType]


def opt_inline(x: i32, c: spy_bool):
    # an inlined body: its result location takes the option type from the
    # peer type of its two returns
    if c:
        return x
    return None


@func()
def use_opt_inline(x: i32, c: spy_bool) -> i32:
    o = opt_inline(x, c)
    return x + opt_width(o)


@func()
def generic_opt_width[T](o: Option[T]) -> i32:
    # the child of an option parameter is a type parameter: a call solves it
    # from the type of the argument (which converts to the option)
    return 3


@func()
def use_generic_opt(x: i32) -> i32:
    return x + generic_opt_width(x)


@func()
def bad_generic_null(x: i32) -> i32:
    # the null value is not a ``T``: it cannot solve the child of an option
    return generic_opt_width(None)


class SpyOptionTest(TestCase):
    def test_null_value(self) -> None:
        self.assertTrue(option_as_null())

    def test_returning_an_option(self) -> None:
        # a value on one path, the null value on the other
        self.assertEqual(use_maybe(10, True), 11)
        self.assertEqual(use_maybe(10, False), 11)

    def test_null_argument(self) -> None:
        self.assertEqual(null_argument(10), 11)

    def test_option_through_a_result_pointer(self) -> None:
        # ``Option[Large]`` is returned through a result pointer, and the
        # construction is delivered into the payload of the option
        self.assertEqual(use_maybe_large(10, True), 12)
        self.assertEqual(use_maybe_large(10, False), 12)

    def test_option_of_a_pointer(self) -> None:
        self.assertEqual(use_maybe_ptr(10, True), 11)
        self.assertEqual(use_maybe_ptr(10, False), 11)

    def test_option_of_a_struct_whose_mirror_reorders_the_pointer(self) -> None:
        self.assertEqual(use_maybe_ptr_mixed(10, True), 14)
        self.assertEqual(use_maybe_ptr_mixed(10, False), 14)

    def test_option_field(self) -> None:
        self.assertEqual(option_field(10, True), 11)
        self.assertEqual(option_field(10, False), 11)

    def test_typeof(self) -> None:
        self.assertTrue(option_field_is_an_option(1))
        self.assertTrue(option_pointer_is_an_option(1, True))

    def test_inlined_body(self) -> None:
        self.assertEqual(use_opt_inline(10, True), 11)
        self.assertEqual(use_opt_inline(10, False), 11)

    def test_generic_child_is_solved(self) -> None:
        self.assertEqual(use_generic_opt(10), 13)

    def test_null_does_not_solve_the_child(self) -> None:
        with self.assertRaises(TypeMismatchError):
            bad_generic_null(1)

    def test_wrong_argument_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_option_argument(1)


# a struct of two pointers: the inner ``Option`` tags on the first one, so the
# outer one of ``Option[Option[TwoPtrs]]`` has to tag on the second
@struct()
class TwoPtrs:
    a: Ptr[i32]
    b: Ptr[i32]


@func()
def maybe_deep(x: i32, c: spy_bool) -> Option[Option[Ptr[i32]]]:
    # the only pointer of ``Ptr[i32]`` is the inner option's tag, so the outer
    # option has none left and carries a ``bool`` tag instead
    y = x
    if c:
        return ref(y)
    return None


@func()
def take_deep(o: Option[Option[Ptr[i32]]]) -> i32:
    return 5


@func()
def use_maybe_deep(x: i32, c: spy_bool) -> i32:
    o = maybe_deep(x, c)
    return x + take_deep(o)


@func()
def maybe_two(a: Ptr[i32], b: Ptr[i32], c: spy_bool) -> Option[Option[TwoPtrs]]:
    # ``TwoPtrs`` has two pointers: the inner option tags on ``a`` (the first),
    # so the outer one tags on ``b`` (the second)
    if c:
        return TwoPtrs(a, b)
    return None


@func()
def take_two(o: Option[Option[TwoPtrs]]) -> i32:
    return 6


@func()
def use_maybe_two(x: i32, c: spy_bool) -> i32:
    y = x
    z = x + 1
    o = maybe_two(ref(y), ref(z), c)
    return x + take_two(o)


class SpyOptionNestingTest(TestCase):
    """The tag of a nested option: an ``Option`` uses one pointer of its child
    as its tag, so it has one fewer than the child and ``Option[Option[T]]``
    has to find ``T``'s *second* pointer (see
    ``sval.find_first_pointer_type_pos``)."""

    def test_the_inner_option_claims_the_only_pointer(self) -> None:
        ptr = spy_type(Ptr[i32])
        opt_ptr = sval.OptionType(ptr)
        self.assertEqual(sval.find_first_pointer_type_pos(ptr), ())
        # the inner option takes that pointer as its tag ...
        self.assertIsNone(sval.find_first_pointer_type_pos(opt_ptr))
        # ... so the outer one has to carry a ``bool`` tag
        outer = sval.OptionType(opt_ptr)
        outer_mir = outer.to_mir_type(MIR_CACHE)
        ptr_mir = ptr.to_mir_type(MIR_CACHE)
        assert isinstance(outer_mir, mir.StructType)
        assert isinstance(ptr_mir, mir.PointerType)
        pointer_size = MIR_CACHE.target.pointer_size
        self.assertEqual(
            mir.estimated_size_of(outer_mir, pointer_size),
            2 * mir.estimated_size_of(ptr_mir, pointer_size),
        )

    def test_the_outer_option_takes_the_second_pointer(self) -> None:
        two = struct_type(TwoPtrs)
        inner = sval.OptionType(two)
        outer = sval.OptionType(inner)
        # the inner option tags on ``a`` ...
        self.assertEqual(sval.find_first_pointer_type_pos(two), (0,))
        # ... so the outer one tags on ``b``
        self.assertEqual(sval.find_first_pointer_type_pos(inner), (0, 1))
        # the two share the representation of the child ...
        self.assertIs(outer.to_mir_type(MIR_CACHE), two.get_mir_type(MIR_CACHE))
        # ... and a further level has no pointer left
        self.assertIsNone(sval.find_first_pointer_type_pos(outer))

    def test_nested_option_of_a_pointer(self) -> None:
        self.assertEqual(use_maybe_deep(10, True), 15)
        self.assertEqual(use_maybe_deep(10, False), 15)

    def test_nested_option_of_a_two_pointer_struct(self) -> None:
        self.assertEqual(use_maybe_two(10, True), 16)
        self.assertEqual(use_maybe_two(10, False), 16)


# ---------------------------------------------------------------------------
# ``and``/``or``: a chain is lowered into a ``hir.Block`` whose operands
# short-circuit with ``hir.BreakIf``s.  An ``if``/``while`` whose condition is
# an ``and`` chain puts its branch body inside the block, so an unwrapped
# option's payload dominates it (see ``astgen``).
# ---------------------------------------------------------------------------


@func()
def record(p: Ptr[i32], v: i32) -> spy_bool:
    # a side effect, to observe whether an operand was evaluated
    p[...] = v
    return True


@func()
def boolop_and(a: i32, b: i32) -> i32:
    if a > 0 and b > 0:
        return 1
    return 0


@func()
def boolop_and_else(a: i32, b: i32) -> i32:
    if a > 0 and b > 0:
        return 1
    else:
        return 2


@func()
def boolop_and_return(a: i32, b: i32) -> i32:
    if a > 0 and b > 0 and a + b > 5:
        return a + b
    return -1


@func()
def boolop_or(a: i32, b: i32) -> i32:
    if a > 0 or b > 0:
        return 1
    return 0


@func()
def boolop_or_else(a: i32, b: i32) -> i32:
    if a > 0 or b > 0:
        return 1
    else:
        return 2


@func()
def boolop_nested(a: i32, b: i32, c: spy_bool) -> i32:
    # an ``or`` chain is an operand of an ``and`` chain
    if (a > 0 or b > 0) and c:
        return 1
    return 0


@func()
def boolop_comptime(a: i32) -> i32:
    # a compile-time operand short-circuits the chain
    flag: Comptime = False
    if a > 0 and flag:
        return 1
    return 0


@func()
def boolop_and_value(a: i32, b: i32) -> spy_bool:
    return a > 0 and b > 0


@func()
def boolop_or_value(a: i32, b: i32) -> spy_bool:
    return a > 0 or b > 0


@func()
def boolop_and_short_circuit(a: i32) -> i32:
    # the ``record`` operand runs only when the first one is true
    x = a
    if a > 0 and record(ref(x), 7):
        pass
    return x


@func()
def boolop_or_short_circuit(a: i32) -> i32:
    # ... and only when the first one is false, for ``or``
    x = a
    if a > 0 or record(ref(x), 7):
        pass
    return x


@func()
def boolop_while(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n and i < 3:
        total = total + i
        i = i + 1
    return total


@func()
def boolop_while_break(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n and i < 5:
        if i == 2:
            break
        total = total + i
        i = i + 1
    else:
        total = total + 100
    return total


@func()
def boolop_while_continue(n: i32) -> i32:
    i: i32 = 0
    total: i32 = 0
    while i < n and i < 4:
        i = i + 1
        if i == 2:
            continue
        total = total + i
    return total


class SpyBoolOpTest(TestCase):
    def test_and(self) -> None:
        self.assertEqual(boolop_and(1, 1), 1)
        self.assertEqual(boolop_and(0, 1), 0)
        self.assertEqual(boolop_and(1, 0), 0)

    def test_and_with_an_else(self) -> None:
        self.assertEqual(boolop_and_else(1, 1), 1)
        self.assertEqual(boolop_and_else(1, 0), 2)

    def test_a_body_that_returns(self) -> None:
        self.assertEqual(boolop_and_return(3, 4), 7)
        self.assertEqual(boolop_and_return(1, 1), -1)
        self.assertEqual(boolop_and_return(0, 4), -1)

    def test_or(self) -> None:
        self.assertEqual(boolop_or(1, 0), 1)
        self.assertEqual(boolop_or(0, 1), 1)
        self.assertEqual(boolop_or(0, 0), 0)

    def test_or_with_an_else(self) -> None:
        self.assertEqual(boolop_or_else(0, 1), 1)
        self.assertEqual(boolop_or_else(0, 0), 2)

    def test_a_nested_chain(self) -> None:
        self.assertEqual(boolop_nested(1, 0, True), 1)
        self.assertEqual(boolop_nested(0, 0, True), 0)
        self.assertEqual(boolop_nested(1, 1, False), 0)

    def test_a_compile_time_operand(self) -> None:
        self.assertEqual(boolop_comptime(1), 0)

    def test_a_chain_as_a_value(self) -> None:
        self.assertTrue(boolop_and_value(1, 1))
        self.assertFalse(boolop_and_value(1, 0))
        self.assertFalse(boolop_and_value(0, 1))
        self.assertTrue(boolop_or_value(1, 0))
        self.assertTrue(boolop_or_value(0, 1))
        self.assertFalse(boolop_or_value(0, 0))

    def test_short_circuiting(self) -> None:
        self.assertEqual(boolop_and_short_circuit(0), 0)
        self.assertEqual(boolop_and_short_circuit(1), 7)
        self.assertEqual(boolop_or_short_circuit(1), 1)
        self.assertEqual(boolop_or_short_circuit(-1), 7)

    def test_a_while_condition(self) -> None:
        self.assertEqual(boolop_while(5), 3)
        self.assertEqual(boolop_while(2), 1)

    def test_break_and_the_else_of_a_while(self) -> None:
        self.assertEqual(boolop_while_break(9), 1)
        self.assertEqual(boolop_while_break(2), 101)

    def test_continue_of_a_while(self) -> None:
        self.assertEqual(boolop_while_continue(5), 8)


# ---------------------------------------------------------------------------
# unwrapping an option: ``expr is None`` / ``expr is not None`` test whether an
# option is absent, and the walrus form ``(name := expr) is not None`` binds
# ``name`` to the option's *payload* - a place, so writing through it writes the
# payload - while the general ``(name := expr)`` binds the option itself.
# ---------------------------------------------------------------------------


@func()
def unwrap_opt(x: i32, c: spy_bool) -> i32:
    # ``o`` is the payload of the option: reading it reads the payload
    if (o := maybe_add(x, 1, c)) is not None:
        return o
    return -1


@func()
def unwrap_opt_written(x: i32, c: spy_bool) -> i32:
    # writing ``o`` writes the payload of the option
    if (o := maybe_add(x, 1, c)) is not None:
        o = o + 10
        return o
    return -1


@func()
def unwrap_opt_augmented(x: i32, c: spy_bool) -> i32:
    if (o := maybe_add(x, 1, c)) is not None:
        o += 100
        return o
    return -1


@func()
def opt_is_none(x: i32, c: spy_bool) -> spy_bool:
    o = maybe_add(x, 1, c)
    return o is None


@func()
def walrus_is_none(x: i32, c: spy_bool) -> spy_bool:
    # the general ``:=`` binds the target to the option itself, which ``is
    # None`` then tests
    return (_ := maybe_add(x, 1, c)) is None


@func()
def unwrap_opt_chain(x: i32, c: spy_bool) -> i32:
    # both names are visible in the branch the ``and`` chain guards
    if (a := maybe_add(x, 1, c)) is not None and (b := maybe_add(x, 100, c)) is not None:
        return a + b
    return -1


@func()
def unwrap_ptr_opt(x: i32, c: spy_bool) -> i32:
    # the option shares the representation of its pointer child: the payload is
    # the pointer itself
    y = x
    if (p := maybe_ptr(ref(y), c)) is not None:
        return p[...]
    return -1


@func()
def unwrap_large_opt(x: i64, c: spy_bool) -> i64:
    if (l := maybe_large(x, c)) is not None:
        return l.a
    return -1


@func()
def unwrap_deep_opt(x: i32, c: spy_bool) -> i32:
    # the outer option carries a ``bool`` tag (the inner one claimed the only
    # pointer): both layers are unwrapped
    if (p := maybe_deep(x, c)) is not None:
        if (q := p) is not None:
            return q[...]
        return -2
    return -1


@func()
def unwrap_comptime_field(x: i32, c: spy_bool) -> i32:
    # an option field of a compile-time aggregate: its payload pointer is taken
    # out of the compile-time storage (a ``ComptimeOptionPtr``)
    h = OptHolder(x if c else None, 1)
    ch: Comptime = h
    if (o := ch.o) is not None:
        return o + 100
    return -1


@func()
def maybe_u0(c: spy_bool) -> Option[u0]:
    if c:
        return 0  # pyright: ignore[reportReturnType]
    return None


@func()
def unwrap_u0(c: spy_bool) -> i32:
    # a zero-sized child: the payload has no storage, only its presence matters
    if (_ := maybe_u0(c)) is not None:
        return 7
    return -1


@func()
def unwrap_declared() -> i32:
    # a declared ``Comptime[Option[T]]`` variable is compile-time option storage
    ch: Comptime[Option[i32]] = 5
    if (o := ch) is not None:
        return o
    return -1


@func()
def bad_is_none(x: i32) -> spy_bool:
    # ``is None`` needs an option
    return x is None  # pyright: ignore[reportUnnecessaryComparison]


@func()
def unwrap_scope_leak(x: i32, c: spy_bool) -> i32:
    # ``o`` is declared in the ``if`` condition's scope: it is not visible after
    # the ``if``, so reading it here is rejected
    if (o := maybe_add(x, 1, c)) is not None:
        pass
    return 0 if o is None else o


@func()
def walrus_duplicate(x: i32, c: spy_bool) -> i32:
    # a ``:=`` target introduces a new variable: a duplicate is rejected
    if (_ := maybe_add(x, 1, c)) is not None and (_ := maybe_add(x, 2, c)) is not None:
        return 1
    return -1


# an option field of a compile-time aggregate whose child is a *pointer* struct:
# the option shares the child's representation, so an absent value nulls the
# pointer the child's first pointer field names (``mir.InsertValue``)
@struct()
class PtrMixedHolder:
    o: Option[PtrMixed]
    n: i32


@func()
def pass_declared_ptr_mixed(x: i32, c: spy_bool) -> i32:
    y = x
    pm = maybe_ptr_mixed(ref(y), 1, c)
    ch: Comptime[Option[PtrMixed]] = pm
    return take_ptr_mixed(ch)


@func()
def comptime_option_ref_in_memory() -> i32:
    # a plain local has no compile-time storage for the pointer: the option is
    # materialized and the local points at that copy (the ``_to_runtime`` of a
    # ``ComptimeOptionPtr``), so writing through it does not touch ``ch``
    ch: Comptime[Option[i32]] = 5
    p = ref(ch)
    p[...] = 7
    if (v := p[...]) is not None and (w := ch) is not None:
        return v * 10 + w
    return -1


@func()
def unwrap_opt_else(x: i32, c: spy_bool) -> i32:
    # an ``if`` whose ``and`` chain guards an unwrap, with an else branch
    if (o := maybe_add(x, 1, c)) is not None and (p := maybe_add(o, 1, c)) is not None:
        return o + p
    else:
        return -1


@func()
def unwrap_opt_while(n: i32) -> i32:
    # an unwrapped option as a ``while`` condition: the body sees the payload
    total: i32 = 0
    i = n
    while (v := maybe_add(i, 1, i > 0)) is not None and v > 1:
        total = total + v
        i = i - 1
    return total


@func()
def unwrap_opt_while_break(n: i32) -> i32:
    # a ``break`` in the body skips the else clause, a natural exit runs it
    total: i32 = 0
    i = n
    while (v := maybe_add(i, 1, i > 0)) is not None and v > 0:
        if v > 3:
            break
        total = total + v
        i = i - 1
    else:
        total = total + 100
    return total


@func()
def unwrap_opt_or_condition(x: i32, c: spy_bool) -> i32:
    # an ``or`` condition is a value: the branch is a plain ``hir.If``
    if maybe_add(x, 1, c) is not None or maybe_add(x, 2, c) is not None:
        return 1
    return -1


@func()
def unwrap_opt_else_scope(x: i32, c: spy_bool) -> i32:
    # the else branch does not see the unwrap's target
    if (o := maybe_add(x, 1, c)) is not None:
        return o
    else:
        return 0 if o is None else o  # pyright: ignore[reportPossiblyUnboundVariable]


@func()
def opt_through_ptr(p: Ptr[Option[i32]]) -> i32:
    # read an option through a pointer to it
    if (v := p[...]) is not None:
        return v
    return -1


@func()
def ptr_opt_through_ptr(p: Ptr[Option[Ptr[i32]]]) -> i32:
    # ... and a pointer-tag option (it shares the pointer's representation)
    if (v := p[...]) is not None:
        return v[...]
    return -1


@func()
def comptime_option_ptr_present() -> i32:
    # ``ref`` of compile-time option storage: the option is materialized and
    # the pointer to that copy is what is passed (``_to_runtime`` of a
    # ``ComptimeOptionPtr``)
    ch: Comptime[Option[i32]] = 5
    return opt_through_ptr(ref(ch))


@func()
def comptime_option_ptr_absent() -> i32:
    ch: Comptime[Option[i32]] = None
    return opt_through_ptr(ref(ch))


@func()
def comptime_option_ptr_runtime(x: i32, c: spy_bool) -> i32:
    # the tag comes from a runtime option value
    o = maybe_add(x, 1, c)
    ch: Comptime[Option[i32]] = o
    return opt_through_ptr(ref(ch))


@func()
def comptime_option_ptr_of_ptr(x: i32, c: spy_bool) -> i32:
    y = x
    o = maybe_ptr(ref(y), c)
    ch: Comptime[Option[Ptr[i32]]] = o
    return ptr_opt_through_ptr(ref(ch))


class SpyOptionUnwrapTest(TestCase):
    def test_reading_the_payload(self) -> None:
        self.assertEqual(unwrap_opt(10, True), 11)
        self.assertEqual(unwrap_opt(10, False), -1)

    def test_writing_through_the_payload(self) -> None:
        self.assertEqual(unwrap_opt_written(10, True), 21)
        self.assertEqual(unwrap_opt_written(10, False), -1)
        self.assertEqual(unwrap_opt_augmented(10, True), 111)

    def test_is_none(self) -> None:
        self.assertFalse(opt_is_none(10, True))
        self.assertTrue(opt_is_none(10, False))
        self.assertFalse(walrus_is_none(10, True))
        self.assertTrue(walrus_is_none(10, False))

    def test_a_chain_of_unwraps(self) -> None:
        self.assertEqual(unwrap_opt_chain(10, True), 121)
        self.assertEqual(unwrap_opt_chain(10, False), -1)

    def test_a_pointer_option(self) -> None:
        self.assertEqual(unwrap_ptr_opt(10, True), 10)
        self.assertEqual(unwrap_ptr_opt(10, False), -1)

    def test_a_large_option(self) -> None:
        self.assertEqual(unwrap_large_opt(10, True), 10)
        self.assertEqual(unwrap_large_opt(10, False), -1)

    def test_a_nested_option(self) -> None:
        self.assertEqual(unwrap_deep_opt(10, True), 10)
        self.assertEqual(unwrap_deep_opt(10, False), -1)

    def test_a_compile_time_option_field(self) -> None:
        self.assertEqual(unwrap_comptime_field(10, True), 110)
        self.assertEqual(unwrap_comptime_field(10, False), -1)

    def test_a_zero_sized_child(self) -> None:
        self.assertEqual(unwrap_u0(True), 7)
        self.assertEqual(unwrap_u0(False), -1)

    def test_a_declared_comptime_option(self) -> None:
        self.assertEqual(unwrap_declared(), 5)

    def test_an_absent_pointer_option_in_compile_time_storage(self) -> None:
        self.assertEqual(pass_declared_ptr_mixed(10, True), 4)
        self.assertEqual(pass_declared_ptr_mixed(10, False), 4)

    def test_a_pointer_to_compile_time_option_storage(self) -> None:
        # a ``Comptime[Option[T]]`` addressed by ``ref`` is materialized: what
        # is passed is a pointer to a fresh copy of the option
        self.assertEqual(comptime_option_ptr_present(), 5)
        self.assertEqual(comptime_option_ptr_absent(), -1)
        self.assertEqual(comptime_option_ptr_runtime(10, True), 11)
        self.assertEqual(comptime_option_ptr_runtime(10, False), -1)

    def test_a_pointer_to_compile_time_pointer_option_storage(self) -> None:
        self.assertEqual(comptime_option_ptr_of_ptr(7, True), 7)
        self.assertEqual(comptime_option_ptr_of_ptr(7, False), -1)

    def test_a_pointer_to_a_compile_time_option(self) -> None:
        # ``ref`` of a compile-time option: the option is materialized and the
        # pointer points at that copy
        self.assertEqual(comptime_option_ref_in_memory(), 75)

    def test_an_if_else_guarded_by_a_chain(self) -> None:
        self.assertEqual(unwrap_opt_else(10, True), 23)
        self.assertEqual(unwrap_opt_else(10, False), -1)

    def test_a_while_guarded_by_a_chain(self) -> None:
        self.assertEqual(unwrap_opt_while(3), 9)
        self.assertEqual(unwrap_opt_while(0), 0)

    def test_break_and_the_else_of_a_while(self) -> None:
        self.assertEqual(unwrap_opt_while_break(5), 0)
        self.assertEqual(unwrap_opt_while_break(1), 102)

    def test_an_or_condition(self) -> None:
        self.assertEqual(unwrap_opt_or_condition(10, True), 1)
        self.assertEqual(unwrap_opt_or_condition(10, False), -1)

    def test_the_else_does_not_see_the_target(self) -> None:
        with self.assertRaises(CompileError):
            unwrap_opt_else_scope(1, True)

    def test_is_none_needs_an_option(self) -> None:
        with self.assertRaises(CompileError):
            bad_is_none(1)

    def test_an_unwrap_target_is_scoped_to_the_branch(self) -> None:
        with self.assertRaises(CompileError):
            unwrap_scope_leak(1, True)

    def test_a_duplicate_unwrap_target_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            walrus_duplicate(1, True)


# ---------------------------------------------------------------------------
# multiple return values: a function annotated ``-> tuple[T1, T2, ...]``
# returns one value per element of the tuple.  The lowered function returns
# one of them by value and delivers every other one through a hidden result
# pointer (see ``sval.make_ret_spec``); the caller either destructures the
# call (``a, b = f()``) or keeps the packed results in a ``Comptime`` variable.
# ---------------------------------------------------------------------------


def mir_signature(handle: Any) -> tuple[tuple[mir.Type, ...], mir.MayBeVoidType]:
    """The lowered MIR signature - the argument types and the return type - of
    the one specialization of a registered function.  The function must have
    been compiled already (by calling one of its wrappers)."""
    entry = handle.get_entry()
    assert len(entry.specs) == 1, 'the function was compiled more than once'
    instance = next(iter(entry.specs.values()))
    return tuple(instance.mir.args), instance.mir.ret_type


@func()
def min_max(a: i32, b: i32) -> tuple[i32, i32]:
    if a < b:
        return a, b
    return b, a


@func()
def use_min_max(a: i32, b: i32) -> i32:
    lo, hi = min_max(a, b)
    return lo * 100 + hi


@func()
def forward_min_max(a: i32, b: i32) -> tuple[i32, i32]:
    return min_max(a, b)


@func()
def use_forward_min_max(a: i32, b: i32) -> i32:
    lo, hi = forward_min_max(a, b)
    return lo * 100 + hi


@func()
def comptime_multi(a: i32) -> i32:
    # the packed result of a call has no runtime type of its own: a ``Comptime``
    # variable is the only place that can hold it
    packed: Comptime = min_max(a, a + 1)
    lo, hi = packed
    return lo * 100 + hi


@func()
def mixed_multi(x: i32) -> tuple[i32, Large, Small]:
    return x, Large(x, x, x, x), Small(x, x + 1)


@func()
def use_mixed_multi(x: i32) -> i32:
    n, large, small = mixed_multi(x)
    return n * 10000 + small.b * 1000 + large.d * 100 + small.a


@func()
def big_first(x: i32) -> tuple[Large, i32]:
    return Large(x, x, x, x), x + 100


@func()
def use_big_first(x: i32) -> i32:
    large, n = big_first(x)
    return n * 1000 + large.d


@func()
def two_larges(x: i32) -> tuple[Large, Large]:
    return Large(x, x, x, x), Large(x + 1, x + 1, x + 1, x + 1)


@func()
def use_two_larges(x: i32) -> i32:
    p, q = two_larges(x)
    return p.d * 100 + q.d


@func()
def void_and_value(x: i32) -> tuple[void, i32]:
    return None, x + 1


@func()
def use_void_and_value(x: i32) -> i32:
    _, n = void_and_value(x)  # pyright: ignore[reportAssignmentType]
    return n


@func()
def python_pair(a: i64, b: i64) -> tuple[i64, i64]:
    # only ever called from Python, so the i32 wrappers above keep a single
    # specialization each
    return b, a


@func()
def nested_multi(x: i32) -> tuple[i32, tuple[i32, Large], Small]:
    return x, (x + 1, Large(x, x, x, x)), Small(x, x + 2)


@func()
def use_nested_multi(x: i32) -> i32:
    n, (m, large), small = nested_multi(x)
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def forward_nested_multi(x: i32) -> tuple[i32, tuple[i32, Large], Small]:
    return nested_multi(x)


@func()
def use_forward_nested_multi(x: i32) -> i32:
    n, (m, large), small = forward_nested_multi(x)
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def comptime_nested_multi(x: i32) -> i32:
    packed: Comptime = nested_multi(x)
    n, (m, large), small = packed
    return n * 100000 + m * 10000 + large.d * 100 + small.b


@func()
def nested_group_first(x: i32) -> tuple[tuple[Large, i32], i32]:
    return (Large(x, x, x, x), x + 1), x + 2


@func()
def use_nested_group_first(x: i32) -> i32:
    (large, n), m = nested_group_first(x)
    return large.d * 1000 + n * 100 + m


@func()
def python_nested_pair(a: i64, b: i64) -> tuple[i64, tuple[i64, i64], i64]:
    return a, (b, a + b), b


@func()
def bad_group_target(x: i32) -> i32:
    # a group the caller does not unpack into a tuple target: packing it into
    # a single place of its own is not supported yet (pyright cannot tell)
    _n, _inner, _small = nested_multi(x)
    return x


@func()
def bad_nested_arity(x: i32) -> i32:
    _n, (_m, _large), _small, _extra = nested_multi(x)  # pyright: ignore
    return x


@func()
def bad_nested_ellipsis(x: i32) -> tuple[i32, tuple[i32, ...]]:
    return x  # pyright: ignore[reportReturnType]


@func()
def bad_multi_assign(x: i32) -> i32:
    # a packed result into a variable that is not ``Comptime``: pyright cannot
    # tell, hence the ignore
    _packed = min_max(x, x + 1)
    return x


@func()
def bad_multi_arity(x: i32) -> i32:
    _a, _b, _c = min_max(x, x + 1)  # pyright: ignore
    return x


@func()
def bad_ellipsis_return(x: i32) -> tuple[i32, ...]:
    return x  # pyright: ignore[reportReturnType]


class SpyMultiReturnTest(TestCase):
    """A function annotated ``-> tuple[T1, T2, ...]`` returns several values:
    the lowered function returns one of them by value and delivers every other
    one through a hidden result pointer (see ``sval.make_ret_spec``), and the
    caller destructures the call or keeps the packed results in a ``Comptime``
    variable."""

    def test_destructuring_two_scalars(self) -> None:
        self.assertEqual(use_min_max(2, 7), 207)
        self.assertEqual(use_min_max(9, 3), 309)

    def test_lowered_signature_of_two_scalars(self) -> None:
        # the first result that fits in registers is returned by value, the
        # other one through a result pointer: ``fn(i32, i32, *i32) -> i32``
        self.assertEqual(use_min_max(2, 7), 207)
        args, ret = mir_signature(min_max)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(args, (i32_mir, i32_mir, mir.PointerType(i32_mir, False)))
        self.assertEqual(ret, i32_mir)

    def test_forwarding_a_multi_value_call(self) -> None:
        # ``return f()`` where both functions return two values writes each
        # result into the enclosing function's own result location
        self.assertEqual(use_forward_min_max(2, 7), 207)

    def test_packed_into_a_comptime_variable(self) -> None:
        self.assertEqual(comptime_multi(2), 203)

    def test_python_side_call(self) -> None:
        # the Python side allocates the storage of the result pointers and
        # returns the values as a tuple
        self.assertEqual(python_pair(2, 7), (7, 2))

    def test_nested_destructuring(self) -> None:
        # the middle result is a ``tuple[...]`` of its own: the leaves are
        # delivered separately and the caller regroups them
        self.assertEqual(use_nested_multi(3), 340305)

    def test_lowered_signature_of_a_nested_return(self) -> None:
        # the leaves are delivered depth first: the first scalar returns by
        # value, the other scalar and both aggregates through a pointer each
        self.assertEqual(use_nested_multi(3), 340305)
        args, ret = mir_signature(nested_multi)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(args[0], i32_mir)
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(i32_mir, False),
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(struct_mirror(Small), False),
            ),
        )

    def test_a_leading_group_still_returns_a_scalar_by_value(self) -> None:
        self.assertEqual(use_nested_group_first(3), 3405)
        args, ret = mir_signature(nested_group_first)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(i32_mir, False),
            ),
        )

    def test_forwarding_a_nested_multi_value_call(self) -> None:
        self.assertEqual(use_forward_nested_multi(3), 340305)

    def test_nested_packed_into_a_comptime_variable(self) -> None:
        self.assertEqual(comptime_nested_multi(3), 340305)

    def test_python_side_call_of_a_nested_return(self) -> None:
        self.assertEqual(python_nested_pair(2, 7), (2, (7, 9), 7))

    def test_a_group_target_that_is_not_a_tuple_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_group_target(3)

    def test_nested_arity_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_nested_arity(3)

    def test_a_nested_varying_number_of_results_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_nested_ellipsis(3)

    def test_aggregates_go_through_result_pointers(self) -> None:
        # a scalar returns by value, a large struct and a small one through a
        # result pointer each: ``fn(i32, *Large, *Small) -> i32``
        self.assertEqual(use_mixed_multi(3), 34303)
        args, ret = mir_signature(mixed_multi)
        self.assertEqual(ret, mir.IntType(32, True))
        self.assertEqual(
            args[1:],
            (
                mir.PointerType(struct_mirror(Large), False),
                mir.PointerType(struct_mirror(Small), False),
            ),
        )

    def test_a_large_leading_result_still_returns_the_scalar(self) -> None:
        # the by-value result need not be the first one
        self.assertEqual(use_big_first(3), 103003)
        args, ret = mir_signature(big_first)
        self.assertEqual(ret, mir.IntType(32, True))
        self.assertEqual(
            args[1:],
            (mir.PointerType(struct_mirror(Large), False),),
        )

    def test_every_result_through_a_pointer_returns_void(self) -> None:
        self.assertEqual(use_two_larges(3), 304)
        args, ret = mir_signature(two_larges)
        self.assertIs(ret, mir.VOID)
        self.assertEqual(len(args), 3)

    def test_a_zero_sized_result_is_the_unit_value(self) -> None:
        # a zero-sized result has no place of its own: it is delivered as its
        # unit value, and the other result still returns by value
        self.assertEqual(use_void_and_value(3), 4)
        args, ret = mir_signature(void_and_value)
        self.assertEqual(args, (mir.IntType(32, True),))
        self.assertEqual(ret, mir.IntType(32, True))

    def test_packing_into_a_plain_variable_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_multi_assign(3)

    def test_arity_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_multi_arity(3)

    def test_a_varying_number_of_results_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bad_ellipsis_return(3)


# ---------------------------------------------------------------------------
# the primitives of error handling: the payload union and the error union in
# the spy type system (``sval``), and their lowering (``mir``/``lower``)
# ---------------------------------------------------------------------------


def _union_lowering_fn() -> mir.Function:
    """A hand-built MIR function that writes an ``i32`` through a ``BitCast``
    of a union's address and branches on it with a ``Switch``:
    ``f(n) = 10 if n == 0, 20 if n == 1, else 30``."""
    i32_mir = mir.IntType(32, True)
    payload = mir.StructType(
        'union_payload',
        (mir.FormalArg('a', i32_mir), mir.FormalArg('b', i32_mir)),
    )
    union = mir.UnionType('union', payload)
    fn = mir.Function('union_lowering_test', [i32_mir], [None], i32_mir)
    entry = fn.entry
    slot = entry.emit(mir.Alloca(union))
    cell = entry.emit(mir.BitCast(slot, mir.PointerType(i32_mir)))
    entry.emit(mir.Store(cell, mir.Param(0, i32_mir)))
    value = entry.emit(mir.Load(cell))
    case0 = mir.BasicBlock()
    case1 = mir.BasicBlock()
    other = mir.BasicBlock()
    entry.emit(mir.Switch(value, other, ((0, case0), (1, case1))))
    case0.emit(mir.Ret(mir.Int(10, i32_mir)))
    case1.emit(mir.Ret(mir.Int(20, i32_mir)))
    other.emit(mir.Ret(mir.Int(30, i32_mir)))
    mir.normalize(fn)
    fn.is_complete = True
    return fn


class SpyErrorUnionPrimitiveTest(TestCase):
    """The type-system and lowering primitives of error handling: the result
    type spreads into a value, an error code and a payload union (see
    ``sval.make_ret_spec``), and the payload union is an untagged union read and
    written through a ``BitCast``."""

    def test_empty_error_union_is_the_unit_type(self) -> None:
        empty = sval.ResultType(sval.VoidType(), sval.FrozenArraySet())
        self.assertEqual(empty.tag_bits, 0)
        self.assertEqual(empty.code_type, sval.IntType(0, False))
        self.assertIsNotNone(empty.get_unit_value())
        self.assertIsNone(empty.to_mir_type(MIR_CACHE))

    def test_error_code_width_is_the_smallest(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        # a third, distinct exception (its size does not matter here)
        third = struct_type(ExternMixed)
        void = sval.VoidType()
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small,))).tag_bits, 1)
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small, large))).tag_bits, 2)
        # ``types`` is a set: a duplicate is dropped
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet((small, large, small))).tag_bits, 2
        )
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet((small, large, third))).tag_bits, 2
        )
        # a function that returns no value has no "no error" code: its i-th
        # exception is tagged ``i``, so a single one needs no code at all
        empty = sval.EmptyType()
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet()).tag_bits, 0)
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small,))).tag_bits, 0)
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small, large))).tag_bits, 1)
        self.assertEqual(
            sval.ResultType(empty, sval.FrozenArraySet((small, large, third))).tag_bits, 2
        )
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small,))).code_of(small), 0)
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small,))).code_of(small), 1)

    def test_payload_union_uses_the_largest_variant_and_is_interned(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        one = sval.UnionType(frozenset((small, large)))
        two = sval.UnionType(frozenset((small, large)))
        self.assertIs(one.storage_variant(MIR_CACHE), large)
        self.assertIs(one.to_mir_type(MIR_CACHE), two.to_mir_type(MIR_CACHE))
        self.assertIsNone(sval.UnionType(frozenset()).to_mir_type(MIR_CACHE))
        self.assertEqual(
            sval.UnionType(frozenset()).get_unit_value(),
            sval.UnionValue(sval.UnionType(frozenset())),
        )

    def test_make_ret_spec_spreads_the_error_union(self) -> None:
        small = struct_type(Small)
        i32_type = sval.IntType(32, True)
        type = sval.ResultType(i32_type, sval.FrozenArraySet((small,)))
        spec = sval.make_ret_spec(type, MIR_CACHE)
        assert isinstance(spec, sval.RetTuple)
        self.assertIs(spec.type, type)
        result, code, payload = spec.values
        assert isinstance(result, sval.RetValue)
        self.assertIs(result.type, i32_type)
        assert isinstance(code, sval.RetValue)
        assert isinstance(payload, sval.RetValue)
        self.assertEqual(code.type, sval.IntType(1, False))
        self.assertIsInstance(payload.type, sval.UnionType)
        # the i32 is returned by value; its code and payload through pointers
        self.assertFalse(result.via_result_ptr)
        self.assertTrue(code.via_result_ptr)
        self.assertTrue(payload.via_result_ptr)
        self.assertEqual(len(list(sval.iter_ret_leaves(spec))), 3)

    def test_a_value_less_function_returns_its_small_payload_by_value(self) -> None:
        small = struct_type(Small)
        # no value to return, one exception: the code is ``u0`` (zero-sized) and
        # the payload union takes the by-value slot
        type = sval.ResultType(sval.EmptyType(), sval.FrozenArraySet((small,)))
        spec = sval.make_ret_spec(type, MIR_CACHE)
        assert isinstance(spec, sval.RetTuple)
        _value, code, payload = spec.values
        assert isinstance(code, sval.RetValue)
        assert isinstance(payload, sval.RetValue)
        self.assertEqual(code.type, sval.IntType(0, False))
        self.assertFalse(code.via_result_ptr)
        self.assertFalse(payload.via_result_ptr)
        self.assertIs(sval.ret_returned_type(spec), payload.type)

    def test_error_union_subtyping_is_set_inclusion(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        void = sval.VoidType()
        empty = sval.ResultType(void, sval.FrozenArraySet())
        one = sval.ResultType(void, sval.FrozenArraySet((small,)))
        both = sval.ResultType(void, sval.FrozenArraySet((small, large)))
        self.assertTrue(empty.is_subtype_of(one))
        self.assertTrue(one.is_subtype_of(both))
        self.assertFalse(both.is_subtype_of(one))
        self.assertFalse(one.is_subtype_of(sval.ResultType(void, sval.FrozenArraySet((large,)))))

    def test_error_union_peer_is_the_union_in_delivery_order(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        void = sval.VoidType()
        peer = sval.ResultType(void, sval.FrozenArraySet((small,))).resolve_peer_type(
            sval.ResultType(void, sval.FrozenArraySet((large, small))),
        )
        self.assertEqual(peer, sval.ResultType(void, sval.FrozenArraySet((small, large))))
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet()).resolve_peer_type(
                sval.ResultType(void, sval.FrozenArraySet((small,))),
            ),
            sval.ResultType(void, sval.FrozenArraySet((small,))),
        )

    def test_success_is_the_value_of_the_empty_error_union(self) -> None:
        empty = sval.ResultType(sval.VoidType(), sval.FrozenArraySet())
        success = empty.get_unit_value()
        assert success is not None
        self.assertIsInstance(success, sval.Success)
        self.assertEqual(sval.type_of(success), empty)

    def test_lowering_a_bitcast_and_a_switch(self) -> None:
        fn = _union_lowering_fn()
        globals: StrBiMap[mir.GlobalValue] = StrBiMap()
        globals.add('union_lowering_test', fn)
        backend = LLVMBackend()
        native = backend.compile(set(), globals, _GLOBAL_CONTEXT.target_info())[fn]
        self.assertEqual(native.call(ctypes.c_int32(0)), 10)
        self.assertEqual(native.call(ctypes.c_int32(1)), 20)
        self.assertEqual(native.call(ctypes.c_int32(7)), 30)
        text = '\n'.join(native.print_all())
        # the LLVM IR pointers are untyped (``ptr``), so reinterpreting a
        # pointer type as another lowers to nothing
        self.assertNotIn('bitcast', text)
        self.assertIn('switch', text)


# ---------------------------------------------------------------------------
# raising and propagating exceptions: a function that may raise carries an
# error code and a payload next to its result (see ``sval.ResultType``),
# and a call carries the error to its caller (remapping the tag)
# ---------------------------------------------------------------------------


@struct()
class ErrorA(Exception):
    code: i32


@struct()
class ErrorB(Exception):
    n: i32


@func(exceptions=ErrorA)
def raise_a(n: i32) -> i32:
    if n < 0:
        raise ErrorA(7)
    return n + 1


@func(exceptions=ErrorB)
def raise_b(n: i32) -> i32:
    if n < 0:
        raise ErrorB(9)
    return n + 2


@func(exceptions=(ErrorA, ErrorB))
def forward_raise(n: i32) -> i32:
    return raise_a(n) + 10


@func(exceptions=(ErrorA,))
def catch_bound(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorA as e:
        return e.code + 100


@func(exceptions=(ErrorA, ErrorB))
def catch_multi(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorB:
        return 1
    except ErrorA as e:
        return e.code + 200
    except:  # noqa: E722
        return -1


@func(exceptions=(ErrorA, ErrorB))
def escape_through(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorB:
        return 1


@func()
def catch_without_declaring(n: i32) -> i32:
    # the try catches ``ErrorA`` in its own error space, so the function itself
    # never raises and declares nothing
    try:
        return raise_a(n)
    except ErrorA as e:
        return e.code + 500


@func()
def raise_undeclared(n: i32) -> i32:
    # the function declares no exception, so raising one is rejected when the
    # error is recorded (see ``HirRunner._add_function_exception``)
    if n < 0:
        raise ErrorA(7)
    return n + 1


# inlined plain Python functions: their bodies are emitted into their call
# sites, so an error they raise or let through belongs to the error space - and
# to the try blocks - enclosing the *caller*, exactly like an error raised at
# the call site itself (see ``HirRunner._active_try``)


def inline_raise(n: i32) -> i32:
    if n < 0:
        raise ErrorA(7)
    return n + 1


def inline_catch(n: i32) -> i32:
    # the inline body has a try of its own: a raise crosses two frames before
    # reaching the try (the body of the raise is inlined into this one)
    try:
        return inline_raise(n)
    except ErrorA as e:
        return e.code + 100


def inline_reraise(n: i32) -> i32:
    # the clause matches nothing the callee raises, so the error is re-raised
    # out of the body's own try - into the caller's, not out of the function
    try:
        return inline_raise(n)
    except ErrorB:
        return 1


def inline_forward(n: i32) -> i32:
    # a native call inside an inlined body: the error it carries is delivered
    # into the caller's space too
    return raise_a(n) + 10


def inline_no_return(n: i32) -> i32:
    # an inlined body whose every path raises: it never falls through to its
    # caller, so the caller's code after the call is dead (see
    # ``HirRunner._pop_frame``)
    raise ErrorA(7)


def inline_raise_either(n: i32) -> i32:
    # both paths of the body raise, so it does not fall through either
    if n < 0:
        raise ErrorA(7)
    raise ErrorB(9)


def inline_forward_no_return(n: i32) -> i32:
    # an inlined body whose only path raises through another inlined body
    return inline_no_return(n) + 3


def inline_raise_two(n: i32) -> i32:
    # an inlined body that may raise either of two exceptions
    if n > 10:
        raise ErrorA(7)
    if n > 0:
        raise ErrorB(9)
    return n + 1


@func()
def catch_inline_raise(n: i32) -> i32:
    try:
        return inline_raise(n)
    except ErrorA as e:
        return e.code + 100


@func()
def catch_inline_own_try(n: i32) -> i32:
    return inline_catch(n) + 1000


@func()
def catch_inline_reraise(n: i32) -> i32:
    try:
        return inline_reraise(n)
    except ErrorA as e:
        return e.code + 100


@func()
def catch_inline_call(n: i32) -> i32:
    try:
        return inline_forward(n) + 10
    except ErrorA as e:
        return e.code + 900


@func()
def catch_inline_in_branch(n: i32) -> i32:
    # the raise sits in a branch of the try body, so the other branch of the
    # inline body still falls into the code after the call
    try:
        if n < 0:
            inline_raise(n)
            return 1
        return 2
    except ErrorA as e:
        return e.code + 100


@func(exceptions=(ErrorA, ErrorB))
def raise_in_clause(n: i32) -> i32:
    # a raise inside a clause body belongs to the try *enclosing* the clause,
    # never to the clause's own try again
    try:
        raise_b(n)
    except ErrorB:
        try:
            raise_a(n)
        except ErrorB:
            return 1
        return 2
    return 3


@func(exceptions=(ErrorA,))
def forward_no_return(n: i32) -> i32:
    # the ``+ 10`` after the call is dead: the inlined body never returns
    return inline_no_return(n) + 10


@func(exceptions=(ErrorA,))
def nested_forward_no_return(n: i32) -> i32:
    return inline_forward_no_return(n) + 10


@func()
def catch_no_return(n: i32) -> i32:
    try:
        inline_no_return(n)
        return 1
    except ErrorA as e:
        return e.code + 100


@func()
def catch_raise_either(n: i32) -> i32:
    try:
        inline_raise_either(n)
        return 1
    except ErrorA as e:
        return e.code + 100
    except ErrorB as e:
        return e.n + 200


@func()
def raise_in_inline_undeclared(n: i32) -> i32:
    # the function proper declares nothing and the raise has no try to be
    # caught by: it is rejected when the error is tagged
    return inline_raise(n) + 10


@struct()
class ErrorC(Exception):
    code: i32


@func(exceptions="infer")
def inferred_raise(n: i32) -> i32:
    if n < 0:
        raise ErrorC(7)
    return n + 1


@func(exceptions="infer")
def inferred_return_first(n: i32) -> i32:
    # the successful return is typed before the raise that widens the set, so
    # the return path clears the error code before the set is even known
    if n >= 0:
        return n + 1
    raise ErrorC(7)


@func(exceptions="infer")
def inferred_forward(n: i32) -> i32:
    return inferred_raise(n) + 10


@func(exceptions="infer")
def inferred_catch(n: i32) -> i32:
    try:
        return inferred_raise(n)
    except ErrorC as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_two(n: i32) -> i32:
    return raise_a(n) + raise_b(n)


@func(exceptions="infer")
def nested_catch(n: i32) -> i32:
    try:
        try:
            return raise_a(n)
        except ErrorB:
            return -1
    except ErrorA as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_inline_catch(n: i32) -> i32:
    # the clause catches only ``ErrorA``, so only the ``ErrorB`` the inlined body
    # lets through is inferred into this function's own set
    try:
        return inline_raise_two(n)
    except ErrorA as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_no_return(n: i32) -> i32:
    # an inferred exception set leaves the declared return type in place, even
    # though the inlined body never returns and no value is ever stored
    return inline_no_return(n) + 10


@func(exceptions="infer")
def inferred_wider_return(n: i32) -> i64:
    # ... and it is the *declared* type that fixes the value result, not the
    # type of the values the body stores (``n + 1`` here is an ``i32``)
    if n < 0:
        raise ErrorA(7)
    return n + 1


@struct()
class ErrorBig(Exception):
    a: i64
    b: i64
    c: i64


@func()
def make_big(n: i32) -> ErrorBig:
    return ErrorBig(n, n + 1, n + 2)


@func(exceptions="infer")
def raise_big(n: i32) -> i32:
    # a call returning an aggregate is raised through a result pointer: the
    # delivery into the (still inferred) space is deferred to its commit
    if n < 0:
        raise make_big(n)
    return n + 1


@func(exceptions="infer")
def catch_big(n: i32) -> i32:
    try:
        raise_big(n)
    except ErrorBig as e:
        return e.a + e.b + e.c
    return 0


def call_with_error(handle: Any, arg: int) -> tuple[int, int, int]:
    """Compile ``handle`` (a Python-side call is rejected after compiling it)
    and invoke its native form directly, with the hidden error-code and payload
    pointers filled in by this helper.  Returns ``(result, error code, payload
    read as i32)``; the payload is only meaningful when the code is not zero."""
    entry = handle.get_entry()
    if len(entry.specs) == 0:
        with TestCase().assertRaises(SpyError):
            handle(arg)
    assert len(entry.specs) == 1, entry.specs
    instance = next(iter(entry.specs.values()))
    native = instance.wrapper_fn or instance.native_fn
    assert native is not None
    code = ctypes.c_uint8(255)
    payload = ctypes.c_int32(-1)
    result = native.call(
        ctypes.c_int32(arg),
        ctypes.c_void_p(ctypes.addressof(code)),
        ctypes.c_void_p(ctypes.addressof(payload)),
    )
    return int(result), int(code.value), int(payload.value)


class SpyErrorUnionTest(TestCase):
    """``raise`` delivers an exception into the function's error location - an
    error code and a payload - and ends the path; a call of a function that
    may raise carries the error to its caller, remapping the tag to the
    caller's own exception set."""

    def test_a_raising_function_lowers_to_a_code_and_a_payload(self) -> None:
        self._compile(raise_a, 1)
        args, ret = mir_signature(raise_a)
        i32_mir = mir.IntType(32, True)
        # ``fn(i32, *u1, *payload) -> i32``: one exception needs one bit for
        # its two tags (no error and the exception)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(args[0], i32_mir)
        self.assertEqual(args[1], mir.PointerType(mir.IntType(1, False), False))
        self.assertIsInstance(args[2], mir.PointerType)

    def test_a_function_that_raises_nothing_has_no_error_part(self) -> None:
        # the empty exception set is the unit type: an exception-free function's
        # lowered signature has neither an error code nor a payload
        self.assertEqual(catch_without_declaring(5), 6)
        args, ret = mir_signature(catch_without_declaring)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(args, (i32_mir,))
        self.assertEqual(ret, i32_mir)

    def test_raising_an_undeclared_exception_is_rejected(self) -> None:
        # the function declares no exception, so the error is rejected when it
        # is tagged (``HirRunner._add_function_exception``), with a hint about the declaration
        with self.assertRaises(CompileError) as ctx:
            raise_undeclared(5)
        self.assertIn('cannot raise', str(ctx.exception))

    def test_a_caller_widens_the_error_code(self) -> None:
        self._compile(forward_raise, 1)
        args, _ = mir_signature(forward_raise)
        # two exceptions need two bits (three tags)
        self.assertEqual(args[1], mir.PointerType(mir.IntType(2, False), False))

    def test_the_declared_exceptions_are_recorded(self) -> None:
        entry = raise_a.get_entry()  # pyright: ignore
        signature = entry.hir.signature
        assert signature.exceptions is not None
        self.assertEqual(list(signature.exceptions.values), [struct_type(ErrorA)])

    def test_calling_a_raising_function_from_python_is_rejected(self) -> None:
        with self.assertRaises(SpyError) as ctx:
            raise_a(1)
        self.assertIn('not supported', str(ctx.exception))

    def _compile(self, handle: Any, arg: int) -> None:
        # a call from Python compiles the function before the boundary rejects
        # it: the Python-side handling of errors is not implemented yet
        with self.assertRaises(SpyError):
            handle(arg)


class SpyTryExceptTest(TestCase):
    """``try``/``except`` catches the error a call (or a ``raise``) delivered:
    the clause whose type matches the error code runs, taking the payload for
    its ``as`` name; an error no clause matches re-raises to the enclosing
    handler, or out of the function."""

    def test_catching_the_normal_result_passes_through(self) -> None:
        result, code, _ = call_with_error(catch_bound, 5)
        self.assertEqual((result, code), (6, 0))

    def test_catching_binds_the_payload(self) -> None:
        result, code, _ = call_with_error(catch_bound, -3)
        self.assertEqual((result, code), (107, 0))

    def test_a_plain_clause_catches_without_binding(self) -> None:
        result, code, _ = call_with_error(catch_multi, -3)
        self.assertEqual((result, code), (207, 0))

    def test_multiple_clauses_pick_the_matching_one(self) -> None:
        self.assertEqual(call_with_error(catch_multi, 5)[:2], (6, 0))
        self.assertEqual(call_with_error(catch_multi, -3)[:2], (207, 0))

    def test_an_unmatched_error_escapes_the_try(self) -> None:
        _result, code, payload = call_with_error(escape_through, -3)
        # the two exceptions need two bits, and the code is non-zero: the
        # error escaped the ``try`` and reached the function's caller
        self.assertNotEqual(code, 0)
        self.assertEqual(payload, 7)

    def test_a_caught_exception_need_not_be_declared(self) -> None:
        # the function declares nothing, so Python can call it directly
        self.assertEqual(catch_without_declaring(5), 6)
        self.assertEqual(catch_without_declaring(-3), 507)


class SpyInlineErrorTest(TestCase):
    """Errors of an inlined body.  An inlined plain Python function is part of
    its caller, so its errors are delivered into - and caught by - the error
    space and the try blocks enclosing the *caller*: the error location and the
    handler are read from one innermost-*open*-try lookup across the inlined
    frames (see ``HirRunner._active_try``), which keeps the two in step."""

    def test_the_callers_try_catches_the_raise(self) -> None:
        self.assertEqual(catch_inline_raise(5), 6)
        self.assertEqual(catch_inline_raise(-3), 107)

    def test_the_bodys_own_try_catches_the_raise(self) -> None:
        # the raise is inlined *into* the body whose try catches it, so the two
        # sit in different frames
        self.assertEqual(catch_inline_own_try(5), 1006)
        self.assertEqual(catch_inline_own_try(-3), 1107)

    def test_a_body_with_no_falling_path_hands_its_error_to_the_caller(self) -> None:
        # the inlined body never reaches the caller's continuation, so the code
        # after the call - ``return 1`` here - is dead and must not be typed
        self.assertEqual(catch_no_return(3), 107)

    def test_a_body_whose_paths_all_raise_hands_its_error_to_the_caller(self) -> None:
        self.assertEqual(catch_raise_either(-3), 107)
        self.assertEqual(catch_raise_either(3), 209)

    def test_an_error_of_a_body_with_no_falling_path_escapes_the_function(self) -> None:
        # the error leaves the inlined body, and the function with it: the ``+
        # 10`` after the call is dead all the same
        _result, code, payload = call_with_error(forward_no_return, 3)
        self.assertEqual((code, payload), (1, 7))

    def test_the_same_through_two_levels_of_inlining(self) -> None:
        _result, code, payload = call_with_error(nested_forward_no_return, 3)
        self.assertEqual((code, payload), (1, 7))

    def test_an_unmatched_error_of_the_body_re_raises_to_the_caller(self) -> None:
        self.assertEqual(catch_inline_reraise(5), 6)
        self.assertEqual(catch_inline_reraise(-3), 107)

    def test_a_native_call_in_the_body_reaches_the_callers_try(self) -> None:
        # the error of the native call is carried into the caller's space, and
        # the caller's try is the handler the error propagates to
        self.assertEqual(catch_inline_call(5), 26)
        self.assertEqual(catch_inline_call(-3), 907)

    def test_a_raise_in_a_branch_of_the_body(self) -> None:
        # the raising branch leaves the try through the handler, while the other
        # branch of the body still falls into the code after the call
        self.assertEqual(catch_inline_in_branch(5), 2)
        self.assertEqual(catch_inline_in_branch(-3), 107)

    def test_a_raise_in_a_clause_body_is_not_caught_by_it_again(self) -> None:
        # the clause's own try is no longer open while its body is typed, so the
        # inner try's unmatched error belongs to the function's space
        self.assertEqual(call_with_error(raise_in_clause, 5), (3, 0, -1))
        _result, code, payload = call_with_error(raise_in_clause, -3)
        entry = raise_in_clause.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # it is the ``ErrorA`` the inner try raised, tagged in the declared
        # set's own order
        types = list(instance.ret_sig.exceptions.values)
        self.assertEqual((code, payload), (types.index(struct_type(ErrorA)) + 1, 7))

    def test_an_inferred_set_takes_only_the_error_the_try_lets_through(self) -> None:
        # an inferred function whose try catches one of the two exceptions its
        # inlined callee may raise: the caught one never leaves the try, the
        # other escapes it and is inferred into the function's own set
        self.assertEqual(call_with_error(inferred_inline_catch, 0)[:2], (1, 0))
        self.assertEqual(call_with_error(inferred_inline_catch, 20)[:2], (107, 0))
        _result, code, payload = call_with_error(inferred_inline_catch, 5)
        entry = inferred_inline_catch.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # only ``ErrorB`` is inferred, and being the only exception of the space
        # it has tag 1 (one bit for its two tags)
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorB)])
        self.assertEqual((code, payload), (1, 9))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_an_inlined_raise_needs_a_declaration_or_a_try(self) -> None:
        # without either, the error has nowhere to go: it is rejected when it is
        # tagged, naming the declaration that would allow it
        with self.assertRaises(CompileError) as ctx:
            raise_in_inline_undeclared(-3)
        self.assertIn('cannot raise', str(ctx.exception))


class SpyInferTest(TestCase):
    """An inferred exception set (``exceptions="infer"``): the space's set,
    its code width and its payload union follow from what the body actually
    raises or lets through, fixed when the function's analysis ends."""

    def test_an_inferred_set_lowers_to_a_code_and_a_payload(self) -> None:
        result, code, _ = call_with_error(inferred_raise, 5)
        self.assertEqual((result, code), (6, 0))
        result, code, payload = call_with_error(inferred_raise, -3)
        self.assertEqual((result, code, payload), (0, 1, 7))
        entry = inferred_raise.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorC)])
        args = instance.mir.args
        self.assertEqual(args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_an_inferred_error_propagates(self) -> None:
        self.assertEqual(call_with_error(inferred_forward, 5)[:2], (16, 0))
        self.assertEqual(call_with_error(inferred_forward, -3)[1:], (1, 7))

    def test_a_return_before_the_raise_still_clears_the_code(self) -> None:
        # the successful path is typed before the raise widens the set
        self.assertEqual(call_with_error(inferred_return_first, 5)[:2], (6, 0))
        self.assertEqual(call_with_error(inferred_return_first, -3)[1:], (1, 7))

    def test_a_fully_caught_inferred_function_is_callable(self) -> None:
        self.assertEqual(inferred_catch(5), 6)
        self.assertEqual(inferred_catch(-3), 107)

    def test_an_inferred_set_collects_every_callee(self) -> None:
        self.assertEqual(call_with_error(inferred_two, 5)[:2], (13, 0))
        self.assertEqual(call_with_error(inferred_two, -3)[1:], (1, 7))
        entry = inferred_two.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # first-delivery order, and two exceptions need two bits
        self.assertEqual(
            list(instance.ret_sig.exceptions.values),
            [struct_type(ErrorA), struct_type(ErrorB)],
        )
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(2, False), False))

    def test_an_inferred_set_keeps_the_declared_return_type(self) -> None:
        # the exceptions are inferred, the value type is the declared ``i64`` -
        # not the ``i32`` the body stores - so the value is the by-value result
        # and the error code goes through a pointer
        self.assertEqual(call_with_error(inferred_wider_return, 3)[:2], (4, 0))
        self.assertEqual(call_with_error(inferred_wider_return, -3)[1:], (1, 7))
        entry = inferred_wider_return.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorA)])
        self.assertEqual(instance.mir.ret_type, mir.IntType(64, True))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_a_body_that_never_returns_keeps_the_declared_return_type(self) -> None:
        # nothing is stored into the result location at all, so the result type
        # has to come from the declaration
        _result, code, payload = call_with_error(inferred_no_return, 3)
        self.assertEqual((code, payload), (1, 7))
        entry = inferred_no_return.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(instance.mir.ret_type, mir.IntType(32, True))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_nested_tries_hand_over_inward(self) -> None:
        # the inner ``try`` catches nothing, so its error is re-raised into the
        # outer one, which catches it
        self.assertEqual(nested_catch(5), 6)
        self.assertEqual(nested_catch(-3), 107)

    def test_raising_a_call_returning_an_aggregate(self) -> None:
        # ``raise make_big(n)`` hands a result pointer to the callee, deferred
        # until the inferred space's payload type is known
        self.assertEqual(catch_big(-3), -6)
        self.assertEqual(catch_big(5), 0)


@func()
def loop_forever(n: i32):
    # a body that never delivers a result and never raises: the function can
    # never return at all (a ``mir.NoReturn`` function)
    while True:
        n = n + 1


@func()
def call_loop_forever(n: i32) -> i32:
    # the ``return -1`` after the call is dead: the call never comes back, and
    # the declared return type keeps the caller's own result an ``i32``
    if n < 0:
        loop_forever(n)
        return -1
    return n + 1


@func(exceptions=(ErrorA,))
def always_raises(n: i32):
    # no value to return and one exception: the error code is zero-sized
    # (``u0``) and the payload union takes the by-value result
    raise ErrorA(n)


@func()
def catch_always_raises(n: i32) -> i32:
    try:
        always_raises(n)
        return 1
    except ErrorA as e:
        return e.code + 100


@func()
def never_declared(n: i32) -> Never:
    # ``-> Never`` declares that no value is ever returned: the body may not
    # return, and one that also raises nothing cannot return at all
    while True:
        n = n + 1


@func()
def call_never_declared(n: i32) -> i32:
    if n < 0:
        never_declared(n)
        return -1
    return n + 1


@func()
def forward_never(n: i32) -> Never:
    # a noreturn call ends the path: the implicit fallthrough of the body is
    # never reached (and so is not rejected)
    never_declared(n)


@func()
def call_forward_never(n: i32) -> i32:
    if n < 0:
        forward_never(n)
        return -1
    return n + 1


@func(exceptions=(ErrorA,))
def raises_never(n: i32) -> Never:
    # the declared empty value behaves like the inferred one of ``always_raises``
    raise ErrorA(n)


@func()
def catch_raises_never(n: i32) -> i32:
    try:
        raises_never(n)
        return 1
    except ErrorA as e:
        return e.code + 200


@func()
def returns_from_never(n: i32) -> Never:
    # the return is rejected: the function has no value to return (and the
    # store of its value into the empty result location is a no-op)
    return n  # pyright: ignore


@func()
def falls_through_never(n: i32) -> Never:  # pyright: ignore
    # falling off the end of the body is a returning path too
    n = n + 1


def inline_never(n: i32) -> Never:
    # an inlined body that always raises: its declaration holds, and the caller's
    # code after the call is dead (the body reaches no continuation)
    raise ErrorA(n)


@func(exceptions="infer")
def catch_inline_never(n: i32) -> i32:
    try:
        inline_never(n)
        return 1
    except ErrorA as e:
        return e.code + 300


def inline_returns_from_never(n: i32) -> Never:
    # an inlined body has no convention of its own, so its declaration is what
    # rejects the return
    return n  # pyright: ignore


@func()
def call_inline_returns_from_never(n: i32) -> i32:
    return inline_returns_from_never(n) + 1


def inline_falls_through_never(n: i32) -> Never:  # pyright: ignore
    n = n + 1


@func()
def call_inline_falls_through_never(n: i32) -> i32:
    inline_falls_through_never(n)
    return 1


@func(exceptions="infer")
def forward_always_raises(n: i32) -> i32:
    # the callee's payload union is a subset of this function's own (which also
    # holds ``ErrorB``): the by-value payload is written into the function's own
    # payload through a pointer reinterpretation (a union cannot be converted),
    # and the error is then only tagged (no copy)
    if n > 100:
        raise ErrorB(n)
    return always_raises(n) + 1


@func(exceptions="infer")
def catch_forward_always_raises(n: i32) -> i32:
    try:
        return forward_always_raises(n)
    except ErrorA as e:
        return e.code + 100


class SpyNoReturnTest(TestCase):
    """A function that cannot return: a body that never delivers a result and
    raises nothing lowers to a ``mir.NoReturn`` function - its LLVM form is
    marked ``noreturn`` and a call of it ends the block it sits in, so the code
    after the call is dead - while a value-less function that raises carries
    error codes that start at 0 (one exception needs no code at all)."""

    def test_a_value_less_function_without_errors_is_noreturn(self) -> None:
        self.assertEqual(call_loop_forever(5), 6)
        args, ret = mir_signature(loop_forever)
        self.assertIs(ret, mir.NORETURN)
        self.assertEqual(args, (mir.IntType(32, True),))
        # the dead ``return -1`` was never typed, so the caller's result is
        # still the declared ``i32``
        _args, caller_ret = mir_signature(call_loop_forever)
        self.assertEqual(caller_ret, mir.IntType(32, True))

    def test_the_noreturn_function_is_marked_in_the_ir(self) -> None:
        self.assertEqual(call_loop_forever(5), 6)
        entry = loop_forever.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        native = instance.wrapper_fn or instance.native_fn
        assert native is not None
        text = '\n'.join(native.print_all())
        self.assertIn('define void', text)
        self.assertIn('noreturn', text)
        # a call of it ends its block, and LLVM wants every block to end with an
        # explicit terminator
        self.assertIn('unreachable', text)

    def test_a_declared_never_function_is_noreturn(self) -> None:
        # ``-> Never`` declares the empty return type, which with an empty
        # exception set is a function that can never return
        self.assertEqual(call_never_declared(5), 6)
        args, ret = mir_signature(never_declared)
        self.assertIs(ret, mir.NORETURN)
        self.assertEqual(args, (mir.IntType(32, True),))

    def test_a_declared_never_function_matches_the_inferred_one(self) -> None:
        # the declared empty value lowers to the same result type as the one
        # ``always_raises`` infers from its body: no code, the payload by value
        self.assertEqual(catch_raises_never(7), 207)
        self.assertEqual(mir_signature(raises_never), mir_signature(always_raises))

    def test_a_never_function_forwarding_a_noreturn_call(self) -> None:
        # the call ends the path, so the body's fallthrough never runs
        self.assertEqual(call_forward_never(5), 6)
        _args, ret = mir_signature(forward_never)
        self.assertIs(ret, mir.NORETURN)

    def test_returning_from_a_never_function_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            returns_from_never(5)
        self.assertIn('cannot return', str(ctx.exception))

    def test_falling_through_a_never_function_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            falls_through_never(5)
        self.assertIn('fall off its end', str(ctx.exception))

    def test_an_inlined_never_body_may_not_return(self) -> None:
        self.assertEqual(catch_inline_never(7), 307)
        with self.assertRaises(CompileError) as ctx:
            call_inline_returns_from_never(5)
        self.assertIn('cannot return', str(ctx.exception))

    def test_an_inlined_never_body_may_not_fall_through(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            call_inline_falls_through_never(5)
        self.assertIn('fall off its end', str(ctx.exception))

    def test_a_value_less_function_with_one_exception_has_no_code(self) -> None:
        self.assertEqual(catch_always_raises(7), 107)
        args, ret = mir_signature(always_raises)
        # no error code at all: the i-th exception of a value-less function is
        # tagged ``i``, so a single one needs no code, and the payload union is
        # the by-value result
        self.assertEqual(args, (mir.IntType(32, True),))
        self.assertIsInstance(ret, mir.UnionType)

    def test_a_by_value_payload_fills_a_wider_union(self) -> None:
        # the caller's own payload union is wider than the callee's, so the
        # returned union is written through a reinterpreted pointer and the error
        # is carried on as a tag alone
        self.assertEqual(call_with_error(catch_forward_always_raises, 7)[:2], (107, 0))
        _result, code, payload = call_with_error(forward_always_raises, 7)
        self.assertEqual((code, payload), (2, 7))
        _result, code, payload = call_with_error(forward_always_raises, 105)
        self.assertEqual((code, payload), (1, 105))
        entry = forward_always_raises.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(
            list(instance.ret_sig.exceptions.values),
            [struct_type(ErrorB), struct_type(ErrorA)],
        )


# ---------------------------------------------------------------------------
# cross-context calls: a spy function compiled in one host context may reach
# the functions and structs another context registered.  A handle reached
# across contexts is re-bound to the calling context (see
# ``dsl._Context.resolve_global``), so every context parses, compiles and
# names its own copies - a struct declared below has one type object per
# context, and a spy body only ever sees its own context's objects.
# ---------------------------------------------------------------------------


@struct()
class CrossPair:
    a: i32
    b: i32

    def total(self) -> i32:
        return self.a + self.b


@struct()
class CrossBox[T]:
    v: T

    def get(self) -> T:
        return self.v


# a second host context: its own backend and symbol table, so it links the
# functions and structs it reaches into its own modules
_CROSS_CONTEXT = _Context(LLVMBackend())


@_CROSS_CONTEXT.func()
def cross_call_global_fn(a: i32, b: i32) -> i32:
    return smoke_test(a, b) + 1


@_CROSS_CONTEXT.func()
def cross_build_global_struct(x: i32) -> i32:
    p: CrossPair = CrossPair(x, x + 1)
    return p.total()


@_CROSS_CONTEXT.func()
def cross_build_global_generic(x: i32) -> i32:
    b: CrossBox[i32] = CrossBox(x)
    return b.get()


@_CROSS_CONTEXT.struct()
class CrossLocalStruct:
    a: i32

    # the method is registered in the *global* context (``func`` is its
    # decorator): the struct's own context re-binds it when it builds the head
    @func()
    def doubled(self) -> i32:
        return self.a * 2


@_CROSS_CONTEXT.func()
def cross_build_local_struct(x: i32) -> i32:
    c: CrossLocalStruct = CrossLocalStruct(x)
    return c.doubled()


@func()
def cross_target_fn(n: i32) -> i32:
    return n + 1


@_CROSS_CONTEXT.func()
def cross_call_target_fn(n: i32) -> i32:
    return cross_target_fn(n) * 2


class SpyCrossContextTest(TestCase):
    """A spy function compiled in a second host context reaches the functions
    and structs the global context registered.  Every handle is re-bound to
    the calling context, so each context compiles its own copies - the struct
    types of one context are not the types of another."""

    def test_calling_a_global_function(self) -> None:
        self.assertEqual(cross_call_global_fn(2, 3), 6)

    def test_constructing_a_global_struct(self) -> None:
        self.assertEqual(cross_build_global_struct(4), 9)

    def test_constructing_a_global_generic_struct(self) -> None:
        self.assertEqual(cross_build_global_generic(5), 5)

    def test_calling_a_global_method_on_a_local_struct(self) -> None:
        self.assertEqual(cross_build_local_struct(6), 12)

    def test_a_global_function_resolves_to_this_contexts_copy(self) -> None:
        # the call compiles ``smoke_test`` anew in the second context: its
        # entry there is not the global one, whose spec was compiled into the
        # global context's modules (and named in its symbol table)
        self.assertEqual(cross_call_global_fn(2, 3), 6)
        own = _CROSS_CONTEXT.resolve_global(smoke_test)
        assert own is not None
        self.assertIsNot(own, smoke_test.get_entry())  # pyright: ignore
        self.assertEqual(len(own.specs), 1)  # pyright: ignore

    def test_a_global_struct_resolves_to_this_contexts_type(self) -> None:
        own = _CROSS_CONTEXT.resolve_global(CrossPair)
        assert isinstance(own, sval.StructType)
        global_type = struct_type(CrossPair)
        self.assertIsNot(own, global_type)
        # the copy names the same struct: the same fields, with the same types
        for field in ('a', 'b'):
            self.assertEqual(own.field_type(field), global_type.field_type(field))

    def test_the_global_context_is_not_polluted(self) -> None:
        # the second context resolves a copy of the callee and specializes
        # *its* copy, so the global entry the global context registered keeps
        # its own specialization set (empty here - nothing in the global
        # context calls it)
        self.assertEqual(cross_call_target_fn(3), 8)
        global_entry = cross_target_fn.get_entry()  # pyright: ignore
        self.assertEqual(len(global_entry.specs), 0)  # pyright: ignore
        own = _CROSS_CONTEXT.resolve_global(cross_target_fn)
        assert own is not None
        self.assertEqual(len(own.specs), 1)  # pyright: ignore


@func()
def dst_target_fn(x: i32) -> i32:
    return x + 1


@func()
def dst_local(n: i32) -> i32:
    # a function value has no runtime representation of its own: a runtime
    # location (the local's memory) cannot hold one
    _f: spy_typeof(dst_target_fn) = dst_target_fn  # pyright: ignore
    return n


@func()
def dst_return():
    # ... and neither can a return deliver one
    return dst_target_fn


@func()
def opaque_roundtrip(x: i32) -> i32:
    # a pointer to an opaque type is a void pointer: it round-trips through
    # ``ptr_cast`` without ever naming a value of the opaque type
    p = ptr_cast(ref(x), Ptr[Opaque])
    q = ptr_cast(p, Ptr[i32])
    return q[...]


@func()
def unsized_ptr_index(x: i32) -> i32:
    # a pointer to an unsized array converts to the multi pointer of its
    # elements: ``*[?]T`` and ``*T`` carry the same address
    a = array(x, x + 1, x + 2, x + 3)
    p = ptr_cast(ref(a), Ptr[Array[i32, None]])  # pyright: ignore
    m: MultiPtr[i32] = p  # pyright: ignore
    return m[2]


@struct()
class FamCarrier:
    # a struct with an unsized-array field: a C flexible array member
    n: i32
    data: Array[i32, None]  # pyright: ignore


@func()
def fam_ptr_index(x: i32) -> i32:
    a = array(x, x + 1, x + 2, x + 3)
    w = ptr_cast(ref(a), Ptr[FamCarrier])
    m: MultiPtr[i32] = ref(w.data)  # pyright: ignore
    return m[1]


@func()
def fam_subscript_rejected(x: i32) -> i32:
    a = array(x, x + 1)
    w = ptr_cast(ref(a), Ptr[FamCarrier])
    return w.data[0]  # pyright: ignore


@func()
def fam_read_by_value(x: i32) -> i32:
    a = array(x, x + 1)
    w = ptr_cast(ref(a), Ptr[FamCarrier])
    return w.data  # pyright: ignore


@struct()
class ZeroArrayCarrier:
    # a zero-sized struct whose only field is a zero-length array: it has no MIR
    # mirror of its own, but its alignment follows the field's element type
    data: Array[i64, 0]  # pyright: ignore


@func()
def layout_size_i32() -> usize:
    return size_of(i32)


@func()
def layout_align_i32() -> usize:
    return align_of(i32)


@func()
def layout_size_ptr() -> usize:
    return size_of(Ptr[i32])


@func()
def layout_field_size() -> usize:
    return layout_of(i32).size


@func()
def layout_field_align() -> usize:
    return layout_of(i32).align


@func()
def layout_size_void() -> usize:
    return size_of(void)


@func()
def layout_align_void() -> usize:
    return align_of(void)


@func()
def layout_size_zero_array() -> usize:
    return size_of(Array[i64, 0])  # pyright: ignore


@func()
def layout_align_zero_array() -> usize:
    # ``align_of(T[0]) == align_of(T)``
    return align_of(Array[i64, 0])  # pyright: ignore


@func()
def layout_align_zero_array_ptr() -> usize:
    return align_of(Array[Ptr[i32], 0])  # pyright: ignore


@func()
def layout_size_zero_array_struct() -> usize:
    return size_of(ZeroArrayCarrier)


@func()
def layout_align_zero_array_struct() -> usize:
    return align_of(ZeroArrayCarrier)


@func()
def layout_size_unsized_array() -> usize:
    return size_of(Array[i32, None])  # pyright: ignore


@func()
def layout_align_unsized_array() -> usize:
    # ``align_of([?]T) == align_of(T)``
    return align_of(Array[i32, None])  # pyright: ignore


@func()
def layout_size_fam() -> usize:
    return size_of(FamCarrier)


@func()
def layout_align_fam() -> usize:
    return align_of(FamCarrier)


@func()
def layout_size_func_type() -> usize:
    # a function type is dynamically sized: it has no fixed layout
    return size_of(IntUnary)


@func()
def layout_align_unsized_opaque() -> usize:
    return align_of(Array[Opaque, None])  # pyright: ignore


@func()
def ptr_cast_index(x: i32) -> i32:
    # a pointer reinterpreted by ``ptr_cast`` indexes by its *new* pointee:
    # the stride is the element type, not the array the address came from
    a = array(x, x + 1, x + 2, x + 3)
    m = ptr_cast(ref(a), MultiPtr[i32])
    return m[2]


def _make_struct(*fields: tuple[str, sval.Type]) -> sval.StructType:
    head = sval.StructTypeHead('T')
    for name, type in fields:
        head.add_field(name, type)
    return head.specialize(())


def _fn_type() -> sval.FunctionType:
    return sval.FunctionType((), sval.VoidType())


class SpyTypeClassifyTest(TestCase):
    """``Type.classify``: how a spy type maps onto runtime code, computed from
    the type alone (it never asks for the MIR mirror).  The composite kinds
    follow their members, a compile-time-only member outranking a
    dynamically-sized one, which outranks the zero-sized case."""

    def test_ordinary_types(self) -> None:
        for type in (sval.BoolType(), sval.FloatType(64), sval.IntType(32, True),
                     _make_struct(('a', sval.IntType(32, True)))):
            self.assertEqual(type.classify(), sval.SpecialTypeKind.NONE)
            self.assertFalse(type.is_zst())

    def test_zero_sized_types(self) -> None:
        for type in (sval.VoidType(), sval.NullType(), sval.EmptyType(),
                     sval.IntType(0, False), sval.ValueType(3),
                     _make_struct(('v', sval.VoidType()))):
            self.assertTrue(type.is_zst())

    def test_compile_time_only_types(self) -> None:
        for type in (sval.TypeType(), sval.TypeVar('C'), sval.AnyIntType(),
                     sval.TupleType((sval.IntType(32, True),), False),
                     sval.ResultType(sval.VoidType(), sval.FrozenArraySet()), sval.AnyFunction()):
            self.assertEqual(type.classify(), sval.SpecialTypeKind.COMPTIME)
            self.assertFalse(type.is_zst())

    def test_a_function_type_is_dynamically_sized(self) -> None:
        self.assertEqual(_fn_type().classify(), sval.SpecialTypeKind.DST)
        self.assertFalse(_fn_type().is_zst())

    def test_a_pointer_is_always_sized(self) -> None:
        # ... even one to a dynamically-sized type: only its value form is
        # unsized
        self.assertEqual(sval.PointerType(_fn_type()).classify(), sval.SpecialTypeKind.NONE)
        self.assertEqual(
            sval.PointerType(sval.IntType(32, True), sval.TypeVar('C')).classify(),
            sval.SpecialTypeKind.COMPTIME,
        )

    def test_array_kinds(self) -> None:
        i32_type = sval.IntType(32, True)
        self.assertEqual(sval.ArrayType(i32_type, 3).classify(), sval.SpecialTypeKind.NONE)
        # a zero-length array holds no storage whatever its element type
        self.assertTrue(sval.ArrayType(_fn_type(), 0).is_zst())
        self.assertTrue(sval.ArrayType(sval.TypeType(), 0).is_zst())
        self.assertTrue(sval.ArrayType(sval.VoidType(), 3).is_zst())
        self.assertEqual(
            sval.ArrayType(_fn_type(), 3).classify(), sval.SpecialTypeKind.DST
        )
        self.assertEqual(
            sval.ArrayType(sval.TypeType(), 3).classify(), sval.SpecialTypeKind.COMPTIME
        )
        # a length that is not solved yet leaves the layout unknown
        self.assertEqual(
            sval.ArrayType(i32_type, sval.TypeVar('L')).classify(),
            sval.SpecialTypeKind.COMPTIME,
        )

    def test_option_kinds(self) -> None:
        self.assertEqual(
            sval.OptionType(sval.IntType(32, True)).classify(), sval.SpecialTypeKind.NONE
        )
        # an option is never zero-sized: it keeps whether a value is there
        self.assertEqual(
            sval.OptionType(sval.VoidType()).classify(), sval.SpecialTypeKind.NONE
        )
        self.assertEqual(
            sval.OptionType(_fn_type()).classify(), sval.SpecialTypeKind.DST
        )
        self.assertEqual(
            sval.OptionType(sval.TypeType()).classify(), sval.SpecialTypeKind.COMPTIME
        )

    def test_union_kinds(self) -> None:
        i32_type = sval.IntType(32, True)
        self.assertTrue(sval.UnionType(frozenset()).is_zst())
        self.assertTrue(sval.UnionType(frozenset((sval.VoidType(),))).is_zst())
        self.assertEqual(sval.UnionType(frozenset((i32_type,))).classify(), sval.SpecialTypeKind.NONE)
        self.assertEqual(
            sval.UnionType(frozenset((_fn_type(),))).classify(), sval.SpecialTypeKind.DST
        )
        self.assertEqual(
            sval.UnionType(frozenset((sval.TypeType(),))).classify(), sval.SpecialTypeKind.COMPTIME
        )

    def test_struct_kinds(self) -> None:
        i32_type = sval.IntType(32, True)
        self.assertTrue(_make_struct().is_zst())
        self.assertEqual(
            _make_struct(('f', _fn_type())).classify(), sval.SpecialTypeKind.DST
        )
        self.assertEqual(
            _make_struct(('t', sval.TypeType())).classify(), sval.SpecialTypeKind.COMPTIME
        )
        # the compile-time kind outranks the dynamically-sized one
        self.assertEqual(
            _make_struct(('f', _fn_type()), ('t', sval.TypeType())).classify(),
            sval.SpecialTypeKind.COMPTIME,
        )
        # ... and the dynamically-sized one outranks a field with storage
        self.assertEqual(
            _make_struct(('a', i32_type), ('f', _fn_type())).classify(),
            sval.SpecialTypeKind.DST,
        )

    def test_a_function_type_is_passed_by_reference(self) -> None:
        self.assertIs(sval.pass_by_ref(_fn_type(), MIR_CACHE), TriState.TRUE)
        self.assertIs(sval.pass_by_ref(sval.IntType(32, True), MIR_CACHE), TriState.FALSE)

    def test_the_mir_cache_is_the_contexts_own(self) -> None:
        # the interning table belongs to a host context, so two contexts do not
        # share the MIR types they intern
        self.assertIsNot(_GLOBAL_CONTEXT.mir_lower_cache, _CROSS_CONTEXT.mir_lower_cache)
        union = sval.UnionType(frozenset((sval.IntType(32, True),)))
        global_mir = union.to_mir_type(_GLOBAL_CONTEXT.mir_lower_cache)
        # ... while one context reuses the one it made
        self.assertIs(union.to_mir_type(_GLOBAL_CONTEXT.mir_lower_cache), global_mir)
        self.assertIsNot(union.to_mir_type(_CROSS_CONTEXT.mir_lower_cache), global_mir)


class SpyDstTest(TestCase):
    """A dynamically-sized type (a function type) has no runtime value: a
    runtime location cannot hold one and a return cannot deliver one."""

    def test_a_local_of_a_function_type_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            dst_local(3)
        self.assertIn('dynamically-sized', str(ctx.exception))

    def test_returning_a_function_value_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            dst_return()
        self.assertIn('dynamically-sized', str(ctx.exception))


class SpyOpaqueAndUnsizedTest(TestCase):
    """An opaque type (``syntax.Opaque``) and an unsized array
    (``syntax.Array[T, None]``): dynamically-sized types that have no value of
    their own.  A pointer to one is a value - a ``void*`` for an opaque type, a
    plain pointer to the elements of an unsized array - and an unsized-array (or
    opaque) field is a struct's flexible member (a C FAM), placed last."""

    def test_an_opaque_type_is_dynamically_sized(self) -> None:
        opaque = sval.OpaqueType()
        self.assertIs(opaque.classify(), sval.SpecialTypeKind.DST)
        self.assertIsNone(opaque.to_mir_type(MIR_CACHE))
        # a pointer to it is a void pointer
        self.assertEqual(
            sval.PointerType(opaque).to_mir_type(MIR_CACHE), mir.PointerType(mir.VOID)
        )

    def test_an_unsized_array_is_dynamically_sized(self) -> None:
        unsized = sval.ArrayType(sval.IntType(32, True), None)
        self.assertIs(unsized.classify(), sval.SpecialTypeKind.DST)
        self.assertIsNone(unsized.to_mir_type(MIR_CACHE))
        self.assertEqual(str(unsized), 'i32[?]')
        # a pointer to it is a plain pointer to its elements (``*[?]T -> *T``)
        self.assertEqual(
            sval.PointerType(unsized).to_mir_type(MIR_CACHE),
            mir.PointerType(mir.IntType(32, True)),
        )

    def test_a_pointer_to_an_unsized_array_converts_to_a_multi_pointer(self) -> None:
        i32_type = sval.IntType(32, True)
        single = sval.PointerType(sval.ArrayType(i32_type, None))
        multi = sval.PointerType(i32_type, variant=sval.PointerVariant.MULTI)
        self.assertTrue(single.is_subtype_of(multi))
        # ... but not the other way around
        self.assertFalse(multi.is_subtype_of(single))

    def test_the_opaque_pointer_round_trips(self) -> None:
        self.assertEqual(opaque_roundtrip(41), 41)

    def test_a_pointer_to_an_unsized_array_is_indexable(self) -> None:
        self.assertEqual(unsized_ptr_index(10), 12)

    def test_a_fam_field_is_placed_last(self) -> None:
        carrier = struct_type(FamCarrier)
        mirror = carrier.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        # ``n`` keeps its storage, ``data`` is the flexible member
        self.assertEqual([f.name for f in mirror.fields], ['n'])
        self.assertEqual(mirror.fam_type, mir.IntType(32, True))
        # ``data`` sits past ``n``: position 1 of the mirror
        self.assertEqual(carrier.get_field_mir_indices(MIR_CACHE), (0, 1))
        # the FAM adds no size, but its element alignment
        self.assertEqual(mir.estimated_size_of(mirror, 8), 4)
        self.assertEqual(mir.estimated_alignment_of(mirror, 8), 4)

    def test_the_fam_address_is_indexable(self) -> None:
        # ``ref(w.data)`` is ``*[?]i32``: it converts to ``MultiPtr[i32]`` and
        # names the elements that follow ``n``
        self.assertEqual(fam_ptr_index(10), 12)

    def test_an_unsized_array_is_not_subscripted(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            fam_subscript_rejected(1)
        self.assertIn('dynamically-sized array', str(ctx.exception))

    def test_a_fam_field_is_not_read_by_value(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            fam_read_by_value(1)
        self.assertIn('dynamically-sized', str(ctx.exception))

    def test_only_the_first_dst_field_is_reachable(self) -> None:
        i32_type = sval.IntType(32, True)
        s = _make_struct(
            ('n', i32_type),
            ('a', sval.ArrayType(i32_type, None)),
            ('b', sval.OpaqueType()),
        )
        # the first DST field is the FAM, the second is left with no position
        self.assertEqual(s.get_field_mir_indices(MIR_CACHE), (0, 1, None))
        mirror = s.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['n'])
        self.assertEqual(mirror.fam_type, mir.IntType(32, True))

    def test_a_nested_dst_struct_is_placed_last(self) -> None:
        i32_type = sval.IntType(32, True)
        inner = _make_struct(('n', i32_type), ('data', sval.ArrayType(i32_type, None)))
        outer = _make_struct(('x', i32_type), ('inner', inner))
        self.assertIs(outer.classify(), sval.SpecialTypeKind.DST)
        mirror = outer.get_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['x'])
        self.assertIs(mirror.fam_type, inner.get_mir_type(MIR_CACHE))

    def test_an_option_of_a_dst_puts_it_last(self) -> None:
        option = sval.OptionType(sval.OpaqueType())
        mirror = option.to_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.StructType)
        self.assertEqual([f.name for f in mirror.fields], ['tag'])
        self.assertIs(mirror.fam_type, mir.VOID)

    def test_a_dst_result_is_delivered_through_a_result_pointer(self) -> None:
        # a non-default convention: ``() -> Opaque`` lowers to ``(*void) -> void``
        fn = sval.FunctionType((), sval.OpaqueType())
        self.assertEqual(
            fn.to_mir_type(MIR_CACHE),
            mir.FunctionType((mir.PointerType(mir.VOID),), mir.VOID),
        )
        # a C convention forces the result by value, which a DST has no size for
        fn_c = sval.FunctionType((), sval.OpaqueType(), callconv='c')
        with self.assertRaises(CompileError):
            fn_c.to_mir_type(MIR_CACHE)

# ---------------------------------------------------------------------------
# tagged unions: ``A | B``, the tag test ``isinstance(u, A)`` / the unwrap
# ``isinstance(e := u, A)``, and ``match (e := u): case A(): ...`` (which binds
# the payload) / ``match u: case A(): ...`` (which tests only)
# ---------------------------------------------------------------------------


@struct()
class TU_A:
    x: i32


@struct()
class TU_B:
    y: i32


@struct()
class TU_C:
    z: i32


@struct()
class TU_Z1:
    pass


@struct()
class TU_Z2:
    pass


# the pointer variant's type, behind a name (``isinstance`` of a subscripted
# marker is spy syntax the Python type checker does not model)
TU_PTR_I32: Any = Ptr[i32]


@func()
def tu_make(sel: i32) -> TU_A | TU_B:
    if sel == 0:
        return TU_A(3)
    return TU_B(4)


@func()
def tu_tag_of(u: TU_A | TU_B) -> i32:
    # ``isinstance`` without an unwrap is an ordinary boolean test
    if isinstance(u, TU_A):
        return 1
    return 2


@func()
def tu_tag_of_make(sel: i32) -> i32:
    return tu_tag_of(tu_make(sel))


@func()
def tu_unwrap(u: TU_A | TU_B) -> i32:
    if isinstance(a := u, TU_A):
        return a.x
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_unwrap_make(sel: i32) -> i32:
    return tu_unwrap(tu_make(sel))


@func()
def tu_negated(u: TU_A | TU_B) -> i32:
    if not isinstance(u, TU_A):
        return 1
    return 0


@func()
def tu_negated_make(sel: i32) -> i32:
    return tu_negated(tu_make(sel))


@func()
def tu_widen(u: TU_A | TU_B) -> i32:
    # a value of a subset union converts to the wider one (the tag is remapped)
    v: TU_A | TU_B | TU_C = u
    if isinstance(c := v, TU_C):
        return c.z
    if isinstance(a := v, TU_A):
        return a.x
    if isinstance(b := v, TU_B):
        return b.y
    return -1


@func()
def tu_widen_make(sel: i32) -> i32:
    return tu_widen(tu_make(sel))


@func()
def tu_match(u: TU_A | TU_B) -> i32:
    match (_ := u):
        case TU_A():
            return 1
        case TU_B():
            return 2
    return -1


@func()
def tu_match_make(sel: i32) -> i32:
    return tu_match(tu_make(sel))


@func()
def tu_match_wildcard(u: TU_A | TU_B) -> i32:
    match (_ := u):
        case TU_A():
            return 1
        case _:
            return 2


@func()
def tu_match_wildcard_make(sel: i32) -> i32:
    return tu_match_wildcard(tu_make(sel))


@func()
def tu_match_tag(u: TU_A | TU_B) -> i32:
    # a plain-name subject: the tag is tested, no payload is bound
    match u:
        case TU_A():
            return 1
        case TU_B():
            return 2
    return -1


@func()
def tu_match_tag_make(sel: i32) -> i32:
    return tu_match_tag(tu_make(sel))


@func()
def tu_match_no_shadow(u: TU_A | TU_B) -> i32:
    # no payload is bound, so the subject name still names the union in the
    # case body, where it can be unwrapped
    match u:
        case TU_A():
            if isinstance(a := u, TU_A):
                return a.x
            return -1
        case _:
            return -2


@func()
def tu_match_no_shadow_make(sel: i32) -> i32:
    return tu_match_no_shadow(tu_make(sel))


@func()
def tu_scalar(sel: i32) -> i32:
    u: i32 | i64
    if sel == 0:
        u = 5
    else:
        u = 100
    if isinstance(i := u, i32):
        return i
    return -1


@func()
def tu_zst(sel: i32) -> i32:
    # every variant is zero-sized: only the tag is stored
    u: TU_Z1 | TU_Z2
    if sel == 0:
        u = TU_Z1()
    else:
        u = TU_Z2()
    if isinstance(_ := u, TU_Z1):
        return 1
    return 2


@func()
def tu_single() -> i32:
    # ``A | A`` de-duplicates to a single variant, which has no tag of its own
    u: TU_A | TU_A  # noqa: PYI016
    u = TU_A(9)
    if isinstance(a := u, TU_A):
        return a.x
    return -1


@func()
def tu_ptr(p: Ptr[i32]) -> i32:
    u: Ptr[i32] | i32
    u = p
    if isinstance(pp := u, TU_PTR_I32):
        return pp[...]
    return 0


@func()
def tu_ptr_driver() -> i32:
    x: i32 = 42
    return tu_ptr(ref(x))


@func()
def tu_comptime() -> i32:
    # a declared ``Comptime`` tagged union: its own place form (see
    # ``ComptimeTaggedUnionPtr``)
    u: Comptime[TU_A | TU_B] = TU_A(2)
    if isinstance(a := u, TU_A):
        return a.x
    if isinstance(b := u, TU_B):
        return b.y
    return -1


@func()
def tu_typeof(u: TU_A | TU_B) -> i32:
    if spy_typeof(u) == (TU_A | TU_B):
        return 1
    return 0


@func()
def tu_typeof_make(sel: i32) -> i32:
    return tu_typeof(tu_make(sel))


@func()
def tu_large_make(sel: i32) -> Large | TU_A:
    # a union large enough to be returned through a result pointer
    if sel == 0:
        return Large(1, 2, 3, 4)
    return TU_A(5)


@func()
def tu_large_use(sel: i32) -> i32:
    u: Large | TU_A = tu_large_make(sel)
    if isinstance(a := u, TU_A):
        return a.x
    return 0


@struct()
class TU_Holder:
    u: TU_A | TU_B
    n: i32


@func()
def tu_field(sel: i32) -> i32:
    h: TU_Holder
    if sel == 0:
        h = TU_Holder(TU_A(6), 1)
    else:
        h = TU_Holder(TU_B(7), 2)
    if isinstance(a := h.u, TU_A):
        return a.x
    return 0


@func()
def tu_field_write(sel: i32) -> i32:
    h: TU_Holder = TU_Holder(TU_A(1), 0)
    if sel == 0:
        h.u = TU_A(8)
    else:
        h.u = TU_B(9)
    if isinstance(a := h.u, TU_A):
        return a.x
    return -1


@func()
def tu_bad_variant() -> i32:
    u: TU_A | TU_B
    u = TU_A(1)
    if isinstance(u, TU_C):
        return 1
    return 0


@func()
def tu_bad_pattern() -> i32:
    u: TU_A | TU_B
    u = TU_A(1)
    match (_ := u):
        case TU_A(x):
            return x
        case _:
            return 0


# ---------------------------------------------------------------------------
# calling a struct's function through the class name, and inherited methods
# ---------------------------------------------------------------------------


class _ClassNamePlain:
    # a plain class (no ``@struct()``) used as a namespace of functions
    def bar() -> i32:  # pyright: ignore
        return 7

    def baz(y: i32) -> i32:  # pyright: ignore
        return y + 1


class _ClassNameBase:
    # a plain base class: a struct that derives from it inherits its methods
    def total(self, x: i32) -> i32:
        return self.a + x  # pyright: ignore


@struct()
class ClassNameLib:
    x: i32

    @staticmethod
    def foo() -> i32:
        return 5

    def add(self, y: i32) -> i32:
        return self.x + y


@struct()
class ClassNameChild(_ClassNameBase):
    a: i32


@struct()
class ClassNameGeneric[T]:
    v: T

    def get(self) -> T:
        return self.v


@func()
def class_name_static() -> i32:
    return ClassNameLib.foo()


@func()
def class_name_static_on_value(a: i32) -> i32:
    v = ClassNameLib(a)
    return v.foo()


@func()
def class_name_method_class_name(a: i32) -> i32:
    v = ClassNameLib(a)
    return ClassNameLib.add(v, 10)


@func()
def class_name_method_value(a: i32) -> i32:
    v = ClassNameLib(a)
    return v.add(10)


@func()
def class_name_plain() -> i32:
    return _ClassNamePlain.bar()  # pyright: ignore


@func()
def class_name_plain_method(a: i32) -> i32:
    return _ClassNamePlain.baz(a)  # pyright: ignore


@func()
def class_name_inherited_class_name(a: i32) -> i32:
    c = ClassNameChild(a)
    return ClassNameChild.total(c, 5)


@func()
def class_name_inherited_value(a: i32) -> i32:
    c = ClassNameChild(a)
    return c.total(5)


@func()
def class_name_missing() -> i32:
    return ClassNameLib.missing()  # pyright: ignore


@func()
def class_name_template(a: i32) -> i32:
    return ClassNameGeneric.get(a)  # pyright: ignore


class SpyClassNameCallTest(TestCase):
    """Calling a struct's function through the class name (``Foo.m(...)``,
    ``Foo[i32].m(x)``): the call passes every argument explicitly, with no
    implicit ``self``; a plain class (no ``@struct()``) works as a namespace
    of functions, and a struct inherits the methods of a plain base."""

    def test_a_static_method_through_the_class_name(self) -> None:
        self.assertEqual(class_name_static(), 5)

    def test_a_static_method_on_a_value_takes_no_self(self) -> None:
        self.assertEqual(class_name_static_on_value(9), 5)

    def test_a_method_through_the_class_name_passes_self_explicitly(self) -> None:
        self.assertEqual(class_name_method_class_name(1), 11)

    def test_a_method_on_a_value_still_passes_the_receiver(self) -> None:
        self.assertEqual(class_name_method_value(1), 11)

    def test_a_plain_class_is_a_namespace(self) -> None:
        self.assertEqual(class_name_plain(), 7)
        self.assertEqual(class_name_plain_method(41), 42)

    def test_a_struct_inherits_a_base_method_through_the_class_name(self) -> None:
        self.assertEqual(class_name_inherited_class_name(100), 105)

    def test_a_struct_inherits_a_base_method_on_a_value(self) -> None:
        self.assertEqual(class_name_inherited_value(100), 105)

    def test_a_missing_method_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            class_name_missing()

    def test_a_struct_template_has_to_be_specialized(self) -> None:
        with self.assertRaises(CompileError):
            class_name_template(1)

    def test_std_mem_allocator_methods_are_inherited(self) -> None:
        dyn = struct_type(DynamicAllocator)
        for name in ('alloc', 'resize', 'new', 'deinit', 'new_array', 'resize_array'):
            self.assertIsNotNone(dyn.get_method(name))


class SpyLayoutOfTest(TestCase):
    """``std.mem.layout_of`` (and the ``size_of``/``align_of`` built on it):
    the size and alignment of a spy type, measured by the lowerer through the
    ``mir.Sizeof``/``mir.Alignof`` instructions - folded at compile time for a
    zero-sized type, which has no MIR mirror of its own."""

    def test_primitives(self) -> None:
        self.assertEqual(layout_size_i32(), 4)
        self.assertEqual(layout_align_i32(), 4)
        self.assertEqual(layout_size_ptr(), MIR_CACHE.target.pointer_size)

    def test_the_layout_field_access(self) -> None:
        self.assertEqual(layout_field_size(), 4)
        self.assertEqual(layout_field_align(), 4)

    def test_a_zero_sized_type(self) -> None:
        self.assertEqual(layout_size_void(), 0)
        self.assertEqual(layout_align_void(), 1)

    def test_a_zero_length_array_aligns_to_its_element(self) -> None:
        # ``align_of(T[0]) == align_of(T)``
        self.assertEqual(layout_size_zero_array(), 0)
        self.assertEqual(layout_align_zero_array(), 8)
        self.assertEqual(layout_align_zero_array_ptr(), MIR_CACHE.target.pointer_size)

    def test_a_zero_sized_struct_aligns_to_its_member(self) -> None:
        # the struct has no MIR mirror, so its alignment is read off its
        # structure: the maximum over its fields
        self.assertEqual(layout_size_zero_array_struct(), 0)
        self.assertEqual(layout_align_zero_array_struct(), 8)

    def test_an_unsized_array_has_no_size_but_its_elements_alignment(self) -> None:
        # ``align_of([?]T) == align_of(T)``
        self.assertEqual(layout_size_unsized_array(), 0)
        self.assertEqual(layout_align_unsized_array(), 4)

    def test_a_flexible_member_struct(self) -> None:
        self.assertEqual(layout_size_fam(), 4)
        self.assertEqual(layout_align_fam(), 4)

    def test_a_type_with_no_fixed_layout_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            layout_size_func_type()
        with self.assertRaises(CompileError):
            layout_align_unsized_opaque()


class SpyTaggedUnionTest(TestCase):
    def test_tag_test_without_unwrap(self) -> None:
        self.assertEqual(tu_tag_of_make(0), 1)
        self.assertEqual(tu_tag_of_make(1), 2)

    def test_unwrap_binds_the_payload(self) -> None:
        self.assertEqual(tu_unwrap_make(0), 3)
        self.assertEqual(tu_unwrap_make(1), 4)

    def test_negated_test(self) -> None:
        self.assertEqual(tu_negated_make(0), 0)
        self.assertEqual(tu_negated_make(1), 1)

    def test_a_subset_union_converts(self) -> None:
        # the tag is remapped when the variants' order differs
        self.assertEqual(tu_widen_make(0), 3)
        self.assertEqual(tu_widen_make(1), 4)

    def test_match(self) -> None:
        self.assertEqual(tu_match_make(0), 1)
        self.assertEqual(tu_match_make(1), 2)

    def test_match_wildcard(self) -> None:
        self.assertEqual(tu_match_wildcard_make(0), 1)
        self.assertEqual(tu_match_wildcard_make(1), 2)

    def test_match_without_binding(self) -> None:
        # ``match u:`` (a plain-name subject) tests the tag only
        self.assertEqual(tu_match_tag_make(0), 1)
        self.assertEqual(tu_match_tag_make(1), 2)

    def test_match_subject_keeps_naming_the_union(self) -> None:
        # nothing is bound, so the subject name is still the union in the body
        self.assertEqual(tu_match_no_shadow_make(0), 3)
        self.assertEqual(tu_match_no_shadow_make(1), -2)

    def test_scalar_variants(self) -> None:
        # an untyped integer literal takes the first variant it fits
        self.assertEqual(tu_scalar(0), 5)
        self.assertEqual(tu_scalar(1), 100)

    def test_zero_sized_variants(self) -> None:
        self.assertEqual(tu_zst(0), 1)
        self.assertEqual(tu_zst(1), 2)

    def test_single_variant(self) -> None:
        self.assertEqual(tu_single(), 9)

    def test_pointer_variant(self) -> None:
        self.assertEqual(tu_ptr_driver(), 42)

    def test_compile_time_storage(self) -> None:
        self.assertEqual(tu_comptime(), 2)

    def test_typeof(self) -> None:
        self.assertEqual(tu_typeof_make(0), 1)

    def test_a_large_union_through_a_result_pointer(self) -> None:
        self.assertEqual(tu_large_use(0), 0)
        self.assertEqual(tu_large_use(1), 5)

    def test_a_tagged_union_field(self) -> None:
        self.assertEqual(tu_field(0), 6)
        self.assertEqual(tu_field(1), 0)

    def test_writing_a_tagged_union_field(self) -> None:
        self.assertEqual(tu_field_write(0), 8)
        self.assertEqual(tu_field_write(1), -1)

    def test_a_type_that_is_not_a_variant_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            tu_bad_variant()
        self.assertIn('not a variant', str(ctx.exception))

    def test_an_unsupported_match_pattern_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            tu_bad_pattern()


# ---------------------------------------------------------------------------
# deferred bodies (``with syntax.defer():``)
#
# A deferred body runs when the region it is declared in is left: ``defer`` on
# any exit, ``okdefer`` on a normal one and ``errdefer`` on an error one, in
# reverse declaration order.  The test functions observe the effect through a
# pointer to a local, whose address a wrapper passes with ``ref`` (a spy
# function never takes a ``Ptr`` from Python directly).
# ---------------------------------------------------------------------------


@func()
def defer_on_return(p: Ptr[i32]) -> i32:
    with defer():
        p[...] = p[...] + 1
    return 5


@func()
def call_defer_on_return(x: i32) -> i32:
    v = x
    r: i32 = defer_on_return(ref(v))
    return v * 100 + r


@func()
def defer_order(p: Ptr[i32]) -> i32:
    # the bodies run in reverse declaration order
    with defer():
        p[...] = p[...] * 10 + 1
    with defer():
        p[...] = p[...] * 10 + 2
    return 0


@func()
def call_defer_order(x: i32) -> i32:
    v = x
    r: i32 = defer_order(ref(v))
    return v * 100 + r


@func()
def defer_fallthrough(p: Ptr[i32]) -> None:
    with defer():
        p[...] = p[...] + 1


@func()
def call_defer_fallthrough(x: i32) -> i32:
    v = x
    defer_fallthrough(ref(v))
    return v


@func()
def defer_in_branch(p: Ptr[i32], c: i32) -> i32:
    # a ``return`` inside a branch runs the defer too
    with defer():
        p[...] = p[...] + 1
    if c > 0:
        return 1
    return 2


@func()
def call_defer_in_branch(x: i32, c: i32) -> i32:
    v = x
    r: i32 = defer_in_branch(ref(v), c)
    return v * 100 + r


@func()
def defer_on_break(p: Ptr[i32], n: i32) -> i32:
    i: i32 = 0
    while i < n:
        with defer():
            p[...] = p[...] + 1
        if i == 1:
            break
        i = i + 1
    return i


@func()
def call_defer_on_break(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_on_break(ref(v), n)
    return v * 100 + r


@func()
def defer_on_continue(p: Ptr[i32], n: i32) -> i32:
    i: i32 = 0
    while i < n:
        with defer():
            p[...] = p[...] + 1
        i = i + 1
        continue
    return i


@func()
def call_defer_on_continue(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_on_continue(ref(v), n)
    return v * 100 + r


@func()
def defer_ok_only(p: Ptr[i32]) -> i32:
    with okdefer():
        p[...] = p[...] + 1
    return 1


@func()
def call_defer_ok_only(x: i32) -> i32:
    v = x
    r: i32 = defer_ok_only(ref(v))
    return v * 100 + r


@func(exceptions=(ErrorA,))
def defer_err(p: Ptr[i32], n: i32) -> i32:
    # ``okdefer`` runs on the normal return, ``errdefer`` on the raise
    with okdefer():
        p[...] = p[...] + 1
    with errdefer():
        p[...] = p[...] + 100
    if n < 0:
        raise ErrorA(7)
    return n + 1


@func()
def call_defer_err(x: i32, n: i32) -> i32:
    v = x
    r: i32 = 0
    try:
        r = defer_err(ref(v), n)
    except ErrorA:
        r = -1
    return v * 100 + r


@func()
def defer_nested(p: Ptr[i32]) -> i32:
    # a defer body may declare a defer of its own: it runs when the body ends
    with defer():
        p[...] = p[...] * 10 + 1
        with defer():
            p[...] = p[...] * 10 + 2
    return 0


@func()
def call_defer_nested(x: i32) -> i32:
    v = x
    r: i32 = defer_nested(ref(v))
    return v * 100 + r


def inline_defer(p: Ptr[i32]) -> i32:
    # a plain (unregistered) function inlined at its call site: its defer runs
    # when the inlined body is left
    with defer():
        p[...] = p[...] + 1
    return 3


@func()
def call_inline_defer(x: i32) -> i32:
    v = x
    r: i32 = inline_defer(ref(v))
    return v * 100 + r


@func()
def defer_in_and_else(p: Ptr[i32], a: i32, b: i32) -> i32:
    # the ``and`` chain's condition opens a block; when the then body falls off
    # its end, the implicit ``break_if`` out of both blocks runs the defer
    r: i32 = 0
    if a > 0 and b > 0:
        with defer():
            p[...] = p[...] + 1
        r = 1
    else:
        r = 2
    return r


@func()
def call_defer_in_and_else(x: i32, a: i32, b: i32) -> i32:
    v = x
    r: i32 = defer_in_and_else(ref(v), a, b)
    return v * 100 + r


@func()
def defer_err_in_try_body(p: Ptr[i32], n: i32) -> i32:
    # the raise leaves the try body, so a defer declared in it runs
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise ErrorA(7)
        return 1
    except ErrorA:
        return 2


@func()
def call_defer_err_in_try_body(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_err_in_try_body(ref(v), n)
    return v * 100 + r


@func()
def defer_shared_raise(p: Ptr[i32], n: i32) -> i32:
    # two ``raise``s leave the same ``try`` body and reach the same typed clause:
    # they share one copy of the defer, and the payload phi of the clause merges
    # the two raises at the chain entry (each path still catches its own payload)
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise ErrorA(1)
        raise ErrorA(2)
    except ErrorA as e:
        return e.code


@func()
def call_defer_shared_raise(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_shared_raise(ref(v), n)
    return v * 100 + r


@func(exceptions=ErrorA)
def raise_a_never(n: i32) -> Never:
    # a call that always raises: it has no normal continuation
    raise ErrorA(n)


@func()
def defer_shared_call(p: Ptr[i32], n: i32) -> i32:
    # the same through the error dispatch of two calls: both share the chain and
    # deliver their payload to the clause's merged phi
    try:
        with errdefer():
            p[...] = p[...] + 100
        if n < 0:
            raise_a_never(1)
        raise_a_never(2)
    except ErrorA as e:
        return e.code


@func()
def call_defer_shared_call(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_shared_call(ref(v), n)
    return v * 100 + r


@func()
def defer_err_caught_outside(p: Ptr[i32], n: i32) -> i32:
    # the raise is caught inside the same function, so the function body is not
    # left and its errdefer does not run
    with errdefer():
        p[...] = p[...] + 100
    try:
        if n < 0:
            raise ErrorA(7)
        return 1
    except ErrorA:
        return 2


@func()
def call_defer_err_caught_outside(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_err_caught_outside(ref(v), n)
    return v * 100 + r


@func()
def defer_in_for(p: Ptr[i32], n: i32) -> i32:
    # the loop body sits inside the ``for`` desugaring's ``try``; its falling
    # end runs the defer each iteration
    total: i32 = 0
    for i in range(n):
        with defer():
            p[...] = p[...] + 1
        total = total + i
    return total


@func()
def call_defer_in_for(x: i32, n: i32) -> i32:
    v = x
    r: i32 = defer_in_for(ref(v), n)
    return v * 100 + r


@func()
def defer_with_if(p: Ptr[i32], c: i32) -> i32:
    # the body of a defer may branch: its copy carries the branch along
    with defer():
        if c > 0:
            p[...] = p[...] + 1
        else:
            p[...] = p[...] + 2
    return 0


@func()
def call_defer_with_if(x: i32, c: i32) -> i32:
    v = x
    r: i32 = defer_with_if(ref(v), c)
    return v * 100 + r


@func()
def defer_comptime_if(p: Ptr[i32]) -> i32:
    # the chosen branch of a compile-time ``if`` fell through: its defer runs
    if spy_typeof(p) == Ptr[i32]:
        with defer():
            p[...] = p[...] + 1
    return 0


@func()
def call_defer_comptime_if(x: i32) -> i32:
    v = x
    r: i32 = defer_comptime_if(ref(v))
    return v * 100 + r


@func()
def defer_bad_return() -> i32:
    with defer():
        return 1
    return 0


@func()
def defer_bad_break_outer() -> i32:
    i: i32 = 0
    while i < 3:
        with defer():
            break
        i = i + 1
    return i


class SpyDeferTest(TestCase):
    def test_a_defer_runs_on_a_return(self) -> None:
        self.assertEqual(call_defer_on_return(0), 105)

    def test_bodies_run_in_reverse_order(self) -> None:
        self.assertEqual(call_defer_order(0), 2100)

    def test_a_defer_runs_when_the_body_falls_through(self) -> None:
        self.assertEqual(call_defer_fallthrough(0), 1)

    def test_a_return_in_a_branch_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_in_branch(0, 1), 101)
        self.assertEqual(call_defer_in_branch(0, 0), 102)

    def test_a_break_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_on_break(0, 0), 0)
        self.assertEqual(call_defer_on_break(0, 1), 101)
        self.assertEqual(call_defer_on_break(0, 2), 201)

    def test_a_continue_runs_the_defer(self) -> None:
        self.assertEqual(call_defer_on_continue(0, 0), 0)
        self.assertEqual(call_defer_on_continue(0, 3), 303)

    def test_okdefer_runs_on_a_normal_exit(self) -> None:
        self.assertEqual(call_defer_ok_only(0), 101)

    def test_errdefer_runs_only_on_an_error(self) -> None:
        self.assertEqual(call_defer_err(0, 5), 106)
        self.assertEqual(call_defer_err(0, -1), 9999)

    def test_a_defer_inside_a_defer_body(self) -> None:
        self.assertEqual(call_defer_nested(0), 1200)

    def test_an_inlined_body_runs_its_defer(self) -> None:
        self.assertEqual(call_inline_defer(0), 103)

    def test_a_defer_in_an_and_chain_body(self) -> None:
        self.assertEqual(call_defer_in_and_else(0, 1, 1), 101)
        self.assertEqual(call_defer_in_and_else(0, 1, 0), 2)
        self.assertEqual(call_defer_in_and_else(0, 0, 1), 2)

    def test_a_defer_in_a_try_body_runs_on_a_caught_raise(self) -> None:
        self.assertEqual(call_defer_err_in_try_body(0, 5), 1)
        self.assertEqual(call_defer_err_in_try_body(0, -1), 10002)

    def test_two_raises_share_a_defer_and_keep_their_payloads(self) -> None:
        self.assertEqual(call_defer_shared_raise(0, -1), 10001)
        self.assertEqual(call_defer_shared_raise(0, 5), 10002)

    def test_two_calls_share_a_defer_and_keep_their_payloads(self) -> None:
        self.assertEqual(call_defer_shared_call(0, -1), 10001)
        self.assertEqual(call_defer_shared_call(0, 5), 10002)

    def test_an_errdefer_is_not_run_when_the_error_is_caught_inside(self) -> None:
        self.assertEqual(call_defer_err_caught_outside(0, 5), 1)
        self.assertEqual(call_defer_err_caught_outside(0, -1), 2)

    def test_a_defer_inside_a_for_body(self) -> None:
        self.assertEqual(call_defer_in_for(0, 0), 0)
        self.assertEqual(call_defer_in_for(0, 3), 303)

    def test_a_defer_body_with_a_branch(self) -> None:
        self.assertEqual(call_defer_with_if(0, 1), 100)
        self.assertEqual(call_defer_with_if(0, 0), 200)

    def test_a_defer_in_a_compile_time_branch(self) -> None:
        self.assertEqual(call_defer_comptime_if(0), 100)

    def test_a_return_inside_a_defer_body_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            defer_bad_return()

    def test_a_break_out_of_a_defer_body_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            defer_bad_break_outer()


# ---------------------------------------------------------------------------
# function types: ``@func_type`` declares a spy function type from a Protocol's
# ``__call__``, and a value of a pointer to one is called through the pointer
# (the only callable form: a function type itself is dynamically sized)
# ---------------------------------------------------------------------------


@struct()
class FnError(Exception):
    code: i32


@func_type(exceptions=(FnError,))
class Fallible(Protocol):
    # a spy-convention function pointer that may raise ``FnError``: the pointed-
    # to function carries the error code and payload like a compiled spy
    # function (the default calling convention)
    def __call__(self, n: i32) -> i32: ...


@func_type(callconv='c')
class CBinary(Protocol):
    # a C function pointer: every argument by value, the result by value, no
    # exceptions
    def __call__(self, a: i32, b: i32) -> i32: ...


@func_type(callconv='c')
class CWithDefault(Protocol):
    def __call__(self, a: i32, b: i32 = 10) -> i32: ...


@struct(extern_c=True)
class Wide:
    a: i64
    b: i64
    c: i64


@func_type(callconv='c')
class CGetWide(Protocol):
    def __call__(self) -> Wide: ...


@func_type(callconv='c', exceptions=(FnError,))
class CBad(Protocol):
    def __call__(self, n: i32) -> i32: ...


@func_type(exceptions=FnError)
class FallibleSingle(Protocol):
    # the single-type spelling of ``exceptions`` (see ``_normalize_exceptions``)
    def __call__(self, n: i32) -> i32: ...


@func(exceptions=FnError)
def raise_fn_error(n: i32) -> i32:
    if n < 0:
        raise FnError(1)
    return n + 1


@func()
def call_fallible(f: Fallible, n: i32) -> i32:
    # a DST parameter is passed by reference; the call carries the pointer's
    # error code and payload, routed to the try here
    try:
        return f(n)
    except FnError as e:
        return e.code + 1000


@func()
def call_c_binary(p: ConstPtr[CBinary], a: i32, b: i32) -> i32:
    # a pointer to a function type is an ordinary value
    return p[...](a, b)


@func()
def call_c_default(p: ConstPtr[CWithDefault], a: i32) -> i32:
    # the argument the call leaves out takes the pointer type's default
    return p[...](a)


@func(callconv='c', exceptions='infer')
def c_inferred_raise(n: i32) -> i32:
    # a C function may not raise; the inferred exception set is rejected once
    # the body is typed
    if n < 0:
        raise FnError(1)
    return n


@func(callconv='c')
def c_multi(n: i32) -> tuple[i32, i32]:
    # a C function may return only one value
    return n, n + 1


@decl_func("abs")
def c_abs(x: i32) -> i32:
    # an external function declared by its C link name (the default callconv)
    ...


@decl_func("bad", exceptions=FnError)
def c_bad_decl(x: i32) -> i32:
    # a C function may not declare exceptions: resolving this is rejected
    ...


@func()
def call_c_abs(x: i32) -> i32:
    # the declared name is a function pointer value: the call goes through it
    return c_abs(x)


@func()
def ptr_target_incr(n: i32) -> i32:
    return n + 1


@func()
def call_spy_func_ptr(n: i32) -> i32:
    # ``as_func_ptr`` materializes the address of a spy function
    p = as_func_ptr(spy_typeof(ptr_target_incr), ptr_target_incr)  # pyright: ignore
    return p[...](n)


@func_type()
class IntUnary(Protocol):
    def __call__(self, n: i32) -> i32: ...


@func()
def apply_int_unary(p: ConstPtr[IntUnary], n: i32) -> i32:
    return p[...](n)


@func()
def call_spy_func_ptr_typed(n: i32) -> i32:
    # the pointer of a spy function converts to a declared function-pointer type
    p = as_func_ptr(spy_typeof(ptr_target_incr), ptr_target_incr)  # pyright: ignore
    return apply_int_unary(p, n)


class SpyFuncTypeTest(TestCase):
    """``@func_type``: the annotations of a Protocol's ``__call__`` (its
    receiver dropped) become a ``sval.FunctionType``, resolved in the context
    that names the declaration."""

    def test_a_function_type_resolves_from_the_call_signature(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(Fallible)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual([a.name for a in t.args], ['n'])
        self.assertEqual(t.args[0].type, sval.IntType(32, True))
        self.assertEqual(t.return_type, sval.IntType(32, True))
        self.assertEqual(tuple(t.exceptions), (struct_type(FnError),))
        self.assertEqual(t.callconv, 'default')

    def test_a_default_of_a_function_type_parameter_is_carried(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CWithDefault)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual(t.args[1].default_value, 10)

    def test_a_pointer_to_a_function_type_is_a_pointer(self) -> None:
        t = sval.as_value(ConstPtr[CBinary], _GLOBAL_CONTEXT)
        assert isinstance(t, sval.PointerType)
        self.assertIsInstance(t.elem, sval.FunctionType)
        self.assertIs(t.is_const, True)

    def test_a_c_function_type_forces_every_argument_by_value(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CBinary)
        assert isinstance(t, sval.FunctionType)
        i32_mir = mir.IntType(32, True)
        # no by-ref argument and no result pointer: ``fn(i32, i32) -> i32``
        self.assertEqual(
            t.to_mir_type(MIR_CACHE), mir.FunctionType((i32_mir, i32_mir), i32_mir, 'c', False),
        )

    def test_a_c_function_type_returns_an_aggregate_by_value(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CGetWide)
        assert isinstance(t, sval.FunctionType)
        mirror = t.to_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.FunctionType)
        # a wide aggregate would go through a result pointer under the default
        # convention; the C one returns it by value (no hidden pointer argument)
        self.assertEqual(mirror.args, ())
        self.assertIsInstance(mirror.return_type, mir.StructType)

    def test_a_c_function_type_may_not_declare_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(CBad)
        self.assertIn('may not declare exceptions', str(ctx.exception))


class SpyFnPtrCallTest(TestCase):
    """A runtime function pointer is called through the address its value
    carries; the call is emitted like a call of a compiled function."""

    def test_calling_a_c_function_pointer(self) -> None:
        ptr_type = sval.as_value(ConstPtr[CBinary], _GLOBAL_CONTEXT)
        proto = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        callback = proto(lambda a, b: a + b)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_c_binary(spy_as(pointer, ptr_type), 2, 3), 5)  # pyright: ignore

    def test_the_default_of_a_function_pointer_is_filled_in(self) -> None:
        ptr_type = sval.as_value(ConstPtr[CWithDefault], _GLOBAL_CONTEXT)
        proto = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        callback = proto(lambda a, b: a + b)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_c_default(spy_as(pointer, ptr_type), 5), 15)  # pyright: ignore

    def test_calling_a_spy_function_pointer(self) -> None:
        # the default convention: the callee carries the error code and payload
        # through trailing pointers
        ptr_type = _GLOBAL_CONTEXT.resolve_global(Fallible)
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        )

        def success(n: int, code_ptr: int, payload_ptr: int) -> int:
            ctypes.cast(code_ptr, ctypes.POINTER(ctypes.c_uint8))[0] = 0
            return n + 1

        callback = proto(success)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_fallible(spy_as(pointer, ptr_type), 5), 6)  # pyright: ignore

    def test_the_error_of_a_spy_function_pointer_is_routed(self) -> None:
        ptr_type = _GLOBAL_CONTEXT.resolve_global(Fallible)
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        )

        def fail(n: int, code_ptr: int, payload_ptr: int) -> int:
            ctypes.cast(code_ptr, ctypes.POINTER(ctypes.c_uint8))[0] = 1
            ctypes.cast(payload_ptr, ctypes.POINTER(ctypes.c_int32))[0] = 7
            return 0

        callback = proto(fail)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_fallible(spy_as(pointer, ptr_type), 5), 1007)  # pyright: ignore


class SpyCallconvTest(TestCase):
    """A non-default calling convention forces every argument by value, the
    result by value, and forbids raising and multiple results."""

    def test_a_c_function_may_not_infer_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_inferred_raise(1)
        self.assertIn('may not raise', str(ctx.exception))

    def test_a_c_function_may_not_return_several_values(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_multi(1)
        self.assertIn('only one value', str(ctx.exception))


class SpyFrozenArraySetTest(TestCase):
    """``util.FrozenArraySet``: an immutable, ordered, deduplicated collection
    whose equality and hashing are by value (it may be a field of a frozen
    dataclass)."""

    def test_it_keeps_the_insertion_order_and_drops_duplicates(self) -> None:
        s = FrozenArraySet((1, 2, 1, 3, 2))
        self.assertEqual(s.values, (1, 2, 3))
        self.assertEqual(list(s), [1, 2, 3])
        self.assertEqual(len(s), 3)

    def test_membership_and_indexing(self) -> None:
        s: FrozenArraySet[str] = FrozenArraySet(('a', 'b', 'c'))
        self.assertIn('b', s)
        self.assertNotIn('z', s)
        self.assertEqual(s[0], 'a')
        self.assertEqual(s.index('c'), 2)
        self.assertIsNone(s.index_of('z'))
        with self.assertRaises(ValueError):
            s.index('z')

    def test_equality_and_hashing_are_by_value(self) -> None:
        # the order is part of the value (an error-code or tag order is)
        self.assertEqual(FrozenArraySet((1, 2)), FrozenArraySet((1, 2)))
        self.assertNotEqual(FrozenArraySet((1, 2)), FrozenArraySet((2, 1)))
        self.assertEqual(hash(FrozenArraySet((1, 2))), hash(FrozenArraySet((1, 2))))
        self.assertEqual(len({FrozenArraySet((1, 2)), FrozenArraySet((1, 2))}), 1)


class SpyExceptionsArgumentTest(TestCase):
    """``@func``/``@func_type`` take the exception set as a tuple or as a single
    exception type (``exceptions=ErrorA`` means ``exceptions=(ErrorA,)``)."""

    def test_a_func_type_accepts_a_single_exception(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(FallibleSingle)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual(tuple(t.exceptions), (struct_type(FnError),))

    def test_a_func_accepts_a_single_exception(self) -> None:
        # a call from Python compiles the function before the boundary rejects it
        with self.assertRaises(SpyError):
            raise_fn_error(1)
        entry = raise_fn_error.get_entry()  # pyright: ignore
        assert entry.hir.signature.exceptions is not None
        self.assertEqual(
            list(entry.hir.signature.exceptions.values), [struct_type(FnError)],
        )


class SpyDeclFuncTest(TestCase):
    """``@decl_func`` declares an external function by its link name: the
    decorated name resolves to a function pointer (``sval.DeclareFunction``)."""

    def test_a_declared_function_is_a_c_function_pointer(self) -> None:
        decl = _GLOBAL_CONTEXT.resolve_global(c_abs)
        assert isinstance(decl, sval.DeclareFunction)
        self.assertEqual(decl.linkname, 'abs')
        self.assertEqual(decl.type.callconv, 'c')
        self.assertEqual(decl.type.exceptions, sval.FrozenArraySet())
        self.assertEqual(decl.get_type(), sval.PointerType(decl.type, is_const=True))

    def test_calling_a_declared_c_function(self) -> None:
        self.assertEqual(call_c_abs(-5), 5)
        self.assertEqual(call_c_abs(7), 7)

    def test_it_lowers_to_an_extern_symbol(self) -> None:
        self.assertEqual(call_c_abs(-1), 1)
        entry = call_c_abs.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.native_fn is not None
        self.assertIn('abs', '\n'.join(instance.native_fn.print_all()))

    def test_a_c_declaration_may_not_declare_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(c_bad_decl)
        self.assertIn('may not declare exceptions', str(ctx.exception))


class SpyAsFuncPtrTest(TestCase):
    """``syntax.as_func_ptr`` materializes the address of a spy function as a
    ``ConstPtr`` of its function type."""

    def test_the_pointer_of_a_spy_function_can_be_called(self) -> None:
        self.assertEqual(call_spy_func_ptr(5), 6)

    def test_the_pointer_converts_to_a_declared_function_pointer_type(self) -> None:
        self.assertEqual(call_spy_func_ptr_typed(10), 11)


# ---------------------------------------------------------------------------
# compile-time reflection: ``std.reflect.type_info`` describes a compile-time
# type (``type``) as a ``TypeInfo`` value the interpreter builds while running
# the HIR - the ``type`` a body names, or the one ``spy.typeof`` yields
# ---------------------------------------------------------------------------


@struct()
class ReflectPoint:
    ab: i32
    c: i32


@struct()
class ReflectA:
    x: i32


@struct()
class ReflectB:
    y: i32


@struct()
class ReflectBox[T]:
    value: T


@func()
def reflect_int_bits() -> i32:
    info: Comptime = type_info(i32)
    if isinstance(v := info, IntType):
        return v.bits
    return -1


@func()
def reflect_int_signed() -> spy_bool:
    info: Comptime = type_info(i32)
    if isinstance(v := info, IntType):
        return v.signed
    return False


@func()
def reflect_typeof_bits(x: i64) -> i32:
    info: Comptime = type_info(spy_typeof(x))
    if isinstance(v := info, IntType):
        return v.bits
    return -1


@func()
def reflect_ptr_not_const() -> spy_bool:
    info: Comptime = type_info(Ptr[i32])
    if isinstance(v := info, PointerType):
        return v.is_const
    return True


@func()
def reflect_ptr_child() -> spy_bool:
    info: Comptime = type_info(Ptr[i32])
    if isinstance(v := info, PointerType):
        return v.child == i32
    return False


@func()
def reflect_array_child() -> spy_bool:
    info: Comptime = type_info(Array[i32, 3])
    if isinstance(v := info, ArrayType):
        return v.child == i32
    return False


@func()
def reflect_option_child() -> spy_bool:
    info: Comptime = type_info(Option[i64])
    if isinstance(v := info, OptionType):
        return v.child == i64
    return False


@func()
def reflect_field_count() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        return v.fields.length
    return -1


@func()
def reflect_field_name_byte() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[0]
        name: Comptime = field.name
        return name.ptr[0]
    return -1


@func()
def reflect_field_name_arith() -> i32:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        field: Comptime = v.fields.ptr[0]
        name: Comptime = field.name
        p: Comptime = name.ptr + 1
        return p[...]
    return -1


@func()
def reflect_tag_count() -> i32:
    info: Comptime = type_info(ReflectA | ReflectB)
    if isinstance(v := info, TaggedUnionType):
        return v.types.length
    return -1


@func()
def reflect_plain_has_head() -> spy_bool:
    info: Comptime = type_info(ReflectPoint)
    if isinstance(v := info, StructType):
        return v.head is not None
    return False


@func()
def reflect_generic_has_head() -> spy_bool:
    info: Comptime = type_info(ReflectBox[i32])
    if isinstance(v := info, StructType):
        return v.head is not None
    return False


class SpyReflectTest(TestCase):
    """``std.reflect.type_info`` at compile time."""

    def test_an_integer_type_reflects_its_width_and_signedness(self) -> None:
        self.assertEqual(reflect_int_bits(), 32)
        self.assertTrue(reflect_int_signed())

    def test_the_type_of_a_value_reflects_like_the_type(self) -> None:
        self.assertEqual(reflect_typeof_bits(0), 64)

    def test_a_pointer_type_reflects_its_child_and_constness(self) -> None:
        self.assertFalse(reflect_ptr_not_const())
        self.assertTrue(reflect_ptr_child())

    def test_an_array_type_reflects_its_element_type(self) -> None:
        self.assertTrue(reflect_array_child())

    def test_an_option_type_reflects_its_child_type(self) -> None:
        self.assertTrue(reflect_option_child())

    def test_a_struct_type_reflects_its_fields(self) -> None:
        self.assertEqual(reflect_field_count(), 2)
        # the first field is named ``ab``: its bytes are readable through the
        # ``ConstSlicePtr[u8]`` the name is stored as
        self.assertEqual(reflect_field_name_byte(), ord('a'))
        self.assertEqual(reflect_field_name_arith(), ord('b'))

    def test_a_tagged_union_type_reflects_its_variants(self) -> None:
        self.assertEqual(reflect_tag_count(), 2)

    def test_a_generic_struct_reflects_its_head(self) -> None:
        # a non-generic struct has no template head; a specialization of a
        # generic one names the head it was specialized from
        self.assertFalse(reflect_plain_has_head())
        self.assertTrue(reflect_generic_has_head())


# ---------------------------------------------------------------------------
# operators: the complete integer/float operator set (true division, floor
# division, exponentiation and the bitwise/shift operators), the augmented
# form of every one of them, and struct operator overloading
# ---------------------------------------------------------------------------


@func()
def int_true_div(a: i32, b: i32) -> f64:
    return a / b


@func()
def int_floor_div(a: i32, b: i32) -> i32:
    return a // b


@func()
def float_floor_div(a: f64, b: f64) -> f64:
    return a // b


@func()
def int_pow_const(a: i32) -> i32:
    return a ** 3


@func()
def int_pow_neg_const(a: i32) -> f64:
    return a ** -2


@func()
def int_pow_runtime(a: i32, e: i32) -> f64:
    return a ** e


@func()
def uint_pow_runtime(a: u64, e: u64) -> f64:
    return a ** e


@func()
def float_pow(a: f64, b: f64) -> f64:
    return a ** b


@func()
def float_pow_over_the_unroll_cap(a: f64) -> f64:
    # a compile-time exponent above ``CompileVars.max_exp_unroll`` becomes a
    # runtime loop instead of an unfolded sequence
    return a ** 5000


@func()
def comptime_pow() -> i32:
    return 2 ** 10


@func()
def bitwise_ops(a: i32, b: i32) -> i32:
    return (a | b) + (a & b) + (a ^ b) + (a << 1) + (a >> 1) + ~a


@func()
def bitwise_assign(a: i32, b: i32) -> i32:
    a |= b
    a &= b
    a ^= b
    a <<= 1
    a >>= 1
    return a


@func()
def augmented_ops(a: i32, b: i32) -> f64:
    a += b
    a -= b
    a *= b
    a //= b
    return a / b


@struct()
class OpVec:
    """A struct that overloads the arithmetic, comparison, unary and bitwise
    operators through magic methods (undecorated, so each is inlined)."""

    v: i32

    def __add__(self, other: OpVec) -> OpVec:
        return OpVec(self.v + other.v)

    def __radd__(self, k: i32) -> OpVec:
        return OpVec(self.v + k)

    def __iadd__(self, other: OpVec) -> OpVec:  # pyright: ignore  # noqa: PYI034
        self.v = self.v + other.v
        return self

    def __eq__(self, other: OpVec) -> spy_bool:  # pyright: ignore
        return self.v == other.v

    def __lt__(self, other: OpVec) -> spy_bool:
        return self.v < other.v

    def __neg__(self) -> OpVec:
        return OpVec(-self.v)

    def __bool__(self) -> spy_bool:
        return self.v != 0

    def __and__(self, other: OpVec) -> OpVec:
        return OpVec(self.v & other.v)


@func()
def overload_add(a: i32, b: i32) -> i32:
    return (OpVec(a) + OpVec(b)).v


@func()
def overload_reflected_add(a: i32) -> i32:
    return (a + OpVec(5)).v


@func()
def overload_inplace_add(a: i32, b: i32) -> i32:
    x: OpVec = OpVec(a)
    x += OpVec(b)
    return x.v


@func()
def overload_less_than(a: i32, b: i32) -> spy_bool:
    return OpVec(a) < OpVec(b)


@func()
def overload_eq(a: i32, b: i32) -> spy_bool:
    return OpVec(a) == OpVec(b)


@func()
def overload_negate(a: i32) -> i32:
    return (-OpVec(a)).v


@func()
def overload_bool(a: i32) -> spy_bool:
    # ``__bool__`` answers the ``AsBool`` an ``if`` condition (or a ``not``)
    # runs, so this exercises it directly
    if OpVec(a):  # noqa: SIM103
        return True
    return False


@func()
def overload_bitand(a: i32, b: i32) -> i32:
    return (OpVec(a) & OpVec(b)).v


class SpyOperatorTest(TestCase):
    """The complete integer/float operator set, the augmented form of every
    operator, and the struct magic methods."""

    def test_integer_division_is_true_division(self) -> None:
        self.assertAlmostEqual(int_true_div(7, 2), 3.5)
        self.assertAlmostEqual(int_true_div(-7, 2), -3.5)

    def test_integer_floor_division(self) -> None:
        self.assertEqual(int_floor_div(7, 2), 3)
        self.assertEqual(int_floor_div(-7, 2), -4)
        self.assertEqual(int_floor_div(7, -2), -4)
        self.assertEqual(int_floor_div(-7, -2), 3)

    def test_float_floor_division(self) -> None:
        self.assertAlmostEqual(float_floor_div(7.5, 2.0), 3.0)
        self.assertAlmostEqual(float_floor_div(-7.5, 2.0), -4.0)

    def test_compile_time_exponent(self) -> None:
        self.assertEqual(int_pow_const(3), 27)
        self.assertEqual(comptime_pow(), 1024)

    def test_negative_compile_time_exponent(self) -> None:
        self.assertAlmostEqual(int_pow_neg_const(2), 0.25)

    def test_runtime_signed_exponent(self) -> None:
        self.assertAlmostEqual(int_pow_runtime(2, 10), 1024.0)
        self.assertAlmostEqual(int_pow_runtime(2, -3), 0.125)
        self.assertAlmostEqual(int_pow_runtime(5, 0), 1.0)

    def test_runtime_unsigned_exponent(self) -> None:
        self.assertAlmostEqual(uint_pow_runtime(spy_as(2, u64), spy_as(5, u64)), 32.0)

    def test_float_exponent(self) -> None:
        self.assertAlmostEqual(float_pow(4.0, 0.5), 2.0)
        self.assertAlmostEqual(float_pow(2.0, 10.0), 1024.0)

    def test_an_exponent_over_the_unroll_cap_is_a_runtime_loop(self) -> None:
        self.assertAlmostEqual(float_pow_over_the_unroll_cap(1.0001), 1.6486800559, places=6)

    def test_bitwise_operators(self) -> None:
        # 12|10=14, 12&10=8, 12^10=6, 12<<1=24, 12>>1=6, ~12=-13
        self.assertEqual(bitwise_ops(0b1100, 0b1010), 45)

    def test_bitwise_augmented_assignment(self) -> None:
        self.assertEqual(bitwise_assign(0b1100, 0b1010), 0)

    def test_every_operator_has_an_augmented_form(self) -> None:
        self.assertAlmostEqual(augmented_ops(12, 4), 3.0)

    def test_struct_add(self) -> None:
        self.assertEqual(overload_add(2, 3), 5)

    def test_struct_reflected_add(self) -> None:
        self.assertEqual(overload_reflected_add(10), 15)

    def test_struct_inplace_add(self) -> None:
        self.assertEqual(overload_inplace_add(2, 3), 5)

    def test_struct_comparisons(self) -> None:
        self.assertTrue(overload_less_than(2, 3))
        self.assertFalse(overload_less_than(3, 2))
        self.assertTrue(overload_eq(3, 3))
        self.assertFalse(overload_eq(3, 4))

    def test_struct_negate(self) -> None:
        self.assertEqual(overload_negate(7), -7)

    def test_struct_bool(self) -> None:
        self.assertFalse(overload_bool(0))
        self.assertTrue(overload_bool(1))

    def test_struct_bitand(self) -> None:
        self.assertEqual(overload_bitand(0b1100, 0b1010), 8)


# ---------------------------------------------------------------------------
# the undefined literal: ``std.core.undefined`` is a value of any type, leaving
# its destination undefined (see ``sval.UndefinedType``/``sval.UntypedUndefined``)
# ---------------------------------------------------------------------------


def identity_i32(x: i32) -> i32:
    return x


@func()
def undef_scalar(n: i32) -> i32:
    x: i32 = undefined()
    spy_typeof(x)
    return n


@func()
def undef_pointer(n: i32) -> i32:
    p: Ptr[i32] = undefined()
    spy_typeof(p)
    return n


@func()
def undef_struct(n: i32) -> i32:
    v: ReflectA = undefined()
    spy_typeof(v)
    return n


@func()
def undef_option(n: i32) -> i32:
    o: Option[i32] = undefined()
    spy_typeof(o)
    return n


@func()
def undef_union(n: i32) -> i32:
    u: ReflectA | ReflectB = undefined()
    spy_typeof(u)
    return n


@func()
def undef_argument(n: i32) -> i32:
    # the literal converts to the parameter's type
    identity_i32(undefined())
    return n


@func()
def undef_inferred(n: i32, c: i32) -> i32:
    # the slot takes the peer type of the value and the literal
    x = n if c > 0 else undefined()
    if c == 12345:
        return x
    return n


@func()
def undef_comptime() -> i32:
    x: Comptime = undefined()
    spy_typeof(x)
    return 1


@func()
def undef_comptime_option() -> i32:
    o: Comptime[Option[i32]] = undefined()
    spy_typeof(o)
    return 2


@func()
def undef_comptime_union() -> i32:
    u: Comptime[ReflectA | ReflectB] = undefined()
    spy_typeof(u)
    return 3


@func()
def undef_return(n: i32) -> i32:
    # the literal converts at the function's result location
    if n > 0:
        return undefined()
    return n


@func()
def call_undef_return(n: i32) -> i32:
    undef_return(n)
    return n


@func()
def undef_if() -> i32:
    if undefined():
        return 1
    return 0


@func()
def undef_comptime_if() -> i32:
    x: Comptime[spy_bool] = undefined()
    if x:
        return 1
    return 0


@func()
def undef_while() -> i32:
    while undefined():
        return 1
    return 0


@func()
def undef_and_condition(a: i32) -> i32:
    if a > 0 and undefined():
        return 1
    return 0


class SpyUndefinedTest(TestCase):
    """The ``std.core.undefined`` literal: a value of any type."""

    def test_the_literal_converts_to_a_scalar(self) -> None:
        self.assertEqual(undef_scalar(7), 7)

    def test_the_literal_converts_to_a_pointer(self) -> None:
        self.assertEqual(undef_pointer(7), 7)

    def test_the_literal_converts_to_a_struct(self) -> None:
        self.assertEqual(undef_struct(7), 7)

    def test_the_literal_converts_to_an_option(self) -> None:
        self.assertEqual(undef_option(7), 7)

    def test_the_literal_converts_to_a_tagged_union(self) -> None:
        self.assertEqual(undef_union(7), 7)

    def test_the_literal_converts_to_an_argument_type(self) -> None:
        self.assertEqual(undef_argument(7), 7)

    def test_the_literal_peers_with_a_value(self) -> None:
        self.assertEqual(undef_inferred(7, 1), 7)
        self.assertEqual(undef_inferred(7, -1), 7)

    def test_the_literal_is_a_compile_time_value(self) -> None:
        self.assertEqual(undef_comptime(), 1)

    def test_a_compile_time_option_is_left_undefined(self) -> None:
        self.assertEqual(undef_comptime_option(), 2)

    def test_a_compile_time_union_is_left_undefined(self) -> None:
        self.assertEqual(undef_comptime_union(), 3)

    def test_the_literal_converts_at_a_return_location(self) -> None:
        self.assertEqual(call_undef_return(5), 5)
        self.assertEqual(call_undef_return(-5), -5)

    def test_an_undefined_if_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_if()

    def test_a_compile_time_undefined_bool_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_comptime_if()

    def test_an_undefined_while_condition_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_while()

    def test_an_undefined_and_operand_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            undef_and_condition(1)


all_tests = [
    SpyFunctionCallTest,
    SpyIfExprTest,
    SpyWhileTest,
    SpyForTest,
    SpyMethodSelfTest,
    SpyInlineLoopTest,
    SpyUnrollForTest,
    SpyAnnotationTest,
    SpyComptimeMarkerTest,
    SpyStructTest,
    SpyStructDefaultsTest,
    SpyComptimeStructTest,
    SpyComptimeArrayTest,
    SpyStructMirrorTest,
    SpyGenericStructTest,
    SpyPointerTest,
    SpyMultiPointerTest,
    SpySlicePtrTest,
    SpyTypeValueTest,
    SpyAggregateArgumentTest,
    SpyArrayTest,
    SpyTupleTest,
    SpyZeroSizedResultTest,
    SpyCompileLogTest,
    SpyBoolOpTest,
    SpyOptionTest,
    SpyOptionNestingTest,
    SpyOptionUnwrapTest,
    SpyTaggedUnionTest,
    SpyMultiReturnTest,
    SpyErrorUnionPrimitiveTest,
    SpyErrorUnionTest,
    SpyTryExceptTest,
    SpyInferTest,
    SpyNoReturnTest,
    SpyCrossContextTest,
    SpyTypeClassifyTest,
    SpyDstTest,
    SpyDeferTest,
    SpyFuncTypeTest,
    SpyFnPtrCallTest,
    SpyCallconvTest,
    SpyFrozenArraySetTest,
    SpyExceptionsArgumentTest,
    SpyDeclFuncTest,
    SpyAsFuncPtrTest,
    SpyReflectTest,
    SpyOperatorTest,
    SpyUndefinedTest,
]

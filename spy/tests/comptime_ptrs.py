from typing import Protocol
from unittest import TestCase

from ..compiler import i32, syntax
from ..compiler.dsl import func, func_type, struct
from ..compiler.syntax import Comptime, Option, Ptr, as_func_ptr, ref

# ---------------------------------------------------------------------------
# compile-time pointers as arguments of a *non-inline* function: a compile-time
# pointer (the address of a ``Comptime`` variable, of its field, or of a
# compile-time aggregate/tuple/option/union) passed to a compiled function is
# carried as a ``fn.ComptimePtrArg`` - the pointee is copied into the callee on
# every call, so the callee can never alias (nor write back to) the caller's
# place.  The same applies to a closure capture of a compiled closure.
# ---------------------------------------------------------------------------


@struct()
class Little:
    a: i32
    b: i32


@struct()
class V_A:
    x: i32


@struct()
class V_B:
    y: i32


# -- scalar pointee ---------------------------------------------------------


@func()
def take_i32(p: Ptr[i32]) -> i32:
    return p[...] + 1


@func()
def bump_i32(p: Ptr[i32]) -> i32:
    p[...] = p[...] + 1
    return p[...]


@func()
def comptime_ptr_scalar() -> i32:
    n: Comptime = 5
    return take_i32(ref(n))


@func()
def comptime_ptr_scalar_is_copied() -> i32:
    # the callee writes to its own copy: ``n`` is left alone
    n: Comptime = 5
    bumped = bump_i32(ref(n))
    return bumped * 100 + n


@func()
def comptime_ptr_in_a_comptime_local() -> i32:
    n: Comptime = 5
    p: Comptime = ref(n)
    return take_i32(p)


# -- aggregate pointee ------------------------------------------------------


@func()
def take_little(p: Ptr[Little]) -> i32:
    return p[...].a + p[...].b


@func()
def comptime_ptr_aggregate() -> i32:
    s: Comptime = Little(1, 2)
    return take_little(ref(s))


@func()
def comptime_ptr_aggregate_in_a_local() -> i32:
    s: Comptime = Little(1, 2)
    p: Comptime = ref(s)
    return take_little(p)


# -- nested pointee ---------------------------------------------------------


@func()
def take_ptr_ptr_i32(p: Ptr[Ptr[i32]]) -> i32:
    return p[...][...] + 1


@func()
def comptime_ptr_nested() -> i32:
    n: Comptime = 5
    p: Comptime = ref(n)
    return take_ptr_ptr_i32(ref(p))


# -- tuple pointee ----------------------------------------------------------


@func()
def take_tuple_ptr(p: Ptr[tuple[i32, i32]]) -> i32:
    return p[...][0] + p[...][1]


@func()
def comptime_ptr_tuple() -> i32:
    t: Comptime = (3, 4)
    return take_tuple_ptr(ref(t))


# -- option pointee ---------------------------------------------------------


@func()
def take_option_ptr(p: Ptr[Option[i32]]) -> i32:
    if (v := p[...]) is not None:
        return v + 1
    return -1


@func()
def comptime_ptr_option_present() -> i32:
    o: Comptime[Option[i32]] = 5
    return take_option_ptr(ref(o))


@func()
def comptime_ptr_option_absent() -> i32:
    o: Comptime[Option[i32]] = None
    return take_option_ptr(ref(o))


# -- tagged union pointee ---------------------------------------------------


@func()
def take_union_ptr(p: Ptr[V_A | V_B]) -> i32:
    if isinstance(v := p[...], V_A):
        return v.x + 1
    return -1


@func()
def comptime_ptr_union() -> i32:
    u: Comptime[V_A | V_B] = V_A(5)
    return take_union_ptr(ref(u))


@func()
def comptime_ptr_union_other_variant() -> i32:
    u: Comptime[V_A | V_B] = V_B(9)
    return take_union_ptr(ref(u))


# -- a compile-time variable as a ``Comptime`` parameter --------------------


@func()
def take_comptime(i: Comptime[i32]) -> i32:
    return i + 1


@func()
def comptime_var_to_comptime_param() -> i32:
    i: Comptime = 5
    return take_comptime(i)


# -- compiled closures capturing a compile-time place -----------------------


@func()
def compiled_closure_captures_a_comptime_aggregate() -> i32:
    s: Comptime = Little(1, 2)

    @syntax.closure(inline=False)
    def f() -> i32:
        return s.a + s.b

    return f()


@func()
def compiled_closure_mutates_its_captured_copy() -> i32:
    # the closure writes to its own copy of ``s``: the outer ``s`` is left alone
    s: Comptime = Little(1, 2)

    @syntax.closure(inline=False)
    def f() -> i32:
        s.a = 100
        return s.a + s.b

    return f() * 1000 + s.a


# -- a compile-time capture baked into a function pointer -------------------


@func_type()
class OneFn(Protocol):
    def __call__(self, x: i32) -> i32: ...


@func()
def comptime_capture_as_func_ptr() -> i32:
    # a compile-time capture needs no runtime parameter, so it is baked into the
    # specialization and the closure is still an ordinary runtime function
    n: Comptime = 5

    @syntax.closure(inline=False)
    def f(x: i32) -> i32:
        return x + n

    p = as_func_ptr(OneFn, f)
    return p[...](1)


class SpyComptimePtrTest(TestCase):
    """Compile-time pointers passed to a non-inline function (and closure
    captures of a compiled closure): the pointee is copied at every call, so the
    callee never aliases - nor writes back to - the caller's place."""

    def test_a_scalar_pointee(self) -> None:
        self.assertEqual(comptime_ptr_scalar(), 6)

    def test_the_pointee_is_copied(self) -> None:
        # ``bump_i32`` increments its own copy; the compile-time ``n`` is unchanged
        self.assertEqual(comptime_ptr_scalar_is_copied(), 605)

    def test_a_pointer_in_a_comptime_local(self) -> None:
        self.assertEqual(comptime_ptr_in_a_comptime_local(), 6)

    def test_an_aggregate_pointee(self) -> None:
        self.assertEqual(comptime_ptr_aggregate(), 3)

    def test_an_aggregate_pointer_in_a_comptime_local(self) -> None:
        self.assertEqual(comptime_ptr_aggregate_in_a_local(), 3)

    def test_a_nested_pointee(self) -> None:
        self.assertEqual(comptime_ptr_nested(), 6)

    def test_a_tuple_pointee(self) -> None:
        self.assertEqual(comptime_ptr_tuple(), 7)

    def test_an_option_pointee(self) -> None:
        self.assertEqual(comptime_ptr_option_present(), 6)
        self.assertEqual(comptime_ptr_option_absent(), -1)

    def test_a_tagged_union_pointee(self) -> None:
        self.assertEqual(comptime_ptr_union(), 6)
        self.assertEqual(comptime_ptr_union_other_variant(), -1)

    def test_a_compile_time_var_as_a_comptime_param(self) -> None:
        self.assertEqual(comptime_var_to_comptime_param(), 6)

    def test_a_compiled_closure_captures_a_comptime_aggregate(self) -> None:
        self.assertEqual(compiled_closure_captures_a_comptime_aggregate(), 3)

    def test_a_compiled_closure_mutates_its_captured_copy(self) -> None:
        self.assertEqual(compiled_closure_mutates_its_captured_copy(), 102001)

    def test_a_comptime_capture_baked_into_a_function_pointer(self) -> None:
        self.assertEqual(comptime_capture_as_func_ptr(), 6)


all_tests = [
    SpyComptimePtrTest,
]

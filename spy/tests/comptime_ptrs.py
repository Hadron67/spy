from typing import Literal, Protocol
from unittest import TestCase

from ..compiler import (
    Array,
    CompileError,
    Comptime,
    ConstComptime,
    Option,
    Ptr,
    array,
    closure,
    const_comptime,
    func,
    func_type,
    i32,
    ref,
    struct,
    sval,
    usize,
)
from ..std.core import as_func_ptr, coerce

# ---------------------------------------------------------------------------
# compile-time pointers crossing a *non-inline* boundary: a compile-time pointer
# to a mutable ``Comptime`` variable is a ``ComptimePtr`` into a mutable
# compile-time box, and lowering it to a runtime value/address is now rejected
# (``cannot use a pointer to a mutable compile-time variable as a runtime
# value``).  A *const* pointer - the address of a ``ConstComptime`` variable -
# does cross: it is lowered to a global static constant the callee points at
# (see the behaviour doc §4.2/§4.3).  The old ``fn.ComptimePtrArg`` (which
# copied the pointee into the callee on every call) is gone, so the callee
# aliases the caller's global constant instead of a copy.
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
def comptime_ptr_scalar() -> i32:
    n: Comptime = 5
    const_n: ConstComptime[i32] = n
    return take_i32(ref(const_n))


@func()
def comptime_ptr_in_a_comptime_local() -> i32:
    n: Comptime = 5
    const_n: ConstComptime[i32] = n
    p: Comptime = ref(const_n)
    return take_i32(p)


# (the old ``comptime_ptr_scalar_is_copied`` / ``test_the_pointee_is_copied`` is
# gone: the callee no longer gets a copy of the pointee - ``fn.ComptimePtrArg``
# was removed - so there is no write-back-not-visible behavior to test)


# -- aggregate pointee ------------------------------------------------------


@func()
def take_little(p: Ptr[Little]) -> i32:
    return p[...].a + p[...].b


@func()
def comptime_ptr_aggregate() -> i32:
    s: Comptime = Little(1, 2)
    const_s: ConstComptime[Little] = s
    return take_little(ref(const_s))


@func()
def comptime_ptr_aggregate_in_a_local() -> i32:
    s: Comptime = Little(1, 2)
    const_s: ConstComptime[Little] = s
    p: Comptime = ref(const_s)
    return take_little(p)


# -- nested pointee ---------------------------------------------------------


@func()
def take_ptr_ptr_i32(p: Ptr[Ptr[i32]]) -> i32:
    return p[...][...] + 1


@func()
def comptime_ptr_nested() -> i32:
    n: Comptime = 5
    const_n: ConstComptime[i32] = n
    p: ConstComptime[Ptr[i32]] = ref(const_n)
    return take_ptr_ptr_i32(ref(p))


# -- tuple pointee ----------------------------------------------------------


@func()
def take_tuple_ptr(p: Ptr[tuple[i32, i32]]) -> i32:
    return p[...][0] + p[...][1]


@func()
def comptime_ptr_tuple() -> i32:
    # the elements are spelled out: the pointee type of ``take_tuple_ptr`` is
    # ``tuple[i32, i32]``, and a plain ``(3, 4)`` would keep them untyped (see
    # task §18.1)
    t: Comptime = (coerce(3, i32), coerce(4, i32))
    const_t: ConstComptime = t
    return take_tuple_ptr(ref(const_t))


# -- option pointee ---------------------------------------------------------


@func()
def take_option_ptr(p: Ptr[Option[i32]]) -> i32:
    if (v := p[...]) is not None:
        return v + 1
    return -1


@func()
def comptime_ptr_option_present() -> i32:
    o: Comptime[Option[i32]] = 5
    const_o: ConstComptime[Option[i32]] = o
    return take_option_ptr(ref(const_o))


@func()
def comptime_ptr_option_absent() -> i32:
    o: Comptime[Option[i32]] = None
    const_o: ConstComptime[Option[i32]] = o
    return take_option_ptr(ref(const_o))


# -- tagged union pointee ---------------------------------------------------
# a fully compile-time tagged union is a compile-time aggregate: build the
# variant in a ``Comptime`` variable, then take a const pointer to it


@func()
def take_union_ptr(p: Ptr[V_A | V_B]) -> i32:
    if isinstance(v := p[...], V_A):
        return v.x + 1
    return -1


@func()
def comptime_ptr_union() -> i32:
    v: Comptime = V_A(5)
    u: Comptime[V_A | V_B] = v
    const_u: ConstComptime[V_A | V_B] = u
    return take_union_ptr(ref(const_u))


@func()
def comptime_ptr_union_other_variant() -> i32:
    v: Comptime = V_B(9)
    u: Comptime[V_A | V_B] = v
    const_u: ConstComptime[V_A | V_B] = u
    return take_union_ptr(ref(const_u))


# -- a runtime index into a compile-time array ------------------------------
# a read-only compile-time constant (a ``ConstComptime``) is materialized into a
# global constant when a runtime index reads it (its variable stays compile-time);
# a mutable ``Comptime`` variable has no runtime identity, so a runtime index is
# still rejected


@func()
def const_comptime_array_read_with_a_runtime_index(i: usize) -> i32:
    arr: ConstComptime[Array[i32, Literal[3]]] = array(
        coerce(10, i32), coerce(20, i32), coerce(30, i32), length=3,
    )
    return arr[i]


@func()
def const_comptime_untyped_array_read_with_a_runtime_index(i: usize) -> i32:
    # an untyped element has no runtime representation, so the constant cannot
    # be materialized: a clean error, not a crash
    const_comptime()
    arr = array(10, 20, 30)
    return arr[i]


@func()
def comptime_array_read_with_a_runtime_index(i: usize) -> i32:
    arr: Comptime[Array[i32, Literal[3]]] = array(
        coerce(10, i32), coerce(20, i32), coerce(30, i32), length=3,
    )
    return arr[i]


# -- an option/union field whose tag is only known at runtime ---------------
# a runtime-tag option/union is a whole runtime value, so it lives in a plain
# variable (see the behaviour doc §5.3)


@struct()
class WithOpt:
    o: Option[i32]


@struct()
class WithUnion:
    u: V_A | V_B


@func()
def take_with_opt_ptr(p: Ptr[WithOpt]) -> i32:
    if (v := p[...].o) is not None:
        return v + 1
    return -1


@func()
def comptime_ptr_runtime_tag_option(n: i32) -> i32:
    # ``n`` coerces to a runtime ``Option[i32]`` (its tag is only known at
    # runtime): the field of the runtime ``s`` holds it
    s: WithOpt = WithOpt(n)
    return take_with_opt_ptr(ref(s))


@func()
def take_with_union_ptr(p: Ptr[WithUnion]) -> i32:
    if isinstance(v := p[...].u, V_A):
        return v.x + 1
    return -1


@func()
def comptime_ptr_runtime_tag_union(u: V_A | V_B) -> i32:
    # ``u`` is a runtime tagged union (its tag is only known at runtime)
    s: WithUnion = WithUnion(u)
    return take_with_union_ptr(ref(s))


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

    @closure(inline=False)
    def f() -> i32:
        return s.a + s.b

    return f()


@func()
def compiled_closure_captures_a_comptime_tuple() -> i32:
    # a captured tuple is a compile-time pointer of its own: the capture is
    # baked into the closure's struct type (see the behaviour doc §9.3)
    t: Comptime = (coerce(3, i32), coerce(4, i32))

    @closure(inline=False)
    def f() -> i32:
        return t[0] + t[1]

    return f()


# (the old ``compiled_closure_mutates_its_captured_copy`` /
# ``test_a_compiled_closure_mutates_its_captured_copy`` is gone: the capture no
# longer copies the pointee, so a compiled closure cannot mutate a private
# copy of a captured compile-time place)


@func()
def compiled_closure_captures_a_tuple_with_a_runtime_element(x: i32) -> i32:
    # a tuple of runtime values is a plain runtime value: the closure captures
    # it like any other runtime variable
    t = (x, x + 5)

    @closure(inline=False)
    def f() -> i32:
        return t[0] + t[1]

    return f()


@func()
def compiled_closure_captures_an_aggregate_with_a_runtime_element(x: i32) -> i32:
    s = Little(x, 2)

    @closure(inline=False)
    def f() -> i32:
        return s.a + s.b

    return f()


# -- a compile-time capture baked into a function pointer -------------------


@func_type()
class OneFn(Protocol):
    def __call__(self, x: i32) -> i32: ...


@func()
def comptime_capture_as_func_ptr() -> i32:
    # a compile-time capture needs no runtime parameter, so it is baked into the
    # specialization and the closure is an ordinary runtime function
    n: Comptime = 5

    @closure(inline=False)
    def f(x: i32) -> i32:
        return x + n

    p = as_func_ptr(OneFn, f)
    return p[...](1)


class SpyComptimePtrTest(TestCase):
    """Compile-time pointers crossing a non-inline boundary: a pointer to a
    mutable ``Comptime`` variable cannot be lowered to a runtime value, while a
    const pointer to a ``ConstComptime`` variable lowers to a global static
    constant the callee aliases (see the behaviour doc §4.2/§4.3)."""

    def test_a_scalar_pointee(self) -> None:
        self.assertEqual(comptime_ptr_scalar(), 6)

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

    def test_a_nested_runtime_tag_option(self) -> None:
        # the option field's tag is only known at runtime
        self.assertEqual(comptime_ptr_runtime_tag_option(5), 6)

    def test_a_nested_runtime_tag_union(self) -> None:
        self.assertEqual(comptime_ptr_runtime_tag_union(V_A(5)), 6)
        self.assertEqual(comptime_ptr_runtime_tag_union(V_B(5)), -1)

    def test_a_compile_time_var_as_a_comptime_param(self) -> None:
        self.assertEqual(comptime_var_to_comptime_param(), 6)

    def test_a_compiled_closure_captures_a_comptime_aggregate(self) -> None:
        self.assertEqual(compiled_closure_captures_a_comptime_aggregate(), 3)

    def test_a_compiled_closure_captures_a_comptime_tuple(self) -> None:
        self.assertEqual(compiled_closure_captures_a_comptime_tuple(), 7)

    def test_a_compiled_closure_captures_a_tuple_with_a_runtime_element(self) -> None:
        self.assertEqual(compiled_closure_captures_a_tuple_with_a_runtime_element(10), 25)

    def test_a_compiled_closure_captures_an_aggregate_with_a_runtime_element(self) -> None:
        self.assertEqual(
            compiled_closure_captures_an_aggregate_with_a_runtime_element(10), 12,
        )

    def test_a_comptime_capture_baked_into_a_function_pointer(self) -> None:
        self.assertEqual(comptime_capture_as_func_ptr(), 6)

    def test_a_const_comptime_array_read_with_a_runtime_index(self) -> None:
        # a read-only compile-time constant crosses into runtime storage so that
        # a runtime index reads it
        self.assertEqual(const_comptime_array_read_with_a_runtime_index(0), 10)
        self.assertEqual(const_comptime_array_read_with_a_runtime_index(2), 30)

    def test_a_mutable_comptime_array_with_a_runtime_index_is_rejected(self) -> None:
        # a mutable compile-time variable has no runtime identity of its own
        with self.assertRaises(CompileError) as ctx:
            comptime_array_read_with_a_runtime_index(0)
        self.assertIn('runtime index', str(ctx.exception))

    def test_a_const_comptime_array_of_untyped_elements_is_rejected(self) -> None:
        # the untyped element has no runtime representation: a clean compiler
        # error rather than a crash
        with self.assertRaises(CompileError):
            const_comptime_untyped_array_read_with_a_runtime_index(0)


_I32 = sval.IntType(32, True)
_I8 = sval.IntType(8, True)


class SpyPointerObjectTypeTest(TestCase):
    """``sval.pointer_object_type``: the object a compile-time pointer names when
    it is materialized into runtime storage - its pointee, or the array's tail
    from the element it names on."""

    def test_a_scalar_pointee(self) -> None:
        box = sval.Box('x', _I32)
        ptr = sval.ComptimePtr(box, 0, sval.PointerType(_I32))
        self.assertEqual(sval.pointer_object_type(ptr), _I32)

    def test_a_pointer_into_an_unknown_box(self) -> None:
        # the box holds no type yet: there is nothing wider to name
        box = sval.Box('x')
        ptr = sval.ComptimePtr(box, 0, sval.PointerType(_I32))
        self.assertEqual(sval.pointer_object_type(ptr), _I32)

    def test_the_first_element_of_an_array_is_the_whole_array(self) -> None:
        array = sval.ArrayType(_I32, 3)
        box = sval.Box('a', array)
        ptr = sval.ComptimePtr(
            box, 0, sval.PointerType(_I32, variant=sval.PointerVariant.MULTI),
        )
        self.assertEqual(sval.pointer_object_type(ptr), array)

    def test_a_later_element_is_the_array_tail(self) -> None:
        array = sval.ArrayType(_I32, 3)
        box = sval.Box('a', array)
        ptr = sval.ComptimePtr(box, 2, sval.PointerType(_I32))
        self.assertEqual(sval.pointer_object_type(ptr), sval.ArrayType(_I32, 1))

    def test_the_tail_counts_cells_not_elements(self) -> None:
        # a two-cell element at index 1 sits at cell offset 2, and the tail holds
        # the remaining two elements (``length - offset`` would say one)
        pair = sval.TupleType((_I32, _I32), False)
        array = sval.ArrayType(pair, 3)
        box = sval.Box('p', array)
        ptr = sval.ComptimePtr(box, 2, sval.PointerType(pair))
        self.assertEqual(sval.pointer_object_type(ptr), sval.ArrayType(pair, 2))

    def test_a_pointer_of_another_pointee_is_not_an_element(self) -> None:
        # a pointee that is not the array's element type (a field of the box's
        # object, say) names only itself
        box = sval.Box('a', sval.ArrayType(_I32, 3))
        ptr = sval.ComptimePtr(box, 0, sval.PointerType(_I8))
        self.assertEqual(sval.pointer_object_type(ptr), _I8)


all_tests = [
    SpyComptimePtrTest,
    SpyPointerObjectTypeTest,
]

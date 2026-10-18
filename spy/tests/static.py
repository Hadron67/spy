from typing import Any, Literal
from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i32,
    i64,
    usize,
)
from ..compiler import bool as spy_bool
from ..compiler.dsl import decl_func, func, struct
from ..compiler.syntax import (
    Array,
    ConstComptime,
    ConstPtr,
    Option,
    Ptr,
    array,
    comptime,
    const_comptime,
    ref,
)
from ..std.core import coerce


@func()
def static_i32() -> i32:
    # a compile-time value turned into a global static constant: a read-only
    # ``ConstComptime`` variable's pointer lowers to it automatically (see the
    # behaviour doc §4.3, replacing ``as_static_ptr``)
    comptime()
    x: i32 = 42
    const_x: ConstComptime[i32] = x
    p = ref(const_x)
    return p[...]


@func()
def static_f64() -> f64:
    comptime()
    x: f64 = 1.5
    const_x: ConstComptime[f64] = x
    p = ref(const_x)
    return p[...]


@func()
def static_bool() -> spy_bool:
    comptime()
    x: spy_bool = True
    const_x: ConstComptime[spy_bool] = x
    p = ref(const_x)
    return p[...]


@struct()
class StaticPair:
    a: i32
    b: i32


@struct()
class StaticNested:
    p: StaticPair
    n: i32


@func()
def static_struct() -> i32:
    comptime()
    s: StaticPair = StaticPair(3, 4)
    const_s: ConstComptime[StaticPair] = s
    p = ref(const_s)
    return p[...].a * 10 + p[...].b


@func()
def static_nested_struct() -> i32:
    comptime()
    s: StaticNested = StaticNested(StaticPair(1, 2), 3)
    const_s: ConstComptime[StaticNested] = s
    p = ref(const_s)
    return p[...].p.a * 100 + p[...].p.b * 10 + p[...].n


@func()
def static_array() -> i32:
    comptime()
    a: Array[i32, Literal[3]] = array(1, 2, 3, length=3)
    const_a: ConstComptime[Array[i32, Literal[3]]] = a
    p = ref(const_a)
    return p[...][0] * 100 + p[...][1] * 10 + p[...][2]


@func()
def static_option_present() -> i32:
    comptime()
    o: Option[i32] = 7
    const_o: ConstComptime[Option[i32]] = o
    p = ref(const_o)
    if (v := p[...]) is not None:
        return v
    return -1


@func()
def static_option_absent() -> i32:
    comptime()
    o: Option[i32] = None
    const_o: ConstComptime[Option[i32]] = o
    p = ref(const_o)
    if (v := p[...]) is not None:
        return v
    return -1


@func()
def static_option_pointer_absent() -> spy_bool:
    # the child of the option holds a pointer, so the pointer itself is the
    # absent tag: the constant is a null pointer
    comptime()
    o: Option[Ptr[i32]] = None
    const_o: ConstComptime[Option[Ptr[i32]]] = o
    p = ref(const_o)
    return p[...] is None


@func()
def static_option_struct() -> i32:
    # an option whose child is an aggregate: the payload is a struct value
    comptime()
    o: Option[StaticPair] = StaticPair(3, 4)
    const_o: ConstComptime[Option[StaticPair]] = o
    p = ref(const_o)
    if (v := p[...]) is not None:
        return v.a * 10 + v.b
    return -1


@func()
def static_option_struct_absent() -> i32:
    comptime()
    o: Option[StaticPair] = None
    const_o: ConstComptime[Option[StaticPair]] = o
    p = ref(const_o)
    if (v := p[...]) is not None:
        return v.a
    return -1


@struct()
class StaticUnionA:
    x: i32
    w: i32


@struct()
class StaticUnionB:
    y: i64
    z: i64


@struct()
class StaticUnionHolder:
    tag: i32
    u: StaticUnionA | StaticUnionB
    tail: i32


@struct()
class StaticUnionEmpty:
    pass


@func()
def static_union_storage() -> i64:
    # ``StaticUnionB`` is the storage (largest) variant: the constant is that
    # variant itself
    comptime()
    v: StaticUnionB = StaticUnionB(9, 0)
    comptime()
    u: StaticUnionA | StaticUnionB = v
    const_u: ConstComptime[StaticUnionA | StaticUnionB] = u
    p = ref(const_u)
    if isinstance(b := p[...], StaticUnionB):
        return b.y
    return -1


@func()
def static_union_smaller() -> i32:
    # ``StaticUnionA`` is smaller than the storage variant: the constant is laid
    # out the way clang lays out such a global, ``{<variant>, [pad x i8]}``
    comptime()
    v: StaticUnionA = StaticUnionA(5, 0)
    comptime()
    u: StaticUnionA | StaticUnionB = v
    const_u: ConstComptime[StaticUnionA | StaticUnionB] = u
    p = ref(const_u)
    if isinstance(a := p[...], StaticUnionA):
        return a.x
    return -1


@func()
def static_union_field() -> i32:
    # a union member forces the struct's constant to a literal type whose layout
    # reproduces the named one (the tail field still reads back correctly)
    comptime()
    v: StaticUnionA = StaticUnionA(5, 0)
    comptime()
    h: StaticUnionHolder = StaticUnionHolder(1, v, 9)
    const_h: ConstComptime[StaticUnionHolder] = h
    p = ref(const_h)
    if isinstance(a := p[...].u, StaticUnionA):
        return a.x * 100 + p[...].tail
    return -1


@func()
def static_union_array_read(index: usize) -> i32:
    # a runtime index into a compile-time array of union constants: the address
    # into the literal-laid-out global is computed byte-wise (see the lowering)
    comptime()
    v: StaticUnionA = StaticUnionA(5, 0)
    comptime()
    u: StaticUnionA | StaticUnionB = v
    arr: ConstComptime[Array[StaticUnionA | StaticUnionB, Literal[2]]] = array(
        u, u, length=2,
    )
    if isinstance(a := arr[index], StaticUnionA):
        return a.x
    return -1


@func()
def static_union_zst_variant() -> i32:
    # a zero-sized variant: it holds no payload, so the union storage stays
    # undefined
    comptime()
    v: StaticUnionEmpty = StaticUnionEmpty()
    comptime()
    u: StaticUnionEmpty | StaticUnionB = v
    const_u: ConstComptime[StaticUnionEmpty | StaticUnionB] = u
    p = ref(const_u)
    if isinstance(_ := p[...], StaticUnionEmpty):
        return 1
    return -1


@func()
def static_rejects_runtime(x: i32) -> i32:
    const_x: ConstComptime[i32] = x
    p = ref(const_x)
    return p[...]


@func()
def static_pointer_to_struct() -> i32:
    # a pointer in the constant is not followed: the value it points at becomes
    # a global of its own and the pointer points at it
    comptime()
    v: StaticPair = StaticPair(3, 4)
    const_v: ConstComptime[StaticPair] = v
    comptime()
    p: Ptr[StaticPair] = ref(const_v)
    const_p: ConstComptime[Ptr[StaticPair]] = p
    q = ref(const_p)
    return q[...][...].a * 10 + q[...][...].b


@struct()
class StaticPtrHolder:
    p: Ptr[StaticPair]
    n: i32


@func()
def static_struct_with_pointer() -> i32:
    comptime()
    v: StaticPair = StaticPair(3, 4)
    const_v: ConstComptime[StaticPair] = v
    comptime()
    ptr: Ptr[StaticPair] = ref(const_v)
    comptime()
    h: StaticPtrHolder = StaticPtrHolder(ptr, 7)
    const_h: ConstComptime[StaticPtrHolder] = h
    q = ref(const_h)
    return q[...].p[...].a * 10 + q[...].n


@func()
def static_static_pointer() -> i32:
    # another static constant (a global reference, not a register) is allowed as
    # the value: the global holds the pointer to the first one
    comptime()
    v: i32 = 9
    const_v: ConstComptime[i32] = v
    comptime()
    p: ConstPtr[i32] = ref(const_v)
    const_p: ConstComptime[ConstPtr[i32]] = p
    q = ref(const_p)
    return q[...][...]


@decl_func('abs')
def c_abs(x: i32) -> i32:
    ...


@func()
def static_function_pointer(x: i32) -> i32:
    # a function pointer is a global reference, not a register: it may stand in
    # a static constant
    const_f: ConstComptime = c_abs
    q = ref(const_f)
    return q[...](x)


@func()
def static_rejects_a_type() -> i32:
    comptime()
    t = i32
    const_comptime()
    const_t = t
    p = ref(const_t)
    return p[...]  # pyright: ignore[reportReturnType]


# ---------------------------------------------------------------------------
# the ``syntax.const_comptime()`` statement marker: the *unannotated* spelling of
# ``x: ConstComptime = e`` (``x = e`` with the marker declares a read-only
# compile-time constant of the type its value determines, see ``astgen``)
# ---------------------------------------------------------------------------


@func()
def const_marker_scalar() -> i32:
    # the value determines the type of the constant: a typed compile-time value
    # (an untyped integer literal is compile-time-only and has no runtime
    # representation to point at)
    const_comptime()
    x = coerce(42, i32)
    p = ref(x)
    return p[...]


@func()
def const_marker_typed_declaration() -> i32:
    # the marker also combines with a plain annotation, like ``comptime()`` does
    const_comptime()
    x: i32 = 42
    p = ref(x)
    return p[...]


@func()
def const_marker_struct() -> i32:
    const_comptime()
    s = StaticPair(3, 4)
    p = ref(s)
    return p[...].a * 10 + p[...].b


@struct()
class ConstHolder:
    p: ConstPtr[StaticPair]


@func()
def make_const_holder() -> ConstHolder:
    # the reported shape: the constant's address ends up in a value the function
    # *returns*, so it must name a global - a copy materialized on this frame's
    # stack would dangle as soon as the function returned
    const_comptime()
    s = StaticPair(3, 4)
    return ConstHolder(ref(s))


@func()
def const_marker_escapes() -> i32:
    h = make_const_holder()
    return h.p[...].a * 10 + h.p[...].b


@func()
def const_marker_rejects_a_write() -> i32:
    # the constant's box is read-only once its declaration committed: the store
    # below is a compile error.
    const_comptime()
    c = coerce(42, i32)
    c = coerce(43, i32)
    return c


@func()
def const_marker_rejects_a_bound_name() -> i32:
    # the marker must precede a *fresh* declaration
    x: i32 = 1
    const_comptime()
    x = 42
    return x


@func()
def const_marker_rejects_a_runtime_value(x: i32) -> i32:
    const_comptime()
    y = x
    return y


@func()
def const_marker_on_an_annotated_declaration() -> i32:
    # the annotation already says ``ConstComptime``: the marker would repeat it
    const_comptime()
    x: ConstComptime[i32] = 42
    return x


class StaticPtrTest(TestCase):
    """A compile-time value turned into a global static constant through a
    read-only ``ConstComptime`` variable and ``ref`` (see the behaviour doc §4.3;
    it replaces the old ``std.core.as_static_ptr``)."""

    def test_a_scalar(self) -> None:
        self.assertEqual(static_i32(), 42)
        self.assertAlmostEqual(static_f64(), 1.5)
        self.assertTrue(static_bool())

    def test_a_struct(self) -> None:
        self.assertEqual(static_struct(), 34)
        self.assertEqual(static_nested_struct(), 123)

    def test_an_array(self) -> None:
        self.assertEqual(static_array(), 123)

    def test_an_option(self) -> None:
        self.assertEqual(static_option_present(), 7)
        self.assertEqual(static_option_absent(), -1)
        self.assertTrue(static_option_pointer_absent())
        self.assertEqual(static_option_struct(), 34)
        self.assertEqual(static_option_struct_absent(), -1)

    def test_a_tagged_union(self) -> None:
        self.assertEqual(static_union_storage(), 9)
        self.assertEqual(static_union_smaller(), 5)
        self.assertEqual(static_union_zst_variant(), 1)

    def test_a_union_field(self) -> None:
        self.assertEqual(static_union_field(), 509)

    def test_a_union_constant_is_read_the_way_clang_reads_it(self) -> None:
        # a constant whose union holds a variant other than the storage one is
        # laid out with a literal type: an address into it is byte-based, and an
        # aggregate is read out through a memcpy - clang never uses a GEP or a
        # load that claims the canonical type, which the object does not have
        self.assertEqual(static_union_smaller(), 5)
        text = _module_text(static_union_smaller)
        self.assertIn('getelementptr inbounds i8, ptr @global', text)
        self.assertIn('call void @llvm.memcpy', text)
        self.assertNotIn('getelementptr inbounds %tagged_union', text)

    def test_a_runtime_index_into_a_union_constant_array(self) -> None:
        # the byte offset of a runtime index is the index scaled by the element
        # size (a byte GEP), and the element itself is read through a memcpy
        self.assertEqual(static_union_array_read(0), 5)
        self.assertEqual(static_union_array_read(1), 5)
        text = _module_text(static_union_array_read)
        self.assertIn('mul i64 %', text)
        self.assertIn('getelementptr inbounds i8, ptr @global, i64 %', text)
        self.assertIn('call void @llvm.memcpy', text)

    def test_a_runtime_value_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            static_rejects_runtime(1)

    def test_a_pointer_to_a_struct(self) -> None:
        self.assertEqual(static_pointer_to_struct(), 34)

    def test_a_pointer_field(self) -> None:
        self.assertEqual(static_struct_with_pointer(), 37)

    def test_a_pointer_to_another_static_constant(self) -> None:
        self.assertEqual(static_static_pointer(), 9)

    def test_a_function_pointer(self) -> None:
        self.assertEqual(static_function_pointer(-5), 5)

    def test_a_compile_time_only_value_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            static_rejects_a_type()


def _module_text(handle: Any) -> str:
    """The LLVM IR of the module ``handle`` was compiled into; the handle's
    Python-side call has to have run first (it is what compiles it)."""
    entry = handle.get_entry()  # pyright: ignore
    instance = next(iter(entry.specs.values()))
    native = instance.wrapper_fn or instance.native_fn
    assert native is not None
    return '\n'.join(native.print_all())


class ConstComptimeMarkerTest(TestCase):
    """``syntax.const_comptime()`` marks the *unannotated* declaration that
    follows as a read-only compile-time constant (``x = e`` with the marker is
    ``x: ConstComptime = e``): ``ref(x)`` then lowers to a global static constant
    the same way the annotated spelling does (see the behaviour doc §4.3)."""

    def test_a_scalar(self) -> None:
        self.assertEqual(const_marker_scalar(), 42)
        self.assertEqual(const_marker_typed_declaration(), 42)
        self.assertIn('constant i32 42', _module_text(const_marker_scalar))

    def test_an_aggregate(self) -> None:
        self.assertEqual(const_marker_struct(), 34)

    def test_the_constant_lowers_to_a_global(self) -> None:
        self.assertEqual(const_marker_struct(), 34)
        text = _module_text(const_marker_struct)
        # the value is a read-only global the pointer points at - not a copy
        # materialized on the declaring frame's stack (which is what a *mutable*
        # compile-time variable would lower to)
        self.assertIn('constant %StaticPair', text)
        self.assertNotIn('alloca %StaticPair', text)

    def test_a_pointer_escaping_the_declaring_frame(self) -> None:
        self.assertEqual(const_marker_escapes(), 34)
        text = _module_text(make_const_holder)
        self.assertIn('constant %StaticPair', text)
        self.assertNotIn('alloca %StaticPair', text)

    def test_a_write_after_the_commit_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            const_marker_rejects_a_write()

    def test_the_marker_must_precede_a_fresh_declaration(self) -> None:
        with self.assertRaises(CompileError):
            const_marker_rejects_a_bound_name()

    def test_a_runtime_value_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            const_marker_rejects_a_runtime_value(1)

    def test_the_marker_may_not_repeat_the_annotation(self) -> None:
        with self.assertRaises(CompileError):
            const_marker_on_an_annotated_declaration()


all_tests = [
    StaticPtrTest,
    ConstComptimeMarkerTest,
]

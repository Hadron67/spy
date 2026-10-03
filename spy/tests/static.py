from unittest import TestCase

from ..compiler import (
    CompileError,
    f64,
    i32,
    i64,
)
from ..compiler import bool as spy_bool
from ..compiler.dsl import decl_func, func, struct
from ..compiler.syntax import (
    Array,
    ConstPtr,
    Option,
    Ptr,
    array,
    comptime,
    ref,
)
from ..std.core import as_static_ptr


@func()
def static_i32() -> i32:
    # a compile-time value turned into a global static constant: the pointer is
    # a ``ConstPtr`` to the location the value is stored in
    comptime()
    x: i32 = 42
    p = as_static_ptr(x)
    return p[...]


@func()
def static_f64() -> f64:
    comptime()
    x: f64 = 1.5
    p = as_static_ptr(x)
    return p[...]


@func()
def static_bool() -> spy_bool:
    comptime()
    x: spy_bool = True
    p = as_static_ptr(x)
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
    p = as_static_ptr(s)
    return p[...].a * 10 + p[...].b


@func()
def static_nested_struct() -> i32:
    comptime()
    s: StaticNested = StaticNested(StaticPair(1, 2), 3)
    p = as_static_ptr(s)
    return p[...].p.a * 100 + p[...].p.b * 10 + p[...].n


@func()
def static_array() -> i32:
    comptime()
    a: Array[i32, 3] = array(1, 2, 3)  # pyright: ignore
    p = as_static_ptr(a)
    return p[...][0] * 100 + p[...][1] * 10 + p[...][2]


@func()
def static_option_present() -> i32:
    comptime()
    o: Option[i32] = 7
    p = as_static_ptr(o)
    if (v := p[...]) is not None:
        return v
    return -1


@func()
def static_option_absent() -> i32:
    comptime()
    o: Option[i32] = None
    p = as_static_ptr(o)
    if (v := p[...]) is not None:
        return v
    return -1


@func()
def static_option_pointer_absent() -> spy_bool:
    # the child of the option holds a pointer, so the pointer itself is the
    # absent tag: the constant is a null pointer
    comptime()
    o: Option[Ptr[i32]] = None
    p = as_static_ptr(o)
    return p[...] is None


@func()
def static_option_struct() -> i32:
    # an option whose child is an aggregate: the payload is a struct value
    comptime()
    o: Option[StaticPair] = StaticPair(3, 4)
    p = as_static_ptr(o)
    if (v := p[...]) is not None:
        return v.a * 10 + v.b
    return -1


@func()
def static_option_struct_absent() -> i32:
    comptime()
    o: Option[StaticPair] = None
    p = as_static_ptr(o)
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
    p = as_static_ptr(u)
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
    p = as_static_ptr(u)
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
    p = as_static_ptr(h)
    if isinstance(a := p[...].u, StaticUnionA):
        return a.x * 100 + p[...].tail
    return -1


@func()
def static_union_zst_variant() -> i32:
    # a zero-sized variant: it holds no payload, so the union storage stays
    # undefined
    comptime()
    v: StaticUnionEmpty = StaticUnionEmpty()
    comptime()
    u: StaticUnionEmpty | StaticUnionB = v
    p = as_static_ptr(u)
    if isinstance(_ := p[...], StaticUnionEmpty):
        return 1
    return -1


@func()
def static_rejects_runtime(x: i32) -> i32:
    p = as_static_ptr(x)
    return p[...]


@func()
def static_pointer_to_struct() -> i32:
    # a pointer in the constant is not followed: the value it points at becomes
    # a global of its own and the pointer points at it
    comptime()
    v: StaticPair = StaticPair(3, 4)
    comptime()
    p: Ptr[StaticPair] = ref(v)
    q = as_static_ptr(p)
    return q[...][...].a * 10 + q[...][...].b


@struct()
class StaticPtrHolder:
    p: Ptr[StaticPair]
    n: i32


@func()
def static_struct_with_pointer() -> i32:
    comptime()
    v: StaticPair = StaticPair(3, 4)
    comptime()
    ptr: Ptr[StaticPair] = ref(v)
    comptime()
    h: StaticPtrHolder = StaticPtrHolder(ptr, 7)
    q = as_static_ptr(h)
    return q[...].p[...].a * 10 + q[...].n


@func()
def static_static_pointer() -> i32:
    # another static constant (a global reference, not a register) is allowed as
    # the value: the global holds the pointer to the first one
    comptime()
    v: i32 = 9
    comptime()
    p: ConstPtr[i32] = as_static_ptr(v)
    q = as_static_ptr(p)
    return q[...][...]


@decl_func('abs')
def c_abs(x: i32) -> i32:
    ...


@func()
def static_function_pointer(x: i32) -> i32:
    # a function pointer is a global reference, not a register: it may stand in
    # a static constant
    q = as_static_ptr(c_abs)
    return q[...](x)


@func()
def static_rejects_a_type() -> i32:
    comptime()
    t = i32
    p = as_static_ptr(t)
    return p[...]  # pyright: ignore[reportReturnType]


class StaticPtrTest(TestCase):
    """``std.core.as_static_ptr``: a compile-time value turned into a global
    static constant (see ``mir.GlobalConstant``)."""

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


all_tests = [
    StaticPtrTest,
]

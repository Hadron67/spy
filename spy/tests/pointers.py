from unittest import TestCase

from ..compiler import (
    i32,
    syntax,
)
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    ConstPtr,
    Ptr,
    ref,
)
from .structs import Small

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


all_tests = [
    SpyPointerTest,
]

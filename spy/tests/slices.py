from typing import TYPE_CHECKING, Literal
from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    i64,
    syntax,
    u64,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Array,
    Comptime,
    ConstMultiPtr,
    ConstPtr,
    MultiPtr,
    Option,
    Ptr,
    array,
    ref,
)
from ..std import (
    ConstSlicePtr,
    SlicePtr,
    arr_slice,
    const_arr_slice,
)
from .generics import TwoI64
from .structs import Small
from .types import ptr_cast_index

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


all_tests = [
    SpyMultiPointerTest,
    SpySliceIteratorTest,
    SpySlicePtrTest,
    SpySubscriptOverloadTest,
    SpyTypeValueTest,
    SpyAggregateArgumentTest,
]

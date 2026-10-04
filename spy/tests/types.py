from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    i64,
    mir,
    sval,
    usize,
    void,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, func, struct
from ..compiler.syntax import (
    Array,
    MultiPtr,
    Opaque,
    Ptr,
    array,
    ptr_cast,
    ref,
)
from ..compiler.util import TriState
from ..std.mem import align_of, layout_of, size_of
from .cross_context import _CROSS_CONTEXT
from .func_types import IntUnary
from .structs import MIR_CACHE, struct_type


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
    p = ptr_cast(ref(a), Ptr[Array[i32, None]])
    m: MultiPtr[i32] = p  # pyright: ignore
    return m[2]


@struct()
class FamCarrier:
    # a struct with an unsized-array field: a C flexible array member
    n: i32
    data: Array[i32, None]


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


all_tests = [
    SpyTypeClassifyTest,
    SpyDstTest,
    SpyOpaqueAndUnsizedTest,
    SpyLayoutOfTest,
]

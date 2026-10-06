from unittest import TestCase

from ..compiler import bool as spy_bool
from ..compiler import (
    i32,
    i64,
    syntax,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..std import Numeric
from ..std.reflect import (
    StrDictType,
    TupleType,
    type_info,
)

# ---------------------------------------------------------------------------
# a parameter whose annotation is a ``tuple[...]`` has no runtime
# representation of its own (``sval.TupleType``): its elements are passed
# separately (flattened) and rebound as one place tree at the callee, exactly
# like ``*args``.  The body reads the tuple with a compile-time integer index,
# ``x[i]``, and it nests.
# ---------------------------------------------------------------------------


@func()
def sum_pair(x: tuple[i32, i32]) -> i32:
    return x[0] + x[1]


@func()
def sum_nested(x: tuple[i32, tuple[i32, i32]]) -> i32:
    # a nested tuple element is a tuple of its own
    return x[0] + x[1][0] + x[1][1]


@func()
def sum_triple(x: tuple[i32, i32, i32]) -> i32:
    return x[0] + x[1] + x[2]


@func()
def first[T: Numeric](x: tuple[T, T]) -> T:
    # ``T`` is solved from the element types of the provided tuple
    return x[0]


@func()
def pair_and_extra(a: i32, x: tuple[i32, i32]) -> i32:
    # a tuple parameter after an ordinary one
    return a + x[0] + x[1]


@func()
def forward(x: tuple[i32, i32]) -> i32:
    # passing a tuple parameter on to another function
    return sum_pair(x)


@func()
def call_with_comptime(a: i32) -> i32:
    # the second element is a compile-time value: it is carried as a value and
    # not passed in MIR (see ``ArgNode``)
    return sum_pair((a, 5))


def sum_pair_inline(x: tuple[i32, i32]) -> i32:
    # an undecorated (inlined) function may take a tuple parameter too
    return x[0] + x[1]


@func()
def call_inline(a: i32, b: i32) -> i32:
    return sum_pair_inline((a, b))


# a struct is *not* flattened: as an element of a tuple it is a single runtime
# argument (by value when small, by reference when large), and the body reads
# its fields through the tuple element


@struct()
class Pair:
    a: i32
    b: i32

    def total(self) -> i32:
        return self.a + self.b


@struct()
class Big:
    # larger than the by-value limit: passed by reference
    a: i64
    b: i64
    c: i64

    def total(self) -> i64:
        return self.a + self.b + self.c


@func()
def sum_pair_struct(x: tuple[Pair, i32]) -> i32:
    return x[0].a + x[0].b + x[1]


@func()
def total_pair_struct(x: tuple[Pair, i32]) -> i32:
    return x[0].total() + x[1]


@func()
def sum_big(x: tuple[Big, i64]) -> i64:
    return x[0].a + x[0].b + x[0].c + x[1]


@func()
def total_big(x: tuple[Big, i64]) -> i64:
    return x[0].total() + x[1]


@func()
def call_big(a: i64, b: i64, c: i64) -> i64:
    # a large struct element built in spy and passed on
    return total_big((Big(a, b, c), 100))


@func()
def forward_big(x: tuple[Big, i64]) -> i64:
    return sum_big(x)


@func()
def call_forward_big(a: i64, b: i64, c: i64) -> i64:
    return forward_big((Big(a, b, c), 7))


@func()
def call_big_comptime(a: i64, b: i64, c: i64) -> i64:
    # a large struct element beside a compile-time element
    return sum_big((Big(a, b, c), 5))


# ---------------------------------------------------------------------------
# compile-time reflection of a parameter's *shape*: an unannotated parameter
# takes whatever shape the call provides, and the body classifies it at compile
# time with ``std.reflect.type_info`` - a tuple, a nested tuple or a
# ``**kwargs``-style ``dict[str, T]``.  Only the branch that matches the
# argument's shape is compiled, so it may read the parameter's values.
# ---------------------------------------------------------------------------


@func()
def classify(x) -> i32:
    syntax.comptime()
    info = type_info(spy_typeof(x))
    if isinstance(v := info, TupleType):
        return 10 + v.types.length
    if isinstance(v := info, StrDictType):
        return 30 + v.entries.length
    return -1


@func()
def reflect_sum(x) -> i64:
    syntax.comptime()
    info = type_info(spy_typeof(x))
    if isinstance(v := info, TupleType):
        if v.types.length == 2:
            # the second element is a tuple itself: read its elements too
            syntax.comptime()
            second = type_info(v.types.ptr[1])
            if isinstance(second, TupleType):
                return 20 + x[0] + x[1][0] + x[1][1]
        # a flat tuple: walk it by the length the reflection gives
        total: i64 = 0
        syntax.comptime()
        i: i32 = 0
        syntax.unroll()
        while i < v.types.length:
            total = total + x[i]
            i = i + 1
        return total
    if isinstance(v := info, StrDictType):
        # a dictionary: walk its entries and read each by its reflected name
        dict_total: i64 = 0
        syntax.comptime()
        j: i32 = 0
        syntax.unroll()
        while j < v.entries.length:
            syntax.comptime()
            entry = v.entries.ptr[j]
            dict_total = dict_total + x[entry.name]
            j = j + 1
        return 100 + dict_total
    return -1


@func()
def call_reflect_sum() -> i64:
    return reflect_sum((1, 2, 3))


@func()
def first_dict_entry_name(x) -> i32:
    syntax.comptime()
    info = type_info(spy_typeof(x))
    if isinstance(v := info, StrDictType):
        syntax.comptime()
        entry = v.entries.ptr[0]
        return ord(entry.name[0:1])
    return -1


@func()
def first_dict_entry_is_i64(x) -> spy_bool:
    syntax.comptime()
    info = type_info(spy_typeof(x))
    if isinstance(v := info, StrDictType):
        syntax.comptime()
        entry = v.entries.ptr[0]
        return entry.type == i64
    return False


@func()
def dict_value_is_tuple(x) -> spy_bool:
    syntax.comptime()
    info = type_info(spy_typeof(x))
    if isinstance(v := info, StrDictType):
        syntax.comptime()
        entry = v.entries.ptr[0]
        syntax.comptime()
        child = type_info(entry.type)
        if isinstance(child, TupleType):
            syntax.comptime()
            value = x[entry.name]
            return value[0] + value[1] == 3
    return False


class SpyTupleArgTest(TestCase):
    """A ``tuple[...]`` parameter: the elements of the argument are passed
    separately and rebound as the parameter's tuple."""

    def test_a_pair(self) -> None:
        self.assertEqual(sum_pair((20, 22)), 42)

    def test_a_triple(self) -> None:
        self.assertEqual(sum_triple((1, 2, 3)), 6)

    def test_a_nested_tuple(self) -> None:
        self.assertEqual(sum_nested((1, (2, 3))), 6)

    def test_a_generic_tuple(self) -> None:
        self.assertEqual(first((7, 9)), 7)

    def test_a_tuple_after_an_ordinary_parameter(self) -> None:
        self.assertEqual(pair_and_extra(1, (2, 3)), 6)

    def test_forwarding_a_tuple(self) -> None:
        self.assertEqual(forward((3, 4)), 7)

    def test_a_compile_time_element(self) -> None:
        self.assertEqual(call_with_comptime(10), 15)

    def test_an_inlined_function(self) -> None:
        self.assertEqual(call_inline(2, 3), 5)

    def test_a_small_struct_element(self) -> None:
        # a small struct crosses by value
        self.assertEqual(sum_pair_struct((Pair(3, 4), 5)), 12)
        self.assertEqual(total_pair_struct((Pair(3, 4), 5)), 12)

    def test_a_large_struct_element(self) -> None:
        # a large struct (larger than the by-value limit) crosses by reference
        self.assertEqual(sum_big((Big(1, 2, 3), 100)), 106)
        self.assertEqual(total_big((Big(1, 2, 3), 100)), 106)

    def test_a_large_struct_built_in_spy(self) -> None:
        self.assertEqual(call_big(1, 2, 3), 106)

    def test_forwarding_a_large_struct(self) -> None:
        self.assertEqual(call_forward_big(1, 2, 3), 13)

    def test_a_large_struct_beside_a_compile_time_element(self) -> None:
        self.assertEqual(call_big_comptime(1, 2, 3), 11)

    def test_classify_a_tuple(self) -> None:
        self.assertEqual(classify((1, 2, 3)), 13)
        self.assertEqual(classify((7, 8)), 12)

    def test_classify_a_nested_tuple(self) -> None:
        self.assertEqual(classify((1, (2, 3))), 12)

    def test_classify_a_dict(self) -> None:
        self.assertEqual(classify({'a': 1, 'b': 2}), 32)
        self.assertEqual(classify({'x': 1}), 31)

    def test_reflect_and_read_a_tuple(self) -> None:
        self.assertEqual(reflect_sum((1, 2, 3)), 6)
        self.assertEqual(reflect_sum((7, 8)), 15)

    def test_reflect_and_read_a_nested_tuple(self) -> None:
        self.assertEqual(reflect_sum((1, (2, 3))), 26)

    def test_reflect_and_read_a_dict(self) -> None:
        self.assertEqual(reflect_sum({'a': 1, 'b': 2}), 103)
        self.assertEqual(reflect_sum({'x': 5}), 105)

    def test_a_spy_constructed_tuple(self) -> None:
        self.assertEqual(call_reflect_sum(), 6)

    def test_a_dict_entry_reflects_its_name_and_type(self) -> None:
        self.assertEqual(first_dict_entry_name({'abc': 1}), ord('a'))
        self.assertTrue(first_dict_entry_is_i64({'abc': 1}))

    def test_a_dict_value_that_is_a_tuple(self) -> None:
        self.assertTrue(dict_value_is_tuple({'p': (1, 2)}))


all_tests = [
    SpyTupleArgTest,
]

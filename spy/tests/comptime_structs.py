from unittest import TestCase

from ..compiler import bool as spy_bool
from ..compiler import (
    i32,
    i64,
    syntax,
)
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Comptime,
    ref,
)
from .generics import OuterTwo, Pair, StructHolder, TwoI64
from .pointers import incr_ptr
from .structs import Blank, Holder, Small, sum_small

# ---------------------------------------------------------------------------
# compile-time structs: a struct built in an inline slot - an expression
# temporary or a ``Comptime`` variable - is an aggregate whose fields are their
# own compile-time places, so a field is read and written at compile time: a
# field assignment is folded in Python, a field read may condition a
# compile-time loop marked with ``syntax.unroll()``, and a whole aggregate is
# copied field by field (see ``sval.Box``).
#
# A struct with *any* runtime part is a whole runtime value now: a ``Comptime``
# variable only holds compile-time values, so such a struct lives in a plain
# variable (memory) instead.
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
    # a struct with a runtime part is a whole runtime value: it lives in a plain
    # variable rather than a ``Comptime`` one
    s: Small = Small(x, 2)
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


# (the old ``comptime_struct_ref_in_memory`` / ``test_a_pointer_in_memory_points_at_a_copy``
# is gone: a compile-time pointer now aliases the box globally, so there is no
# "the local points at a copy" behavior - a plain local holding ``ref(s)`` is
# inferred compile-time storage and aliases ``s`` just like an annotated one)


@func()
def comptime_struct_method() -> i32:
    # a compiled method takes the aggregate's address: the aggregate lives in a
    # plain variable, so that address is a real runtime pointer
    p: Pair[i32] = Pair[i32](1, 2)
    return p.total()


@func()
def comptime_struct_argument() -> i32:
    return sum_small(Small(1, 2))


@func()
def comptime_struct_choose(c: spy_bool) -> i32:
    # both branches are runtime regions, so the aggregate lives in one runtime
    # storage: each branch keeps its own value
    s: Small = Small(1, 1) if c else Small(2, 2)
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
    # cap reports it instead of unrolling forever (see ``inline_bad_cond``).
    # The declared type keeps the struct in memory (an unannotated ``Toggle``
    # of compile-time values would be inferred compile-time)
    s: Toggle = Toggle(True, 0)

    syntax.unroll()
    while s.on:
        s.on = False
    return s.n


@func()
def comptime_struct_nested_copy() -> i32:
    # a whole-aggregate copy of a struct with a struct field: the field place is
    # an aggregate of its own
    h: Comptime = StructHolder(Pair[i32](1, 2), 3)
    b: Comptime = h
    return b.p.a * 100 + b.p.b * 10 + b.extra


@func()
def comptime_struct_from_a_runtime_value(x: i32) -> i32:
    # a whole *runtime* struct value is an ordinary runtime value: a plain
    # variable holds it (there is no more splitting into one store per field)
    s: Small = Small(x, 2)
    c = s
    return c.a + c.b


@func()
def comptime_struct_declared_from_a_runtime_value(x: i32) -> i32:
    # ... the same into a declared plain variable
    s: Small = Small(x, 2)
    c: Small = s
    return c.a + c.b


@func()
def comptime_struct_reassigned_from_a_runtime_value(x: i32) -> i32:
    # a whole runtime value assigned after a construction: the plain variable
    # holds the new value
    c: Small = Small(x, 1)
    s: Small = Small(x, 3)
    c = s
    return c.a + c.b


@func()
def comptime_struct_of_one_field_from_a_runtime_value(x: i64) -> i64:
    # a struct whose mirror *is* its one stored field: a runtime value of it is
    # an ordinary runtime value
    o = OuterTwo(TwoI64(x, 1))
    c: OuterTwo = o
    return c.inner.b


@func()
def comptime_struct_runtime_field_by_ref(x: i32) -> i32:
    # a runtime field of a runtime aggregate lives in memory, so handing its
    # address to a native function hands it a real pointer and the write reaches
    # the field
    s: Small = Small(x, 2)
    incr_ptr(ref(s.a))
    return s.a


@func()
def comptime_struct_field_written_in_branches(c: spy_bool, x: i32) -> i32:
    # the field holds the runtime value ``x`` in memory, so writing it in both
    # branches of a runtime ``if`` merges in that memory and the choice is a
    # runtime one (the last write of the walk no longer wins)
    s: Small = Small(x, 1)
    if c:
        s.a = 5
    else:
        s.a = 9
    return s.a + s.b


@func()
def comptime_struct_from_a_nested_runtime_value(x: i32) -> i32:
    # a nested aggregate field holding a runtime struct is an ordinary runtime
    # value too
    inner: Pair[i32] = Pair[i32](x, 2)
    h: StructHolder = StructHolder(inner, 3)
    return h.p.a + h.p.b + h.extra


@func()
def comptime_struct_from_a_runtime_choice(c: spy_bool, x: i32) -> i32:
    # both branches assign a whole runtime value: each runtime branch writes its
    # own value into the runtime storage
    a: Small = Small(x, 2)
    b: Small = Small(x, 3)
    s: Small = a if c else b
    return s.a + s.b


@func()
def comptime_struct_argument_from_a_variable(x: i32) -> i32:
    # a runtime aggregate handed to a native function: its fields are read out
    # of memory
    s: Small = Small(x, 2)
    return sum_small(s)


@func()
def comptime_struct_field_comparison() -> spy_bool:
    # a field read is a compile-time constant, so an operator on it is folded
    # in Python
    s: Comptime = Small(1, 2)
    return s.a == 1


@func()
def comptime_struct_field_written_in_a_runtime_loop(x: i32) -> i32:
    # a runtime field is memory, so a runtime loop may write it
    s: Small = Small(x, 1)
    y: i32 = 0
    while y < 3:
        s.a = s.a + 1
        y = y + 1
    return s.a


@func()
def comptime_struct_declared_field_written_in_a_runtime_loop(x: i32) -> i32:
    # ... likewise for a declared plain variable
    s: Small = Small(x, 1)
    y: i32 = 0
    while y < 3:
        s.a = s.a + 1
        y = y + 1
    return s.a


@func()
def comptime_struct_zst_in_a_comptime_local() -> spy_bool:
    # a zero-sized aggregate is an aggregate too: it is held by its places, not
    # by a compile-time box
    b: Comptime = Blank(None)
    return spy_typeof(b) == Blank


class SpyComptimeStructTest(TestCase):
    """Compile-time structs: a struct built in an inline slot - an expression
    temporary or a ``Comptime`` variable - is an aggregate whose fields are
    their own compile-time places, so a field read or write is folded in
    Python (a field value may condition a compile-time loop marked with
    ``syntax.unroll()``).  A struct with a runtime part is a whole runtime
    value and lives in a plain variable instead."""

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

    def test_native_argument(self) -> None:
        self.assertEqual(comptime_struct_argument(), 3)

    def test_field_conditions_a_compile_time_loop(self) -> None:
        self.assertEqual(comptime_struct_unroll(), 1)

    def test_a_runtime_branch_writes_the_aggregate(self) -> None:
        # the two branches of the runtime ``if`` are two runtime regions, so the
        # aggregate lives in one runtime storage: each branch keeps its own value
        self.assertEqual(comptime_struct_choose(True), 1)
        self.assertEqual(comptime_struct_choose(False), 2)

    def test_nested_aggregate_copy(self) -> None:
        self.assertEqual(comptime_struct_nested_copy(), 123)

    def test_a_whole_runtime_value_into_a_plain_variable(self) -> None:
        # a whole runtime struct value is an ordinary runtime value
        self.assertEqual(comptime_struct_from_a_runtime_value(5), 7)

    def test_a_whole_runtime_value_into_a_declared_plain_variable(self) -> None:
        self.assertEqual(comptime_struct_declared_from_a_runtime_value(5), 7)

    def test_a_whole_runtime_value_after_a_construction(self) -> None:
        self.assertEqual(comptime_struct_reassigned_from_a_runtime_value(5), 8)

    def test_a_runtime_field_of_such_a_variable_is_addressable(self) -> None:
        # the runtime field is memory, not a box: taking its address hands the
        # callee a real pointer, so the write reaches the field
        self.assertEqual(comptime_struct_runtime_field_by_ref(5), 6)

    def test_a_runtime_field_written_in_branches(self) -> None:
        # both branches write the field's memory, so the choice is a runtime one
        self.assertEqual(comptime_struct_field_written_in_branches(True, 3), 6)
        self.assertEqual(comptime_struct_field_written_in_branches(False, 3), 10)

    def test_a_runtime_field_written_in_a_runtime_loop(self) -> None:
        self.assertEqual(comptime_struct_field_written_in_a_runtime_loop(10), 13)
        self.assertEqual(comptime_struct_declared_field_written_in_a_runtime_loop(10), 13)

    def test_a_nested_runtime_aggregate_field(self) -> None:
        self.assertEqual(comptime_struct_from_a_nested_runtime_value(5), 10)

    def test_a_runtime_choice_of_whole_values(self) -> None:
        # each branch writes its own whole value into the runtime storage, so the
        # choice is a runtime one
        self.assertEqual(comptime_struct_from_a_runtime_choice(True, 5), 7)
        self.assertEqual(comptime_struct_from_a_runtime_choice(False, 5), 8)

    def test_a_native_argument_from_a_plain_variable(self) -> None:
        self.assertEqual(comptime_struct_argument_from_a_variable(5), 7)

    def test_field_comparison(self) -> None:
        self.assertTrue(comptime_struct_field_comparison())

    def test_zero_sized_aggregate_in_a_comptime_local(self) -> None:
        self.assertTrue(comptime_struct_zst_in_a_comptime_local())

    def test_a_struct_whose_mirror_is_its_field(self) -> None:
        # a struct of one stored field mirrors to that field itself: a runtime
        # value of it still reads the field out of the value
        self.assertEqual(comptime_struct_of_one_field_from_a_runtime_value(4), 1)


all_tests = [
    SpyComptimeStructTest,
]

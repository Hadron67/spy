from unittest import TestCase

from ..compiler import (
    CompileError,
    TypeMismatchError,
    i8,
    i32,
    i64,
    mir,
    sval,
    u0,
)
from ..compiler import bool as spy_bool
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import func, struct
from ..compiler.syntax import (
    Comptime,
    Option,
    Ptr,
    ref,
)
from .structs import MIR_CACHE, Large, Small, spy_type, struct_type

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


all_tests = [
    SpyOptionTest,
    SpyOptionNestingTest,
    SpyBoolOpTest,
    SpyOptionUnwrapTest,
]

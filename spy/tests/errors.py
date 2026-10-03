import ctypes
from typing import Any, Never
from unittest import TestCase

from ..compiler import (
    CompileError,
    SpyError,
    i32,
    i64,
    mir,
    sval,
)
from ..compiler.dsl import _GLOBAL_CONTEXT, func, struct
from ..compiler.lower import LLVMBackend
from ..compiler.util import StrBiMap
from .returns import mir_signature
from .structs import MIR_CACHE, ExternMixed, Large, Small, struct_type

# ---------------------------------------------------------------------------
# the primitives of error handling: the payload union and the error union in
# the spy type system (``sval``), and their lowering (``mir``/``lower``)
# ---------------------------------------------------------------------------


def _union_lowering_fn() -> mir.Function:
    """A hand-built MIR function that writes an ``i32`` through a ``BitCast``
    of a union's address and branches on it with a ``Switch``:
    ``f(n) = 10 if n == 0, 20 if n == 1, else 30``."""
    i32_mir = mir.IntType(32, True)
    payload = mir.StructType(
        'union_payload',
        (mir.FormalArg('a', i32_mir), mir.FormalArg('b', i32_mir)),
    )
    union = mir.UnionType('union', payload)
    fn = mir.Function('union_lowering_test', [i32_mir], [None], i32_mir)
    entry = fn.entry
    slot = entry.emit(mir.Alloca(union))
    cell = entry.emit(mir.BitCast(slot, mir.PointerType(i32_mir)))
    entry.emit(mir.Store(cell, mir.Param(0, i32_mir)))
    value = entry.emit(mir.Load(cell))
    case0 = mir.BasicBlock()
    case1 = mir.BasicBlock()
    other = mir.BasicBlock()
    entry.emit(mir.Switch(value, other, ((0, case0), (1, case1))))
    case0.emit(mir.Ret(mir.Int(10, i32_mir)))
    case1.emit(mir.Ret(mir.Int(20, i32_mir)))
    other.emit(mir.Ret(mir.Int(30, i32_mir)))
    mir.normalize(fn)
    fn.is_complete = True
    return fn


class SpyErrorUnionPrimitiveTest(TestCase):
    """The type-system and lowering primitives of error handling: the result
    type spreads into a value, an error code and a payload union (see
    ``sval.make_ret_spec``), and the payload union is an untagged union read and
    written through a ``BitCast``."""

    def test_empty_error_union_is_the_unit_type(self) -> None:
        empty = sval.ResultType(sval.VoidType(), sval.FrozenArraySet())
        self.assertEqual(empty.tag_bits, 0)
        self.assertEqual(empty.code_type, sval.IntType(0, False))
        self.assertIsNotNone(empty.get_unit_value())
        self.assertIsNone(empty.to_mir_type(MIR_CACHE))

    def test_error_code_width_is_the_smallest(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        # a third, distinct exception (its size does not matter here)
        third = struct_type(ExternMixed)
        void = sval.VoidType()
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small,))).tag_bits, 1)
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small, large))).tag_bits, 2)
        # ``types`` is a set: a duplicate is dropped
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet((small, large, small))).tag_bits, 2
        )
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet((small, large, third))).tag_bits, 2
        )
        # a function that returns no value has no "no error" code: its i-th
        # exception is tagged ``i``, so a single one needs no code at all
        empty = sval.EmptyType()
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet()).tag_bits, 0)
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small,))).tag_bits, 0)
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small, large))).tag_bits, 1)
        self.assertEqual(
            sval.ResultType(empty, sval.FrozenArraySet((small, large, third))).tag_bits, 2
        )
        self.assertEqual(sval.ResultType(empty, sval.FrozenArraySet((small,))).code_of(small), 0)
        self.assertEqual(sval.ResultType(void, sval.FrozenArraySet((small,))).code_of(small), 1)

    def test_payload_union_uses_the_largest_variant_and_is_interned(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        one = sval.UnionType(frozenset((small, large)))
        two = sval.UnionType(frozenset((small, large)))
        self.assertIs(one.storage_variant(MIR_CACHE), large)
        self.assertIs(one.to_mir_type(MIR_CACHE), two.to_mir_type(MIR_CACHE))
        self.assertIsNone(sval.UnionType(frozenset()).to_mir_type(MIR_CACHE))
        self.assertEqual(
            sval.UnionType(frozenset()).get_unit_value(),
            sval.UnionValue(sval.UnionType(frozenset())),
        )

    def test_make_ret_spec_spreads_the_error_union(self) -> None:
        small = struct_type(Small)
        i32_type = sval.IntType(32, True)
        type = sval.ResultType(i32_type, sval.FrozenArraySet((small,)))
        spec = sval.make_ret_spec(type, MIR_CACHE)
        assert isinstance(spec, sval.RetTuple)
        self.assertIs(spec.type, type)
        result, code, payload = spec.values
        assert isinstance(result, sval.RetValue)
        self.assertIs(result.type, i32_type)
        assert isinstance(code, sval.RetValue)
        assert isinstance(payload, sval.RetValue)
        self.assertEqual(code.type, sval.IntType(1, False))
        self.assertIsInstance(payload.type, sval.UnionType)
        # the i32 is returned by value; its code and payload through pointers
        self.assertFalse(result.via_result_ptr)
        self.assertTrue(code.via_result_ptr)
        self.assertTrue(payload.via_result_ptr)
        self.assertEqual(len(list(sval.iter_ret_leaves(spec))), 3)

    def test_a_value_less_function_returns_its_small_payload_by_value(self) -> None:
        small = struct_type(Small)
        # no value to return, one exception: the code is ``u0`` (zero-sized) and
        # the payload union takes the by-value slot
        type = sval.ResultType(sval.EmptyType(), sval.FrozenArraySet((small,)))
        spec = sval.make_ret_spec(type, MIR_CACHE)
        assert isinstance(spec, sval.RetTuple)
        _value, code, payload = spec.values
        assert isinstance(code, sval.RetValue)
        assert isinstance(payload, sval.RetValue)
        self.assertEqual(code.type, sval.IntType(0, False))
        self.assertFalse(code.via_result_ptr)
        self.assertFalse(payload.via_result_ptr)
        self.assertIs(sval.ret_returned_type(spec), payload.type)

    def test_error_union_subtyping_is_set_inclusion(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        void = sval.VoidType()
        empty = sval.ResultType(void, sval.FrozenArraySet())
        one = sval.ResultType(void, sval.FrozenArraySet((small,)))
        both = sval.ResultType(void, sval.FrozenArraySet((small, large)))
        self.assertTrue(empty.is_subtype_of(one))
        self.assertTrue(one.is_subtype_of(both))
        self.assertFalse(both.is_subtype_of(one))
        self.assertFalse(one.is_subtype_of(sval.ResultType(void, sval.FrozenArraySet((large,)))))

    def test_error_union_peer_is_the_union_in_delivery_order(self) -> None:
        small = struct_type(Small)
        large = struct_type(Large)
        void = sval.VoidType()
        peer = sval.ResultType(void, sval.FrozenArraySet((small,))).resolve_peer_type(
            sval.ResultType(void, sval.FrozenArraySet((large, small))),
        )
        self.assertEqual(peer, sval.ResultType(void, sval.FrozenArraySet((small, large))))
        self.assertEqual(
            sval.ResultType(void, sval.FrozenArraySet()).resolve_peer_type(
                sval.ResultType(void, sval.FrozenArraySet((small,))),
            ),
            sval.ResultType(void, sval.FrozenArraySet((small,))),
        )

    def test_success_is_the_value_of_the_empty_error_union(self) -> None:
        empty = sval.ResultType(sval.VoidType(), sval.FrozenArraySet())
        success = empty.get_unit_value()
        assert success is not None
        self.assertIsInstance(success, sval.Success)
        self.assertEqual(sval.type_of(success), empty)

    def test_lowering_a_bitcast_and_a_switch(self) -> None:
        fn = _union_lowering_fn()
        globals: StrBiMap[mir.GlobalValue] = StrBiMap()
        globals.add('union_lowering_test', fn)
        backend = LLVMBackend()
        native = backend.compile(set(), globals, _GLOBAL_CONTEXT.target_info())[fn]
        self.assertEqual(native.call(ctypes.c_int32(0)), 10)
        self.assertEqual(native.call(ctypes.c_int32(1)), 20)
        self.assertEqual(native.call(ctypes.c_int32(7)), 30)
        text = '\n'.join(native.print_all())
        # the LLVM IR pointers are untyped (``ptr``), so reinterpreting a
        # pointer type as another lowers to nothing
        self.assertNotIn('bitcast', text)
        self.assertIn('switch', text)


# ---------------------------------------------------------------------------
# raising and propagating exceptions: a function that may raise carries an
# error code and a payload next to its result (see ``sval.ResultType``),
# and a call carries the error to its caller (remapping the tag)
# ---------------------------------------------------------------------------


@struct()
class ErrorA(Exception):
    code: i32


@struct()
class ErrorB(Exception):
    n: i32


@func(exceptions=ErrorA)
def raise_a(n: i32) -> i32:
    if n < 0:
        raise ErrorA(7)
    return n + 1


@func(exceptions=ErrorB)
def raise_b(n: i32) -> i32:
    if n < 0:
        raise ErrorB(9)
    return n + 2


@func(exceptions=(ErrorA, ErrorB))
def forward_raise(n: i32) -> i32:
    return raise_a(n) + 10


@func(exceptions=(ErrorA,))
def catch_bound(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorA as e:
        return e.code + 100


@func(exceptions=(ErrorA, ErrorB))
def catch_multi(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorB:
        return 1
    except ErrorA as e:
        return e.code + 200
    except:  # noqa: E722
        return -1


@func(exceptions=(ErrorA, ErrorB))
def escape_through(n: i32) -> i32:
    try:
        return raise_a(n)
    except ErrorB:
        return 1


@func()
def catch_without_declaring(n: i32) -> i32:
    # the try catches ``ErrorA`` in its own error space, so the function itself
    # never raises and declares nothing
    try:
        return raise_a(n)
    except ErrorA as e:
        return e.code + 500


@func()
def raise_undeclared(n: i32) -> i32:
    # the function declares no exception, so raising one is rejected when the
    # error is recorded (see ``HirRunner._add_function_exception``)
    if n < 0:
        raise ErrorA(7)
    return n + 1


# inlined plain Python functions: their bodies are emitted into their call
# sites, so an error they raise or let through belongs to the error space - and
# to the try blocks - enclosing the *caller*, exactly like an error raised at
# the call site itself (see ``HirRunner._active_try``)


def inline_raise(n: i32) -> i32:
    if n < 0:
        raise ErrorA(7)
    return n + 1


def inline_catch(n: i32) -> i32:
    # the inline body has a try of its own: a raise crosses two frames before
    # reaching the try (the body of the raise is inlined into this one)
    try:
        return inline_raise(n)
    except ErrorA as e:
        return e.code + 100


def inline_reraise(n: i32) -> i32:
    # the clause matches nothing the callee raises, so the error is re-raised
    # out of the body's own try - into the caller's, not out of the function
    try:
        return inline_raise(n)
    except ErrorB:
        return 1


def inline_forward(n: i32) -> i32:
    # a native call inside an inlined body: the error it carries is delivered
    # into the caller's space too
    return raise_a(n) + 10


def inline_no_return(n: i32) -> i32:
    # an inlined body whose every path raises: it never falls through to its
    # caller, so the caller's code after the call is dead (see
    # ``HirRunner._pop_frame``)
    raise ErrorA(7)


def inline_raise_either(n: i32) -> i32:
    # both paths of the body raise, so it does not fall through either
    if n < 0:
        raise ErrorA(7)
    raise ErrorB(9)


def inline_forward_no_return(n: i32) -> i32:
    # an inlined body whose only path raises through another inlined body
    return inline_no_return(n) + 3


def inline_raise_two(n: i32) -> i32:
    # an inlined body that may raise either of two exceptions
    if n > 10:
        raise ErrorA(7)
    if n > 0:
        raise ErrorB(9)
    return n + 1


@func()
def catch_inline_raise(n: i32) -> i32:
    try:
        return inline_raise(n)
    except ErrorA as e:
        return e.code + 100


@func()
def catch_inline_own_try(n: i32) -> i32:
    return inline_catch(n) + 1000


@func()
def catch_inline_reraise(n: i32) -> i32:
    try:
        return inline_reraise(n)
    except ErrorA as e:
        return e.code + 100


@func()
def catch_inline_call(n: i32) -> i32:
    try:
        return inline_forward(n) + 10
    except ErrorA as e:
        return e.code + 900


@func()
def catch_inline_in_branch(n: i32) -> i32:
    # the raise sits in a branch of the try body, so the other branch of the
    # inline body still falls into the code after the call
    try:
        if n < 0:
            inline_raise(n)
            return 1
        return 2
    except ErrorA as e:
        return e.code + 100


@func(exceptions=(ErrorA, ErrorB))
def raise_in_clause(n: i32) -> i32:
    # a raise inside a clause body belongs to the try *enclosing* the clause,
    # never to the clause's own try again
    try:
        raise_b(n)
    except ErrorB:
        try:
            raise_a(n)
        except ErrorB:
            return 1
        return 2
    return 3


@func(exceptions=(ErrorA,))
def forward_no_return(n: i32) -> i32:
    # the ``+ 10`` after the call is dead: the inlined body never returns
    return inline_no_return(n) + 10


@func(exceptions=(ErrorA,))
def nested_forward_no_return(n: i32) -> i32:
    return inline_forward_no_return(n) + 10


@func()
def catch_no_return(n: i32) -> i32:
    try:
        inline_no_return(n)
        return 1
    except ErrorA as e:
        return e.code + 100


@func()
def catch_raise_either(n: i32) -> i32:
    try:
        inline_raise_either(n)
        return 1
    except ErrorA as e:
        return e.code + 100
    except ErrorB as e:
        return e.n + 200


@func()
def raise_in_inline_undeclared(n: i32) -> i32:
    # the function proper declares nothing and the raise has no try to be
    # caught by: it is rejected when the error is tagged
    return inline_raise(n) + 10


@struct()
class ErrorC(Exception):
    code: i32


@func(exceptions="infer")
def inferred_raise(n: i32) -> i32:
    if n < 0:
        raise ErrorC(7)
    return n + 1


@func(exceptions="infer")
def inferred_return_first(n: i32) -> i32:
    # the successful return is typed before the raise that widens the set, so
    # the return path clears the error code before the set is even known
    if n >= 0:
        return n + 1
    raise ErrorC(7)


@func(exceptions="infer")
def inferred_forward(n: i32) -> i32:
    return inferred_raise(n) + 10


@func(exceptions="infer")
def inferred_catch(n: i32) -> i32:
    try:
        return inferred_raise(n)
    except ErrorC as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_two(n: i32) -> i32:
    return raise_a(n) + raise_b(n)


@func(exceptions="infer")
def nested_catch(n: i32) -> i32:
    try:
        try:
            return raise_a(n)
        except ErrorB:
            return -1
    except ErrorA as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_inline_catch(n: i32) -> i32:
    # the clause catches only ``ErrorA``, so only the ``ErrorB`` the inlined body
    # lets through is inferred into this function's own set
    try:
        return inline_raise_two(n)
    except ErrorA as e:
        return e.code + 100


@func(exceptions="infer")
def inferred_no_return(n: i32) -> i32:
    # an inferred exception set leaves the declared return type in place, even
    # though the inlined body never returns and no value is ever stored
    return inline_no_return(n) + 10


@func(exceptions="infer")
def inferred_wider_return(n: i32) -> i64:
    # ... and it is the *declared* type that fixes the value result, not the
    # type of the values the body stores (``n + 1`` here is an ``i32``)
    if n < 0:
        raise ErrorA(7)
    return n + 1


@struct()
class ErrorBig(Exception):
    a: i64
    b: i64
    c: i64


@func()
def make_big(n: i32) -> ErrorBig:
    return ErrorBig(n, n + 1, n + 2)


@func(exceptions="infer")
def raise_big(n: i32) -> i32:
    # a call returning an aggregate is raised through a result pointer: the
    # delivery into the (still inferred) space is deferred to its commit
    if n < 0:
        raise make_big(n)
    return n + 1


@func(exceptions="infer")
def catch_big(n: i32) -> i32:
    try:
        raise_big(n)
    except ErrorBig as e:
        return e.a + e.b + e.c
    return 0


def call_with_error(handle: Any, arg: int) -> tuple[int, int, int]:
    """Compile ``handle`` (a Python-side call is rejected after compiling it)
    and invoke its native form directly, with the hidden error-code and payload
    pointers filled in by this helper.  Returns ``(result, error code, payload
    read as i32)``; the payload is only meaningful when the code is not zero."""
    entry = handle.get_entry()
    if len(entry.specs) == 0:
        with TestCase().assertRaises(SpyError):
            handle(arg)
    assert len(entry.specs) == 1, entry.specs
    instance = next(iter(entry.specs.values()))
    native = instance.wrapper_fn or instance.native_fn
    assert native is not None
    code = ctypes.c_uint8(255)
    payload = ctypes.c_int32(-1)
    result = native.call(
        ctypes.c_int32(arg),
        ctypes.c_void_p(ctypes.addressof(code)),
        ctypes.c_void_p(ctypes.addressof(payload)),
    )
    return int(result), int(code.value), int(payload.value)


class SpyErrorUnionTest(TestCase):
    """``raise`` delivers an exception into the function's error location - an
    error code and a payload - and ends the path; a call of a function that
    may raise carries the error to its caller, remapping the tag to the
    caller's own exception set."""

    def test_a_raising_function_lowers_to_a_code_and_a_payload(self) -> None:
        self._compile(raise_a, 1)
        args, ret = mir_signature(raise_a)
        i32_mir = mir.IntType(32, True)
        # ``fn(i32, *u1, *payload) -> i32``: one exception needs one bit for
        # its two tags (no error and the exception)
        self.assertEqual(ret, i32_mir)
        self.assertEqual(args[0], i32_mir)
        self.assertEqual(args[1], mir.PointerType(mir.IntType(1, False), False))
        self.assertIsInstance(args[2], mir.PointerType)

    def test_a_function_that_raises_nothing_has_no_error_part(self) -> None:
        # the empty exception set is the unit type: an exception-free function's
        # lowered signature has neither an error code nor a payload
        self.assertEqual(catch_without_declaring(5), 6)
        args, ret = mir_signature(catch_without_declaring)
        i32_mir = mir.IntType(32, True)
        self.assertEqual(args, (i32_mir,))
        self.assertEqual(ret, i32_mir)

    def test_raising_an_undeclared_exception_is_rejected(self) -> None:
        # the function declares no exception, so the error is rejected when it
        # is tagged (``HirRunner._add_function_exception``), with a hint about the declaration
        with self.assertRaises(CompileError) as ctx:
            raise_undeclared(5)
        self.assertIn('cannot raise', str(ctx.exception))

    def test_a_caller_widens_the_error_code(self) -> None:
        self._compile(forward_raise, 1)
        args, _ = mir_signature(forward_raise)
        # two exceptions need two bits (three tags)
        self.assertEqual(args[1], mir.PointerType(mir.IntType(2, False), False))

    def test_the_declared_exceptions_are_recorded(self) -> None:
        entry = raise_a.get_entry()  # pyright: ignore
        signature = entry.hir.signature
        assert signature.exceptions is not None
        self.assertEqual(list(signature.exceptions.values), [struct_type(ErrorA)])

    def test_calling_a_raising_function_from_python_is_rejected(self) -> None:
        with self.assertRaises(SpyError) as ctx:
            raise_a(1)
        self.assertIn('not supported', str(ctx.exception))

    def _compile(self, handle: Any, arg: int) -> None:
        # a call from Python compiles the function before the boundary rejects
        # it: the Python-side handling of errors is not implemented yet
        with self.assertRaises(SpyError):
            handle(arg)


class SpyTryExceptTest(TestCase):
    """``try``/``except`` catches the error a call (or a ``raise``) delivered:
    the clause whose type matches the error code runs, taking the payload for
    its ``as`` name; an error no clause matches re-raises to the enclosing
    handler, or out of the function."""

    def test_catching_the_normal_result_passes_through(self) -> None:
        result, code, _ = call_with_error(catch_bound, 5)
        self.assertEqual((result, code), (6, 0))

    def test_catching_binds_the_payload(self) -> None:
        result, code, _ = call_with_error(catch_bound, -3)
        self.assertEqual((result, code), (107, 0))

    def test_a_plain_clause_catches_without_binding(self) -> None:
        result, code, _ = call_with_error(catch_multi, -3)
        self.assertEqual((result, code), (207, 0))

    def test_multiple_clauses_pick_the_matching_one(self) -> None:
        self.assertEqual(call_with_error(catch_multi, 5)[:2], (6, 0))
        self.assertEqual(call_with_error(catch_multi, -3)[:2], (207, 0))

    def test_an_unmatched_error_escapes_the_try(self) -> None:
        _result, code, payload = call_with_error(escape_through, -3)
        # the two exceptions need two bits, and the code is non-zero: the
        # error escaped the ``try`` and reached the function's caller
        self.assertNotEqual(code, 0)
        self.assertEqual(payload, 7)

    def test_a_caught_exception_need_not_be_declared(self) -> None:
        # the function declares nothing, so Python can call it directly
        self.assertEqual(catch_without_declaring(5), 6)
        self.assertEqual(catch_without_declaring(-3), 507)


class SpyInlineErrorTest(TestCase):
    """Errors of an inlined body.  An inlined plain Python function is part of
    its caller, so its errors are delivered into - and caught by - the error
    space and the try blocks enclosing the *caller*: the error location and the
    handler are read from one innermost-*open*-try lookup across the inlined
    frames (see ``HirRunner._active_try``), which keeps the two in step."""

    def test_the_callers_try_catches_the_raise(self) -> None:
        self.assertEqual(catch_inline_raise(5), 6)
        self.assertEqual(catch_inline_raise(-3), 107)

    def test_the_bodys_own_try_catches_the_raise(self) -> None:
        # the raise is inlined *into* the body whose try catches it, so the two
        # sit in different frames
        self.assertEqual(catch_inline_own_try(5), 1006)
        self.assertEqual(catch_inline_own_try(-3), 1107)

    def test_a_body_with_no_falling_path_hands_its_error_to_the_caller(self) -> None:
        # the inlined body never reaches the caller's continuation, so the code
        # after the call - ``return 1`` here - is dead and must not be typed
        self.assertEqual(catch_no_return(3), 107)

    def test_a_body_whose_paths_all_raise_hands_its_error_to_the_caller(self) -> None:
        self.assertEqual(catch_raise_either(-3), 107)
        self.assertEqual(catch_raise_either(3), 209)

    def test_an_error_of_a_body_with_no_falling_path_escapes_the_function(self) -> None:
        # the error leaves the inlined body, and the function with it: the ``+
        # 10`` after the call is dead all the same
        _result, code, payload = call_with_error(forward_no_return, 3)
        self.assertEqual((code, payload), (1, 7))

    def test_the_same_through_two_levels_of_inlining(self) -> None:
        _result, code, payload = call_with_error(nested_forward_no_return, 3)
        self.assertEqual((code, payload), (1, 7))

    def test_an_unmatched_error_of_the_body_re_raises_to_the_caller(self) -> None:
        self.assertEqual(catch_inline_reraise(5), 6)
        self.assertEqual(catch_inline_reraise(-3), 107)

    def test_a_native_call_in_the_body_reaches_the_callers_try(self) -> None:
        # the error of the native call is carried into the caller's space, and
        # the caller's try is the handler the error propagates to
        self.assertEqual(catch_inline_call(5), 26)
        self.assertEqual(catch_inline_call(-3), 907)

    def test_a_raise_in_a_branch_of_the_body(self) -> None:
        # the raising branch leaves the try through the handler, while the other
        # branch of the body still falls into the code after the call
        self.assertEqual(catch_inline_in_branch(5), 2)
        self.assertEqual(catch_inline_in_branch(-3), 107)

    def test_a_raise_in_a_clause_body_is_not_caught_by_it_again(self) -> None:
        # the clause's own try is no longer open while its body is typed, so the
        # inner try's unmatched error belongs to the function's space
        self.assertEqual(call_with_error(raise_in_clause, 5), (3, 0, -1))
        _result, code, payload = call_with_error(raise_in_clause, -3)
        entry = raise_in_clause.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # it is the ``ErrorA`` the inner try raised, tagged in the declared
        # set's own order
        types = list(instance.ret_sig.exceptions.values)
        self.assertEqual((code, payload), (types.index(struct_type(ErrorA)) + 1, 7))

    def test_an_inferred_set_takes_only_the_error_the_try_lets_through(self) -> None:
        # an inferred function whose try catches one of the two exceptions its
        # inlined callee may raise: the caught one never leaves the try, the
        # other escapes it and is inferred into the function's own set
        self.assertEqual(call_with_error(inferred_inline_catch, 0)[:2], (1, 0))
        self.assertEqual(call_with_error(inferred_inline_catch, 20)[:2], (107, 0))
        _result, code, payload = call_with_error(inferred_inline_catch, 5)
        entry = inferred_inline_catch.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # only ``ErrorB`` is inferred, and being the only exception of the space
        # it has tag 1 (one bit for its two tags)
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorB)])
        self.assertEqual((code, payload), (1, 9))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_an_inlined_raise_needs_a_declaration_or_a_try(self) -> None:
        # without either, the error has nowhere to go: it is rejected when it is
        # tagged, naming the declaration that would allow it
        with self.assertRaises(CompileError) as ctx:
            raise_in_inline_undeclared(-3)
        self.assertIn('cannot raise', str(ctx.exception))


class SpyInferTest(TestCase):
    """An inferred exception set (``exceptions="infer"``): the space's set,
    its code width and its payload union follow from what the body actually
    raises or lets through, fixed when the function's analysis ends."""

    def test_an_inferred_set_lowers_to_a_code_and_a_payload(self) -> None:
        result, code, _ = call_with_error(inferred_raise, 5)
        self.assertEqual((result, code), (6, 0))
        result, code, payload = call_with_error(inferred_raise, -3)
        self.assertEqual((result, code, payload), (0, 1, 7))
        entry = inferred_raise.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorC)])
        args = instance.mir.args
        self.assertEqual(args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_an_inferred_error_propagates(self) -> None:
        self.assertEqual(call_with_error(inferred_forward, 5)[:2], (16, 0))
        self.assertEqual(call_with_error(inferred_forward, -3)[1:], (1, 7))

    def test_a_return_before_the_raise_still_clears_the_code(self) -> None:
        # the successful path is typed before the raise widens the set
        self.assertEqual(call_with_error(inferred_return_first, 5)[:2], (6, 0))
        self.assertEqual(call_with_error(inferred_return_first, -3)[1:], (1, 7))

    def test_a_fully_caught_inferred_function_is_callable(self) -> None:
        self.assertEqual(inferred_catch(5), 6)
        self.assertEqual(inferred_catch(-3), 107)

    def test_an_inferred_set_collects_every_callee(self) -> None:
        self.assertEqual(call_with_error(inferred_two, 5)[:2], (13, 0))
        self.assertEqual(call_with_error(inferred_two, -3)[1:], (1, 7))
        entry = inferred_two.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        # first-delivery order, and two exceptions need two bits
        self.assertEqual(
            list(instance.ret_sig.exceptions.values),
            [struct_type(ErrorA), struct_type(ErrorB)],
        )
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(2, False), False))

    def test_an_inferred_set_keeps_the_declared_return_type(self) -> None:
        # the exceptions are inferred, the value type is the declared ``i64`` -
        # not the ``i32`` the body stores - so the value is the by-value result
        # and the error code goes through a pointer
        self.assertEqual(call_with_error(inferred_wider_return, 3)[:2], (4, 0))
        self.assertEqual(call_with_error(inferred_wider_return, -3)[1:], (1, 7))
        entry = inferred_wider_return.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(list(instance.ret_sig.exceptions.values), [struct_type(ErrorA)])
        self.assertEqual(instance.mir.ret_type, mir.IntType(64, True))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_a_body_that_never_returns_keeps_the_declared_return_type(self) -> None:
        # nothing is stored into the result location at all, so the result type
        # has to come from the declaration
        _result, code, payload = call_with_error(inferred_no_return, 3)
        self.assertEqual((code, payload), (1, 7))
        entry = inferred_no_return.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(instance.mir.ret_type, mir.IntType(32, True))
        self.assertEqual(instance.mir.args[1], mir.PointerType(mir.IntType(1, False), False))

    def test_nested_tries_hand_over_inward(self) -> None:
        # the inner ``try`` catches nothing, so its error is re-raised into the
        # outer one, which catches it
        self.assertEqual(nested_catch(5), 6)
        self.assertEqual(nested_catch(-3), 107)

    def test_raising_a_call_returning_an_aggregate(self) -> None:
        # ``raise make_big(n)`` hands a result pointer to the callee, deferred
        # until the inferred space's payload type is known
        self.assertEqual(catch_big(-3), -6)
        self.assertEqual(catch_big(5), 0)


@func()
def loop_forever(n: i32):
    # a body that never delivers a result and never raises: the function can
    # never return at all (a ``mir.NoReturn`` function)
    while True:
        n = n + 1


@func()
def call_loop_forever(n: i32) -> i32:
    # the ``return -1`` after the call is dead: the call never comes back, and
    # the declared return type keeps the caller's own result an ``i32``
    if n < 0:
        loop_forever(n)
        return -1
    return n + 1


@func(exceptions=(ErrorA,))
def always_raises(n: i32):
    # no value to return and one exception: the error code is zero-sized
    # (``u0``) and the payload union takes the by-value result
    raise ErrorA(n)


@func()
def catch_always_raises(n: i32) -> i32:
    try:
        always_raises(n)
        return 1
    except ErrorA as e:
        return e.code + 100


@func()
def never_declared(n: i32) -> Never:
    # ``-> Never`` declares that no value is ever returned: the body may not
    # return, and one that also raises nothing cannot return at all
    while True:
        n = n + 1


@func()
def call_never_declared(n: i32) -> i32:
    if n < 0:
        never_declared(n)
        return -1
    return n + 1


@func()
def forward_never(n: i32) -> Never:
    # a noreturn call ends the path: the implicit fallthrough of the body is
    # never reached (and so is not rejected)
    never_declared(n)


@func()
def call_forward_never(n: i32) -> i32:
    if n < 0:
        forward_never(n)
        return -1
    return n + 1


@func(exceptions=(ErrorA,))
def raises_never(n: i32) -> Never:
    # the declared empty value behaves like the inferred one of ``always_raises``
    raise ErrorA(n)


@func()
def catch_raises_never(n: i32) -> i32:
    try:
        raises_never(n)
        return 1
    except ErrorA as e:
        return e.code + 200


@func()
def returns_from_never(n: i32) -> Never:
    # the return is rejected: the function has no value to return (and the
    # store of its value into the empty result location is a no-op)
    return n  # pyright: ignore


@func()
def falls_through_never(n: i32) -> Never:  # pyright: ignore
    # falling off the end of the body is a returning path too
    n = n + 1


def inline_never(n: i32) -> Never:
    # an inlined body that always raises: its declaration holds, and the caller's
    # code after the call is dead (the body reaches no continuation)
    raise ErrorA(n)


@func(exceptions="infer")
def catch_inline_never(n: i32) -> i32:
    try:
        inline_never(n)
        return 1
    except ErrorA as e:
        return e.code + 300


def inline_returns_from_never(n: i32) -> Never:
    # an inlined body has no convention of its own, so its declaration is what
    # rejects the return
    return n  # pyright: ignore


@func()
def call_inline_returns_from_never(n: i32) -> i32:
    return inline_returns_from_never(n) + 1


def inline_falls_through_never(n: i32) -> Never:  # pyright: ignore
    n = n + 1


@func()
def call_inline_falls_through_never(n: i32) -> i32:
    inline_falls_through_never(n)
    return 1


@func(exceptions="infer")
def forward_always_raises(n: i32) -> i32:
    # the callee's payload union is a subset of this function's own (which also
    # holds ``ErrorB``): the by-value payload is written into the function's own
    # payload through a pointer reinterpretation (a union cannot be converted),
    # and the error is then only tagged (no copy)
    if n > 100:
        raise ErrorB(n)
    return always_raises(n) + 1


@func(exceptions="infer")
def catch_forward_always_raises(n: i32) -> i32:
    try:
        return forward_always_raises(n)
    except ErrorA as e:
        return e.code + 100


class SpyNoReturnTest(TestCase):
    """A function that cannot return: a body that never delivers a result and
    raises nothing lowers to a ``mir.NoReturn`` function - its LLVM form is
    marked ``noreturn`` and a call of it ends the block it sits in, so the code
    after the call is dead - while a value-less function that raises carries
    error codes that start at 0 (one exception needs no code at all)."""

    def test_a_value_less_function_without_errors_is_noreturn(self) -> None:
        self.assertEqual(call_loop_forever(5), 6)
        args, ret = mir_signature(loop_forever)
        self.assertIs(ret, mir.NORETURN)
        self.assertEqual(args, (mir.IntType(32, True),))
        # the dead ``return -1`` was never typed, so the caller's result is
        # still the declared ``i32``
        _args, caller_ret = mir_signature(call_loop_forever)
        self.assertEqual(caller_ret, mir.IntType(32, True))

    def test_the_noreturn_function_is_marked_in_the_ir(self) -> None:
        self.assertEqual(call_loop_forever(5), 6)
        entry = loop_forever.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        native = instance.wrapper_fn or instance.native_fn
        assert native is not None
        text = '\n'.join(native.print_all())
        self.assertIn('define void', text)
        self.assertIn('noreturn', text)
        # a call of it ends its block, and LLVM wants every block to end with an
        # explicit terminator
        self.assertIn('unreachable', text)

    def test_a_declared_never_function_is_noreturn(self) -> None:
        # ``-> Never`` declares the empty return type, which with an empty
        # exception set is a function that can never return
        self.assertEqual(call_never_declared(5), 6)
        args, ret = mir_signature(never_declared)
        self.assertIs(ret, mir.NORETURN)
        self.assertEqual(args, (mir.IntType(32, True),))

    def test_a_declared_never_function_matches_the_inferred_one(self) -> None:
        # the declared empty value lowers to the same result type as the one
        # ``always_raises`` infers from its body: no code, the payload by value
        self.assertEqual(catch_raises_never(7), 207)
        self.assertEqual(mir_signature(raises_never), mir_signature(always_raises))

    def test_a_never_function_forwarding_a_noreturn_call(self) -> None:
        # the call ends the path, so the body's fallthrough never runs
        self.assertEqual(call_forward_never(5), 6)
        _args, ret = mir_signature(forward_never)
        self.assertIs(ret, mir.NORETURN)

    def test_returning_from_a_never_function_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            returns_from_never(5)
        self.assertIn('cannot return', str(ctx.exception))

    def test_falling_through_a_never_function_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            falls_through_never(5)
        self.assertIn('fall off its end', str(ctx.exception))

    def test_an_inlined_never_body_may_not_return(self) -> None:
        self.assertEqual(catch_inline_never(7), 307)
        with self.assertRaises(CompileError) as ctx:
            call_inline_returns_from_never(5)
        self.assertIn('cannot return', str(ctx.exception))

    def test_an_inlined_never_body_may_not_fall_through(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            call_inline_falls_through_never(5)
        self.assertIn('fall off its end', str(ctx.exception))

    def test_a_value_less_function_with_one_exception_has_no_code(self) -> None:
        self.assertEqual(catch_always_raises(7), 107)
        args, ret = mir_signature(always_raises)
        # no error code at all: the i-th exception of a value-less function is
        # tagged ``i``, so a single one needs no code, and the payload union is
        # the by-value result
        self.assertEqual(args, (mir.IntType(32, True),))
        self.assertIsInstance(ret, mir.UnionType)

    def test_a_by_value_payload_fills_a_wider_union(self) -> None:
        # the caller's own payload union is wider than the callee's, so the
        # returned union is written through a reinterpreted pointer and the error
        # is carried on as a tag alone
        self.assertEqual(call_with_error(catch_forward_always_raises, 7)[:2], (107, 0))
        _result, code, payload = call_with_error(forward_always_raises, 7)
        self.assertEqual((code, payload), (2, 7))
        _result, code, payload = call_with_error(forward_always_raises, 105)
        self.assertEqual((code, payload), (1, 105))
        entry = forward_always_raises.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.ret_sig is not None
        self.assertEqual(
            list(instance.ret_sig.exceptions.values),
            [struct_type(ErrorB), struct_type(ErrorA)],
        )


all_tests = [
    SpyErrorUnionPrimitiveTest,
    SpyErrorUnionTest,
    SpyTryExceptTest,
    SpyInlineErrorTest,
    SpyInferTest,
    SpyNoReturnTest,
]

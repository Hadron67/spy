import ctypes
from typing import Protocol
from unittest import TestCase

from ..compiler import (
    CompileError,
    SpyError,
    i32,
    i64,
    mir,
    sval,
)
from ..compiler import as_ as spy_as
from ..compiler import typeof as spy_typeof
from ..compiler.dsl import _GLOBAL_CONTEXT, decl_func, func, func_type, struct
from ..compiler.syntax import (
    ConstPtr,
    as_func_ptr,
)
from ..compiler.util import FrozenArraySet
from .structs import MIR_CACHE, struct_type

# ---------------------------------------------------------------------------
# function types: ``@func_type`` declares a spy function type from a Protocol's
# ``__call__``, and a value of a pointer to one is called through the pointer
# (the only callable form: a function type itself is dynamically sized)
# ---------------------------------------------------------------------------


@struct()
class FnError(Exception):
    code: i32


@func_type(exceptions=(FnError,))
class Fallible(Protocol):
    # a spy-convention function pointer that may raise ``FnError``: the pointed-
    # to function carries the error code and payload like a compiled spy
    # function (the default calling convention)
    def __call__(self, n: i32) -> i32: ...


@func_type(callconv='c')
class CBinary(Protocol):
    # a C function pointer: every argument by value, the result by value, no
    # exceptions
    def __call__(self, a: i32, b: i32) -> i32: ...


@func_type(callconv='c')
class CWithDefault(Protocol):
    def __call__(self, a: i32, b: i32 = 10) -> i32: ...


@struct(extern_c=True)
class Wide:
    a: i64
    b: i64
    c: i64


@func_type(callconv='c')
class CGetWide(Protocol):
    def __call__(self) -> Wide: ...


@func_type(callconv='c', exceptions=(FnError,))
class CBad(Protocol):
    def __call__(self, n: i32) -> i32: ...


@func_type(exceptions=FnError)
class FallibleSingle(Protocol):
    # the single-type spelling of ``exceptions`` (see ``_normalize_exceptions``)
    def __call__(self, n: i32) -> i32: ...


@func(exceptions=FnError)
def raise_fn_error(n: i32) -> i32:
    if n < 0:
        raise FnError(1)
    return n + 1


@func()
def call_fallible(f: Fallible, n: i32) -> i32:
    # a DST parameter is passed by reference; the call carries the pointer's
    # error code and payload, routed to the try here
    try:
        return f(n)
    except FnError as e:
        return e.code + 1000


@func()
def call_c_binary(p: ConstPtr[CBinary], a: i32, b: i32) -> i32:
    # a pointer to a function type is an ordinary value
    return p[...](a, b)


@func()
def call_c_default(p: ConstPtr[CWithDefault], a: i32) -> i32:
    # the argument the call leaves out takes the pointer type's default
    return p[...](a)


@func(callconv='c', exceptions='infer')
def c_inferred_raise(n: i32) -> i32:
    # a C function may not raise; the inferred exception set is rejected once
    # the body is typed
    if n < 0:
        raise FnError(1)
    return n


@func(callconv='c')
def c_multi(n: i32) -> tuple[i32, i32]:
    # a C function may return only one value
    return n, n + 1


@decl_func("abs")
def c_abs(x: i32) -> i32:
    # an external function declared by its C link name (the default callconv)
    ...


@decl_func("bad", exceptions=FnError)
def c_bad_decl(x: i32) -> i32:
    # a C function may not declare exceptions: resolving this is rejected
    ...


@func()
def call_c_abs(x: i32) -> i32:
    # the declared name is a function pointer value: the call goes through it
    return c_abs(x)


@func()
def ptr_target_incr(n: i32) -> i32:
    return n + 1


@func()
def call_spy_func_ptr(n: i32) -> i32:
    # ``as_func_ptr`` materializes the address of a spy function
    p = as_func_ptr(spy_typeof(ptr_target_incr), ptr_target_incr)  # pyright: ignore
    return p[...](n)


@func_type()
class IntUnary(Protocol):
    def __call__(self, n: i32) -> i32: ...


@func()
def apply_int_unary(p: ConstPtr[IntUnary], n: i32) -> i32:
    return p[...](n)


@func()
def call_spy_func_ptr_typed(n: i32) -> i32:
    # the pointer of a spy function converts to a declared function-pointer type
    p = as_func_ptr(spy_typeof(ptr_target_incr), ptr_target_incr)  # pyright: ignore
    return apply_int_unary(p, n)


class SpyFuncTypeTest(TestCase):
    """``@func_type``: the annotations of a Protocol's ``__call__`` (its
    receiver dropped) become a ``sval.FunctionType``, resolved in the context
    that names the declaration."""

    def test_a_function_type_resolves_from_the_call_signature(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(Fallible)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual([a.name for a in t.args], ['n'])
        self.assertEqual(t.args[0].type, sval.IntType(32, True))
        self.assertEqual(t.return_type, sval.IntType(32, True))
        self.assertEqual(tuple(t.exceptions), (struct_type(FnError),))
        self.assertEqual(t.callconv, 'default')

    def test_a_default_of_a_function_type_parameter_is_carried(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CWithDefault)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual(t.args[1].default_value, 10)

    def test_a_pointer_to_a_function_type_is_a_pointer(self) -> None:
        t = sval.as_value(ConstPtr[CBinary], _GLOBAL_CONTEXT)
        assert isinstance(t, sval.PointerType)
        self.assertIsInstance(t.elem, sval.FunctionType)
        self.assertIs(t.is_const, True)

    def test_a_c_function_type_forces_every_argument_by_value(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CBinary)
        assert isinstance(t, sval.FunctionType)
        i32_mir = mir.IntType(32, True)
        # no by-ref argument and no result pointer: ``fn(i32, i32) -> i32``
        self.assertEqual(
            t.to_mir_type(MIR_CACHE), mir.FunctionType((i32_mir, i32_mir), i32_mir, 'c', True),
        )

    def test_a_c_function_type_returns_an_aggregate_by_value(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(CGetWide)
        assert isinstance(t, sval.FunctionType)
        mirror = t.to_mir_type(MIR_CACHE)
        assert isinstance(mirror, mir.FunctionType)
        # a wide aggregate would go through a result pointer under the default
        # convention; the C one returns it by value (no hidden pointer argument)
        self.assertEqual(mirror.args, ())
        self.assertIsInstance(mirror.return_type, mir.StructType)

    def test_a_c_function_type_may_not_declare_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(CBad)
        self.assertIn('may not declare exceptions', str(ctx.exception))


class SpyFnPtrCallTest(TestCase):
    """A runtime function pointer is called through the address its value
    carries; the call is emitted like a call of a compiled function."""

    def test_calling_a_c_function_pointer(self) -> None:
        ptr_type = sval.as_value(ConstPtr[CBinary], _GLOBAL_CONTEXT)
        proto = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        callback = proto(lambda a, b: a + b)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_c_binary(spy_as(pointer, ptr_type), 2, 3), 5)  # pyright: ignore

    def test_the_default_of_a_function_pointer_is_filled_in(self) -> None:
        ptr_type = sval.as_value(ConstPtr[CWithDefault], _GLOBAL_CONTEXT)
        proto = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        callback = proto(lambda a, b: a + b)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_c_default(spy_as(pointer, ptr_type), 5), 15)  # pyright: ignore

    def test_calling_a_spy_function_pointer(self) -> None:
        # the default convention: the callee carries the error code and payload
        # through trailing pointers
        ptr_type = _GLOBAL_CONTEXT.resolve_global(Fallible)
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        )

        def success(n: int, code_ptr: int, payload_ptr: int) -> int:
            ctypes.cast(code_ptr, ctypes.POINTER(ctypes.c_uint8))[0] = 0
            return n + 1

        callback = proto(success)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_fallible(spy_as(pointer, ptr_type), 5), 6)  # pyright: ignore

    def test_the_error_of_a_spy_function_pointer_is_routed(self) -> None:
        ptr_type = _GLOBAL_CONTEXT.resolve_global(Fallible)
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        )

        def fail(n: int, code_ptr: int, payload_ptr: int) -> int:
            ctypes.cast(code_ptr, ctypes.POINTER(ctypes.c_uint8))[0] = 1
            ctypes.cast(payload_ptr, ctypes.POINTER(ctypes.c_int32))[0] = 7
            return 0

        callback = proto(fail)
        pointer = ctypes.cast(callback, ctypes.c_void_p)
        self.assertEqual(call_fallible(spy_as(pointer, ptr_type), 5), 1007)  # pyright: ignore


class SpyCallconvTest(TestCase):
    """A non-default calling convention forces every argument by value, the
    result by value, and forbids raising and multiple results."""

    def test_a_c_function_may_not_infer_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_inferred_raise(1)
        self.assertIn('may not raise', str(ctx.exception))

    def test_a_c_function_may_not_return_several_values(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            c_multi(1)
        self.assertIn('only one value', str(ctx.exception))


class SpyFrozenArraySetTest(TestCase):
    """``util.FrozenArraySet``: an immutable, ordered, deduplicated collection
    whose equality and hashing are by value (it may be a field of a frozen
    dataclass)."""

    def test_it_keeps_the_insertion_order_and_drops_duplicates(self) -> None:
        s = FrozenArraySet((1, 2, 1, 3, 2))
        self.assertEqual(s.values, (1, 2, 3))
        self.assertEqual(list(s), [1, 2, 3])
        self.assertEqual(len(s), 3)

    def test_membership_and_indexing(self) -> None:
        s: FrozenArraySet[str] = FrozenArraySet(('a', 'b', 'c'))
        self.assertIn('b', s)
        self.assertNotIn('z', s)
        self.assertEqual(s[0], 'a')
        self.assertEqual(s.index('c'), 2)
        self.assertIsNone(s.index_of('z'))
        with self.assertRaises(ValueError):
            s.index('z')

    def test_equality_and_hashing_are_by_value(self) -> None:
        # the order is part of the value (an error-code or tag order is)
        self.assertEqual(FrozenArraySet((1, 2)), FrozenArraySet((1, 2)))
        self.assertNotEqual(FrozenArraySet((1, 2)), FrozenArraySet((2, 1)))
        self.assertEqual(hash(FrozenArraySet((1, 2))), hash(FrozenArraySet((1, 2))))
        self.assertEqual(len({FrozenArraySet((1, 2)), FrozenArraySet((1, 2))}), 1)


class SpyExceptionsArgumentTest(TestCase):
    """``@func``/``@func_type`` take the exception set as a tuple or as a single
    exception type (``exceptions=ErrorA`` means ``exceptions=(ErrorA,)``)."""

    def test_a_func_type_accepts_a_single_exception(self) -> None:
        t = _GLOBAL_CONTEXT.resolve_global(FallibleSingle)
        assert isinstance(t, sval.FunctionType)
        self.assertEqual(tuple(t.exceptions), (struct_type(FnError),))

    def test_a_func_accepts_a_single_exception(self) -> None:
        # a call from Python compiles the function before the boundary rejects it
        with self.assertRaises(SpyError):
            raise_fn_error(1)
        entry = raise_fn_error.get_entry()  # pyright: ignore
        assert entry.hir.signature.exceptions is not None
        self.assertEqual(
            list(entry.hir.signature.exceptions.values), [struct_type(FnError)],
        )


class SpyDeclFuncTest(TestCase):
    """``@decl_func`` declares an external function by its link name: the
    decorated name resolves to a function pointer (``sval.DeclareFunction``)."""

    def test_a_declared_function_is_a_c_function_pointer(self) -> None:
        decl = _GLOBAL_CONTEXT.resolve_global(c_abs)
        assert isinstance(decl, sval.DeclareFunction)
        self.assertEqual(decl.linkname, 'abs')
        self.assertEqual(decl.type.callconv, 'c')
        self.assertEqual(decl.type.exceptions, sval.FrozenArraySet())
        self.assertEqual(decl.get_type(), sval.PointerType(decl.type, is_const=True))

    def test_calling_a_declared_c_function(self) -> None:
        self.assertEqual(call_c_abs(-5), 5)
        self.assertEqual(call_c_abs(7), 7)

    def test_it_lowers_to_an_extern_symbol(self) -> None:
        self.assertEqual(call_c_abs(-1), 1)
        entry = call_c_abs.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        assert instance.native_fn is not None
        self.assertIn('abs', '\n'.join(instance.native_fn.print_all()))

    def test_a_c_declaration_may_not_declare_exceptions(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            _GLOBAL_CONTEXT.resolve_global(c_bad_decl)
        self.assertIn('may not declare exceptions', str(ctx.exception))


class SpyAsFuncPtrTest(TestCase):
    """``syntax.as_func_ptr`` materializes the address of a spy function as a
    ``ConstPtr`` of its function type."""

    def test_the_pointer_of_a_spy_function_can_be_called(self) -> None:
        self.assertEqual(call_spy_func_ptr(5), 6)

    def test_the_pointer_converts_to_a_declared_function_pointer_type(self) -> None:
        self.assertEqual(call_spy_func_ptr_typed(10), 11)


all_tests = [
    SpyFuncTypeTest,
    SpyFnPtrCallTest,
    SpyCallconvTest,
    SpyFrozenArraySetTest,
    SpyExceptionsArgumentTest,
    SpyDeclFuncTest,
    SpyAsFuncPtrTest,
]

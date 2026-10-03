from unittest import TestCase
from ..compiler import (
    CompileError,
    SpyError,
    TypeMismatchError,
    compile_log,
    f32,
    f64,
    i8,
    i32,
    i64,
    mir,
    sval,
    syntax,
    u0,
    u64,
    usize,
    void,
)
from ..compiler.dsl import _GLOBAL_CONTEXT, _Context, decl_func, func, func_type, struct
from ..compiler.lower import LLVMBackend

from .basics import smoke_test
from .structs import struct_type


# ---------------------------------------------------------------------------
# cross-context calls: a spy function compiled in one host context may reach
# the functions and structs another context registered.  A handle reached
# across contexts is re-bound to the calling context (see
# ``dsl._Context.resolve_global``), so every context parses, compiles and
# names its own copies - a struct declared below has one type object per
# context, and a spy body only ever sees its own context's objects.
# ---------------------------------------------------------------------------


@struct()
class CrossPair:
    a: i32
    b: i32

    def total(self) -> i32:
        return self.a + self.b


@struct()
class CrossBox[T]:
    v: T

    def get(self) -> T:
        return self.v


# a second host context: its own backend and symbol table, so it links the
# functions and structs it reaches into its own modules
_CROSS_CONTEXT = _Context(LLVMBackend())


@_CROSS_CONTEXT.func()
def cross_call_global_fn(a: i32, b: i32) -> i32:
    return smoke_test(a, b) + 1


@_CROSS_CONTEXT.func()
def cross_build_global_struct(x: i32) -> i32:
    p: CrossPair = CrossPair(x, x + 1)
    return p.total()


@_CROSS_CONTEXT.func()
def cross_build_global_generic(x: i32) -> i32:
    b: CrossBox[i32] = CrossBox(x)
    return b.get()


@_CROSS_CONTEXT.struct()
class CrossLocalStruct:
    a: i32

    # the method is registered in the *global* context (``func`` is its
    # decorator): the struct's own context re-binds it when it builds the head
    @func()
    def doubled(self) -> i32:
        return self.a * 2


@_CROSS_CONTEXT.func()
def cross_build_local_struct(x: i32) -> i32:
    c: CrossLocalStruct = CrossLocalStruct(x)
    return c.doubled()


@func()
def cross_target_fn(n: i32) -> i32:
    return n + 1


@_CROSS_CONTEXT.func()
def cross_call_target_fn(n: i32) -> i32:
    return cross_target_fn(n) * 2


class SpyCrossContextTest(TestCase):
    """A spy function compiled in a second host context reaches the functions
    and structs the global context registered.  Every handle is re-bound to
    the calling context, so each context compiles its own copies - the struct
    types of one context are not the types of another."""

    def test_calling_a_global_function(self) -> None:
        self.assertEqual(cross_call_global_fn(2, 3), 6)

    def test_constructing_a_global_struct(self) -> None:
        self.assertEqual(cross_build_global_struct(4), 9)

    def test_constructing_a_global_generic_struct(self) -> None:
        self.assertEqual(cross_build_global_generic(5), 5)

    def test_calling_a_global_method_on_a_local_struct(self) -> None:
        self.assertEqual(cross_build_local_struct(6), 12)

    def test_a_global_function_resolves_to_this_contexts_copy(self) -> None:
        # the call compiles ``smoke_test`` anew in the second context: its
        # entry there is not the global one, whose spec was compiled into the
        # global context's modules (and named in its symbol table)
        self.assertEqual(cross_call_global_fn(2, 3), 6)
        own = _CROSS_CONTEXT.resolve_global(smoke_test)
        assert own is not None
        self.assertIsNot(own, smoke_test.get_entry())  # pyright: ignore
        self.assertEqual(len(own.specs), 1)  # pyright: ignore

    def test_a_global_struct_resolves_to_this_contexts_type(self) -> None:
        own = _CROSS_CONTEXT.resolve_global(CrossPair)
        assert isinstance(own, sval.StructType)
        global_type = struct_type(CrossPair)
        self.assertIsNot(own, global_type)
        # the copy names the same struct: the same fields, with the same types
        for field in ('a', 'b'):
            self.assertEqual(own.field_type(field), global_type.field_type(field))

    def test_the_global_context_is_not_polluted(self) -> None:
        # the second context resolves a copy of the callee and specializes
        # *its* copy, so the global entry the global context registered keeps
        # its own specialization set (empty here - nothing in the global
        # context calls it)
        self.assertEqual(cross_call_target_fn(3), 8)
        global_entry = cross_target_fn.get_entry()  # pyright: ignore
        self.assertEqual(len(global_entry.specs), 0)  # pyright: ignore
        own = _CROSS_CONTEXT.resolve_global(cross_target_fn)
        assert own is not None
        self.assertEqual(len(own.specs), 1)  # pyright: ignore


all_tests = [
    SpyCrossContextTest,
]

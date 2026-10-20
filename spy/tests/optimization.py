"""The JIT's LLVM optimization pipeline (``lower.LLVMBackend._optimize``) and
the compile-time options that select it (``opt_level``/``opt_size``, see
``fn.OptimizationOptions``).

The suite runs at the default level (the pipeline on); the functions here
belong to host contexts of their own, one per level, so every test exercises the
level it names whatever the default is.

``NativeFn.print_all`` reports the module as it was *lowered*, whatever the
level - the pipeline runs on the copy the JIT compiles -, so a test cannot read
the optimized IR back.  Optimizing a module of its own
(``test_the_pipeline_simplifies_the_module``) is what shows the pipeline at work;
the rest checks that a level changes nothing about what a program computes, and
that an option that is not a level is rejected.
"""

from unittest import TestCase

from llvmlite import binding as llvm

from ..compiler import (
    CompileError,
    Context,
    LLVMBackend,
    OptimizationOptions,
    _optimization_options,
    i32,
)


def _context_with_options(**options: int) -> Context:
    ctx = Context(LLVMBackend())
    for name, value in options.items():
        ctx.set_compile_option(name, value)
    return ctx


_CTX_OFF = _context_with_options(opt_level=0)
_CTX_SPEED = _context_with_options(opt_level=3)
_CTX_SIZE = _context_with_options(opt_level=2, opt_size=2)

_CTX_BUG = _context_with_options(opt_level=5)
_CTX_NOT_INT = _context_with_options(opt_level='2')  # pyright: ignore[reportArgumentType]
_CTX_BAD_SIZE = _context_with_options(opt_level=3, opt_size=1)


# a body with work the pipeline folds and unrolls (each iteration adds exactly
# one to ``total``): every level has to deliver the same result

@_CTX_OFF.func()
def sum_off(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        total = total + (i * i + 1) - (i * i)
    return total


@_CTX_SPEED.func()
def sum_speed(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        total = total + (i * i + 1) - (i * i)
    return total


@_CTX_SIZE.func()
def sum_size(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        total = total + (i * i + 1) - (i * i)
    return total


@_CTX_BUG.func()
def sum_bug(n: i32) -> i32:
    return n + 1


@_CTX_NOT_INT.func()
def sum_not_int(n: i32) -> i32:
    return n + 1


@_CTX_BAD_SIZE.func()
def sum_bad_size(n: i32) -> i32:
    return n + 1


class SpyOptimizationPipelineTest(TestCase):
    """The pipeline itself, run on a module of its own."""

    def test_the_pipeline_simplifies_the_module(self) -> None:
        # a body with a redundant slot and a redundant ``+ 0``: the pipeline
        # forwards the store and folds the addition away
        text = (
            'define i32 @f(i32 %a, i32 %b) {\n'
            '  %1 = alloca i32\n'
            '  store i32 %a, ptr %1\n'
            '  %2 = load i32, ptr %1\n'
            '  %3 = add i32 %2, %b\n'
            '  %4 = add i32 %3, 0\n'
            '  ret i32 %4\n'
            '}\n'
        )
        backend = LLVMBackend()
        module = llvm.parse_assembly(text)
        self.assertIn('alloca', str(module))
        backend._optimize(module, OptimizationOptions(2, 0))
        optimized = str(module)
        self.assertNotIn('alloca', optimized)
        self.assertEqual(optimized.count('add i32'), 1)


class SpyOptimizationTest(TestCase):
    """Every level computes the same result."""

    def _check(self, fn) -> None:
        for n in (0, 1, 10, 1000):
            self.assertEqual(fn(n), n)  # pyright: ignore

    def test_the_pipeline_off(self) -> None:
        self._check(sum_off)

    def test_the_pipeline_at_the_speed_level(self) -> None:
        self._check(sum_speed)

    def test_the_pipeline_at_the_size_level(self) -> None:
        self._check(sum_size)


class SpyOptimizationOptionTest(TestCase):
    """The host selects the levels through the compile-time options
    ``opt_level``/``opt_size``; anything else is a compile error."""

    def test_the_default_is_the_speed_level_two(self) -> None:
        options = _optimization_options(Context(LLVMBackend()))
        self.assertEqual(options, OptimizationOptions(2, 0))

    def test_a_level_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            sum_bug(1)
        self.assertIn('opt_level', str(ctx.exception))

    def test_a_level_that_is_not_an_integer_is_rejected(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            sum_not_int(1)
        self.assertIn('must be an integer', str(ctx.exception))

    def test_a_size_level_needs_the_speed_level_two(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            sum_bad_size(1)
        self.assertIn('opt_size', str(ctx.exception))


all_tests = [
    SpyOptimizationPipelineTest,
    SpyOptimizationTest,
    SpyOptimizationOptionTest,
]

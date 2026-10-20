"""Source locations of compile errors.

The compiler attaches the AST node / HIR instruction that failed to a
``CompileError`` (see ``compiler.source``), together with the call sites the
error propagated through.  These tests pin down the mapping from the dedented
source ``astgen`` parses back to the original file, the whole-range caret of a
snippet, and the flat ``called from`` call chain.
"""

import importlib.util
import inspect
import warnings
from unittest import TestCase, skipUnless

from ..compiler import (
    GLOBAL_CONTEXT,
    CompileError,
    func,
    i32,
    struct,
    u8,
    unroll,
)
from ..compiler.errors import SourceLoc
from ..compiler.source import SourceFile, format_location, ipython_cell_name
from ..std.core import compile_error

_IPYTHON_AVAILABLE = importlib.util.find_spec('IPython') is not None


def _line_of(handle, needle: str) -> int:
    """The absolute line of the first source line of ``handle`` containing
    ``needle``."""
    lines, start = inspect.getsourcelines(handle.fn if hasattr(handle, 'fn') else handle)
    for index, line in enumerate(lines):
        if needle in line:
            return start + index
    raise AssertionError(f'{needle!r} not found in the source of {handle!r}')


# ---------------------------------------------------------------------------
# the failing bodies: each is compiled by its first call in the tests below
# ---------------------------------------------------------------------------


@func()
def diag_unresolved_name() -> i32:
    return foo  # noqa: F821  # pyright: ignore[reportUndefinedVariable]


@func()
def diag_store(x: i32) -> i32:
    y: u8 = x  # noqa: F841
    return 0


@func()
def diag_callee(x: i32) -> i32:
    y: u8 = x  # noqa: F841
    return 0


@func()
def diag_caller(x: i32) -> i32:
    return diag_callee(x)


@func()
def diag_closure_caller(x: i32) -> i32:
    def inner(v: i32) -> i32:
        y: u8 = v  # noqa: F841
        return 0

    return inner(x)


@func()
def diag_bad_param(x: list) -> i32:
    return 0


@func()
def diag_bad_return() -> list:
    return 0  # type: ignore[return-value]  # pyright: ignore[reportReturnType]


@func()
def diag_compile_error() -> i32:
    compile_error('boom')
    return 0


@func()
def diag_unused_pragma() -> None:
    unroll()


class DiagBadStructSource:
    x: list


DiagBadStruct = struct()(DiagBadStructSource)


class SpyDiagnosticTest(TestCase):
    """``CompileError.loc``, its source snippet and its call chain."""

    def test_an_astgen_error_marks_the_whole_expression(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_unresolved_name()
        error = ctx.exception
        self.assertIsNotNone(error.loc)
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_unresolved_name, 'return foo'))
        self.assertEqual(error.loc.end_column - error.loc.column, len('foo'))
        self.assertEqual(error.loc.function, 'diag_unresolved_name')

    def test_the_snippet_marks_the_whole_range(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_unresolved_name()
        text = str(ctx.exception)
        self.assertIn("error: name 'foo' is not defined", text)
        self.assertIn('^' * len('foo'), text)
        self.assertIn('note: in function `diag_unresolved_name`', text)

    def test_an_interp_error_points_at_the_statement(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_store(1)
        error = ctx.exception
        self.assertIsNotNone(error.loc)
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_store, 'y: u8 = x'))
        self.assertIn('cannot convert', error.message)

    def test_a_callee_error_records_the_call_site(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_caller(1)
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_callee, 'y: u8 = x'))
        self.assertEqual(error.loc.function, 'diag_callee')
        self.assertEqual(len(error.frames), 1)
        self.assertEqual(error.frames[0].kind, 'called from')
        self.assertEqual(
            error.frames[0].loc.line if error.frames[0].loc is not None else None,
            _line_of(diag_caller, 'diag_callee(x)'),
        )
        self.assertIn('note: called from `diag_caller', str(error))

    def test_a_closure_body_error_records_the_inline_call_site(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_closure_caller(1)
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_closure_caller, 'y: u8 = v'))
        self.assertEqual(error.loc.function, 'inner')
        self.assertGreaterEqual(len(error.frames), 1)
        self.assertEqual(
            error.frames[0].loc.line if error.frames[0].loc is not None else None,
            _line_of(diag_closure_caller, 'inner(x)'),
        )

    def test_a_parameter_annotation_error_points_at_the_annotation(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_bad_param(1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_bad_param, 'x: list'))
        self.assertEqual(error.loc.end_column - error.loc.column, len('list'))

    def test_a_return_annotation_error_points_at_the_annotation(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_bad_return()
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_bad_return, '-> list'))
        self.assertEqual(error.loc.end_column - error.loc.column, len('list'))

    def test_compile_error_points_at_its_call(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_compile_error()
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.message, 'boom')
        self.assertEqual(error.loc.line, _line_of(diag_compile_error, "compile_error('boom')"))

    def test_a_dangling_pragma_points_at_the_marker(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            diag_unused_pragma()
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(diag_unused_pragma, 'unroll()'))

    def test_a_struct_field_error_points_at_the_annotation(self) -> None:
        with self.assertRaises(CompileError) as ctx:
            GLOBAL_CONTEXT.resolve_global(DiagBadStruct)
        error = ctx.exception
        assert error.loc is not None
        self.assertEqual(error.loc.line, _line_of(DiagBadStructSource, 'x: list'))

    @skipUnless(_IPYTHON_AVAILABLE, 'IPython is not installed')
    def test_an_ipython_cell_path_is_rendered_as_in(self) -> None:
        from IPython.core.interactiveshell import InteractiveShell

        with warnings.catch_warnings():
            # IPython warns when ~/.ipython is not writable in the test sandbox
            warnings.simplefilter('ignore', UserWarning)
            shell = InteractiveShell.instance()
        shell.run_cell('def _diag_ipython_cell(x):\n    return x\n')
        fn = shell.user_ns['_diag_ipython_cell']
        path = inspect.getsourcefile(fn)
        assert path is not None

        cell = ipython_cell_name(path)
        self.assertIsNotNone(cell)
        assert cell is not None
        self.assertRegex(cell, r'^In\[\d+\]$')

        source_file = SourceFile(path, (), first_line=1)
        loc = SourceLoc(source_file, 1, 1, 1, 1)
        self.assertTrue(format_location(loc).startswith(f'{cell}:'))


all_tests = [
    SpyDiagnosticTest,
]

"""Source locations and diagnostic formatting.

``astgen`` parses the source of a Python function with ``inspect`` and the
``ast`` module.  This module turns the resulting ``ast`` nodes into
:class:`SourceLoc` values that point back into the *original* file - the
dedented text ``astgen`` parses is only an intermediate representation, so a
line number must be shifted by the first line ``inspect`` reported and a column
by the indentation ``textwrap.dedent`` removed.

Locations are carried through the compiler: every HIR instruction remembers the
node it was generated from (``hir.Inst.loc``), and ``CompileError`` stores the
location of the failure plus the call sites the error propagated through.  The
formatting helpers at the bottom of this module render one diagnostic, with the
source line(s) under the location and a caret spanning the whole range.
"""

import ast
import inspect
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class SourceFile:
    """The source text of one file a location points into.

    ``lines`` holds the original source split into lines (indentation kept),
    and ``dedent`` is the number of leading characters ``textwrap.dedent``
    removed from every line before the AST was parsed.  The AST's ``col_offset``
    is relative to the dedented line, so the original column is
    ``col_offset + dedent``.
    """

    path: str
    lines: tuple[str, ...] = field(repr=False)
    dedent: int = 0
    # the absolute line number of ``lines[0]``: ``inspect.getsourcelines``
    # returns a slice of the file, not the whole file
    first_line: int = 1

    def loc(self, node: ast.AST, function: str | None = None) -> SourceLoc:
        """The location of ``node``, with lines already made absolute by
        :func:`ast.increment_lineno` (see :func:`source_of_object`).

        A function/class definition only highlights its header - the ``def``/
        ``class`` keyword and the name - rather than its whole body, which would
        make an error about the declaration cover pages of unrelated code."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._header_loc(node, 'def', node.name, function)
        if isinstance(node, ast.ClassDef):
            return self._header_loc(node, 'class', node.name, function)
        line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', None) or line
        col = self._column(line, getattr(node, 'col_offset', 0) + self.dedent)
        end_col = self._column(
            end_line, (getattr(node, 'end_col_offset', None) or 0) + self.dedent,
        )
        return SourceLoc(self, line, col, end_line, end_col, function)

    def function_loc(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        function: str,
    ) -> SourceLoc:
        """The header location of a function definition or lambda."""
        if isinstance(node, ast.Lambda):
            return self.loc(node, function)
        return self._header_loc(node, 'def', node.name, function)

    def line_loc(self, line: int, function: str | None = None) -> SourceLoc:
        """A one-character location on ``line``, for failures with no node."""
        return SourceLoc(self, line, 1, line, 1, function)

    def _header_loc(
        self, node: ast.AST, keyword: str, name: str, function: str | None,
    ) -> SourceLoc:
        line = getattr(node, 'lineno', 1)
        col = self._column(line, getattr(node, 'col_offset', 0) + self.dedent)
        # ``def f`` / ``class C``: mark the keyword and the name
        end_col = col + len(keyword) + 1 + len(name)
        return SourceLoc(self, line, col, line, end_col, function)

    def line_text(self, line: int) -> str | None:
        """The text of absolute ``line``, or None outside this slice."""
        index = line - self.first_line
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return None

    def _column(self, line: int, byte_col: int) -> int:
        """A 1-based character column from a byte offset into ``line``."""
        text = self.line_text(line)
        if text is None:
            return byte_col + 1
        prefix = text.encode('utf-8')[:byte_col].decode('utf-8', errors='replace')
        return len(prefix) + 1


@dataclass(frozen=True, slots=True)
class SourceLoc:
    """One range in a source file.

    ``line``/``column`` are 1-based and point at the start of the range;
    ``end_line``/``end_column`` are 1-based and point *past* the end, the same
    convention as the ``ast`` module (``end_col_offset``).  ``function`` is the
    spy function the node belongs to, when there is one.
    """

    file: SourceFile
    line: int
    column: int
    end_line: int
    end_column: int
    function: str | None = None


@dataclass(frozen=True, slots=True)
class TraceFrame:
    """One note of a diagnostic's call chain."""

    kind: Literal['in', 'called from']
    function: str | None
    loc: SourceLoc | None


def _dedent_margin(text: str, dedented: str) -> int:
    """The number of leading characters ``textwrap.dedent`` removed."""
    for original, stripped in zip(text.splitlines(), dedented.splitlines()):
        if stripped.strip():
            return len(original) - len(stripped)
    return 0


def source_of_object(obj: Any) -> tuple[SourceFile, ast.Module] | None:
    """The original source text and AST of a Python function or class.

    ``None`` is returned when ``inspect`` cannot find the source (a dynamically
    defined object); callers turn that into their own error.  The AST's line
    numbers are absolute (``ast.increment_lineno`` shifted them by the first
    line ``inspect`` reported).
    """
    try:
        lines, start = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    text = ''.join(lines)
    dedented = textwrap.dedent(text)
    path = inspect.getsourcefile(obj) or _code_filename(obj) or '<unknown>'
    source = SourceFile(
        path, tuple(text.splitlines()), _dedent_margin(text, dedented), start,
    )
    tree = ast.parse(dedented, filename=path)
    ast.increment_lineno(tree, start - 1)
    return source, tree


def fallback_loc(obj: Any) -> SourceLoc | None:
    """A line-only location for an object whose source is unavailable."""
    code = getattr(obj, '__code__', None)
    if code is None:
        return None
    path = code.co_filename
    name = getattr(obj, '__name__', None)
    return SourceLoc(SourceFile(path, ()), code.co_firstlineno, 1, code.co_firstlineno, 1, name)


def _code_filename(obj: Any) -> str | None:
    code = getattr(obj, '__code__', None)
    return None if code is None else code.co_filename


def ipython_cell_name(path: str) -> str | None:
    """The ``In[N]`` label IPython knows ``path`` by, if any.

    A Jupyter cell is compiled from a temporary file whose name is a hash of
    the cell's code (``ipykernel.compiler.get_file_name``), so the number in
    the name is not the cell number.  ``CachingCompiler`` keeps the mapping
    from that filename to the execution count that compiled it; the same map
    is what IPython itself uses to render ``Cell In[N]`` in tracebacks.

    ``None`` outside IPython, for a real source file, or when the interpreter
    does not know the filename.  Importing IPython is lazy: the compiler has
    no hard dependency on it.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None:
        return None
    compiler = getattr(shell, 'compile', None)
    if compiler is None:
        return None
    formatter = getattr(compiler, 'format_code_name', None)
    if formatter is not None:
        try:
            formatted = formatter(path)
        except (AttributeError, TypeError):
            formatted = None
        if formatted is not None:
            return formatted[1]
    mapping = getattr(compiler, '_filename_map', None)
    if not mapping:
        return None
    number = mapping.get(path)
    return None if number is None else f'In[{number}]'


def format_location(loc: SourceLoc) -> str:
    path = ipython_cell_name(loc.file.path) or loc.file.path
    return f'{path}:{loc.line}:{loc.column}'


def format_diagnostic(
    message: str,
    loc: SourceLoc | None,
    frames: Sequence[TraceFrame] = (),
    *,
    snippet: bool = True,
) -> str:
    """Render one diagnostic: the primary location, its source snippet and the
    call chain the error propagated through."""
    if loc is None:
        out = [f'error: {message}']
    else:
        out = [f'{format_location(loc)}: error: {message}']
        if snippet:
            out.extend(_format_snippet(loc))
        if loc.function is not None:
            out.append(f'note: in function `{loc.function}`')
    for frame in frames:
        where = '' if frame.loc is None else f' at {format_location(frame.loc)}'
        if frame.kind == 'called from':
            if frame.function is None:
                out.append(f'note: called from{where}')
            else:
                out.append(f'note: called from `{frame.function}`{where}')
        else:
            if frame.function is None:
                out.append(f'note: in{where}')
            else:
                out.append(f'note: in function `{frame.function}`{where}')
    return '\n'.join(out)


def _format_snippet(loc: SourceLoc, max_lines: int = 8) -> list[str]:
    """The source lines a location covers, each followed by a caret run."""
    first = loc.file.first_line
    lines = loc.file.lines
    if not lines:
        return []
    start = max(first, loc.line)
    end = min(first + len(lines) - 1, max(loc.end_line, start))
    if start > first + len(lines) - 1:
        return []
    # a huge range (a whole block) would bury the message: show its first and
    # last lines only
    if end - start + 1 > max_lines:
        shown = list(range(start, start + max_lines // 2))
        shown.append(end)
    else:
        shown = list(range(start, end + 1))
    width = len(str(end))
    out: list[str] = []
    for number in shown:
        text = loc.file.line_text(number)
        assert text is not None
        out.append(f'  {number:>{width}} | {text}')
        out.append(f'  {"":>{width}} | {_caret(loc, number, text)}')
    return out


def _caret(loc: SourceLoc, line: int, text: str) -> str:
    """The caret run marking the part of ``line`` the location covers."""
    if line == loc.line:
        start = loc.column
    else:
        start = 1
    if line == loc.end_line:
        end = loc.end_column
    else:
        end = len(text) + 1
    # a zero-width range (a missing node end) still gets one caret
    length = max(1, end - start)
    return ' ' * (start - 1) + '^' * length

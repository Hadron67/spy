"""Exceptions raised by the spy package.

``CompileError`` signals that a function could not be JIT-compiled: an
unsupported construct, an operation that does not type-check, a type
mismatch discovered while "running" the HIR, etc.  It carries the source
location of the failure and the call sites the error propagated through (see
:mod:`spy.compiler.source`), which :meth:`CompileError.format` renders as one
diagnostic.  ``args[0]``/``message`` stay the bare message.

``TypeMismatchError`` signals that the Python values passed to a spy
function at the call boundary do not match the function signature (it
subclasses ``TypeError`` so that ordinary ``except TypeError`` code sees
it).
"""

from typing import Literal

from .source import SourceLoc, TraceFrame, format_diagnostic


class SpyError(Exception):
    pass


class CompileError(SpyError):
    """A compilation failure, optionally located in the source.

    ``loc`` is the primary location - the AST node or HIR instruction that
    failed.  ``frames`` holds the call sites the failure propagated through,
    innermost first, so that an error in a callee shows where it was called
    from.  ``attach_loc`` only fills an empty primary location, keeping the
    deepest (most specific) one; ``add_call_frame`` appends to the call chain.
    """

    message: str
    loc: SourceLoc | None
    frames: list[TraceFrame]

    def __init__(
        self,
        message: str,
        loc: SourceLoc | None = None,
        frames: list[TraceFrame] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.loc = loc
        self.frames = [] if frames is None else frames

    def attach_loc(self, loc: SourceLoc | None) -> None:
        """Fill the primary location when it is not known yet."""
        if self.loc is None and loc is not None:
            self.loc = loc

    def add_call_frame(
        self, function: str | None, loc: SourceLoc | None,
        kind: Literal['in', 'called from'] = 'called from',
    ) -> None:
        """Append one call-chain note, dropping an immediately repeated one."""
        if function is None and loc is None:
            return
        frame = TraceFrame(kind, function, loc)
        if self.frames and self.frames[-1] == frame:
            return
        self.frames.append(frame)

    def format(self, *, snippet: bool = True) -> str:
        """The full diagnostic: location, source snippet and call chain."""
        return format_diagnostic(self.message, self.loc, self.frames, snippet=snippet)

    def __str__(self) -> str:
        return self.format()


class CoerceError(CompileError):
    """A value could not be materialized as a requested spy type (see
    ``interp.HirRunner._coerce``).  It is the failure of one *coercion*, and
    callers that only want to know whether a coercion is possible catch it
    (``interp.HirRunner._try_coerce``); every other compile error keeps being a
    plain ``CompileError``."""


class TypeMismatchError(TypeError, SpyError):
    pass

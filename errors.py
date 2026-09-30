"""Exceptions raised by the spy package.

``CompileError`` signals that a function could not be JIT-compiled: an
unsupported construct, an operation that does not type-check, a type
mismatch discovered while "running" the HIR, etc.

``TypeMismatchError`` signals that the Python values passed to a spy
function at the call boundary do not match the function signature (it
subclasses ``TypeError`` so that ordinary ``except TypeError`` code sees
it).
"""


class SpyError(Exception):
    pass


class CompileError(SpyError):
    pass


class CoerceError(CompileError):
    """A value could not be materialized as a requested spy type (see
    ``interp.HirRunner._coerce``).  It is the failure of one *coercion*, and
    callers that only want to know whether a coercion is possible catch it
    (``interp.HirRunner._try_coerce``); every other compile error keeps being a
    plain ``CompileError``."""


class TypeMismatchError(TypeError, SpyError):
    pass

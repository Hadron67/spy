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


class TypeMismatchError(TypeError, SpyError):
    pass

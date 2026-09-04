import types as pytypes
from abc import abstractmethod
from typing import Any

from . import astgen, mir
from .fn import FunctionEntry
from .type import Type as SpyType
from .type import Value


class FunctionResolver:
    """The compile-time host of one :class:`HirRunner`.

    ``dsl.JitContext`` implements this interface; the interpreter cannot
    import the host directly (the host imports the interpreter), so the
    host inherits this class instead.  The interface exposes only what
    running a function body needs: the parsed HIR of callees and the
    resolution of native call targets (which may compile a callee on
    the fly).
    """
    @abstractmethod
    def hir_of_plain_fn(self, fn: pytypes.FunctionType) -> astgen.FunctionIR:
        """Parse (and cache) the HIR of a Python function, does not work on registered functions: use FunctionEntry.hir instead."""
        raise NotImplementedError

    @abstractmethod
    def resolve_call(self, entry: FunctionEntry, arg_types: tuple[SpyType, ...]) -> tuple[mir.Value, mir.Type]:
        """Resolve the callable value of one callee specialization from
        inside a compiled function: functions that are not compiled yet
        are compiled (MIR-wise) and defined in the module of the caller;
        functions of earlier modules are returned as symbols."""
        raise NotImplementedError

    @abstractmethod
    def resolve_global(self, value: Any) -> Value | None:
        """The spy value a global object referenced inside a function
        body resolves to.  A function registered in this host - reached
        as the raw function object or through the callable view its
        decorated name binds to - resolves to its function entry
        (creating the entry of an aot function that is not used yet);
        any other object is not a spy value of this host and returns
        ``None`` (the object stays a plain compile-time Python value)."""
        raise NotImplementedError

import types as pytypes
from abc import abstractmethod
from typing import Any

from . import astgen, mir, sval
from .fn import FunctionEntry


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
    def resolve_call(
        self, entry: FunctionEntry, arg_types: tuple[sval.Type, ...]
    ) -> tuple[mir.Value, sval.Type, sval.FunctionCallInfo]:
        """Resolve the callable value of one callee specialization from
        inside a compiled function: functions that are not compiled yet
        are compiled (MIR-wise) and defined in the module of the caller;
        functions of earlier modules are returned as symbols.

        Returns the callable value (a :class:`mir.Function` or a
        :class:`mir.Symbol`), the *logical spy return type* of the
        callee (the type its callers see, in ``type.py`` - all the
        interpreter's type checks happen on it) and the *call lowering
        plan* of the callee signature (a :class:`FunctionCallInfo`, see
        ``type.function_call_info``), which the interpreter follows when
        it emits the ``mir.Call``."""
        raise NotImplementedError

    @abstractmethod
    def resolve_global(self, value: Any) -> sval.Value | None:
        """The spy value a global object referenced inside a function
        body resolves to.  A function registered in this host - reached
        as the raw function object or through the callable view its
        decorated name binds to - resolves to its function entry
        (creating the entry of an aot function that is not used yet);
        any other object is not a spy value of this host and returns
        ``None`` (the object stays a plain compile-time Python value)."""
        raise NotImplementedError

    @abstractmethod
    def resolve_method(self, struct: sval.StructType, name: str) -> tuple[Any, bool] | None:
        """The method ``name`` of the struct type ``struct``, as the
        interpreter needs it for a method call ``x.name(...)``: a pair of
        the method - the entry of a registered ``@aot``/``@jit`` method,
        or the plain Python function of an undecorated method (inlined on
        call) - and whether its ``self`` is passed by pointer
        (``ptr_self``).  Returns None when the struct has no such method
        (a field of that name is read by ``astgen`` through ``FieldAddr``
        instead)."""
        raise NotImplementedError

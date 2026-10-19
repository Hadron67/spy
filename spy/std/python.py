"""Bindings to the CPython C API (``std.python``).

The module has two layers:

* the raw declarations at the bottom, one ``@decl_func()`` per C API function,
  keeping the C name and the C reference-counting rules;
* the ``PyObject`` methods and the ``GILState`` helpers above them, which are
  small ``inline=True`` wrappers over the raw calls.

The wrappers do **not** take the GIL: the JIT entry point calls native code
with the GIL released (``ctypes.CFUNCTYPE``), so a caller that wants to touch
the C API has to hold it itself:

.. code-block:: python

    state = GILState.acquire()
    with defer(): state.release()

    mod = PyObject.import_module("sys")
    with defer(): mod.decref()

    info = mod.getattr("version_info")
    with defer(): info.decref()

Registering the ``decref`` right after a new reference is created is the
intended ownership style.  A *borrowed* reference (``tuple_get``,
``list_get``, ``dict_get``, ``PyErr_Occurred``, ...) must not be decref'd, and
a *stolen* reference (``tuple_set``, ``list_set``) transfers its ownership to
the container even when the call fails.

``Option[Ptr[PyObject]]`` is the nullable ``PyObject*`` of the C API: the
pointer option uses the null pointer as its absent tag, so it crosses the C
boundary verbatim.
"""

from enum import Enum
from typing import Never

from ..compiler import (
    ConstMultiPtr,
    Opaque,
    Option,
    Ptr,
    c_int,
    c_long,
    comptime,
    decl_func,
    enum,
    f64,
    func,
    i32,
    i64,
    isize,
    ref,
    struct,
    u8,
    u64,
    usize,
)
from .core import ConstSlicePtr, compile_error, gstr
from .float import float_truncate
from .int import IntCastError, int_cast, try_int_cast
from .reflect import IntType, type_info

# ---------------------------------------------------------------------------
# types, constants and the GIL state
# ---------------------------------------------------------------------------


@struct()
class PyError(Exception):
    """A failed CPython C API call (or a failed conversion of a Python value).

    The operation that failed has already reported its own message (the
    wrappers call ``PyErr_Print`` before raising), so this exception carries
    no payload of its own.
    """


@enum(c_int)
class GILState(Enum):
    """The state ``PyGILState_Ensure`` returns, passed back to
    ``PyGILState_Release``.  ``acquire``/``release`` are the short spelling of
    the two C API calls."""

    LOCKED = 0
    UNLOCKED = 1

    @staticmethod
    @func(inline=True)
    def acquire() -> GILState:
        return PyGILState_Ensure()

    @func(inline=True)
    def release(self) -> None:
        PyGILState_Release(self)


@enum(c_int)
class PyConstant(Enum):
    """The constant ids ``Py_GetConstant``/``Py_GetConstantBorrowed`` take
    (``Py_CONSTANT_*`` in ``object.h``)."""

    NONE = 0
    FALSE = 1
    TRUE = 2
    ELLIPSIS = 3
    NOT_IMPLEMENTED = 4
    ZERO = 5
    ONE = 6
    EMPTY_STR = 7
    EMPTY_BYTES = 8
    EMPTY_TUPLE = 9


@enum(c_int)
class PyCompareOp(Enum):
    """The comparison operators ``PyObject_RichCompare*`` take
    (``Py_LT``..``Py_GE`` in ``object.h``)."""

    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5


@func(inline=True)
def raise_py_error() -> Never:
    """Raise ``PyError`` for the CPython error currently set (if any), after
    printing it.  The caller must hold the GIL."""
    if PyErr_Occurred() is not None:
        PyErr_Print()
    raise PyError()


# ---------------------------------------------------------------------------
# PyObject: reference counting and the singleton constants
# ---------------------------------------------------------------------------


@struct()
class PyObject:
    inner: Opaque

    @func(inline=True)
    def incref(self) -> None:
        Py_IncRef(ref(self))

    @func(inline=True)
    def decref(self) -> None:
        Py_DecRef(ref(self))

    @staticmethod
    @func(inline=True)
    def constant(constant: PyConstant) -> Ptr[PyObject]:
        """The borrowed reference to one of the CPython singleton constants."""
        if (p := Py_GetConstantBorrowed(constant)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def new_constant(constant: PyConstant) -> Ptr[PyObject]:
        """A new reference to one of the CPython singleton constants."""
        if (p := Py_GetConstant(constant)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def none() -> Ptr[PyObject]:
        return PyObject.constant(PyConstant.NONE)

    @staticmethod
    @func(inline=True)
    def new_none() -> Ptr[PyObject]:
        return PyObject.new_constant(PyConstant.NONE)

    @func(inline=True)
    def is_none(self) -> bool:
        return self.rich_compare_bool(PyObject.none(), PyCompareOp.EQ)

    # -- attributes ---------------------------------------------------------

    @func(inline=True)
    def getattr(self, name) -> Ptr[PyObject]:
        """The attribute ``name`` (a compile-time string) as a new
        reference."""
        if (p := PyObject_GetAttrString(ref(self), gstr(name))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def try_getattr(self, name) -> Option[Ptr[PyObject]]:
        """``getattr`` that returns the absent option instead of raising when
        the lookup fails.  The error indicator is cleared, so a missing
        attribute (or any other lookup error) reads as ``None``."""
        p = PyObject_GetAttrString(ref(self), gstr(name))
        if p is None:
            PyErr_Clear()
        return p

    @func(inline=True)
    def getattr_obj(self, name: Ptr[PyObject]) -> Ptr[PyObject]:
        """``getattr`` with a runtime ``str`` name (a new reference)."""
        if (p := PyObject_GetAttr(ref(self), name)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def setattr(self, name, value: Ptr[PyObject]) -> None:
        if PyObject_SetAttrString(ref(self), gstr(name), value) != 0:
            raise_py_error()

    @func(inline=True)
    def setattr_obj(self, name: Ptr[PyObject], value: Ptr[PyObject]) -> None:
        if PyObject_SetAttr(ref(self), name, value) != 0:
            raise_py_error()

    @func(inline=True)
    def hasattr(self, name) -> bool:
        return PyObject_HasAttrString(ref(self), gstr(name)) != 0

    @func(inline=True)
    def hasattr_obj(self, name: Ptr[PyObject]) -> bool:
        return PyObject_HasAttr(ref(self), name) != 0

    @func(inline=True)
    def delattr(self, name) -> None:
        if PyObject_DelAttrString(ref(self), gstr(name)) != 0:
            raise_py_error()

    @func(inline=True)
    def delattr_obj(self, name: Ptr[PyObject]) -> None:
        if PyObject_DelAttr(ref(self), name) != 0:
            raise_py_error()

    # -- calls --------------------------------------------------------------

    @func(inline=True)
    def call(self, args: Option[Ptr[PyObject]] = None) -> Ptr[PyObject]:
        """Call the object with the argument tuple ``args`` (``None`` for no
        arguments); the result is a new reference."""
        if (p := PyObject_CallObject(ref(self), args)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def call_no_args(self) -> Ptr[PyObject]:
        if (p := PyObject_CallNoArgs(ref(self))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def call_one_arg(self, arg: Ptr[PyObject]) -> Ptr[PyObject]:
        if (p := PyObject_CallOneArg(ref(self), arg)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def call_kw(self, args: Ptr[PyObject], kwargs: Option[Ptr[PyObject]] = None) -> Ptr[PyObject]:
        if (p := PyObject_Call(ref(self), args, kwargs)) is not None:
            return p
        raise_py_error()

    # -- subscripting -------------------------------------------------------

    @func(inline=True)
    def getitem(self, key: Ptr[PyObject]) -> Ptr[PyObject]:
        if (p := PyObject_GetItem(ref(self), key)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def setitem(self, key: Ptr[PyObject], value: Ptr[PyObject]) -> None:
        if PyObject_SetItem(ref(self), key, value) != 0:
            raise_py_error()

    @func(inline=True)
    def delitem(self, key: Ptr[PyObject]) -> None:
        if PyObject_DelItem(ref(self), key) != 0:
            raise_py_error()

    @func(inline=True)
    def length(self) -> usize:
        n = PyObject_Length(ref(self))
        if n < 0:
            raise_py_error()
        return int_cast(n, usize)

    # -- object protocol ----------------------------------------------------

    @func(inline=True)
    def is_true(self) -> bool:
        v = PyObject_IsTrue(ref(self))
        if v < 0:
            raise_py_error()
        return v != 0

    @func(inline=True)
    def to_str(self) -> Ptr[PyObject]:
        if (p := PyObject_Str(ref(self))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def to_repr(self) -> Ptr[PyObject]:
        if (p := PyObject_Repr(ref(self))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def get_iter(self) -> Ptr[PyObject]:
        if (p := PyObject_GetIter(ref(self))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def iter_next(self) -> Option[Ptr[PyObject]]:
        """The next item of the iterator (a new reference), or the absent
        option when the iterator is exhausted.  A real error is raised."""
        p = PyIter_Next(ref(self))
        if p is None and PyErr_Occurred() is not None:
            raise_py_error()
        return p

    @func(inline=True)
    def is_instance(self, cls: Ptr[PyObject]) -> bool:
        v = PyObject_IsInstance(ref(self), cls)
        if v < 0:
            raise_py_error()
        return v != 0

    @func(inline=True)
    def rich_compare(self, other: Ptr[PyObject], op: PyCompareOp) -> Ptr[PyObject]:
        """``self op other`` as a new reference (the result of the rich
        comparison)."""
        if (p := PyObject_RichCompare(ref(self), other, op)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def rich_compare_bool(self, other: Ptr[PyObject], op: PyCompareOp) -> bool:
        """Whether ``self op other`` holds (the result coerced to a bool)."""
        v = PyObject_RichCompareBool(ref(self), other, op)
        if v < 0:
            raise_py_error()
        return v != 0

    @func(inline=True)
    def is_callable(self) -> bool:
        return PyCallable_Check(ref(self)) != 0

    @func(inline=True)
    def type_obj(self) -> Ptr[PyObject]:
        if (p := PyObject_Type(ref(self))) is not None:
            return p
        raise_py_error()

    # -- conversions --------------------------------------------------------

    @staticmethod
    @func(inline=True)
    def from_i64(value: i64) -> Ptr[PyObject]:
        if (p := PyLong_FromLongLong(value)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def from_u64(value: u64) -> Ptr[PyObject]:
        if (p := PyLong_FromUnsignedLongLong(value)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def from_f64(value: f64) -> Ptr[PyObject]:
        if (p := PyFloat_FromDouble(value)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def from_bool(value: bool) -> Ptr[PyObject]:
        flag: i32 = 0
        if value:
            flag = 1
        if (p := PyBool_FromLong(flag)) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def from_str(value: ConstSlicePtr[u8]) -> Ptr[PyObject]:
        """A ``str`` from a UTF-8 slice (the bytes are copied)."""
        if (p := PyUnicode_FromStringAndSize(value.ptr, int_cast(value.length, isize))) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def from_bytes(value: ConstSlicePtr[u8]) -> Ptr[PyObject]:
        if (p := PyBytes_FromStringAndSize(value.ptr, int_cast(value.length, isize))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def as_i64(self) -> i64:
        v = PyLong_AsLongLong(ref(self))
        if v == -1 and PyErr_Occurred() is not None:
            raise_py_error()
        return v

    @func(inline=True)
    def as_u64(self) -> u64:
        v = PyLong_AsUnsignedLongLong(ref(self))
        if PyErr_Occurred() is not None:
            raise_py_error()
        return v

    @func(inline=True)
    def as_f64(self) -> f64:
        v = PyFloat_AsDouble(ref(self))
        if v == -1.0 and PyErr_Occurred() is not None:
            raise_py_error()
        return v

    @func(inline=True)
    def as_bool(self) -> bool:
        return self.is_true()

    @func(inline=True)
    def as_int[T: int](self, typ: type[T]) -> T:
        """``self`` as the integer type ``typ``, checking that the value fits
        the target's range."""
        comptime()
        info = type_info(typ)
        if isinstance(i := info, IntType):
            comptime()
            signed = i.signed
            if signed:
                try:
                    return try_int_cast(self.as_i64(), typ)
                except IntCastError:
                    raise_py_error()
            else:
                try:
                    return try_int_cast(self.as_u64(), typ)
                except IntCastError:
                    raise_py_error()
        compile_error("not an integer type")

    @func(inline=True)
    def as_float[T: float](self, typ: type[T]) -> T:
        return float_truncate(self.as_f64(), typ)

    @func(inline=True)
    def as_utf8(self) -> ConstSlicePtr[u8]:
        """The UTF-8 bytes of a ``str``, borrowed from the object (valid while
        the object is alive)."""
        n: isize = 0
        p = PyUnicode_AsUTF8AndSize(ref(self), ref(n))
        if (s := p) is not None:
            return ConstSlicePtr(s, int_cast(n, usize))
        raise_py_error()

    @func(inline=True)
    def as_bytes(self) -> ConstSlicePtr[u8]:
        """The bytes of a ``bytes`` object, borrowed from it (valid while the
        object is alive)."""
        p = PyBytes_AsString(ref(self))
        if (s := p) is not None:
            n = PyBytes_Size(ref(self))
            if n < 0:
                raise_py_error()
            return ConstSlicePtr(s, int_cast(n, usize))
        raise_py_error()

    # -- tuples -------------------------------------------------------------

    @staticmethod
    @func(inline=True)
    def new_tuple(size: usize) -> Ptr[PyObject]:
        if (p := PyTuple_New(int_cast(size, isize))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def tuple_size(self) -> usize:
        n = PyTuple_Size(ref(self))
        if n < 0:
            raise_py_error()
        return int_cast(n, usize)

    @func(inline=True)
    def tuple_get(self, index: usize) -> Ptr[PyObject]:
        """The borrowed reference to the item at ``index``."""
        if (p := PyTuple_GetItem(ref(self), int_cast(index, isize))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def tuple_set(self, index: usize, item: Ptr[PyObject]) -> None:
        """Store ``item`` at ``index``.  The C API **steals** the reference to
        ``item`` (also when the store fails), so the caller must not decref it
        afterwards."""
        if PyTuple_SetItem(ref(self), int_cast(index, isize), item) != 0:
            raise_py_error()

    # -- lists --------------------------------------------------------------

    @staticmethod
    @func(inline=True)
    def new_list(size: usize) -> Ptr[PyObject]:
        if (p := PyList_New(int_cast(size, isize))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def list_size(self) -> usize:
        n = PyList_Size(ref(self))
        if n < 0:
            raise_py_error()
        return int_cast(n, usize)

    @func(inline=True)
    def list_get(self, index: usize) -> Ptr[PyObject]:
        """The borrowed reference to the item at ``index``."""
        if (p := PyList_GetItem(ref(self), int_cast(index, isize))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def list_set(self, index: usize, item: Ptr[PyObject]) -> None:
        """Store ``item`` at ``index``.  The C API **steals** the reference to
        ``item`` (also when the store fails), so the caller must not decref it
        afterwards."""
        if PyList_SetItem(ref(self), int_cast(index, isize), item) != 0:
            raise_py_error()

    @func(inline=True)
    def list_append(self, item: Ptr[PyObject]) -> None:
        """Append ``item``; the reference is **not** stolen."""
        if PyList_Append(ref(self), item) != 0:
            raise_py_error()

    @func(inline=True)
    def list_insert(self, index: usize, item: Ptr[PyObject]) -> None:
        """Insert ``item`` before ``index``; the reference is **not** stolen."""
        if PyList_Insert(ref(self), int_cast(index, isize), item) != 0:
            raise_py_error()

    # -- dictionaries -------------------------------------------------------

    @staticmethod
    @func(inline=True)
    def new_dict() -> Ptr[PyObject]:
        if (p := PyDict_New()) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def dict_size(self) -> usize:
        n = PyDict_Size(ref(self))
        if n < 0:
            raise_py_error()
        return int_cast(n, usize)

    @func(inline=True)
    def dict_get(self, key: Ptr[PyObject]) -> Option[Ptr[PyObject]]:
        """The borrowed reference to the value of ``key``, or the absent
        option when the key is missing.  A real error is raised."""
        p = PyDict_GetItemWithError(ref(self), key)
        if p is None and PyErr_Occurred() is not None:
            raise_py_error()
        return p

    @func(inline=True)
    def dict_get_str(self, key) -> Option[Ptr[PyObject]]:
        """``dict_get`` with a compile-time string key."""
        p = PyDict_GetItemString(ref(self), gstr(key))
        if p is None and PyErr_Occurred() is not None:
            raise_py_error()
        return p

    @func(inline=True)
    def dict_set(self, key: Ptr[PyObject], value: Ptr[PyObject]) -> None:
        """Store ``value`` under ``key``; the reference is **not** stolen."""
        if PyDict_SetItem(ref(self), key, value) != 0:
            raise_py_error()

    @func(inline=True)
    def dict_set_str(self, key, value: Ptr[PyObject]) -> None:
        """Store ``value`` under the compile-time string ``key``; the
        reference is **not** stolen."""
        if PyDict_SetItemString(ref(self), gstr(key), value) != 0:
            raise_py_error()

    @func(inline=True)
    def dict_del(self, key: Ptr[PyObject]) -> None:
        if PyDict_DelItem(ref(self), key) != 0:
            raise_py_error()

    @func(inline=True)
    def dict_del_str(self, key) -> None:
        if PyDict_DelItemString(ref(self), gstr(key)) != 0:
            raise_py_error()

    # -- modules ------------------------------------------------------------

    @staticmethod
    @func(inline=True)
    def import_module(name) -> Ptr[PyObject]:
        """Import the module ``name`` (a compile-time string); the result is a
        new reference."""
        if (p := PyImport_ImportModule(gstr(name))) is not None:
            return p
        raise_py_error()

    @staticmethod
    @func(inline=True)
    def import_module_obj(name_obj: Ptr[PyObject]) -> Ptr[PyObject]:
        """Import the module named by the ``str`` object ``name_obj``; the
        result is a new reference."""
        if (p := PyImport_Import(name_obj)) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def module_dict(self) -> Ptr[PyObject]:
        """The borrowed reference to the module's ``__dict__``."""
        if (p := PyModule_GetDict(ref(self))) is not None:
            return p
        raise_py_error()

    @func(inline=True)
    def module_add_object_ref(self, name, value: Ptr[PyObject]) -> None:
        if PyModule_AddObjectRef(ref(self), gstr(name), value) != 0:
            raise_py_error()

    @func(inline=True)
    def module_add_int_constant(self, name, value: c_long) -> None:
        if PyModule_AddIntConstant(ref(self), gstr(name), value) != 0:
            raise_py_error()

    @func(inline=True)
    def module_add_string_constant(self, name, value) -> None:
        if PyModule_AddStringConstant(ref(self), gstr(name), gstr(value)) != 0:
            raise_py_error()


# ---------------------------------------------------------------------------
# raw CPython C API declarations (same names, same reference-counting rules)
# ---------------------------------------------------------------------------


@decl_func()
def PyGILState_Ensure() -> GILState: ...


@decl_func()
def PyGILState_Release(state: GILState) -> None: ...


@decl_func()
def Py_IncRef(obj: Ptr[PyObject]) -> None: ...


@decl_func()
def Py_DecRef(obj: Ptr[PyObject]) -> None: ...


@decl_func()
def Py_GetConstant(constant: PyConstant) -> Option[Ptr[PyObject]]: ...


@decl_func()
def Py_GetConstantBorrowed(constant: PyConstant) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyErr_Occurred() -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyErr_Clear() -> None: ...


@decl_func()
def PyErr_Print() -> None: ...


@decl_func()
def PyErr_GetRaisedException() -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyErr_SetRaisedException(exc: Ptr[PyObject]) -> None: ...


@decl_func()
def PyObject_GetAttrString(obj: Ptr[PyObject], name: ConstMultiPtr[u8]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_GetAttr(obj: Ptr[PyObject], name: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_SetAttrString(obj: Ptr[PyObject], name: ConstMultiPtr[u8], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_SetAttr(obj: Ptr[PyObject], name: Ptr[PyObject], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_HasAttrString(obj: Ptr[PyObject], name: ConstMultiPtr[u8]) -> i32: ...


@decl_func()
def PyObject_HasAttr(obj: Ptr[PyObject], name: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_DelAttrString(obj: Ptr[PyObject], name: ConstMultiPtr[u8]) -> i32: ...


@decl_func()
def PyObject_DelAttr(obj: Ptr[PyObject], name: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_CallObject(callable: Ptr[PyObject], args: Option[Ptr[PyObject]]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_CallNoArgs(callable: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_CallOneArg(callable: Ptr[PyObject], arg: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_Call(callable: Ptr[PyObject], args: Ptr[PyObject], kwargs: Option[Ptr[PyObject]]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_GetItem(obj: Ptr[PyObject], key: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_SetItem(obj: Ptr[PyObject], key: Ptr[PyObject], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_DelItem(obj: Ptr[PyObject], key: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_Length(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyObject_IsTrue(obj: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_Str(obj: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_Repr(obj: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_GetIter(obj: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyIter_Next(iter: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_IsInstance(obj: Ptr[PyObject], cls: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_RichCompare(obj1: Ptr[PyObject], obj2: Ptr[PyObject], op: PyCompareOp) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyObject_RichCompareBool(obj1: Ptr[PyObject], obj2: Ptr[PyObject], op: PyCompareOp) -> i32: ...


@decl_func()
def PyCallable_Check(obj: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyObject_Type(obj: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyLong_FromLongLong(value: i64) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyLong_FromUnsignedLongLong(value: u64) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyLong_AsLongLong(obj: Ptr[PyObject]) -> i64: ...


@decl_func()
def PyLong_AsUnsignedLongLong(obj: Ptr[PyObject]) -> u64: ...


@decl_func()
def PyLong_AsDouble(obj: Ptr[PyObject]) -> f64: ...


@decl_func()
def PyFloat_FromDouble(value: f64) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyFloat_AsDouble(obj: Ptr[PyObject]) -> f64: ...


@decl_func()
def PyBool_FromLong(value: i32) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyUnicode_FromStringAndSize(value: ConstMultiPtr[u8], size: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyUnicode_FromString(value: ConstMultiPtr[u8]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyUnicode_AsUTF8(obj: Ptr[PyObject]) -> Option[ConstMultiPtr[u8]]: ...


@decl_func()
def PyUnicode_AsUTF8AndSize(obj: Ptr[PyObject], size: Ptr[isize]) -> Option[ConstMultiPtr[u8]]: ...


@decl_func()
def PyUnicode_GetLength(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyBytes_FromStringAndSize(value: ConstMultiPtr[u8], size: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyBytes_AsString(obj: Ptr[PyObject]) -> Option[ConstMultiPtr[u8]]: ...


@decl_func()
def PyBytes_Size(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyTuple_New(size: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyTuple_Size(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyTuple_GetItem(obj: Ptr[PyObject], index: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyTuple_SetItem(obj: Ptr[PyObject], index: isize, item: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyList_New(size: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyList_Size(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyList_GetItem(obj: Ptr[PyObject], index: isize) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyList_SetItem(obj: Ptr[PyObject], index: isize, item: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyList_Append(obj: Ptr[PyObject], item: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyList_Insert(obj: Ptr[PyObject], index: isize, item: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyDict_New() -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyDict_Size(obj: Ptr[PyObject]) -> isize: ...


@decl_func()
def PyDict_GetItemWithError(obj: Ptr[PyObject], key: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyDict_GetItemString(obj: Ptr[PyObject], key: ConstMultiPtr[u8]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyDict_SetItem(obj: Ptr[PyObject], key: Ptr[PyObject], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyDict_SetItemString(obj: Ptr[PyObject], key: ConstMultiPtr[u8], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyDict_DelItem(obj: Ptr[PyObject], key: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyDict_DelItemString(obj: Ptr[PyObject], key: ConstMultiPtr[u8]) -> i32: ...


@decl_func()
def PyImport_ImportModule(name: ConstMultiPtr[u8]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyImport_Import(name: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyImport_AddModule(name: ConstMultiPtr[u8]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyImport_GetModuleDict() -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyModule_GetDict(module: Ptr[PyObject]) -> Option[Ptr[PyObject]]: ...


@decl_func()
def PyModule_AddObjectRef(module: Ptr[PyObject], name: ConstMultiPtr[u8], value: Ptr[PyObject]) -> i32: ...


@decl_func()
def PyModule_AddIntConstant(module: Ptr[PyObject], name: ConstMultiPtr[u8], value: c_long) -> i32: ...


@decl_func()
def PyModule_AddStringConstant(module: Ptr[PyObject], name: ConstMultiPtr[u8], value: ConstMultiPtr[u8]) -> i32: ...

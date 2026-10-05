"""The Python side of the native call boundary.

Everything that happens when Python calls a spy function or method, or
constructs a spy struct, lives here: binding the Python values to their spy
types, marshaling them into the calling convention the Python-entry thunk
exposes (``fn.FunctionInstance.thunk_call_sig``/``thunk_ret``), calling it
through ctypes, and turning the results back into Python values.

A spy value that lives on the Python side is one of:

* :class:`_StructInstance` - a spy struct value: a handle over its
  :class:`_StructInstanceData` (the struct type, the allocation that holds its
  bytes, kept alive by the value, and the offset of this value inside it - a
  nested field view shares its parent's allocation, so it writes through to the
  parent);
* :class:`_PtrInstance` - a non-null spy pointer (or a dynamically-sized
  function value, which is a pointer too);
* :class:`_ExceptionInstance` - the struct instance of a raised spy exception,
  which is also a Python ``Exception``.

An ``Option[T]`` and a tagged union are *unwrapped* at the boundary: a
``None`` or the value of the ``T``/variant itself.  ``spy.as_`` returns the
typed scalar constant itself (``sval.Int``/``sval.Float``/a plain bool).

This module does not import ``dsl``: the declaration handles there delegate to
the functions here.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from . import mir, sval
from .errors import CompileError, SpyError
from .fn import (
    ArgList,
    CallSignature,
    FunctionInstance,
    ProvidedArg,
    SpecializedRuntimeArg,
    plain_provided_arg,
)
from .lower import to_ctype
from .sval import (
    AnyValue,
    CompileContext,
    MirLowerCache,
    RetSpec,
    RetTuple,
    RetValue,
    SpecialTypeKind,
    StructType,
    StructTypeHead,
    Type,
    find_first_pointer_type_pos,
    iter_ret_leaves,
    type_of,
)
from .util import frozendict

# the bit width the untyped integer literal of a Python call is marshaled to
# (see ``sval.type_of``): an ``int`` argument becomes an ``i64``
_INT_LITERAL_BITS = 64


class CallableHandle(Protocol):
    """A callable spy handle - a registered function or method, or a plain
    Python function reached at the boundary.  Calling it runs the Python-side
    call (``dsl`` implements it)."""

    def __call__(self, *args: Any, **kwds: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Python-side spy values
# ---------------------------------------------------------------------------


def _address(owner: Any, offset: int) -> int:
    """The address ``offset`` bytes into the allocation ``owner``: a ctypes
    object (or a plain integer, for a value with no Python-side owner)."""
    base = owner if isinstance(owner, int) else ctypes.addressof(owner)
    return base + offset


class _StructInstanceData:
    """The data of a spy struct value on the Python side: the struct's spy type,
    the allocation the bytes of the value live in (``_owner``, kept alive by the
    value) and the offset of the value inside it (a nested field view shares its
    parent's allocation, so it writes through to the parent).  ``_cache`` is the
    MIR-mirror interning table the layout is computed from.

    The fields and methods of the value are reached through :meth:`get_attr` /
    :meth:`set_attr` rather than ``__getattr__``/``__setattr__``, so that the
    :class:`_StructInstance` wrapping it can implement attribute access without
    re-entering itself.  It is held by exactly one :class:`_StructInstance`."""

    __slots__ = ('_cache', '_offset', '_owner', 'type')

    def __init__(self, type: StructType, owner: Any, offset: int, cache: MirLowerCache) -> None:
        self.type = type
        self._owner = owner
        self._offset = offset
        self._cache = cache

    def _base(self) -> int:
        return _address(self._owner, self._offset)

    def get_attr(self, name: str, instance: _StructInstance) -> Any:
        """The value of the field ``name``, or the method ``name`` bound to
        ``instance``."""
        index = self.type.field_index(name)
        if index is not None:
            return _read_field(self, self.type, index, self._cache)
        method = self.type.get_method(name)
        if method is not None:
            return _BoundMethod(instance, method)
        raise AttributeError(f'{self.type} has no field or method {name!r}')

    def set_attr(self, name: str, value: Any) -> None:
        """Write ``value`` into the field ``name``."""
        index = self.type.field_index(name)
        if index is None:
            raise AttributeError(f'{self.type} has no field {name!r}')
        _write_field(self, self.type, index, value, self._cache)

    def __repr__(self) -> str:
        return f'{self.type}(...)'


class _StructInstance:
    """A spy struct value held on the Python side: a handle over its
    :class:`_StructInstanceData`.  ``p.x`` reads the field, ``p.x = v`` writes
    it, and ``p.m`` is the method ``m`` bound to this value.  Assigning a name
    that is not a field falls back to an ordinary attribute, which is what
    ``Exception``'s own attributes (:class:`_ExceptionInstance`) and ``_data``
    itself need.

    It deliberately has no ``__slots__``: :class:`_ExceptionInstance` derives
    from it *and* from ``Exception``, whose layout does not combine with a
    class that has slots of its own."""

    def __init__(self, type: StructType, owner: Any, offset: int, cache: MirLowerCache) -> None:
        object.__setattr__(self, '_data', _StructInstanceData(type, owner, offset, cache))

    def __getattr__(self, name: str) -> Any:
        # the data is read with ``object.__getattribute__``, so that a missing
        # ``_data`` cannot re-enter ``__getattr__``
        return object.__getattribute__(self, '_data').get_attr(name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        data = object.__getattribute__(self, '_data')
        if data.type.field_index(name) is None:
            # not a spy field: an ordinary attribute of the wrapper (the
            # ``Exception`` attributes of ``_ExceptionInstance``, ``_data``...)
            object.__setattr__(self, name, value)
            return
        data.set_attr(name, value)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, '_data'))


class _PtrInstance:
    """A non-null spy pointer (or a dynamically-sized function value) held on
    the Python side.  ``_owner`` is the allocation the address lives in (or a
    plain integer when the value points into memory Python does not own);
    ``_offset`` is the address within it."""

    __slots__ = ('_offset', '_owner', 'type')

    def __init__(self, type: Type, owner: Any, offset: int = 0) -> None:
        self.type = type
        self._owner = owner
        self._offset = offset

    def _base(self) -> int:
        return _address(self._owner, self._offset)

    def __repr__(self) -> str:
        return f'{self.type}({self._base():#x})'


class _ExceptionInstance(Exception, _StructInstance):
    """The struct instance of a raised spy exception, which is a Python
    ``Exception`` too (so ``except Exception`` catches it)."""

    __slots__ = ()

    def __init__(self, type: StructType, owner: Any, offset: int, cache: MirLowerCache) -> None:
        _StructInstance.__init__(self, type, owner, offset, cache)
        object.__setattr__(self, 'args', ())


class _BoundMethod:
    """A method of a struct instance: calling it passes the instance as the
    method's ``self`` (its first parameter, a ``Ptr[Self]``)."""

    __slots__ = ('_handle', '_self')

    def __init__(self, self_obj: _StructInstance, handle: Any) -> None:
        self._self = self_obj
        self._handle = handle

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return cast(CallableHandle, self._handle)(self._self, *args, **kwds)


# ---------------------------------------------------------------------------
# marshaling: spy values <-> the bytes at a boundary location
# ---------------------------------------------------------------------------


def _py(value: Any) -> Any:
    """The plain Python value a compile-time constant denotes (a typed
    ``sval.Int``/``Float`` constant, ``None``/null/void become ``None``)."""
    match value:
        case sval.Int() | sval.Float():
            return value.value
        case sval.Null() | sval.Void() | None:
            return None
        case _:
            return value


def _mir(type: Type, cache: MirLowerCache) -> mir.Type | None:
    return type.to_mir_type(cache)


def _scalar_ctype(type: Type, cache: MirLowerCache) -> Any:
    """The ctypes scalar type a leaf of the given spy type crosses the boundary
    as (a pointer or a dynamically-sized type crosses as a ``c_void_p``)."""
    if isinstance(type, sval.PointerType) or type.classify() == SpecialTypeKind.DST:
        return ctypes.c_void_p
    mir_type = _mir(type, cache)
    assert mir_type is not None, f'{type} has no runtime representation'
    return to_ctype(mir_type)


def _field_offset(type: StructType, index: int, cache: MirLowerCache) -> int:
    """The byte offset of the declaration-ordered field ``index`` in the memory
    of a value of ``type`` (only valid for a field with storage)."""
    if type.mirror_is_a_field(cache):
        return 0
    mir_type = type.get_mir_type(cache)
    assert isinstance(mir_type, mir.StructType)
    position = type.get_field_mir_indices(cache)[index]
    assert position is not None, 'a field with storage has a mirror position'
    name, _ = to_ctype(mir_type)._fields_[position]
    return cast(int, getattr(to_ctype(mir_type), name).offset)


def _child_offset(type: Type, index: int, cache: MirLowerCache) -> int:
    """The byte offset of the ``index``-th child (see ``get_type_children``)
    in the memory of a value of ``type``."""
    match type:
        case StructType():
            return _field_offset(type, index, cache)
        case sval.ArrayType():
            elem = type.elem.to_mir_type(cache)
            assert elem is not None
            return ctypes.sizeof(to_ctype(elem)) * index
        case sval.ComplexType():
            mir_type = type.to_mir_type(cache)
            assert isinstance(mir_type, mir.StructType)
            ctype = to_ctype(mir_type)
            name, _ = ctype._fields_[index]
            return cast(int, getattr(ctype, name).offset)
        case sval.OptionType():
            return 0
        case _:
            raise SpyError(f'cannot navigate the layout of {type}')


def _option_shape(
    type: sval.OptionType, cache: MirLowerCache,
) -> tuple[Literal['void', 'niche', 'struct'], tuple[int, ...]]:
    """How an option is represented (see ``sval.OptionType.to_mir_type``):
    ``'void'`` (a zero-sized child, only the tag is stored), ``'niche'`` (the
    child itself is the representation, a pointer at the given position is the
    tag) or ``'struct'`` (a ``(tag, value)`` struct)."""
    child = type.child
    if child.is_zst():
        return 'void', ()
    position = find_first_pointer_type_pos(child)
    if position is not None:
        return 'niche', position
    return 'struct', ()


def _read(type: Type, owner: Any, offset: int, cache: MirLowerCache) -> Any:
    """The Python value of the spy value of ``type`` stored ``offset`` bytes
    into ``owner``."""
    if type.is_zst():
        return None
    match type:
        case sval.OptionType():
            shape, position = _option_shape(type, cache)
            if shape == 'void':
                # a zero-sized child: the tag alone says whether a value is there,
                # and every value of the child equals its unit value (None)
                return None
            if shape == 'niche':
                if _read_pointer_at(type.child, position, owner, offset, cache) == 0:
                    return None
                return _read(type.child, owner, offset, cache)
            child_offset = _option_value_offset(type, cache)
            if not _bool_reader(type, owner, offset, cache):
                return None
            return _read(type.child, owner, offset + child_offset, cache)
        case sval.TaggedUnionType():
            return _read_tagged_union(type, owner, offset, cache)
        case StructType():
            return _StructInstance(type, owner, offset, cache)
        case sval.ArrayType():
            raise SpyError('an array cannot cross the Python boundary yet')
        case sval.ComplexType():
            # a complex number crosses as a Python ``complex``: read its real and
            # imaginary parts out of the two float fields of its mirror
            re = _read(type.elem, owner, offset + _child_offset(type, 0, cache), cache)
            im = _read(type.elem, owner, offset + _child_offset(type, 1, cache), cache)
            return complex(re, im)
        case _:
            value = _scalar_ctype(type, cache).from_address(_address(owner, offset)).value
            if isinstance(type, (sval.PointerType,)) or type.classify() == SpecialTypeKind.DST:
                return None if not value else _PtrInstance(type, value)
            return value


def _write(type: Type, value: Any, owner: Any, offset: int, cache: MirLowerCache) -> None:
    """Store the Python value ``value`` as a spy value of ``type`` ``offset``
    bytes into ``owner``."""
    if type.is_zst():
        return
    address = _address(owner, offset)
    match type:
        case sval.OptionType():
            _write_option(type, value, owner, offset, cache)
        case sval.TaggedUnionType():
            _write_tagged_union(type, value, owner, offset, cache)
        case StructType():
            source = _as_instance(value, type)
            mir_type = type.get_mir_type(cache)
            assert mir_type is not None
            ctypes.memmove(address, source._data._base(), ctypes.sizeof(to_ctype(mir_type)))
        case sval.ArrayType():
            raise SpyError('an array cannot cross the Python boundary yet')
        case sval.ComplexType():
            # a complex number crosses as a Python ``complex``; a real value
            # (float or integer) has a zero imaginary part
            value = _py(value)
            if isinstance(value, complex):
                re, im = value.real, value.imag
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpyError(f'cannot use {value!r} as a complex value')
            else:
                re, im = value, 0.0
            _write(type.elem, re, owner, offset + _child_offset(type, 0, cache), cache)
            _write(type.elem, im, owner, offset + _child_offset(type, 1, cache), cache)
        case _:
            pointer = isinstance(type, sval.PointerType) or type.classify() == SpecialTypeKind.DST
            if pointer:
                _scalar_ctype(type, cache).from_address(address).value = _as_address(value)
            else:
                _scalar_ctype(type, cache).from_address(address).value = _py(value)


def _as_address(value: Any) -> int | None:
    """The address a pointer-valued Python value carries (``None`` for the null
    pointer)."""
    match value:
        case _PtrInstance():
            return value._base()
        case _StructInstance():
            return value._data._base()
        case ctypes.c_void_p():
            return value.value
        case None:
            return None
        case int():
            return value
        case _:
            raise SpyError(f'cannot use {value!r} as a pointer')


def pointer_value(value: Any, type: Type) -> _PtrInstance:
    """A pointer-valued Python value (``spy.as_(address, Ptr[...])``): the raw
    address it carries, tied to its spy type."""
    address = _as_address(value)
    return _PtrInstance(type, 0 if address is None else address)


def _as_instance(value: Any, type: StructType) -> _StructInstance:
    if isinstance(value, _StructInstance):
        data = value._data
        if data.type is not type:
            raise SpyError(f'cannot use a {data.type} value as a {type} value')
        return value
    raise SpyError(f'cannot use {value!r} as a {type} value')


def _bool_reader(type: sval.OptionType, owner: Any, offset: int, cache: MirLowerCache) -> bool:
    mir_type = type.to_mir_type(cache)
    assert isinstance(mir_type, mir.StructType)
    name, _ = to_ctype(mir_type)._fields_[0]
    field_offset = cast(int, getattr(to_ctype(mir_type), name).offset)
    return cast(bool, ctypes.c_bool.from_address(_address(owner, offset + field_offset)).value)


def _option_value_offset(type: sval.OptionType, cache: MirLowerCache) -> int:
    mir_type = type.to_mir_type(cache)
    assert isinstance(mir_type, mir.StructType)
    name, _ = to_ctype(mir_type)._fields_[1]
    return cast(int, getattr(to_ctype(mir_type), name).offset)


def _read_pointer_at(type: Type, position: tuple[int, ...], owner: Any, offset: int, cache: MirLowerCache) -> int:
    for index in position:
        offset += _child_offset(type, index, cache)
        type = type.get_type_children()[index]
    assert isinstance(type, sval.PointerType)
    return cast(int, ctypes.c_void_p.from_address(_address(owner, offset)).value or 0)


def _write_pointer_at(type: Type, position: tuple[int, ...], owner: Any, offset: int, cache: MirLowerCache, value: int | None) -> None:
    for index in position:
        offset += _child_offset(type, index, cache)
        type = type.get_type_children()[index]
    ctypes.c_void_p.from_address(_address(owner, offset)).value = value


def _read_field(instance: _StructInstanceData, type: StructType, index: int, cache: MirLowerCache) -> Any:
    field = type.fields().get_by_id(index)
    if field.type.is_zst():
        return None
    return _read(field.type, instance._owner, instance._offset + _field_offset(type, index, cache), cache)


def _write_field(instance: _StructInstanceData, type: StructType, index: int, value: Any, cache: MirLowerCache) -> None:
    field = type.fields().get_by_id(index)
    if field.type.is_zst():
        return
    _write(field.type, value, instance._owner, instance._offset + _field_offset(type, index, cache), cache)


def _write_option(type: sval.OptionType, value: Any, owner: Any, offset: int, cache: MirLowerCache) -> None:
    shape, position = _option_shape(type, cache)
    value = _py(value)
    if shape == 'void':
        ctypes.c_bool.from_address(_address(owner, offset)).value = value is not None
        return
    if shape == 'niche':
        if value is None:
            _write_pointer_at(type.child, position, owner, offset, cache, None)
            return
        _write(type.child, value, owner, offset, cache)
        return
    child_offset = _option_value_offset(type, cache)
    ctypes.c_bool.from_address(_address(owner, offset)).value = value is not None
    if value is not None:
        _write(type.child, value, owner, offset + child_offset, cache)


def _payload_offset(type: sval.TaggedUnionType, cache: MirLowerCache) -> int:
    mir_type = type.to_mir_type(cache)
    assert isinstance(mir_type, mir.StructType)
    name, _ = to_ctype(mir_type)._fields_[1]
    return cast(int, getattr(to_ctype(mir_type), name).offset)


def _read_tagged_union(type: sval.TaggedUnionType, owner: Any, offset: int, cache: MirLowerCache) -> Any:
    shape = sval.tagged_union_shape(type, cache)
    if shape == 'single':
        return _read(type.types[0], owner, offset, cache)
    tag_mir = type.tag_type().to_mir_type(cache)
    assert isinstance(tag_mir, mir.IntType)
    tag = to_ctype(tag_mir).from_address(_address(owner, offset)).value
    if shape == 'tag_only':
        return None
    variant = type.types[tag]
    return _read(variant, owner, offset + _payload_offset(type, cache), cache)


def _write_tagged_union(type: sval.TaggedUnionType, value: Any, owner: Any, offset: int, cache: MirLowerCache) -> None:
    shape = sval.tagged_union_shape(type, cache)
    if shape == 'single':
        _write(type.types[0], value, owner, offset, cache)
        return
    if isinstance(value, _StructInstance):
        variant_type: Type = value._data.type
    else:
        variant_type = type_of(_py(value))
    index = type.variant_index_for(variant_type)
    if index is None:
        raise SpyError(f'{value!r} is not a variant of {type}')
    tag_mir = type.tag_type().to_mir_type(cache)
    assert isinstance(tag_mir, mir.IntType)
    to_ctype(tag_mir).from_address(_address(owner, offset)).value = index
    if shape == 'tag_payload':
        _write(type.types[index], value, owner, offset + _payload_offset(type, cache), cache)


# ---------------------------------------------------------------------------
# the argument list of one Python-side call
# ---------------------------------------------------------------------------


def arg_spy_type(value: Any) -> Type | None:
    """The spy type of one Python argument (``None`` for a value with no spy
    type)."""
    if isinstance(value, _StructInstance):
        return value._data.type
    if isinstance(value, _PtrInstance):
        return value.type
    return type_of(value, _INT_LITERAL_BITS)


def boundary_arg(value: Any, context: CompileContext) -> Any:
    """One Python argument of a call, in the spy domain: a Python-side spy
    value passes through, anything else is resolved by ``sval.as_value``."""
    if isinstance(value, (_StructInstance, _PtrInstance)):
        return value
    return sval.as_value(value, context)


def provided_arglist(sig: Any, arglist: ArgList[Any]) -> ArgList[ProvidedArg]:
    """The ``provided`` argument list of one Python-boundary call (see
    ``Signature.solve_param_types``)."""
    positional: list[ProvidedArg] = []
    for (name, param), value in zip(sig.positional.items(), arglist.positional):
        type = arg_spy_type(value)
        if param.is_type_value:
            if not isinstance(value, Type):
                raise CompileError(
                    f"the argument of parameter '{name}' must be a spy type"
                )
            positional.append((type, value))
        else:
            positional.append(plain_provided_arg(type))
    varargs = tuple(plain_provided_arg(arg_spy_type(v)) for v in arglist.varargs)
    kwargs = frozendict(
        (k, plain_provided_arg(arg_spy_type(v))) for k, v in arglist.kwargs.items()
    )
    return ArgList(tuple(positional), varargs, kwargs)


# ---------------------------------------------------------------------------
# calling a compiled function
# ---------------------------------------------------------------------------


def invoke(
    instance: FunctionInstance,
    call_sig: CallSignature,
    arglist: ArgList[Any],
    context: CompileContext,
) -> Any:
    """Call the thunk of the compiled specialization ``instance`` from Python:
    marshal the bound arguments, allocate an out pointer per result leaf, call
    the native thunk, and turn the results back into Python values (raising a
    spy exception when the error code says so)."""
    cache = context.mir_cache()
    native = instance.wrapper_fn
    thunk_sig = instance.thunk_call_sig
    thunk_ret = instance.thunk_ret
    assert native is not None and thunk_sig is not None and thunk_ret is not None

    temps: list[Any] = []
    py_args: list[Any] = []
    for (_name, arg), (_tname, targ), value in zip(
        call_sig.positional, thunk_sig.positional, arglist.positional,
    ):
        if not isinstance(targ, SpecializedRuntimeArg):
            continue
        argument, buffer = _marshal_arg(targ.type, value, targ.is_ref, cache)
        py_args.append(argument)
        if buffer is not None:
            temps.append(buffer)

    leaves = _storage_leaves(thunk_ret)
    buffers: list[Any] = []
    for leaf in leaves:
        mir_type = leaf.type.to_mir_type(cache)
        assert mir_type is not None
        buffer = to_ctype(mir_type)()
        buffers.append(buffer)
        py_args.append(ctypes.c_void_p(ctypes.addressof(buffer)))

    native.call(*py_args)
    return _read_result(instance, buffers, context)


def _marshal_arg(type: Type, value: Any, is_ref: bool, cache: MirLowerCache) -> tuple[Any, Any]:
    """The native argument and (when the argument has to stay alive until the
    call) the buffer that holds it.  A pointer-like value (a pointer, or a
    dynamically-sized function value) crosses as the address it carries; an
    aggregate the thunk takes by pointer (``is_ref``) crosses as the address of
    a buffer holding it; anything else crosses by value."""
    if isinstance(type, sval.PointerType) or type.classify() == SpecialTypeKind.DST:
        return ctypes.c_void_p(_as_address(value)), None
    if is_ref:
        mir_type = type.to_mir_type(cache)
        assert mir_type is not None, f'{type} has no runtime representation'
        buffer = to_ctype(mir_type)()
        _write(type, value, buffer, 0, cache)
        return ctypes.c_void_p(ctypes.addressof(buffer)), buffer
    # a scalar crosses by value: build it in a temporary buffer and read it out
    ctype = _scalar_ctype(type, cache)
    buffer = ctype()
    _write(type, value, buffer, 0, cache)
    return buffer.value, None


def _storage_leaves(spec: RetSpec) -> list[RetValue]:
    return [leaf for leaf in iter_ret_leaves(spec) if leaf.type.get_unit_value() is None]


def _read_result(
    instance: FunctionInstance,
    buffers: list[Any],
    context: CompileContext,
) -> Any:
    """Turn the out pointers of a finished call back into Python values,
    raising the spy exception the error code names."""
    cache = context.mir_cache()
    ret_sig = instance.ret_sig
    assert ret_sig is not None
    value_spec = ret_sig.ret_type_spec
    exceptions = list(ret_sig.exceptions.values)

    consumed = 0

    def read(spec: RetSpec) -> Any:
        nonlocal consumed
        match spec:
            case RetValue():
                if spec.type.get_unit_value() is not None:
                    return None
                buffer = buffers[consumed]
                consumed += 1
                return _read(spec.type, buffer, 0, cache)
            case RetTuple():
                return tuple(read(value) for value in spec.values)

    # the value part first, then the error code and the payload
    if not _value_is_empty(value_spec):
        value = read(value_spec)
    else:
        value = None

    if len(exceptions) > 0:
        result_type = ret_sig.result_type()
        code_type = result_type.code_type
        code = 0
        if code_type.get_unit_value() is None:
            code = _read(code_type, buffers[consumed], 0, cache)
            consumed += 1
        if _value_is_empty(value_spec):
            index = code
        else:
            index = code - 1
        if index >= 0 and index < len(exceptions):
            payload = _read_error_payload(exceptions[index], buffers[consumed], cache)
            raise payload
    return value


def _value_is_empty(spec: RetSpec | None) -> bool:
    return isinstance(spec, RetValue) and isinstance(spec.type, sval.EmptyType)


def _read_error_payload(exception: Type, buffer: Any, cache: MirLowerCache) -> _ExceptionInstance:
    assert isinstance(exception, StructType), 'an exception is a spy struct'
    return _ExceptionInstance(exception, buffer, 0, cache)


# ---------------------------------------------------------------------------
# class-name access: ``Foo[i32]`` and ``Foo.m``
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpecializedStruct(sval.StructTypeApplication):
    """The Python-level ``Foo[i32]`` of a struct template: like a
    ``sval.StructTypeApplication`` (so ``sval.as_value`` still resolves it in an
    annotation scope), plus the resolving context, and *callable* - a
    construction (``Foo[i32](...)``)."""

    context: CompileContext

    def _type(self) -> StructType:
        type = sval.as_value(self, self.context)
        assert isinstance(type, StructType)
        return type

    def __call__(self, *args: Any, **kwds: Any) -> _StructInstance:
        type = self._type()
        return construct(type.head, type.generic_args, args, kwds, self.context)

    def __getattr__(self, name: str) -> Any:
        """``Foo[i32].m``: the method ``m`` of this specialization, called with
        no implicit ``self``."""
        if name.startswith('_'):
            raise AttributeError(name)
        method = self._type().get_method(name)
        if method is None:
            raise AttributeError(f'{self._type()} has no method {name!r}')
        return method


def specialized_struct(struct: Any, key: Any, context: CompileContext) -> SpecializedStruct:
    args = key if isinstance(key, tuple) else (key,)
    return SpecializedStruct(struct, args, context)


# ---------------------------------------------------------------------------
# constructing a struct
# ---------------------------------------------------------------------------


def construct(
    head: StructTypeHead,
    generic_args: tuple[Any, ...] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    context: CompileContext,
) -> _StructInstance:
    """Construct a spy struct from Python: ``Foo(...)`` (``generic_args`` is
    ``None``, the arguments of the specialization are inferred from the
    provided field values) or ``Foo[...](...)`` (``generic_args`` are the
    explicit ones)."""
    cache = context.mir_cache()
    type = _resolve_struct(head, generic_args, args, kwargs)
    mir_type = type.get_mir_type(cache)
    if mir_type is None:
        # a zero-sized struct has no storage
        return _StructInstance(type, 0, 0, cache)
    buffer = to_ctype(mir_type)()

    fields = type.fields()
    provided: dict[int, Any] = {}
    if len(args) > len(fields.by_id):
        raise TypeError(
            f'{head.name_base} takes {len(fields.by_id)} field(s) but {len(args)} were given'
        )
    for index, value in enumerate(args):
        provided[index] = value
    for name, value in kwargs.items():
        index = fields.by_key.get(name)
        if index is None:
            raise TypeError(f'{head.name_base} has no field {name!r}')
        if index in provided:
            raise TypeError(f"got multiple values for field '{name}'")
        provided[index] = value

    for index, field in enumerate(fields.values()):
        if field.type.is_zst():
            continue
        offset = _field_offset(type, index, cache)
        if index in provided:
            _write(field.type, boundary_arg(provided[index], context), buffer, offset, cache)
        elif field.default is not None:
            _write(field.type, field.default, buffer, offset, cache)
        else:
            raise TypeError(
                f'missing a value for field {field.name!r} of {head.name_base}'
            )
    return _StructInstance(type, buffer, 0, cache)


def _resolve_struct(
    head: StructTypeHead,
    generic_args: tuple[Any, ...] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> StructType:
    """The struct specialization a Python construction builds: the explicit
    arguments, the head's only specialization when it is not generic, or the
    one the provided field values determine."""
    if generic_args is not None:
        return head.specialize(generic_args)
    if len(head.generic_args) == 0:
        return head.specialize(())
    solver = sval.TypeVarSolver()
    fields = head.fields
    provided: dict[int, Any] = {}
    for index, value in enumerate(args):
        provided[index] = value
    for name, value in kwargs.items():
        index = fields.by_key.get(name)
        if index is not None:
            provided[index] = value
    for index, value in provided.items():
        if index < 0 or index >= len(fields.by_id):
            continue
        value_type = arg_spy_type(value)
        if value_type is not None:
            solver.add_constraint(value_type, fields.get_by_id(index).type, True)
    solver.finish()
    solved = solver.get_solved()
    resolved: list[AnyValue] = []
    for type_var in head.generic_args:
        if type_var not in solved:
            raise CompileError(
                f'cannot infer the generic argument {type_var.name} of struct '
                f'{head.name_base} from this construction; give it explicitly, '
                f'e.g. {head.name_base}[...](...)'
            )
        resolved.append(solved[type_var])
    return head.specialize(tuple(resolved))

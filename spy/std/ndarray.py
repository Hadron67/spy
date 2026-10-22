import ctypes
from typing import TYPE_CHECKING, Any, cast

from ..compiler import (
    Array,
    InferComptime,
    MultiPtr,
    Option,
    Ptr,
    c64,
    c128,
    comptime,
    f32,
    f64,
    func,
    i8,
    i16,
    i32,
    i64,
    struct,
    sval,
    typeof,
    u8,
    u16,
    u32,
    u64,
    unroll,
    usize,
)
from ..compiler import (
    bool as spy_bool,
)
from .core import compile_error, const_array, panic, sstr, unwrap_or
from .core import slice as SpySlice


@struct()
class NDArray[T, N: int]:
    """A multi-dimensional view of an array of ``T``.

    ``ptr`` points at the first element, ``size`` holds the length of every
    dimension and ``stride`` the distance (in elements) between two neighbours
    along each dimension.  The view owns no storage: slicing returns another
    ``NDArray`` over the same elements.
    """

    ptr: MultiPtr[T]
    size: Array[usize, N]
    stride: Array[usize, N]

    @func(inline=True)
    def __spy_getitemptr__(self, indices) -> Ptr[T]:
        """``a[i, j, ...]``: the place of one element.

        A subscript must name **every** dimension - spy's subscript always
        yields one element, so there is no numpy-style partial indexing here
        (use :meth:`slice` for that).  Every index has to be a ``usize``.
        """
        if N == 1:
            # ``a[i]`` passes the index itself, not a tuple (spy only builds a
            # tuple from a subscript with several indices)
            if indices >= self.size[0]:
                panic(sstr('index out of bounds'))
            return self.ptr + indices * self.stride[0]
        else:
            if len(indices) != N:
                compile_error('an NDArray subscript has to give one index per dimension')
            offset: usize = 0
            unroll()
            for i in range(cast(int, N)):
                index: InferComptime = indices[i]
                if index >= self.size[i]:
                    panic(sstr('index out of bounds'))
                offset += index * self.stride[i]
            return self.ptr + offset

    @func(inline=True)
    def slice(self, *slices) -> NDArray[T, N]:
        """A view of the elements the per-dimension ``slices`` name.

        Every dimension takes one argument: a ``usize`` selects a single index
        of that dimension (the result keeps the dimension with length 1), and a
        ``std.slice`` object - the value ``a:b`` builds, or one made with
        :func:`span` - names a range.  A slice bound left out is 0 for the
        start and the dimension's length for the end; a step other than 1 is
        rejected.
        """
        if len(slices) != N:
            compile_error('NDArray.slice takes one argument per dimension')
        offset: usize = 0
        new_size: Array[usize, N] = cast(Array[usize, N], const_array(0, cast(int, N)))
        unroll()
        for i in range(cast(int, N)):
            s: InferComptime = slices[i]
            comptime()
            is_slice = typeof(s) == SpySlice[usize]
            if is_slice:
                begin = unwrap_or(s.start, 0)
                end = unwrap_or(s.end, self.size[i])
                step = unwrap_or(s.step, 1)
                if step != 1:
                    panic(sstr('NDArray.slice does not support a step'))
                if begin > end or end > self.size[i]:
                    panic(sstr('slice out of bounds'))
                offset += begin * self.stride[i]
                new_size[i] = end - begin
            else:
                if s >= self.size[i]:
                    panic(sstr('index out of bounds'))
                offset += s * self.stride[i]
                new_size[i] = 1
        return NDArray[T, N](self.ptr + offset, new_size, self.stride)

    if TYPE_CHECKING:
        # only so that the Python type checker accepts the ``a[i, j]`` spellings:
        # the subscript goes through ``__spy_getitemptr__`` at compile time
        def __getitem__(self, index: int | tuple[int, ...]) -> T: ...
        def __setitem__(self, index: int | tuple[int, ...], value: T) -> None: ...


@func(inline=True)
def span(start: Option[usize], end: Option[usize] = None) -> SpySlice[usize]:
    """The ``std.slice`` object of a range - the value ``a:b`` builds.

    ``span(0, 2)`` is ``0:2``, ``span(None, 2)`` is ``:2`` and
    ``span(None, None)`` is the whole dimension.
    """
    return SpySlice(start, end)


# The spy scalar type of each supported numpy dtype, keyed by ``(kind, itemsize)``.
_DTYPE_TYPES: dict[tuple[str, int], sval.Type] = {
    ('b', 1): cast(sval.Type, spy_bool),
    ('i', 1): cast(sval.Type, i8),
    ('i', 2): cast(sval.Type, i16),
    ('i', 4): cast(sval.Type, i32),
    ('i', 8): cast(sval.Type, i64),
    ('u', 1): cast(sval.Type, u8),
    ('u', 2): cast(sval.Type, u16),
    ('u', 4): cast(sval.Type, u32),
    ('u', 8): cast(sval.Type, u64),
    ('f', 4): cast(sval.Type, f32),
    ('f', 8): cast(sval.Type, f64),
    ('c', 8): cast(sval.Type, c64),
    ('c', 16): cast(sval.Type, c128),
}


def _dtype_elem_type(dtype: Any) -> sval.Type:
    """The spy scalar type a numpy dtype holds."""
    if not getattr(dtype, 'isnative', True):
        raise TypeError(f'a non-native-endian numpy dtype is not supported: {dtype!r}')
    spy_type = _DTYPE_TYPES.get((dtype.kind, dtype.itemsize))
    if spy_type is None:
        raise TypeError(f'a numpy dtype of {dtype!r} has no spy element type')
    return spy_type


def from_ndarray(arr: Any, context: sval.CompileContext | None = None) -> Any:
    """A spy ``NDArray`` view over the numpy array ``arr`` - a zero-copy
    conversion, usable from Python (for tests and host code).  The result is a
    ``glue._StructInstance`` of the matching ``NDArray[T, N]`` specialization.

    ``T`` follows the dtype, ``N`` is ``arr.ndim`` and the strides are turned
    from bytes into element counts.  Negative strides are rejected.  The numpy
    array is kept alive by the returned value (as an ordinary Python attribute,
    never part of the spy value); a value the spy code hands back is a fresh
    wrapper without that reference, so the caller has to keep ``arr`` alive
    itself in that case.
    """
    from ..compiler import GLOBAL_CONTEXT, glue

    if context is None:
        context = GLOBAL_CONTEXT
    elem_type = _dtype_elem_type(arr.dtype)
    itemsize: int = arr.dtype.itemsize
    strides: list[int] = []
    for stride in arr.strides:
        if stride < 0:
            raise ValueError('a numpy array with a negative stride is not supported')
        if stride % itemsize != 0:
            raise ValueError(f'the stride {stride} is not a multiple of the itemsize {itemsize}')
        strides.append(stride // itemsize)
    head = context.resolve_global(NDArray)
    assert isinstance(head, sval.StructTypeHead), 'NDArray is not a struct template'
    nd = glue.construct(
        head,
        (elem_type, len(arr.shape)),
        (None, tuple(arr.shape), tuple(strides)),
        {},
        context,
    )
    nd.ptr = ctypes.c_void_p(arr.ctypes.data)
    nd._keepalive = arr
    return nd

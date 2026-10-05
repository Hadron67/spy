from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    u8,
    usize,
)
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
    ConstMultiPtr,
)
from ..std import (
    ConstSlicePtr,
    gstr,
    sstr,
)

# ---------------------------------------------------------------------------
# byte strings: a ``bytes`` literal/value exists only at compile time (it has no
# runtime representation), a string literal is encoded to ``bytes`` at parse
# time, and a byte string may be subscripted / sliced at compile time.  The
# ``std.core.gstr``/``std.core.sstr`` builtins turn one into a runtime
# ``ConstMultiPtr[u8]``/``ConstSlicePtr[u8]``, and ``ord`` yields the encoding of
# a one-byte byte string.
# ---------------------------------------------------------------------------


@func()
def bytes_index() -> i32:
    s: Comptime = b'abc'
    return ord(s[0]) * 100 + ord(s[2])


@func()
def bytes_slice_index() -> i32:
    s: Comptime = b'hello'
    t: Comptime = s[1:4]
    return ord(t[0]) * 100 + ord(t[2])


@func()
def bytes_omitted_bounds() -> i32:
    s: Comptime = b'abcde'
    head: Comptime = s[:2]
    tail: Comptime = s[2:]
    return ord(head[0]) * 100 + ord(head[1]) + ord(tail[0]) * 10000 + ord(tail[2]) * 100000


@func()
def ord_of_literal() -> i32:
    return ord(b'a')


@func()
def str_literal_is_encoded() -> i32:
    # a ``str`` literal is encoded to ``bytes`` at parse time
    s: Comptime = 'A'
    return ord(s[0])


@func()
def gstr_read() -> i32:
    s: Comptime = b'AB'
    p: ConstMultiPtr[u8] = gstr(s)
    return p[1]


@func()
def sstr_read() -> i32:
    s: Comptime = b'XY'
    sl: ConstSlicePtr[u8] = sstr(s)
    return sl[1]


@func()
def sstr_length() -> usize:
    s: Comptime = b'abcd'
    sl: ConstSlicePtr[u8] = sstr(s)
    return sl.length


@func()
def gstr_twice() -> i32:
    # both calls name the same bytes: they share one global constant
    s: Comptime = b'dup'
    a: ConstMultiPtr[u8] = gstr(s)
    b: ConstMultiPtr[u8] = gstr(s)
    return a[0] + b[1]


@func()
def ord_of_two_bytes() -> i32:
    return ord(b'ab')


@func()
def ord_of_non_bytes() -> i32:
    return ord(1)  # pyright: ignore


@func()
def bytes_index_out_of_bounds() -> i32:
    s: Comptime = b'a'
    return ord(s[1])


@func()
def bytes_slice_out_of_bounds() -> i32:
    s: Comptime = b'abc'
    t: Comptime = s[1:9]
    return ord(t[0])


class SpyBytesTest(TestCase):
    """Compile-time byte strings and the ``gstr``/``sstr``/``ord`` builtins."""

    def test_a_byte_string_can_be_indexed(self) -> None:
        self.assertEqual(bytes_index(), ord('a') * 100 + ord('c'))

    def test_a_byte_string_can_be_sliced(self) -> None:
        self.assertEqual(bytes_slice_index(), ord('e') * 100 + ord('l'))

    def test_a_slice_bound_may_be_left_out(self) -> None:
        expected = ord('a') * 100 + ord('b') + ord('c') * 10000 + ord('e') * 100000
        self.assertEqual(bytes_omitted_bounds(), expected)

    def test_ord_yields_the_byte(self) -> None:
        self.assertEqual(ord_of_literal(), ord('a'))

    def test_a_string_literal_is_encoded_to_bytes(self) -> None:
        self.assertEqual(str_literal_is_encoded(), ord('A'))

    def test_gstr_yields_a_pointer_to_the_bytes(self) -> None:
        self.assertEqual(gstr_read(), ord('B'))

    def test_sstr_yields_a_slice_of_the_bytes(self) -> None:
        self.assertEqual(sstr_read(), ord('Y'))

    def test_sstr_length_is_the_byte_count(self) -> None:
        # the global carries a trailing NUL, but the slice length does not count it
        self.assertEqual(sstr_length(), 4)

    def test_identical_bytes_share_one_global(self) -> None:
        self.assertEqual(gstr_twice(), ord('d') + ord('u'))
        entry = gstr_twice.get_entry()  # pyright: ignore
        instance = next(iter(entry.specs.values()))
        native = instance.wrapper_fn or instance.native_fn
        assert native is not None
        text = '\n'.join(native.print_all())
        self.assertEqual(text.count('c"dup\\00"'), 1)

    def test_ord_requires_exactly_one_byte(self) -> None:
        with self.assertRaises(CompileError):
            ord_of_two_bytes()

    def test_ord_requires_a_byte_string(self) -> None:
        with self.assertRaises(CompileError):
            ord_of_non_bytes()

    def test_an_out_of_bounds_index_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bytes_index_out_of_bounds()

    def test_an_out_of_bounds_slice_is_rejected(self) -> None:
        with self.assertRaises(CompileError):
            bytes_slice_out_of_bounds()


all_tests = [
    SpyBytesTest,
]

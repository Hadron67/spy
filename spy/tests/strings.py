from unittest import TestCase

from ..compiler import (
    CompileError,
    i32,
    u8,
    usize,
)
from ..compiler import bool as spy_bool
from ..compiler.dsl import func
from ..compiler.syntax import (
    Comptime,
    ConstMultiPtr,
)
from ..std import (
    ConstSlicePtr,
    encode_length,
    ptr_cast,
    sstr,
)
from ..std.int import int_cast

# ---------------------------------------------------------------------------
# compile-time strings: a ``bytes`` byte string and a ``str`` text string both
# exist only at compile time (they have no runtime representation) and may be
# subscripted / sliced / compared / concatenated at compile time.  A ``str`` is
# a sequence of *characters* (``len`` is the character count, ``encode_length``
# the UTF-8 byte count), encoded to UTF-8 when it is laid into memory.  Either
# lowers into a runtime pointer/slice through ``ptr_cast`` (see the behaviour doc
# §4.3, replacing the old ``std.core.gstr``) or through ``sstr``, and ``ord``
# yields the encoding of a one-byte byte string or the code point of a
# single-character text string.
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
def str_literal_is_a_text_string() -> i32:
    # a ``str`` literal is a compile-time text string (not encoded to bytes)
    s: Comptime = 'A'
    return ord(s[0])


@func()
def str_index() -> i32:
    s: Comptime = 'abc'
    return ord(s[0]) * 100 + ord(s[2])


@func()
def str_slice_index() -> i32:
    s: Comptime = 'hello'
    t: Comptime = s[1:4]
    return ord(t[0]) * 100 + ord(t[2])


@func()
def str_concat() -> i32:
    s: Comptime = 'ab' + 'cd'
    return ord(s[2]) * 100 + ord(s[3])


@func()
def str_compare() -> spy_bool:
    s: Comptime = 'foo' + 'bar'
    return s == 'foobar'


@func()
def str_char_and_byte_length() -> i32:
    # ``len`` counts characters, ``encode_length`` the UTF-8 bytes
    s: Comptime = 'héllo'
    return int_cast(len(s), i32) * 100 + int_cast(encode_length(s), i32)


@func()
def str_ptr_cast_read() -> i32:
    s: Comptime = 'XY'
    p: ConstMultiPtr[u8] = ptr_cast(s, ConstMultiPtr[u8])
    return p[1]


@func()
def str_sstr_length() -> usize:
    # the slice length is the encoded byte count
    s: Comptime = 'hello'
    sl: ConstSlicePtr[u8] = sstr(s)
    return sl.length


@func()
def str_sstr_multibyte_length() -> usize:
    s: Comptime = 'héllo'
    sl: ConstSlicePtr[u8] = sstr(s)
    return sl.length


@func()
def gstr_read() -> i32:
    s: Comptime = b'AB'
    p: ConstMultiPtr[u8] = ptr_cast(s, ConstMultiPtr[u8])
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
    a: ConstMultiPtr[u8] = ptr_cast(s, ConstMultiPtr[u8])
    b: ConstMultiPtr[u8] = ptr_cast(s, ConstMultiPtr[u8])
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
    """Compile-time byte strings and the ``ptr_cast``/``sstr``/``ord`` spellings."""

    def test_a_byte_string_can_be_indexed(self) -> None:
        self.assertEqual(bytes_index(), ord('a') * 100 + ord('c'))

    def test_a_byte_string_can_be_sliced(self) -> None:
        self.assertEqual(bytes_slice_index(), ord('e') * 100 + ord('l'))

    def test_a_slice_bound_may_be_left_out(self) -> None:
        expected = ord('a') * 100 + ord('b') + ord('c') * 10000 + ord('e') * 100000
        self.assertEqual(bytes_omitted_bounds(), expected)

    def test_ord_yields_the_byte(self) -> None:
        self.assertEqual(ord_of_literal(), ord('a'))

    def test_a_str_literal_is_a_text_string(self) -> None:
        self.assertEqual(str_literal_is_a_text_string(), ord('A'))

    def test_a_text_string_can_be_indexed(self) -> None:
        self.assertEqual(str_index(), ord('a') * 100 + ord('c'))

    def test_a_text_string_can_be_sliced(self) -> None:
        self.assertEqual(str_slice_index(), ord('e') * 100 + ord('l'))

    def test_a_text_string_can_be_concatenated(self) -> None:
        self.assertEqual(str_concat(), ord('c') * 100 + ord('d'))

    def test_a_text_string_can_be_compared(self) -> None:
        self.assertTrue(str_compare())

    def test_len_is_characters_and_encode_length_is_bytes(self) -> None:
        # 'héllo' is 5 characters and 6 UTF-8 bytes
        self.assertEqual(str_char_and_byte_length(), 5 * 100 + 6)

    def test_a_text_string_lowers_through_ptr_cast(self) -> None:
        self.assertEqual(str_ptr_cast_read(), ord('Y'))

    def test_sstr_of_a_text_string_counts_encoded_bytes(self) -> None:
        self.assertEqual(str_sstr_length(), 5)
        self.assertEqual(str_sstr_multibyte_length(), 6)

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

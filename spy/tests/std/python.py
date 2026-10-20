import sys
from typing import Never
from unittest import TestCase

from ...compiler import defer, f64, func, i8, i32, i64, u8, u64, usize
from ...std.core import sstr
from ...std.python import GILState, PyCompareOp, PyError, PyObject
from ..structs import struct_type

# ---------------------------------------------------------------------------
# ``std.python``: the CPython C API bindings
#
# Every function below holds the GIL for its whole body (the JIT entry point
# calls native code with the GIL released) and follows the ownership style of
# the module: a new reference registers its ``decref`` immediately with
# ``defer``; a borrowed reference is never decref'd; a stolen reference
# (``tuple_set``/``list_set``) is not decref'd by the caller.
# ---------------------------------------------------------------------------


@func(exceptions="infer")
def version_micro() -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    mod = PyObject.import_module("sys")
    with defer(): mod[...].decref()

    info = mod[...].getattr("version_info")
    with defer(): info[...].decref()

    micro = info[...].getattr("micro")
    with defer(): micro[...].decref()

    return micro[...].as_i64()


@func(exceptions="infer")
def int_roundtrip(x: i64) -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_i64(x)
    with defer(): obj[...].decref()

    return obj[...].as_i64()


@func(exceptions="infer")
def u64_roundtrip(x: u64) -> u64:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_u64(x)
    with defer(): obj[...].decref()

    return obj[...].as_u64()


@func(exceptions="infer")
def as_i32(x: i64) -> i32:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_i64(x)
    with defer(): obj[...].decref()

    return obj[...].as_int(i32)


@func(exceptions="infer")
def as_u8(x: u64) -> u8:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_u64(x)
    with defer(): obj[...].decref()

    return obj[...].as_int(u8)


@func(exceptions="infer")
def as_i8_overflows() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_i64(300)
    with defer(): obj[...].decref()

    try:
        obj[...].as_int(i8)
        return False
    except PyError:
        return True


@func(exceptions="infer")
def float_roundtrip(x: f64) -> f64:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_f64(x)
    with defer(): obj[...].decref()

    return obj[...].as_f64()


@func(exceptions="infer")
def bool_roundtrip(x: bool) -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_bool(x)
    with defer(): obj[...].decref()

    return obj[...].as_bool()


@func(exceptions="infer")
def str_utf8_len() -> usize:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_str(sstr("héllo"))
    with defer(): obj[...].decref()

    return obj[...].as_utf8().length


@func(exceptions="infer")
def bytes_len() -> usize:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_bytes(sstr(b"abc"))
    with defer(): obj[...].decref()

    return obj[...].as_bytes().length


@func(exceptions="infer")
def call_len_of_list() -> u64:
    state = GILState.acquire()
    with defer(): state.release()

    builtins = PyObject.import_module("builtins")
    with defer(): builtins[...].decref()


    fn = builtins[...].getattr("len")
    with defer(): fn[...].decref()

    lst = PyObject.new_list(0)
    with defer(): lst[...].decref()

    one = PyObject.from_i64(1)
    lst[...].list_append(one)
    one[...].decref()

    result = fn[...].call_one_arg(lst)
    with defer(): result[...].decref()

    return result[...].as_u64()


@func(exceptions="infer")
def tuple_roundtrip() -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    t = PyObject.new_tuple(1)
    with defer(): t[...].decref()

    # ``tuple_set`` steals the reference to ``value``: no decref for it
    value = PyObject.from_i64(11)
    t[...].tuple_set(0, value)

    item = t[...].tuple_get(0)
    return item[...].as_i64()


@func(exceptions="infer")
def list_roundtrip() -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    lst = PyObject.new_list(0)
    with defer(): lst[...].decref()

    first = PyObject.from_i64(1)
    lst[...].list_append(first)
    first[...].decref()

    second = PyObject.from_i64(2)
    lst[...].list_append(second)
    second[...].decref()

    # ``list_set`` steals the new item: no decref for it
    replacement = PyObject.from_i64(9)
    lst[...].list_set(0, replacement)

    return lst[...].list_get(0)[...].as_i64() + lst[...].list_get(1)[...].as_i64()


@func(exceptions="infer")
def dict_roundtrip() -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    d = PyObject.new_dict()
    with defer(): d[...].decref()

    value = PyObject.from_i64(17)
    with defer(): value[...].decref()
    d[...].dict_set_str("x", value)

    if (got := d[...].dict_get_str("x")) is not None:
        return got[...].as_i64()
    return -1


@func(exceptions="infer")
def iter_sum() -> i64:
    state = GILState.acquire()
    with defer(): state.release()

    lst = PyObject.new_list(0)
    with defer(): lst[...].decref()

    one = PyObject.from_i64(1)
    lst[...].list_append(one)
    one[...].decref()

    two = PyObject.from_i64(2)
    lst[...].list_append(two)
    two[...].decref()

    three = PyObject.from_i64(3)
    lst[...].list_append(three)
    three[...].decref()

    it = lst[...].get_iter()
    with defer(): it[...].decref()

    total: i64 = 0
    while True:
        if (item := it[...].iter_next()) is not None:
            with defer(): item[...].decref()
            total = total + item[...].as_i64()
        else:
            break
    return total


@func(exceptions="infer")
def rich_compare_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    one = PyObject.from_i64(1)
    with defer(): one[...].decref()

    two = PyObject.from_i64(2)
    with defer(): two[...].decref()

    less = one[...].rich_compare(two, PyCompareOp.LT)
    with defer(): less[...].decref()

    return less[...].as_bool() and one[...].rich_compare_bool(two, PyCompareOp.NE)


@func(exceptions="infer")
def none_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    none = PyObject.none()
    value = PyObject.from_bool(True)
    with defer(): value[...].decref()

    return none[...].is_none() and not value[...].is_none()


@func(exceptions="infer")
def error_path() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    mod = PyObject.import_module("sys")
    with defer(): mod[...].decref()

    if mod[...].try_getattr("definitely_missing") is not None:
        return False
    try:
        mod[...].getattr("definitely_missing")
        return False
    except PyError:
        return True


@func(exceptions="infer")
def callable_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    builtins = PyObject.import_module("builtins")
    with defer(): builtins[...].decref()

    fn = builtins[...].getattr("len")
    with defer(): fn[...].decref()

    lst = PyObject.new_list(0)
    with defer(): lst[...].decref()

    return fn[...].is_callable() and not lst[...].is_callable()


@func(exceptions="infer")
def refcount_incref() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    sys_mod = PyObject.import_module("sys")
    with defer(): sys_mod[...].decref()

    getrefcount = sys_mod[...].getattr("getrefcount")
    with defer(): getrefcount[...].decref()

    obj = PyObject.new_list(0)
    with defer(): obj[...].decref()

    before = getrefcount[...].call_one_arg(obj)
    with defer(): before[...].decref()

    obj[...].incref()
    after = getrefcount[...].call_one_arg(obj)
    with defer(): after[...].decref()

    obj[...].decref()
    return after[...].as_i64() == before[...].as_i64() + 1


@func(exceptions="infer")
def attr_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    types = PyObject.import_module("types")
    with defer(): types[...].decref()

    ns_cls = types[...].getattr("SimpleNamespace")
    with defer(): ns_cls[...].decref()

    ns = ns_cls[...].call_no_args()
    with defer(): ns[...].decref()

    value = PyObject.from_i64(5)
    with defer(): value[...].decref()

    ns[...].setattr("x", value)
    if not ns[...].hasattr("x"):
        return False
    got = ns[...].getattr("x")
    with defer(): got[...].decref()
    if got[...].as_i64() != 5:
        return False

    ns[...].delattr("x")
    return not ns[...].hasattr("x")


@func(exceptions="infer")
def type_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    obj = PyObject.from_i64(1)
    with defer(): obj[...].decref()

    typ = obj[...].type_obj()
    with defer(): typ[...].decref()

    return obj[...].is_instance(typ)


@func(exceptions="infer")
def module_dict_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    mod = PyObject.import_module("sys")
    with defer(): mod[...].decref()

    d = mod[...].module_dict()
    return d[...].dict_get_str("version_info") is not None


@func(exceptions="infer")
def module_constants_check() -> bool:
    state = GILState.acquire()
    with defer(): state.release()

    types = PyObject.import_module("types")
    with defer(): types[...].decref()

    module_cls = types[...].getattr("ModuleType")
    with defer(): module_cls[...].decref()

    name = PyObject.from_str(sstr("spy_test_module"))
    with defer(): name[...].decref()

    mod = module_cls[...].call_one_arg(name)
    with defer(): mod[...].decref()

    value = PyObject.from_i64(7)
    with defer(): value[...].decref()
    mod[...].module_add_object_ref("value", value)
    mod[...].module_add_int_constant("number", 42)
    mod[...].module_add_string_constant("text", "hello")

    number = mod[...].getattr("number")
    with defer(): number[...].decref()

    text = mod[...].getattr("text")
    with defer(): text[...].decref()

    stored = mod[...].getattr("value")
    with defer(): stored[...].decref()

    return number[...].as_i64() == 42 and stored[...].as_i64() == 7 and text[...].as_utf8().length == 5


@func(exceptions=PyError)
def raise_py_error_from_spy() -> Never:
    raise PyError()


class SpyPythonTest(TestCase):
    def test_the_gil_wrapped_import_and_attribute_chain(self) -> None:
        self.assertEqual(version_micro(), sys.version_info.micro)

    def test_integer_roundtrip(self) -> None:
        self.assertEqual(int_roundtrip(0), 0)
        self.assertEqual(int_roundtrip(-7), -7)
        self.assertEqual(int_roundtrip(2**62), 2**62)

    def test_unsigned_roundtrip(self) -> None:
        self.assertEqual(u64_roundtrip(2**63 + 1), 2**63 + 1)

    def test_checked_integer_casts(self) -> None:
        self.assertEqual(as_i32(-42), -42)
        self.assertEqual(as_u8(255), 255)
        self.assertTrue(as_i8_overflows())

    def test_float_roundtrip(self) -> None:
        self.assertEqual(float_roundtrip(1.5), 1.5)
        self.assertEqual(float_roundtrip(-0.25), -0.25)

    def test_bool_roundtrip(self) -> None:
        self.assertIs(bool_roundtrip(True), True)
        self.assertIs(bool_roundtrip(False), False)

    def test_utf8_length(self) -> None:
        self.assertEqual(str_utf8_len(), 6)

    def test_bytes_length(self) -> None:
        self.assertEqual(bytes_len(), 3)

    def test_calling_a_python_callable(self) -> None:
        self.assertEqual(call_len_of_list(), 1)

    def test_tuple_set_steals_and_tuple_get_borrows(self) -> None:
        self.assertEqual(tuple_roundtrip(), 11)

    def test_list_operations(self) -> None:
        self.assertEqual(list_roundtrip(), 11)

    def test_dict_operations(self) -> None:
        self.assertEqual(dict_roundtrip(), 17)

    def test_iteration(self) -> None:
        self.assertEqual(iter_sum(), 6)

    def test_rich_comparison(self) -> None:
        self.assertTrue(rich_compare_check())

    def test_none(self) -> None:
        self.assertTrue(none_check())

    def test_missing_attribute_raises_py_error(self) -> None:
        self.assertTrue(error_path())

    def test_callable_check(self) -> None:
        self.assertTrue(callable_check())

    def test_manual_incref_decref(self) -> None:
        self.assertTrue(refcount_incref())

    def test_attributes(self) -> None:
        self.assertTrue(attr_check())

    def test_type_and_is_instance(self) -> None:
        self.assertTrue(type_check())

    def test_module_dict(self) -> None:
        self.assertTrue(module_dict_check())

    def test_module_constants(self) -> None:
        self.assertTrue(module_constants_check())

    def test_py_error_crosses_the_python_boundary(self) -> None:
        with self.assertRaises(Exception) as ctx:
            raise_py_error_from_spy()
        self.assertIs(ctx.exception._data.type, struct_type(PyError))  # pyright: ignore


all_tests = [SpyPythonTest]

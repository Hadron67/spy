"""Integration tests for the spy JIT (``spy``).

The functions under test are defined at module level and registered with
the ordinary ``@spy.func()`` decorator; a function body may call the
other registered functions by name (they are module globals, resolved by
the compile-time interpreter) exactly like a user would.  The
undecorated ``add_inline`` is deliberately left unregistered: it stays a
plain Python function and is inlined at its call sites.

The tests are grouped by feature into the submodules of this package; each
submodule defines an ``all_tests`` list of its test cases, and this module
collects them.  The global host context caches specializations, so the
tests share the compiled functions; a test that needs a fresh compilation
calls a function no earlier test has compiled.
"""

from unittest import TestCase, TestSuite

from .annotations import all_tests as annotations_tests
from .arrays import all_tests as arrays_tests
from .basics import all_tests as basics_tests
from .classname import all_tests as classname_tests
from .comptime import all_tests as comptime_tests
from .comptime_structs import all_tests as comptime_structs_tests
from .cross_context import all_tests as cross_context_tests
from .defer import all_tests as defer_tests
from .errors import all_tests as errors_tests
from .func_types import all_tests as func_types_tests
from .generics import all_tests as generics_tests
from .ifexpr import all_tests as ifexpr_tests
from .inline_loops import all_tests as inline_loops_tests
from .loops import all_tests as loops_tests
from .operators import all_tests as operators_tests
from .options import all_tests as options_tests
from .pointers import all_tests as pointers_tests
from .reflect import all_tests as reflect_tests
from .returns import all_tests as returns_tests
from .slices import all_tests as slices_tests
from .static import all_tests as static_tests
from .structs import all_tests as structs_tests
from .tagged_unions import all_tests as tagged_unions_tests
from .types import all_tests as types_tests
from .undefined import all_tests as undefined_tests

all_tests: list[type[TestCase]] = [
    *basics_tests,
    *loops_tests,
    *inline_loops_tests,
    *structs_tests,
    *annotations_tests,
    *comptime_tests,
    *generics_tests,
    *pointers_tests,
    *slices_tests,
    *arrays_tests,
    *ifexpr_tests,
    *comptime_structs_tests,
    *options_tests,
    *returns_tests,
    *errors_tests,
    *tagged_unions_tests,
    *types_tests,
    *cross_context_tests,
    *classname_tests,
    *defer_tests,
    *func_types_tests,
    *reflect_tests,
    *operators_tests,
    *undefined_tests,
    *static_tests,
]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for case in all_tests:
        suite.addTests(loader.loadTestsFromTestCase(case))
    return suite

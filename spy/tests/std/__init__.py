"""Tests for the ``spy.std`` modules."""

from .atomic import all_tests as atomic_tests
from .int import all_tests as int_tests
from .mem import all_tests as mem_tests
from .openmp import all_tests as openmp_tests

all_tests = [
    *atomic_tests,
    *int_tests,
    *mem_tests,
    *openmp_tests,
]

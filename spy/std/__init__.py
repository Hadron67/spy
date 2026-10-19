"""The spy standard library.

``std`` re-exports the core types defined in :mod:`spy.std.core` and the
``syntax`` markers a spy function body may use, so that ``from std import
T`` names any of them.
"""

from . import c as c
from . import io as io
from . import llvm as llvm
from . import math as math
from . import mem as mem
from . import openmp as openmp
from . import python as python
from . import reflect as reflect
from . import vec as vec
from .core import *

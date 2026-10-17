"""Float casts (``std.float``).

``float_truncate`` is a builtin (see
``interp.HirRunner._builtin_float_truncate``).  It is the explicit ``f64`` ->
``f32`` narrowing: the implicit conversion (``std.core.coerce`` and every
store) only widens, so narrowing a float has to be spelled out.
"""

from typing import Any

from ..compiler import builtin_func


@builtin_func
def float_truncate[T](value: Any, typ: type[T]) -> T: ...

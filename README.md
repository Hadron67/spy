# spy - System Python

[中文文档](README_zh.md)

`spy` is a package that **JIT-compiles Python functions into machine code**. You write ordinary Python — just decorate a function with `@spy.func()` to register it; on the first call, a **specialized** version is compiled for the types of the actual arguments, and every call after that is a native call that never goes through Python again. The function body is "executed" at compile time with concrete types, so Zig-like **comptime** semantics are supported (`spy.typeof`, compile-time `if`, generics, inlining ordinary Python functions, etc.).

Functions must be defined in real source files (compilation needs the function's source, so interactive environments are not supported).

## Example

```python
from spy.compiler import func, u64
from spy.compiler import as_ as spy_as


@func()                         # register; compiled on first call
def add[T](a: T, b: T) -> T:
    return a + b


@func()
def add_u64(a: u64, b: u64) -> u64:
    return a + b


print(add(1, 2))                                  # compiles add(i64, i64)
print(add(1.5, 2.25))                             # compiles add(f64, f64)
print(add_u64(spy_as(2**63 - 1, u64), spy_as(2, u64)))
```

See [doc/en/features.md](doc/en/features.md) for detailed feature documentation.

## Project structure

- [doc/](doc/) — documentation
- [spy/](spy/) — the Python package
- [spy/compiler/](spy/compiler/) — the compiler: Python source → machine code
- [spy/std/](spy/std/) — standard library
- [spy/tests/](spy/tests/) — integration tests

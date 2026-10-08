# Instructions

## Python specific
* For interpreter: use `python3` instead of `python`.
* Always use relative imports if possible.
* Always use type annotations, and prefer concrete types in annotations. Widecard types such as `Any` or `object` should only be used when necessary. Forward reference in type annotation is supported by current version of Python, so do not use `Any` for forward reference.
* A method **must** use `self`, otherwise, define it as a function.
* When using `@dataclass` annotation, always pass `slots=True`
* When using object identity dicts, use `map: dict[util.IdentityKey[K], V]` and `map[util.IdentityKey(key)] = val` instead of `map: dict[int, V]` and `map[id(key)] = val`. If `util.IdentityKey` is not defined, define it first.
* Local variables should be assigned first, i.e., write
```python
a: Foo | None = None
if foo():
    a = bar()
else:
    a = baz()
check(a)
```
instead of
```python
if foo():
    a = bar()
else:
    a = baz()
check(a)
```

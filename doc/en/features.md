# spy feature list

This document lists the language features and observable behavior already implemented by the **current code** (`spy/spy/compiler` + `spy/spy/std`).

> Note: this document is organized by "language surface" — that is, what a spy programmer can write and its semantics.
> How the compiler's internal stages implement these behaviors is described in `doc/implementation.md`.

---

## 0. Overview

`spy` compiles ordinary Python functions into native machine code. After a function is registered with `@spy.func()`, the first call compiles,
according to the argument types, a **specialization** for that type; afterwards it is a native call that no longer goes through Python.

Functions must be defined in a real source file (compilation needs the function's source code).

For the compilation pipeline and the implementation of each stage, see `doc/implementation.md`.

### 0.1 Example conventions

All ```python blocks in this document are **runnable**, complete examples: they assume the following **shared import** (in real code
the public entry point of `spy` is `spy.compiler`; here the equivalent form `import spy.compiler as spy` is used):

```python
from spy.compiler import (
    func, struct, decl_func, func_type, Enum,
    i8, i16, i32, i64, u8, u16, u32, u64, u0, usize, isize,
    f32, f64, c64, c128,
    c_char, c_uchar, c_short, c_ushort, c_int, c_uint,
    c_long, c_ulong, c_longlong, c_ulonglong,
    Ptr, ConstPtr, MultiPtr, ConstMultiPtr, Option,
    typeof, compile_log, as_ as spy_as,
)
from spy.compiler.syntax import (
    Array, Opaque, array,
    Comptime, ConstComptime, InferComptime,
    comptime, const_comptime, infer_comptime, unroll,
    ref, ptr_cast, as_func_ptr, closure,
    defer, okdefer, errdefer, OK, RAISE, UNWIND, ERR, ALL,
)
from spy.std import (
    ConstSlicePtr, SlicePtr, Numeric, StopIteration,
    arr_slice, const_arr_slice, const_array, range, slice, sstr, undefined,
)
from spy.std.core import (
    panic, catch_unwind, UnwindException, coerce, typed_undefined, bitcast,
    unwrap, unwrap_or, compile_error,
)
from spy.std.c import snprintf
from spy.std.mem import Layout, layout_of, size_of, align_of
from spy.std.int import IntCastError, int_cast, try_int_cast, truncate, valid_range
from spy.std.atomic import (
    MemoryOrder, RmwOp,
    atomic_load, atomic_store, atomic_rmw, atomic_cmpxchg, atomic_fence, atomic_fetch_add,
)
from spy.std.reflect import TypeInfo, IntType, type_info, reify
from typing import TYPE_CHECKING, Literal, Never, NoReturn
```

> Code blocks can be run in bulk for verification with `tools/run_doc_examples.py`: the script materializes each ```python block into a real
> `.py` file (`spy` needs source code) and then executes it; the first runnable block is prepended to every other block as the shared import.
> Run: `python3 tools/run_doc_examples.py doc/features.md`.

---

## 1. Type system

### 1.1 Integers and floats

- Fixed-width integers `u8/u16/u32/u64`, `i8/i16/i32/i64`; zero-bit integer `u0` (ZST).
- Floats `f32/f64`.
- `spy.usize` / `spy.isize`: integers determined by the target platform's pointer width.
- C integer types `c_char/c_uchar`, `c_short/c_ushort`, `c_int/c_uint`, `c_long/c_ulong`, `c_longlong/c_ulonglong`:
  the width (and the signedness of `c_char`) is determined by the target platform (for example `c_long` is 32-bit on Windows and 64-bit on common 64-bit Unix).
- Untyped integer literals: the literal `1` initially has no concrete type; it is determined by the place it is written into or the type it is paired with.
- Integer subtyping follows range inclusion: `i8` is a subtype of `i16`, and `u8` is not a subtype of `i8`. Integers are not floats.

```python
@func()
def arith(a: i32, b: i32) -> i32:
    return a + b * 2 - (a // b) + (a % b)

@func()
def to_float(a: i32, b: i32) -> f64:
    return a / b                 # integer / is true division -> f64

@func()
def width_and_literals() -> i32:
    x: i32 = 1                   # the literal's type comes from the annotation
    y: i64 = x                   # widening is implicit
    return x + 2

@func()
def target_sized(n: usize) -> usize:
    return n + 1

@func()
def c_types(x: c_int) -> c_int:
    return x + 1

assert arith(7, 3) == 12
assert to_float(3, 2) == 1.5
assert width_and_literals() == 3
assert target_sized(5) == 6
assert c_types(41) == 42
```

### 1.2 Booleans

`spy.bool`. There is no "truthiness conversion": the conditions of `if`/`while`/`not`/`and`/`or` must all be `spy.bool`
(when an object provides `__bool__` it is also called, see §7.2).

```python
@func()
def both(a: bool, b: bool) -> bool:
    return a and b

@func()
def in_range(x: i32) -> bool:
    return x > 0 and x < 10

assert both(True, False) is False
assert in_range(5) is True and in_range(20) is False
```

### 1.3 Pointers

`syntax.Ptr[T]` (mutable single pointer), `ConstPtr[T]` (read-only single pointer), `MultiPtr[T]` / `ConstMultiPtr[T]`
(multi pointer, subscriptable like an array).

- A single pointer is written `p[...]` (dereference, C's `*p`), a multi pointer is written `p[i]` (the place of the i-th element).
- Pointers are **non-null**; `Option[Ptr[T]]` is what gives a nullable pointer.
- A mutable pointer implicitly converts to const (`Ptr[T]` → `ConstPtr[T]`); the reverse does not hold.
- A pointer to an array (`*[N]T`) can be treated as a multi pointer to its elements.
- A pointer to a ZST is "dangling but non-null", not a null pointer.
- `syntax.ref(a)` takes an address; `syntax.ptr_cast(p, T)` reinterprets the pointer type (no checks); `p[...]` dereferences.

```python
@func()
def incr(p: Ptr[i32]) -> i32:
    p[...] = p[...] + 1          # ``p[...]`` is an lvalue
    return p[...]

@func()
def use_incr(x: i32) -> i32:
    v: i32 = x
    incr(ref(v))                 # take the address of a local
    return v

@func()
def pointer_demo(x: i32) -> i32:
    v: i32 = x
    p: Ptr[i32] = ref(v)
    cp: ConstPtr[i32] = p        # mutable -> const is implicit
    q: Ptr[i32] = ptr_cast(p, Ptr[i32])
    q[...] = cp[...] + 1
    return v

@func()
def low_byte(x: i32) -> u8:
    v: i32 = x
    return ptr_cast(ref(v), Ptr[u8])[...]   # reinterpret the address

assert use_incr(5) == 6
assert pointer_demo(5) == 6
assert low_byte(0x41) == 0x41
```

### 1.4 Arrays and slices

- `syntax.Array[T, N]`: `N` `T`s laid out in a row; the length is a **value** type parameter, written as `Literal[N]` in annotations.
  The length may be omitted as `Array[T, None]` (**unsized array**, DST).
- `syntax.array(a1, a2, ...)` constructs an array in place: the length is the number of arguments, and the element type is the type of the target place, or otherwise the common type of all elements;
  there is also a `length=` keyword that exists **only for type checkers** (the actual length follows the number of arguments).
- `a[i]` is an **lvalue** of the i-th element (readable/writable/`+=`); a compile-time constant subscript checks out of bounds, a runtime subscript does not.
- Slice types live in `std`: `std.SlicePtr[T]` / `std.ConstSlicePtr[T]` (the struct `{ptr, length}`).
  `std.arr_slice(p)` / `std.const_arr_slice(p)` turn a pointer to an array into a slice of the whole thing;
  a multi-pointer slice `p[a:b]` yields `SlicePtr(p+a, b-a)` (lower bound defaults to 0, upper bound to the length, step must be 1).
- A slice is iterable (element values) and has `refs()` (element places); going out of bounds `panic`s.
- `std.core.const_array(elem, length)`: an array of `length` equal elements (the element expression is evaluated only once and may be a runtime value).
- Arrays **cannot cross the Python calling boundary** (see §10), so the examples below call indirectly through spy functions.

```python
@func()
def sum_array(a: Array[i32, Literal[3]]) -> i32:
    return a[0] + a[1] + a[2]

@func()
def use_sum(x: i32) -> i32:
    a = array(x, x + 1, x + 2, length=3)
    return sum_array(a)

@func()
def build_array(x: i32) -> i32:
    a = array(x, x + 1, x + 2, length=3)
    a[0] = 7
    return a[0] + a[1] + a[2]

@func()
def slice_array(x: i32) -> i32:
    a = array(x, x + 1, x + 2, length=3)
    s = arr_slice(ref(a))
    t = s.slice(1, 3)            # elements 1..2
    return s[0] + t[0] + t[1]

@func()
def repeat(n: i32) -> i32:
    a = const_array(n, 4)
    return a[0] + a[3]

assert use_sum(1) == 6
assert build_array(1) == 12
assert slice_array(1) == 6       # 1 + 2 + 3
assert repeat(2) == 4
```

### 1.5 Option

`syntax.Option[T]` (equivalent to `T | None`):

- A value is either a `T` or `None`.
- Both `T` and `None` implicitly convert to `Option[T]`; when a single place receives both `T` and `None`, the type becomes `Option[T]`.
- Testing and unwrapping: `expr is None` / `expr is not None`; the walrus syntax `(name := expr) is not None` binds `name`
  to the writable place of the inner value; plain `(name := expr)` binds `name` to the option itself.
- **Not yet implemented**: pattern matching an option; under the borrow-style representation there are still edge cases where absence is misjudged (see `doc/implementation.md`).

```python
@func()
def maybe_add(x: i32, y: i32, c: bool) -> Option[i32]:
    if c:
        return x + y
    return None

@func()
def bumped(x: i32, y: i32, c: bool) -> i32:
    if (v := maybe_add(x, y, c)) is not None:
        v += 1                   # writes the payload of the option
        return v
    return 0

@func()
def is_absent(o: Option[i32]) -> bool:
    return o is None

assert bumped(3, 4, True) == 8
assert bumped(3, 4, False) == 0
assert maybe_add(1, 2, True) == 3
assert maybe_add(1, 2, False) is None    # an Option crosses as None
assert is_absent(None) is True and is_absent(5) is False
```

### 1.6 Tuples

`tuple[T1, T2, ...]`:

- When all elements have a runtime representation, a tuple is an ordinary runtime aggregate value (copyable ⇔ all elements are copyable).
- In a **variable/field/expression value** position it is "a tuple value": `t[i]` is an lvalue of the i-th element (`i` is a compile-time integer; out-of-bounds/negative/runtime subscripts are an error),
  `len(t)` is a compile-time integer, and the whole thing can be copied (`b = a`).
- In a **parameter/return annotation** position it is interpreted specially: a parameter `tuple[...]` is **flattened passing** (see §6.4), a return `tuple[...]` is **multiple return values** (see §6.5).
- Elements without a runtime representation (types, byte strings, DST function types) make the whole tuple a compile-time value only.
- Tuple `+=` / `+` are **no longer supported** (a tuple is now an ordinary runtime aggregate with a fixed arity).

```python
@func()
def tuple_use(a: i32, b: i32) -> i32:
    t = (a, b)                   # a runtime tuple value
    t[1] = t[1] + 1
    return t[0] + t[1] + len(t)

assert tuple_use(3, 4) == 3 + 5 + 2
```

### 1.7 Untagged union (`UnionType`)

An exception's payload is an **untagged union**: the same piece of storage is interpreted as different variants, and the variants share the same space.
Union **values cannot be converted to one another**; only the storage can be reinterpreted. A union that occupies no storage (no variants, or all variants ZST) is only a compile-time value.
It has no direct source syntax and appears only in exception payloads and reflection (see §5.5, §8.8).

### 1.8 Tagged union

`A | B | C`:

- The value is exactly one of the variants, carrying a variant tag (the variant order is part of the type's identity).
- Testing/unwrapping: `isinstance(u, A)`; the walrus `isinstance(e := u, A)` binds `e` to the variant value (writable);
  `match u:` + `case A():` (test the variant) / `case A() as e:` (bind the variant value) / `case _:` (wildcard, must come last).
- A subset union converts to a superset union; `A | A` deduplicates to a single variant;
  a union whose tag is known only at runtime is always an ordinary runtime value.

```python
@struct()
class A:
    x: i32

@struct()
class B:
    y: f64

@func()
def tu_value(u: A | B) -> i32:
    if isinstance(e := u, A):        # bind ``e`` to the variant payload
        return e.x
    return -1

@func()
def tu_make(c: bool) -> A | B:
    if c:
        return A(5)
    return B(1.5)

@func()
def tu_match(u: A | B) -> i32:
    match u:
        case A() as a:
            return a.x
        case _:
            return 0

assert tu_value(A(7)) == 7
assert tu_value(B(1.0)) == -1
assert tu_make(True).x == 5         # the union unwraps to the variant value
assert tu_make(False).y == 1.5
assert tu_match(A(9)) == 9
```

### 1.9 Complex numbers

Complex types `spy.c64` / `spy.c128`: the real part and the imaginary part are each a floating-point element.

- A Python `complex` maps to `c128`; there is an implicit `T → Complex[T]` conversion for reals/integers (imaginary part 0), and `Complex[f32]` widens to `Complex[f64]`.
- Supports `+ - * /` (component-wise) and the fields `z.real` / `z.imag`.
- `std.reflect.type_info` yields the `ComplexType` reflection variant.

```python
@func()
def cmul(z: c128, w: c128) -> c128:
    return z * w

@func()
def cparts(z: c128) -> f64:
    return z.real + z.imag

@func()
def widen(x: f64) -> c128:
    return x                     # implicit T -> Complex[T]

assert cmul(1 + 2j, 3 + 4j) == (1 + 2j) * (3 + 4j)
assert cparts(3 + 4j) == 7.0
assert widen(2.5) == complex(2.5, 0.0)
```

### 1.10 Integer enums

A class that inherits `spy.Enum` (the standard `enum.Enum`) with integer member values declares a spy type:

- It can be used directly as a parameter/return value/field annotation; its members are compile-time values.
- **It is not an integer**: it takes part in no arithmetic/bit operations, and is not compared/converted with `int` or another enum; only `==`/`!=` between members of the same enum are supported (comparing member identity).
- Methods are supported (`@func()`-compiled, undecorated inlined, `@staticmethod`); a method's `self` is **by value** by default (use `@func(self_arg="ptr"/"const ptr")` to switch to by reference).
- At the Python boundary: a member passed in is recognized as an enum value, and an enum return value/field is restored to a Python member.

```python
class TriState(Enum):
    UNKNOWN = 0
    TRUE = 1
    FALSE = 2

@func()
def negate_state(s: TriState) -> TriState:
    if s == TriState.TRUE:
        return TriState.FALSE
    if s == TriState.FALSE:
        return TriState.TRUE
    return TriState.UNKNOWN

@func()
def state_is_unknown(s: TriState) -> bool:
    return s == TriState.UNKNOWN

assert negate_state(TriState.TRUE) is TriState.FALSE
assert negate_state(TriState.UNKNOWN) is TriState.UNKNOWN
assert state_is_unknown(TriState.UNKNOWN) is True
```

### 1.11 Structs

See §7.

### 1.12 ZST (zero-sized type) and DST

- **ZST**: the `void` of `-> None`, `u0`, structs whose fields are all ZST, arrays whose elements are ZST or whose length is 0, `Null`, `undefined`,
  and so on. A ZST takes no runtime space, but a ZST result still delivers a unit value as usual.
- **DST (dynamically-sized type)**: DST function types, `Opaque`, unsized arrays; only **a pointer to it** is a value, and it cannot be passed/returned by value or read as a whole.
  The first DST field in a struct is a flexible array member, and later DST fields occupy no slot.
- **Types with no runtime representation (purely compile time)**: type values (`type`), byte strings (`bytes`), DST function types, the empty type (`Never`).
- A pointer to a ZST is non-null (but does not point to actual memory).

```python
@func()
def do_nothing(x: i32) -> None:
    return None

@func()
def zero_bit(x: u0) -> i32:
    return 1

@func()
def opaque_roundtrip(p: Ptr[Opaque]) -> Ptr[Opaque]:
    return p

@func()
def take_opaque(x: i32, c: bool) -> bool:
    v: i32 = x
    p: Ptr[Opaque] = ptr_cast(ref(v), Ptr[Opaque])
    if c:                        # ``opaque_roundtrip`` is compiled, not run
        opaque_roundtrip(p)
    return c

assert do_nothing(1) is None
assert zero_bit(spy_as(0, u0)) == 1
assert take_opaque(1, False) is False
```

### 1.13 Empty type (`Never`)

The empty type (`Never`): it has no values at all. `typing.Never`/`NoReturn` map to it, and it is used to declare functions that never return.
It is the result type of a function whose "body never delivers a value": the body may not contain `return`, and cannot fall through to the end.
Empty type + no exceptions thrown ⇒ a `noreturn` function (see §5.6).

```python
@struct()
class MyError(Exception):
    code: i32

@func(exceptions=MyError)
def always_raise() -> Never:
    raise MyError(1)

@func()
def spins() -> Never:
    while True:
        pass

@func()
def maybe_spin(c: bool) -> bool:
    if c:                        # ``spins`` is compiled (a noreturn call)
        spins()
    return c

try:
    always_raise()
except Exception as e:          # a spy exception crosses as an Exception instance
    assert e.code == 1
assert maybe_spin(False) is False
```

---

## 2. Runtime values and compile-time values

A value is either a **runtime value** (it has a runtime representation) or a **compile-time value** (it exists only at compile time):

- Compile-time values cover: scalar constants, type values, compile-time aggregate values, compile-time Option/tagged unions, `Null`/`Void`/undefined,
  compile-time pointers, and ordinary Python `int/float/complex/bytes/bool`; in addition there are compile-time objects such as function values.

**Key point**: there is no such thing as "a compile-time value in the form of a runtime value". An aggregate any of whose components is a runtime value
**is a runtime value as a whole** and cannot be stored in a `Comptime` variable/parameter (to hold it, use an ordinary variable).

For the concrete storage mechanism of compile-time variables, see `doc/implementation.md`.

```python
@struct()
class Small:
    a: i32
    b: i32

@func()
def runtime_aggregate(x: i32) -> i32:
    s = Small(x, 2)              # a runtime struct lives in an ordinary variable
    return s.a + s.b

@func()
def comptime_box(x: i32) -> i32:
    c: Comptime[i32] = 5         # a compile-time box, not memory
    return c + x

@func()
def inferred(x: i32) -> i32:
    infer_comptime()
    t = 3                        # an untyped literal stays a compile-time box
    return t + x

assert runtime_aggregate(1) == 3
assert comptime_box(1) == 6
assert inferred(1) == 4
```

---

## 3. Variables and declarations

### 3.1 Ordinary variables

- `name = expr`: when the name is already bound this is an assignment; only a name that has never been bound **declares** a block-local variable.
  A declaration does not escape its block; an assignment to an **outer** variable writes that outer variable.
- An unannotated ordinary declaration / a destructuring target yields an ordinary runtime variable.
- A local variable needs a type that can be materialized at runtime: `x = 1` (unannotated, untyped integer literal) is an error and must be written as `x: i32 = 1` or `x: Comptime = 1`.
- `name <operator>= expr` (all binary operators) and tuple unpacking assignment `a, b = e` (nestable) are supported.
  Destructuring **reads the source and writes the targets in subscript order**, so `a, b = b, a` is a known defect (the right-hand side reads values that may already be overwritten).
- Only `=` and `+=` assignments are supported (no chained assignment `a = b = e`).

```python
@func()
def locals_demo(x: i32) -> i32:
    y: i32 = x + 1
    z = y * 2                    # unannotated: runtime storage, type from the value
    z += 1
    return z

@func()
def destructure(a: i32, b: i32) -> i32:
    x, y = a, b
    x, y = y, x
    return x * 10 + y            # the swap is a known flaw: reads source in order

assert locals_demo(1) == 5
assert destructure(1, 2) == 22
```

### 3.2 Type-annotated local variables

- `name: T = expr`: declares a new block-local variable with the given type `T` (the value is converted to `T`).
  Annotating is declaring: the same name cannot be annotated twice.
- The type in the annotation is itself a compile-time value expression (it may name type parameters).

```python
@func()
def annotated_local(x: i32) -> i64:
    y: i64 = x                   # the value is coerced to the declared type
    return y + 1

assert annotated_local(4) == 5
```

### 3.3 Compile-time variables

- `x: Comptime = e` / `x: Comptime[T] = e`: a **compile-time variable**.
  The value written must be **fully known at compile time**; writing a runtime value into a `Comptime` variable is a compile error.
- The statement marker `syntax.comptime()` followed by a **new declaration** is equivalent to a `Comptime` annotation.
  The marker must come **immediately before the marked declaration**, and may only mark a **new declaration** (not an assignment to an already-bound name).
- A `Comptime` variable can hold: type values, compile-time scalars, compile-time aggregates, compile-time tuples/dicts, compile-time Options/unions, compile-time pointers, function values, and so on.

```python
@func()
def comptime_holds_type[T](x: T) -> i32:
    comptime()
    t = typeof(x)                # a type value, kept in a compile-time variable
    if t == i32:                 # a compile-time ``if``
        return 1
    return 0

@func()
def comptime_marker_typed() -> i32:
    comptime()
    i: i32 = 0
    i = i + 1                    # still a compile-time box
    return i

assert comptime_holds_type(spy_as(1, i32)) == 1
assert comptime_holds_type(spy_as(1.5, f64)) == 0
assert comptime_marker_typed() == 1
```

### 3.4 The compile-time constant `ConstComptime`

- `x: ConstComptime = e` / `x: ConstComptime[T] = e` (an initial value is required), or the marker `syntax.const_comptime()` + a new declaration.
- After initialization it is read-only; writing it afterwards is a compile error.
- It is the source of "compile-time global constants": after initialization, `ref(x)` yields a pointer to a global constant.

```python
@func()
def static_read() -> i32:
    const_x: ConstComptime[i32] = 42
    p = ref(const_x)             # automatically becomes a global constant
    return p[...]

assert static_read() == 42
```

### 3.5 `InferComptime` (inferred storage)

- `x: InferComptime = e` / `x: InferComptime[T] = e`, or the marker `syntax.infer_comptime()` + a new declaration.
- The storage form is **determined by the value written in**: when all written values are compile-time values it is a compile-time variable, otherwise it is an ordinary runtime variable.
- Use: a single declaration can both hold compile-time-only values (untyped integers, type values) and be materialized into memory to accept runtime values.
- **Local variable declarations only**: writing `InferComptime` on a parameter is a compile error.

```python
@func()
def infer_literal(x: i32) -> i32:
    n: InferComptime = 3         # the untyped literal -> a compile-time box
    return n + x

@func()
def infer_runtime(x: i32) -> i32:
    n: InferComptime = x         # a runtime value -> memory
    n += 1
    return n

assert infer_literal(1) == 4
assert infer_runtime(1) == 2
```

---

## 4. Expressions and operators

- Arithmetic `+ - * / // % **` (integers and floats), bit operations `| & ^ << >> ~`, unary minus, `not` (only `spy.bool`),
  comparisons `== != < <= > >=` (**chained comparisons are not supported**).
- When both sides are compile-time constants it is folded at compile time; when a runtime value is involved it is computed at runtime.
- Integer `/` is true division (the result defaults to `f64`); integer `//` defaults to floor division; float `//` is `floor(a/b)`.
- `**`: both integer and float exponents are supported. **Float `%` is not implemented** (it is an error).
- Every binary operator has a corresponding compound assignment `a <operator>= b`.
- Conditional expression `a if c else b`: the condition must be `spy.bool`; a compile-time condition keeps only the selected branch.
- Short-circuit `and`/`or`: operands are evaluated in order, and the boolean value decides whether to continue.
- `cast(T, value)`: no-op; `len(x)`: the `__len__` of a tuple/array/slice/struct; `ord(b)`: the encoding of a single-byte byte string.
- Subscript `expr[i]`: arrays/tuples/multi pointers/slices/byte strings/dicts/`$args`/`$kwargs`; `p[...]` dereferences a single pointer;
  a struct can overload the subscript via `__spy_getitemptr__` (see §7.8).
- For operator overloading see §7.2.

```python
@func()
def operators(a: i32, b: i32) -> i32:
    return (a + b) * 2 - a // b + a % b + (a << 1) - (b >> 1)

@func()
def choose(a: i32, b: i32, c: bool) -> i32:
    return a if c else b

@func()
def logic(a: bool, b: bool) -> bool:
    return a and not b

@func()
def true_div(a: i32, b: i32) -> f64:
    return a / b

@func()
def pow_int(a: i32) -> i32:
    return a ** 3

@func()
def pow_float(a: f64, b: f64) -> f64:
    return a ** b

@func()
def byte_ord() -> i32:
    return ord(b'A')

assert operators(7, 3) == 32
assert choose(1, 2, True) == 1 and choose(1, 2, False) == 2
assert logic(True, False) is True
assert true_div(3, 2) == 1.5
assert pow_int(2) == 8
assert pow_float(2.0, 10.0) == 1024.0
assert byte_ord() == 65
```

---

## 5. Statements and control flow

### 5.1 `if` / `elif` / `else`

When the condition is a compile-time constant only the selected branch is kept (the other branch is neither compiled nor analyzed).
A name unpacked by `:=` in the condition is visible in its body. **No truthiness conversion** (the condition must be `spy.bool`).

```python
@func()
def sign(x: i32) -> i32:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

assert sign(5) == 1 and sign(-5) == -1 and sign(0) == 0
```

### 5.2 `while` / `while-else`

The condition is re-evaluated each round; `break` jumps past the loop (skipping `else`), `continue` goes back to the condition.
A `while False:` that is false at compile time generates no loop body.

```python
@func()
def count_down(n: i32) -> i32:
    total: i32 = 0
    while n > 0:
        total += n
        n -= 1
    return total

@func()
def find_limit(limit: i32) -> i32:
    i: i32 = 0
    while i < limit:
        if i * i >= limit:
            break
        i += 1
    else:
        return -1                # only on a natural exit (break skips it)
    return i

assert count_down(3) == 6
assert find_limit(10) == 4
assert find_limit(-1) == -1
```

### 5.3 `for`

`for exprs in iter:` is based on the iterator protocol: first `iter.__iter__()` is taken, then each round `__next__()` is called and the result bound to `exprs`;
on `StopIteration` the loop ends (running `else` first). `range` and `StopIteration` come from `std`.
The loop variable is bound in the loop body's sub-scope and is not visible after the loop.

```python
@func()
def sum_range(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        total += i
    return total

@func()
def skip_odd(n: i32) -> i32:
    total: i32 = 0
    for i in range(n):
        if i % 2 == 0:
            continue
        total += i
    return total

assert sum_range(4) == 6         # 0+1+2+3
assert skip_odd(6) == 1 + 3 + 5
```

### 5.4 `break` / `continue`

`break` leaves the innermost loop; `continue` returns to the loop condition. A `break`/`continue` inside a `try` body within a loop breaks out/continues directly.
(For examples of `break`/`continue` see §5.2, §5.3.)

### 5.5 Exceptions `raise` / `try` / `except`

- `raise E(...)` constructs an exception and throws it.
- `except E as e` binds `e` to the exception value (no copy).
- Clauses are matched in order; a bare `except:` catches all exceptions the function may throw; an except not matched by any clause is dead code (not compiled).
- `try-else` and `try-finally` are **not supported**; at least one except is required; a **bare `raise` is not supported**.
- `@func(exceptions=...)` declares the exception set: a single type, a tuple, or `"infer"` (inferred from the function body).

```python
@struct()
class MyError(Exception):
    code: i32

@func(exceptions=MyError)
def may_fail(x: i32) -> i32:
    if x < 0:
        raise MyError(1)
    return x

@func()
def catch_fail(x: i32) -> i32:
    try:
        return may_fail(x)
    except MyError as e:
        return e.code

assert catch_fail(5) == 5
assert catch_fail(-1) == 1
```

### 5.6 `noreturn`

A function whose value part is the empty type and which throws no exceptions never returns (`noreturn`): calling it is a `noreturn` call, and code after the call site is dead code.
(For an example see `spins`/`maybe_spin` in §1.13.)

### 5.7 `defer`

- `with syntax.defer(): body` registers `body` with the enclosing region and runs it when **leaving that region**.
- Which kinds of exit trigger it is decided by the `defer(flags)` bit mask: `OK` (normal exit), `RAISE` (error exit), `UNWIND` (panic unwinding);
  `okdefer()` = `defer(OK)`, `errdefer()` = `defer(ERR)` (`ERR = RAISE | UNWIND`), and no-argument `defer()` = `ALL`.
- Multiple defers in the same region run in **reverse** declaration order, and inner regions run before outer ones.
- `body` must not jump out of the defer block (a `return`/`break`/`continue`/`raise` that escapes is an error).

```python
@func()
def add_on_exit(p: Ptr[i32]) -> i32:
    with defer():
        p[...] = p[...] + 100    # runs when the region is left
    p[...] = p[...] + 1
    return p[...]

@func()
def use_add_on_exit(x: i32) -> i32:
    v: i32 = x
    add_on_exit(ref(v))
    return v

assert use_add_on_exit(1) == 102
```

### 5.8 `panic` / `catch_unwind`

- `std.core.panic(data)` throws a panic carrying `PanicData`.
- `std.core.catch_unwind(fn)` (`fn` must be a **non-inlined closure** compiled into a runtime function) catches a panic and packs the data into `std.core.UnwindException`,
  thrown along the ordinary exception path (read with `except UnwindException as e: e.data`).
- `may_panic` defaults to `True` for `@func`/`@syntax.closure`. An uncaught panic that unwinds to the Python boundary aborts the process.

```python
@func()
def panic_i32(n: i32) -> i32:
    panic(n)                     # never returns
    return 0

@func()
def caught() -> i32:
    @closure(inline=False)
    def body() -> i32:
        return panic_i32(7)

    try:
        return catch_unwind(body)
    except UnwindException:
        return 42

assert caught() == 42
```

### 5.9 `match`

For a tagged union: `match u:` + `case A():` / `case A() as e:` / `case _:` (the wildcard must come last).
Case guards and subpatterns are not supported (`case A(x):` is an error). (For an example see `tu_match` in §1.8.)

---

## 6. Functions

### 6.1 Definition and specialization

- Register with `@spy.func()`; the first call compiles a **specialization** for the argument types, and a function may have multiple specializations.
- An ordinary (unregistered) Python function called inside a function body is **inlined**; the same holds for undecorated struct methods.
- A parameter's type is determined by: the annotation (after substituting solved generic parameters) > the type inferred from the argument > the type of the default value.

```python
@func()
def add_generic[T: Numeric](a: T, b: T) -> T:
    return a + b

def add_plain(a: i32, b: i32) -> i32:      # an ordinary Python function: inlined
    return a + b

@func()
def use_inline(a: i32, b: i32) -> i32:
    return add_plain(a, b)

assert add_generic(1, 2) == 3              # a Python int marshals as i64
assert add_generic(1.5, 2.5) == 4.0
assert use_inline(2, 3) == 5
```

### 6.2 Parameters

- Positional parameters, keyword parameters, default parameters, and `*args`/`**kwargs` are supported.
- **Positional-only/keyword-only parameters are not supported** (Python `posonlyargs`/`kwonlyargs` are an error).
- Compile-time parameter `x: Comptime` / `x: Comptime[T]`: passed only at compile time, and the argument must be fully known at compile time.
- **A parameter type with no runtime representation ⇒ an implicit compile-time parameter** (type values, byte strings, DST function types, the empty type, etc.):
  it automatically becomes a compile-time parameter without an explicit `Comptime` (e.g. `def sstr(s: bytes)`).
- A `type[X]` parameter: accepts a **type value**.
- `*args`/`**kwargs`: see §8.3.

```python
@func()
def with_default(a: i32, b: i32 = 10) -> i32:
    return a + b

@func()
def comptime_param(x: Comptime[i32], y: i32) -> i32:
    return x + y

@func()
def call_comptime_param(y: i32) -> i32:
    return comptime_param(1, y)   # the first argument is a compile-time value

assert with_default(1) == 11
assert with_default(1, 2) == 3
assert call_comptime_param(2) == 3
```

### 6.3 C variadic declarations

- `@decl_func()` / `@func_type(callconv='c')`: a **non-annotated** `*args` after the last named parameter represents C's `...`.
- A call may pass any number of extra arguments; each is passed by value with its own type, with **C default argument promotion** applied (integers narrower than `int` promote to `int`, `f32`→`f64`).
- An annotated `*args`, `**kwargs`, and function types with the default callconv do not allow C varargs.
- The **definition of a C variadic function** (reading arguments with `va_arg`) is **not yet implemented** (only declaration and calling are possible).

```python
@func()
def c_vararg_call(n: i32) -> i32:
    buf = typed_undefined(Array[u8, Literal[16]])
    s = arr_slice(ref(buf))
    fmt: ConstMultiPtr[u8] = ptr_cast(b'%d', ConstMultiPtr[u8])
    return snprintf(s.ptr, s.length, fmt, n)   # ``n`` goes to ``*args``

assert c_vararg_call(42) == 2
```

### 6.4 `tuple[...]` parameters (flattened passing)

- Parameter annotation `tuple[T1, ...]`: arguments are passed **flattened** by element, and the callee side reassembles a tuple value.
- The function body takes elements with the compile-time integer subscript `x[i]`, nestable.
- Struct elements are **not** flattened (one argument as a whole).
- An unannotated parameter accepts any shape given at the call site, and compile-time reflection can read the shape out.
- Tuple/dict-typed fields (`Foo.t: tuple[i32, i32]`) follow **exactly the same rules** as struct fields.

```python
@func()
def sum_tuple(p: tuple[i32, i32, i32]) -> i32:
    return p[0] + p[1] + p[2]

@func()
def sum_nested(p: tuple[i32, tuple[i32, i32]]) -> i32:
    return p[0] + p[1][0] + p[1][1]

@func()
def use_tuple(a: i32, b: i32, c: i32) -> i32:
    return sum_tuple((a, b, c))

assert use_tuple(1, 2, 3) == 6
assert sum_nested((1, (2, 3))) == 6        # a Python tuple is flattened too
```

### 6.5 Multiple return values

- Return annotation `tuple[T1, T2, ...]` = returning multiple values.
- Call side: `a, b = f()` destructures; `return f()` forwards.
- **Packing**: an all-runtime result can be packed into a **plain tuple variable** `packed = f()` (afterwards it can be destructured again with `x, y = packed`, and `len`, `packed[i]` work);
  the packed tuple keeps the result grouping — one element per result *element*, a nested result being a tuple of its own (so `packed[i][j]` works and `len` is the top-level arity);
  when it contains compile-time leaves (type values/const pointers/byte strings/function values), the packed tuple has no runtime representation and the target must be a `Comptime` variable.
- **Compile-time return values**: a return value fully known at compile time occupies no runtime result.
  A non-inline call takes the value of the callee's **last analyzed `return`** (it does not check whether the return paths agree).
- `tuple[T, ...]` (variadic) is an error at any nesting level.

```python
@func()
def min_max(a: i32, b: i32) -> tuple[i32, i32]:
    if a < b:
        return a, b
    return b, a

@func()
def use_min_max(a: i32, b: i32) -> i32:
    lo, hi = min_max(a, b)     # destructuring
    return lo * 10 + hi

@func()
def packed(a: i32, b: i32) -> i32:
    p = min_max(a, b)          # pack the results into a plain tuple variable
    lo, hi = p
    return lo + hi

@func()
def nested() -> tuple[i32, tuple[i32, i32]]:
    return 1, (2, 3)

@func()
def use_packed_nested() -> i32:
    p = nested()               # the packed tuple keeps the nesting
    return p[1][0] + p[1][1]   # 5

assert min_max(3, 1) == (1, 3)   # the Python side gets a tuple
assert use_min_max(3, 1) == 13
assert packed(3, 1) == 4
assert use_packed_nested() == 5
```

### 6.6 Generics

- PEP 695 `[T]` syntax (requires Python 3.13+); `T` is solved from the argument types.
- Argument types for parameters annotated with the same `T` are unified into a common type, and the arguments are then converted to it.
- **When a return annotation is declared it determines the return type of that specialization**; a recursive function must have a return annotation.
- Direct recursion, mutual recursion, and generic multi-type recursion are all supported.

```python
@func()
def identity[T](x: T) -> T:
    return x

@func()
def fact(n: i32) -> i32:       # recursion needs the return annotation
    if n <= 1:
        return 1
    return n * fact(n - 1)

assert identity(5) == 5
assert identity(1.5) == 1.5
assert fact(5) == 120
```

### 6.7 Closures (= an anonymous struct with `__call__`)

- A `def`/`lambda` may be defined inside a function body; they capture outer variables **by reference** (`move=False` by default) or by value (`move=True`).
- `@syntax.closure(inline=True, exceptions=..., callconv='default', may_panic=True, move=False)`:
  the only allowed nested def decorator; `inline=False` compiles into a runtime function. A `lambda` is always forced inline.
- A closure **is just a value** (its fields are the captures, `__call__` is the body); `f(...)` calls `f.__call__(...)`.
- Because a closure is a struct value, all the old restrictions are lifted: it can be stored in variables/fields, passed as an argument, returned, and put into runtime storage.
- **Restrictions**: no `nonlocal`; **mutually recursive closures are not supported** (see `doc/issues.md` #6); a closure body referencing the outer function's type parameters is not supported.

```python
def apply_inline(f, x: i32) -> i32:        # an inlined plain Python function
    return f(x)

@func()
def closure_capture(x: i32) -> i32:
    @closure()
    def add(y: i32) -> i32:
        return x + y           # captures ``x`` by reference
    return add(10)

@func()
def closure_compiled(x: i32) -> i32:
    @closure(inline=False)
    def add(y: i32) -> i32:
        return x + y
    return add(10)

@func()
def lambda_use(x: i32) -> i32:
    return apply_inline(lambda y: y + 1, x)

assert closure_capture(5) == 15
assert closure_compiled(5) == 15
assert lambda_use(5) == 6
```

### 6.8 `as_func_ptr`

`syntax.as_func_ptr(T, f)`: `T` is only for the Python type checker (the implementation ignores it); `f` must be a **struct that has `__call__` and is a ZST**;
the result is the **`__call__` with the `self` parameter removed**. An ordinary spy function and an extern function can be taken as a constant function pointer **just by using their name**.

```python
@func_type()
class BinOp:
    def __call__(self, a: i32, b: i32) -> i32: ...

@func()
def use_func_ptr(a: i32, b: i32) -> i32:
    @closure(inline=False)
    def add(x: i32, y: i32) -> i32:
        return x + y
    f: ConstPtr[BinOp] = as_func_ptr(BinOp, add)   # the first argument is ignored
    return f[...](a, b)

assert use_func_ptr(2, 3) == 5
```

### 6.9 Recursion

- Direct recursion, mutual recursion, and generic multi-type recursion are all supported; a recursive function's return type must be determinable from its annotation (see §6.6).

---

## 7. Structs

### 7.1 Declaration and fields

- `@spy.struct(extern_c=False, copyable="inherit")` turns an annotated class into a spy struct; the annotated fields form its field set.
- Fields are arranged in declaration order and may be nested (`p.inner.a`); they are readable, assignable, and `+=`-able (at any nesting depth).
- `y = x` copies the struct.
- A value given to a field in the class body is a default value; a field without a default value must be fully provided at construction (the same holds for ZST fields).
- `__init__` **cannot** be defined in the class.
- **`is_comptime` fields** (Zig semantics): they take part in no copy and may be omitted at construction; accessing such a field of a runtime struct returns the default value directly.
- **`is_ref` fields**: logically a reference; readable only, writes do not propagate through.

```python
@struct()
class Point:
    x: i32
    y: i32

@struct()
class Cfg:
    a: i32
    b: i32 = 7

@func()
def use_point(a: i32) -> i32:
    p = Point(a, 3)              # positional args by declaration order
    p.y += 5
    q = Point(y=1, x=2)          # keyword args by name
    return p.x + p.y + q.x

@func()
def use_cfg(a: i32) -> i32:
    c = Cfg(a)                   # ``b`` falls back to its default
    return c.a + c.b

assert use_point(2) == 12
assert use_cfg(1) == 8
```

### 7.2 Methods

- A method decorated with `@spy.func()` in a class is a spy method compiled into a native call; an undecorated method is **inlined** at the call site.
- A method's `self` is by default a writable pointer to itself (`Ptr[Self]`); `@func(self_arg="const ptr")` gives a read-only pointer, `"value"` passes by value.
- Methods are inherited (taken from the whole MRO, with subclass overrides taking precedence).
- Calling through the class name `Foo.m(...)` (including `Foo[i32].m(x)`) does **not pass `self` implicitly**; `@staticmethod` has no receiver.
- `@struct()` may be omitted: an ordinary class can use `Cls.fn(...)` as a namespace.
- Operator overloading: `__add__`/`__sub__`/… (with their reflected `__r*__` and in-place `__i*__` versions), comparisons `__eq__`/`__lt__`/…,
  unary `__neg__`/`__invert__`, `__bool__` as a condition, and `__len__`; for subscript overloading see §7.8.

```python
@struct()
class Point:
    x: i32
    y: i32

    def total(self) -> i32:          # undecorated: inlined
        return self.x + self.y

    @func()
    def doubled(self) -> i32:        # decorated: a native call through ``self``
        return self.total() * 2

@struct()
class Math:
    @staticmethod
    def double(x: i32) -> i32:
        return x * 2

@func()
def use_point(a: i32) -> i32:
    p = Point(a, 3)
    p.y += 5
    return p.doubled()

@func()
def use_static(x: i32) -> i32:
    return Math.double(x)

assert use_point(2) == 20            # (2 + 8) * 2
assert use_static(4) == 8
assert Math.double(5) == 10          # class-name call: no implicit self
```

### 7.3 Construction

- `Point(a, b)` constructs in place: it writes directly into the target place, and nested construction writes in place (no copy).
- Positional arguments follow the field declaration order, keyword arguments go by name; every field must receive a value (a default value may be omitted).
- Default values **do not take part in generic parameter inference**; it is determined only by the provided field values.

```python
@struct()
class Inner:
    v: i32

@struct()
class Outer:
    inner: Inner

@func()
def build(x: i32) -> i32:
    o = Outer(Inner(x))          # the inner struct is built in place
    o.inner.v += 1
    return o.inner.v

assert build(1) == 2
```

### 7.4 Generic structs

- `class Foo[T]` is a template and `Foo[i32]` is a specialization; the `T` in field annotations is substituted in (recursively).
- Generic parameters may be omitted at construction: the specialization is determined by the type of the target place; when the type is unknown it is inferred from the types of the values written into the fields.
- A method call `x.m(...)` ≡ `typeof(x).m(x, ...)`; a method may have its own type parameters; a struct's type parameters can be used as values inside the body.

```python
@struct()
class Box[T]:
    value: T

    @func()
    def get(self) -> T:
        return self.value

@func()
def use_box(x: i32) -> i32:
    b = Box[i32](x)              # explicit specialization
    return b.get()

assert use_box(5) == 5
assert Box[i32](7).value == 7
```

### 7.5 Layout

- The memory layout of a non-`extern_c` struct is decided by the compiler (fields may be reordered and ZST fields take no space); `extern_c=True` uses the C layout
  (keeping declaration order, usable across the C boundary).
- The layout can be queried through `std.mem.layout_of`/`size_of`/`align_of` (see §8.8).

### 7.6 Copyability and argument passing/return

- `copyable=True|False|"inherit"` (the default `"inherit"` = copyable only if every field is copyable); a DST is never copyable.
- A non-copyable value can be constructed in place, passed by reference, returned through the result location, and have its fields read/written through an address, but **cannot be read out as a whole**.
- **Arguments are passed by value** (semantically always a copy); non-copyable ones are forced to by reference.
- **Return**: a small struct is returned by value; a large struct is returned through the result location; non-copyable ones are forced to return through the result location.

```python
@struct(copyable=False)
class Handle:
    id: i32

@func()
def handle_id(h: Handle) -> i32:     # a non-copyable value is passed by reference
    return h.id

@func()
def make_handle(x: i32) -> Handle:   # returned through a result pointer
    return Handle(x)

assert handle_id(make_handle(7)) == 7
```

### 7.7 Use from the Python side

- `Point(1, 2)` constructs an instance (omitted fields take their default values; a generic specialization is inferred from the field values or given explicitly as `Foo[i32](...)`).
- Instance fields are readable and writable; methods can be called (`p.m()` passes `self` implicitly; `Foo.m(p, ...)` does not).
- An instance can be an argument or a return value across the boundary; the layout is handled by the boundary layer.
- A nested field is a **view** into the parent instance (writing the child field writes the parent instance).

```python
@struct()
class Pt:
    x: i32
    y: i32

    def total(self) -> i32:
        return self.x + self.y

@func()
def swap_xy(p: Pt) -> Pt:
    return Pt(p.y, p.x)

p = Pt(1, 2)
assert p.x == 1 and p.y == 2         # a struct instance crosses the boundary
assert p.total() == 3                # a bound method
assert Pt.total(p) == 3              # class-name call passes ``self`` explicitly
q = swap_xy(p)
assert q.x == 2 and q.y == 1
```

### 7.8 Subscript overloading (`__spy_getitemptr__`)

A struct value can be subscripted like an array (`expr[i]`): define the method `__spy_getitemptr__(self, index) -> Ptr[T]`
(or `ConstPtr[T]` / `MultiPtr[T]` / `ConstMultiPtr[T]`), which **returns a pointer to the element** — that is, the place the `i`-th element lives in.

- `expr[i]` is exactly **the place the method's return value points to**: readable, assignable, `+=`-able, and you can keep taking fields/subscripts.
- The method may be undecorated (inlined at the call site) or compiled into a runtime method with `@func()`; it must return a **pointer**, and returning a value is an error.
- The class usually also defines a pair of `__getitem__`/`__setitem__` (placed in `if TYPE_CHECKING:`) **only so that a Python type checker** accepts
  the `expr[i]` form — the compiler does not call them (at compile time a subscript goes through `__spy_getitemptr__`).
- `std.SlicePtr`/`std.ConstSlicePtr`/`std.vec.Vec` use it to implement subscripts.

```python
@struct()
class IntArray:
    ptr: MultiPtr[i32]

    def __spy_getitemptr__(self, index: i64) -> MultiPtr[i32]:
        return self.ptr + index          # the place of the i-th element

    if TYPE_CHECKING:
        # only so a type checker accepts ``v[i]``; never called by the compiler
        def __getitem__(self, index: int) -> i32: ...
        def __setitem__(self, index: int, value: i32) -> None: ...

@func()
def subscript_use(x: i32) -> i32:
    a = array(x, x + 1, x + 2)
    v = IntArray(arr_slice(ref(a)).ptr)
    v[2] = 100                           # writes through the returned place
    return v[0] + v[1] + v[2]

assert subscript_use(1) == 1 + 2 + 100
```

---

## 8. Compile time (comptime)

### 8.1 Type probes and compile-time logging

- `spy.typeof(x)`: queries the static type (returning a type value); it is a **type probe**: it only types and has no runtime behavior; spy functions referenced in it are triggered to compile. If a probed call throws and leaves the probe, it is an error.
- `spy.compile_log(...)`: prints at compile time (no runtime action); bytes are rendered as text and runtime values as `<T>`.
- `spy.as_`: usable only at the Python calling boundary (calling it inside a function body is an error).

```python
@func()
def probe(x: i32) -> i32:
    if typeof(x) == i32:         # a type probe: no code, no runtime effect
        return 1
    return 0

@func()
def logged(x: i32) -> i32:
    compile_log('compiling logged')   # prints at compile time only
    return x + 1

assert probe(1) == 1
assert logged(1) == 2
```

### 8.2 Compile-time parameters (supplement)

- A type with no runtime representation as a parameter ⇒ an implicit compile-time parameter (see §6.2).
- **const only**: a pointer to a compile-time constant can be passed as a `Comptime` argument to a **non-inlined** function (passed by identity, aliasable);
  a non-const compile-time pointer cannot cross a non-inline boundary (see §8.5).

```python
@func()
def read_const(p: Ptr[i32]) -> i32:
    return p[...]                # the callee aliases the global constant

@func()
def use_const_ptr() -> i32:
    const_x: ConstComptime[i32] = 99
    return read_const(ref(const_x))

assert use_const_ptr() == 99
```

### 8.3 `*args` / `**kwargs`

- In the function body `args` collects positional arguments and `kwargs` collects keyword arguments by name.
- `len(args)`, `args[i]` (compile-time integer subscript), `kwargs[b'name']` (compile-time byte string name).
- Element/value types may be annotated or inferred; `*args`/`**kwargs` elements do not support `type[X]`; when needed, traverse with `syntax.unroll()`.
- **Call-site unpacking (`f(*args, **kwargs)`) is not yet implemented**: `**kw` is silently ignored and `*x` is an error (see `doc/issues.md` #1).

```python
@func()
def sum_all(*args: i32) -> i32:
    total: i32 = 0
    unroll()
    for i in range(len(args)):
        total += args[i]
    return total

@func()
def named_sum(**kwargs: i32) -> i32:
    return kwargs[b'a'] + kwargs[b'b']

assert sum_all(1, 2, 3) == 6
assert named_sum(a=1, b=2) == 3
```

### 8.4 `dict[str, T]` parameters

- A `**kwargs` parameter / `dict[str, T]` parameter passes arguments flattened by name; access with `len(kwargs)`, `kwargs[b'name']` (compile-time byte string names only, **runtime key lookup is not supported**).
- Because it has a runtime representation, a dict value can be stored in ordinary variables/fields and passed by value or by reference.
- The `dict[str, T]` annotation itself is not yet parsed.

### 8.5 Compile-time pointers

- Address-of/offset: `ref(a)` (`a` is a compile-time place) yields a compile-time pointer; `p[...]` reads/writes the place it points to.
- A **const pointer** can cross a non-inline boundary (as a `Comptime` argument, captured by a compiled closure, as a non-inline function's return value); it is passed by **identity** and is aliasable.
- A **non-const compile-time pointer** **materializes a copy** when crossing a non-inline boundary: the callee's writes do not affect the original place.
- A compile-time pointer obtained through `ptr_cast` cannot be read/written through.
- A pointer to a compile-time constant and a byte string constant are both automatically materialized as global constants.

```python
@struct()
class Small:
    a: i32
    b: i32

@func()
def aliasing() -> i32:
    c: Comptime[Small] = Small(1, 2)
    p: Comptime = ref(c)         # a compile-time pointer into the box
    p[...].a = 10                # writes the aggregate (aliases the box)
    return c.a + c.b

assert aliasing() == 12
```

### 8.6 Compile-time aggregates and loops

- For an aggregate fully known at compile time, all field/element reads and writes are folded at compile time; a whole copy is by value.
- Compile-time `if`: the condition must be fully known at compile time; the unselected branch is neither compiled nor analyzed (a `compile_error` in it does not trigger).
- Compile-time loop `syntax.unroll()`: the marker applies to the immediately following `while`/`for`; the condition must be a compile-time value; the loop body must advance compile-time state,
  otherwise exceeding `max_loop_unroll` (1024 by default) is an error. `syntax.unroll()` must come immediately before the marked loop.

```python
@struct()
class Small:
    a: i32
    b: i32

@func()
def comptime_struct() -> i32:
    c: Comptime[Small] = Small(1, 2)
    c.a = c.a + 10               # a compile-time field write
    return c.a + c.b

@func()
def unrolled_sum() -> i32:
    total: i32 = 0
    unroll()
    for i in range(4):
        total += i
    return total

assert comptime_struct() == 13
assert unrolled_sum() == 6
```

### 8.7 Other compile-time builtins

- `std.core.undefined()`: undefined of any type, typed according to its type when written into a target; undefined as a condition is an error.
- `std.core.compile_error(msg)`: aborts compilation with a compile-time byte string message; a non-byte-string/runtime message is an error; a dead branch does not trigger.
- `std.core.coerce(value, T)`: concretizes value into type `T` (the same conversion as store).
- `std.core.compile_option(name)`: **declared but not implemented** (calling it is an error).
- `std.core.compile(fn)`: manually compiles a **non-generic** function value (precompiles its default specialization).
- `std.core.bitcast(value, T)` / `std.int.truncate(value, T)`: same-bit-width reinterpretation of a scalar / integer narrowing.

```python
@func()
def dead_branch() -> i32:
    comptime()
    c = False
    if c:
        compile_error('never compiled')   # a dead branch is not compiled
    return 1

@func()
def pick(c: bool, a: i32, b: i32) -> i32:
    v: i32 = undefined()
    if c:
        v = a
    else:
        v = b
    return v

@func()
def narrowed(x: i32) -> i8:
    return coerce(x, i8)

@func()
def same_bits(x: f32) -> u32:
    return bitcast(x, u32)

@func()
def truncated(x: u32) -> u8:
    return truncate(x, u8)

assert dead_branch() == 1
assert pick(True, 1, 2) == 1 and pick(False, 1, 2) == 2
assert narrowed(128) == -128
assert same_bits(1.0) == 0x3F800000
assert truncated(300) == 44
```

### 8.8 Reflection

- `std.reflect.type_info(type) -> TypeInfo` / `reify(info) -> type`;
  variants `IntType`/`PointerType`/`ArrayType`/`StructType`/`TaggedUnionType`/`OptionType`/`TupleType`/`StrDictType`/`ComplexType`/`UnionType`.
- `reflect.StructField` has the `is_comptime` / `is_ref` markers; `reflect.StructType.head` (`None` for non-generic).
- `reify`ing a struct **rebuilds** it (not identity-equal to the original struct, but with the same fields/field types).
- `std.mem.layout_of`/`size_of`/`align_of`: also accept tuple/dict types.

```python
@func()
def int_bits(t: type[i32]) -> i32:
    comptime()
    info = type_info(t)
    if isinstance(i := info, IntType):
        return i.bits
    return 0

@func()
def reified() -> i32:
    comptime()
    t = reify(type_info(i32))
    if t == i32:
        return 1
    return 0

@func()
def int_size() -> i32:
    return coerce(size_of(i32), i32)

assert int_bits(i32) == 32
assert reified() == 1
assert int_size() == 4
```

### 8.9 Compile-time byte strings

- `bytes` exists only at compile time (no runtime representation); a `str` literal is encoded into `bytes` (`str` is not allowed inside a spy function).
- It can be subscripted `b[i]` and sliced `b[a:c]` (lower bound defaults to 0, upper bound to the length, step must be 1, out of bounds is an error); `ord(b)` takes the encoding of a single-byte byte string.
- Materialized at runtime: `ptr_cast(s, ConstMultiPtr[u8])` or `std.core.sstr(s) -> ConstSlicePtr[u8]` (the length excludes the NUL).
- Byte strings with the same content share a global constant; a local holding a byte string must be a compile-time variable.

```python
@func()
def byte_at() -> i32:
    s: Comptime = b'ABC'
    return ord(s[1])

@func()
def second_byte() -> i32:
    s: Comptime = b'xy'
    p: ConstMultiPtr[u8] = ptr_cast(s, ConstMultiPtr[u8])
    return coerce(p[1], i32)

@func()
def str_len() -> i32:
    s = sstr(b'hello')
    return coerce(s.length, i32)

assert byte_at() == 66
assert second_byte() == 121
assert str_len() == 5
```

---

## 9. Standard library

| Module | Contents |
|---|---|
| `std.core` | `Numeric`, `StopIteration`, `slice`, `range`, `SlicePtr`/`ConstSlicePtr`, `arr_slice`/`const_arr_slice`/`const_arr_ptr`, `NullException`, `unwrap`/`unwrap_or`, `undefined`, `coerce`/`typed_undefined`, `const_array`, `compile_error`, `sstr`, `compile`, `bitcast`, `DestructorFn`/`OpaqueWithDestructor`/`PanicData`/`deinit_panic_data`/`UnwindException`, `panic`, `catch_unwind`, `compile_option` (not implemented) |
| `std.reflect` | `TypeInfo` and its variants, `StructField`, `type_info`, `reify` |
| `std.int` | `truncate` (builtin), `IntCastError`, `valid_range`, `try_int_cast`/`int_cast` |
| `std.mem` | `Layout`, `layout_of`/`size_of`/`align_of`, `AllocError`, `Allocator` (protocol), `AllocatorVtable`, `DynamicAllocator`, `CAllocator` |
| `std.c` | `malloc`/`free`/`realloc`/`puts`/`snprintf` (extern declarations), `CError`, `FILE`, `fopen`/`fclose`/`fread`/`fwrite` |
| `std.io` | `IOError`/`PartialWriteError`/`PartialReadError`, `Writable`/`Readable` (protocols) |
| `std.vec` | `Vec[T, A]` dynamic array |
| `std.atomic` | `MemoryOrder`/`RmwOp`, `atomic_load`/`atomic_store`/`atomic_rmw`/`atomic_cmpxchg`/`atomic_fence`, convenience wrappers |
| `std.openmp` | `load_openmp_runtime`, `parallel_loop` |

### 9.1 Atomic operations

- The operand (pointee) must be an integer, a float, or a pointer (`bool` is not supported; the integer width must be a power of two bytes).
- `atomic_cmpxchg` does not accept a float pointee; it returns `(old value, whether swapped)`, and the failure ordering cannot be `RELEASE`/`ACQ_REL` and cannot be stronger than success.
- The operations of `atomic_rmw` are restricted by the kind of pointee (all of them for integers; only `XCHG`/`ADD`/`SUB` for floats; only `XCHG` for pointers); it also accepts **complex** pointees.
- Each operation takes a `MemoryOrder`; all except `atomic_fence` have a `volatile` switch.
- Convenience wrappers: `atomic_swap`, `atomic_fetch_add/sub/and/or/xor`.

```python
@func()
def bump(p: Ptr[i32]) -> i32:
    return atomic_fetch_add(p, 1)     # returns the old value

@func()
def use_bump(x: i32) -> i32:
    v: i32 = x
    old = bump(ref(v))
    return old * 10 + v

assert use_bump(5) == 56
```

### 9.2 OpenMP

`std.openmp.parallel_loop(length, kernel)` implements `#pragma omp parallel for` (static schedule).
Before use, `load_openmp_runtime()` is required (idempotent, loads `libomp`).

---

## 10. The Python ↔ spy calling boundary

- Mapping from Python values to spy types: `bool`→`bool`, `int`→`i64`, `float`→`f64`, `complex`→`c128`,
  `bytes`→a compile-time byte string, `None`→`Null`, `syntax.Ptr[T]`→a pointer; use `spy.as_(value, T)` to specify the type explicitly.
- A struct instance supports field reads/writes, method calls, and view semantics; pointers are non-null.
- `Option[T]` and tagged unions are **unwrapped** when going back to Python: `Option[T]` is a `T` value or `None`, and a union is the value of its concrete variant.
- Complex numbers ↔ Python `complex`; integer enums ↔ Python enum members.
- A spy function that may raise throws a struct instance inheriting from `Exception` when called from Python (`except Exception as e` can catch it and read its fields).
- Multiple return values are packed into a Python `tuple` (nested results nest `tuple`s), and an aggregate returned by value is unpacked into an instance.
- Compile-time results: a scalar with a natural Python mirror is converted to `int`/`float`, and the rest are handed over as compiler objects as-is.
- **Arrays currently cannot cross the Python boundary** (reading/writing them is an error directly).

```python
@struct()
class Pt:
    x: i32
    y: i32

@func()
def swap_xy(p: Pt) -> Pt:
    return Pt(p.y, p.x)

@func()
def id64(x: u64) -> u64:
    return x

p = Pt(1, 2)                          # a struct instance crosses the boundary
assert p.x == 1 and p.y == 2
q = swap_xy(p)
assert q.x == 2 and q.y == 1
assert id64(spy_as(2 ** 40, u64)) == 2 ** 40   # spy.as_ picks an explicit type
```

---

## 11. Implemented / unimplemented boundary

**Implemented (highlights)**: see the sections above — scalars/pointers/arrays/slices/Option/tagged unions/union/complex/enums/structs;
multiple return values; generics (functions and structs); closures (i.e. anonymous structs); `Comptime`/`ConstComptime`/`InferComptime`;
compile-time pointers (including const across a non-inline boundary); compile-time aggregates/loops/reflection; `defer`; exceptions and `noreturn`;
panic/`catch_unwind`; atomic operations; OpenMP; Python-side interoperability (structs/enums/option/union/exceptions/multiple return values).

**Not implemented / known limitations**:

- Assignment is only `=` (including tuple unpacking) and `+=`; **chained assignment `a = b = e` only handles the first target** (silently).
- Positional-only/keyword-only parameters, chained comparisons, single-pointer subscript `p[i]` (a single pointer only has `p[...]`).
- Call-site unpacking `f(*args, **kwargs)` (`**kw` is silently ignored, `*x` is an error).
- The **definition** of C variadic functions (`va_arg`).
- Tuple `+=` / `+` (removed); float `%`; byte-string operations other than subscript/slice/`ord`/`ptr_cast`/`sstr`.
- Arrays: arrays with a runtime length, conversions between arrays, arrays crossing the Python boundary.
- `dict[str, T]` annotation parsing; pattern matching an option; closure `nonlocal`; mutually recursive closures; a closure referencing the outer type parameters.
- Declared "compile-time-only" return types (`-> bytes` / `-> type` etc., see `doc/issues.md` #7).
- `std.core.compile_option`: declared but not implemented.
- Destructive move semantics (`copyable=False` only means non-copyable; no move checking is done).

> For individual implementation defects and minimal reproductions see `doc/issues.md`.

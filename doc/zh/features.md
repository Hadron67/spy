# spy 特性清单

本文档列举 **当前代码**（`spy/spy/compiler` + `spy/spy/std`）已经实现的语言特性与可观察行为。

> 说明：本文按「语言表面」组织——即一个 spy 程序员能写出的东西及其语义。
> 编译器内部各阶段如何实现这些行为见 `doc/implementation.md`。

---

## 0. 总览

`spy` 把普通 Python 函数编译成原生机器码。用 `@spy.func()` 注册函数后，首次调用即按实参类型
编译出该类型的一个**特化**（specialization），之后是原生调用，不再经过 Python。

函数必须定义在真实源码文件里（编译需要函数的源码）。

编译流程与各阶段实现见 `doc/implementation.md`。

### 0.1 示例约定

本文所有 ```python 代码块都是**可运行**的完整示例：它们假定下面这段**共享导入**。
`spy` 的公开入口就是 `spy` 包本身，因此真实代码可以直接 `import spy` 并使用 `spy.func`、`spy.u64`、`spy.typeof` 等；
下面的示例为了保持函数体简洁，把这些常用名字直接导入：

```python
from spy import (
    func, struct, decl_func, func_type, Enum,
    i8, i16, i32, i64, u8, u16, u32, u64, u0, usize, isize,
    f32, f64, c64, c128,
    c_char, c_uchar, c_short, c_ushort, c_int, c_uint,
    c_long, c_ulong, c_longlong, c_ulonglong,
    Ptr, ConstPtr, MultiPtr, ConstMultiPtr, Option,
    typeof,
)
from spy.compiler.syntax import (
    Array, Opaque, array,
    Comptime, ConstComptime, InferComptime,
    comptime, const_comptime, infer_comptime, unroll,
    ref, closure,
    defer, okdefer, errdefer, OK, RAISE, UNWIND, ERR, ALL,
)
from spy.std import (
    ConstSlicePtr, SlicePtr, Numeric, StopIteration,
    arr_slice, const_arr_slice, const_array, range, slice, sstr, undefined,
)
from spy.std.core import (
    panic, catch_unwind, UnwindException, coerce, typed_undefined, bitcast,
    ptr_cast, as_func_ptr, unwrap, unwrap_or, compile_error, compile_log, lub_type,
    int_to_float_unchecked, float_to_int_unchecked,
)
from spy.std.c import snprintf
from spy.std.math import sin, cos, tan, asin, acos, atan, atan2, sqrt, cbrt, pow, exp, log
from spy.std.mem import Layout, layout_of, size_of, align_of
from spy.std.int import (
    IntCastError, FloatCastError, int_cast, try_int_cast, truncate, valid_range,
    int_to_float, try_int_to_float, float_to_int, try_float_to_int,
)
from spy.std.float import float_truncate
from spy.std.atomic import (
    MemoryOrder, RmwOp,
    atomic_load, atomic_store, atomic_rmw, atomic_cmpxchg, atomic_fence, atomic_fetch_add,
)
from spy.std.reflect import TypeInfo, IntType, FloatType, type_info, reify
from typing import TYPE_CHECKING, Literal, Never, NoReturn
```

> 代码块可以用 `tools/run_doc_examples.py` 批量运行验证：脚本把每个 ```python 块物化成一个真正的
> `.py` 文件（`spy` 需要源码）后执行，第一个可运行块作为共享导入前置到其余每个块。
> 运行：`python3 tools/run_doc_examples.py doc/features.md`。

---

## 1. 类型系统

### 1.1 整数与浮点

- 定宽整数 `u0`..`u64`、`i0`..`i64`（覆盖 0 到 64 位的全部宽度）；`u0`/`i0` 是零位整数（ZST）。
- 浮点 `f32/f64`。
- `spy.usize` / `spy.isize`：按目标平台指针宽度决定的整数。
- C 整数类型 `c_char/c_uchar`、`c_short/c_ushort`、`c_int/c_uint`、`c_long/c_ulong`、`c_longlong/c_ulonglong`：
  宽度（及 `c_char` 的有符号性）由目标平台决定（例如 `c_long` 在 Windows 为 32 位、常见 64 位 Unix 为 64 位）。
- 未定型的整数字面量：字面量 `1` 起初没有具体类型，由它被写入的位置或与它配对的类型决定。
- 整数子类型按范围包含：`i8` 是 `i16` 的子类型，`u8` 不是 `i8` 的子类型。整数与浮点之间：整型是浮点的子类型，当且仅当该浮点能**无损**表示它的整个范围（`i16`/`u8 → f32`、`i32`/`u32` → `f64`；`i32` → `f32`、`i64`/`u64` → `f64` 都不行）。编译期整数则按其**值**判断。

```python
@func()
def arith(a: i32, b: i32) -> i32:
    return a + b * 2 - (a // b) + (a % b)

@func()
def to_float(a: i32, b: i32) -> f64:
    # 整数能无损转成 f64，真除法可直接用（见 §4）
    return a / b

@func()
def to_float_explicit(x: u64) -> f64:
    # u64 超出 f64 的精确整数范围：必须显式转换（超范围会 panic）
    return int_to_float(x, f64)

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

### 1.2 布尔

`spy.bool`。没有「真值转换」：`if`/`while`/`not`/`and`/`or` 的条件都必须是 `spy.bool`
（对象提供 `__bool__` 时也会被调用，见 §7.2）。

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

### 1.3 指针

`syntax.Ptr[T]`（可变单指针）、`ConstPtr[T]`（只读单指针）、`MultiPtr[T]` / `ConstMultiPtr[T]`
（多指针，可像数组一样下标）。

- 单指针写 `p[...]`（解引用，C 的 `*p`），多指针写 `p[i]`（第 i 个元素的地址——本身就是单指针，元素的值再 `p[i][...]`）。
- 指针是**非空**的；`Option[Ptr[T]]` 才是可空指针。
- 可变指针可隐式转 const（`Ptr[T]` → `ConstPtr[T]`）；反之不行。
- 指向数组的指针（`*[N]T`）可当成元素的多指针。
- 指向 ZST 的指针是「dangling 但非空」，不是空指针。
- `syntax.ref(a)` 取地址；`std.core.ptr_cast(p, T)` 重解释指针类型（不做检查）；`p[...]` 解引用。

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

### 1.4 数组与切片

- `syntax.Array[T, N]`：`N` 个 `T` 排成一行；长度是**值**类型参数，注解里用 `Literal[N]` 写。
  长度可以缺省为 `Array[T, None]`（**无长度数组**，DST）。
- `syntax.array(a1, a2, ...)` 就地构造数组：长度取实参个数，元素类型取目标位置的类型、否则取所有元素的共同类型；
  还有一个**只为类型检查器服务**的 `length=` 关键字（实际长度以实参个数为准）。
- `a[i]` 是第 i 个元素的**左值**（可读/写/`+=`）；编译期常量下标检查越界，运行时下标不检查。
- 切片类型在 `std`：`std.SlicePtr[T]` / `std.ConstSlicePtr[T]`（结构体 `{ptr, length}`）。
  `std.arr_slice(p)` / `std.const_arr_slice(p)` 把指向数组的指针变成整段切片；
  多指针切片 `p[a:b]` 得到 `SlicePtr(p+a, b-a)`（缺省下界 0、上界长度、step 必须为 1）。
- 切片可迭代（元素值）与 `refs()`（元素指针）；越界会 `panic`。
- `std.core.const_array(elem, length)`：`length` 个相同元素的数组（元素表达式只求值一次，可为运行时值）。
- 数组**不能跨 Python 调用边界**（见 §10），因此下面用 spy 函数间接调用。

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

`syntax.Option[T]`（等价 `T | None`）：

- 值要么是一个 `T`，要么是 `None`。
- `T` 与 `None` 都能自动转成 `Option[T]`；同一个位置同时收到 `T` 与 `None` 时类型即 `Option[T]`。
- 判断与解包：`expr is None` / `expr is not None`（对任意类型可用：非 option 值永不缺失，判断恒为 `False`）；海牙语法 `(name := expr) is not None` 把 `name`
  绑定到内层值的可写位置（该形式要求 option）；一般 `(name := expr)` 把 `name` 绑定到 option 本身。
- **尚未实现**：模式匹配 option；借用式表示下还存在误判缺失的边界情况（见 `doc/implementation.md`）。

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

### 1.6 元组（tuple）

`tuple[T1, T2, ...]`：

- 当所有元素都有运行时表示时，元组是一个普通运行时聚合值（可拷贝 ⇔ 所有元素可拷贝）。
- 在**变量/字段/表达式值**位置是「一个元组值」：`t[i]` 是第 i 个元素的左值（`i` 为编译期整数，越界/负/运行时下标报错），
  `len(t)` 是编译期整数，可整体拷贝（`b = a`）。
- 在**形参/返回注解**位置被特殊解释：形参 `tuple[...]` 是**铺平传参**（见 §6.4），返回 `tuple[...]` 是**多返回值**（见 §6.5）。
- 无运行时表示的元素（类型、字节串、DST 函数类型）使整个元组只能是编译期值。
- **不再支持**元组 `+=` / `+`（元组现在是普通运行时聚合，元数固定）。

```python
@func()
def tuple_use(a: i32, b: i32) -> i32:
    t = (a, b)                   # a runtime tuple value
    t[1] = t[1] + 1
    return t[0] + t[1]

@func()
def tuple_len(a: i32, b: i32, c: i32) -> usize:
    t = (a, b, c)                # a runtime tuple value
    return len(t)                # a compile-time integer

assert tuple_use(3, 4) == 3 + 5
assert tuple_len(1, 2, 3) == 3
```

### 1.7 无标签 union（`UnionType`）

异常的 payload 是一个**无标签并集**（union）：同一块存储按不同变体解读，各变体共用同一空间。
union **值之间不能转换**，只能重解释存储。不占存储的 union（无变体或变体全为 ZST）只是编译期值。
它没有直接的源码语法，只在异常 payload 与反射中出现（见 §5.5、§8.8）。

### 1.8 带标签 union（tagged union）

`A | B | C`：

- 值是恰好其中一个变体，附带变体标签（变体顺序属于类型身份）。
- 判断/解包：`isinstance(u, A)`；海牙 `isinstance(e := u, A)` 把 `e` 绑定到变体值（可写）；
  `match u:` + `case A():`（测变体）/ `case A() as e:`（绑定变体值）/ `case _:`（通配，必须最后）。
- 子集 union 可转成超集 union；`A | A` 去重为单变体；
  tag 运行时才知道的 union 一律是普通运行时值。

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

### 1.9 复数

复数类型 `spy.c64` / `spy.c128`：实部与虚部各是一个浮点元素。

- Python `complex` 映射到 `c128`；**浮点**实数有一条隐式 `T → Complex[T]` 转换（虚部 0），`Complex[f32]` 可扩宽到 `Complex[f64]`。整数在能**无损**转成元素浮点时也走同一条转换（`i32 → c128`；编译期常量按值检查，`z: c128 = 3` 可写），不能无损转换的整数（如 `i64`）要先显式转成浮点。
- 支持 `+ - * /`（逐分量）与字段 `z.real` / `z.imag`。
- `std.reflect.type_info` 给出 `ComplexType` 反射变体。

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

@func()
def widen_int_literal() -> c128:
    z: c128 = 3                  # 编译期整数常量按值检查，仍可转
    return z

@func()
def widen_int(r: i32) -> c128:
    return r                     # i32 -> f64 无损，因而 i32 -> c128 也隐式

assert cmul(1 + 2j, 3 + 4j) == (1 + 2j) * (3 + 4j)
assert cparts(3 + 4j) == 7.0
assert widen(2.5) == complex(2.5, 0.0)
assert widen_int_literal() == complex(3, 0.0)
assert widen_int(3) == complex(3, 0.0)
```

### 1.10 整数枚举

继承 `spy.Enum`（标准 `enum.Enum`）、成员值为整数的类声明一个 spy 类型：

- 可直接作形参/返回值/字段注解；成员是编译期值。
- **不是一个整数**：不参与算术/位运算、不与 `int` 或另一个枚举比较/转换；只支持同枚举之间的 `==`/`!=`（比较成员身份）。
- 支持方法（`@func()` 编译、未装饰内联、`@staticmethod`）；方法的 `self` **默认按值**（可用 `@func(self_arg="ptr"/"const ptr")` 改按引用）。
- Python 边界：成员入参识别为枚举值，枚举返回值/字段还原为 Python 成员。

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

### 1.11 结构体（struct）

见 §7。

### 1.12 ZST（零大小类型）与 DST

- **ZST**：`-> None` 的 `void`、`u0`、字段全为 ZST 的结构体、元素全为 ZST 的元组/`dict[str, T]`、元素为 ZST 或长度为 0 的数组、`Null`、`undefined`
  等。ZST 不占运行时空间，但 ZST 结果照常交付单位值（`sval.Type.get_unit_value`）。
- **DST（动态大小类型）**：DST 函数类型、`Opaque`、无长度数组；只有**指向它的指针**是值，不能按值传递/返回/整体读取。
  结构体里第一个 DST 字段是弹性数组，后续 DST 字段不占位。
- **无运行时表示的类型（纯编译期）**：类型值（`type`）、字节串（`bytes`）、DST 函数类型、空类型（`Never`）。
- 指向 ZST 的指针是非空的（但不指向实际内存）。

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
assert zero_bit(u0(0)) == 1
assert take_opaque(1, False) is False
```

### 1.13 空类型（`Never`）

空类型（`Never`）：一个值都没有。`typing.Never`/`NoReturn` 映射到它，用于声明永不返回的函数。
它是「函数体从不交付值」的函数的结果类型：体内不能有 `return`，也不能落穿到末尾。
空类型 + 不抛异常 ⇒ `noreturn` 函数（见 §5.6）。

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

## 2. 运行时值与编译期值

一个值要么是**运行时值**（有运行时表示），要么是**编译期值**（只在编译期存在）：

- 编译期值涵盖：标量常量、类型值、编译期聚合值、编译期 Option/tagged union、`Null`/`Void`/undefined、
  编译期指针，以及普通 Python `int/float/complex/bytes/bool`；此外还有函数值、不用 `@func` 装饰的
  Python 函数（`PyFunction`，见 §6.10）等编译期对象。

**关键点**：不存在「以运行时值形式存在的编译期值」。任何一个组成部分是运行时值的聚合，
**整体就是运行时值**，不能存进 `Comptime` 变量/参数（要持有它就用普通变量）。

编译期变量的具体存储机制见 `doc/implementation.md`。

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

## 3. 变量与声明

### 3.1 普通变量

- `name = expr`：名字已绑定时赋值；完全没绑定过的名字才**声明**一个块局部变量。
  声明不逃出所在块；对**外层**变量的赋值写的就是外层变量。
- 无注解的普通声明 / 解构目标得到普通运行时变量。
- 局部变量需要有能落到运行时的类型：`x = 1`（未注解、未定型整数字面量）报错，需写成 `x: i32 = 1` 或 `x: Comptime = 1`。
- 支持 `name <算符>= expr`（全部二元算符）与元组解包赋值 `a, b = e`（可嵌套）。
  解构**按下标顺序读源、写目标**，因此 `a, b = b, a` 是已知缺陷（右侧读到可能已被覆盖的值）。
- 赋值仅支持 `=` 与 `+=`（无链式赋值 `a = b = e`）。

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

### 3.2 类型标注的局部变量

- `name: T = expr`：声明一个新的块局部变量并给定类型 `T`（值转换到 `T`）。
  标注即声明：同一个名字不能被标注两次。
- 标注里的类型本身是编译期值表达式（可命名类型参数）。

```python
@func()
def annotated_local(x: i32) -> i64:
    y: i64 = x                   # the value is coerced to the declared type
    return y + 1

assert annotated_local(4) == 5
```

### 3.3 编译期变量

- `x: Comptime = e` / `x: Comptime[T] = e`：**编译期变量**。
  写入的值必须**完全编译时已知**；把运行时值写进 `Comptime` 变量是编译错误。
- 语句标记 `syntax.comptime()` + 紧随其后的一条**新声明**等价于 `Comptime` 标注。
  该标记必须**紧接在被标记的声明之前**，且只能标记**新声明**（不能标记对已绑定名字的赋值）。
- 一个 `Comptime` 变量可持有：类型值、编译期标量、编译期聚合、编译期元组/字典、编译期 Option/union、编译期指针、函数值等。

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

assert comptime_holds_type(i32(1)) == 1
assert comptime_holds_type(f64(1.5)) == 0
assert comptime_marker_typed() == 1
```

### 3.4 编译期常量 `ConstComptime`

- `x: ConstComptime = e` / `x: ConstComptime[T] = e`（必须带初值），或标记 `syntax.const_comptime()` + 一条新声明。
- 初始化后即只读；之后写它是编译错误。
- 它是「编译期全局常量」的来源：初始化后 `ref(x)` 得到一个指向全局常量的指针；**只读常量数组**（或其它只读常量聚合体）也可以直接用运行期下标读取——该访问会把它物化成一个全局常量。可变编译期变量没有运行期身份，仍需编译期下标。

```python
@func()
def static_read() -> i32:
    const_x: ConstComptime[i32] = 42
    p = ref(const_x)             # automatically becomes a global constant
    return p[...]

@func()
def const_array_read(i: usize) -> i32:
    arr: ConstComptime[Array[i32, Literal[3]]] = array(10, 20, 30)
    return arr[i]                # a runtime index materializes it into a global constant

assert static_read() == 42
assert const_array_read(1) == 20
```

### 3.5 `InferComptime`（推断存储）

- `x: InferComptime = e` / `x: InferComptime[T] = e`，或标记 `syntax.infer_comptime()` + 一条新声明。
- 存储形式**由写进去的值**决定：写入的值全是编译期值时为编译期变量，否则为普通运行时变量。
- 用途：一条声明既能持有编译期专属值（未定型整数、类型值），又能落内存接受运行时值。
- **仅限局部变量声明**：形参写 `InferComptime` 是编译错误。

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

## 4. 表达式与运算符

- 算术 `+ - * / // % **`（整数与浮点）、位运算 `| & ^ << >> ~`、一元负号、`not`（仅 `spy.bool`）、
  比较 `== != < <= > >=`（**不支持链式比较**）。
- 两侧都是编译期常量时在编译期折叠；含运行时值时在运行时计算。
- 整数 `/` 是真除法（结果默认为 `f64`）：两侧都是编译期常量时在编译期折叠；含运行时整数时，只要整数能**无损**转成 `f64`（`i32`/`u32`/`i16`/…，见 §1.1）就能直接参与（如 `a / b`），否则也要么显式转成浮点（`int_to_float`），要么用 `//`。整数 `//` 默认向下取整；浮点 `//` 是 `floor(a/b)`。
- `**`：整数与浮点指数都支持。**浮点 `%` 未实现**（报错）。
- 每个二元算符有对应的复合赋值 `a <算符>= b`。
- 条件表达式 `a if c else b`：条件必须 `spy.bool`；编译期条件只保留选中分支。
- 短路 `and`/`or`：操作数依次求值，按布尔值决定是否继续。
- `cast(T, value)`：no-op；`len(x)`：元组/数组/切片/结构体的 `__len__`；`ord(b)`：单字节字节串的编码。
- 下标 `expr[i]`：数组/元组/多指针/切片/字节串/字典/`$args`/`$kwargs`；`p[...]` 解引用单指针；
  结构体可通过 `__spy_getitemptr__` 重载下标（见 §7.8）。
- 运算符重载见 §7.2。

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
    # 整数能无损转成 f64，真除法可直接用
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

## 5. 语句与控制流

### 5.1 `if` / `elif` / `else`

条件为编译期常量时只保留选中分支（另一分支不编译、不分析）。
条件里 `:=` 解包出的名字在其 body 里可见。**无真值转换**（条件必须是 `spy.bool`）。

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

每轮重新求值条件；`break` 跳到循环后（跳过 `else`），`continue` 回到条件。
编译期为假的 `while False:` 不生成循环体。

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

`for exprs in iter:` 基于迭代器协议：先取 `iter.__iter__()`，每轮调用 `__next__()` 并把结果绑到 `exprs`；
遇到 `StopIteration` 结束循环（此时先跑 `else`）。`range` 与 `StopIteration` 来自 `std`。
循环变量绑在循环体子作用域里，循环后不可见。

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

`break` 离开最内层循环；`continue` 回到循环条件。循环内的 `try` body 里的 `break`/`continue` 直接跳出/继续。
（`break`/`continue` 的示例见 §5.2、§5.3。）

### 5.5 异常 `raise` / `try` / `except`

- `raise E(...)` 构造异常并抛出。
- `except E as e` 把 `e` 绑定到异常值（不拷贝）。
- 子句按顺序匹配；裸 `except:` 捕获本函数会抛出的所有异常；没被任何子句命中的 except 是死代码（不编译）。
- `try-else`、`try-finally` **不支持**；必须有至少一个 except；**裸 `raise` 不支持**。
- `@func(exceptions=...)` 声明异常集：单个类型、元组、或 `"infer"`（从函数体推断）。

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

值部分是空类型且不抛异常的函数永远回不来（`noreturn`）：调用它是 `noreturn` 调用，调用点之后的代码是死代码。
（示例见 §1.13 的 `spins`/`maybe_spin`。）

### 5.7 `defer`

- `with syntax.defer(): body` 把 `body` 登记到所在区域，在**离开该区域**时运行。
- 触发哪些离开由 `defer(flags)` 位掩码决定：`OK`（正常离开）、`RAISE`（错误离开）、`UNWIND`（panic 展开）；
  `okdefer()` = `defer(OK)`，`errdefer()` = `defer(ERR)`（`ERR = RAISE | UNWIND`），无参 `defer()` = `ALL`。
- 同一区域多个 defer 按声明**反序**运行，内层区域先于外层。
- `body` 不得跳出 defer 块（`return`/`break`/`continue`/`raise` 越出即报错）。

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

- `std.core.panic(data)` 抛出一个携带 `PanicData` 的 panic。
- `std.core.catch_unwind(fn)`（`fn` 必须是编译成运行时函数的**非内联闭包**）捕获 panic，把数据包成 `std.core.UnwindException`
  沿普通异常路径抛出（`except UnwindException as e: e.data` 读取）。
- `@func`/`@syntax.closure` 的 `may_panic` 默认 `True`。未捕获的 panic 展开到 Python 边界时，由该函数的入口 thunk 捕获并作为 `std.core.UnwindException` 重新抛出（`e.data` 是 payload），不再中止进程。

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

对 tagged union：`match u:` + `case A():` / `case A() as e:` / `case _:`（通配必须最后）。
不支持 case 守卫与子模式（`case A(x):` 报错）。（示例见 §1.8 的 `tu_match`。）

---

## 6. 函数

### 6.1 定义与特化

- `@spy.func()` 注册；首次调用按实参类型编译出一个**特化**，同函数可有多个特化。
- **不用 `@func` 装饰的函数是编译期 Python 辅助函数**（`PyFunction`，见 §6.10）：只能在编译期调用，函数体由 Python 原样执行。
- `@func(inline=True)` 把注册的函数标成**内联**：函数体展开到调用处，可以接收运行期实参。未装饰的结构体方法也是内联的。
- 形参类型由：注解（替换已求解的泛型参数后）> 实参推断出的类型 > 默认值的类型决定。
- `@func(ignore_signature=True)` 忽略形参与返回**类型**注解（它们只供 Python 类型检查器阅读）：形参类型一律由实参推断、返回类型由函数体推断。`Comptime`/`Comptime[T]` 不是类型，其标记仍保留（参数仍是编译期参数）。

```python
@func()
def add_generic[T: Numeric](a: T, b: T) -> T:
    return a + b

@func(inline=True)
def add_plain(a: i32, b: i32) -> i32:      # an inlined spy function
    return a + b

@func()
def use_inline(a: i32, b: i32) -> i32:
    return add_plain(a, b)

assert add_generic(1, 2) == 3              # a Python int marshals as i64
assert add_generic(1.5, 2.5) == 4.0
assert use_inline(2, 3) == 5
```

### 6.2 参数

- 支持位置参数、关键字参数、默认参数、`*args`/`**kwargs`。
- **仅位置/仅关键字参数不支持**（Python `posonlyargs`/`kwonlyargs` 报错）。
- 编译期参数 `x: Comptime` / `x: Comptime[T]`：只在编译期传递，实参必须完全编译时已知。
- **形参类型无运行时表示 ⇒ 隐式编译期形参**（类型值、字节串、DST 函数类型、空类型等）：
  自动成为编译期形参，无需显式 `Comptime`（如 `def sstr(s: bytes)`）。
- `type[X]` 形参：接受一个**类型值**。
- `*args`/`**kwargs`：见 §8.3。

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

### 6.3 C 变参声明

- `@decl_func()` / `@func_type(callconv='c')`：最后一个具名参数后一个**不带注解**的 `*args` 表示 C 的 `...`。
- 调用可传任意多个额外实参，每个按值、以自身类型传递，并施加 **C 默认实参提升**（窄于 `int` 的整型提升为 `int`，`f32`→`f64`）。
- `*args` 带注解、`**kwargs`、默认 callconv 的函数类型都不允许 C 变参。
- **C 变参函数的定义**（用 `va_arg` 读实参）**尚未实现**（只能声明与调用）。

```python
@func()
def c_vararg_call(n: i32) -> i32:
    buf = typed_undefined(Array[u8, Literal[16]])
    s = arr_slice(ref(buf))
    fmt: ConstMultiPtr[u8] = ptr_cast(b'%d', ConstMultiPtr[u8])
    return snprintf(s.ptr, s.length, fmt, n)   # ``n`` goes to ``*args``

assert c_vararg_call(42) == 2
```

### 6.4 `tuple[...]` 形参（铺平传参）

- 形参注解 `tuple[T1, ...]`：实参按元素**铺平**传递，callee 侧重组成一个元组值。
- 函数体用编译期整数下标 `x[i]` 取元素，可嵌套。
- 结构体元素**不**铺平（整体一个实参）。
- 无注解参数接受调用点给的任意形状，编译期反射能读出形状。
- 元组/字典类型字段（`Foo.t: tuple[i32, i32]`）与结构体字段**完全同规则**。

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

### 6.5 多返回值

- 返回注解 `tuple[T1, T2, ...]` = 返回多个值。
- 调用侧：`a, b = f()` 解构；`return f()` 转发。
- **打包**：全运行时结果可打包进**普通元组变量** `packed = f()`（之后可 `x, y = packed` 再解构、`len`、`packed[i]`）；
  打包保留结果的分组——每个「结果元素」一个元素，嵌套结果是它自己的元组（故 `packed[i][j]` 可用，`len` 为顶层 arity）；
  含编译期叶（类型值/const 指针/字节串/函数值）时打包出的元组无运行时表示，目标必须是 `Comptime` 变量。
- **编译时返回值**：完全编译时已知的返回值不占运行时结果。
  非 inline 调用取 callee **最后一条被分析的 `return`** 的值（不检查各 return 路径是否一致）。
- `tuple[T, ...]`（变长）在任意嵌套层级报错。

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
    p = nested()               # 打包出的元组保留嵌套
    return p[1][0] + p[1][1]   # 5

assert min_max(3, 1) == (1, 3)   # the Python side gets a tuple
assert use_min_max(3, 1) == 13
assert packed(3, 1) == 4
assert use_packed_nested() == 5
```

### 6.6 泛型

- PEP 695 `[T]` 语法（需 Python 3.13+）；`T` 由实参类型求解。
- 形参注解为同一 `T` 的实参类型统一成一个共同类型，实参再转换到它。
- **声明了返回注解时它决定该特化的返回类型**；递归函数必须有返回注解。
- 直接递归、互递归、泛型多类型递归都支持。

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

### 6.7 闭包（= 带 `__call__` 的匿名 struct）

- 函数体内可再定义 `def`/`lambda`，它们**按引用**捕获外层变量（`move=False` 默认）或按值（`move=True`）。
- `@syntax.closure(inline=True, exceptions=..., callconv='default', may_panic=True, move=False)`：
  只允许的嵌套 def 装饰器；`inline=False` 编译成运行时函数。`lambda` 一律强制内联。
- 闭包**就是一个值**（字段是捕获，`__call__` 是函数体）；`f(...)` 即调用 `f.__call__(...)`。
- 因为闭包是 struct 值，旧限制全部取消：可存变量/字段、传参、返回、放进运行时存储。
- **限制**：无 `nonlocal`；**互递归闭包不支持**（见 `doc/issues.md` #6）；闭包体引用外层函数的类型参数不支持。

```python
@func(inline=True)
def apply_inline(f, x: i32) -> i32:        # an inlined spy function
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

`std.core.as_func_ptr(T, f)`：`T` 只给 Python 类型检查器看（实现忽略）；`f` 必须是一个**有 `__call__` 且为 ZST 的 struct**；
结果是**去掉 `self` 参数的 `__call__`**。普通 spy 函数与 extern 函数**直接用其名字**即可取到常函数指针。

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

### 6.9 递归

- 直接递归、互递归、泛型多类型递归都支持；递归函数必须能由注解确定返回类型（见 §6.6）。

### 6.10 编译期 Python 函数（`PyFunction`）

- **不用 `@func` 装饰**的普通 Python 函数是一个**编译期值**（`sval.PyFunction`，没有运行时表示）：它**只能在编译期被调用**，
  函数体由 Python **原样执行**（不是编译成 spy 代码）；`@func(inline=True)` 才是「内联 spy 函数」。
- 实参与返回值都必须是**编译期量**，且只能是普通 Python 对象：`int`（含 `bool`）、`float`、`complex`、`bytes`、`None`；
  Python `str` 在边界处按 UTF-8 编码成 spy 的字节串（与源码里的字符串字面量一致）。
  和 Python 调用边界一样，`Null`/`TypedNull`/`Void` 都传成 `None`，`Option` 自动解包成载荷值、tagged union 解包成变体值（载荷/变体是聚合时会因不是普通对象而被拒）；
  返回 `None` 则变成 `Null`（option 的缺失值）。
  其它实参（运行期值、spy 类型值、普通聚合、指针等）或其它返回值一律编译报错。
- 返回值写进调用处的结果位置，因此取该位置的类型（推断出的位置保留值自身的类型）。绑参由 Python 完成（默认值、`*args`/`**kwargs` 都照常）。
- `PyFunction` 是普通编译期值，可存进 `Comptime` 变量/参数，也可作为 comptime 实参传给别的 spy 函数。

```python
def double(n: int) -> int:                 # not registered: runs at compile time
    return n * 2

def shout(s: bytes) -> str:
    return s.decode("utf-8").upper()

@func()
def use_py_function() -> i32:
    f: Comptime = double                   # stored in a compile-time variable
    return f(21)

@func()
def use_py_function_str():
    return shout(b"hey")                   # b'HEY': a str crosses as its UTF-8 bytes

assert use_py_function() == 42
assert use_py_function_str() == b"HEY"
```

---

## 7. 结构体

### 7.1 声明与字段

- `@spy.struct(extern_c=False, copyable="inherit")` 把带注解的类变成 spy 结构体；注解字段构成其字段集。
- 字段按声明顺序排列、支持嵌套（`p.inner.a`）；可读、可赋、可 `+=`（可任意嵌套）。
- `y = x` 拷贝结构体。
- 类体里给字段的值是默认值；没有默认值的字段必须构造时写全（ZST 字段也一样）。
- 类里**不能**定义 `__init__`。
- **`is_comptime` 字段**（Zig 语义）：不参与拷贝、构造时可省略；访问运行时 struct 的这类字段直接返回默认值。
- **`is_ref` 字段**：逻辑上是引用，只可读、不可写透。

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

### 7.2 方法

- 类里 `@spy.func()` 装饰的方法是编译成原生调用的 spy 方法；未装饰的方法在调用处**内联**。
- 方法 `self` 默认是指向自身的可写指针（`Ptr[Self]`）；`@func(self_arg="const ptr")` 得只读指针，`"value"` 按值传。
- 方法可继承（取自整个 MRO，子类覆盖优先）。
- 通过类名调用 `Foo.m(...)`（含 `Foo[i32].m(x)`）**不隐式传 `self`**；`@staticmethod` 不带接收者。
- `@struct()` 可省略：普通类可用 `Cls.fn(...)` 当命名空间。
- 运算符重载：`__add__`/`__sub__`/…（各自反射版 `__r*__` 与原地版 `__i*__`）、比较 `__eq__`/`__lt__`/…、
  一元 `__neg__`/`__invert__`、作为条件的 `__bool__`、以及 `__len__`；下标重载见 §7.8。

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

### 7.3 构造

- `Point(a, b)` 就地构造：直接写入目标位置，嵌套构造就地写入（无拷贝）。
- 位置实参按字段声明顺序，关键字实参按名；每个字段都要得到值（默认值可省）。
- 默认值**不参与泛型参数推断**，只由提供的字段值决定。

```python
@struct()
class Inner:
    v: i32

@struct()
class Outer:
    inner: Inner

@func()
def build(x: i32) -> i32:
    o = Outer(Inner(x))          # the inner struct is built into its destination
    o.inner.v += 1
    return o.inner.v

assert build(1) == 2
```

### 7.4 泛型结构体

- `class Foo[T]` 是模板，`Foo[i32]` 是特化；字段注解里的 `T` 被替换进去（可递归）。
- 构造可省略泛型参数：特化由目标位置的类型确定；类型未知时由写进各字段的值的类型推断。
- 方法调用 `x.m(...)` ≡ `typeof(x).m(x, ...)`；方法可有自己的类型参数；结构体的类型参数可在体内当值使用。

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

### 7.5 布局

- 非 `extern_c` 结构体的内存布局由编译器决定（字段可能被重排、ZST 字段不占空间）；`extern_c=True` 采用 C 布局
  （保持声明顺序，可跨 C 边界）。
- 布局可通过 `std.mem.layout_of`/`size_of`/`align_of` 查询（见 §8.8）。

### 7.6 拷贝性与传参/返回

- `copyable=True|False|"inherit"`（默认 `"inherit"` = 每字段可拷贝才可拷贝）；DST 一律不可拷贝。
- 不可拷贝的值可就地构造、按引用传递、经返回位置返回、经地址读写字段，但**不能整体读出**。
- **传参按值**（语义上总是拷贝）；不可拷贝的强制按引用。
- **返回**：小结构体按值返回；大结构体经返回位置返回；不可拷贝的强制经返回位置返回。

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

### 7.7 Python 侧使用

- `Point(1, 2)` 构造一个实例（省略字段取默认值；泛型特化由字段值推断或显式 `Foo[i32](...)`）。
- 实例字段可读可写；方法可调用（`p.m()` 隐式传 `self`；`Foo.m(p, ...)` 不隐式传）。
- 实例可作实参、返回值跨边界；布局由边界层处理。
- 嵌套字段是父实例的**视图**（写入子字段即写入父实例）。

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

### 7.8 下标重载（`__spy_getitemptr__`）

一个结构体值可以像数组一样被下标（`expr[i]`）：定义方法 `__spy_getitemptr__(self, index) -> Ptr[T]`
（或 `ConstPtr[T]` / `MultiPtr[T]` / `ConstMultiPtr[T]`），它**返回元素的指针**——即第 `i` 个元素的地址。

- `expr[i]` 就是该方法返回值**指向的那个元素**（编译器在需要值的地方自动解引用）：可读、可赋值、可 `+=`，也可继续取字段/下标。
- 方法可以未装饰（调用处内联）或用 `@func()` 编译成运行时方法；返回的必须是**指针**，返回一个值会报错。
- 类里通常再写一对 `__getitem__`/`__setitem__`（放在 `if TYPE_CHECKING:` 里）**只为 Python 类型检查器**接受
  `expr[i]` 的写法——编译器不调用它们（下标编译时走 `__spy_getitemptr__`）。
- `std.SlicePtr`/`std.ConstSlicePtr`/`std.vec.Vec` 就用它实现下标。

```python
@struct()
class IntArray:
    ptr: MultiPtr[i32]

    def __spy_getitemptr__(self, index: i64) -> MultiPtr[i32]:
        return self.ptr + index          # the address of the i-th element

    if TYPE_CHECKING:
        # only so a type checker accepts ``v[i]``; never called by the compiler
        def __getitem__(self, index: int) -> i32: ...
        def __setitem__(self, index: int, value: i32) -> None: ...

@func()
def subscript_use(x: i32) -> i32:
    a = array(x, x + 1, x + 2)
    v = IntArray(arr_slice(ref(a)).ptr)
    v[2] = 100                           # writes through the returned pointer
    return v[0] + v[1] + v[2]

assert subscript_use(1) == 1 + 2 + 100
```

---

## 8. 编译期（comptime）

### 8.1 类型探针与编译期日志

- `spy.typeof(x)`：查询静态类型（返回类型值），是一个**类型探针**：只做类型化、无运行时行为；其中引用的 spy 函数会被触发编译。被探测的调用若抛错离开探针则报错。
- `std.core.compile_log(...)`：编译期打印（运行时无动作）；bytes 渲染为文本，运行时值渲染为 `<T>`。

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

### 8.2 编译期参数（补充）

- 无运行时表示的类型作形参 ⇒ 隐式编译期形参（见 §6.2）。
- **仅 const**：指向编译期常量的指针可以作为 `Comptime` 实参传给**非内联**函数（按身份传递、可别名）；
  非 const 的编译期指针不能跨非内联边界（见 §8.5）。

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

- 函数体里 `args` 收集位置实参、`kwargs` 按名收集关键字实参。
- `len(args)`、`args[i]`（编译期整数下标）、`kwargs[b'name']`（编译期字节串名）。
- 元素/值类型可注解也可推断；`*args`/`**kwargs` 元素不支持 `type[X]`；需要时配 `syntax.unroll()` 遍历。
- **调用点展开（`f(*args, **kwargs)`）尚未实现**：`**kw` 被静默忽略，`*x` 报错（见 `doc/issues.md` #1）。

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

### 8.4 `dict[str, T]` 形参

- `**kwargs` 形参 / `dict[str, T]` 形参按名铺平传参；访问 `len(kwargs)`、`kwargs[b'name']`（仅编译期字节串名，**不支持运行时键查找**）。
- 由于有运行时表示，字典值可存普通变量/字段、按值或引用传递。
- `dict[str, T]` 注解本身尚未解析。

### 8.5 编译期指针

- 取址/偏移：`ref(a)` 得到指向 `a` 的编译期指针；`p[...]` 读写它指向的位置。
- **const 指针**可以跨非内联边界（作 `Comptime` 实参、被编译闭包捕获、作非内联函数返回值），按**身份**传递、可别名。
- **非 const 编译期指针**跨非内联边界时会**物化一份拷贝**：callee 的写入影响不到原位置。
- 经 `ptr_cast` 得到的编译期指针不能读写透。
- 指向编译期常量的指针、字节串常量都会自动落成全局常量。

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

### 8.6 编译期聚合与循环

- 完全编译时已知的聚合，字段/元素读写全部在编译期折叠；整体拷贝按值。
- 编译期 `if`：条件必须是完全编译时已知；未选分支不编译、不分析（其中的 `compile_error` 不触发）。
- 编译期循环 `syntax.unroll()`：标记紧跟其后的 `while`/`for`；条件必须是编译期值；循环体必须推进编译期状态，
  否则超过 `max_loop_unroll`（默认 1024）报错。`syntax.unroll()` 必须紧接在被标记的循环之前。

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

### 8.7 其他编译期内建

- `std.core.undefined()`：任意类型的 undefined，写进目标时按其类型定型；undefined 作条件报错。
- `std.core.compile_error(msg)`：以编译期字节串为消息中止编译；非字节串/运行时消息报错；死分支不触发。
- `std.core.coerce(value, T)`：把 value 具体化成类型 `T`——与 store 相同的**隐式转换**，只允许同类型或扩宽：整型按范围覆盖（`u32 → u64`、`i32 → i64`），浮点 `f32 → f64`，**无损**的整数 → 浮点（运行期按类型范围、编译期按值，见 §1.1），合法的指针转换，以及（能无损转成元素浮点的）实数 `→ Complex[T]`；**收窄与有损的整数 → 浮点一律报错**。编译期值按值检查（`x: i8 = 12` 可以）。收窄要用 `std.int.truncate`/`int_cast` 或 `std.float.float_truncate`，有损的整数 → 浮点用 `std.int.int_to_float`。
- `std.core.lub_type(t1, t2)`：两个类型的最小上界——两者共同的 peer 类型（`t1.resolve_peer_type(t2)`），如类型与其 `Option`、变体与其 tagged union、能无损容纳它的整型的无类型整数或浮点。结果是编译期类型值，可用于命名声明类型或喂给 `type[X]` 形参；两者无共同类型时得到 `None`。peer 规则对两个参数是对称的，唯一例外是签名里的 result 类型（返回类型 + 异常集）：此时取 `t1` 的返回类型、`t1` 的异常在前，所以这些情形下结果与参数顺序有关。
- `std.core.compile_option(name)`：读取宿主上下文里存的编译期选项（`CompileContext.get_compile_option`）；未设置时是缺失值（`None`）。
- `std.core.compile(fn)`：手动编译一个**非泛型**函数值（预编译其默认特化）。
- `std.core.bitcast(value, T)` / `std.int.truncate(value, T)` / `std.float.float_truncate(value, T)`：标量的同位宽重解释 / 整数收窄 / 浮点收窄。
- `std.core.int_to_float_unchecked(value, T)` / `std.core.float_to_int_unchecked(value, T)`：整数↔浮点转换（`sitofp`/`uitofp` 与 `fptosi`/`fptoui`，后者向零取整），**不做任何检查**；编译期值装不进目标类型时得到 `undefined`。

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
def coerced_widen(x: i32) -> i64:
    return coerce(x, i64)        # coerce 只做同类型/扩宽

@func()
def same_bits(x: f32) -> u32:
    return bitcast(x, u32)

@func()
def truncated(x: u32) -> u8:
    return truncate(x, u8)

@func()
def float_truncated(x: f64) -> f32:
    return float_truncate(x, f32)

@func()
def to_f64(x: u64) -> f64:
    return int_to_float(x, f64)          # exact, else panics

@func()
def to_i32(x: f64) -> i32:
    return float_to_int(x, i32)          # in range, else panics

@func()
def widen(x: i32) -> i64:
    comptime()
    t = lub_type(i32, i64)               # i32 与 i64 的最小上界
    y: t = x                             # 算出的类型命名声明类型
    return y

assert dead_branch() == 1
assert pick(True, 1, 2) == 1 and pick(False, 1, 2) == 2
assert coerced_widen(3) == 3
assert float_truncated(1.5) == 1.5
assert same_bits(1.0) == 0x3F800000
assert truncated(300) == 44
assert widen(3) == 3
assert to_f64(9007199254740992) == 9007199254740992.0
assert to_i32(3.9) == 3
```

### 8.8 反射

- `std.reflect.type_info(type) -> TypeInfo` / `reify(info) -> type`；
  变体 `IntType`/`FloatType`/`PointerType`/`ArrayType`/`StructType`/`TaggedUnionType`/`OptionType`/`TupleType`/`StrDictType`/`ComplexType`/`UnionType`。
- `reflect.StructField` 含 `is_comptime` / `is_ref` 标记；`reflect.StructType.head`（非泛型为 `None`）。
- `reify` 一个 struct 会**重新构建**（与原始 struct 不 identity 相等，但字段/字段类型相同）。
- `std.mem.layout_of`/`size_of`/`align_of`：也接受元组/字典类型。

```python
@func()
def int_bits(t: type[i32]) -> i32:
    comptime()
    info = type_info(t)
    if isinstance(i := info, IntType):
        return i.bits
    return 0

@func()
def float_bits(t: type[f32]) -> i32:
    comptime()
    info = type_info(t)
    if isinstance(i := info, FloatType):
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
    return int_cast(size_of(i32), i32)

assert int_bits(i32) == 32
assert float_bits(f32) == 32
assert reified() == 1
assert int_size() == 4
```

### 8.9 编译期字节串

- `bytes` 只在编译期存在（无运行时表示）；`str` 字面量被编码成 `bytes`（spy 函数内不允许 `str`）。
- 可下标 `b[i]` 与切片 `b[a:c]`（下界缺省 0、上界缺省长度、step 必须 1，越界报错）；`ord(b)` 取单字节字节串的编码。
- 落成运行时：`ptr_cast(s, ConstMultiPtr[u8])` 或 `std.core.sstr(s) -> ConstSlicePtr[u8]`（长度不含 NUL）。
- 内容相同的字节串共用一个全局常量；持有字节串的局部量必须是编译期变量。

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
    return int_cast(s.length, i32)

assert byte_at() == 66
assert second_byte() == 121
assert str_len() == 5
```

---

## 9. 标准库

| 模块 | 内容 |
|---|---|
| `std.core` | `Numeric`、`StopIteration`、`slice`、`range`、`SlicePtr`/`ConstSlicePtr`、`arr_slice`/`const_arr_slice`/`const_arr_ptr`、`NullException`、`unwrap`/`unwrap_or`、`undefined`、`coerce`/`typed_undefined`/`lub_type`、`const_array`、`compile_error`、`sstr`、`compile`、`bitcast`、`int_to_float_unchecked`/`float_to_int_unchecked`、`DestructorFn`/`OpaqueWithDestructor`/`PanicData`/`deinit_panic_data`/`UnwindException`、`panic`、`catch_unwind`、`compile_option` |
| `std.reflect` | `TypeInfo` 及变体、`StructField`、`type_info`、`reify` |
| `std.int` | `truncate`（内建）、`IntCastError`、`valid_range`、`try_int_cast`/`int_cast`、`FloatCastError`、`try_int_to_float`/`int_to_float`、`try_float_to_int`/`float_to_int` |
| `std.float` | `float_truncate`（内建） |
| `std.mem` | `Layout`、`layout_of`/`size_of`/`align_of`、`AllocError`、`Allocator`（协议）、`AllocatorVtable`、`DynamicAllocator`、`CAllocator` |
| `std.c` | `malloc`/`free`/`realloc`/`puts`/`snprintf`（extern 声明）、`sin`/`sinf`/`cos`/`cosf`/`tan`/`tanf`/`atan`/`atanf`/`atan2`/`atan2f`/`sqrt`/`sqrtf`/`cbrt`/`cbrtf`/`pow`/`powf`/`exp`/`expf`/`log`/`logf`/`asin`/`asinf`/`acos`/`acosf`（libm）、`CError`、`FILE`、`fopen`/`fclose`/`fread`/`fwrite` |
| `std.io` | `IOError`/`PartialWriteError`/`PartialReadError`、`Writable`/`Readable`（协议） |
| `std.math` | `sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`sqrt`/`cbrt`/`pow`/`exp`/`log`（实现由编译期选项 `sin_mode`/`cos_mode`/`tan_mode`/`asin_mode`/`acos_mode`/`atan_mode`/`atan2_mode`/`sqrt_mode`/`cbrt_mode`/`pow_mode`/`exp_mode`/`log_mode` 在编译期选定：`llvm`/`libc`/`software`） |
| `std.vec` | `Vec[T, A]` 动态数组 |
| `std.atomic` | `MemoryOrder`/`RmwOp`、`atomic_load`/`atomic_store`/`atomic_rmw`/`atomic_cmpxchg`/`atomic_fence`、便捷包装 |
| `std.openmp` | `load_openmp_runtime`、`parallel_loop` |

### 9.1 原子操作

- 操作对象（pointee）必须是整数、浮点或指针（不支持 `bool`；整数宽度须为字节大小的 2 的幂）。
- `atomic_cmpxchg` 不接受浮点 pointee；返回 `(旧值, 是否交换)`，failure ordering 不能是 `RELEASE`/`ACQ_REL`、也不能强于 success。
- `atomic_rmw` 的操作按 pointee 种类受限（整数全部；浮点只 `XCHG`/`ADD`/`SUB`；指针只 `XCHG`），还接受**复数** pointee。
- 每个操作带一个 `MemoryOrder`；除 `atomic_fence` 外都有 `volatile` 开关。
- 便捷包装：`atomic_swap`、`atomic_fetch_add/sub/and/or/xor`。

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

`std.openmp.parallel_loop(length, kernel, reduce_op=None)` 实现 `#pragma omp parallel for`（static 调度）。
使用前须 `load_openmp_runtime()`（幂等，加载 `libomp`）。

- `kernel` 是接收下标 `i`（类型 `typeof(length)`，`i32`/`u32`/`i64`/`u64` 之一）的闭包；每个线程拿到 `[0, length)` 的一个切片。
- 若 `kernel` 有返回值（单个，或用 `tuple[...]` 注解的多返回值）且给出 `reduce_op`，则把各次迭代的返回值归约，
  并把结果**作为 `parallel_loop` 的返回值**（多返回值以多返回形式交付）。
- `reduce_op` 以 `reduce_op(acc, v)` 调用：两个操作数各是「kernel 的返回值」本身——单返回值即该值，多返回值即整个返回值元组（**不会**掉平成每元素一个实参）。
- 每个线程先用自己的切片折叠出一个 partial，再在临界区（`__kmpc_critical`）内并入共享累加器；合并顺序不确定，
  因此 `reduce_op` 需满足结合律（最好还有交换律）。
- `length == 0` 时没有迭代可归约，返回累加器的初始（`undefined`）值。
- `kernel` 返回 void 时不做归约，`reduce_op` 被忽略（只执行循环体）。

```python
@func()
def parallel_sum(n: i32) -> i64:
    def kernel(i: i32) -> i64:
        return coerce(i, i64)          # 显式转换（typing.cast 在 spy 里只是空操作）

    def add(a: i64, b: i64) -> i64:
        return a + b

    return parallel_loop(n, kernel, add)   # sum(0..n-1)

@func()
def parallel_sum_count(n: i32) -> i64:
    def kernel(i: i32):
        return coerce(i, i64), coerce(1, i64)   # 多返回值

    def add(a, b):
        return a[0] + b[0], a[1] + b[1]

    total, count = parallel_loop(n, kernel, add)
    return total * 1000 + count
```

### 9.3 数学

`std.math.sin(x)` / `std.math.cos(x)` / `std.math.tan(x)` 返回 `f32` 或 `f64` 的 `x` 的正弦/余弦/正切（结果类型与 `x` 相同），`std.math.asin(x)` / `std.math.acos(x)` 返回其反正弦（值域 `[-pi/2, pi/2]`）/反余弦（值域 `[0, pi]`）（`|x| > 1` 为 NaN），`std.math.atan(x)` 返回其反正切（值域 `(-pi/2, pi/2)`），`std.math.atan2(y, x)` 返回 `y/x` 的反正切（值域 `[-pi, pi]`，两个参数是相容的浮点类型，结果取它们的共同类型，并按两者的符号选择象限），`std.math.sqrt(x)` 返回其平方根（非负 `x`），`std.math.cbrt(x)` 返回其立方根（可为负 `x`），`std.math.pow(x, y)` 返回 `x` 的 `y` 次幂（两个参数是相容的浮点类型，结果取它们的共同类型），`std.math.exp(x)` 返回 `e**x`，`std.math.log(x)` 返回自然对数 `ln(x)`（`x` 为 `f32` 或 `f64`）。具体用哪个实现在**编译期**由宿主的编译期选项 `sin_mode`/`cos_mode`/`tan_mode`/`asin_mode`/`acos_mode`/`atan_mode`/`atan2_mode`/`sqrt_mode`/`cbrt_mode`/`pow_mode`/`exp_mode`/`log_mode` 选定：

- `llvm`：LLVM 内建（`llvm.sin.*`/`llvm.cos.*`/`llvm.tan.*`/`llvm.asin.*`/`llvm.acos.*`/`llvm.atan.*`/`llvm.atan2.*`/`llvm.sqrt.*`/`llvm.cbrt.*`/`llvm.pow.*`/`llvm.exp.*`/`llvm.log.*`；其中 `llvm.cbrt.*` 是 LLVM 21 才引入的内建，更早的 LLVM 上不可用）；
- `libc`：C 标准库的 `sinf`/`sin`/`cosf`/`cos`/`tanf`/`tan`/`asinf`/`asin`/`acosf`/`acos`/`atanf`/`atan`/`atan2f`/`atan2`/`sqrtf`/`sqrt`/`cbrtf`/`cbrt`/`powf`/`pow`/`expf`/`exp`/`logf`/`log`（即 `std.c` 中的 extern 声明）；
- `software`（默认值，选项未设置时也是它）：musl 的 `sin`/`sinf`/`cos`/`cosf`/`tan`/`tanf` 的纯 spy 移植（`std.math.trig`，共用内核与参数规约）；musl 的 `asin`/`asinf`/`acos`/`acosf`/`atan`/`atanf`/`atan2`/`atan2f` 的在 `std.math.invtrig`（`asin`/`acos` 共用 `R(x**2)` 有理逼近，`atan` 按区间使用奇多项式，`atan2` 按象限规约到 `atan`；`f32` 路径在 `f64` 中求值并在最后舍入）；musl 的 `sqrt`/`sqrtf` 的纯 spy 移植在 `std.math.sqrt`，`cbrt`/`cbrtf` 的在 `std.math.cbrt`，`pow`/`powf` 的在 `std.math.pow`，`exp`/`expf` 的在 `std.math.exp`（`f64` 复用 `pow` 的 exp 内核），`log`/`logf` 的在 `std.math.log`（使用完整的 `log_data.c`/`logf_data.c` 数据表）。

只有被选中的实现会被类型推导并 lowering，因此选择在运行时没有开销；非法取值是编译错误。`sin`/`cos` 与 `sqrt` 的软件实现对 `f32` 与 `f64`
都是正确舍入的，`tan`/`tanf` 的软件实现误差约小于 1 ULP，`cbrt` 的软件实现误差小于 0.667 ULP，`pow`/`powf` 的软件实现（musl 的 Arm `pow`/`powf` 移植）误差分别小于 0.54 与 0.82 ULP，`exp`/`expf` 的误差约小于 0.51 ULP，`log`/`logf` 的误差分别约小于 0.51 与 0.82 ULP，`asin`/`acos`/`atan`/`atan2` 的软件实现误差约小于 1 ULP。`sqrt` 的负参数（以及 NaN）得到 NaN，`sqrt(-0.0)` 是 `-0.0`；`cbrt` 是奇函数（`cbrt(-x) == -cbrt(x)`），负参数正常计算，`cbrt(±0)` 是 `±0`；`pow` 的 `x**0` 是 `1`（含 `0**0`），负底数配整数指数正常计算（偶数得正、奇数得负，`pow(-0.0, 3)` 是 `-0.0`），负底数配非整数指数得到 NaN，`|x|` 与 `y` 的 `±inf`、溢出与下溢都按 C 的 `pow` 处理；`exp(±0)` 是 `1`，`exp(-inf)` 是 `+0`，`exp(+inf)` 是 `+inf`，`exp(NaN)` 是 NaN；`log(1)` 是 `+0`，`log(±0)` 是 `-inf`，`log(+inf)` 是 `+inf`，负参数与 `log(NaN)` 是 NaN，次正规参数正常计算。`asin(±1)` 是 `±pi/2`，`acos(1)` 是 `+0`、`acos(-1)` 是 `pi`，`|x| > 1`（含 NaN）得到 NaN，`asin` 是奇函数、`acos(±0)` 是 `pi/2`。`atan` 是奇函数，`atan(±0)` 是 `±0`，`atan(±inf)` 是 `±pi/2`，NaN 得到 NaN；`atan2` 的带符号零、四个象限、`±inf` 与 NaN 按 C 的 `atan2` 处理（例如 `atan2(±0, +1)` 是 `±0`、`atan2(±0, -1)` 是 `±pi`、`atan2(±inf, ±inf)` 是 `±pi/4` 或 `±3pi/4`）。

```python
@func()
def my_sin(x: f64) -> f64:
    return sin(x)

@func()
def my_cos(x: f64) -> f64:
    return cos(x)

@func()
def my_tan(x: f64) -> f64:
    return tan(x)

@func()
def my_sqrt(x: f64) -> f64:
    return sqrt(x)

@func()
def my_cbrt(x: f64) -> f64:
    return cbrt(x)

@func()
def my_pow(x: f64, y: f64) -> f64:
    return pow(x, y)

@func()
def my_exp(x: f64) -> f64:
    return exp(x)

@func()
def my_log(x: f64) -> f64:
    return log(x)

@func()
def my_asin(x: f64) -> f64:
    return asin(x)

@func()
def my_acos(x: f64) -> f64:
    return acos(x)

@func()
def my_atan(x: f64) -> f64:
    return atan(x)

@func()
def my_atan2(y: f64, x: f64) -> f64:
    return atan2(y, x)

assert abs(my_sin(0.5) - 0.479425538604203) < 1e-15
assert abs(my_sin(1.0) - 0.8414709848078965) < 1e-15
assert abs(my_cos(0.5) - 0.8775825618903728) < 1e-15
assert abs(my_tan(0.5) - 0.5463024898437905) < 1e-15
assert abs(my_sqrt(2.0) - 1.4142135623730951) < 1e-15
assert my_sqrt(4.0) == 2.0
assert my_sqrt(0.25) == 0.5
assert my_cbrt(8.0) == 2.0
assert my_cbrt(-8.0) == -2.0
assert abs(my_cbrt(2.0) - 1.2599210498948732) < 1e-15
assert my_pow(2.0, 10.0) == 1024.0
assert abs(my_pow(2.0, 0.5) - 1.4142135623730951) < 1e-15
assert my_pow(-8.0, 3.0) == -512.0
assert my_exp(0.0) == 1.0
assert abs(my_exp(1.0) - 2.718281828459045) < 1e-15
assert my_log(1.0) == 0.0
assert abs(my_log(2.718281828459045) - 1.0) < 1e-15
assert abs(my_asin(0.5) - 0.5235987755982988) < 1e-15
assert abs(my_acos(0.5) - 1.0471975511965976) < 1e-15
assert my_acos(1.0) == 0.0
assert abs(my_acos(-1.0) - 3.141592653589793) < 1e-15
assert abs(my_atan(1.0) - 0.7853981633974483) < 1e-15
assert abs(my_atan2(1.0, 1.0) - 0.7853981633974483) < 1e-15
assert abs(my_atan2(1.0, -1.0) - 2.356194490192345) < 1e-15
```

---

## 10. Python ↔ spy 调用边界

- Python 值到 spy 类型的映射：`bool`→`bool`，`int`→`i64`，`float`→`f64`，`complex`→`c128`，
  `bytes`→编译期字节串，`None`→`Null`，`syntax.Ptr[T]`→指针；用 `T(value)`（如 `u64(x)`）显式指定类型。
- 结构体实例支持字段读写、方法调用与视图语义；指针是非空的。
- `Option[T]` 与 tagged union 回到 Python 时**脱壳**：`Option[T]` 是 `T` 值或 `None`，union 是其具体变体的值。
- 复数 ↔ Python `complex`；整数枚举 ↔ Python 枚举成员。
- 可能 raise 的 spy 函数从 Python 调用时抛出一个继承 `Exception` 的结构体实例（`except Exception as e` 可捕获读字段）。
- 多返回值打包成 Python `tuple`（嵌套结果嵌套 `tuple`），按值返回的聚合解包成实例。
- 编译期结果：有自然 Python 镜像的标量转成 `int`/`float`，其余原样交出编译器对象。
- **数组目前不能跨 Python 边界**（读取/写入时直接报错）。

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
assert id64(u64(2 ** 40)) == 2 ** 40   # u64(x) picks an explicit type
```

---

## 11. 已实现 / 未实现边界

**已实现（要点）**：见上文各节——标量/指针/数组/切片/Option/tagged union/union/complex/枚举/结构体；
多返回值；泛型（函数与结构体）；内联函数（`@func(inline=True)`）与编译期 Python 函数（`PyFunction`）；
闭包（即匿名 struct）；`Comptime`/`ConstComptime`/`InferComptime`；
编译期指针（含 const 跨非内联边界）；编译期聚合/循环/反射；`defer`；异常与 `noreturn`；
panic/`catch_unwind`；原子操作；OpenMP（含归约）；Python 侧互操作（结构体/枚举/option/union/异常/多返回值）。

**未实现 / 已知限制**：

- 赋值仅 `=`（含元组解包）与 `+=`；**链式赋值 `a = b = e` 只处理第一个目标**（静默）。
- 仅位置/仅关键字参数、链式比较、单指针下标 `p[i]`（单指针只有 `p[...]`）。
- 调用点展开 `f(*args, **kwargs)`（`**kw` 被静默忽略、`*x` 报错）。
- C 变参函数的**定义**（`va_arg`）。
- 元组 `+=` / `+`（已移除）；浮点 `%`；字节串除下标/切片/`ord`/`ptr_cast`/`sstr` 外的运算。
- 数组：运行时长度的数组、数组之间的转换、数组跨 Python 边界。
- `dict[str, T]` 注解解析；option 的模式匹配；闭包 `nonlocal`；互递归闭包；闭包引用外层类型参数。
- 声明的「编译期专属」返回类型（`-> bytes` / `-> type` 等，见 `doc/issues.md` #7）。
- 破坏性 move 语义（`copyable=False` 只表示不可拷贝，不做 move 检查）。

> 逐条实现缺陷与最小复现见 `doc/issues.md`。

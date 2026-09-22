# spy - System Python

`spy` 是一个把 Python 函数 **JIT 编译成机器码**的包。你用普通的 Python 写出函数，用 `@spy.func()` 装饰注册；spy 在编译期以具体参数类型"运行"函数体（带类似 Zig 的 **comptime** 语义：`spy.typeof`、编译期 `if`、普通 Python 函数的**内联**等），生成有类型的中间表示，最后用 LLVM 编译成本地代码。之后的每次调用都是原生调用，不再经过 Python。

编译流程：

```
Python 源码 ──astgen──▶ 无类型 HIR ──interp──▶ 有类型 MIR ──lower──▶ LLVM IR/机器码
```

- `astgen`：用 `inspect.getsource` 取得函数源码，翻译成线性的无类型指令流（HIR），并把函数签名（形参/默认值/泛型参数/返回注解）转成 spy 域的 `fn.Signature`；
- `interp`：在**编译期**以具体的参数类型逐条"运行"HIR——纯编译期操作直接在 Python 中求值，需要落到运行时的操作才发出带类型的 MIR 指令（此时控制流是结构化区域树，见下文）；
- `lower`：把 MIR 机械地映射到 LLVM IR（用 `llvm` 的文本 IR 构造器），再用 llvmlite 的 MCJIT 编译成原生代码。

> 函数必须定义在真实源码文件中（`inspect.getsource` 需要源码，交互式环境里无法使用）。

## 快速上手

```python
import spy

@spy.func()               # 注册；首次调用时按实参类型编译（一个函数可有多个特化）
def add[T](a: T, b: T) -> T:
    return a + b

print(add(1, 2))          # 编译并调用 add(i64, i64)
print(add(1.5, 2.25))     # 生成另一个特化 add(f64, f64)

@spy.func()
def add_u64(a: spy.u64, b: spy.u64) -> spy.u64:
    return a + b

print(add_u64(spy.as_(2**63 - 1, spy.u64), spy.as_(2, spy.u64)))  # 大数 u64 正常往返
```

装饰器返回一个可调用的 wrapper（`dsl._RegisteredFn`）。`spy.func`/`spy.struct` 是同一个全局 context 的方法，因此注册的函数可以互相调用；普通（未装饰的）Python 函数在函数体内被调用时会被**内联**。

## 类型

Python 值在调用边界按以下规则映射：

| Python 值 | spy 类型 |
|---|---|
| `bool` | `spy.bool` |
| `int` | `spy.i64`（见 `dsl._INT_LITERAL_BITS`） |
| `float` | `spy.f64` |
| `str` | `const u8*`（只作为常量指针传递，尚不支持运算） |
| `syntax.Ptr[T]` | `sval.PointerType`（见下） |

可用的类型注解值：`spy.bool`、`spy.u8/u16/u32/u64`、`spy.i8/i16/i32/i64`、`spy.f32/f64`、`spy.void`。想以非默认类型传参时用 `spy.as_(value, T)`：

```python
add_u64(spy.as_(2**63 - 1, spy.u64), spy.as_(2, spy.u64))
```

**形参类型**按以下顺序确定：注解（替换掉已求解的泛型参数后）、实参 marshaled 出的类型、默认值的 spy 类型；形参写了注解时注解生效，实参在调用点转换到该类型。类型注解同时也是**编译期值**：`spy.typeof(x)` 返回 `x` 的静态类型，可以与类型值比较做编译期分发。

**单位类型（ZST）**：`-> None` 的 void 类型 `VoidType` 是一个**零大小类型**（zero-sized type，ZST）；零位整数 `spy.u0`，字段全为 ZST、没有字段的结构体，以及元素为 ZST 或长度为 0 的数组同样是 ZST。ZST 没有运行时表示——`to_mir_type` 对 ZST 返回 MIR 的 void 类型（`mir.VOID`，因此返回类型是 ZST 的函数就返回 void）——ZST 的 slot 不落内存、不产生 load/store，结构体里的 ZST 字段不占布局、不进入 MIR 结构体。**ZST 参数同样跳过**：不进入 MIR 签名、调用时不传参，函数体内读到的是该类型的单位值。**ZST 结果照常交付**：返回类型是 ZST 的调用同样不产生寄存器（callee 返回 void），但调用仍把结果的单位值写进它的 result location——因此 `y = f(x)`（`f` 返回 `None`）会把 `y` 绑定为单位值、类型为该 ZST，丢弃结果的表达式语句也不会留下无类型的临时 slot。估计大小与对齐（`sval.estimated_size_of`、`estimated_alignment_of`，用来决定结构体的返回与传参方式）对 ZST 分别取 0 与 1。在编译期，"无值"用其单位值 `sval.Void()` 表示。

泛型：函数可以用 PEP 695 的 `[T]` 语法（需要 Python 3.13+）。`T` 由实参类型求解；形参注解为同一个 `T` 的实参类型会被统一成一个共同类型，实参再转换到它。**声明了返回注解时，它决定该特化的返回类型**（递归函数必须有，见下）。

## 已实现的功能

### 语言与表达式

- 算术 `+ - * %`（整数）与 `+ - * /`（浮点）、一元负号、`not`（只作用于 `spy.bool`）、比较 `== != < <= > >=`；数值操作若含运行时值，生成原生指令；若两侧都是编译期常量，则直接在编译期算出结果。整数与浮点混用时提升为浮点；整数 `/`、`//`、`**`（浮点的 `//`、`**` 亦然）尚未实现，会报错。
- **条件表达式** `a if c else b`：条件必须是 `spy.bool` 值（spy 没有真值转换）。条件为编译期常量时只保留选中的分支，另一个分支不会被编译；否则在 MIR 里就是一个普通的分支，两个分支把各自的值写进**同一个 result location**（因此两侧的类型要能互相 resolve，否则报错）。
- **局部变量与块级作用域**：`name = expr` 只存回 `name` 已经绑定的那个 slot（本块或任一外层块的，包括形参），只有完全没绑定过的名字才会**声明**一个块局部变量（新分配一个可寻址 slot）；声明不会逃出所在块，块结束后该名字不再绑定，但块内对**外层**变量的赋值写的就是那个变量本身。变量在 MIR 里都是 alloca（内存），单次存取的 slot 之后由 `opt` 折回寄存器。HIR/MIR 是 wasm 式树状结构、无 phi，跨分支的写入与读取靠内存顺序语义（外层变量在分支里被赋值时不能在寄存器里，所以走内存）。块 = 函数体与各 `if` 分支体：函数体是**最外层块**，其作用域初始持有各参数。支持 `name += expr`（只支持 `+=`）与元组解包赋值（`a, b = e`，可嵌套）；赋值目标可以是名字、名字元组，或（运行时结构体值的）字段链。局部变量需要有能落到运行时的类型——用未注解的整数字面量初始化（`x = 1`）会报错，编译期值局部变量（如 `t = spy.typeof(x)`）也暂不支持。
- 一个函数必须保证每条运行路径都以 `return` 结束（否则编译报错），且各 `return` 的类型一致（或与返回注解一致）。**void 函数**（返回注解为 `-> None`，或无返回注解且函数体从不返回值）除外：允许函数体"落穿"结束，也允许裸 `return` 提前退出。

### 编译期（comptime）

- `spy.typeof(x)`：查询参数/表达式的静态类型（返回类型值），对编译期值也适用。
- `spy.compile_log(...)`：编译期打印日志（运行时无任何动作）。
- 条件为编译期常量的 `if`（例如 `if spy.typeof(a) == spy.u64: ... else: ...`）在编译期折叠，未选中的分支不会被编译。
- `and`/`or`：仅当两侧都是编译期值时才能求值（两侧都会被求值）。
- `spy.as_` 只能用在 Python 调用边界，不能出现在函数体内。

### 运行时控制流

- 运行时 `if`（条件为运行时布尔值时）会被编译为真正的分支。控制流采用**结构化的区域树**（无 basic block/phi）：
  - 每个分支要么以 `return` 结束，要么"落穿"到 `if` 之后的代码继续执行；两个分支都落穿（汇合）也允许——分支的汇合点就是 `if` 的 `End`，且对**外层**变量的赋值写的就是外层 slot（内存），因此跨汇合的状态无需 phi；
  - 支持分支嵌套、连续 `if`、`elif`（即嵌套 `if`）。
- 运行时 `if` 也允许出现在**内联函数体内**（普通函数与未装饰的结构体方法）。内联体以 `mir.Block`/`mir.Break`（类 WASM 的 `br`）结构化：分支中的内联 `return` 把值写入调用方结果位置（内存）后跳出内联体到调用方延续，因此在调用点形成内存汇合（同样无需 phi）；若各路径返回类型不一致、或部分路径落穿/裸 `return`，会像函数本身一样报错。
- 尚无循环（`while`/`for`）；也正因局部变量在 MIR 里都是 alloca（不需要 phi）且没有循环，其它需要真正 phi 的结构才尚未实现。

### 函数与调用

- **模块化编译**：编译一个函数时，把"它 + 它依赖的所有尚未编译的函数"放进同一个 LLVM module 一起 `define`；之前模块已编译过的函数在调用处作为外部符号（按链接名）引用，由进程内唯一的 MCJIT engine 解析。每个特化的原生符号名唯一（重名的函数会被分配不同的名字）。
- **递归**：直接递归、互递归、泛型函数的多类型递归都能工作（调用进行中即可解析到正在编译的函数本身）。递归要求函数返回类型能由注解确定：具体类型，或由参数绑定出的类型参数 `T`；否则报 `requires a return type annotation`。
- 普通（未注册进任何 context 的）Python 函数在体内调用时会被**内联**；未装饰的结构体方法同理。内联函数也可以递归调用自己，但内联体的参数绑定为运行期值，递归驱动参数是运行期值时编译期不会收敛（形同编译期死循环），会在内联嵌套上限（64 层）处报错——需要真正运行期递归的函数请声明为 spy 函数。
- Python 调用侧与函数体**内部**的调用都支持位置参数、关键字参数与默认参数（`*args`/`**kwargs` 尚不支持）。
- 支持把 spy 函数定义在**工厂/闭包**里：捕获的外层变量（数值、类型值、兄弟 spy 函数等）在解析时作为编译期常量嵌入；函数在注册期间引用自己的名字也正常（见 `astgen._resolve_closure`）。

## 结构体

`@spy.struct()` 把带类型注解的 Python 类变成 spy 结构体：注解字段按声明顺序构成布局；类里的方法（`@spy.func()` 装饰或未装饰）成为结构体的方法。结构体目前是**纯编译期类型**——只能在 spy 函数体内注解、构造和调用方法，Python 侧还不能构造实例。

使用示例：

```python
import spy

@spy.struct()
class Point:
    x: spy.i32
    y: spy.i32

    def total(self) -> spy.i32:      # 未装饰的方法：调用处内联
        return self.x + self.y

@spy.func()
def use_point(x: spy.i32) -> spy.i32:
    p = Point(x, 3)                  # 构造：实参按字段声明顺序写入
    p.y += 5
    return p.total() + p.x

assert use_point(2) == 12
```

语义（类似 C）：

- **字段**按声明顺序排列，支持嵌套结构体（`p.inner.a`）；字段可读、可赋值、可 `+=`（`x.h = e`、`x.h += e`，可任意嵌套）。局部结构体变量是一个 alloca，`y = x` 拷贝结构体（改 `y` 不影响 `x`）。
- **传参按值**：结构体实参是调用方结构体的一份拷贝，函数内对参数字段的修改不外溢（大于 16 字节的聚合在原生 ABI 上以指针传递，但语义仍是拷贝）。
- **方法**：在类里定义并用 `@spy.func()` 装饰的是编译成原生调用的 spy 方法（未写返回注解时按函数体推断，什么都不返回就是 void 方法）；未装饰的方法在调用处被**内联**。方法的 `self` 默认按引用传递（因此可以就地写回），`@spy.func(sfv=True)` 让 `self` 按值传递。
- **构造**：`Point(a, b)` 是一个 result-location 构造调用——`p = Point(...)` 或 `return Point(...)` 直接向目标位置的 slot 写字段，嵌套构造（`Bar(Foo(...), ...)`）把内层结构体直接建在外层字段里，不产生拷贝。位置实参按字段声明顺序写入，关键字实参按名指定；字段必须都得到值，只有零大小类型（ZST）的字段可以省略。类里**不能**定义 `__init__`（自定义构造函数尚未支持）。
- **返回结构体**：函数可以返回结构体（含方法）。返回方式由返回类型决定（`sval.returns_via_result_ptr`）：默认**小结构体（≤16 字节）按值返回**——spy 之间直接走 LLVM 聚合返回；**大结构体经 result 指针返回**——MIR 阶段就给函数追加一个 result 指针形参并返回 void，callee 直接写进调用方的结果位置。`return expr` 语句本身也走 RLS：表达式（含调用）直接写进函数的结果位置。
- **布局**：非 `extern_c` 结构体由编译器布局——字段按对齐重排（对齐小的在前），ZST 字段不占位置；只含一个非 ZST 字段的结构体，其 MIR 镜像是该字段本身（无包装结构体）；没有非 ZST 字段的镜像为 void。`@spy.struct(extern_c=True)` 保持声明顺序、并保留包装结构体，以便匹配 C ABI。
- `spy.typeof(x) == Point` 可在编译期按结构体类型分发。

**泛型结构体**：`class Foo[T]` 声明的是结构体*模板*，`Foo[i32]` 是它的一个特化：字段注解里的 `T` 被替换进去（可递归，如 `inner: Pair[T]`）。**构造时可以省略泛型参数**（`Foo(...)`）——此时特化由构造位置（result location）的类型确定（如返回注解已声明的 `return Foo(...)`、或类型已知的结构体字段）；构造位置类型未知时（如赋给一个新的局部变量），泛型实参由**写进各字段的值**的类型推断（与泛型调用解类型参数同理），推不出（如某类型参数压根不出现在任何被赋值的字段里）才必须显式写出 `Foo[T](...)`，否则报错。方法调用 `x.m(...)` 等价于 `typeof(x).m(x, ...)`——`m` 的签名以结构体**模板**作为 `self` 的类型，调用时把该特化的类型实参代入方法签名，因此返回注解 `-> T`、形参注解 `b: T1` 都会取到具体类型。方法也可以有自己的类型参数（`def m[U](self, b: U)`），它与结构体的类型参数一并参与求解（同名时方法自己的遮蔽结构体的）。结构体的类型参数还能**在函数体内当值使用**（如 `Foo[T](...)`、`spy.typeof(x) == T`）：帧里保存了本次调用解出的类型实参。

```python
@spy.struct()
class Pair[T]:
    a: T
    b: T

    @spy.func()
    def total(self) -> T:
        return self.a + self.b

@spy.func()
def use_pair(x: spy.i32) -> spy.i32:
    p = Pair[i32](x, 3)         # 显式特化：局部变量的 slot 类型未知
    return p.total()            # 方法携带 {T: i32}
```

## 指针

`syntax.Ptr[T]` 是 C 的 `T*`：`Ptr[T, C]` 的 `C` 是 const 性（默认 `False`，const 时写 `Literal[True]`），也可以是一个在调用时求解的类型参数。`syntax.ref(a)` 取 `a` 的地址（C 的 `&a`），`p[...]` 解引用（C 的 `*p`）：

```python
import spy
from spy.syntax import Ptr, ref

@spy.func()
def incr(p: Ptr[spy.i32]) -> spy.i32:
    p[...] = p[...] + 1         # ``p[...]`` 是一个左值
    return p[...]

@spy.func()
def use_incr(x: spy.i32) -> spy.i32:
    v = x
    r = incr(ref(v))            # 取局部变量的地址：incr 就地改写了 v
    return v * 100 + r          # 6 * 100 + 6
```

- **类型**：注解里的 `Ptr[T]` 被 `sval.as_value` 转成 `sval.PointerType`（元素类型 `T`、const 性 `C`）；未求解的 `C`（还是个类型参数）没有运行时表示。
- **取地址**：`ref(a)` 是一个值（指针），内容就是 `a` 的地址——`a` 可寻址（变量、形参、字段、`p[...]`）时直接就是它的地址，否则（字面量、算术结果等）先落进一个临时 slot 再取。
- **解引用**：`p[...]` 表示 `p` 指向的那个位置，和变量一样是一个**引用**（左值）：可读、可赋值（`p[...] = v`）、可 `+=`、可传给按引用传递的形参。`p.x`（自动解引用）与 `p[...].x` 取到的都是同一个字段。
- **const**：可变指针可以隐式转成 const 指针，反过来不行。
- **泛型**：指针类型参与类型参数求解——`Ptr[T]` 求解 `T`，`Ptr[T, C]` 连 const 性一起求解（`C` 解成 `True`/`False`）。

## 数组

`syntax.Array[T, N]` 是 `N` 个 `T` 排成一行构成的数组类型，`syntax.array(a1, a2, ...)` 构造数组：

```python
import spy
from typing import Literal
from spy.syntax import Array, array

@spy.func()
def total(a: Array[spy.i32, Literal[2]]) -> spy.i32:
    return a[0] + a[1]

@spy.func()
def use_array(x: spy.i32) -> spy.i32:
    a = array(x, x + 1, length=2)   # ``length`` 只给类型检查看
    a[0] = 7
    return total(a) + a[0]
```

- **类型**：注解里的 `Array[T, N]` 被 `sval.as_value` 转成 `sval.ArrayType`（元素类型 `T`、长度 `N`）。长度是一个**值**类型参数，所以写的时候要用 `Literal[N]`（和指针的 const 性写 `Literal[True]` 一样）；元素类型也可以是未求解的类型参数。
- **构造**：`array(a1, a2, ...)` 与结构体构造一样是 result-location 构造调用——元素按顺序直接写进数组的存储，嵌套构造（数组套数组、结构体里的数组字段）不产生拷贝。**长度取实参的个数**；元素类型取目标位置已声明的类型，否则取所有元素的共同类型（每个元素都得有一个能落地的类型，所以 `array(1, 2)` 这种没写类型的整数字面量会报错）。因为 Python 类型系统无法从实参推出长度，`array` 签名里带一个**只为类型检查**服务的 `length: N = 0` 关键字（不写时 pyright 认为长度是 0）；编译以实参个数为准，并且忽略这个关键字。
- **下标**：`a[i]` 是第 `i` 个元素的**位置**（左值）——可读、可赋值、可 `+=`，也可以继续取字段（`a[0].x`）或继续下标（`a[0][1]`）。下标类型暂时固定为 `u64`（将来的 `usize` 会按目标平台取具体整数类型）；编译期常量下标会检查越界，运行期下标不检查。指向数组的指针同样可以下标：`p[...][i]`。
- **ZST**：元素是 ZST、或长度为 0 的数组本身是 ZST——没有存储、没有运行时表示：每个元素都等于元素类型的单位值，`a[i]` 不产生地址（`a[i] = v` 也就什么都不写）。
- **返回与传参**：和结构体同一套规则（`sval.returns_via_result_ptr`/`pass_by_ref`）——不超过 16 字节按值、更大的经 result 指针；数组的值可以整体拷贝（`b = a`）。

## 模块结构

| 文件 | 作用 |
|---|---|
| `__init__.py` | 公开接口：`func`、`struct`、`typeof`、`compile_log`、`as_`，以及各类型常量（`spy.i32` 等） |
| `dsl.py` | `func`/`struct` 装饰器、注册与全局 context（`_Context`）、Python 侧调用入口（实参绑定、特化、原生调用） |
| `astgen.py` | 源码 → 无类型 HIR；把函数签名（注解/默认值/泛型参数）转成 spy 域的 `fn.Signature` |
| `hir.py` | 无类型 HIR 指令定义 |
| `interp.py` | 编译期运行 HIR → 有类型 MIR（comptime 语义所在）；结构体的字段寻址、方法分发与就地构造 |
| `mir.py` | MIR 类型、指令与区域树定义 |
| `opt.py` | MIR 清理：把单次存取的 slot 折回寄存器、删除无用 slot |
| `llvm.py` | LLVM IR 的文本构造器 |
| `lower.py` | MIR → LLVM IR → 机器码（llvmlite MCJIT） |
| `fn.py` | 函数签名（`Signature`：形参绑定、类型参数求解、返回类型推导）、函数值与编译产物、链接名表（`SymbolTable`）与函数入口 thunk |
| `sval.py` | spy 类型系统（含结构体类型）、编译期值、Python 值 → spy 域的映射（`as_value`）与类型参数约束求解（`TypeVarSolver`） |
| `syntax.py` | 函数体内使用的语法标记：指针类型 `Ptr`、取地址 `ref`、数组类型 `Array` 与构造 `array` |
| `errors.py` | `SpyError`、`CompileError`、`TypeMismatchError`（同时是 `TypeError` 子类） |
| `binop.py` | 运算符的字面量类型 |
| `builtins.py` | 函数体内使用的 `spy.*` builtin |
| `util.py` | 共用工具 |
| `tests.py` | 集成测试 |

## 尚未实现 / 已知限制

- `while`/`for` 循环、运行时 `and`/`or`（目前只支持编译期操作数）。
- 赋值仅支持 `=`（含元组解包）与 `+=`（无链式赋值 `a = b = e`、其它增强赋值、类型标注赋值 `x: T = ...`、编译期值局部变量）。
- `*args`/`**kwargs`、仅位置/仅关键字参数、链式比较、对**指针**的下标 `p[i]`（指针只有解引用 `p[...]` 可用；数组的 `a[i]` 已实现）。
- 整数 `/`、`//`、`**`（浮点的 `//`、`**` 亦然）；字符串的运算。
- 结构体：Python 侧实例表示（因此返回结构体、或带结构体参数的函数还不能从 Python 侧直接调用）、通过类名访问方法（如 `Foo[i32].m(x)`）、结构体整体比较。
- 数组：运行时长度的数组（`syntax.MultiPtr`）、切片、数组之间的转换（如 `i32[2]` → `i64[2]`）、以及 Python 侧实例表示（带数组参数/返回值的函数还不能从 Python 侧直接调用）。
- 普通 Python 函数的内联不支持运行期递归（递归驱动参数是运行期值时会在内联嵌套上限处报错，而非编译期展开）；运行期的函数值调用（把函数存进变量/字段后再调用）也尚未实现。

## 运行测试

```sh
python3 -m unittest spy.tests        # 只跑 spy 的测试
python3 run_tests.py                  # 仓库根目录：跑全部测试
```

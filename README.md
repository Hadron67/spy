# spy - System Python

`spy` 是一个把 Python 函数 **JIT 编译成机器码**的包。你用普通的 Python 写出函数，通过 `JitContext` 的装饰器注册，spy 在编译期以具体参数类型"运行"函数体（带类似 Zig 的 **comptime** 语义：`spy.typeof`、编译期 `if` 等），生成有类型的中间表示，最后用 LLVM 编译成本地代码。之后的每次调用都是原生调用，不再经过 Python。

编译流程：

```
Python 源码 ──astgen──▶ 无类型 HIR ──interp──▶ 有类型 MIR ──lower──▶ LLVM IR/机器码
```

- `astgen`：用 `inspect.getsource` 取得函数源码，翻译成线性的无类型指令流（HIR），并把函数签名（形参/默认值/泛型参数/返回注解）转成 spy 域的 `fn.Signature`；
- `interp`：在**编译期**以具体的参数类型逐条"运行"HIR——纯编译期操作直接在 Python 中求值，需要落到运行时的操作才发出带类型的 MIR 指令（此时控制流是结构化区域树，见下文）；
- `lower`：把 MIR 机械地映射到 LLVM IR，用 llvmlite（MCJIT）编译成原生代码。

> 函数必须定义在真实源码文件中（`inspect.getsource` 需要源码，交互式环境里无法使用）。

## 快速上手

```python
import symlat.spy as spy

cache = spy.JitContext()

@cache.jit()              # 惰性：首次调用时按实参类型编译（可多个特化）
def add[T](a: T, b: T) -> T:
    return a + b

@cache.aot()              # 惰性：首次使用时按（必须具体、齐全的）注解编译；一个函数只有一个编译实例
def add_u64(a: spy.u64, b: spy.u64) -> spy.u64:
    return a + b

print(add(1, 2))          # 编译并调用 add(spy.i32, spy.i32)
print(add(1.0, 2.0))      # 生成另一个特化 add(spy.f64, spy.f64)
print(add_u64(2**63 - 1, 2))  # 大数 u64 正常往返
```

装饰器返回一个可调用的 wrapper；同一个 `JitContext` 里注册的函数可以互相调用。

## 类型

Python 值在调用边界按以下规则映射：

| Python 值 | spy 类型 |
|---|---|
| `bool` | `spy.bool` |
| `int` | `spy.i32`（默认有符号 32 位） |
| `float` | `spy.f64` |
| `str` | 编译为 `u8` 数组（目前只作为 `const u8*` 常量传递，不支持运算） |

可用的类型注解值：`spy.bool`、`spy.u8/u16/u32/u64`、`spy.i8/i16/i32/i64`、`spy.f32/f64`。想以非默认类型传参时用 `spy.as_(value, T)`：

```python
add_u64(spy.as_(2**63 - 1, spy.u64), spy.as_(2, spy.u64))
```

类型注解同时也是**编译期值**：`spy.typeof(x)` 返回 `x` 的静态类型，可以与类型值比较做编译期分发。

**单位类型（ZST）**：`-> None` 的 void 类型 `VoidType` 是一个**零大小类型**（zero-sized type，ZST）。ZST 没有运行时表示——`to_mir_type` 对任何 ZST 都返回 `None`（MIR 里不存在 void 类型，函数返回类型是 ZST 就返回 void，即 `None`）——ZST 的 slot 不落内存、不产生 load/store，结构体里的 ZST 字段不占布局、不会进入 MIR 结构体（零位整型、字段全为 ZST 或空的结构体也属于 ZST）。**ZST 函数参数同样跳过**：不进入 MIR 签名、调用时不传参（方法按值绑定的 `self` 若是 ZST 结构体也一样跳过），函数体内该参数读到的是类型的单位值。在编译期，"无值"用其单位值 `sval.Void()` 表示，取代原先的 `None` 哨兵；读取 ZST 字段直接得到该类型的单位值。

泛型：函数可以用 PEP 695 的 `[T]` 语法（需要 Python 3.13+）。在 `jit` 模式下参数注解不影响特化的选取（由实参 marshaled 出的类型决定），但注解为同一个 `T` 的参数必须统一成同一类型；**声明了返回注解时，它决定该特化的返回类型**（递归函数必须有，见下）。

## 已实现的功能

### 语言与表达式

- 算术 `+ - * %`（整数）与 `+ - * /`（浮点）、一元负号、`not`、比较 `== != < <= > >=`；整数与浮点混用时提升为浮点。
- 数值操作若含运行时值，生成原生指令；若两侧都是编译期常量，则直接在编译期算出结果。
- **局部变量与块级作用域**：`name = expr` 声明一个块局部变量（为该块新分配一个可寻址 slot，遮蔽外层同名绑定）并初始化；同一块内对该名的后续赋值只写回该 slot。变量全部走 alloca（内存），HIR/MIR 是 wasm 式树状结构、无 phi，跨分支的写入与读取都靠内存顺序语义。块 = 函数体与各 `if` 分支体：函数体是**最外层块**，其作用域初始持有各参数（因此函数体顶层的 `x = ...` 写回参数 slot）。声明不会逃出所在块：块结束后该名字不再绑定，块外读它会落到外层同名绑定（Python 全局/捕获变量）或报未定义。支持 `name += expr`；暂不支持 `a = b = e` 链式赋值、`a, b = ...` 解包、除 `+=` 外的增强赋值，也暂不支持编译期值局部变量（如 `t = spy.typeof(x)`）。
- 一个函数必须保证每条运行路径都以 `return` 结束（否则编译报错），且各 `return` 的类型一致（或与返回注解一致）。**void 函数**（返回注解为 `-> None`，或方法/无注解函数体从不返回值）除外：允许函数体“落穿”结束，也允许裸 `return` 提前退出。

### 编译期（comptime）

- `spy.typeof(x)`：查询参数/表达式的静态类型（返回类型值），也支持编译期值。
- `spy.compile_log(...)`：编译期打印日志（运行时无任何动作）。
- 条件为编译期常量的 `if`（例如 `if spy.typeof(a) == spy.u64: ... else: ...`）在编译期折叠，未选中的分支不会被编译。
- `spy.as_` 只能用在 Python 调用边界，不能出现在函数体内。

### 运行时控制流

- 运行时 `if`（条件为运行时布尔值时）会被编译为真正的分支。控制流采用**结构化的区域树**（无 basic block/phi，为将来把 `defer` 之类保留到 MIR→LLVM 阶段再展开而设计）：
  - 每个分支要么以 `return` 结束，要么"落穿"到 `if` 之后的代码继续执行；两个分支都落穿（汇合）也允许——分支的汇合点就是 `if` 的 `End`（MIR 里落穿的分支本来就跳到它后面的代码），且块级作用域（分支内声明的名字不逃出分支）使跨汇合的状态只走内存槽，无需 phi；
  - 支持分支嵌套、连续 `if`、`elif`（即嵌套 `if`）。
- 运行时 `if` 也允许出现在**内联函数体内**（普通函数与未装饰的结构体方法）。内联体以 `Block`/`Break`（类 WASM 的 `br`）结构化：分支中的内联 `return` 把值写入调用方结果位置（内存）后跳出内联体到调用方延续，因此在调用点形成内存汇合（同样无需 phi）；若各路径返回类型不一致、或部分路径返回值而另一些路径落穿/裸 `return`，会像函数 proper 一样报错。
- 尚无循环（`while`/`for`）；也正因局部变量全部走 alloca（不需要 phi）且没有循环，其它需要真正 phi 的结构才尚未实现。

### 函数与调用

- **模块化编译**：编译一个函数时，把"它 + 它依赖的所有尚未编译的函数"放进同一个 LLVM module 一起 `define`；之前模块已编译过的函数在调用处作为外部符号引用、链接期把地址接上。每个特化的原生符号名唯一（重名的函数会被分配不同的名字）。
- **递归**：直接递归、互递归、泛型函数的多类型递归都能工作（调用进行中即可解析到正在编译的函数本身）。递归要求函数返回类型能由注解确定：具体类型，或由参数绑定出的类型参数 `T`；否则报"需要返回类型注解"。
- 普通（未注册进任何 context 的）Python 函数在体内调用时会被**内联**；函数体内也可以递归调用自己，但内联体的参数绑定为运行期值，递归驱动参数是运行期值时编译期不会收敛（形同编译期死循环），会在内联嵌套上限处报错——需要真正运行期递归的函数请声明为 spy 函数（见下）。结构体中未装饰的方法同样是普通函数，被调用时内联。
- Python 调用侧支持位置参数、关键字参数、默认参数；函数体**内部**的调用只支持位置参数，但被调 spy 函数的默认参数仍可生效。
- 支持把 spy 函数定义在**工厂/闭包**里：
  - 捕获的外层变量（数值、类型值、兄弟 spy 函数等）在解析时作为编译期常量嵌入——`def make(k): @cache.jit() def f(x): return x * k`；
  - `@aot` 函数在工厂内注册、甚至自引用（递归）都正常。
- 同一个原始函数也可以先后注册进**不同**的 `JitContext`，每个 context 各自持有独立的编译状态。

## 结构体

`@cache.struct()` 把带类型注解的 Python 类变成 spy 结构体：注解字段按声明顺序构成布局；类里的方法（`@cache.aot`/`@cache.jit` 装饰或未装饰）成为结构体的 spy 方法。结构体实例在 Python 侧持有与 LLVM 布局一致的原生内存（ctypes），因此既能在 Python 侧直接读写字段、调用方法，也能按值或按指针传给 spy 函数。

使用示例：
```python
import spy

cache = spy.JitContext()

@cache.struct()
class Foo:
    a: spy.u64
    b: spy.u32

@cache.struct()
class Bar:
    foo: Foo
    h: spy.i32

    @cache.aot(ptr_self = True)  # 相当于 def hkm(self: spy.ptr(Bar))：self 按指针传递
    def hkm(self):
        self.foo.a += 34
        self.h += 2

@cache.aot()
# def example(bar: Bar) -> None 是 void 函数：函数体可以没有 return
def example(bar: Bar) -> None:
    bar.hkm()
    bar1 = Bar(Foo(1, 3), 5)   # 在 jit 函数里新建结构体
    bar1.hkm()

bar = Bar(Foo(1, 2), 3)
example(bar)
assert bar.h == 3              # 值语义：example 拿到的是 bar 的副本，改动不外溢
bar.hkm()                      # Python 侧方法调用：ptr_self 使改动直接落在 bar 的内存上
assert bar.h == 5
assert bar.foo.a == 35
```

语义（类似 C）：

- **字段**按声明顺序排列，支持嵌套结构体（`bar.foo.a`）；字段可读、可赋值、可 `+=`（`x.h = e`、`x.h += e`、任意嵌套链）。局部结构体变量是一个 alloca，`y = x` 拷贝结构体（改 `y` 不影响 `x`）。
- **传参按值**：结构体实参是调用方结构体的一份拷贝，函数内对参数字段的修改不外溢。
- **指针参数**：方法的 `self` 可用 `@cache.aot(ptr_self=True)`（相当于注解 `self: spy.ptr(Bar)`）按指针传递——`bar.hkm()` 会就地修改 `bar`；不带 `ptr_self`（默认）相当于 `self: Bar`（按值）。`spy.ptr(...)`/`spy.ref(...)` 作为普通参数类型/实参语法尚未实现。
- **返回结构体**：函数可以返回结构体（含方法）。返回方式由返回类型决定，统一在类型层判定（`type.returns_via_result_ptr`，把判定写成独立函数以便将来扩展：数组类型、或允许用户按结构体手动指定）：默认**小结构体（≤16 字节）按值返回**——spy 之间直接走 LLVM 聚合返回，Python 侧由生成的 out-buffer thunk 转换；**大结构体经 result 指针返回**——MIR 阶段就给函数追加一个 result 指针形参并返回 void，callee 直接写进调用方的结果位置，`return f(...)` 链零拷贝直通。`return expr` 语句本身也走 RLS：表达式（含调用）直接写进函数的结果位置。
- **构造**：`Bar(a, b)` 是一个 result-location 构造调用——在 jit 函数里 `x = Bar(...)` 直接向 `x` 的 slot 写字段（Python 侧则创建原生实例）。没有自定义 `__init__` 时，实参按声明顺序写入字段；类里定义 `def __init__(self, ...)` 则它是构造函数（`self` 指向结果内存，可直接赋字段；`@aot`/`@jit` 或未装饰均可）。
- **方法**：在类里定义并用 `@cache.aot(...)`/`@cache.jit(...)` 装饰的是编译成原生调用的 spy 方法（未写返回注解时按函数体推断，什么都不返回就是 void 方法）；未装饰的方法在调用处被**内联**。方法也可在结构体创建之后添加：`Bar.methods['name'] = handle或函数`。
- `spy.typeof(x) == Foo` 可在编译期按结构体类型分发。

## 模块结构

| 文件 | 作用 |
|---|---|
| `dsl.py` | `JitContext`、`jit`/`aot`/`struct` 装饰器、注册与模块编译调度、外部符号链接、结构体实例（ctypes）的构造与封送 |
| `astgen.py` | 源码 → 无类型 HIR；把函数签名（注解/默认值/泛型参数）转成 spy 域的 `fn.Signature` |
| `hir.py` | 无类型 HIR 指令定义 |
| `interp.py` | 编译期运行 HIR → 有类型 MIR（comptime 语义所在）；结构体的字段寻址、方法分发与构造 |
| `mir.py` | MIR 类型（含结构体类型）、指令与区域树定义 |
| `lower.py` | MIR → LLVM IR → 机器码（MCJIT）；结构体按值传参 |
| `fn.py` | 函数签名（`Signature`：形参绑定、类型参数求解、返回类型推导）与函数值/注册条目：`LazyJitFunction`（jit）、`FunctionValue`（aot） |
| `sval.py` | spy 类型系统（含结构体类型）、Python 值 → spy 域的映射（`as_value`）与类型参数约束求解（`TypeVarSolver`） |
| `errors.py` | `CompileError`、`TypeMismatchError`（`TypeMismatchError` 同时是 `TypeError` 子类） |

## 尚未实现 / 已知限制

- `while`/`for` 循环、运行时 `and`/`or`（目前只支持编译期操作数）。
- 赋值仅支持单 `Name`/字段链目标的 `=` 与 `+=`（无链式/解包/其它增强赋值、无类型标注赋值 `x: T = ...`、无编译期值局部变量）。
- 函数体内调用带关键字参数、链式比较、`*args`/`**kwargs`、仅位置/仅关键字参数。
- 整数 `/`（会提示改用浮点）、`//`、`**`；字符串的运算；非结构体值的运行期属性访问。
- 结构体：不能整体返回/比较结构体值；`spy.ptr(...)`/`spy.ref(...)`、结构体字段默认值等尚未实现。
- 普通 Python 函数的内联不支持运行期递归（递归驱动参数是运行期值时编译在内联嵌套上限处报错，而非编译期展开）；`defer`、运行时函数值调用等尚未实现。

## 运行测试

```sh
python3 -m unittest spy.tests        # 只跑 spy 的测试
python3 run_tests.py                  # 仓库根目录：跑全部测试
```

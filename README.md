# spy - System Python

`spy` 是一个把 Python 函数 **JIT 编译成机器码**的包。你用普通的 Python 写出函数，用 `@spy.func()` 装饰注册；spy 在编译期以具体参数类型"运行"函数体（带类似 Zig 的 **comptime** 语义：`spy.typeof`、编译期 `if`、普通 Python 函数的**内联**等），生成有类型的中间表示，最后用 LLVM 编译成本地代码。之后的每次调用都是原生调用，不再经过 Python。

编译流程：

```
Python 源码 ──astgen──▶ 无类型 HIR ──interp──▶ 有类型 MIR ──lower──▶ LLVM IR/机器码
```

- `astgen`：用 `inspect.getsource` 取得函数源码，翻译成线性的无类型指令流（HIR），并把函数签名（形参/默认值/泛型参数/返回注解）转成 spy 域的 `fn.Signature`；
- `interp`：在**编译期**以具体的参数类型逐条“运行”HIR——纯编译期操作直接在 Python 中求值，需要落到运行时的操作才发出带类型的 MIR 指令（此时控制流是一张以入口块为根的基本块图，每个块以 `jmp`/`br`/`ret` 结束，见下文）；
- `lower`：把 MIR 机械地映射到 LLVM IR（用 `llvm` 的文本 IR 构造器），再用 llvmlite 的 LLJIT（ORC JIT）编译成原生代码。

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
| `complex` | `spy.c128`（`ComplexType[FloatType(64)]`） |
| `bytes` | 编译期字节串（`sval.BytesType`，只在编译期存在；用 `std.core.gstr`/`sstr` 转成运行时指针/切片。`str` 字面量在解析时被编码为 `bytes`，spy 函数内不允许 `str`） |
| `None` | `Null`（`sval.Null()`，类型 `NullType`；是 `Option[T]` 的“缺席”值，也可以当 void 值用） |
| `syntax.Ptr[T]` | `sval.PointerType`（见下） |
| `syntax.Option[T]` | `sval.OptionType`（见下） |

可用的类型注解值：`spy.bool`、`spy.u8/u16/u32/u64`、`spy.i8/i16/i32/i64`、`spy.f32/f64`、`spy.c64/c128`、`spy.void`。C 整数类型 `spy.c_char/c_uchar`、`spy.c_short/c_ushort`、`spy.c_int/c_uint`、`spy.c_long/c_ulong`、`spy.c_longlong/c_ulonglong` 也可用作注解，它们的宽度（以及 `c_char` 的符号）由目标平台的 `TargetInfo` 决定（`c_long` 在 Windows 为 32 位、在常见 64 位 Unix 上为 64 位）。想以非默认类型传参时用 `spy.as_(value, T)`：

```python
add_u64(spy.as_(2**63 - 1, spy.u64), spy.as_(2, spy.u64))
```

**形参类型**按以下顺序确定：注解（替换掉已求解的泛型参数后）、实参 marshaled 出的类型、默认值的 spy 类型；形参写了注解时注解生效，实参在调用点转换到该类型。类型注解同时也是**编译期值**：`spy.typeof(x)` 返回 `x` 的静态类型，可以与类型值比较做编译期分发。

**复数**：`sval.ComplexType`（`spy.c64 = ComplexType(f32)`、`spy.c128 = ComplexType(f64)`）是实部与虚部各为元素浮点类型的复数，运行时落成 `{real, imag}` 结构体。Python `complex` 映射到 `spy.c128`；实数（浮点或整数）到复数有一条隐式转换 `T -> Complex[T]`（虚部为 0，`Complex[f32]` 可扩宽到 `Complex[f64]`）。支持四则运算 `+ - * /`（逐分量内联计算：编译期常量在编译期折叠，运行期发浮点指令），以及字段访问 `z.real` / `z.imag`。编译期反射中 `std.reflect.type_info` 会给出带 `elem`（元素浮点类型）的 `ComplexType` 变体。

**空类型**：`sval.EmptyType`（`empty`）是**空类型**——它一个值都没有；注解里的 `typing.Never`（及其已弃用的别名 `NoReturn`）映射到它，因此可以显式声明一个永远不返回值的函数（`-> Never`，见异常一节）。它是“函数体从不交付值”时那个结果的值部分类型（推理得出的情形亦然），因此没有任何 `return` 路径：这样的函数体里不能出现 `return`，也不能落穿到函数体末尾（都报编译错）。对编译器它表现得像个 ZST：`to_mir_type` 返回 `None`，slot 里存的是“无值”标记 `Void()`。

**payload union**：异常的 payload 是一个**无标签并集** `sval.UnionType`（`mir.UnionType` 的存储就是最大变体，各变体都在偏移 0，因此读写都靠指针重解释 / `BitCast`）；**union 值之间不能转换**，只能重解释存储。不占存储的并集（没有变体，或变体全为 ZST）在编译期只有 `sval.UnionValue`，里面只保存“是哪个 union”——并集值不携带变体（变体由旁边的错误码指明）。

**单位类型（ZST）**：`-> None` 的 void 类型 `VoidType` 是一个**零大小类型**（zero-sized type，ZST）；零位整数 `spy.u0`，字段全为 ZST、没有字段的结构体，以及元素为 ZST 或长度为 0 的数组同样是 ZST。ZST 没有运行时表示——`to_mir_type` 对 ZST 返回 `None`（返回类型是 ZST 的函数因此不返回值）——ZST 的 slot 不落内存、不产生 load/store，结构体里的 ZST 字段不占布局、不进入 MIR 结构体。**ZST 参数同样跳过**：不进入 MIR 签名、调用时不传参，函数体内读到的是该类型的单位值。**ZST 结果照常交付**：返回类型是 ZST 的调用同样不产生寄存器（callee 返回 void），但调用仍把结果的单位值写进它的 result location——因此 `y = f(x)`（`f` 返回 `None`）会把 `y` 绑定为单位值、类型为该 ZST，丢弃结果的表达式语句也不会留下无类型的临时 slot。ZST 的大小是 0，对齐量则没有固定值，而是由 `sval.alignment_of` 按结构算出（数组取元素的对齐，即 `align_of(T[0]) == align_of([?]T) == align_of(T)`；结构体取所有成员对齐量的最大值，空结构体为 1；联合体取所有变体的最大值；单变体 tagged union 取该变体；其余叶子 ZST（void、`u0`、`Null`、`undefined`……）为 1）。`std.mem.layout_of`（以及基于它的 `size_of`/`align_of`）把这些暴露给代码：具体的字节数由 lower 阶段发射的 `mir.Sizeof`/`mir.Alignof` 确定，只有 ZST 和长度未定数组在编译期折叠。**已知不一致（待定）**：普通（有存储的）结构体的 MIR 镜像会丢掉 ZST 字段，因此「结构体取所有成员对齐量的最大值」这条规则只在 ZST 结构体上成立——有存储的结构体里，ZST 字段的对齐量不影响结果（与当前 MIR/LLVM 布局一致，但与「所有成员」的直觉不符，如何统一尚未决定）。在编译期，“无值”用其单位值 `sval.Void()` 表示。

泛型：函数可以用 PEP 695 的 `[T]` 语法（需要 Python 3.13+）。`T` 由实参类型求解；形参注解为同一个 `T` 的实参类型会被统一成一个共同类型，实参再转换到它。**声明了返回注解时，它决定该特化的返回类型**（递归函数必须有，见下）。

## 已实现的功能

### 语言与表达式

- 算术 `+ - * / // % **`（整数与浮点）、位运算 `| & ^ << >>` 与非 `~`、一元负号、`not`（只作用于 `spy.bool`）、比较 `== != < <= > >=`；数值操作若含运行时值，生成原生指令；若两侧都是编译期常量，则直接在编译期算出结果。整数 `/` 是真除法（先转换到 `CompileVars.int_div_type`，默认 `f64`，结果为浮点）；整数 `//` 默认向下取整（可用 `CompileVars.int_trunc_div` 改为向零截断），浮点 `//` 是 `floor(a/b)`（`mir.Arith('/')` + `mir.Floor`，后者 lower 成 `llvm.floor.*`）；`**` 的编译期常量整数指数量用快速幂展开（超过 `CompileVars.max_exp_unroll`，默认 4096，则改用运行时循环），运行时整数指数（有符号或无符号）是 `f64` 的运行时循环（有符号先取绝对值、末尾按需取倒数，循环变量用 `mir.Phi`），浮点指数是 `mir.Pow`（lower 成 `llvm.pow.*`）；浮点 `%` 尚未实现，会报错。每个二元算符都有对应的复合赋值 `a <算符>= b`。
- **条件表达式** `a if c else b`：条件必须是 `spy.bool` 值（spy 没有真值转换）。条件为编译期常量时只保留选中的分支，另一个分支不会被编译；否则在 MIR 里就是一个普通的分支，两个分支把各自的值写进**同一个 result location**（因此两侧的类型要能互相 resolve，否则报错）。
- **`and`/`or`（短路）**：一条链（`a and b and c` 按「链」处理，不是右结合递归）降级成一个 `hir.Block`：每个操作数依次求值、写进 result location，再按它的布尔值 `hir.BreakIf` 跳出块（`and` 在假时跳出、`or` 在真时跳出）——跳出的那个操作数就是结果。编译期操作数让未选中的部分整段不被走（不求值、不参与类型推导）。操作数与结果都必须是 `spy.bool`（`AsBool` 转换，将来支持 `__bool__` 后同样适用）。最后一个操作数用 result location 直接生成（不产生拷贝）；其余操作数先写进 result location 再判断。每个操作数各开一个作用域：其中的 `:=` 不外泄（那个操作数不一定求值过）。
- **局部变量与块级作用域**：`name = expr` 只存回 `name` 已经绑定的那个 slot（本块或任一外层块的，包括形参），只有完全没绑定过的名字才会**声明**一个块局部变量（新分配一个可寻址 slot）；声明不会逃出所在块，块结束后该名字不再绑定，但块内对**外层**变量的赋值写的就是那个变量本身。未标注的变量在 MIR 里都是 alloca（内存），单次存取的 slot 之后由 `opt` 折回寄存器。HIR 是 wasm 式树状结构、MIR 是基本块图，都无 phi，跨分支或跨迭代的写入与读取靠内存顺序语义（外层变量在分支里被赋值时不能在寄存器里，所以走内存）。块 = 函数体与各 `if` 分支体：函数体是**最外层块**，其作用域初始持有各参数。支持 `name <算符>= expr`（全部二元算符）与元组解包赋值（`a, b = e`，可嵌套）；赋值目标可以是名字、名字元组，或（运行时结构体值的）字段链。局部变量需要有能落到运行时的类型——用未注解的整数字面量初始化（`x = 1`）会报错，这时用类型标注声明它的类型即可（见下）。
- **类型标注的局部变量**：`name: T = expr` 声明一个新的块局部变量，并直接给定它的类型 `T`（值会转换到该类型；slot 立即落成运行时内存）。`name: Comptime` 声明一个**编译期变量**（值不在内存里，而在一个编译期 box 里），`name: Comptime[T]` 同时给定它的类型 `T`——这是局部变量持有纯编译期值（如类型：`t: Comptime = spy.typeof(x)`）的方式。标注里的类型就是一个编译期值表达式（可以命名类型参数 `T`，调用时解出具体类型）。标注即声明：同一个名字不能被标注两次。除了写在标注里，也可以用语句标记 `syntax.comptime()` 声明编译期变量（见下文“编译期”）。
- 一个函数必须保证每条运行路径都以 `return` 结束（否则编译报错），且各 `return` 的类型一致（或与返回注解一致）。**void 函数**（返回注解为 `-> None`，或无返回注解且函数体从不返回值）除外：允许函数体"落穿"结束，也允许裸 `return` 提前退出。

### 编译期（comptime）

- `spy.typeof(x)`：查询参数/表达式的静态类型（返回类型值），对编译期值也适用。它是一个 `syntax` 标记而非常规调用：实参**只被类型检查**（其中引用的 spy 函数会被触发编译），但不会发射任何代码，因此没有任何运行时行为——`spy.typeof(bar(foo(), x))` 只类型化（并编译）`bar`/`foo`，不会真正调用它们。
- `spy.compile_log(...)`：编译期打印日志（运行时无任何动作）。
- 条件为编译期常量的 `if`（例如 `if spy.typeof(a) == spy.u64: ... else: ...`）在编译期折叠，未选中的分支不会被编译。
- `spy.as_` 只能用在 Python 调用边界，不能出现在函数体内。
- `Comptime` 标注：`t: Comptime = spy.typeof(x)` 声明一个**编译期变量**，`t: Comptime[T] = ...` 再给定它的类型（见上文“类型标注的局部变量”）；只有编译期变量能持有类型这类纯编译期值。参数也可以这样标注（`def f(x: Comptime[void])`）。同一个意思也可以写成一条语句标记 `syntax.comptime()`（与 `syntax.unroll()` 同类的标记，必须紧接在被标记的声明之前）：`syntax.comptime()` + `a: T = e` 等价于 `a: Comptime[T] = e`，不带类型注解的 `syntax.comptime()` + `a = e` 等价于 `a: Comptime = e`（这时它必须声明一个新名字；标注已经写了 `Comptime` 时再加标记会报错）。

  编译期分**浅**、**深**两层（代码里叫 inline / comptime）：
  - **浅（inline）**：值本身不是运行时值（`RuntimeVal`），可以放进编译期 box 或编译期聚合而不落内存。`name: Comptime` 要求的只是这一层，所以一个编译期变量能持有一个编译期结构（类型、元组、结构体、数组……），即使它里面的元素是运行时值（例如多返回值打包成的元组）。
  - **深（comptime）**：整个值在编译期就完全确定，可以在 Python 里直接算出来。只有这些地方要求深层编译期：编译期 `if`、编译期一元运算的折叠，以及给 `Comptime` 参数传参（实参是运行时值时报错，而不是被默默丢掉）。
- **字节串（`bytes`）**：字节串字面量/值只在编译期存在（`sval.BytesType`，没有运行时表示；`str` 字面量在 `astgen` 里被编码成 `bytes`，spy 函数内不允许 `str`）。编译期字节串可**下标** `b[i]` 与**切片** `b[a:c]`（下界缺省为 0、上界缺省为长度、step 必须为 1，越界报错），两者都产出一个*引用*（`ComptimeVal(sval.ConstRef(<子串>))`；`b[i]` 是单字节子串）。`ord(b)` 取**恰好一个字节**的字节串的编码（一个无类型整数，见 `hir.Ord`）。要把字节串落成运行时值用 `std.core.gstr(b) -> ConstMultiPtr[u8]`（一个持有其字节的全局常量，字节末尾附一个 NUL 结束符）与 `std.core.sstr(b) -> ConstSlicePtr[u8]`（`{gstr(b), 字节数}`，长度**不含**那个 NUL）；内容相同的字节串共用一个全局常量。持有字节串的局部变量必须写成 `Comptime` 变量。

### 运行时控制流

- 运行时 `if`（条件为运行时布尔值时）会被编译为真正的分支。MIR 的控制流是一张**基本块图**（`mir.BasicBlock`，不专门注册、从入口块沿出边可达；循环的回边使它不再是树）：每个块以转移指令结束——无条件转移 `mir.Jmp`、二分支转移 `mir.Br`、或返回 `mir.Ret`（无 phi）：
  - 运行时 `if` 把当前块以 `br` 一分为二；每个分支要么以 `return`（`ret`）结束，要么“落穿”到 `if` 之后的代码继续执行（分支块以 `jmp` 跳向汇合块）。两个分支都落穿（汇合）也允许——对**外层**变量的赋值写的就是外层 slot（内存），因此跨汇合的状态无需 phi；
  - 支持分支嵌套、连续 `if`、`elif`（即嵌套 `if`）。
- 运行时 `if` 也允许出现在**内联函数体内**（普通函数与未装饰的结构体方法）。内联体直接续写调用点所在的块，并为调用方的延续预留一个**出口块**：内联 `return` 把值写入调用方结果位置（内存）后以 `jmp` 跳到出口块，落穿的内联体也汇入出口块，因此在调用点形成内存汇合（同样无需 phi）；若各路径返回类型不一致、或部分路径落穿/裸 `return`，会像函数本身一样报错。
- **`while`/`while`-`else` 循环**：`while cond: body` 降级为一个死循环块（`hir.Loop`…`hir.End`）：每轮在块头重新求值 `cond`，为真则执行 `body`，块体末尾跳回块头；为假则先执行 `else` 子句（若有）再**跳出**循环。`break`（`hir.BreakLoop`）直接跳到循环之后（因此 `while`-`else` 的 `else` 只在条件自然为假时执行，被 `break` 跳过，与 Python 一致），`continue` 跳回块头（重新求值条件，并跳过本轮剩余 body）。循环在 MIR 里就是带回边的普通块，循环携带的变量是普通 alloca（内存），因此无需 phi。条件的 `and` 链另见图下的「`and`/`or` 的短路」。编译期为假的 `while False:` 不会生成 body（只保留选中的分支）。可在循环体内嵌套 `if`/`try`：`break`/`continue` 位于循环内的 `try` 体里时直接跳出/回到块头，except 子句只在该 try 体先抛异常时才走。
- **`for` 循环**：`for exprs in iter: body`（可带 `else`）在 astgen 里降级为一个显式迭代器循环：循环前先取一次迭代器（`%it = iter.__iter__()` 并 commit），循环体放在一个 `try` 里：`exprs = %it.__next__()`，commit，再执行 `body`；`except StopIteration`（`std.StopIteration`）里先跑 `else` 再 `break`。`__next__` 抛 `StopIteration` 即循环结束（`else` 只在这种情况下执行，`break` 跳过它，与 Python 一致），`continue` 开始下一轮。`__iter__`/`__next__` 由迭代器的静态类型解析（`range` 这个内建名在 astgen 里映射到 `std.range`，`StopIteration` 映射到 `std.StopIteration`）。循环变量绑在循环体的子作用域里，循环之后不可见。循环前有一条 `syntax.unroll()` 语句时，外层 `Loop` 标记为编译期循环（见下面的“编译期循环”）：迭代器和循环变量都是编译期值，于是 `try` 能靠编译期抛出的 `StopIteration` 结束循环，整个循环在编译期展开。
- **`and`/`or` 的短路**：降级成一个 `hir.Block`，操作数用 `hir.BreakIf` 短路跳出（见上文「语言与表达式」）。`if`/`while` 的条件是 `and` 链时，分支体直接放在块里（`break_if` 为假时跳出块去走 `else` 分支或跳出循环；有 `else` 的 `if` 用两层块，body 末尾无条件跳出两层以跳过 `else`），这样条件里 `:=` 解包出的 payload 指针支配 body（普通的两路 `if` 分叉不会）。
- **编译期循环**：循环前的一条 `syntax.unroll()` 语句把紧跟其后的 `while`/`for` 标记为编译期循环，`interp` 不为它发出回边，而是把循环体按条件的编译期取值**展开**成多个 body 块（条件必须是编译期值，且循环体要让编译期状态推进，否则展开次数超过 `HirRunner.max_loop_unroll`（默认 1024）时报错）。`break` 跳过全部展开的块（到循环之后），`continue` 跳到下一个 body（重新求值条件），`while`-`else` 的 `else` 在条件转假时执行一次。`syntax.unroll()` 必须紧接在它标记的循环之前，中间不能有其它语句（否则报错）。编译期 `for` 的**迭代器和循环变量都建在内联 slot 里**（`astgen._gen_for`）：编译期迭代器是一个没有运行时表示的聚合，普通表达式临时量放不下它；迭代器的 `__iter__`/`__next__` 是未装饰方法（内联），接收者是编译期值时就在编译期执行，于是 `__next__` 耗尽时抛的 `StopIteration` 是**编译期抛出**、被 `except StopIteration` 静态接住，它子句里的 `break` 结束整段展开——运行时迭代器则没有这个出口，会由展开上限报错。`std.range` 的 `__next__` 把结果值放在 `Comptime[T]` 局部量里，同理是为了让元素类型没有运行时表示（`range(3, 0, 1)` 这种未定型字面量）时也能编译期迭代。
- **异常**：`raise E(...)` 把异常值构造成一个 slot 并发出 `hir.Raise`。函数的返回约定是一个 `sval.ResultType`（正常返回值 + 按错误码排序的异常集），`make_ret_spec` 把它摊成叶子：值、错误码、payload union（结果位置因此是 `ComptimeResultPtr`：value/code/payload 三个位置）。异常不往每个 try 自己的 error space 拷贝，而是在**抛出/调用点**用一个 `switch` 直接分派到最内层能捕获它的 except 子句：每个子句的入口块惰性创建，payload 指针由 `mir.Phi` 汇合送来，`except E as e` 的 `e` 直接绑定到那个指针（`hir.ExceptBind`），不拷贝。被调函数的 error 部分由调用方决定写到哪：若没有外层 try 能捕获、且（当前函数异常集为 infer 或已含被调函数的全部异常），就**直接写进当前函数的返回 payload**（零拷贝；payload 按值返回时也一样——写之前先把目标指针重解释成“到来的那个 union”，因为 union 值之间不能转换，只能重解释存储）；否则写进临时 slot 再按需拷入函数返回位置。没有被任何子句命中的 except 是死代码，其 HIR 不编译不分析；子句按顺序匹配，裸 `except` 的 union 随分派增长、延迟定型（`mir.UnionType`：子集 union 的指针可 bitcast 到超集），目前只把 code/payload 传进去、尚不读取。
- **错误码的编码**：`ResultType.tag_bits` 取函数用到的 tag 所需的最小宽度：有正常返回值时 tag 是 `0..n`（`0` = 无错误），需要 `n.bit_length()` 位；**值部分是空类型**（`sval.EmptyType`：函数体从不交付值，因此不存在 `return` 路径）时没有“无错误”码，第 i 个异常的 tag 就是 `i`，于是单个异常时错误码是 `u0`（零大小）——调用点无从 `switch`，直接静态分派到那唯一的异常。写错误码（异常离开函数，或经 result location 抛出）依赖这个编码，而函数体跑的时候返回值类型可能还没定型，所以写入推迟成 `_PendingErrorCodeWrite`（`mir.Insertion` 占位），在 `finish_function` 里按最终的 `ResultType` 填上；`return` 路径清的 `0` 不用推迟（能 `return` 就不可能是空类型）。
- **noreturn**：值部分是空类型且**不抛异常**的函数永远回不来：它的 MIR 返回值是 `mir.NoReturn`，LLVM 定义上标 `noreturn`，调用它是 `mir.Call(..., mir.NoReturn)`——这个调用**结束所在基本块**（lower 在它后面补一条 `unreachable`），`interp` 在它之后直接 cut，所以调用点之后的代码是死代码（它也不会让调用方的返回值类型被推断成那些死代码的类型）。值部分是空类型但会抛异常的函数照常“返回”（靠错误码），只是调用点没有 `code == 0` 分支。
- **defer 块**：`with syntax.defer(): body` 把 `body` 登记到它所在区域（函数体、`if`/`while` 的分支体、`try` 的 body/子句、`hir.Block`、以及 defer 体自身），在**离开该区域**时运行。触发哪些离开由 `defer(flags)` 的位掩码决定（`flags` 是任意编译期整数表达式，如 `syntax.UNWIND | syntax.OK`；见 `syntax` 的 `OK`/`RAISE`/`UNWIND`/`ERR`/`ALL`）：`OK` 是正常离开（`return`/`break`/`continue`/自然落穿），`RAISE` 是错误离开（`raise`、被传播出去的错误），`UNWIND` 是 panic 展开；`okdefer()`/`errdefer()` 分别是 `defer(OK)`/`defer(ERR)` 的简写（`ERR = RAISE|UNWIND`），无参 `defer()` 是 `ALL`。同一区域内的多个 defer 按声明**反序**运行，内层区域先于外层。astgen 把 `with` 翻成一个 `hir.Defer`（带 `flags`）＋ body ＋匹配的 `hir.End`；`interp` 把 body 发射到一个**独立于当前块**的块树（模板），并把入口登记进当前区域的 defer 列表——body 在源位置并不执行，只有离开区域的转移指令才通过 `defer_blocks`（`mir.Jmp`/`mir.Br` 的两个边各一份/`mir.Ret`/`mir.EndDefer`）携带它（panic 的展开边另有 `mir.CallMayPanic`/`mir.Panic` 的 `unwind_defers`，见下）。因此离开区域的每条路径都会先跑相应的 defer 再继续；载入错误的转移会跨 inlined frame 收集，defer 体内声明的 defer 由该 body 的 `mir.EndDefer` 触发。这些模板是共享的，`mir.instantiate_defers`（在 `normalize` 之后）为每个使用点克隆一份并接好续接，实例化后 CFG 上不再有 `mir.EndDefer`、所有 `defer_blocks` 均为空。`body` 不得跳出 defer 块（`return`/`break`/`continue`/`raise` 越出即报错），目前也不允许声明在编译期（`syntax.unroll()`）循环里。
- **panic / `catch_unwind`**：`std.core.panic(data)` 抛出一个携带 `PanicData` 的 C++ 异常（复用 Itanium ABI 的 `__gxx_personality_v0` 人格函数与 `__cxa_throw` 等运行时；typeinfo 用 `typeid(int)`），`std.core.catch_unwind(fn)`（`fn` 必须是编译成运行时函数的**非内联闭包**）调用它并捕获 panic，把数据包成 `std.core.UnwindException` 沿普通异常路径抛出（可用 `except UnwindException as e: e.data` 读取）。一个可能 panic 的运行时调用，若它所在区域还有会触发 `UNWIND` 的 defer，就降成 `mir.CallMayPanic`：unwind 边先跑这些 defer、再进整个函数共用的 resume 块（`mir.Resume`）；`panic` 本身降成同形的 `mir.Panic`（内联的 `__cxa_allocate_exception`/`__cxa_throw`），`catch_unwind` 降成 `mir.CatchUnwind`（catch landing pad 捕获后由解释器构造 `UnwindException` 分派）。为此 `@func`/`@syntax.closure` 的 `may_panic` 默认 `True`（C 调用约定亦然）。panic 没有 `recover`，只有 `catch_unwind`；一个未被捕获的 panic 展开到 Python 边界会中止进程。

### 函数与调用

- **模块化编译**：编译一个函数时，把"它 + 它依赖的所有尚未编译的函数"放进同一个 LLVM module 一起 `define`；之后该 module 作为一个新命名的 JIT library 链接进进程内唯一的 LLJIT，并把此前链接过的所有 library 都列为自己的前置依赖（LLJIT 的链接顺序不传递，必须逐个列出），于是调用处引用的外部符号按链接名解析到对应 library；未能解析到的外部符号回退到宿主进程。每个特化的原生符号名唯一（重名的函数会被分配不同的名字）。
- **递归**：直接递归、互递归、泛型函数的多类型递归都能工作（调用进行中即可解析到正在编译的函数本身）。递归要求函数返回类型能由注解确定：具体类型，或由参数绑定出的类型参数 `T`；否则报 `requires a return type annotation`。
- 普通（未注册进任何 context 的）Python 函数在体内调用时会被**内联**；未装饰的结构体方法同理。内联函数也可以递归调用自己，但内联体的参数绑定为运行期值，递归驱动参数是运行期值时编译期不会收敛（形同编译期死循环），会在内联嵌套上限（64 层）处报错——需要真正运行期递归的函数请声明为 spy 函数。
- Python 调用侧与函数体**内部**的调用都支持位置参数、关键字参数与默认参数（`*args`/`**kwargs` 尚不支持）。
- 支持把 spy 函数定义在**工厂/闭包**里：捕获的外层变量（数值、类型值、兄弟 spy 函数等）在解析时作为编译期常量嵌入；函数在注册期间引用自己的名字也正常（见 `astgen._resolve_closure`）。
- **闭包**：spy 函数体内可以再定义 `def`/`lambda`，它们**按引用**捕获外层函数的变量（参数、局部量；内部的写回通过捕获量的字段/指针完成）。闭包只存在于编译期：只能在函数内调用，或传给未注册（内联）的 Python 函数；**不能**传给运行时 spy 函数，也不能存进运行时位置——`def` 的名字自动绑定到一个全编译期 slot。默认闭包在调用点**内联**；`@syntax.closure(inline=False)` 让它编译成运行时函数，捕获变量以隐藏的指针参数传入（编译期捕获不进 MIR，作为编译期参数内嵌），`@syntax.closure(exceptions=...)` 指明它可能抛出的异常（同 `@func`，默认为不抛异常）。无捕获且非内联的闭包可用 `syntax.as_func_ptr` 取函数指针。`lambda` 一律强制内联，可直接作为实参（如 `f(lambda e: foo(e, 1))`）。闭包内被赋值的名字是闭包局部（没有 `nonlocal`，见「尚未实现」）。
- **多返回值**：返回注解写成 `tuple[T1, T2, ...]` 时函数返回多个值（一个按值返回，其余经 result 指针交付），调用侧用 `a, b = f()` 解构（见「多返回值」）。

## 结构体

`@spy.struct()` 把带类型注解的 Python 类变成 spy 结构体：注解字段按声明顺序构成布局；类里的方法（`@spy.func()` 装饰或未装饰）成为结构体的方法。结构体既能在 spy 函数体内注解、构造和调用方法，也能从 Python 侧直接构造、读写字段、调用方法，并作为参数/返回值跨边界（见下「Python 侧使用」）。

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
- **方法**：在类里定义并用 `@spy.func()` 装饰的是编译成原生调用的 spy 方法（未写返回注解时按函数体推断，什么都不返回就是 void 方法）；未装饰的方法在调用处被**内联**。方法的 `self` 默认是一个指向自身的指针（`self: Ptr[Self]`）：HIR 直接把 `self` 绑到调用方传进来的那个指针上（`FunctionIR.arg_is_ref`），因此读 `self` 会隐含解引用得到接收者本身、`ref(self)` 是 `Ptr[Self]`，可以就地写回；`@spy.func(sfv=True)` 让 `self` 按值传递（得到一份拷贝）。结构体还可以定义**魔术方法**参与运算符重载：算术与位运算的 `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__`/`__mod__`/`__pow__`/`__or__`/`__and__`/`__xor__`/`__lshift__`/`__rshift__`（各自的反射版 `__r*__` 与原地版 `__i*__`）让 `a + b`、`a += b` 等落到方法上；比较 `__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__`（`<` 等对右操作数用相反的比较方法）；一元 `__neg__`/`__invert__`；以及作为 `if`/`not` 条件的 `__bool__`。左操作数没有对应方法时用右操作数的反射方法（交换操作数）。**通过类名调用**：`Foo.m(...)`（已特化的 `Foo[i32].m(x)` 亦然）按名字解析出方法/函数并**不隐式传 `self`**——实参按序原样传入，`self` 由调用处显式给出（`Foo.m(x, ...)`）；`@staticmethod` 声明的方法不带接收者（`Foo.m()` 与值上的 `v.m()` 都不传 `self`）。`@struct()` 可省略：普通类可用 `Cls.fn(...)` 调用其函数（把类当作命名空间）。**方法可继承**：方法取自整个 MRO（普通类/`Protocol` 基类的函数也成为方法），子类覆盖优先，例如 `std.mem.DynamicAllocator` 继承了 `Allocator` 的 `new`/`deinit`/`new_array`/`resize_array`。
- **构造**：`Point(a, b)` 是一个 result-location 构造调用——`p = Point(...)` 或 `return Point(...)` 直接向目标位置的 slot 写字段，嵌套构造（`Bar(Foo(...), ...)`）把内层结构体直接建在外层字段里，不产生拷贝。位置实参按字段声明顺序写入，关键字实参按名指定；**每个字段都要得到值**：没有默认值的字段必须写全（零大小（ZST）的字段也一样，`Blank(None)` 不能写 `Blank()`），类体里给了值的字段（`y: i32 = 7`）可以省略，省略时把自己的默认值写进去（按字段类型转换，ZST 字段则写不进任何东西——它的值就是单位值）。默认值**不参与泛型参数推断**，只由提供的字段值决定。类里**不能**定义 `__init__`（自定义构造函数尚未支持）。
- **编译期聚合体**（结构体或数组）：只有落进 `Comptime`/`Comptime[T]` 变量（`hir.InlineMode.FULL`）时才不落内存，而是被表示成一个**编译期聚合**（`interp.ComptimeAggregatePtr`：每个字段/元素各有自己的 place，聚合本身是一个指针）：读写在编译期折叠（`s.a = s.a` 是编译期赋值），字段/元素值可以作为编译期 `if` 的条件或 `syntax.unroll()` 循环的条件，嵌套聚合（结构体字段、数组元素）按各自类型递归。**`FULL` 槽里的聚合一律是编译期聚合**，不管字段装的是什么（运行时值也行）——唯一的例外是某次交付需要它自己的单个地址（走 result pointer 的调用，见 `_defer_ptr_convertion`），那时只能落内存。字段的 place 各按自己的类型决定：装编译期值的落成 box（`ComptimeBox`），装运行时值的落成它自己的运行时 place（内存，因此**可以取地址**：`ref(s.a)` 能交给原生函数写透）。因此把一个**整个运行时结构体值**赋给编译期变量（`s = Small(x, 2); c: Comptime = s`）会把值拆成“每字段一次写入”（嵌套聚合递归拆分），而不是把整个结构体物化进内存。整个聚合按 place 拷贝（`b: Comptime = a` 不共享 place）。**表达式临时量**是 `InlineMode.NON_AGGREGATE`，不内联聚合：`f(Pair(x, 3))`、`Pair(x, 3).total()` 这类字面量直接写进运行时 alloca，没有多余的拷贝（运行时值写进 `NON_AGGREGATE` 槽或编译期聚合里装运行时值的字段时**必须**落内存：运行时 `if` 的两条路径只在内存里汇合，编译期 box 只会留下解释器走 body 时最后写的那个值）。`ComptimeBox` 只装非聚合值（标量、类型……），聚合一律用 `ComptimeAggregatePtr`（零大小的聚合也是，它的值就是单位值），元组/`**kwargs` 字典分别用 `ComptimeTuplePtr`/`ComptimeDictPtr`（元组解包目标也是 `ComptimeTuplePtr`）。传给**原生**调用的按值参数或方法接收者时先物化成内存再取地址（`interp._materialize_aggregate`）。注意编译期位置只持有一个值：**编译期值**写进同一批 place 时（运行时 `if` 的两个分支各构造一次，如 `s: Comptime = Small(1, 1) if c else Small(2, 2)`）最后写入的胜出（标量的 `x: Comptime = 1 if c else 2` 也是这个行为）；如果两边的值都是运行时值，则各分支写自己的值到同一批 place，运行时选择的语义得以保留。`ref(聚合)` 得到的是编译期指针：放进 `Comptime` 变量就指向聚合本身（写透会改到聚合），放进普通变量则会物化成一份拷贝的地址（聚合没有自己的地址）。
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

### Python 侧使用

结构体值也能从 Python 侧直接使用。`Point(1, 2)` 构造一个实例：位置实参按字段声明序、关键字实参按名写入，省略的字段取类体默认值（没有默认值的字段必须给全）；泛型结构体 `Foo(...)` 的特化由字段值的类型推断，也可写成 `Foo[i32](...)`。实例的字段可读可写（`p.x`、`p.y = 5`），方法可调用（`p.m()` 隐式传 `self`；`Foo.m(p, ...)`、`Foo[i32].m(p, ...)` 是类名调用，**不**隐式传 `self`，`self` 由调用处显式给出）。实例可以作为实参传给带结构体参数的函数，函数的返回值也是实例：按值/按引用/结果指针的 ABI、字段布局（重排、ZST 字段、单字段镜像）都由边界层（`compiler/glue.py`）透明处理。

```python
@spy.struct()
class Point:
    x: spy.i32
    y: spy.i32

    @spy.func()
    def total(self) -> spy.i32:
        return self.x + self.y

@spy.func()
def make_point(x: spy.i32) -> Point:
    return Point(x, 1)

@spy.func()
def sum_point(p: Point) -> spy.i32:
    return p.total()

p = make_point(5)          # 一个实例
assert p.x == 5
p.y = 2
assert sum_point(p) == 7   # 作为实参传回去
assert Point.total(p) == 7 # 类名调用：显式传 self
```

`Option[T]` 与 tagged union 在回到 Python 时**脱壳**：`Option[T]` 是 `T` 的值或 `None`，union 是其具体 variant 的值；`None` 也可以作为 `Option[T]` 的缺席值传回。可能 raise 的 spy 函数从 Python 调用时，抛出的是一个继承 `Exception` 的异常结构体实例（`except Exception as e` 可捕获并读它的字段）。指针是**非空**的 `Ptr[T]`（`Option[Ptr[T]]` 才是可空指针）。数组、以及自定义 `__eq__` 之外的 Python 侧实例运算尚未实现。

## 指针

`syntax.Ptr[T]` 是 C 的 `T*`（const 的写法是 `syntax.ConstPtr[T]`）。`syntax.ref(a)` 取 `a` 的地址（C 的 `&a`），`p[...]` 解引用（C 的 `*p`）：

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

- **类型**：注解里的 `Ptr[T]` 被 `sval.as_value` 转成 `sval.PointerType`（元素类型 `T`、非 const）；`ConstPtr[T]` 转成 const 的指针类型。**多指针** `MultiPtr[T]`（const 的写法 `ConstMultiPtr[T]`）是同一地址，但可以像数组一样下标（见「数组」）。
- **取地址**：`ref(a)` 是一个值（指针），内容就是 `a` 的地址——`a` 可寻址（变量、形参、字段、`p[...]`）时直接就是它的地址，否则（字面量、算术结果等）先落进一个临时 slot 再取。
- **解引用**：`p[...]` 表示 `p` 指向的那个位置，和变量一样是一个**引用**（左值）：可读、可赋值（`p[...] = v`）、可 `+=`、可传给按引用传递的形参。`p.x`（自动解引用）与 `p[...].x` 取到的都是同一个字段。
- **const**：可变指针（`Ptr`/`MultiPtr`）可以隐式转成对应的 const 指针（`ConstPtr`/`ConstMultiPtr`），反过来不行。
- **`ptr_cast`**：`syntax.ptr_cast(p, T)` 把指针 `p` 强制重解释成指针类型 `T`（运行期就是一次地址重解释，不做检查）；`std.arr_slice`/`std.const_arr_slice` 用它把指向数组的指针变成数组的切片（见「数组」）。
- **泛型**：指针类型参与类型参数求解——`Ptr[T]` 求解 `T`（const 性不再是一个类型参数：它是类型本身，用 `Ptr`/`ConstPtr` 区分）。

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

- **类型**：注解里的 `Array[T, N]` 被 `sval.as_value` 转成 `sval.ArrayType`（元素类型 `T`、长度 `N`）。长度是一个**值**类型参数，所以写的时候要用 `Literal[N]`；元素类型也可以是未求解的类型参数。
- **切片**：`std.arr_slice(p)` / `std.const_arr_slice(p)` 把一个指向数组的指针 `p`（`Ptr[Array[T, N]]` / `ConstPtr[Array[T, N]]`）变成数组整体的切片 `std.SlicePtr[T]` / `std.ConstSlicePtr[T]`（用 `ptr_cast` 把指向数组的指针重解释成元素的多指针，长度取 `N`）。多指针（`MultiPtr[T]`，例如切片的 `ptr` 字段）也可以直接下标/偏移：`p[i]`、`p + n`；对多指针切片 `p[a:b]` 会构造一个切片。切片本身是 `{ptr, length}` 的结构体。
- **构造**：`array(a1, a2, ...)` 与结构体构造一样是 result-location 构造调用——元素按顺序直接写进数组的存储，嵌套构造（数组套数组、结构体里的数组字段）不产生拷贝。**长度取实参的个数**；元素类型取目标位置已声明的类型，否则取所有元素的共同类型（每个元素都得有一个能落地的类型，所以 `array(1, 2)` 这种没写类型的整数字面量会报错）。因为 Python 类型系统无法从实参推出长度，`array` 签名里带一个**只为类型检查**服务的 `length: N = 0` 关键字（不写时 pyright 认为长度是 0）；编译以实参个数为准，并且忽略这个关键字。
- **下标**：`a[i]` 是第 `i` 个元素的**位置**（左值）——可读、可赋值、可 `+=`，也可以继续取字段（`a[0].x`）或继续下标（`a[0][1]`）。下标类型暂时固定为 `u64`（将来的 `usize` 会按目标平台取具体整数类型）；编译期常量下标会检查越界，运行期下标不检查。指向数组的指针同样可以下标：`p[...][i]`。
- **ZST**：元素是 ZST、或长度为 0 的数组本身是 ZST——没有存储、没有运行时表示：每个元素都等于元素类型的单位值，`a[i]` 不产生地址（`a[i] = v` 也就什么都不写）。
- **编译期数组**：和结构体同一套规则——`Comptime` 变量里的数组是元素各有 place 的聚合，元素读写在编译期折叠；表达式临时量里的数组仍然落内存。
- **返回与传参**：和结构体同一套规则（`sval.returns_via_result_ptr`/`pass_by_ref`）——不超过 16 字节按值、更大的经 result 指针；数组的值可以整体拷贝（`b = a`）。

## Option

`syntax.Option[T]`（即 PEP 695 的 `type Option[T] = T | None`）的值要么是一个 `T`，要么是 Python 字面量 `None` 求值出的 `Null`：

```python
import spy
from spy.syntax import Option

@spy.func()
def maybe_add(x: spy.i32, y: spy.i32, c: spy.bool) -> Option[spy.i32]:
    if c:
        return x + y         # 一个 T 值
    return None              # 缺席值
```

- **类型**：注解里的 `Option[T]`（等价写法 `T | None`）被 `sval.as_value` 转成 `sval.OptionType`。`None` 不再是 void 的单位值 `Void()`，而是 `Null()`（类型 `NullType`）；函数返回注解 `-> None` 仍解释为返回 void（`VoidType`）。
- **自动转换**：`T` 与 `NullType` 都能自动转成 `Option[T]`：`coerce` 把 `T` 值标记为“有值”、把 `Null` 标记为“缺席”，`resolve_peer_type` 把二者统一成 `Option[T]`（`NullType` 与 `T` 的 peer、`Option[T1]` 与 `T2`/`Option[T2]` 的 peer 都会往下递归到 child）。因此一个 result location / slot 同时收到 `T` 和 `None` 时，其类型就是 `Option[T]`；`NullType` 也可以自动转成 `VoidType`（`Holder(None, n)` 里的 void 字段照旧可写）。
- **泛型**：`Option[T]` 里的类型参数照常求解（`Option[T]` 实参对 `Option[U]` 形参约束 `T` 对 `U`；只给一个 `T` 值也可以解出 `T`）。
- **内存表示**（`sval.OptionType.to_mir_type` + `find_first_pointer_type_pos`）：如果 `T` 里还有**未被内层 `Option` 占用**的指针（Zig 风格的非空 `Ptr[T]`），就借用它当“是否缺席”的标签，因此 `Option[Ptr[T]]` 的内存布局与 `Ptr[T]` **完全一致**，读到的是空指针即为 `Null`；`T` 是 ZST 时只用一位 `bool`；否则用一个 `(bool, T)` 的结构体（`bool` 是标签，`T` 是值）。**每个 `Option` 层占用一个指针**：`T` 有 `n` 个可用指针，`Option[T]` 就只剩 `n - 1` 个，所以 `Option[Option[T]]` 的外层用 `T` 的**第二个**指针做标签（内存布局仍是 `T` 本身），指针用完后才退回 `bool`/结构体（例：`Option[Option[Ptr[T]]]` 退回 `(bool, Ptr[T])`）。`find_first_pointer_type_pos` 同时决定了有没有可用指针与标签的位置；缺席值 `Null` 落到运行时就是相应的空指针 / `false` 标签。
- **result location**：往 `Option[T]` 的存储里交付一个 `T`（如构造体、return）时，`HirRunner._convert_result_ptr` 把指针转成此时选项 payload 的地址（并按表示设置标签），构造就地在 payload 上进行。
- **判断缺席**：`expr is None` / `expr is not None` 判断一个选项是否缺席（`expr` 必须是 `Option[T]`，否则报编译错）。`hir.IsNull` 求“是否缺席”，`is not None` 再对它取反。
- **解包（海牙语法）**：`(name := expr)` 让 `expr`（一个 `Option[T]`）落成一个可寻址的地方，`name` 绑定到**该选项的指针**上（是它的一个别名），表达式自身也是那个地址。`(name := expr) is not None` 是解包的特例，整体当作一个判断：`name` 改绑到该选项 **payload 的指针**——读 `name` 读出 payload，写 `name`（以及 `name += ...`）写的就是 payload——而表达式的值是“存在”布尔。

  ```python
  @spy.func()
  def bump(x: spy.i32, y: spy.i32, c: spy.bool) -> spy.i32:
      if (v := maybe_add(x, y, c)) is not None:
          v += 1                 # writes the payload of the option
          return v
      return 0
  ```

  海牙语法可以出现在任何表达式位置。`if`/`while` 的条件是 `and` 链时，条件里的名字在其分支体（body）里可见：链中每个操作数都求值过才会进入 body，而 body 就放在同一个 `hir.Block` 里（见「`and`/`or` 的短路」），条件里的指针因此支配它。`or` 链、一般（值）的 `and`/`or` 链、条件表达式（`a if c else b`）的两个分支各自新开一个作用域（`if`/`while` 的条件作用域也把 `else` 排除在外）：这些地方不保证求值过，名字只在其中可见。
- **编译期选项**：选项的编译期存储是 `interp.ComptimeOptionPtr`——一个 `*Option[T]`，标签 `is_null` 是一个**值**（标签没有地址）而 payload 是一个自己的 place；读出来是 `interp.ComptimeOption`（标签 + payload），它的运行时表示用 `mir.InsertValue` 从字段拼出（指针标签的缺席值用 `mir.Select` 置空指针）。`Comptime[Option[T]]` 变量、编译期聚合里的选项字段都是这种存储，海牙解包对它们同样可用。
- **尚未实现**：还没有模式匹配（`match`）；以 `T` 的某个指针当标签的表示下，`T` 自身令该指针为空值（或内层选项为缺席）时会被误读为外层缺席（与 Zig/Rust 的 niche 优化同样的局限）。

## 多返回值

函数的返回注解可以写成一个 `tuple`：`-> tuple[T1, T2, ...]` 表示函数返回**多个值**（不是返回一个元组值——spy 里没有元组值）。

```python
@spy.func()
def min_max(a: spy.i32, b: spy.i32) -> tuple[spy.i32, spy.i32]:
    if a < b:
        return a, b
    return b, a
```

- **降级后的签名**：编译器从返回值里挑**一个**按值返回——即第一个 `sval.returns_via_result_ptr` 说够小、能放进寄存器的类型（标量总是可以；零大小的结果不占位置，只交付其单位值）——其余每个值都给函数追加一个 result 指针形参（`*T`），按返回值顺序排在所有声明形参之后；一个按值结果都没有时函数返回 void。例如 `-> tuple[i32, LargeStruct, SmallStruct]` 降级成 `fn(*LargeStruct, *SmallStruct) -> i32`，`-> tuple[Foo, Bar]`（两个大结构体）降级成 `fn(*Foo, *Bar) -> void`。泛型返回值在调用解出具体类型后重新决定。
- **嵌套多返回值**：返回值里可以再嵌一个 `tuple[...]`，例如 `-> tuple[i32, tuple[i32, Large], Small]` 返回三个值，其中第二个值是它自己的两个值。嵌套的元组在解密后的签名里同样展开成叶子（`sval.RetTuple`/`sval.RetValue`），比如上面这个例子降级成 `fn(*i32, *Large, *Small) -> i32`；元组只承载结构，不占用运行时位置。调用方按同样的嵌套形状解构（`n, (m, large), small = f()`）或转发（`return f()`），而把整个调用打包进一个 `Comptime` 变量时也保留嵌套形状。
- **解构赋值**：`a, b = f()` 把每个结果写进对应目标的地址（目标可嵌套，也不必是新声明的变量）。
- **不解构**：调用结果是一个**打包**的结果（每个结果所在位置的编译期元组），它没有自己的运行时类型，只能放进编译期变量：`a: Comptime = f()`（之后还可以 `x, y = a` 再解构）。用普通变量接一个多值结果会报错，也不能把一个嵌套子组单独打包进一个普通变量（未来会加入单独声明 `Comptime` 目标的方式）。
- **打包结果的索引与长度**：打包进编译期变量的元组是一个元素 place 的树（`ComptimeTuplePtr`），可以用 `len(t)` 取元素个数（编译期整数），用 `t[i]`（编译期整数、不允许负、越界报错）取第 i 个元素的 place（可读可写，嵌套元组同样可再下标）。`len(x)` 对元组返回其长度，对结构体则调用其 `__len__` 方法（见 `hir.Len`）。
- **`return f()`**：把一个多返回值调用直接作为另一个多返回值函数的返回值，逐个写进自己的 result location。
- **Python 侧调用**：返回值打包成 Python 的 `tuple`（嵌套结果打包成嵌套 `tuple`；result 指针的存储由 Python 侧分配）；按值返回的聚合结果（小结构体）解包成实例。
- **限制**：`tuple[T, ...]`（变长）不是固定的返回值集合，在任意嵌套层级都会报错。

## 模块结构

| 文件 | 作用 |
|---|---|
| `compiler/__init__.py` | 公开接口：`func`、`struct`、`typeof`、`compile_log`、`as_`，以及各类型常量（`spy.i32` 等） |
| `compiler/dsl.py` | `func`/`struct` 装饰器、注册与全局 context（`_Context`）、各声明句柄（含 parse 与特化/编译编排） |
| `compiler/glue.py` | Python 侧边界：实参编组与结果脱壳、`_StructInstance`/`_PtrInstance`/异常实例、构造与方法调用 |
| `compiler/astgen.py` | 源码 → 无类型 HIR；把函数签名（注解/默认值/泛型参数）转成 spy 域的 `fn.Signature` |
| `compiler/hir.py` | 无类型 HIR 指令定义 |
| `compiler/interp.py` | 编译期运行 HIR → 有类型 MIR（comptime 语义所在）；结构体的字段寻址、方法分发与就地构造 |
| `compiler/mir.py` | MIR 类型、指令与基本块定义 |
| `compiler/opt.py` | MIR 清理：把单次存取的 slot 折回寄存器（按基本块支配关系判定）、删除无用 slot |
| `compiler/llvm.py` | LLVM IR 的文本构造器 |
| `compiler/lower.py` | MIR → LLVM IR → 机器码（llvmlite LLJIT） |
| `compiler/fn.py` | 函数签名（`Signature`：形参绑定、类型参数求解、返回类型推导）、函数值与编译产物、链接名表（`SymbolTable`）与函数入口 thunk |
| `compiler/sval.py` | spy 类型系统（含结构体类型）、编译期值、Python 值 → spy 域的映射（`as_value`）与类型参数约束求解（`TypeVarSolver`） |
| `compiler/syntax.py` | 函数体内使用的语法标记：指针类型 `Ptr`/`ConstPtr`/`MultiPtr`/`ConstMultiPtr`、取地址 `ref`、指针强转 `ptr_cast`、数组类型 `Array` 与构造 `array`、`Option`、编译期变量标注 `Comptime`、类型探针 `typeof` |
| `compiler/errors.py` | `SpyError`、`CompileError`、`CoerceError`、`TypeMismatchError`（同时是 `TypeError` 子类） |
| `compiler/binop.py` | 运算符的字面量类型 |
| `compiler/builtins.py` | 函数体内使用的 `spy.*` builtin |
| `compiler/util.py` | 共用工具 |
| `std/__init__.py` | 标准库对外入口：把 `std.core` 的类型与 `compiler.syntax` 的标记一并再导出 |
| `std/core.py` | 标准库核心类型：`Numeric`、`StopIteration`、`slice`、`range`、`SlicePtr`/`ConstSlicePtr`、`arr_slice`/`const_arr_slice`，以及 `gstr`/`sstr` 等内置函数 |
| `std/mem.py` | 内存相关工具：`layout_of` / `size_of` / `align_of`（布局反射）与分配器（更多尚未实现） |
| `tests/` | 集成测试（按特性拆分成多个模块） |

## 尚未实现 / 已知限制

- 赋值仅支持 `=`（含元组解包）与 `+=`（无链式赋值 `a = b = e`、其它增强赋值）。
- `*args`/`**kwargs`、仅位置/仅关键字参数、链式比较、对**单指针**的下标 `p[i]`（单指针只有解引用 `p[...]` 可用；多指针的 `p[i]` 与数组的 `a[i]` 已实现）。
- 整数 `/`、`//`、`**`（浮点的 `//`、`**` 亦然）；字节串（`bytes`）除编译期下标/切片与 `ord`/`gstr`/`sstr` 外的运算（拼接、比较、`len` ……）。
- 结构体：Python 侧实例之间的整体比较（实例本身还没有 `__eq__`）。
- 数组：运行时长度的数组、数组之间的转换（如 `i32[2]` → `i64[2]`）、以及 Python 侧实例表示（带数组参数/返回值的函数还不能从 Python 侧直接调用）。切片已由 `std.arr_slice`/`std.const_arr_slice` 提供。
- 类型标注：局部变量的标注按函数体内的表达式求值，支持能当值求出的类型（具体类型、类型参数、结构体及结构体特化）与 `Comptime` 标记；`Ptr[T]`/`ConstPtr[T]`/`MultiPtr[T]`/`ConstMultiPtr[T]`/`Array[T, N]`/`Option[T]` 这类 `syntax` 类型标记在函数体里也是可用作值的表达式（由 `hir.PointerType`/`hir.ArrayType`/`hir.OptionType` 在编译期构造），此外也能写在形参、返回值与结构体字段注解里。
- 多返回值：不能嵌套元组返回值（`-> tuple[i32, tuple[i32, i32]]`）。
- Option：还没有模式匹配（`match`，解包目前用 `is None` / 海牙语法 `(name := expr) is not None`）；以 `T` 的某个指针当标签的表示下，`T` 自身令该指针为空值（或内层选项为缺席）时会被误读为外层缺席（与 Zig/Rust 的 niche 优化同样的局限）。
- 普通 Python 函数的内联不支持运行期递归（递归驱动参数是运行期值时会在内联嵌套上限处报错，而非编译期展开）；运行期的函数值调用（把函数存进变量/字段后再调用）也尚未实现。
- 闭包：没有 `nonlocal`——闭包体内被赋值的名字是闭包局部，对外层变量的写回要通过捕获量的字段/指针；不能把闭包传给运行时函数或存进运行时位置；闭包体引用外层函数的类型参数（泛型参数）暂不支持（闭包可有自己的 `[T]`）；`lambda` 不能写注解，其参数靠实参定型。
- `defer` 块：`body` 不能跳出（`return`/`break`/`continue`/`raise` 越出 defer 块即报错），也不能声明在编译期（`syntax.unroll()`）循环里（展开循环的相邻迭代间没有转移指令可挂）。panic 已实现（`std.core.panic` / `std.core.catch_unwind`），但没有 `recover`，且 `UNWIND` 的 defer 在 defer 体内部抛 panic 时不再向外层收集（同 Rust 的 drop 内 panic）。

## 运行测试

```sh
python3 -m unittest spy.tests        # 只跑 spy 的测试
python3 run_tests.py                  # 仓库根目录：跑全部测试
```

## 一些小TODO
- [ ] sval中的内存估计改成用mir类型来计算
- [ ] 统一 ZST 字段的对齐：`std.mem.layout_of` 对 ZST 结构体按**所有**成员取最大对齐（`sval.alignment_of`），但有存储的结构体在 MIR 里丢掉了 ZST 字段，于是两者对「ZST 字段算不算对齐」不一致（详见「单位类型（ZST）」一节）。

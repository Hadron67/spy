# 编译期（comptime）重构：新行为清单

本文档描述把 spy 的 comptime 语义改成 **Zig 式「完全编译时已知」** 之后，
语言在编译期相关方面的**完整行为清单**。它是「目标行为规格」，即重构完成后
语言应当表现出的行为；每条尽可能注明它与**当前**行为的差异。

状态：已实现（ba7a1502f1eb4b618920287e94c610292c4475cd）

> 说明：本文档**不**描述实现机制，只描述可观察行为。

---

## 0. 总体模型

### 0.1 从「两层」到「一分」

- **现状**：编译期分**浅**（代码里的 inline）与**深**（comptime）两层。浅层允许一个
  `Comptime` 变量持有「不引用寄存器的运行时值」（全局常量、字符串、函数），
  也允许一个编译期聚合里混有运行时值的字段（`ComptimeBox` /
  `ComptimeAggregatePtr` 里各字段 place 可能是内存）。深层才要求「整个值在
  Python 里能直接算出来」。
- **新模型**：**只有一层**——一个值要么**运行时值**，要么**编译期值（完全编译时已知）**。
  不再有「混合/inline 存储」。判定见 §1。
- **删除的类型**：`ComptimeTuple` / `ComptimeDict` / `ComptimeAggregate` /
  `ComptimeOption` / `ComptimeTaggedUnionValue` 及其所有 `Comptime*Ptr` 形式
  （`ComptimeBox` / `ComptimeTuplePtr` / `ComptimeDictPtr` / `ComptimeAggregatePtr` /
  `ComptimeOptionPtr` / `ComptimeTaggedUnionPtr`）。`ComptimeVal`（对普通编译期
  `sval.Value` 的包装）保留。
- **删除的闭包机制**：闭包不再是一个独立的值种类——**嵌套 `def`/`lambda` 变成普通
  struct 值**。删除 `fn.ClosureValue`、`hir.Closure`、`hir.RebuildCapture`、
  `hir.CallInplace.captures`、`fn.CallSignature.captures` 与
  `std.core.as_runtime_closure`（见 §9）。`fn.ClosureFunction`（解析出的嵌套函数体）
  保留为 `hir.MakeClosure` 的输入。
- **新增 / 改造**：
  - `sval.Box`：comptime 变量（comptime alloca）的存储单元，**按对象标识相等**；把聚合体**铺平**保存为标量序列（见 §0.2）。
  - `sval.ComptimePtr`（一个 `sval.Value`）：`{box: Box | bytes, offset: int, type: PointerType}`，
    表示指向 comptime 存储的指针。`offset` 是所指对象**第一个标量**在 box 铺平序列里的
    **起始下标**；alloca 刚返回时 `offset == 0`，代表对象本身。`type` 是「视图指针类型」
    （其 `is_const` 参与相等）。`box` 也可以**直接是一个 `bytes`**（字节串存储：无 cell、只读，
    按内容参与相等，见 §4、§10）。
  - `syntax.ConstComptime` / `syntax.const_comptime()`：声明一个 **commit 后成为
    只读**的 comptime alloca（编译期常量，见 §3.4）。
  - `sval.CastedPtr`：编译期指针被 `ptr_cast` 重解释后的形式（取代
    `interp.ComptimeCastedPtr`，见 §4）。
  - **struct 字段新增标记 `is_comptime` / `is_ref`**（见 §2.4）。
  - **`fn.FunctionValue` 变为可 lower**：`lower` 直接把它 lower 成**函数符号本身**
    （同模块 `mir.Function` / 早先模块 `ExternAnonSymbol` / extern `ExternSymbol`），且
    `FunctionValue.get_type` 改为返回**常函数指针**类型（见 §4.3、§10）。
  - **`hir.MakeClosure` 改为产出「带 `__call__` 的匿名 struct 类型」**（见 §9）。
  - **HIR 的 result location 变为树状 `ResultLocTree`**：解构赋值仍走 RLS，`result_loc`
    是一个地址树；**不引入 `hir.DestructureStore`**、并删除 `hir.TuplePtr`（见 §7.2）。

### 0.2 编译期存储

- 编译期存储由 **Box** 承担：`x: Comptime = e` / `x: Comptime[T] = e` /
  `syntax.comptime(); x = e` 都会产生一个 comptime alloca，其返回值指向一个新
  `Box`（`ComptimePtr{box, offset=0}`）。
- `Box.name_base` = 当前函数的 `name_base` 加上一个 box 计数器；**它只是 MIR 里的命名提示**，
  不是特化键、不要求唯一/稳定、无链接影响（特化键是 `Box` 自身的身份，见 §4.2）。
- Box 把聚合体**铺平**保存为一串标量（`list[ScalarValue]`），`ComptimePtr` 用**起始下标
  `offset`** 寻址。宽度规则：**标量类型（含 ZST 标量）恒占 1**；**聚合体由成员定**
  （空 struct 0 cell、union `1+max` …）；未设置的 cell 读作 `UndefinedScalar`
  （完整规则见 task 文档 §0.2）。
- **`sval.ScalarValue`**：**标量类型（`ScalarType`）的值** union（`Int`/`Float`/`Bool`/`Bytes`/
  类型值/函数值/`Null`/`Void`/`UndefinedScalar`/`ComptimePtr`/`CastedPtr` 等，加普通 Python
  标量/`bytes`）；struct/array/tuple/strdict/complex/option/tagged union/union 是**聚合体**，
  不在其中；`TypedNull` 不算标量（它是 `Option[T]` 的值）。
- `Box.is_const` 的生命周期：
  - alloca 时 `is_const = False`（可写，用于初始化）。
  - `x: Comptime = e` / `syntax.comptime()` 的 box 在 commit 后**仍是
    `is_const = False`**（可变编译期变量）。
  - `x: ConstComptime = e` / `syntax.const_comptime()` 的 box 在 **commit 后变为
    `is_const = True`**（只读编译期常量；再写是编译错误，见 §3.4）。
- **能否进入运行时表示**：`is_const = True` 的 box（编译期常量）由 `lower` **自动 lower 成
  `mir.GlobalConstant`**；**函数值**（`fn.FunctionValue`）则直接 lower 成**函数符号本身**
  （同模块 `mir.Function` / 早先模块 `ExternAnonSymbol` / extern `ExternSymbol`，`get_type` 为常函数指针，
  不经 `GlobalConstant`）；`is_const = False` 的 box（可变编译期变量）**暂不支持 lower**（见 §4.3）。
- Box 的身份是**全局的、非局部的**：一个编译期指针不再随栈帧生灭，因此可以
  逃逸出创建它的位置（见 §4）。

---

## 1. 值的两分法（什么是「编译期值」）

一个解释器值是**编译期值**当且仅当它**完全在编译期已知**，即它是编译器在编译期
能直接持有的对象：

- 一个 `sval.AnyValue`（即 `sval.Value`——标量常量、类型值、编译期聚合值、
  `sval.OptionValue`/`sval.TaggedUnionValue`、`Null`/`Void`/`UndefinedScalar`/`UndefinedAggregate`、
  `sval.ComptimePtr`、`sval.CastedPtr` 等——再加上普通
  Python 标量/`bytes`），或
- 编译器的一个编译期对象，如 `fn.FunctionValue`（其 `get_type` 报出函数指针类型，
  并由 `lower` 自动 lower）。

否则它是**运行时值**（有 MIR 表示、引用寄存器的 load / 计算 / 参数、运行期循环
里的量……）。

**关键变化**：编译期值**严格**是 `sval` 域的编译期值，**不再包含「不引用寄存器的
运行时值」**。旧定义之所以放宽（把全局常量、字符串、函数指针、空指针也算编译期
值），是因为 `as_static_ptr` / `as_func_ptr` 返回的是运行时值（`mir.GlobalConstant`
形式的全局常量）。在新语义下这些都用编译期值表示——`sval.Box`（编译期指针）与
`fn.FunctionValue`——并由 `lower` 自动 lower，因此不再需要那条放宽。

因此**不存在「以运行时值（`mir.GlobalConstant` 等）形式存在的编译期值」**：任何
编译期已知的值，要么是 `sval` 的编译期值，要么是 `fn.FunctionValue` 这类编译期
对象。

**关键变化**：在「完全编译时已知」下，**任何一个组成部分是运行时值的聚合都不是
编译期值**。因此旧语义里「一个 `Comptime` 变量持有一个字段是运行时的结构体」
（`s: Comptime = Small(x, 2)`）不再成立——那样的值整体是运行时值（见 §5 的
行为变化）。

判定在**判定位置**逐层进行（一个容器只有在每个元素都是编译期值时才整体是编译期值）。
这一点与旧 `interp._is_comptime_val` 的「深层」判定一致；区别在于新语义下
判定结果直接决定这个值能不能进入 comptime 变量/参数的存储，而不再有「浅层兜底」。

---

## 2. 类型的运行时表示

### 2.1 新获得运行时表示的类型

- **`tuple[T1, T2, ...]`（`sval.TupleType`）**：**当所有元素都有运行时表示时**，
  它现在**相当于一个 struct**——字段是位置字段，按元素顺序布局，遵循与结构体相同的
  布局规则（对齐重排、只含一个非 ZST 元素时镜像为该元素、全 ZST 时为 ZST、
  可拷贝性 = 所有元素可拷贝）。它不再「没有自己的运行时表示」。若某元素没有运行时
  表示（`type`、字节串、DST 函数类型等，见 §2.2），该元组仍只能是编译期值。
- **`StrDictType`（`dict[str, T]` / `**kwargs` 的形状）**：**当所有条目都有运行时
  表示时**，同样相当于一个 struct——每个具名字段一个条目，顺序为条目声明顺序，
  布局规则同结构体。

### 2.2 仍无运行时表示的类型（纯编译期）

以下类型**依然**没有运行时表示，因此其值只能是编译期值：

- **类型值**（`TypeType`，`spy.typeof` 的结果、`spy.i32` 这类类型常量）。
- **字节串**（`sval.BytesType`）。可下标/切片/`ord`/`len`。字节串**直接 lower 成
  运行时全局常量**（`lower` 在末尾补一个 NUL，见 §10），因此也可被 `ptr_cast`
  成指针；`std.core.gstr` 不再保留，`std.core.sstr` 保留为库函数（见 §10）。
- **DST 函数类型**（`FunctionType` / `AnyFunction`）：这类**类型**没有运行时表示，其值只能
  是编译期值。（`fn.FunctionValue` 的类型是**具体函数指针**——指针类型**有**运行时表示，
  见 §1。）无运行时捕获的闭包（= ZST struct）可经 `as_func_ptr` 变为 `fn.FunctionValue`
  （`get_type` 返回常函数指针），并由 `lower` 自动 lower（见 §4.3、§9）。
- **`EmptyType`**（`Never`）：无值。

（以上任何类型作**形参类型**时按 §3.2 自动成为**编译期形参**。）

### 2.3 位置的类型解释（保留现状）

- **返回注解里的 `tuple[...]`** 仍解释为**多返回值**，不是「返回一个元组值」。
- **函数形参注解里的 `tuple[...]`** 仍是**铺平传参**。
- 这两点与现状一致（见 §7、§8）。
- 因此 `tuple[...]` 的「运行时表示」用于**变量 / 字段 / 表达式值**等其它位置；
  在返回/形参位置它被特殊解释。元组/字典类型字段（`Foo.t: tuple[i32, i32]`）与
  结构体字段**完全同规则**（布局、可拷贝性、传参）。
- **元组/字典形参的「逻辑类型」（Q1）**：callee 侧实际绑定的是**合成的 `StructType`**
  （`$args`/`$kwargs`，字段名分别为 `_0,_1,…` / 条目名，见 §7.4/§8）；因此对这样的形参做
  `std.reflect.type_info` 得到的是 **`StructType`**（**不加特判**），tuple 与 dict 靠**字段名**区分。

### 2.4 struct 字段的两个新标记

- **`is_comptime: bool = False`**：该字段是「编译时已知」的（Zig 语义）。
  - **访问运行时 struct 的这类字段，直接返回该字段的默认值**（不读运行时存储）。
  - 这类字段**不进 MIR**：不占（MIR）布局、不参与拷贝、构造时有默认值即可省略。
    **注意**：这只针对**运行时表示**；在编译期的 **Box 铺平**里它**照常占位**（cell 数 = 其类型 width，
    存构造时的覆盖值）。
  - 若 struct 含**无默认值**的 `is_comptime` 字段，则该 struct 的 `classify()` 返回
    `COMPTIME`（它是纯编译期类型，不能有运行时值）。
- **`is_ref: bool = False`**：该字段逻辑上是一个引用，保存一个**常指针 `*const T`**，
  **只可读、不可写透**（写透是编译错误，见 §13）。
  - 对它做 `field_index_addr` 时会**再额外 load 一次**（得到 pointee 的地址），
    后续**读**作用于 pointee。在编译期的 **Box 铺平**里，该字段的 cell 存一个**编译时指针**
    （`ComptimePtr`），`field_index_addr` 同样额外 load。
  - **用途：`*args` / `**kwargs` 传参**。由于 `ComptimeTuplePtr` / `ComptimeDictPtr`
    已删除，varargs/kwargs 在函数体内只能用 **struct** 表示；其中**传引用**的元素令
    `is_ref = True`（从而额外 load 一次得到实参地址，读作用于实参）。
  - 它是**通用字段标记，与闭包捕获无关**（闭包捕获见 §9）。
- **构造时覆盖 `is_comptime` 字段**：只有**编译时 struct 变量**可以覆盖其默认值
  （覆盖值保存在该编译时聚合值里）；**运行时 struct 变量不能覆盖**——为运行时
  struct 的 `is_comptime` 字段提供实参是编译错误（见 §13）。
- **ZST 判定**：
  - `is_comptime` 字段**不参与**——一个只含 `is_comptime` 字段且都有默认值的 struct
    仍是 ZST。
  - `is_ref` 的判定是**显式例外**：一律**按字段类型**算，**不看**它实际存的指针。
- **`is_comptime` 字段的类型可以没有运行时表示**：因为它不进 MIR，这类字段不影响该 struct 是否
  有运行时表示（一个含「有默认值、类型无运行时表示」的 `is_comptime` 字段的 struct 仍可为
  `ZST`/`NONE`）。因此「类型有运行时表示 ⟺ 所有组成部分都有」**不成立**，`is_comptime` 字段
  是显式例外。
- **`is_ref` 字段的 `T` 为 ZST 时**：该字段在 MIR 里不占存储；对它的 `field_index_addr` 返回
  `undefined(*const T)`，再 load 该「指针」直接得到 `T` 的（单位/undefined）值（`store` 对 ZST 特判），
  故「额外 load 一次」在 ZST 下同样有定义。

---

## 3. 编译期变量与参数

### 3.1 `Comptime` 变量（comptime alloca）

- 声明形式（均保留）：`x: Comptime = e`、`x: Comptime[T] = e`、以及语句标记
  `syntax.comptime()` + 紧随其后的一条声明（`x = e` 或 `x: T = e`）。
  `syntax.comptime()` 也必须紧接在被标记的声明之前，且只能标记**新声明**
  （不能标记对已绑定名字的赋值）——这些错误条件与现状一致。
- 存储：一个 comptime alloca → 一个 **Box**。变量在函数体内的读、写、取地址
  都作用于这个 box（编译期）。
- **写入的值必须完全编译时已知**：把运行时值写进 `Comptime` 变量是编译错误。
  - **变化**：旧语义下 `c: Comptime = x`（`x` 运行时）会落内存、并可在运行期
    循环里写；新语义下这是错误。要一个可写的运行时标量，请用**普通变量**
    （`c = x`，见 §3.3）。
  - 错误信息（建议）：`cannot store a runtime value into a compile-time variable`。
- 声明式 `x: Comptime[T]` 也要求所写值能转换成完全编译期的 `T`。
- **逐字段构造**：声明式的 `x: Comptime[T]`（无初值）可以逐字段写（未设置的 cell 读作
  `UndefinedScalar`），因为 box 的铺平标量一开始就存在。
- 一个 `Comptime` 变量可以持有：类型值、编译期标量、编译期聚合、编译期元组/
  字典、编译期 Option/union、编译期指针（`ComptimePtr`）、函数值等。

### 3.2 编译期参数

- 形参注解 `x: Comptime` / `x: Comptime[T]` 保留：该参数**不占 MIR 实参**，只在
  编译期传递。
- 传参时实参必须完全编译时已知；传运行时值给 `Comptime` 形参是编译错误
  （与现状一致，不静默丢弃）。
- **形参类型无运行时表示 ⇒ 隐式编译期形参**：若形参解析出的类型 `classify() == COMPTIME`
  （类型值、字节串、DST 函数类型、`EmptyType`，以及任何分类为 `COMPTIME` 的类型），该形参**自动**
  成为编译期形参（实现上 `SignatureFormalArg.is_comptime = True`，等价于写成 `x: Comptime[T]`）：
  不占 MIR 实参、只在编译期传递，实参必须完全编译时已知。因此 `def sstr(s: bytes)` 这类库函数
  无需显式 `Comptime`。
- **新能力（仅 const）**：**const** `ComptimePtr`（指向 `is_const = True` 的 box）
  可以作为 `Comptime` 实参传给**非内联**函数。作为编译期实参它**按身份**传递（进特化键，callee 得到的
  comptime place **别名同一 box 同一起始下标 `offset`**，可别名，见 §4）；只有当它必须变成**运行时值/地址**
  时才由**自动 lower** 生成全局常量指针（见 §4.3）。**非 const** 的编译期指针暂**不允许**跨非内联边界
  （见 §4.2）。

### 3.3 普通变量、`is_comptime` 的推断

- 普通变量的存储：
  - **带类型注解（非 `Comptime`）** 的普通声明 → **运行时存储**（内存）——之后可以写运行时值。
  - **无注解**的普通声明 / 解构目标 → **运行时存储**（内存）（`is_comptime = False`）——同样可以写
    运行时值。因此一个无注解变量不能直接持有编译期专属值（如未定型整数）：要持有就写成
    `x: Comptime = e`（见 §3.1）、把类型写出来（`x: i32 = 1`），或写成 `InferComptime`（见下）。
- **`InferComptime`（推断存储）**：声明形式 `x: InferComptime = e`、`x: InferComptime[T] = e`，
  以及语句标记 `syntax.infer_comptime()` + 紧随其后的一条声明（错误条件与 `syntax.comptime()` 相同：
  只能标记**新声明**、必须紧跟其前）。
  - 语义：该 slot 的存储**留给写进去的值**决定（`is_comptime = None`，推断规则见下）。写入的值
    **全是编译期值**时它是编译期存储（Box），否则是运行时存储（内存）。
  - 用途：一条声明既能持有编译期专属值（未定型整数、类型值等），又能落在内存里接受运行时值，
    而不必在源码里写死是哪种。
  - **仅限局部变量声明**：形参（含 `*args`/`**kwargs`、闭包形参）写 `InferComptime` 是编译错误——
    形参总是按签名给出的存储（要编译期形参请写 `Comptime`）。
- 新字段 `PendingSlot.is_comptime: bool | None` 取代旧 `inline_mode`
  （`NONE`/`NON_AGGREGATE`/`FULL`）：
  - `True`：该 slot 一定是编译期存储（comptime alloca / Box）——`Comptime`/`ConstComptime` 变量，
    `unroll` 的编译期 `for` 的迭代器/循环变量槽。
  - `False`：该 slot 一定是运行时存储（内存）——**普通变量声明（含无注解的声明与解构目标）**、
    参数（其存储是该参数的地址）、**普通（非 `unroll`）`for` 的迭代器/循环变量槽**。
  - `None`：**推断**——由写进它的值决定该 slot 是编译期还是运行时。写在源码里就是上述
    `InferComptime`/`syntax.infer_comptime()`；编译器内部的槽（表达式临时量、`*args`/`**kwargs`
    合成 struct 的字段槽、内部的 `def` 名槽、非编译期 `for` 的槽等）也走 `None`。
- **推断规则**（仅 `is_comptime = None` 的内部槽）：在 `CommitSlot` 提交时，当且仅当该 slot 上
  **已有的写入**（store，以及构造式写入）**全部是编译期值**时，它是编译期存储；否则运行时。
  （`Comptime`/`ConstComptime` 声明走 `True`；普通声明走 `False`；表达式临时量等内部槽走 `None` 推断。）
- **`inline_mode → is_comptime` 不是机械替换**：`NONE` / `NON_AGGREGATE` / `FULL`
  各有不同语义，其每个使用点都要按新模型**逐个重新判断**，不能只做名字映射
  （见 §11）。
- **`x = 1`（未注解的整数字面量初始化）**：报错——无注解的声明是**运行时存储**，而字面量没有
  可落地的**类型**（未定型整数），因此由 `_no_runtime_type`/未定型整数路径给出。需类型标注
  （`x: i32 = 1`）或写成 `Comptime` 变量（`x: Comptime = 1`）。
- 已定成运行时存储的 slot 之后写编译期值照常可写；「一次写入决定了存储」的规则
  在 `is_comptime` 为 `True/False` 时不再有歧义，`None` 时按上面的推断。反之，已定成**编译期**
  存储的 slot 之后写**运行时值**是编译错误（同 §3.1）。

### 3.4 `ConstComptime`：编译期常量

- 声明形式：`x: ConstComptime = e`、`x: ConstComptime[T] = e`（与 `Comptime` 平行），
  以及语句标记 `syntax.const_comptime()`（与 `syntax.comptime()` 平行，必须紧接一条
  **新声明**）。
- **必须带初值**：`ConstComptime` 声明必须写成带初值的形式（`x: ConstComptime[T] =
  e`），不能"先声明后赋值"。
- 存储与提交：alloca 时 `is_const = False`（可写，用于初始化），**commit 之后
  `is_const = True`**（只读）。此后写它（`x = ...` / `x.f = ...`）是编译错误。
- 它是「编译期全局常量」的来源：`is_const = True` 的 box 由 `lower` 自动 lower 成
  `mir.GlobalConstant`（见 §4.3）。普通 `Comptime` 变量的 box commit 后仍是可变的，
  因此**不会**被 lower。
- 写入的值必须完全编译时已知（同 §3.1 的要求）。

---

## 4. 编译期指针（`ComptimePtr`）与 `ref`

### 4.1 取址与偏移

- `ref(a)`（`syntax.ref`）保留。`a` 是一个**编译期 place**（`Comptime` 变量、其字段/
  元素、编译期聚合的字段/元素、编译期 Option/union 的 payload）时，结果是
  `ComptimePtr{box, offset}`：
  - 对变量本身：`offset == 0`（即 alloca 的返回值，代表对象本身）。
  - 对第 `i` 个字段/元素：`offset == parent.offset + 该字段/元素在铺平序列里的偏移`
    （铺平顺序见 task §0.2；option/union 的 payload 从 `parent.offset + 1` 起）。
- `ref(a)` 的 `a` 是**运行时 place**（普通变量、参数、运行时字段、`p[...]`）时，
  结果是**运行时指针** `RuntimeVal`（地址），与现状一致。
- `p[...]`（解引用）保留：
  - 对 `ComptimePtr`：按 `offset` 读 `box` 的铺平标量（左值，可读；**box 与指针视图**
    都可写时才可写、可 `+=`；否则不可写，见 §4.2）。
  - 对运行时指针：普通 load/store。
- 字段/元素寻址链在编译期直接**加偏移**（不物化中间值）。
- **`sval.CastedPtr`（编译期指针经 `ptr_cast` 的结果）不能读/写透**：它只改变指针被
  看待的类型；要操作其指向的值，必须先 `ptr_cast` 回（能转换的）原类型再读写。
- **字节串作为源**：字节串编译期值（`str` 已在 astgen 编码为 `bytes`）经 `ptr_cast` 得到的是
  一个 **`box` 为该 `bytes` 值的 `ComptimePtr`**（不是 `CastedPtr`），它由 lower 生成字符串
  全局常量（末尾补 NUL，见 §4.3、§10）。

### 4.2 全局性与别名（核心变化）

- 编译期指针**不再随栈帧复制**。`ComptimePtr` 当且仅当**同一 box 同一 `offset` 同一视图
  类型**时相等（`sval.Box` 是**身份对象**，`ComptimePtr` 为 `@dataclass(frozen=True)`，字段
  `{box, offset, 视图类型}`，其结构化 eq/hash 自动给出；`box` 为 `bytes` 时按**内容**相等）。
- 全局性带来可别名，但**跨非内联边界暂只允许 const 指针**：
  - **const** `ComptimePtr`（指向 `is_const = True` 的 box）**可以**跨非内联边界：
    作为非内联函数的 `Comptime` 实参、被**编译闭包**捕获、作非内联函数返回值。
    作为**编译期实参/返回值/捕获**时它**按身份传递**（同一 box + 同一 `offset` + 同一视图类型进入
    特化键），callee/闭包得到的 comptime place **别名同一 box**、不复制；**只有当它必须变成运行时
    值/地址**时（存进运行时 storage、传给原生函数等）才由**自动 lower** 生成全局常量指针（见 §4.3）。
    可**别名**（读透可见；const 本身不可写）。
    （**变化**：旧语义每个调用前复制 pointee，callee 改的是自己的拷贝、写不回去。）
  - **非 const** `ComptimePtr`（`is_const = False` 的 box，即可变编译期变量/聚合的
    地址）**不报错，而是物化一份拷贝**：当它必须变成运行时值/地址时（作非内联函数的实参、
    作非内联函数返回值、**存进运行时 storage** 等），实现会**新分配一块内存并把 box 的内容
    复制进去**，返回该内存的地址。callee 拿到的是**拷贝**，写不回原 box（效果上等同于旧语义的
    「复制 pointee」，但旧语义不再需要）。**注意**：这只关乎「把指针本身物化成运行时值」；
    闭包捕获是把指针**烘进 struct 类型**（不是物化），不受此影响（见 §9.3）。
  - 内联函数 / 内联闭包内部：编译期指针照常按身份别名、不复制。
  - **编译期返回值经 `RetSpec` 的 `RetComptime` 叶子交付**：完全编译时已知的返回值（含 const
    `ComptimePtr`）不写运行时结果位置；**inline** 调用写进调用点的 `ResultLocTree` 叶子（一个**编译期
    place**），**非 inline** 调用在编译 callee 时得到、调用点写进对应 place（见 §7.3）。
- **特化键**：一个 `ComptimePtr` 实参由「同一 box（身份）同一 `offset` 同一视图类型」决定，
  而不是像旧的 `ComptimePtrArg` 由「内容」决定。两个指向不同 box 的指针是不同的实参；同一
  box 同一偏移同一视图是同一个实参。
- **const**：指针可带**独立的只读视图**（`PointerType.is_const`），与 box 的 `is_const`
  分开。**写透要求「指针视图」与「box」都可写**；任一为只读就报错。可变 `Ptr` 可隐式
  转 const（得到只读视图），反之不行。
- **跨 context**：设计上保证一个 context 的编译期值**不会泄漏**到另一个 context（`Box` /
  `ComptimePtr` / `fn.FunctionValue` 都是本 context 的身份对象）；跨 context 传递的只有可重新
  lower 的编译期值。

### 4.3 自动 lower（替代 `as_static_ptr`）

- **删除 `std.core.as_static_ptr`（及其内建）**：不再需要手动把编译期值转成运行时
  值。
- **自动 lower**：`is_const = True` 的 box（编译期常量）由**编译期常量 lowering（interp 的
  `_const_to_mir` 家族）**自动 lower 成 `mir.GlobalConstant`：先把 box **`load` 成一个完全编译时
  已知的 `sval.Value`**（聚合快照；`is_comptime` 字段已按 box 铺平读出、`is_ref` 字段是其中一个
  `ComptimePtr`），再把**该值**交给 `_const_to_mir` lower——布局按目标的 MIR 表示，故 `is_comptime`
  字段自然被跳过、`is_ref` 字段按指针（指向被指向对象的独立全局）写出；「box 铺平布局 ≠ MIR 布局」
  的差异由「值 → MIR」这一步消化。无需任何显式转换。**函数值**（`fn.FunctionValue`）则**直接 lower 成函数符号本身**
  ——同模块 `mir.Function` / 早先模块 `ExternAnonSymbol` / `@decl_func` 的 `ExternSymbol`
  （`get_type()` 为常函数指针），**不经 `GlobalConstant`**；未被调用的非 generic 函数可经
  `std.core.compile(fn)` 显式触发编译。
  - 迁移示例：
    ```python
    # 旧
    ptr = as_static_ptr(value)
    # 新（value 完全编译时已知）
    const_value: ConstComptime = value
    ptr = ref(const_value)
    ```
    `ref(const_value)` 在编译期语境下是 `ComptimePtr`，在需要运行时指针时由 lower
    自动处理（指向该全局常量）。
- **可变编译期变量（`is_const = False` 的 box）的物化 = 一份拷贝**：需要运行时表示时
  新分配一块内存并把 box 的内容复制进去，返回该内存的地址（见 §4.2）；**不**生成指向该 box 的全局，
  **也不报错**。因此普通 `Comptime` 变量经 `ref` 进入运行时后，callee 写的是拷贝、写不回原变量
  （要共享存储请用 `ConstComptime` + `ref`，它 lower 成全局常量）。
- lower 的表示遵循旧 `as_static_ptr` 的约束与表示：
  - 值必须完全编译时已知（不存在「引用寄存器」的问题——见 §1）。
  - 指针字段**不**被跟随：它指向的内容各自成为独立全局，指针指过去。
  - 已有的全局引用（另一个静态常量、函数指针 / `FunctionValue`）可直接作为常量
    成分。
  - 布局遵循目标的 MIR 表示：小变体的 tagged union 常量按 clang 的
    `{<variant>, [pad]}` 方式布局；ZST 变体不占 payload（`static.py` 用例验证的
    表示）。
- **字节串**：字节串编译期值由其 `ComptimePtr`（`box` 为该 `bytes` 值）自动 lower 成全局
  常量（末尾补 NUL，见 §10）；`gstr` 不再存在。
- `std/core.py` 里依赖 `as_static_ptr` 的地方（如 `std/mem.py` 的
  `DynamicAllocator.create` 的 vtable）按上式迁移（把值放进 `ConstComptime` 再
  `ref`）。

---

## 5. 编译期聚合（struct / array / tuple / option / union / complex）

### 5.1 编译期聚合的存储与读写

- 一个完全编译时已知的聚合在 `Box` 里**铺平**为标量序列（见 §0.2），字段/元素读写全部
  在编译期折叠（`load` 出的整值快照是 `sval.AggregateValue`）。
- 字段读 `s.a`：若是常量则折叠为编译期值；可作为 `if` 条件或 `syntax.unroll()`
  循环条件的编译期判定。
- 字段写 `s.a = ...`：写进 box 的聚合值（编译期赋值）。
- 嵌套聚合（结构体字段、数组/元组元素、Option payload）按各自类型递归，偏移加深。
- 整体拷贝按值：`b: Comptime = a` 不共享（各自新 box）。
- **`is_comptime` 字段**（§2.4）：**运行时 struct** 读它直接给出该字段的**默认值**（不读存储，
  因为该字段根本不进 MIR）；**编译期 struct**（Box 铺平）读它给出 **box cell 里的当前值**
  ——构造时的覆盖值，或未覆盖时的默认值。`is_ref` 字段的 `field_index_addr` 会额外 load
  一次（作用于 pointee）。

### 5.2 与运行时聚合的关系（重要变化）

- 若聚合的**任一**部分是运行时值，则**整个聚合是运行时值**，不能放进 `Comptime`
  变量/参数。要持有它，用**普通变量**（内存）。
  - **变化**：旧语义 `s: Comptime = Small(x, 2)`（`x` 运行时）能编译，且其
    `a` 字段是运行时 place、`b` 字段是编译期 place（混合）；新语义里这报错，
    应写 `s = Small(x, 2)`（普通变量，运行时结构体）。
  - 因此「把整个运行时结构体值赋给编译期变量、拆成每字段一次写入」这一现象
    **消失**。
- 把字面量（编译期）聚合直接传给原生函数的按值参数/方法接收者，仍按需要落到
  内存再取地址（保留现状）。
- 零大小的聚合：其值就是单位值；没有存储（与现状一致）。ZST 字段不占布局。
- 「某次交付需要它的单个地址」的完全编译期聚合：**作为 result location 传给按值返回的
  调用**这一子情形随 `ResultLocTree` / RLS 重构不再适用（N/A）；但 **auto-lower**
  （`ConstComptime` + `ref`，得到运行时常量指针）与**原生函数按引用传参**仍需要该地址
  （见 §4.3），这部分**保留现状**（物化 / 落全局常量）。

### 5.3 Option / tagged union

- **option / tagged union 按聚合体处理**：整体完全已知时铺平为 `[tag, payload...]`
  （tag 是第 0 个标量，payload 从 `offset + 1` 起），读写、解包
  `(name := expr) is not None` 均在编译期。
- **编译期值形态**：present option 是新增的 `sval.OptionValue`，absent 仍是
  `sval.Null`/`sval.TypedNull`；tagged union 统一用 `sval.TaggedUnionValue`
  （删除 `interp.ComptimeOption`/`ComptimeTaggedUnionValue`）。
- 若 tag 只有运行时才知道（例如从一个运行时 Option 读出），则**整个 Option/union
  是运行时值**，走普通运行时表示（`OptionType.to_mir_type` / union 存储 + 指针
  重解释），不再是特殊的 Comptime 形式。
  - **变化**：旧 `ComptimeOptionPtr`/`ComptimeTaggedUnionPtr`「tag 是值、payload 是
    place、tag 可能运行时」的混合形态被取消；tag 运行时的 Option/union 一律是
    普通运行时值。

---

## 6. 编译期控制流

- **编译期 `if`**：条件必须是**完全编译时已知**（编译期值）。为真/假时只保留选中
  分支，未选分支不编译、不分析（其中的 `compile_error` 不触发）。
  - 变化：条件必须是 §1 定义的编译期值（`sval` 编译期值，或 `fn.FunctionValue`
    等编译期对象）；含运行时值的聚合字段不再是编译期值，不能作为条件。
- **编译期循环** `syntax.unroll()`：
  - `while`/`for` 前紧邻的 `syntax.unroll()` 标记保留；条件必须是编译期值；
    循环体必须推进编译期状态，否则超过展开上限（`max_loop_unroll`，默认 1024）
    报错。
  - 编译期 `for` 的迭代器与循环变量仍建在编译期存储（comptime alloca）。
  - **变化**：旧语义里迭代器/循环变量是「浅层 inline」；新语义下它们必须是完全
    编译期值（`std.range` 的元素若是未定型字面量，靠 `Comptime[T]` 局部量保持
    编译期，行为不变）。
  - `syntax.unroll()` 必须紧接在被标记的循环之前，否则报错（保留）。
- `syntax.comptime()` 标记与 `syntax.unroll()` 标记的错误条件保留（见 §3.1、
  §10 的错误清单）。

---

## 7. 元组（tuple）的完整行为

元组从「无运行时表示」变为「相当于 struct」，但**解构、多返回值、铺平传参**的
语义保持不变。

### 7.1 作为运行时值

- 元组值是一个运行时聚合（位置字段），可赋给普通变量、作为字段、按值/按引用
  传参与返回（大小规则同结构体）。注意**形参/返回位置**上 `tuple[...]` 恒被特殊
  解释（铺平传参 / 多返回值，见 §2.3、§7.4），其余位置才是「一个元组值」。
- 元组下标 `t[i]` 是第 `i` 个元素的**左值**：可读可写可 `+=`（`i` 是编译期整数，
  越界/负数/运行时下标报错——保留）。
- `len(t)` 是编译期整数——保留。
- 元组是**可拷贝**当且仅当所有元素可拷贝；非拷贝元素使整体不可拷贝（同结构体）。
- 嵌套元组按元素类型递归（`t[1][0]`）。

### 7.2 解构赋值（RLS + `ResultLocTree`）

解构赋值**不使用** `hir.DestructureStore`（该指令不存在）；它仍然走 **result-location
semantics**，而且结果位置与「值」彻底分开：`type ResultLocTree = hir.Value | tuple[ResultLocTree, ...]`
（纯 place 树，叶子只含 `hir.Value`）与 `type FullResultLoc = ResultLocTree | ReturnLoc`
（`ReturnLoc` **不是** `hir.Value`，语义 =「外层函数的整棵结果位置树」，取代旧 `hir.ResultLoc`）。
「写穿」指令把结果**直接写进目的地**（`FullResultLoc`），不产生值寄存器；其余表达式先算值再 `hir.Store`。

- `a, b, ... = expr`：`_gen_lhs` 把左边的目标（名字 / 字段链 / 嵌套元组 / `_`）建成一棵
  `ResultLocTree`，再以它为 `result_loc` 对 `expr` 做 RLS 生成。
- **`expr` 是 tuple** → 就地生成（与现状相同）：`InitTuple` + 逐元素 `TuplePtrElement`，
  对每个元素递归 `_gen_result_loc`。
- **`expr` 不可拆解**（名字 / 属性 / 下标等）→ 默认分支 `Store(result_loc, value)`：
  `result_loc` 是具体树时 `astgen` 展开为 `target[i] = expr[i]`；是 `ReturnLoc()` 时单个
  `Store(ReturnLoc(), value)`，由 interp 按 `value` 的形状定型结果位置（单值 → 单 place；值形式
  tuple → N 叶树并逐元素 splat）。
- **返回位置**：`return` / lambda 体传 `ReturnLoc()`。返回注解**已声明** → 结果树由注解定型；**省略** →
  初始未定型，**第一次投递**按被投递物形状（`InitTuple` 的 arity / callee 的 `RetSpec` 叶）定型
  （`return f()` 转发即把 callee 的结果叶当作 caller 的结果 place）；互相递归且都省略注解 → 报错。
- 「等式两边都是 tuple」（`a, b = c, d`）由 `astgen` **拆分**为逐元素赋值，所以
  `InitTuple` / `TuplePtrElement` 只作用于**叶子**（如单变量元组赋值 `t = (c, d)`）。
- **RHS 不能是结构体/数组构造子**（解构一个结构体/数组构造在 `astgen` 报错）。
- 目标树 arity 必须与值/结果的元数一致，否则报错（保留）。
- **换手 `a, b = b, a` 不修正**：解构仍是「按下标顺序读源、写目标」，右侧读到可能已
  被覆盖的值（维持现状缺陷）。

**所有带 result location 的 HIR 指令，其目的地字段类型改为 `FullResultLoc`**：

| 指令 | 字段 |
|---|---|
| `hir.Binary` / `hir.Unary` / `hir.Ord` | `ret` |
| `hir.CallInplace` / `hir.CallMethodInplace` | `ret` |
| `hir.InitTuple` | `result_loc` |
| `hir.FinishStruct` / `hir.FinishArray` | 目的地 |
| `hir.TuplePtrElement` | `tuple_ptr` |
| `hir.Store` | `ptr`（`hir.Value | ReturnLoc`）|

**不变**（没有该字段）：`hir.StoreVoidRetloc`，以及所有产生**地址**的指令
（`FieldAddr` / `FieldIndexAddr` / `Subscript` / `Alloca` / `Arg` 等）——它们是 `ResultLocTree` 的**叶子**。

**删除**：`hir.TuplePtr`（原「元组 place」由 `ResultLocTree` 取代）。

### 7.3 多返回值

- 返回注解 `-> tuple[T1, T2, ...]` 仍表示多返回值（一个按值返回，其余经 result
  指针交付；嵌套元组展平成叶子；全经指针时返回 void）——**与现状一致**。
- **编译时返回值**：完全编译时已知的返回值（`classify()==COMPTIME`：const `ComptimePtr`、类型值、
  字节串、函数值等）作为 `RetSpec` 的 **`RetComptime`** 叶子（不占 MIR 结果、不占 by-value 槽）。
  - **inline 调用**：结果位置是调用点的 `ResultLocTree`（直接传进被调帧），callee 在编译期把值写进
    对应叶子（一个**编译期 place**）。
  - **非 inline 调用**：值在**编译 callee 时**得到——即 **callee 的 `interp` 分析过程中最后一条被分析的
    `return` 所返回的值**（每条 `return` 把自己的编译期值写入 callee `FunctionInstance` 上的槽，
    后写覆盖先写）。**本次重构不对「所有 `return` 路径是否给出同一个编译期值」做检查**（见 §15）；
    调用点取出该值后写进对应 place；该 place 必须是编译期 place（否则报错）。
  - **Python 调用方**：编译期叶子不经过内存（没有 out 指针）。`glue._read_result` 直接读 callee
    `FunctionInstance.ret_comptime_values[leaf.index]`，并按「有自然 Python 镜像的标量转成 Python
    `int`/`float`，其余**原样**交出编译器内部对象」交付（`glue._comptime_to_python`）。于是
    `def f(): return 1` 从 Python 调用得到 `1`（`def f(): return (1, 2)` 得到 Python 元组 `(1, 2)`）；
    类型值/指针/函数值等则原样是 `sval` 对象——它们的 Python 侧 API **未定义**（未定型行为）。
  - **未来**：可手动指定把 `classify()==NONE` 的类型也按编译时返回。
- 调用侧：解构 `a, b = f()`（多值结果经 RLS 写进 `ResultLocTree`）、或 `return f()`
  转发（逐个写进自己的 result location）——保留。
- **打包成一个变量**：
  - 旧语义：`packed: Comptime = f()`（结果打包进编译期元组）。
  - 新语义（**仅当所有结果叶 `classify() != COMPTIME` 时**）：由于元组有运行时表示，多值
    结果可以打包进**普通元组变量**：`packed = f()`（目标为单个名字时，其类型在 `CommitSlot`
    处由调用的结果推断为元组类型，并把各结果打包成该元组值）或显式标注元组类型。之后可
    `x, y = packed` 再解构、`len(packed)`、`packed[i]`。
  - **含编译时叶时**：只要任一结果叶 `classify() == COMPTIME`（类型值、const 指针、字节串、
    函数值等），打包出的元组类型就**没有运行时表示**，目标变量必须是 `Comptime` 变量
    （否则报错）；也可以只用解构 `x, y = f()` 而不打包。
  - **变化**：旧语义「只能放进编译期变量」的约束**仅对全运行时结果取消**；`packed: Comptime = f()`
    在结果含运行时值时仍按 §3.1 报错。
- 限制 `tuple[T, ...]`（变长）在任意嵌套层级报错——保留。

### 7.4 元组作参数 / 返回值

- 形参注解 `tuple[T1, ...]`：**铺平传参**，callee 重组为**合成 `StructType`**（不再是元组值本身）；
  元素为编译期值时不占 MIR 实参。可嵌套、可转发、泛型（`tuple[T, T]` 解类型参数）——保留。
- **callee 侧表示（Q1）**：该合成 struct 的字段名为 `_0,_1,…`；其**反射类型是 `StructType`**（不是
  `TupleType`），tuple/dict 形参靠字段名区分。
- 结构体元素**不**铺平（整体一个运行时实参，按值/按引用规则）——保留。
- 返回注解元组 = 多返回值（§7.3）。

### 7.5 `t += u`

- **不再支持**元组 `+=`（旧语义依赖「元组没有自己的存储、由元素 place 串构成、可零拷贝
  增长元数」的混合存储，已被删除）。
- 元数在构造后**固定**（元组是运行时聚合，无法原地增长）。
- `t = t + u` 仍不支持（没有元组 `+`）。
- 原来的 reduce / 累积用法改为**先补全元素类型、再 `reify` 出 tuple 类型、最后逐元素填充**：

  ```python
  # 累积 N 个元素的类型（编译期），再取出该元组类型的值并逐元素填充
  types = const_array(typed_undefined(type), N)
  unroll()
  for i in range(len(types)):
      types[i] = ...                      # 第 i 个元素的类型（编译期值）

  tuple_type = reify(TupleType(const_arr_slice(ref(types)), False))
  ret = typed_undefined(tuple_type)
  unroll()
  for i in range(len(ret)):
      ret[i] = ...                        # 第 i 个元素的值
  ```

  依赖 `std.reflect.reify` / `reflect.TupleType`、`std.core.const_array` / `const_arr_slice` /
  `typed_undefined`。

---

## 8. `StrDict`（`dict[str, T]` / `**kwargs`）的完整行为

- `StrDictType` 从「无运行时表示」变为「相当于 struct」（具名字段）。
- `**kwargs` 形参 / `dict[str, T]` 形参仍**按名铺平**传参，callee 重组为字典值。
- 访问：`len(kwargs)`、`kwargs[b'name']`（编译期字节串名）——保留。
- 反射：`std.reflect.type_info` 的 `StrDictType` 变体（`entries`：name + type）、
  `TupleType` 变体仍可用（见 §10）。但 `**kwargs`/`dict` **形参**绑定的合成 struct（`$kwargs`，
  字段名 = 条目名）反射为 **`StructType`**（Q1，见 §7.4/§10）。
- 无注解的 `dict` 参数：调用点给的任意形状，编译期按反射分类（保留；分类对象是合成 `StructType`）。
- **预期变化**：由于有运行时表示，字典值可存进普通变量/字段、按值或引用传递；
  条目访问 `d['name']` 仍**只支持编译期字节串名**（**不支持运行时键查找**）。

---

## 9. 闭包 =「带 `__call__` 的匿名 struct」

新语义下**没有「闭包」这一独立概念**：嵌套 `def` / `lambda` 求值出的是一个普通
**匿名 struct 值**，带一个 `__call__` 方法（函数体即该嵌套函数）。

### 9.1 `hir.MakeClosure` → 匿名 struct

- `hir.MakeClosure` 产出一个**匿名 struct 类型**（不可写进源码注解，只能由值直接
  持有），其字段是闭包的捕获，`__call__` 是嵌套函数体。
- **捕获的编译时变量** → `is_comptime = True` 的字段，默认值是该变量的**引用**
  （`move = False`）或**值**（`move = True`）。
- **捕获的运行时变量** → **普通字段**：`move = False` 时字段存**指针**（指向被捕获
  变量），`move = True` 时字段存**值**（拷贝）。
- 调用 `f(...)` 即 `f.__call__(...)`；`__call__` 的 `self` **总是按指针**传。
- 该 struct 是否 ZST 决定它能否取函数指针（§9.5）。

### 9.2 `@syntax.closure` 参数

- 参数**全部保留**：`inline` / `exceptions` / `callconv` / `may_panic`（函数相关，
  **透传给生成的 `__call__`**）以及 `move`（捕获属性，`False` 默认按引用、`True`
  按值）。
- **泛型参数、`*args`/`**kwargs`** 按闭包镜像到 `__call__`（保留现状）。

### 9.3 捕获语义

- `move = False`（默认）：**全部按引用**——捕获量全存指针。运行时变量写穿指针
  **写回外层变量**；编译时变量存其引用（`ComptimePtr`）。
- `move = True`：**全部按值**——捕获量全存副本，闭包内的写不回外层。对**运行时**
  捕获量，副本落在普通字段里（可写）；对**编译时**捕获量，「按值」意味着把该**值**
  烘进 struct 类型（见下），因此它实际是**类型级常量**，闭包内对它写入是编译错误
  （并非「可写的局部副本」）。
- 捕获的编译时变量是 `is_comptime` 字段，其**引用/值烘进 struct 类型**；因此每次
  捕获（不同 box）产生**不同的匿名 struct 类型**（沿用旧的「按创建点特化」语义）。
  `hir.MakeClosure` 是**编译时指令**：运行时循环体只编译一次，故闭包类型不会随
  运行迭代重复产生。
- **匿名 struct 类型的身份**：捕获（引用或值）烘进字段的**默认值**，因此 **struct 类型的相等/哈希
  必须包含字段的默认值**（编译时捕获的 `ComptimePtr` 按身份/内容，`move` 捕获的值按结构），
  类型缓存（`MirLowerCache`）也据此区分；`is_comptime` 字段的默认值不参与 MIR 布局缓存。
- **§4.2 的限制不适用于闭包捕获**：编译时指针「挂在 struct 类型上」，即使
  `__call__` 被编译也不报错。**原因**：捕获是把指针**烘进 struct 类型**，不是把指针物化成
  运行时值；读 `is_comptime` 字段的**值**在编译期折叠成常量（可进运行时代码），只有把该
  编译时变量的**地址**（`ComptimePtr`）物化成运行时值才受 §4.2 限制——那是**非 const 编译期
  指针的通用限制，与闭包无关**。**代价**：编译结果可能依赖编译顺序（在编译期写了非
  const 的编译时指针、之后又读它的值），这是用户需自行避免的。

### 9.4 一等值与旧限制的取消

- 因为闭包就是 struct 值，旧的全部限制**自动取消**：可存进变量/字段、可传参、
  可返回、可传给运行时函数、可放进运行时存储。
- `std.core.as_runtime_closure` **删除**（其功能被「闭包即 struct」取代）。

### 9.5 `as_func_ptr`

- `syntax.as_func_ptr(T, f)`：**第一个参数 `T` 只给 Python 类型检查器看，实现忽略之
  （Q3）**——不再校验它。`f` 必须是一个**有 `__call__` 方法的 struct**，且该 struct 是 **ZST**
  （即捕获的运行时量为零）；结果是**去掉 `self` 参数的 `__call__`**（一个不含 `self` 的
  `fn.FunctionValue`，`get_type` 返回**常**函数指针），由 `lower` 自动 lower。实现上包一层：
  体内先 `self_val = typed_undefined(T)`（ZST 的「未定义实例」携带类型级的编译期捕获），
  再 `return self_val.__call__(args…)`（Q2）。
- 普通 spy 函数与 extern 函数**直接用其名字**即可取到函数指针（无需 `as_static_ptr` /
  `as_func_ptr`）：函数值本身即**常**函数指针（`FunctionValue.get_type` 返回 `PointerType(fn_type, is_const=True)`），
  目标函数类型来自函数自身的签名，不必再显式给出 `T`。因此 `as_func_ptr` 现在只服务
  「ZST 闭包」这一种情形。

---

## 10. 反射与 std 影响

- `std.reflect.type_info(type) -> TypeInfo` / `reify(info) -> type` 保留；变体
  `IntType`/`PointerType`/`ArrayType`/`StructType`/`TaggedUnionType`/`OptionType`/
  `TupleType`/`StrDictType`/`ComplexType`/`UnionType` 保留。
- **变化**：`TupleType` / `StrDictType` 现在（在成员都有运行时表示时）描述有运行时
  表示的类型；反射结果（元素类型列表、字典条目）不变，但这样的类型现在可用于运行时
  存储。
- **`reflect.StructField` 新增 `is_comptime` / `is_ref` 字段**：用于呈现 struct 字段的
  这两个标记（含闭包 struct 的 `is_comptime` 捕获字段）。
- **元组/字典形参的反射（Q1）**：`TupleType`/`StrDictType` 变体本身仍存在（用于显式元组/字典
  **类型**），但 `tuple[...]`/`**kwargs` **形参**绑定的合成 struct 反射为 **`StructType`**
  （字段名分别为 `_0,_1,…` / 条目名）；不加特判。
- 字节串：**`std.core.gstr` 不再保留**——字节串编译期值经其 `ComptimePtr`（`box` 为该
  `bytes` 值）由 `lower` 生成运行时全局常量（并在末尾补一个 NUL）。`std.core.sstr` 保留为
  **库函数**：
  ```python
  def sstr(s: bytes) -> ConstSlicePtr[u8]:
      return ConstSlicePtr(ptr_cast(s, ConstMultiPtr[u8]), len(s))
  ```
  （`s: bytes` 因 `bytes` 无运行时表示而**自动成为编译期形参**，见 §3.2；`ptr_cast(s, …)` 得到
  `box=s` 的 `ComptimePtr`。）持有字节串的局部量仍必须是编译期变量。
- `std.mem` 的 `layout_of` / `size_of` / `align_of`：现在也接受元组/字典类型
  （因为它们有运行时布局）。
- `std/mem.py` 的 `DynamicAllocator.create` 等使用 `as_static_ptr` 的地方按 §4.3
  迁移（`const_value: ConstComptime = ...; ref(const_value)`）。
- **`as_func_ptr` 语义调整**：`syntax.as_func_ptr(T, f)` 要求 `f` 是一个**有
  `__call__` 且为 ZST 的 struct**，转换为**去掉 `self` 的 `__call__`** 函数
  （`fn.FunctionValue`，`get_type` 返回**常**函数指针，由 `lower` 自动 lower）。详见 §9.5。
- **`std.core.as_runtime_closure` 删除**：闭包本身即 struct 值，不再需要它。
- **全局与函数值**：普通 spy 函数与 extern 函数**直接用名字**取到**常函数指针**（`FunctionValue.get_type`
  返回 `PointerType(fn_type, is_const=True)`）；全局（函数、类型、捕获常量）是**编译期值**，需要指针时
  得到 `ComptimePtr`。`hir.ConstRef`/`sval.ConstRef` 不再存在（见 §9.5、§11）。
- **函数值的类型**：`fn.FunctionValue.get_type()` 返回**常函数指针** `PointerType(fn_type, is_const=True)`，
  因此 `spy.typeof(f)`（及任何观察函数值类型的地方）现在得到的是**函数指针类型**而非函数类型——
  这是有意的行为变化（见 §14.11）。

---

## 11. 删除项与迁移对照

| 删除/变化 | 新做法 |
|---|---|
| `ComptimeTuple` / `ComptimeTuplePtr` | 元组有运行时表示（struct 式）；编译期元组铺平存进 Box |
| `ComptimeDict` / `ComptimeDictPtr` | 字典有运行时表示（struct 式）；编译期字典铺平存进 Box |
| `ComptimeAggregate` / `ComptimeAggregatePtr` | 编译期聚合铺平为 Box 的标量序列，按 `offset` 寻址 |
| `ComptimeOption` / `ComptimeOptionPtr` | 完全编译期的 Option 是编译期聚合；tag 运行时的 Option 是普通运行时值 |
| `ComptimeTaggedUnionValue` / `ComptimeTaggedUnionPtr` | 同上 |
| `ComptimeBox` | `sval.Box`（+ `ComptimePtr`） |
| `ComptimeCastedPtr` | `sval.CastedPtr`（编译期指针的 `ptr_cast` 结果） |
| `PendingSlot.inline_mode` | `PendingSlot.is_comptime: bool \| None`（**非机械替换**，使用点逐个重判） |
| `ComptimePtrArg`（复制 pointee） | 删除；直接用 `ComptimePtr`（`Box` 身份对象 + 结构化 eq/hash、全局别名；非 const 只能进内联） |
| `hir.ConstRef` / `sval.ConstRef` | 删除；全局在 HIR 里是**值**（需要 place 时装入编译期 box）；「编译期常引用」= `ComptimePtr` |
| `std.core.as_static_ptr` | `const_value: ConstComptime = value; ref(const_value)`（自动 lower） |
| `std.core.gstr` | 删除；字节串直接 lower（末尾补 NUL） |
| `fn.ClosureValue` / `hir.Closure` / `hir.RebuildCapture` | 删除；闭包 = 匿名 struct + `__call__` |
| `sval.ClosureType` | 删除；闭包值即其匿名 struct 类型（DST 函数类型 `AnyFunction`/`FunctionType` 仍保留） |
| `hir.CallInplace.captures` / `fn.CallSignature.captures` | 删除；捕获经 struct 字段（`self`）读取 |
| `std.core.as_runtime_closure` | 删除；闭包本身即 struct 值 |
| `syntax.as_func_ptr` | 保留，改为接受 ZST struct、取去掉 `self` 的 `__call__` |
| `hir.DestructureStore` | **不引入**；解构走 RLS + `ResultLocTree` |
| `hir.TuplePtr` | 删除；由 `ResultLocTree` 取代 |
| 浅层（inline）编译期 | 删除；只有「完全编译时已知」 |

---

## 12. 行为保留清单（重构后仍成立）

以下行为与现状一致，重构后应保持：

- `syntax.comptime()` / `syntax.const_comptime()` 标记的语法与错误条件：必须
  紧接新声明；冗余标注报错；不能标记赋值；不是值；不能用于循环/非声明。
- `x: Comptime[T]` / `x: Comptime` 的语义（除「可持运行时值」一条外）；
  `x: ConstComptime[T]` / `x: ConstComptime` 的语义（必须带初值、commit 后只读）。
- `syntax.unroll()` 编译期循环的语法与展开/上限行为。
- 编译期 `if` 的选择与死分支不编译。
- `spy.typeof` 是类型探针：只类型化、不发代码；死分支不运行；实参里的 spy 函数
  被触发编译；被探测的调用若抛出错误离开探针则报错。（探针语义不变；但函数值本身的类型
  见 §14.11。）
- `spy.compile_log(...)`：编译期打印日志、运行时无动作。
- `std.core.compile_option(name)`：编译期选项（诊断开关）。
- `std.core.compile_error(msg)`：以编译期字节串为消息中止编译；非字节串/运行时
  消息报错；死分支不触发。
- `std.core.undefined()` 的转换与「是编译期值」的性质；undefined 作条件报错。
- `ref` / `ptr_cast` / `p[...]` / 指针 const 隐式转换。
- 多返回值的降级签名（一值按值 + 其余 result 指针；全指针时 void；嵌套展平）。
- 元组参数铺平、元组返回多值、泛型元组；解构赋值走 RLS（结果位置为 `ResultLocTree`）。
- `*args`/`**kwargs` 的绑定与访问（`len`、`args[i]`、`kwargs[b'name']`）；元素/
  值类型可注解或推断；`type[X]` 元素不允许。
- 嵌套 `def`/`lambda` 的捕获/调用、`@syntax.closure` 的 `inline`/`exceptions`/
  `callconv`/`may_panic`/`move` 选项；泛型参数与 `*args`/`**kwargs` 的镜像。
  （旧「闭包不能传给运行时函数 / 不能存运行时位置」等约束**取消**；
  `as_runtime_closure` 删除。）
- `std.reflect` 的 `type_info`/`reify` 语义。
- 泛型（类型参数求解、返回类型推断、构造推断、`Foo[T]` 显式特化）。
- ZST 规则（无存储、不占布局、单位值）。

---

## 13. 错误清单（重构后）

| 触发 | 错误（建议消息） |
|---|---|
| 把运行时值写进 `Comptime` / `ConstComptime` 变量/参数 | `cannot store a runtime value into a compile-time variable` |
| 用含运行时值的聚合初始化 `Comptime` 变量 | 同上（整个值不是编译期值） |
| `Comptime` 参数收到运行时实参 | 报错（不静默丢弃） |
| 写入 `ConstComptime`（`is_const = True` 的 box）/ 写只读的指针视图 | 报错 |
| `ConstComptime` 声明无初值 | 报错（必须带初值） |
| **无注解变量**（运行时存储）持有编译期专属类型（如未定型整数 `x = 1`） | 报错（`_no_runtime_type`：需类型标注 `x: i32 = 1` 或 `Comptime` 变量）|
| 编译期 `if` / `unroll` 条件不是编译期值 | 报错 |
| `syntax.comptime()` / `syntax.const_comptime()` / `syntax.infer_comptime()` 未紧接新声明 / 冗余标注 / 用作值 | 与现状相同的报错 |
| 形参（含 `*args`/`**kwargs`、闭包形参）写 `InferComptime` | 报错（推断仅限局部变量声明） |
| 解构目标 arity 与值元数不符 | 报错 |
| 解构的 RHS 是结构体/数组构造子 | 报错（astgen） |
| 元组下标越界 / 负 / 运行时 | 报错 |
| 元组 `+=`（已不再支持） | 报错 |
| `tuple[T, ...]` 变长返回 | 报错 |
| `as_func_ptr` 的 `f` 不是「有 `__call__` 的 ZST struct」 | 报错 |
| 写透 `is_ref` 字段（存常指针，不可写） | 报错 |
| 为**运行时 struct** 的 `is_comptime` 字段提供构造实参 | 报错（只有编译时 struct 变量能覆盖） |
| 同一存储的多次聚合构造类型不同（`if` 两臂，含两个不同的 ZST struct） | 报错（无 peer 类型，Q5） |
| `compile_error` 非字节串/运行时消息 | 报错 |
| 传递给 `typeof` 的调用抛出错误离开探针 | 报错 |
| 结果位置形状循环依赖（互相递归省略返回注解） | 报错（结果位置无法定型） |

---

## 14. 主要行为变化汇总（旧 → 新）

1. **无混合存储**：`Comptime` 变量/参数只能持完全编译期值；含运行时值的聚合整体
   是运行时值（用普通变量）。
2. **编译期值严格为 `sval` 编译期值 + 编译期对象**（如 `fn.FunctionValue`）：不再
   有「不引用寄存器的运行时值」这一类；不存在运行时值形式的编译期常量。
3. **编译期指针全局化、按身份别名、不复制**：但**跨非内联边界仅限 const 指针**
   （非 const 只能进内联）；编译期返回值经 `RetSpec` 的 `RetComptime` 叶子交付
   （见 §4.2、§7.3）。
4. **`ConstComptime`**：新增 commit 后只读的编译期常量；`is_const = True` 的 box 由
   `lower` **自动 lower** 成 `mir.GlobalConstant`；**函数值**则直接 lower 成**函数符号本身**
   （`mir.Function` / `ExternAnonSymbol` / `ExternSymbol`，不经 `GlobalConstant`）；可变
   编译期变量暂不支持 lower。
5. **`as_static_ptr` 移除**：用 `ConstComptime` + `ref` 取代（自动 lower）；`gstr`
   移除（字节串直接 lower，末尾补 NUL）；`sstr` 保留为库函数；`as_func_ptr` 改为
   把无运行时捕获的闭包转成 `fn.FunctionValue`。
6. **元组/字典有运行时表示**：可存普通变量、作字段、按值/引用传递；解构仍走 RLS，
   结果位置改为树状 `ResultLocTree`（不引入 `DestructureStore`，删除 `hir.TuplePtr`）；
   多值结果可打包进普通元组变量；元组 `+=` **不再支持**。
7. **闭包 = 匿名 struct**：`ClosureValue` 等删除；`MakeClosure` 产出带 `__call__`
   的 struct；捕获成字段（编译时捕获 `is_comptime`、运行时捕获普通字段，`move`
   决定指针还是值）；调用不带 captures；`as_runtime_closure` 删除；`as_func_ptr`
   接受 ZST struct 取 `__call__`。旧闭包限制全部取消。
8. **struct 字段新标记**：`is_comptime`（不进 MIR、ZST 判定排除；运行时 struct 读默认值、
   编译期 struct 读 box cell）、`is_ref`（存常指针、`field_index_addr` 额外 load）。
9. **`PendingSlot.inline_mode` → `is_comptime`**：三态，`None` 表示推断；源码里用
   `InferComptime`/`syntax.infer_comptime()` 显式请求推断（形参不允许）。
10. 旧 `Comptime*Value`/`Comptime*Ptr` 形式全部消失，编译期存储统一为 `Box`；
    `ComptimeCastedPtr` → `sval.CastedPtr`。
11. **函数值的类型**变为**常函数指针**：`fn.FunctionValue.get_type()` 返回
    `PointerType(fn_type, is_const=True)`，故 `spy.typeof(f)` 等观察点从「函数类型」变为「函数指针类型」。
12. **无运行时表示的类型的形参自动成为编译期形参**（`classify()==COMPTIME` ⇒
    `SignatureFormalArg.is_comptime = True`，见 §3.2）。

---

## 15. 遗留问题（待确认）

除一项待决（Q7 后半）外，另有一项**明确不做**的已知限制；其余「明确不做」项见 task 文档 §17：

- **Q7（后半）**：手写指定把 `classify()==NONE` 的类型也按编译时返回。**本次重构暂不决定其实现、不实现**，
  留待后续。
- **非内联调用的编译期返回值不做一致性检查**（见 §7.3）：取 callee 的 `interp` 分析过程中**最后一条
  被分析的 `return`** 返回的编译期值，不检查各 `return` 路径是否给出同一个值。

> 其余原遗留项均已解决并迁入 task 文档正文：Q13→§8.1、Q14/Q15→§0.2/§2.4/§4.1、Q16/Q17→§4.2、
> Q18→§4.1/§4.3/§10、Q20→§7.2，以及更早的 Q5/Q6/Q11/Q19→§4.3/§4.2/§2.1·§2.2/§5.3。

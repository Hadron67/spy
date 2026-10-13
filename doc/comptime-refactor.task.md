# 编译期（comptime）重构：修改任务清单

本文档是 [`comptime-refactor.behaviour.md`](./comptime-refactor.behaviour.md) 的实现落地清单：
把「目标行为规格」翻译成**逐文件、逐函数/类**的新增 / 修改 / 删除，并尽量列出需要清理的旧逻辑。

- 语言：中文（与行为文档一致）。
- 记号：`新增`（新增的类/函数/字段）、`修改`（语义或签名变化）、`删除`（整段移除）。文件路径写项目相对路径。
- 章节编号尽量对齐行为文档 §0–§14，末尾另有「删除清单」「实施顺序」「待确认问题」。
- 尚未确定的点集中在文末 §17；已确认的决定已直接写入正文对应章节。

---

## 0. 总体模型与关键决策

### 0.1 「一分」模型

目标：**只有一层** —— 一个值要么是**运行时值**（有 MIR 表示），要么是**编译期值（完全编译时已知）**。
现有实现的「浅层 inline / 深层 comptime 两层 + 混合存储」被整体删除：

- 删除 interp 中全部「编译期聚合/元组/字典/Option/union 的混合存储形式」：
  `ComptimeTuple`、`ComptimeDict`、`ComptimeTuplePtr`、`ComptimeDictPtr`、`ComptimeAggregate`、
  `ComptimeAggregatePtr`、`ComptimeOption`、`ComptimeOptionPtr`、`ComptimeTaggedUnionValue`、
  `ComptimeTaggedUnionPtr`、`ComptimeBox`、`ComptimeCastedPtr`（共 12 个类）。
- 编译期存储统一为 **`sval.Box`**（编译期 alloca，**铺平保存 `list[ScalarValue]`**）+ **`sval.ComptimePtr{box, offset}`**（一个 `sval.Value`，见 §0.2）。
- 编译期指针的 `ptr_cast` 结果统一为 **`sval.CastedPtr`**。
- 闭包不再是独立值种类：`fn.ClosureValue` / `hir.Closure` / `hir.RebuildCapture` /
  `hir.CallInplace.captures` / `fn.CallSignature.captures` / `std.core.as_runtime_closure` 全删，
  改为 **`hir.MakeClosure` 产出「带 `__call__` 的匿名 struct」**。
- HIR 的 result location 变为树状 **`ResultLocTree`**，并与「值」彻底分开：`hir.py` 里新增
  **`type ResultLocTree = hir.Value | tuple[ResultLocTree, ...]`**（纯 place 树，叶子只含 `hir.Value`）与
  **`type FullResultLoc = ResultLocTree | ReturnLoc`**，其中 **`ReturnLoc` 不是 `hir.Value`**（取代旧的
  `hir.ResultLoc`，语义 =「外层函数的整棵结果位置树」）。`ResultLocTree` 定义从 astgen 提到 `hir.py`；
  删除 `hir.TuplePtr`，**不引入** `hir.DestructureStore`（详见 §7.2）。

### 0.2 编译期存储：`Box`（铺平标量序列）+ `ComptimePtr{box, offset}`

- `x: Comptime = e` / `x: Comptime[T] = e` / `syntax.comptime(); x = e` → 一个 comptime alloca，
  其值是一个 `sval.ComptimePtr`，指向新 `Box` 的 `offset == 0`。
- `x: ConstComptime = e` / `x: ConstComptime[T] = e` / `syntax.const_comptime()` → 同上，
  但 commit 后 `box.is_const = True`。
- `Box.name_base` = 当前函数 `name_base`（见 `interp._class_name_base` /
  `FunctionInstance.mir.name_base` / `hir.MakeClosure` 生成闭包的 `name_base`）加 box 计数器。
- `is_const=True` 的 box 与 `fn.FunctionValue` 由「自动 lower」进入运行时（见 §4.3 映射）。

**数据模型（已定，取代「`Box.value` 持有一个 `sval.AnyValue`」）**：

- **`sval.ScalarType`**（**新增**，`Type` 子类）：所有**标量类型**的基类——`IntType`/`AnyIntType`/
  `FloatType`/`BoolType`/`VoidType`/`NullType`/`UndefinedType`/`EmptyType`/`BytesType`/`TypeType`/
  `AnyType`/`ValueType`/`IntEnumType`/`OpaqueType`/`TypeVar`/指针类型/函数类型（含 `AnyFunction`、DST）。
- **`sval.AggregateType`**（**新增**，`Type` 子类）：所有**聚合类型**的基类——`StructType`/`ArrayType`/
  `TupleType`/`StrDictType`/`ComplexType`/`OptionType`/`TaggedUnionType`/**`UnionType`**（无 tag 的错误
  payload union）。`ResultType` **不参与分类**（纯编译期、会 spread）。于是「标量 vs 聚合」是**类型层次**
  上的静态区分（`isinstance`），取代 interp 里临时的 `_is_aggregate`（现为 `StructType|ArrayType|
  ComplexType`，interp L778）。
- **`sval.ScalarValue`**：**`ScalarType` 的值** union（`Int`/`Float`/`Bool`/`Bytes`/类型值/函数值/`Null`/
  `Void`/**`UndefinedScalar`**/`ComptimePtr`/`CastedPtr` 等，加普通 Python 标量 / `bytes`）。**聚合类型**
  的值（struct / array / tuple / strdict / complex / **option** / **tagged union** / **union**）不在此列；
  `TypedNull` **不再算标量**（它是 `Option[T]` 的值，见 §5.3 / Q19）。
- **`sval.UndefinedScalar`**（**重命名**自 `sval.Undefined`）：**标量类型**的 undefined（`type` 是
  `ScalarType`），是合法的 cell 值。**`sval.UndefinedAggregate`**（**新增**）：**聚合类型**的 undefined 的
  **值形态**（`type` 是 `AggregateType`，不进 `ScalarValue`）；落 box 时按该聚合体展开、**每个标量叶 cell
  都是 `UndefinedScalar(叶类型)`**。`coerce_const`（`UntypedUndefined` 分支）按类型分派：ZST → 单位值、
  标量 `T` → `UndefinedScalar(T)`、聚合 `T` → `UndefinedAggregate(T)`。
- **`sval.Box`**（**不是** `sval.Value`；继承 **`util.IdentityObj`**，按对象身份相等/哈希；因此**不要**用会生成 `__eq__` 的 `@dataclass`）保存：
  - `name_base: str`（**只是 MIR 里的命名提示**，见 §4.2：非特化键、不要求唯一/稳定、无链接影响）；
  - `type: sval.Type`（声明/提交后的类型）；
  - `is_const: bool`；
  - `cells: list[ScalarValue]`：把聚合体**铺平（flatten）**得到的标量序列（标量类型只有 1 个 cell）。
- **`sval.ComptimePtr`**（一个 `sval.Value`）：**`@dataclass(frozen=True, slots=True)`**，字段
  `{box: Box | bytes, offset: int, type: PointerType}`——`type` 是它**看待的指针类型**（`get_type()` 直接返回它，
  其 `is_const` 即「指针视图」的只读性，见 §4.2）。`offset` 是所指对象**第一个标量**在 `box.cells` 中的
  **起始下标**；「对象本身」即 `offset == 0`。
  **`box` 也可以直接是一个 `bytes`**（字节串存储：**无 cell、只读**，`offset` 是字节内偏移；
  由 lower 生成 `mir.GlobalStringValue`，见 §4、§10）。
  **自动 eq/hash（已定，方案 A）**：`Box` 身份相等、`bytes` 内容相等 + 结构化字段 → 两个 `ComptimePtr`
  当且仅当**同一 box、同一 `offset`、同一视图类型**时相等。因此**视图类型（含 `is_const`）参与相等/特化键**：
  同址的 const 与非 const 视图是**不同**实参（见 §4.2）。
- **铺平**（原 §17-Q14/Q15，已定）：按类型结构做一次**深度优先展开**——struct 按字段序、array/tuple
  按元素序、strdict 按条目序、complex 为 `(re, im)`、**option / tagged union 为 `[tag, payload...]`**
  （tag 是第 0 个 cell，payload 从 `+1` 起）；嵌套聚合直接内联拼接，偏移 = 各前缀宽度之和。宽度规则：
  - **聚合体一律由成员定**（与自身是否 `is_zst()` 无关）：struct/array/tuple/strdict = 成员之和
    （含 `is_comptime` 字段；`is_ref` 字段按类型 `*const T` = 1 cell）、complex = 2、option = `1 (tag) + child`、
    tagged union = `1 (tag) + max(variant)`（payload 按**变体最大宽度**取齐，不足留 `UndefinedScalar`）、
    **untagged `UnionType` = `max(variant)`**（其 tag 是旁边的错误码，不占 box cell）。故**空 struct = 0 cell**、
    字段全为 ZST 标量的 struct 每个字段各占 1、单成员聚合体 = 该成员。
  - **标量类型（`ScalarType`）恒占 1 cell**（**含 ZST 标量**）。即「ZST 标量占 1，ZST 聚合体按成员（可为 0）」。
  - **布局只由「类型」决定，与具体值无关**：标量类型的 undefined 是 `UndefinedScalar`（1 个 cell）；
    聚合类型的 undefined（`UndefinedAggregate`）**不落成单个 cell**，而按该聚合体展开、**每个标量叶 cell
    都是 `UndefinedScalar(叶类型)`**（`undefined()` / `x: Comptime[T]` 未初值同理，见下「未设置的 cell」）。
  - **`is_comptime` 字段照常占位**（cell 数 = `scalar_count(字段类型)`，存构造时的覆盖值）；**`is_ref` 字段的
    cell 存一个编译时指针 `sval.ComptimePtr`**，`field_index_addr` 对其额外 load 一次（可跨到另一 box）。
    二者与「不进 MIR」是**两套独立规则**（MIR 布局 ≠ box 铺平），无需统一。
  - **helper**（`sval.py`，带缓存；**不复用** `StructType._calculate_mir`）：`scalar_count(t) -> int`
    （上述 cell 数）与 `field_scalar_offset(t, i) -> int`（第 i 个字段/元素/条目的起始偏移，深度优先
    前缀和；0-cell 成员不推进偏移；option/union 的 `i=0` 是 tag、payload 从 1 起）。「单成员聚合体 =
    该成员」是铺平的**推论**，非复用 `mirror_is_a_field`。
- **读写**：`store` 直接写 `box.cells[offset + i]`（标量），`load` 从 `cells[offset ...]` 读回；
  因此一个**已提交**的聚合也能**逐字段写**，取代被删的 `ComptimeAggregatePtr`/`ComptimeOptionPtr`/
  `ComptimeTaggedUnionPtr`/`ComptimeTuplePtr` 的「每字段一个 place」。不可变的 `sval.AggregateValue`
  只作 `load` 出来的**值快照**，不再充当可变存储。
- **未设置的 cell** 读作 `UndefinedScalar`：`x: Comptime[T]`（无初值）也可逐字段构造（见 §3.1）。

> **注**：`sval.Box` **直接替代**现有 `interp.ComptimeBox`（后者删除），编译期指针值改用
> `sval.ComptimePtr` 包装。

### 0.3 「自动 lower」的落点

行为文档说「由**编译期常量 lowering**自动 lower 成 `mir.GlobalConstant`」。实现上，常量 lowering
**放在 `interp._const_to_mir` 中**，不在 `lower.py`：

- `lower.py` 是 **MIR → LLVM**，`mir.GlobalConstant` 的常量构造/去重实际在 **`interp.py`**
  （`_const_to_mir` 家族 + `_string_global`）里完成，再由 `lower._Lowerer._lower_global_constant/_lower_constant` 落到 LLVM。

因此本清单把「自动 lower」实现为：**在需要运行时表示的边界，自动触发 `interp._const_to_mir` 家族**
（而不是由用户显式调用 `as_static_ptr`）。`lower.py` 除函数值的表示外不改。

**做法（已定）**：lower 一个 box 时**先把 box `load` 成一个完全编译时已知的 `sval.Value`**
（聚合快照；`is_comptime` 字段按 box 铺平读出、`is_ref` 字段是其中一个 `ComptimePtr`），
再把**该值**交给 `_const_to_mir` 家族 lower 成 `mir.GlobalConstant`。于是「box 铺平布局 ≠ MIR 布局」
的差异全部由「值 → MIR」这一步消化：`is_comptime` 字段自然被跳过、`is_ref` 字段按指针写出
（指向被指向对象的独立全局）；lower **不直接读 `box.cells`**。

---

## 1. 值的两分法（`_is_comptime_val` 重写）

### 涉及文件

- `spy/compiler/interp.py`

### 修改

- `_is_comptime_val(val)`：重写。新语义下：

  - `ComptimeVal` → 总是编译期值。
  - `sval.ComptimePtr` → 总是编译期值（它指向 comptime 存储；`Box.is_const` 不影响「是否编译期」）。
    读透（`p[...]`）得到编译期值；`sval.CastedPtr` 同理（不能读写透，但仍是编译期值）。
  - `sval.Box` **不是** `Value`、也不单独作为 `InterpVal` 出现；它只经 `ComptimePtr` 到达，
    因此 `_is_comptime_val` 里无需为它单独分支。
  - `RuntimeVal` → **一律不是编译期值**（删除「不引用寄存器即可视为编译期值」的放宽；对应删除 `_references_register` 在此处的使用）。
  - `PendingSlot` → 保留：`committed` 递归；`declared` 且所有 store 为 inline 编译期值；否则 False。
  - `fn.FunctionValue`（含 `sval.DeclaredFunction`）→ 是编译期对象（`get_type` 报函数指针）。
  - 其它类型值 → 编译期值。

- 删除 `_is_comptime_val` 中针对 `ComptimeTuple/ComptimeDict/ComptimeAggregate/ComptimeOption/
  ComptimeTaggedUnionValue` 及其 `*Ptr` 的分支（这些类本身被删，见 §7）。

### 连带修改

- `_references_register` 保留（`_const_runtime_to_mir` 仍用），但**不再被 `_is_comptime_val` 引用**。
- `_stores_decide_storage`：保留其判据含义，但改名为「可寻址/存储由 store 决定」的判据，且
  输入类型集合随 `TupleType`/`StrDictType` 获得运行时表示而变化（见 §11）。

---

## 2. 类型的运行时表示（`sval.py`）

### 2.0 类型层次：`ScalarType` / `AggregateType`

`spy/compiler/sval.py`：

- **新增** `class ScalarType(Type)`：标量类型基类；把 `IntType`/`AnyIntType`/`FloatType`/`BoolType`/
  `VoidType`/`NullType`/`UndefinedType`/`EmptyType`/`BytesType`/`TypeType`/`AnyType`/`ValueType`/
  `IntEnumType`/`OpaqueType`/`TypeVar`/`PointerType`/函数类型（`FunctionType`/`AnyFunction`）
  改为其子类。
- **新增** `class AggregateType(Type)`：聚合类型基类；`StructType`/`ArrayType`/`TupleType`/`StrDictType`/
  `ComplexType`/`OptionType`/`TaggedUnionType`/`UnionType` 改为其子类。`ResultType` 不参与（纯编译期）。
- `sval.ScalarValue` = `ScalarType` 的值 union；`sval.Undefined` → 重命名为 **`UndefinedScalar`**（标量），
  **新增** **`UndefinedAggregate`**（聚合类型，值形态）。详见 §0.2。
- `interp._is_aggregate`（L778，现为 `StructType|ArrayType|ComplexType`）→ **删除**，改用
  `isinstance(type, sval.AggregateType)`（调用点 L478/L770/L2155/L3849/L3952/L4773/L4933/L7797/L7825/L8655）。
- `MirLowerCache`（L69）→ **修改**：新增 `TupleType → StructType` / `StrDictType → StructType` 两个镜像缓存
  （与既有 `option_struct_mir`/`tagged_union_mir`/`complex_mir` 同构），供 §2.1/§2.2 合成并复用 `StructType` 布局。

### 2.1 `TupleType`

`spy/compiler/sval.py`：

- `TupleType.classify()`（现返回 `COMPTIME`）→ **修改**：当**所有元素都有运行时表示**时按 struct 规则返回
  `ZST` / `NONE`；任一元素无运行时表示（`type`、`BytesType`、DST 函数类型、`EmptyType`）时仍返回 `COMPTIME`。
- `TupleType.to_mir_type(cache)`（现返回 `None`）→ **修改**（原 §17-Q11，已定）：**为每个 `TupleType` 合成一个
  等价的 `sval.StructType`**（字段 = 各元素、位置名），把它**缓存在 `MirLowerCache`**（`TupleType → StructType`），
  再 **lower 该 struct**（返回 `struct.to_mir_type(cache)`）——**整体复用** `StructType._calculate_mir`
  （对齐重排、只含一个非 ZST 字段时 `mirror_is_a_field`、全 ZST 无镜像），不再另写布局/缓存。
- `TupleType.get_field_mir_indices(cache)` / `mirror_is_a_field(cache)`（**新增**）→ 直接**委托给上面缓存的
  struct**（`field_index_addr` / `_extract_aggregate_value` 由此拿到访问器）。
- `TupleType.is_copyable()`（现 `all(t.is_copyable())`）→ 保留，但需与新的布局/`classify` 一致。
- `TupleType.get_type_children()` 保留。

> `tuple[...]` 在**返回注解 / 形参注解**位置仍被特殊解释（多返回值 / 铺平传参），见 §7 与 §2.3。

### 2.2 `StrDictType`

`spy/compiler/sval.py`：

- `StrDictType.classify()` → **修改**：所有条目类型都有运行时表示时返回 struct 式 `NONE`/`ZST`，
  否则 `COMPTIME`。
- `StrDictType.to_mir_type(cache)` → **修改**（原 §17-Q11，已定）：同 `TupleType`——**为每个 `StrDictType` 合成一个
  等价的 `sval.StructType`**（每个具名条目一个字段、顺序为条目声明顺序），缓存到 `MirLowerCache`
  （`StrDictType → StructType`），再 lower 该 struct。`get_field_mir_indices` / `mirror_is_a_field`
  （**新增**）委托给缓存的 struct。
- `StrDictType` 现在是 `frozendict[str, Type]`；保持字段顺序（`frozendict` 保序）。

### 2.2（续）仍无运行时表示的类型（纯编译期）

以下 `classify()==COMPTIME` / `to_mir_type()==None` **保持现状**，无须改：

- `TypeType`、`BytesType`、`FunctionType` / `AnyFunction`（DST）、`EmptyType`。

`BytesType` 额外：字节串**直接 lower 成运行时全局常量**（末尾补 NUL），可被 `ptr_cast` 成指针
（得到 `box` 为该 `bytes` 值的 `ComptimePtr`）。实现落在 interp 的 `_const_to_mir`/`_string_global`
与 `_const_pointer_to_mir` 的「`box` 是 `bytes`」分支（见 §4、§10、§14）。

### 2.3 位置的类型解释（保留）

- 返回注解 `tuple[...]` = 多返回值；形参注解 `tuple[...]` = 铺平传参。**不改**（`parse_function`、
  `Signature`、`make_ret_spec`、`_init_args_from_signature` 的相关分支保留）。
- 元组/字典类型作为**字段 / 变量 / 表达式值**时用新运行时表示。

### 2.4 struct 字段的两个新标记

`spy/compiler/sval.py`：

- `StructField`（`@dataclass(frozen=True)`）→ **新增**字段 `is_comptime: bool = False`、`is_ref: bool = False`。
- `StructTypeHead.add_field(name, type, default)` → **修改**签名，增加 `is_comptime`/`is_ref` 参数
  （或改为传入 `StructField`）。
- `StructType.fields()`（复制 head 字段时用 `replace_type_vars_type`）→ **修改**：保留新标记。
- `StructType.classify()` → **修改**：
  - 含**无默认值**的 `is_comptime` 字段 → 返回 `COMPTIME`（纯编译期类型）。
  - `is_comptime` 字段**不参与** ZST 判定。
- `StructType._calculate_mir()` → **修改**：`is_comptime` 字段**不进 MIR**（如同 ZST：`continue`，不占布局）。
  `is_ref` 字段的**存储/MIR 类型** `= *const T if not T.is_zst() else 无`（`T` 为 ZST 时该字段为 ZST、
  不占布局）；`StructField.type` 仍是逻辑类型 **`T`**，**ZST 判定也按 `T`**（显式例外，不看它实际存的指针）。
- **铺平里的 `is_comptime`/`is_ref`（原 §17-Q15，已定）**：comptime box 的 `cells` **照常含 `is_comptime` 字段**
  （占其类型的 cell 数，存「构造时覆盖值」）；**`is_ref` 字段**的 cell 存一个编译时指针 `sval.ComptimePtr`
  （**恒占 1 cell**，与 `T` 是否 ZST 无关），`field_index_addr` 对其**额外 load 一次**（读到 `ComptimePtr`，
  可跨到另一 box，读写作用于其 pointee）。
  这与「`is_comptime`/`is_ref` 不进 MIR」是**两套独立规则**、无需统一（详见 §0.2 铺平条目）。
- **`is_comptime` 字段的类型可以没有运行时表示**（它不进 MIR）：一个含「有默认值、类型无运行时表示」的
  `is_comptime` 字段的 struct 仍可为 `ZST`/`NONE`（`classify()` 不受该字段类型影响）。故
  「类型有运行时表示 ⟺ 所有组成部分都有」**不成立**。
- **`is_ref` 字段的 `T` 为 ZST 时**：MIR 里不占存储；`field_index_addr` 返回
  `undefined(*const T)`，load 它直接得到 `T` 的值（`store` 对 ZST 特判），故 ZST 下「额外 load」有定义。
- **box 的 lower 走「先 load 成 `sval.Value`、再 lower 该值」**（见 §0.3/§4.3），lower 不直接读 `box.cells`。
- `StructType.is_copyable()` → **修改**：`is_comptime` 字段排除；`is_ref` 按常指针处理。
- `StructType.is_subtype_of` / `get_unit_value` 等按需保持一致。

`spy/compiler/dsl.py`：

- `_RegisteredClass.get_entry()` 读取类体注解建字段处 → **修改**：`is_comptime`/`is_ref` 标记
  **暂不支持在源码中指定**，目前只由编译器内部生成（varargs/kwargs 的 struct 表示，见 §8.1）。
  因此这里只需给 `head.add_field` 传递默认 `False`，不需要新的注解语法。

`spy/compiler/interp.py`：

- `field_index_addr(...)` → **修改**：先取字段的**指针**（如 `mir.Gep` 返回的就是字段指针）。
  对 `is_ref = True` 字段，此时需**再 load 一次**，以得到该字段存储的**指针的值**
  （即 pointee 的地址），后续读作用于 pointee；写透是编译错误。
  - `is_comptime` 字段：**运行时 struct** 下**不取存储地址**，直接返回其**默认值**（编译期值/常量）；
    **编译期 struct**（box 铺平）下返回 **box cell 里的当前值**（覆盖值或缺省默认值）（§5.1）。
- `finish_struct(...)` → **修改**：
  - 运行时 struct 的 `is_comptime` 字段**不接受**构造实参（编译错误，§13）。
  - 编译时 struct 变量可为 `is_comptime` 字段覆盖默认值（覆盖值进编译期聚合值）。
  - `is_comptime` 字段缺省时直接取默认值，不产生 store。
- `exec_field_name_addr(...)`（`s.a` 读）→ **修改**：`is_comptime` 字段运行时 struct 下给默认值、编译期 struct 下给 box cell 的当前值（同 `field_index_addr`）；`is_ref` 走额外 load。

`spy/std/reflect.py`：

- `StructField`（reflect 版）→ **新增** `is_comptime: bool`、`is_ref: bool` 字段。
- `interp._build_struct_field` / `_reflect_struct`（L7394-7500）→ **修改**：填充这两个字段。

---

## 3. 编译期变量与参数

### 3.1 `Comptime` 变量（comptime alloca）

`spy/compiler/interp.py`：

- `alloca(...)`（L4756）→ **修改**：`Comptime` 变量分支改为产出一个 **`sval.Box`**（或持有 box 的
  `PendingSlot.committed`），返回值是 `ComptimePtr{box, offset=0}` 的 `ComptimeVal`。
- `store(ptr, value)`（L3800）→ **修改**：`ComptimePtr` 作为 `ptr` 时按 `offset` 写
  `box.cells[offset + i]`；写运行时值是编译错误 `cannot store a runtime value into a compile-time variable`。
  **写透要求「指针视图」与「box」都可写**（见 §4.2）。
  删除分支：`ComptimeBox`、`ComptimeAggregatePtr`、`ComptimeOptionPtr`、`ComptimeTaggedUnionPtr`、
  `ComptimeTuplePtr` 的整段 store 处理（这些类删除）。
- `load(...)` → **修改**：`ComptimePtr` 按 `offset` 从 `box.cells` 读回。
- `_type_of` → **修改**：`ComptimePtr` 的类型 = 它所携带的 `PointerType`（const 视图与 `Box.is_const` 分开，见 §4.2）；
  删除 `ComptimeBox/ComptimeAggregatePtr/...` 分支。
- `_shallow_normalize` → **修改**：`PendingSlot.committed` 可能是 `sval.ComptimePtr` 的包装，保持归一。

### 3.2 编译期参数

- `SignatureFormalArg.is_comptime`（已有）保留。
- **新增（已定）**：形参解析出的类型 `classify() == COMPTIME` 时**自动**置
  `SignatureFormalArg.is_comptime = True`（等价 `x: Comptime[T]`）。落点：生成 `SignatureFormalArg` 的
  签名求解处（`Signature`/`solve_param_types` 等），并让其不占 MIR 实参。因此 `def sstr(s: bytes)`
  这类库函数无需显式 `Comptime`。
- `interp._init_arg_node`（L1989）/ `_init_ptr_target`（L2071）/ `_init_args_from_signature`（L2169）→ **修改**：
  编译期实参直接以 `sval.ComptimePtr` 承载（删除 `ComptimePtrArg`）；**新增**：const `ComptimePtr`
  （`is_const=True` 的 box）可跨非内联边界，callee 得到同一 `ComptimePtr`（同一 box 同一 `offset`），可别名。
- `interp._check_comptime_args`（L1764）保留语义。
- const `ComptimePtr` 跨非内联边界时**按身份**传递（进特化键、callee 别名同一 box），**不 lower**；
  只有被**物化成运行时值/地址**时才 lower（见 §4.2/§4.3）。

### 3.3 普通变量、`is_comptime` 的推断

`spy/compiler/hir.py`：

- `InlineMode`（`NONE`/`NON_AGGREGATE`/`FULL`，L92-110）→ **删除**。
- `Alloca.inline: InlineMode` → **修改**为 `Alloca.is_comptime: bool | None`（三态）。

`spy/compiler/astgen.py`：

- 所有构造 `hir.Alloca(...)` 的点 → **修改**（逐个重判，非机械替换）。**统一策略（已定）**：
  `Comptime`/`ConstComptime` 声明 → `True`；**带类型注解（非 `Comptime`）的普通声明** → `False`
  （运行时存储，之后可写运行时值）；**无注解的声明、解构目标、表达式临时量、内部槽** → `None`（按写入推断）。
  - `_predeclare_name(name, hir.InlineMode.FULL)`（def 名，L381）→ `is_comptime=None`
    （**不是 `True`**：新模型下闭包是普通 struct 值，捕获运行时变量时含运行时字段、是运行时值，
    `True` 会触发 `cannot store a runtime value…`）。
  - `_predeclare_target`（无注解赋值，L391）→ `None`；`_predeclare_ann`（L433）→ `True`/`False`
    （按声明是否 `Comptime`）。
  - `_gen_stmt` 的 bare `raise` 临时槽（`NON_AGGREGATE`，L504）→ `is_comptime=None`（推断）。
  - `_gen_for`：编译期 `for` 的迭代器/循环变量槽（`FULL`，L707-714、L724-727）→ `True`；
    非编译期 `for` 的槽（`NON_AGGREGATE`/`NONE`）→ `None`。
  - `_gen_lhs` 的 `inline_mode` 参数 → 改为 `is_comptime: bool | None`（默认 `None`）。
  - `_gen_assign`（L925）→ `True if by_marker else None`（无注解声明按写入推断，与解构目标一致）；
    `_gen_ann_assign`（L991）→ `True`/`False`（按注解是否 `Comptime`）；
    `_gen_target_tuple`（L1027）→ 默认 `None`（与 `_gen_lhs` 一致；解构目标与单目标同策略）。
  - `_gen_expr` 的表达式临时槽（`NON_AGGREGATE`，L1341）→ `None`。
  - **`_as_ref`（L1158，`NON_AGGREGATE`）→ `None`**（此点原清单漏列）。

`spy/compiler/interp.py`：

- `PendingSlot.inline_mode: InlineMode` → **修改**为 `PendingSlot.is_comptime: bool | None`。
- `PendingSlot.is_inline(type)`（L464）→ **修改**：改为按 `is_comptime` 三态 + store 判定：
  - `True` → 编译期存储；
  - `False` → 运行时存储（内存）；
  - `None` → **推断**：`_record_pending_store` 提交时，**当且仅当已有写入全部是编译期值**时为编译期。
  - 删除 `NON_AGGREGATE` 的「聚合例外」逻辑（新模型下聚合要么整值编译期要么整值运行时）。
- `alloca(inline=..., declared=..., commit_declared=...)`（L4756）→ **修改**：签名与内部判定。
- `_record_pending_store`（L7753）→ **修改**：`is_inline` 的判据换成新三态 + `_is_comptime_val`。
- `_commit_pending_slot`（L7768）→ **修改**：分派到 box / `sval` 聚合值；删除对
  `ComptimeAggregatePtr` / `ComptimeTuplePtr` / `ComptimeDictPtr` / `ComptimeOptionPtr` /
  `ComptimeTaggedUnionPtr` 的物化分支。
- `_inline_aggregate_of` / `_inline_tuple_of` / `init_comptime_tuple` / `init_comptime_dict` /
  `init_comptime_option` / `init_comptime_tagged_union` / `init_inline_aggregate` / `_fresh_place` /
  `_bind_slot` → **修改或删除**（见 §5、§7）。

- 行为保留：`x = 1`（未注解整数字面量）仍报「无落地类型」错误（现由 `_no_runtime_type`/未定型整数路径给出）。

### 3.4 `ConstComptime`

`spy/compiler/syntax.py`：

- **新增** `type ConstComptime[T = Any] = T`（与 `Comptime` 平行）。
- **新增** `def const_comptime(): ...`（与 `comptime()` 平行）。

`spy/compiler/sval.py`：

- **新增** `unwrap_const_comptime(annotation)`（与 `unwrap_comptime` 平行）或在 `unwrap_comptime` 上扩展返回值
  （三态：`Comptime` / `ConstComptime` / 无）。

`spy/compiler/astgen.py`：

- `_split_comptime(node)`（L999）→ **修改**：额外识别 `ConstComptime`，返回 `(is_comptime, is_const, type_node)`
  或引入新 `_split_comptime_kind`。
- `_gen_ann_assign`（L948）→ **修改**：`ConstComptime` **必须带初值**（`node.value is None` → 编译错误）；
  `Alloca` 增加 `is_const` 信息（或另设 `hir.Alloca.is_const_comptime`）。
- `_gen_assign` / `_gen_syntax_call`：`syntax.const_comptime()` 标记处理（紧跟新声明、不能标记赋值，错误条件同 `comptime()`）。
- `_as_marker`（L311）/ `_Pragma` 家族：新增 `_ConstComptime` 标记类。

`spy/compiler/interp.py`：

- `Box` 提交后 `is_const=True`；此后写它是编译错误（`store` 对 `is_const` box 报错，已有一部分逻辑在 `ComptimeBox.is_const`，迁移即可）。

---

## 4. 编译期指针（`sval.ComptimePtr`）与 `ref`

### 4.1 取址与偏移

`spy/compiler/interp.py`：

- `ref`（`astgen` 的 `syntax.ref`）路径：对**编译期 place** 产生 `sval.ComptimePtr{box, offset}`：
  - 变量本身 `offset == 0`；第 `i` 个字段/元素 `offset == parent.offset + 该字段/元素在铺平序列里的偏移`
    （铺平顺序见 §0.2；option/union 的 payload 从 `parent.offset + 1` 起）。
- `field_index_addr`（L4603）/ `exec_field_name_addr`（L4557）→ **修改**：对编译期指针**加偏移**，
  不物化中间值；偏移用 §0.2 新增的 `sval.field_scalar_offset(t, i)`（宽度用 `sval.scalar_count(t)`）。
  对 `is_ref=True` 字段：字段的 cell 存一个 `ComptimePtr`，需**额外 load 一次**得到 pointee 地址，
  后续读/写作用于 pointee（可能跨到另一 box，见 §2.4/§0.2）。
- `subscript`（L8252）→ **修改**：数组/元组/字典的编译期元素寻址同样加偏移。
- `p[...]` 解引用：对 `ComptimePtr` 按 `offset` 读/写 `box.cells`；只读指针或只读 box 不可写（见 §4.2）。
- `exec_ptr_cast`（L5226）→ **修改**：编译期指针的结果统一为 **`sval.CastedPtr`**；`CastedPtr` 不可读写透
  （与现 `ComptimeCastedPtr` 语义一致，只是换了类型与归属域）。
- **`ptr_cast(字节串)`**（`std.core.sstr` 依赖）：字节串编译期值经 `ptr_cast` 得到的是一个
  **`box` 为该 `bytes` 值的 `ComptimePtr`**（而非 `sval.CastedPtr`），由自动 lower 生成
  `mir.GlobalStringValue`（`ConstPtr[u8]`，末尾补 NUL）；落在 §4.3 的自动 lower 入口
  （与 §2.2（续）一致）。

### 4.2 全局性与别名

- 编译期指针相等由 **`box` + `offset` + 视图类型** 经 `ComptimePtr` 的**结构化
  `@dataclass(frozen=True)` eq/hash** 自动给出（`box` 为 `Box` 时是 `util.IdentityObj`，身份相等/哈希；
  `box` 为 `bytes` 时按**内容**相等——同内容的字节串共用同一特化/全局。box 可变仅影响 `cells` 内容，
  不影响其身份）。
- `fn.ComptimePtrArg`（L161，fn.py）→ **删除**：直接以 `sval.ComptimePtr` 作为 `ArgNode` 叶子
  （`AnyValue` 已含它）；相等/特化键即 `ComptimePtr` 本身（见上）。
- **`Box.name_base` 只是 MIR 里的命名提示（原 §17-Q16，已定）**：不参与特化键（键 = `Box` 身份 +
  `offset` + 视图类型，即上面的 `ComptimePtr` 结构化相等），**不要求唯一/稳定、无链接影响**；若某处拿它
  当 MIR 符号名，同名冲突由 MIR 层消歧即可。
- 跨非内联边界：**只允许 const 指针**（`is_const=True` 的 box）。**作为编译期实参/返回值/捕获时按身份
  传递**（进特化键，callee/闭包**别名同一 box**，不 lower）；**只有被物化成运行时值/地址**时
  （需要运行时表示，如非内联实参 / 非内联返回 / **存进运行时 storage**）才处理：非 const → 报错，
  const → 自动 lower（见 §16）。**闭包捕获不算物化**（指针烘进 struct 类型），不受此限（见 §9.3）。
- **跨 context（已定）**：设计上保证一个 context 的编译期值**不会泄漏**到另一个 context
  （`Box`/`ComptimePtr`/`FunctionValue` 都是本 context 的身份对象）；跨 context 传递的只有可重新
  lower 的编译期值。
- **const（已定）**：指针可带**独立的只读视图**（`PointerType.is_const`），与 `Box.is_const` 分开。
  **写透要求「指针视图」与「box」都可写**；任一为只读就报错。可变 `Ptr` 可隐式转 const。
- 特化键：`ComptimePtr` 本身（同一 box + 同一 `offset` + 同一视图类型）。视图类型（含 `is_const`）
  参与相等，故同址的 const / 非 const 视图是不同实参。
- 编译期返回值经 `RetSpec` 的「编译时返回值」叶子交付（见 §7.3）。
- **可变 box 与特化缓存（原 §17-Q17，已定；属「用户需避免」的使用约束）**：同一 `ComptimePtr{box, offset}`
  在写前/写后命中同一特化（键取 box 身份、不随值变化），故**用户不得依赖「可变 comptime 值决定特化」**——
  不做检测/失效。风险面本就窄：非 const box **不跨非内联边界**（上一条），跨边界的只有 const box（值不变）；
  仅 §9.3 的闭包编译顺序依赖需用户留意。

### 4.3 自动 lower（替代 `as_static_ptr`）

`spy/compiler/interp.py`：

- **删除** `_as_static_ptr`（L7075）及 `_call_builtin` 中 `as_static_ptr` 的分派。
- 保留并复用：`_const_to_mir`、`_const_runtime_to_mir`、`_const_pointer_to_mir`、`_const_struct_to_mir`、
  `_const_array_to_mir`、`_const_option_to_mir`、`_absent_path_constant`、`_const_tagged_union_to_mir`
  （作为「自动 lower 引擎」）。**新增**入口（已定）：给定一个 `is_const=True` 的 box，**先 `load` 成
  完全编译时已知的 `sval.Value`**（聚合快照），再对该值跑 `_const_to_mir` 家族，自动产生
  `mir.GlobalConstant`（指针类型 `ConstPtr[T]`）；给定 `fn.FunctionValue` 走函数符号（见下）。
  **不直接读 `box.cells`**——「box 铺平布局 ≠ MIR 布局」的差异由「值 → MIR」消化
  （`is_comptime` 字段被跳过、`is_ref` 字段按指针写出）。
- `fn.FunctionValue.get_type()`（fn.py L917）→ **修改**：返回**常函数指针**类型 `PointerType(fn_type, is_const=True)`
  （`fn_type = signature.as_non_generic_fn_type() or AnyFunction()`）；与 `sval.DeclareFunction.get_type`（已是 `PointerType(type, True)`）一致。
- 非 const box（普通 `Comptime` 变量）**不支持 lower**：其 `ref` 进不了运行时，只在编译期/内联范围使用。
- **`mir`/`lower` 侧：函数值的自动 lower（原 §17-Q5 遗留，已定）→ 直接是函数符号本身，不经 `GlobalConstant`**：
  - **同模块**的 spy 函数 / 闭包 `__call__` → `mir.Function`（`FunctionValue` 取其实例化后的
    `FunctionInstance.mir`）；`mir.Function` 是 `GlobalValue`，其 `get_type()` =
    `PointerType(FunctionType(args, ret), is_const=True)`（常函数指针），与 `sval.FunctionValue.get_type()`
    一致，可直接作操作数（in-module 符号）或 `mir.GlobalConstant` 初始化器成员（如 vtable 的函数指针）。
  - **早先模块**里编译的函数 → `mir.ExternAnonSymbol`（link name 由 `fn.SymbolTable` 解析）。
  - **`@decl_func` 外部函数**（`sval.DeclareFunction`）→ `mir.ExternSymbol`（现状不变）。
  - 前提：该 `FunctionValue` 的 `FunctionInstance` 已编译（`mir.Function.get_type()` 断言 `is_complete`）。
    未被调用的**非 generic** 函数值可经 **`std.core.compile(fn)`**（见 §10）显式请求编译。

**删除 `ConstRef`：全局是「值」，函数值即常函数指针（Q5，已定）**

`hir.ConstRef`（`hir.py:137`）与 `sval.ConstRef`（`sval.py:1243`）**删除**：前者的「指向全局的引用」与新的
`ComptimePtr` 重复，后者的「指向编译期值的常指针」也由 `ComptimePtr` 承担。做法：

- **astgen**：`_resolve_global(name)` → 返回 **`hir.Const(obj)`**（值）；`_try_resolve_object` 解包 `.value`。
  `_gen_name(name)` → 返回 **`ArgEntry[hir.Value]`**：本地 binding `ArgEntry(slot, True)`，全局 `ArgEntry(<hir.Const>, False)`。
  于是**全局在 HIR 里是值（`is_ref=False`）**，`ArgEntry.is_ref` 只对 binding 的 slot 为真。
  `_gen_expr` Name 直接用该 `ArgEntry`；`is_struct = (not is_ref) and isinstance(value, hir.Const) and _is_struct_class(value.value)`。
- **`hir.Subscript.base` / `hir.Call*.callee` 仍是 place**：需要 place 时由 `_as_ref` 把全局**值** alloca 进一个
  comptime box（`Alloca`+`Store`+`CommitSlot`），再当 place 用。因此 `Foo[T]`（非 syntax marker 的模板特化）
  会「box → subscript → `subscript` 解包 box」绕一圈（语义仍清晰：subscript/callee 恒为 place）。
  两条静态快路径（`Literal[X]`、`_gen_type_ctor` 处理 `Ptr[T]`/`Array[T,N]`/`Option[T]`…）**保留**，不走 box。
- **astgen 具体落点**：`_make_load` 删 `ConstRef` 分支；`Attribute` 的全局折叠**移到 `_as_ref` 之前**；
  `Subscript` 通用路径记录 `is_global` 并 box base、去掉原 “subscript of non-reference” 报错；
  `_gen_call` 结构不变（callee 经 `_as_ref`）；`_gen_struct_ctor(callee.value, …)` 收到的是 `hir.Const`
  （`hir.FinishStruct.struct` 的语义/文档由「引用」改为「值」）。
- **interp 具体落点**：`_callee_object` 的 `ComptimeVal(obj) → obj`（不再解 `sval.ConstRef`），并**新增解包
  `ComptimePtr`**（box → 值）；`operand` 删 `case hir.ConstRef()`；`load` 删 `sval.ConstRef` 分支；
  `subscript` 先 `load` 出 base 再判 `sval.StructTypeHead`；去掉 `_declared_type`/`_class_name_base`/
  `_call_class_method`/`call_method`/`_val_to_node` 中对 `sval.ConstRef` 的处理。
- **「编译期常引用」一律用 `ComptimePtr`**（指向一个 `ConstComptime`/`is_const=True` 的 box，视图类型
  `PointerType(elem, is_const=True)`）：原先 `sval.ConstRef` 的两处构造改由它承担——
  - `_subscript_bytes`（`b[i]`/`b[a:c]`，interp L8434）→ `ComptimePtr`（`box` = 该子串 `bytes`）；
  - `_init_arg_node` 的编译期叶子（interp L2066）→ **仍包一层指针**：`ComptimePtr`（指向持有该值的
    const box），因为 `hir.Arg` 都是参数的 **place**，不能是裸值。
- **`_generate_call_body`（interp L6954）修正**：原来直接把 `hir.Const(closure)`（值）传给 `hir.CallInplace`
  作 callee（只是碰巧能跑），改为先 `Alloca`/`Store(closure)`/`CommitSlot` 再把该 place 作 callee，与
  `_gen_call` 一致。

---

## 5. 编译期聚合（struct / array / tuple / option / union / complex）

### 5.1 存储与读写

- 完全编译时已知的聚合 → 在 `Box` 里**铺平**为标量序列（§0.2），字段/元素读写全部在编译期折叠；
  `load` 出的整值快照是 **`sval.AggregateValue`**（不再充当可变存储）。
- `store`/`load`/`field_index_addr`/`_extract_aggregate_value` → **修改**：编译期聚合的字段/元素读写全部在编译期折叠，
  用 `offset` 加偏移寻址。
- `_extract_aggregate_value`（L4060）/ `as_comptime_aggregate`（L4098）/ `_split_runtime_aggregate`（L4125）→
  **修改或删除**：新模型下聚合值统一为 `Box` 铺平存储（编译期）或 `RuntimeVal`（运行时），
  不再有 `ComptimeAggregate`/`ComptimeAggregatePtr` 两个中间形式（见 §7）。
- 整体拷贝按值：`b: Comptime = a` → 各自新 box（`_coerce`/`store` 中保证）。

### 5.2 与运行时聚合的关系

- `store`/`_coerce`/`_convert_inst` → **修改**：若聚合任一成员是运行时值，则**整体是运行时值**；
  写进 `Comptime` 变量报错（`cannot store a runtime value into a compile-time variable`）。
- **删除**「把运行时聚合拆分进 inline `Comptime` 变量、每字段一次 store」的路径：
  `_split_runtime_aggregate`、`store` 中 `InlineMode.FULL + _is_aggregate` 的整段
  （现 `store` L3848-3889）。
- 字面量聚合传给原生函数按值参数仍落内存取址——保留。
- ZST 聚合：单位值、无存储——保留。

### 5.3 Option / tagged union

- **option / tagged union 按聚合体处理**（§0.2）：整体完全已知时铺平为 `[tag, payload...]`
  （tag 是第 0 个标量，payload 从 `offset + 1` 起）；读写/解包全在编译期。
- **编译期值形态（原 §17-Q19，已定）**：present option 是**新增** `sval.OptionValue`——
  `class OptionValue(Value): value: AnyValue`，`get_type()` = `OptionType(type_of(value))`；absent 仍是
  `sval.Null`（无类型）/`sval.TypedNull`（有类型）。`OptionValue` 是**值形态**（同 `AggregateValue`/
  `TaggedUnionValue`），**不进 `ScalarValue`**（`TypedNull` 亦已从 `ScalarValue` 移出）。tagged union 统一用
  **`sval.TaggedUnionValue`**（删除 `interp.ComptimeOption`/`ComptimeTaggedUnionValue`）。
- `sval.coerce_const` 的 `OptionType` 分支（sval L3629）→ **修改**：absent（`Null`/`TypedNull`）原样；
  present 不再直接返回 child 值，而是包成 **`OptionValue(coerce_const(value, child))`**。
- 同步更新（把 `ComptimeOption` 换成 `OptionValue`/`TypedNull`）：`interp._is_null`（L8081，absent 判定
  要同时认 `Null` 与 `TypedNull`）、`_const_option_to_mir`（L7208）、`slice_object`（L8494）。
- tag 运行时才知的 → **普通运行时值**（`OptionType.to_mir_type` / union 存储 + 指针重解释）。
- **删除** `ComptimeOption`/`ComptimeOptionPtr`/`ComptimeTaggedUnionValue`/`ComptimeTaggedUnionPtr`
  及其在 `store`/`load`/`_type_of`/`_val_to_node`/`_init_ptr_target`/`_init_container_target` 里的分支；
  由运行时表示或铺平存储/`sval.OptionValue`/`sval.TaggedUnionValue` 取代。

---

## 6. 编译期控制流

`spy/compiler/interp.py`：

- 编译期 `if`（`_exec_if`，L3056）→ **修改**：条件必须是 §1 的新编译期值。
- `syntax.unroll()` 循环（`_exec_loop`/`_unroll_inline_loop`/`_exec_break_loop`/`_cut`，L3092-3671）→ **修改**：
  迭代器/循环变量必须是完全编译期值（`std.range` 元素靠 `Comptime[T]` 局部量保持编译期，行为不变）。
- `_runtime_position()`（L7731）→ **修改**：与 `PendingSlot.is_comptime` 推断配合，修掉 `issues.md` #5 的
  「带类型的 `comptime()` 局部量落成内存」缺口（新模型下 `Comptime` 变量恒为编译期存储、不再降级）。
  运行时控制流内的编译期写入**本次不做检查**（见 §17）。

---

## 7. 元组（tuple）的完整行为

### 7.1 作为运行时值

- `TupleType` 获得运行时表示（§2.1）。
- `hir.Tuple`（值形式，hir.py L493）/ `hir.InitTuple` / `hir.TuplePtrElement` → **修改**：
  产出一个运行时聚合值 / 就地初始化位置字段。
- `interp.init_tuple`（L8922）/ `tuple_ptr_element`（L8959）→ **修改**：从「tuple of places」改为
  在目的地就地初始化元组的元素字段（要么运行时聚合的 `field_index_addr`，要么编译期聚合的 `offset`）。
- `interp.subscript` 的 `_subscript_comptime_tuple`（L8374）→ **修改/删除**：元组下标改为普通聚合元素寻址
  （`t[i]` 是元素左值，`i` 是编译期整数，越界/负/运行时下标报错）。
- `_tuple_places_type`（L1456）/ `ComptimeTuple` / `ComptimeTuplePtr` → **删除**；`_type_of` 中元组分支改写。
- `len(t)`（`exec_len` L8208）→ **修改**：通过新的元组类型长度。

### 7.2 解构赋值（RLS + `ResultLocTree`）

**RLS（结果位置语义）**：一批「写穿」指令把结果**直接写进目的地**（`FullResultLoc`），不产生值寄存器；
其余表达式先算出值、再 `hir.Store`。写穿的源形式由 `astgen._gen_result_loc` 分派（见下）。

`spy/compiler/hir.py`：

- **新增**（`hir.py`）：`type ResultLocTree = hir.Value | tuple[ResultLocTree, ...]` 与
  `type FullResultLoc = ResultLocTree | ReturnLoc`；**`ReturnLoc` 是新的独立节点**（**不是** `hir.Value`），
  语义 =「外层函数的**整棵**结果位置树」，**取代** 旧的 `hir.ResultLoc`（`Value`，删除）。
  于是「结果位置树」与「值」彻底分开：`ResultLocTree` 的叶子只含 `hir.Value`（place）。
- **修改**：目的地字段从 `Value` 改为 **`FullResultLoc`**——`hir.Binary.ret`、`hir.Unary.ret`、`hir.Ord.ret`、
  `hir.CallInplace.ret`、`hir.CallMethodInplace.ret`、`hir.InitTuple.result_loc`、`hir.FinishStruct`/`FinishArray`
  的目的地、`hir.TuplePtrElement.tuple_ptr`（`return (a, b)` 需要 `TuplePtrElement(ReturnLoc(), i)`）、
  `astgen._gen_result_loc` 的参数。
- **修改**：`hir.Store.ptr` 由 `hir.Value` 改为 **`hir.Value | ReturnLoc`**——`return <非可拆解表达式>`
  （名字 / 值形式 tuple）经 `_gen_result_loc` 的默认分支走的就是 `Store(result_loc, value)`，`result_loc` 可为
  `ReturnLoc()`。语义：`value` 是单值 → 把结果位置定型为单 place（或形状校验后 store）；`value` 是值形式
  tuple（N 元）→ 定型为 N 叶树（已定型则校验）并**逐元素 splat** 进各叶 place（`ret = a, b; return ret`）。
- **删除** `hir.TuplePtr`（L501）。
- **不新增** `hir.DestructureStore`。
- 不变：`FinishStruct`/`FinishArray`、`InitTuple`/`TuplePtrElement`（只作用于叶子）、
  `StoreVoidRetloc`、产生地址的指令（`FieldAddr`/`FieldIndexAddr`/`Subscript`/`Alloca`/`Arg` 等）。

`spy/compiler/astgen.py`：

- `_gen_target_tuple`（L1027）→ **修改**：返回 Python `tuple[hir.Value, ...]`（`ResultLocTree` 结点），
  不再 `self.add(hir.TuplePtr(...))`。
- `_gen_lhs`（L865）→ **修改**：返回值类型 `ResultLocTree`。
- `_gen_result_loc`（L1591）→ **修改**：参数 `result_loc: FullResultLoc`（`return expr` / lambda 体传 `ReturnLoc()`）。
  按 `expr` 分派——前六类是**写穿**（直接把结果写进 `result_loc`，不产生值寄存器）：
  - 两边都是 tuple（`a, b = c, d`）→ astgen **拆分**为逐元素赋值。
  - `ast.Call()`（非 `syntax.*`）→ `_gen_call`（callee 分派见下）；
  - `ast.UnaryOp()`（非 `not`）→ **`hir.Unary`**；`ast.BinOp()` → **`hir.Binary`**；
    `ast.BoolOp()` → `_gen_boolop`（`hir.Block` + `Store`/`break_if`）；
  - `ast.Tuple()` → **`hir.InitTuple`** + 逐元素 `hir.TuplePtrElement`（递归；base 可为 `ReturnLoc`）；
  - `ast.IfExp()` → 条件 + `hir.If`/`hir.Else`/`hir.End`（两臂递归，同一 `result_loc`）；
  - 其余 → 默认分支 `hir.Store(result_loc, value)`：`result_loc` 是**具体树**时 astgen 展开为 `target[i] = expr[i]`；
    是 `ReturnLoc()` 时单个 `Store(ReturnLoc(), value)`，由 interp 定型/splat。
  - RHS 是结构体/数组构造子 → 报错。
- `_gen_boolop`（L1644）→ **修改**：`result_loc` 参数 → `FullResultLoc`（`return a and b` 会把 `ReturnLoc()` 传进来，
  内部走 `Store`/`break_if`）。
- `_gen_call`（L1724）→ **修改**：`result_loc: FullResultLoc`；按 callee 分派：
  - `cast(T, v)` → 无（递归 `v`）；`syntax.array(...)` → `_gen_array_ctor`（`hir.FinishArray`）；
    `isinstance(v, T)` → `_gen_isinstance`；`ord(x)` → **`hir.Ord`**；`len(x)` → `hir.Len` + `Store`；
  - `node.func` 是 `ast.Attribute`（`x.h(...)`）→ **`hir.CallMethodInplace`**；
  - callee 是 struct 构造 → `_gen_struct_ctor`（`hir.FinishStruct`）；
  - 其它 → **`hir.CallInplace`**。
- `_gen_struct_ctor` / `_gen_array_ctor` 的 `result_loc: FullResultLoc` → **修改**（`FinishStruct`/`FinishArray`）。
- `_gen_augassign`（L1054）→ **修改**：元组 `+=` 不再支持（见 §7.5）。

`spy/compiler/interp.py`：

- `_deliver_result`（L9758）/ `_deliver_packed_tuple`（L9881）/ `_pair_places`（L9895）/ `_fresh_places`（L9936）→
  **修改**：result location 支持「树」——`Call*Inplace.ret` 为树时逐叶交付。
- **`ComptimeResultPtr.value` 由单个 place 改为 `ResultLocTree | None`**（`code`/`payload` 字段保留，
  `ComptimeResultPtr` 本身保留）。即 `InlineFrame.ret_loc` 仍是 `ComptimeResultPtr`，但其 `.value` 现在是树：
  `_start_inline`（L10044）把该调用的结果树放进 `ret_loc.value`；`_reserve_result_loc` / `_current_result_loc` /
  `_result_places`（L1477）/ `_function_result` 相应改为吃/产树。
- **结果位置的定型（原 §17-Q20，已定）**：函数结果位置（`ComptimeResultPtr.value`）：
  - 返回注解**已声明** → 由注解 `RetSpec` 给出的**静态树**；
  - 返回注解**省略** → 初始 **`None`（未定型）**，在**第一次投递**时按**被投递物自身的形状**定型并分配 place：
    `InitTuple(ReturnLoc(), n)` → `n` 叶；`return f()` → `f` 的 `RetSpec` 值叶；单值 → 单叶。
  定型后即函数的结果树、此后静态；后续 return 形状不符 → 编译错误；`_finish_function` 用各叶类型产
  `ReturnSignature`（未投递过 → `EmptyType`，同今天）。
- `store(ptr, value)`（L3800）→ **修改**：`ptr` 为 `ReturnLoc`（→ 外层函数结果位置）时按 `value` 的形状定型
  （单值 → 单 place；值形式 tuple → N 叶树并逐元素 splat）；否则按普通 place 处理。
- **删除** `_result_place_tree`（L7857）与「记录型 FULL 结果槽」：形状不再来自 recorded stores，而来自被投递物
  （callee spec / `InitTuple` 的 arity）。
- `init_tuple`/`tuple_ptr_element` 见 §7.1。

### 7.3 多返回值

`spy/compiler/sval.py`：

- **新增 `RetComptime` 叶子**：`RetComptime{type: Type, index: int}`（**不带值、不带 `via_result_ptr`**），
  `RetSpec = RetValue | RetTuple | RetComptime`。判定用 `classify()`：注解展开出的叶子类型
  `classify() == COMPTIME` → `RetComptime`（其余按现有 `RetValue`/`RetTuple`）。它在交付约定上等同
  **via-result-ptr**：**不占 by-value 槽、不生成 MIR result 实参**。
- **编译期返回值的值不在 spec 里**（`RetSpec` 仍 frozen）：另存在 **callee `FunctionInstance` 上的一个 `list`**，
  `index` 是按 `RetComptime` 叶子深序编号的下标；每次编译该 callee specialization 时填充。
  **填充规则（已定）**：每条被 `interp` 分析的 `return` 把自己的编译期值写入对应槽，**后写覆盖先写**，
  故最终值 = **最后一条被分析的 `return` 所返回的值**；**不检查**各 `return` 路径是否一致（见 §17）。
- `make_ret_spec` / `result_leaves` → **修改**：按 `classify()` 产出 `RetComptime`；by-value 选择只在
  `RetValue` 叶子中挑，`RetComptime` 跳过。
- `iter_ret_leaves` → **仍产出 `RetComptime`**（保持「一个 spec 叶 ↔ 一个预留 place」的对齐，供
  `_deliver_result` 的 `zip` 用）；**所有面向 MIR 的使用点显式过滤 `RetValue`**：`ret_returned_type`、
  `FunctionType.to_mir_type`（sval L2144）、`fn._storage_leaves`（L1049）/ `glue._storage_leaves`（L705）、
  `fn.Signature.specialize`（L752，非默认 callconv 的不可拷贝检查）。`ret_by_value_index` **遍历计数含它**
  （保下标与 `_result_places`/`_pair_places` 一致），但**选中条件必须是 `RetValue`**（`RetComptime` 无
  `via_result_ptr`，否则 AttributeError）。
- `ret_spec_value_is_empty` 不受影响（只看根 `RetValue`）。`ReturnSignature.ret_spec`（fn.py L230）/
  `FunctionType.ret_spec`（sval L2099）→ **传导**。

`spy/compiler/interp.py`：

- **两条取回路径**：
  - **inline 调用**：把该调用的结果树（`ResultLocTree`；若 `ret` 是 `ReturnLoc()` 则解析/定型为外层的）
    直接作为 `InlineFrame.ret_loc.value` 传入；body 在编译期把返回值写进树的对应叶子（编译期 place）。**不经 list**。
  - **非 inline 调用**：callee 编译成独立 MIR 函数，编译期返回值经 `FunctionInstance` 上的 list 带出
    （值 = 最后一条被分析的 `return`，见上）；调用点按 `index` 取出，`store` 到 `_pair_places` 预留的对应 place。**place 不一定是编译时的**——
    把编译时值存进运行时 place 合法；只有返回类型的 `classify() == COMPTIME` 时 place 才必须是编译时
    （否则 alloca 处报错）。目前只对 `classify()==COMPTIME` 走 `RetComptime`，故现况必然是编译时 place；
    将来会支持手动指定把 `classify()==NONE` 的类型也按编译时返回。
- `_ret_spec_place`（L2195）/ `_commit_ret_places`（L2263）→ **修改**：为 `RetComptime` 叶预留一个
  （编译期）place 并提交。
- `_deliver_result`（L9758）/ `_pair_places`（L9895）→ **修改**：`RetComptime` 叶不成 MIR 实参，
  改为「取 list 值 → `store` 到对应 place」。
- `_materialize_ret_sig`（L2242）/ `_make_runtime_call` → **修改**（报错文案 / 传导）。
- **打包成一个变量**：**当所有结果叶 `classify() != COMPTIME` 时**，`packed = f()`（普通元组变量）——
  `CommitSlot` 处由调用结果推断元组类型并打包。**若任一结果叶 `classify() == COMPTIME`**（类型值 /
  const 指针 / 字节串 / 函数值等），打包出的元组类型无运行时表示，目标变量必须是 `Comptime`。
  把「删除『只能进编译期变量』约束」改为**仅对全运行时结果删除**：`_no_runtime_type` 里针对
  `TupleType` 的提示文案改为**只对运行时元组删除、保留含编译时叶的提示**。
- `tuple[T, ...]`（变长）在任意嵌套层级报错——保留。

### 7.4 元组作参数 / 返回值

- 形参 `tuple[T1, ...]`（含 `*args`）**铺平传参**——**保留**；callee 侧重组成一个**合成 `StructType`**
  （不再是 `ComptimeTuplePtr`），规则见 §8.1。
- 元素为编译期值时不占 MIR 实参——**保留**：合成 struct 里该字段 `is_comptime=True`、`default=该编译时值`
  （值烘进类型，不 store、不进 MIR 布局），见 §8.1。
- 结构体元素不铺平——保留。

### 7.5 `t += u`

`spy/compiler/interp.py`：

- `binary_assign`（L6081）中元组分支 → **删除/改为报错**（元组 `+=` 不再支持）。
- `_tuple_extend_in_place`（L6101）、`_tuple_extend_element`（L6131）→ **删除**。
- 「reduce/累积」用例改用「**先补全元素类型、再 `reify` 出 tuple 类型、最后逐元素填充**」：

  ```python
  types = const_array(typed_undefined(type), N)
  unroll()
  for i in range(len(types)):
      types[i] = ...
  tuple_type = reify(TupleType(const_arr_slice(ref(types)), False))
  ret = typed_undefined(tuple_type)
  unroll()
  for i in range(len(ret)):
      ret[i] = ...
  ```

  依赖 `std.reflect.reify` / `reflect.TupleType`、`std.core.const_array` / `const_arr_slice` /
  `typed_undefined`（文档/测试更新）。

---

## 8. `StrDict`（`dict[str, T]` / `**kwargs`）

- `StrDictType` 获得运行时表示（§2.2）。
- `**kwargs` 形参 / `dict[str, T]` 形参仍按名铺平传参——保留。
- `interp` 中 `ComptimeDict`/`ComptimeDictPtr` 及相关分支（`store`/`load`/`_type_of`/`_val_to_node`/
  `_init_ptr_target`/`_init_args_from_signature`/`_subscript_comptime_dict`）→ **删除**，
  改为普通聚合（编译期 `sval.AggregateValue` 或运行时 struct）。
- `_dict_arg_entries`（L1290）/ `_subscript_comptime_dict`（L8393）→ **修改**。
- 条目访问 `d['name']` 仍**只支持编译期字节串名**（无运行时键查找）。

### 8.1 `*args` / `**kwargs` 的合成 struct（原 §17-Q13，已定）

`*args` 的形参节点是 `tuple[ArgNode, ...]`、`**kwargs` 的是 `frozendict[str, ArgNode]`；两者（连同嵌套在
其内的 tuple/frozendict 元素）在调用点**铺平**（§7.4），callee 侧改由**合成 struct** 承载，取代被删的
`ComptimeTuplePtr`/`ComptimeDictPtr`。`interp._init_args_from_signature`（L2169）改为：

- **叶节点参数**（`RuntimeArgNode`、编译时值叶子）：处理**不变**。
- **`tuple[ArgNode, ...]` / `frozendict[str, ArgNode]` 参数**：生成对应的 **`StructType`**（字段 = 元素，
  顺序即元素序；`frozendict` 的字段名 = 条目名）并 **`alloca` 一次**，再逐元素把值/引用写入字段：
  - 元素是**编译时值** → 字段 `is_comptime=True`、`default=该编译时值`：值烘进 struct **类型**
    （类型级常量，写是编译错误，同 §9.3 `move` 捕获），**不 store、不进 MIR 布局**、不占 MIR 实参
    （落实 §7.4）。
  - 元素是 **`RuntimeArgNode`** → 字段类型 = `node.type`，**`is_ref = node.is_ref`**：`is_ref=False` 写入值，
    `is_ref=True` 写入引用（该实参本就以指针铺平，写的正是那个指针；读侧 `field_index_addr` 对其额外
    load 一次，§2.4）。
  - 元素是**另一个 `tuple`/`frozendict`** → **递归**生成嵌套 `StructType` 作为外层字段（按值嵌入，
    `is_ref=False`）；**不再单独 `alloca`**（那会多一次拷贝），而是以外层字段的位置（指针/偏移）为基址，
    在其上直接写子字段。
- 因此整个组装是 **单次 `alloca` + 按字段偏移的递归写入**。例：

  ```python
  def foo(a: i32, *args)
      b1 = args[0]

  a: i64 = 12
  foo(12, Large(...), Large(...), (a, a, Large(...), Small(...)))
  # args 的类型:
  #   struct { ref _0: Large; ref _1: Large;
  #            _2: struct { _0: i64; _1: i64; ref _2: Large; _3: Small } }
  # MIR: fn(i64, *const Large, *const Large, i64, i64, *const Large, Small)
  # 初始化（只 alloca args 一次）:
  #   args[0] = %1; args[1] = %2
  #   args[2][0] = %3; args[2][1] = %4; args[2][2] = %5; args[2][3] = %6
  ```

- `is_ref` **完全对齐 `RuntimeArgNode.is_ref`**（`is_ref=True` 只出现在实参本就以引用铺平时），
  故不存在「对寄存器实参取地址」的冲突。
- **`hir.Spread`（源码 `*`/`**` 转发）→ 保留，改为按 struct 字段展开**：`hir.Spread.value` 求值为该
  struct 的 place，`*` 按字段声明序展开为位置实参、`**` 按字段名展开为关键字实参；`eval_call_args`
  （L9161）/ `_spread_positional`（L9186）/ `_spread_kwargs`（L9196）改为读 struct 字段，去掉对
  `ComptimeTuplePtr`/`ComptimeDictPtr` 的依赖。

---

## 9. 闭包 =「带 `__call__` 的匿名 struct」

### 9.1 `hir.MakeClosure` → 匿名 struct

`spy/compiler/hir.py`：

- `hir.MakeClosure`（L442）→ **修改**：产出匿名 struct 类型（字段=捕获，`__call__`=嵌套函数体）；
  删除 `captures` 语义改为字段。
- **匿名 struct 类型的身份（已定）**：捕获烘进字段的**默认值**，故 `StructField` 的相等/哈希
  必须包含默认值（`ComptimePtr` 身份/内容、`move` 捕获的值按结构），不同捕获 ⇒ 不同类型；
  `MirLowerCache` 的键也要包含它，同时 `is_comptime` 字段不进 MIR 布局缓存。
- `hir.Closure`（L160）→ **删除**（捕获经 struct 字段读取）。
- `hir.RebuildCapture`（L466）→ **删除**。
- `hir.CallInplace.captures`（L361）→ **删除**。
- `hir.Spread`（L318）→ **修改**：`ComptimeTuplePtr`/`ComptimeDictPtr` 删除后，`*`/`**` 转发改为对应
  struct 的**字段展开**（`*` 按字段序、`**` 按字段名），见 §8.1。

`spy/compiler/fn.py`：

- `ClosureValue`（L974）→ **删除**。
- `FunctionValue.captures`（L894 参数/属性）→ **删除**。
- `CallSignature.captures`（L279）→ **删除**。
- `ClosureFunction`（L922）→ **保留**（作为 `hir.MakeClosure` 输入）。
- `ArgNode` 联合（L133）：`ComptimePtrArg` **删除**（改用 `sval.ComptimePtr` 叶子）、`CompoundArgNode` **修改**（随 §4/§5 变化）。

`spy/compiler/interp.py`：

- `exec_make_closure`（L5261）→ **修改**：构造匿名 struct 类型 + 字段（编译时捕获 `is_comptime=True`
  字段、默认值=引用或值；运行时捕获普通字段，`move` 决定指针/值），把 `__call__` 挂为方法。
- `_init_capture_values`（L2187）→ **修改/删除**。
- `_runtime_position` / `_inline_capture_values`（L9475）→ **修改**。
- `ClosureValue` 相关调用点（`_call_function_entry` L9362、`_val_to_node`、`_request_function` 等）→ **修改**。

### 9.2 `@syntax.closure` 参数

- 参数全部保留：`inline`/`exceptions`/`callconv`/`may_panic`（透传给 `__call__`）、`move`（捕获属性）。
- `astgen._parse_closure_decorators`（L1847）→ **修改**：新增 `move`；`_gen_closure`（L1934）传导。

### 9.3 捕获语义

- `move=False`：全部按引用（编译时捕获存 `ComptimePtr`；运行时捕获存指针）。
- `move=True`：全部按值（编译时捕获烘进 struct 类型 → 类型级常量，写是编译错误；运行时捕获存副本）。
- 每次捕获产生不同匿名 struct 类型（按创建点特化）。
- §4.2 的限制不适用于闭包捕获：捕获是把指针**烘进 struct 类型**（不是物化）；读 `is_comptime`
  字段的**值**在编译期折叠成常量，只有把捕获的编译时变量**地址**物化成运行时值才受 §4.2 限制
  （那是非 const 编译期指针的通用限制，与闭包无关）。
- `_build_runtime_closure`（L6816）/ `_collect_capture_fields`（L6852）/ `_build_call_function`（L6893）/
  `_generate_call_body`（L6922）/ `_fill_runtime_closure`（L6968）/ `_capture_leaf_values`（L6986）/
  `_content_leaf_values`（L7004）/ `_rebuild_capture`（L7041）→ **删除，完全重新实现**：
  按新的匿名 struct 捕获语义从头写 `exec_make_closure`，**不**沿用原 `as_runtime_closure` 的整套计划。
- `RuntimeClosurePlan`（L1800）/ `_runtime_closure_counter`（L1887）/ `_builtin_as_runtime_closure`（L6769）/
  `_runtime_closure_args`（L6801）→ **删除**。

### 9.4 一等值与旧限制取消

- 闭包可存变量/字段、传参、返回、传给运行时函数、进运行时存储——由「闭包即 struct」自动获得。
- `std.core.as_runtime_closure`（std/core.py L279，std/__init__.py 导出）→ **删除**。

### 9.5 `as_func_ptr`

`spy/compiler/interp.py`：

- `exec_as_func_ptr`（L5326）→ **修改**：`f` 必须是有 `__call__` 的 **ZST** struct；
  结果是去掉 `self` 参数的 `__call__`（`fn.FunctionValue`，`get_type` 报函数指针），由自动 lower 进入运行时。
- 普通 spy 函数 / extern 函数：直接用名字取函数指针（`FunctionValue.get_type` 报函数指针类型）。
- `syntax.as_func_ptr`（syntax.py L167）→ **修改**：语义调整（ZST 闭包专用）。

---

## 10. 反射与 std 影响

`spy/std/reflect.py`：

- `StructField` → **新增** `is_comptime`/`is_ref`（报告的 `type` 是**逻辑类型 `T`**，不是 `*const T`）。
- 变体 `TupleType`/`StrDictType` 描述有运行时表示的类型——结构不变，语义变化（文档更新）。
- `interp._build_type_info`/`_build_struct_field`/`_build_dict_entry`/`_reify_struct`/`_reify_strdict` → **修改**。

`spy/std/core.py`：

- **删除** `gstr`（L257）。
- **删除** `as_static_ptr`（L265）。
- **删除** `as_runtime_closure`（L279）。
- `sstr`（L262）→ **修改**为库函数：
  ```python
  def sstr(s: bytes) -> ConstSlicePtr[u8]:
      return ConstSlicePtr(ptr_cast(s, ConstMultiPtr[u8]), len(s))
  ```
  （已确认：`ptr_cast` 对字节串编译期值可用；`len(s)` 为字节数，不含 NUL；`s: bytes` 按 §3.2
  因 `bytes` 无运行时表示而**自动成为编译期形参**）
- 字节串由其 `ComptimePtr`（`box` 为该 `bytes` 值）lower 成全局常量（末尾补 NUL）：
  `interp._string_global`（L6329）保留，删除 `_builtin_gstr`（L6345）/ `_builtin_sstr`（L6352）
  与 `_call_builtin` 中分派；`ptr_cast(字节串)` 即走这条（得到 `ConstPtr[u8]`，见 §4.1/§4.3）。
- **新增** `compile[T: Callable](fn: T)`（`@builtin_func def compile[T: Callable](fn: T)`）：**手动编译**一个
  **非 generic** 的函数值——从其声明构造默认 `CallSignature` 并请求编译（`_request_function`），使其可被
  自动 lower 成 `mir.Function`（见 §4.3）。generic 函数不在此列（需显式类型实参）。interp 侧新增 builtin 处理。

`spy/std/__init__.py`：

- 更新导出：删 `as_static_ptr`/`as_runtime_closure`/`gstr`；新增 `compile`。

`spy/std/mem.py`：

- `DynamicAllocator.create`（L93）→ **修改**：`as_static_ptr(vtable)` 换成
  `const_value: ConstComptime = vtable; ref(const_value)`（自动 lower）。

`spy/std/openmp.py`：

- `_ident`（L112）→ **修改**：`as_static_ptr(ident)` → `ConstComptime` + `ref`。
- `parallel_loop`（L170）→ **修改**：`as_runtime_closure(kernel)` 删除；`kernel` 直接作为 struct 闭包。
- 相关 `gstr`（L112）→ 字节串直接 lower。

`spy/std/int.py`：

- `try_int_cast`（L38）等处 `comptime()` + `Comptime` 变量——按新三态复核（大多不变）。

`spy/compiler/interp.py`：

- `_layout_builtin`/`_layout_of`/`_sizeof_value`/`_alignof_value`/`_resolve_layout_type`（L7307-7390）→ **修改**：
  接受元组/字典类型（有运行时布局）；对无运行时表示的类型报错。

---

## 11. 删除项与迁移对照（汇总）

| 删除/变化 | 新做法 | 落点 |
|---|---|---|
| `ComptimeTuple`/`ComptimeTuplePtr` | 元组有运行时表示；编译期元组存 `Box` 的 `sval.AggregateValue` | interp §7 |
| `ComptimeDict`/`ComptimeDictPtr` | 字典有运行时表示 | interp §8 |
| `ComptimeAggregate`/`ComptimeAggregatePtr` | `Box` 铺平标量存储，按 `offset` 寻址 | interp §5 |
| `ComptimeOption`/`ComptimeOptionPtr` | 完全编译期=聚合；tag 运行时=普通运行时值 | interp §5.3 |
| `ComptimeTaggedUnionValue`/`ComptimeTaggedUnionPtr` | 同上 | interp §5.3 |
| `ComptimeBox` | `sval.Box`（+ `ComptimePtr`） | interp/sval §3 |
| `ComptimeCastedPtr` | `sval.CastedPtr` | interp/sval §4.1 |
| `interp._is_aggregate` | `sval.AggregateType`（类型层次判定） | interp/sval §0.2/§2.0 |
| `sval.Undefined` | `sval.UndefinedScalar`（+ 新增 `UndefinedAggregate`） | sval §0.2 |
| 函数值的自动 lower | `mir.Function`（同模块）/ `ExternAnonSymbol`（早先模块）/ `ExternSymbol`（extern），不经 `GlobalConstant` | interp §4.3 |
| `TupleType`/`StrDictType` 的镜像 | 合成等价 `StructType`（缓存于 `MirLowerCache`）后 lower，复用 `StructType._calculate_mir` | sval §2.0/§2.1/§2.2 |
| `PendingSlot.inline_mode` | `PendingSlot.is_comptime: bool\|None` | hir/interp §3.3 |
| `InlineMode` | `bool\|None` | hir §3.3 |
| `ComptimePtrArg`（复制 pointee） | 删除；直接用 `ComptimePtr`（`Box` 为身份对象，结构化 eq/hash；非 const 只能进内联） | fn/interp §4.2 |
| `hir.ConstRef` | 删除；全局在 HIR 里是**值**（`hir.Const`），需要 place 时 `_as_ref` 装进 box | astgen/interp §4.3 |
| `sval.ConstRef` | 删除；「指向编译期值的常引用」由 `ComptimePtr`（`ConstComptime` box）承担 | interp §4.3 |
| `std.core.as_static_ptr` | `ConstComptime` + `ref`（自动 lower） | std/interp §4.3 |
| `std.core.gstr` | 删除；字节串直接 lower | std/interp §10 |
| `fn.ClosureValue`/`hir.Closure`/`hir.RebuildCapture` | 删除；闭包=匿名 struct + `__call__` | fn/hir/interp §9 |
| `sval.ClosureType` | 删除；闭包值即其匿名 struct 类型（保留 DST 的 `AnyFunction`/`FunctionType`） | sval/fn/interp §2 |
| `hir.CallInplace.captures`/`fn.CallSignature.captures` | 删除；捕获经 struct 字段（`self`） | hir/fn §9 |
| `std.core.as_runtime_closure` | 删除 | std §9 |
| `syntax.as_func_ptr` | 保留，改为 ZST struct 取 `__call__` | syntax/interp §9.5 |
| `hir.DestructureStore` | **不引入** | hir §7.2 |
| `hir.TuplePtr` | 删除；由 `ResultLocTree` 取代 | hir/astgen §7.2 |
| 浅层（inline）编译期 | 删除；只有「完全编译时已知」 | 全体 §0 |

---

## 12. 实施顺序（建议）

按「先立数据模型、再改类型、后改流程、最后清理 std/测试」推进，尽量保持每一步可编译：

1. **sval 数据模型**：新增 `Box`/`ComptimePtr`/`CastedPtr`、`ScalarType`/`AggregateType` 类型层次
   （`Undefined`→`UndefinedScalar` + 新增 `UndefinedAggregate`）、`StructField.is_comptime/is_ref`、
   `RetSpec` 编译时返回值叶子；`TupleType`/`StrDictType` 运行时表示（§0.2、§2、§5.1、§7.3）。
2. **hir 结构**：`ResultLocTree`、`Alloca.is_comptime`、删 `InlineMode`/`TuplePtr`/`Closure`/`RebuildCapture`/
   `CallInplace.captures`；各 RLS 指令 `ret` 改树（§3.3、§7.2、§9）。
3. **astgen**：`_split_comptime`(+ConstComptime)、`Alloca` 构造点重判、`_gen_target_tuple` 返回树、
   `_gen_result_loc` 树化、闭包 `move`、`const_comptime()` 标记（§3、§7.2、§9）。
4. **interp 核心**：`_is_comptime_val`、`pending slot` 三态、`store`/`load`/`_type_of` 用 `Box`+`ComptimePtr`、
   聚合/Option/union 统一（§1、§3、§5、§8）。
5. **interp 指针/RLS/返回**：`ref`/`field_index_addr`/`exec_ptr_cast`/`subscript`、`_deliver_result` 树化、
   `RetSpec` 编译时叶子、自动 lower（§4、§7）。
6. **interp 闭包**：`exec_make_closure` 重写、删 `as_runtime_closure` 家族、`exec_as_func_ptr`（§9）。
7. **std + 反射**：`core.py`/`mem.py`/`openmp.py`/`int.py`/`reflect.py`/`__init__.py`（§10）。
8. **测试**：迁移/新增用例（§13）。

---

## 13. 测试影响

需要更新或新增的用例文件（`spy/spy/tests/`）：

- `comptime.py`、`comptime_ptrs.py`、`comptime_structs.py`：混合存储用例需改为
  「整体编译期」或「普通变量运行时」；`as_static_ptr` 用例改 `ConstComptime`+`ref`。
- `tuple_ops.py`：元组 `+=` 用例**全部按 §7.5 的模式改写或删除**；新增元组作运行时值的用例。
- `tuples`/`tuple_args.py`：元组运行时表示、打包进普通变量。
- `closures.py`：`as_runtime_closure` 用例删除；改为闭包=struct 的一等值用例；
  `@syntax.closure(move=...)` 新增用例。
- `varargs.py`/`varargs_combos.py`：`*args`/`**kwargs` 用 struct + `is_ref` 表示。
- `strings.py`/`c_varargs.py`：`gstr` 删除 → 直接字节串；`sstr` 仍可用。
- `reflect.py`：`StructField.is_comptime/is_ref`、`TupleType`/`StrDictType` 运行时表示。
- `static.py`：`as_static_ptr` → `ConstComptime`+`ref`。
- `inline_loops.py`/`loops.py`：编译期循环/`is_comptime` 推断。
- `structs.py`：`is_comptime`/`is_ref` 字段。
- 新增：`ConstComptime`、const 编译期指针跨非内联边界、非 const 跨边界报错、写 const box 报错、
  写运行时值进 `Comptime` 报错。

---

## 14. 主要行为变化 → 代码映射（旧 → 新）

1. **无混合存储** → `_is_comptime_val` + `PendingSlot.is_comptime` + `store`/`load`（interp §1/§3/§5）。
2. **编译期值严格为 `sval` 值 + 编译期对象** → `_is_comptime_val`（删 `RuntimeVal` 放宽）。
3. **编译期指针全局化、按身份别名、不复制** → `sval.ComptimePtr`（`Box` 身份 + `offset`，自动 eq/hash）（§4.2）。
4. **`ConstComptime` + 自动 lower** → `Box.is_const` + `_const_to_mir` 自动入口（§3.4/§4.3）。
5. **`as_static_ptr`/`gstr` 移除、`sstr` 库函数、`as_func_ptr` 调整** → std §10。
6. **元组/字典有运行时表示、`ResultLocTree`、多值打包** → §2/§7。
7. **闭包=匿名 struct** → §9。
8. **struct 字段新标记** → §2.4。
9. **`inline_mode → is_comptime`** → §3.3。
10. **旧 `Comptime*` 形式消失、`ComptimeCastedPtr → CastedPtr`** → §11 表。
11. **函数值类型**：`fn.FunctionValue.get_type()` 返回**常函数指针**（§4.3）→ `spy.typeof(f)` 等观察点变化（§10）。

---

## 15. 行为保留清单（回归目标）

- `syntax.comptime()`/`syntax.const_comptime()` 标记语法与错误条件。
- `syntax.unroll()` 语法/上限/展开。
- 编译期 `if` 选择与死分支不编译。
- `spy.typeof` 探针、`spy.compile_log`、`compile_option`、`compile_error`、`undefined()`。
- `ref`/`ptr_cast`/`p[...]`/指针 const 隐式转换。
- 多返回值降级签名（一值按值 + 其余 result 指针；全指针时 void；嵌套展平）。
- 元组参数铺平、元组返回多值、泛型元组；解构走 RLS。
- `*args`/`**kwargs` 绑定与访问。
- 嵌套 `def`/`lambda` 捕获/调用、`@syntax.closure` 选项；泛型镜像。
- `std.reflect` 语义；泛型求解；ZST 规则。

---

## 16. 错误清单落点（§13 行为文档）

错误消息**照用**行为文档给出的英文串（如 `cannot store a runtime value into a compile-time variable`）。

| 触发 | 落点 |
|---|---|
| 运行时值写进 `Comptime`/`ConstComptime` | `interp.store`（`ComptimePtr` 分支） |
| 含运行时值的聚合初始化 `Comptime` | `interp.store`/`_coerce`（聚合整体判定） |
| `Comptime` 参数收到运行时实参 | `interp._check_comptime_args` |
| 写只读的编译期指针（指针视图只读或 `Box.is_const` 只读） | `interp.store`（`ComptimePtr`） |
| `ConstComptime` 无初值 | `astgen._gen_ann_assign` |
| 物化非 const 编译期指针为运行时值/地址 | `interp`（`ComptimePtr` 的 lower 入口/`_val_to_node`） |
| 编译期 `if`/`unroll` 条件非编译期值 | `interp._exec_if`/`_exec_loop` |
| `comptime()`/`const_comptime()` 错误用法 | `astgen._gen_stmt`/`_gen_syntax_call` |
| 解构 arity 不符 | `interp.store`（树）/`astgen` |
| 解构 RHS 是结构体/数组构造 | `astgen._gen_result_loc` |
| 元组下标越界/负/运行时 | `interp.subscript` |
| 元组 `+=` | `interp.binary_assign` |
| `tuple[T, ...]` 变长返回 | `sval.result_leaves` |
| `as_func_ptr` 的 `f` 非「有 `__call__` 的 ZST struct」 | `interp.exec_as_func_ptr` |
| 写透 `is_ref` 字段 | `interp.field_index_addr`/`store` |
| 运行时 struct 的 `is_comptime` 字段给构造实参 | `interp.finish_struct` |
| `compile_error` 非字节串/运行时消息 | 现有 `_builtin_compile_error` |
| `typeof` 调用抛错离开探针 | 现有探针逻辑 |
| 结果位置形状循环依赖（互相递归省略返回注解） | `interp`（结果位置定型，§7.2） |

---

## 17. 待确认问题

已确认的决定均已直接写入上文对应章节（§0.2/§0.3/§2.0/§2.1/§2.2/§2.4/§3.1/§3.3/§3.4/§4.1/§4.2/§4.3/
§5.2/§5.3/§7.2/§7.3/§7.4/§8.1/§9.3/§9.5/§10/§16）。原 Q5（含遗留）/Q6/Q7（前半）/Q11/Q13/Q14/Q15/Q16/Q17/Q18/
Q19/Q20 已解决并迁入正文（Q5→§4.3/§10、Q6/Q16/Q17→§4.2、Q7 前半→§7.3、Q11→§2.0/§2.1/§2.2、Q13→§8.1、
Q14/Q15→§0.2/§2.4/§4.1、Q18→§4.1/§4.3/§10、Q19→§5.3、Q20→§7.2）。以下为仍**未决**的点：

- **Q7 遗留（手写指定编译期返回）**：把 `classify()==NONE` 的类型也按编译时返回（正文 §7.3 已引用）。
  **本次重构暂不决定其实现、不实现**，留待后续。
- **非内联调用的编译期返回值不做一致性检查（本次明确不做）**：取 callee `interp` 分析时**最后一条被分析
  的 `return`** 返回的编译期值，不检查各 `return` 路径是否一致（正文 §7.3）。
- **运行时控制流内的编译期写入不检测（本次明确不做）**：新模型把 `Comptime` 变量固定为编译期存储、
  不再降级为内存，因此「运行时条件/循环内的编译期写入」在语义上需要拒绝，但**本次任务不加任何此类检查**
  （已知限制，由用户避免）；仅在 §6 收敛 `_runtime_position` 以修掉 `issues.md` #5。
> **限制（原 Q20，已定并落 §7.2）**：互相递归且都省略返回注解的函数（结果形状循环依赖）无法定型——
> 报错 / 需显式注解。

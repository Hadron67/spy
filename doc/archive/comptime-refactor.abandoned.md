# Comptime 语义重构：区分「深编译时（`Comptime`）」与「内联（`Inline`）」

**本方案已废弃**

本文是 **`Comptime` / `Inline` 存储语义重构**的设计与实施清单。目标：去掉 `hir.InlineMode` 这个把两件事混在一起的枚举，把「值是否深编译时」与「聚合体是否内联存储」拆成 `PendingSlot` 上的两个正交字段，并让语言层的标记分别对应它们。相关既有结论见 `doc/issues.md`、`doc/archive/comptimebox-refactor.md`。

---

## 1. 背景：现有模型的问题

现状（重构前）：

- `hir.InlineMode` 有三个值：`NONE`（恒内存）、`NON_AGGREGATE`（可内联标量，不内联聚合）、`FULL`（可内联任意值含聚合）。`PendingSlot` 只有一个 `inline_mode` 字段。
- `FULL` 其实是**推断模式**：`PendingSlot.is_inline` 要求「每个 store 都可内联」（`_PendingStore.is_inline = 同块 and _is_comptime_val(value)`）。于是同一个 `FULL` 既被用户 `Comptime` 变量使用，也被内部「需要聚合/必须 place 形式」的槽（返回位置、`subscript` 方法槽、`_inline_arg_place`、tuple 元素 place、tagged-union 字段 place……）使用，两种用途混在一起。
- `Comptime` 这个标记的语义因此不清晰：它既不是「一定内联」，也不是「一定深编译时」，而是「按 store 推断」。这带来两个面：
  - **#4**：`c: Comptime[i32] = 5` 在 `CommitSlot` 已 fixed 成 `ComptimeBox`，之后在另一个运行期块写运行时值 → 报错。
  - **#5**：`k: Comptime[u64] = 0` 的 `Alloca` 被 `astgen._predeclare` 提升到入口块，初始化 `Store` 落在前置**编译期**循环的退出块 → `runtime_pos` 不等 → 判为不可内联 → 落内存 → 编译期下标报 “a tuple element index has to be a compile-time integer”。

另外，`interp` 里已经存在一套**深/浅**概念：`_is_comptime_val`（深：值在编译期完全确定，且不含引用寄存器的运行时值）与各种 `Comptime*Ptr` place 树（浅：值以一个编译期 place 持有，叶子可落内存）。本次重构正是把语言标记对齐到这两个既有概念上。

---

## 2. 语义与行为的改变

### 2.1 两个正交维度

`PendingSlot` 用两个字段取代 `inline_mode`：

```python
@dataclass(slots=True)
class PendingSlot(InterpVal):
    insertion: mir.Insertion
    is_comptime: bool | None = None   # 深编译时；None = infer（判据不变）
    allow_inline: bool = False        # 聚合体是否内联存储（字段各自 place）
    runtime_pos: mir.BasicBlock | None = None
    stores: list[_PendingAction] = field(default_factory=list)
    committed: InterpVal | None = None
    declared: sval.Type | None = None
```

两者都是**深**的：

- `is_comptime` 是「**整个值**是否深编译时」：对聚合而言，只有**所有**字段都深编译时，聚合才是 `is_comptime=True`。
- `allow_inline` 是「**整个聚合**是否内联」：inline 聚合指针的字段继续 `allow_inline=True`（嵌套聚合照此递归）。

由这两个字段推出的三种已定存储：

| `is_comptime` | `allow_inline` | 存储形式 |
|---|---|---|
| `False` | `False` | **完全运行时**：`RuntimeVal`（内存） |
| `True` | （自动 `True`） | **完全编译时**：`ComptimeBox`（标量）/ 全编译期 place 树 |
| `False` | `True` | **混合**：inline place 树，字段各自落 box 或内存 |
| `None` | 任意 | **infer**：由 store 推断出上面三者之一 |

- `is_comptime = True` 时**自动** `allow_inline = True`（深聚合必然内联）。
- 标量没有「混合」形态：标量在 `is_comptime=False` 时一律落内存。
- `is_comptime = None` 的推断**与旧 `PendingSlot.is_inline` 严格等价**（见 2.3）：它既决定标量是否落 `ComptimeBox`，也决定聚合是否落 place 树；后者只在 `allow_inline=True` 时才成立（也就是旧 `NON_AGGREGATE` 对聚合的排除项）。因此「默认」槽（`is_comptime=None, allow_inline=False`）仍能承接编译期标量/tuple，只是不内联 struct/array/complex。

### 2.2 标记与注解

| 标记 / 注解 | `is_comptime` | `allow_inline` | 说明 |
|---|---|---|---|
| `Comptime` / `Comptime[T]` / `syntax.comptime()` | `True` | `True` | **深编译时**：值必须在编译期确定为编译期值 |
| `Inline` / `Inline[T]` / `syntax.inline()`（**新增**） | `None` | `True` | **浅编译时 place**：承接旧 `Comptime` 的角色（聚合内联、叶子可运行时） |
| 无标记的普通变量 | `False` | `False` | 运行时内存 |
| （内部）表达式临时量 | `None` | `False` | infer；不内联聚合 |

- `syntax.inline()` 与 `syntax.comptime()` 同构：`syntax.inline()` + `a: T = e` ≡ `a: Inline[T] = e`；+ `a = e` ≡ `a: Inline = e`。它们的「必须紧跟被标记的声明」「必须声明新名字」等校验规则完全一致。
- **不对 `Comptime` 做额外检查**：`Comptime` 只是把槽标成 `is_comptime=True`，若写进引用寄存器的运行时值，仍由 `ComptimeBox` 既有的 `cannot store a runtime value into a compile-time place` 报错兜住。
- `is_comptime=None`（`Inline` 与内部临时量）的推断仍以 `_PendingStore` 的「同块 and `_is_comptime_val(value)`」为判据，`PendingSlot.runtime_pos` 语义不变；额外只加了「聚合类类型必须 `allow_inline` 才内联」这道门（见 2.3、3.7）。

### 2.3 存储判定（替换 `PendingSlot.is_inline`）

```python
def resolve_storage(slot, type):
    ic = slot.is_comptime
    if ic is None:
        # 判据与旧 ``is_inline`` 等价：每个 action 都允许让槽成为编译期 place，
        # 且「聚合类」类型只有在显式允许内联（allow_inline）时才内联——这就是旧的
        # NON_AGGREGATE 排除项。标量/tuple/dict/option/union 仍可推断为内联，这样
        # 表达式临时量才能承接没有运行期表示的编译期量（见 3.7）。
        allow = slot.allow_inline or not _is_aggregate(type)
        ic = allow and all(action.data.info()[1] for action in slot.stores)
    if _is_aggregate(type) or type 是 tuple/str-dict/option/tagged-union:
        if ic or slot.allow_inline:      # 不能化简成 slot.allow_inline：(None, False)
                                         # 的标量/tuple 仍可能推断出 ic=True
            tree = 建 inline place 树（字段/元素/子项各自 place，递归 allow_inline=True）
            tree.is_comptime = ic and（该 tree 的子 place 是否都深编译期）
            return tree                       # InlineAggregatePtr / InlineTuplePtr / ...
        return 内存                            # RuntimeVal
    return ComptimeBox(type, value) if ic else 内存   # 标量
```

- `_is_aggregate`（`interp.py:778`，= struct/array/complex）**保持**，它就是「旧的聚合排除项」的判别子：tuple/dict/option/tagged-union 不算聚合，旧 `NON_AGGREGATE` 对它们是**允许内联**的，新规则也照此保留。
- `_PendingActionData.info()` 的第二个返回值沿用「该 action 允许让槽成为编译期 place」的含义；`_PendingAggregate`/`_PendingTuple` 恒 `True`，`_PendingPtrConvertion` 恒 `False`，`_PendingStore` 为「同块 and 深编译期值」。
- `_stores_decide_storage` 保持；它只决定「声明的 `Inline`/叶子槽是否把存储交给 store」。
- 与旧 `is_inline` 的逐条对照（`store` 全可内联时）：

  | 类型 | 旧 `is_inline` | 新 `resolve_storage` |
  |---|---|---|
  | struct/array/complex | `False`（内存） | `allow = allow_inline or False = False` → 内存 |
  | 标量 | `True`（`ComptimeBox`） | `ic=True` → `ComptimeBox` |
  | tuple/dict/option/union | `True`（place 树） | `ic=True` → place 树 |
  | 任一类型（有跨块 store） | `False` | `ic=False` → 内存 |

- 零尺寸类型不进本函数的新分支：仍由 `_commit_pending_slot` 的 `type.get_unit_value()` 分支处理（`interp.py:7824`，得到 `ComptimeBox(unit)`）——`alloca` 的「直接返回指针」路径（见 2.4）也必须走同一判定，不能对 `is_comptime=False` 的零尺寸类型直接发 `mir.Alloca`。
- `tree.is_comptime` 必须由 tree 的**子 place** 推导，不能在建 tree 时就当成缓存字段（此时子 place 只是未提交的 `PendingSlot`），也不能依赖后续会被就地改写的 `Comptime*Ptr` 字段；用 `_is_comptime_val` 按需计算。详见 2.5 的说明。

### 2.4 `alloca` 的约定：类型与存储至少一个未知才用 `PendingSlot`

`PendingSlot` 只表示「最终类型或最终存储尚未定」。因此当 `declared is not None and is_comptime is not None`（类型和存储都已知）时，`alloca` **不建 `PendingSlot`，直接返回对应的指针**：

- `Comptime[T]`（`is_comptime=True`）→ `ComptimeBox`（标量）或全编译期 place 树（聚合）。
- 普通标注变量 `T`（`is_comptime=False, allow_inline=False`）→ 直接发 `mir.Alloca`，返回 `RuntimeVal`；但**零尺寸类型例外**：它没有内存，仍要按 unit 值返回 `ComptimeBox(unit)`（见 2.3 末段），不能走 `mir.Alloca`。

其余情况才建 `PendingSlot`：

- `Comptime`（裸，无类型）→ `is_comptime=True`，类型未知 → `PendingSlot`（在 `CommitSlot` 处按类型定成 box / place 树）。
- `Inline` / `Inline[T]`（`is_comptime=None`）→ 存储未知 → `PendingSlot`。
- 内部临时量（`is_comptime=None, allow_inline=False`，即「默认」）与叶子 place（`allow_inline=True`，需要延迟提交以让运行时值落内存）→ `PendingSlot`。

配套：

- `alloca` 的 `is_comptime` 默认值**保持 `None`**，不要改成 `False`：astgen 里那些 `NON_AGGREGATE` 临时量（`_as_ref`/retloc/`raise`/`it`）迁到「默认」后仍要靠 `None` 的推断去承接编译期标量（否则会丢内联，见 3.7）。
- `_init_arg_node`（`interp.py:2034`，旧默认 `NONE`，把运行期参数写进内存）必须**显式** `is_comptime=False`：它是标量 + 空 stores，沿用新默认 `None` 会被推断成 `ComptimeBox`，再存入 `mir.Param` 就会报 `cannot store a runtime value into a compile-time place`。
- `_commit_pending_slot` 对**非** `PendingSlot` 参数改为「已实体化」语义：转去 `_commit_place_tree(...)`（提交它内部未提交的叶子 place），不再报 `can only commit a pending slot`。这样 `hir.CommitSlot` 对直接返回的 place 也能安全执行。
- 去掉 `alloca` 的 `commit_declared` 形参：是否延迟提交由上面「类型/存储是否已知」一以贯之。

### 2.5 类型重命名

因为 “comptime” 现在专指 `is_comptime`（深），旧的 `Comptime*Ptr` 等表示的是 **inline place**，故统一改名。**`ComptimeBox` 与 `ComptimeVal` 保持不变**（`ComptimeBox` 只装编译时值；其字段类型是 `InterpVal` 是因为 `mir.GlobalConstant` 也算编译期已知常数、只是无法表示成 `sval`）。

| 旧 | 新 |
|---|---|
| `ComptimeAggregatePtr` | `InlineAggregatePtr`（+ 字段 `is_comptime: bool`） |
| `ComptimeTuplePtr` | `InlineTuplePtr`（+ `is_comptime: bool`） |
| `ComptimeDictPtr` | `InlineDictPtr`（+ `is_comptime: bool`） |
| `ComptimeOptionPtr` | `InlineOptionPtr`（+ `is_comptime: bool`） |
| `ComptimeTaggedUnionPtr` | `InlineTaggedUnionPtr`（+ `is_comptime: bool`） |
| `ComptimeAggregate` | `InlineAggregate` |
| `ComptimeTuple` | `InlineTuple` |
| `ComptimeDict` | `InlineDict` |
| `ComptimeOption` | `InlineOption` |
| `ComptimeTaggedUnionValue` | `InlineTaggedUnionValue` |
| `ComptimeCastedPtr` | `InlineCastedPtr` |
| `ComptimeResultPtr` | `InlineResultPtr` |
| `fn.ComptimePtrArg` | `fn.InlinePtrArg` |
| `ComptimeBox` / `ComptimeVal` | **保持** |

- 每个聚合 inline **指针**新增 `is_comptime: bool`（聚合整体的深编译期性＝各字段之与）；值形态不加。
- 该 `is_comptime` **不能是构造时算好的缓存字段**：`ComptimeOptionPtr.is_null/payload_ptr`（`interp.py:4000`/`8069`）、`ComptimeTaggedUnionPtr.tag/payload_ptr`（`interp.py:4299`/`4325-4344`）、`ComptimeTuplePtr.values`（`interp.py:6126`，`t += u` 就地增长）都会在构造后被改写，缓存值会过期；并且建 tree 时子 place 还是未提交的 `PendingSlot`。建议改为按需计算（复用 `_is_comptime_val`）或在这些写点上同步维护。
- 相关辅助函数同步改名：`init_comptime_option`→`init_inline_option`、`init_comptime_tuple`→`init_inline_tuple`、`init_comptime_dict`→`init_inline_dict`、`init_comptime_tagged_union`→`init_inline_tagged_union`、`_store_comptime_tagged_union`→`_store_inline_tagged_union`、`_as_comptime_option`→`_as_inline_option` 等。
- 讲「编译期值 / 深编译期」的名字**保持**：`_is_comptime_val`、`_to_comptime`、`_comptime_index`、`_comptime_int`、`_comptime_py_op`、`_comptime_py_value`、`sval.unwrap_comptime`、参数上的 `is_comptime`。

### 2.6 #4 / #5 的行为

- **#5**：把下标变量声明为 `Comptime`（`k: Comptime[u64] = 0`）即可——`is_comptime=True` 直接定成编译期 place，不再受编译期循环退出块的 `runtime_pos` 影响。若声明成 `Inline[u64]`（`is_comptime=None`，沿用旧推断），仍会因跨块落内存而失败；`Comptime` 是显式逃生口。
- **#4**：变成明文语义——`c: Comptime[i32] = 5` 立即（或提交时）定成 `ComptimeBox`，之后写运行时值由 box 报错。

### 2.7 迁移

- 旧「浅」用法改 `Inline` / `syntax.inline()`：`std/core.py` 的 `range.__next__`（`comptime(); result = self.start`）、以及测试里 `Comptime[i32] = <运行时值>`、`Comptime = <含运行时元素的元组/分支聚合>` 等。
- 保持 `Comptime`：持类型 / 函数值 / 纯编译期标量 / 纯编译期聚合处（`typeof(...)`、`sb/rs/...`、`K/LT/ST`、`n: Comptime = 5`、`const_array(7, 4)`、`RtTagA(2)`、`Option[u0]` 等）。

---

## 3. 详细修改清单

### 3.1 `spy/compiler/syntax.py`

- 新增 `type Inline[T = Any] = T`（与 `type Comptime[T = Any] = T` 并列）。
- 新增 `def inline(): ...`（与 `def comptime(): ...` 并列，均为不可直接调用的语法标记）。

### 3.2 `spy/compiler/astgen.py`

- 新增标记对象 `_Inline`（`_Comptime` 的孪生）与 `_INLINE = _Inline()`；`_as_marker` 里识别 `syntax.inline` → `_INLINE`。
- `_predeclare`：`_COMPTIME`/`_INLINE` 的 `composable` 处理并列（两者都让紧随的声明「就地声明」）。
- 把 `_split_comptime` 推广为 `_split_storage_marker(node) -> (is_comptime, allow_inline, type_node)`：
  - `syntax.Comptime` / `syntax.Comptime[T]` → `is_comptime=True`；
  - `syntax.Inline` / `syntax.Inline[T]` → `allow_inline=True`（`is_comptime=None`）；
  - 其余 → 都不是。
- `_predeclare_ann` / `_gen_ann_assign` / `_gen_assign(by_marker)`：
  - 由「`is_comptime` 布尔」改为两个标记；
  - `hir.Alloca` 参数从 `hir.InlineMode.*` 改为 `is_comptime`/`allow_inline`（见 3.3）；
  - 报错文案（“already declared compile-time …”）扩为两种标记的冲突检查，并新增 `syntax.inline()` 的同款文案。
- 逐点替换 `hir.InlineMode` 引用（共约 16 处）：

  | 位置 | 旧 | 新 |
  |---|---|---|
  | `_predeclare_name(stmt.name, …)` | `FULL` | `allow_inline=True` |
  | `_predeclare_target(..., …)` | `NONE` | `is_comptime=False` |
  | `_predeclare_ann` / `_gen_ann_assign` | `FULL if 标记 else NONE` | 由标记给出 `is_comptime`/`allow_inline` |
  | `_gen_stmt` 的 `Raise` slot | `NON_AGGREGATE` | 默认（`is_comptime=None, allow_inline=False`） |
  | `_gen_for` 的 `it` | `FULL if is_inline else NON_AGGREGATE` | `allow_inline=is_inline` |
  | `_gen_for` 的 `iter_slot` | `FULL` | `allow_inline=True` |
  | `_gen_for` 的循环变量 | `FULL if is_inline else NONE` | 对应 `is_comptime`/`allow_inline` |
  | `_gen_lhs` / `_gen_target_tuple` 默认 | `NONE` | `is_comptime=False` |
  | `_gen_assign(by_marker)` | `FULL if by_marker else NONE` | 由标记给出 |
  | `_as_ref` / `_gen_expr` retloc | `NON_AGGREGATE` | 默认 |

- `_gen_syntax_call`：`syntax.comptime()` 的“必须是声明前标记”文案旁加 `syntax.inline()`。

### 3.3 `spy/compiler/hir.py`

- 删除 `InlineMode` 枚举；`Alloca` 的 `inline: InlineMode` 改为 `is_comptime: bool | None = None`、`allow_inline: bool = False`。
- 更新模块头（“the slot only becomes real memory …”一段）与 `Alloca` docstring（三种模式 → 两个字段）。

### 3.4 `spy/compiler/interp.py`

**数据结构 / 类型**

- `PendingSlot`：`inline_mode: InlineMode` → `is_comptime: bool | None = None`、`allow_inline: bool = False`；更新 docstring。
- `PendingSlot.is_inline` → 新的 `resolve_storage`（见 2.3）。
- 重命名（2.5 全表）：`Comptime*Ptr`/`Comptime*`（值形态）/`ComptimeCastedPtr`/`ComptimeResultPtr`；`ComptimeBox`/`ComptimeVal` 保持。
- 给 `InlineAggregatePtr/TuplePtr/DictPtr/OptionPtr/TaggedUnionPtr` 增加 `is_comptime: bool`，但**不要**做成构造时算好的缓存字段（这些指针会被就地改写，且建 tree 时子 place 还是未提交的 `PendingSlot`）；按需计算（复用 `_is_comptime_val`）或在这些写点上同步维护，详见 2.5。

**`alloca` / 提交**

- `alloca(self, declared=None, is_comptime=None, allow_inline=False) -> InterpVal`：
  - `is_comptime=True` → 强制 `allow_inline=True`；
  - `declared is not None and is_comptime is not None` → 直接返回 `ComptimeBox`/place 树/内存指针（不建 `PendingSlot`）；
  - 否则建 `PendingSlot`，`declared` 可决定时按现有逻辑延迟或立即建 place 树。
- 删除 `commit_declared` 形参；`_exec_inst` 的 `hir.Alloca` 相应简化。
- `_commit_pending_slot`：非 `PendingSlot` 参数改走 `_commit_place_tree`。
- 逐点替换 `InlineMode`（约 40 处）：

  | 位置 | 旧 | 新 |
  |---|---|---|
  | `_reserve_result_loc` code/payload、`_declared_comptime_place`、`_ret_spec_place`、`_build_catch_path`、`_materialize_union`、`_materialize_aggregate`、`_tuple_extend_element`、`_builtin_panic`、`_rebuild_capture`、`_make_runtime_call` 各点 | `NONE` | `is_comptime=False` |
  | `_init_arg_node`（裸 `alloca()`，接运行期参数） | `NONE`（默认值） | `is_comptime=False`（**不能**沿用新默认 `None`，空 stores 会被推断成 Box） |
  | `_operand_place`、`_cmp_overload`、`as_bool`、`exec_len` | `NON_AGGREGATE` | 默认（语义与旧 `NON_AGGREGATE` 等价，见 3.7） |
  | `_reserve_result_loc` value、`subscript`、`_fresh_places`(RetValue)、`_inline_arg_place`、`_fresh_place`、tuple 元素、tagged-union 字段 place、`field_index_addr` 的 `FULL` | `FULL` | `allow_inline=True` |
  | `store`：`ptr.inline_mode == FULL` | `FULL` | `ptr.allow_inline` |
  | `store`：`ptr.inline_mode != NONE`（tuple 分支） | `!= NONE` | `ptr.is_comptime is not False` |
  | `field_index_addr`：`alloca(ptr.inline_mode)` | 继承 `inline_mode` | 继承容器的 `is_comptime`/`allow_inline` |

**其它**

- `_PendingStore`：字段 `is_inline` 的语义沿用（“允许成为编译期 place”），可在注释里改名说明。
- `field_index_addr`/`exec_field_name_addr` 里对未提交 slot 取字段 place 的逻辑不变（继承容器字段）。
- `finish_struct`（`interp.py:8760`）/`finish_array`（`interp.py:8894`）的 `dest.is_inline(...)` 改为「按类型解析后是否会内联」的谓词：对 struct/array 等价于 `slot.allow_inline`（`(None, False)` 的聚合必须解析为内存，不能内联）。
- `init_tuple`（`interp.py:8940`）与 `store` 的 tuple 分支（`interp.py:3877`）的 `!= NONE` 判定 → `is_comptime is not False`：`None` 的 `NON_AGGREGATE` 临时量仍可内联 tuple，`False`（真内存槽）被排除。

### 3.5 `spy/compiler/fn.py`

- `ComptimePtrArg` → `InlinePtrArg`（含 `ArgNode` 类型别名、`solve_param_types`/`specialize`/`_thunk_call_sig` 等引用点）。
- 参数上的 `is_comptime`（`SignatureFormalArg`、`ClosureFunction`、`vararg_is_comptime`/`kwarg_is_comptime`）表示「编译期参数（深）」，**保持**。
- 同一次重命名还要覆盖 `spy/compiler/glue.py`（`from .fn import ... ComptimePtrArg`，及 `marshal` 里对它的引用），以及 `interp.py` 的 `_COMPTIME_PTR_FORMS`、`_val_to_node` 等处的 `ComptimePtrArg`。

### 3.6 测试与文档

- 测试迁移（浅语义 → `Inline`/`syntax.inline()`）：`tests/annotations.py::comptime_typed_local`、`comptime_local_holds_a_tuple`；`tests/comptime.py::comptime_marker_holds_a_runtime_value`、`comptime_scalar_written_in_a_runtime_loop`、`comptime_declared_scalar_written_in_a_runtime_loop`、`comptime_marker_*`（按需）；`tests/comptime_structs.py::comptime_struct_choose` 等。
- `#5` 的复现/规避：改为把下标变量声明成 `Comptime`（`k: Comptime[u64] = 0`）；追加运行时元素的元组改用 `Inline`。
- `doc/issues.md`：更新 #4/#5 的说明与规避。
- `README.md`：术语同步——引入 `Inline`，`Comptime` 改述为“深编译时”，`Comptime*Ptr` → `Inline*Ptr`，浅/深两层的表述与新标记对齐（涉及“类型标注的局部变量”“编译期（comptime）”“编译期聚合体”“Option”“多返回值”等小节）。
- `doc/archive/comptimebox-refactor.md`：把 `ComptimeBox`/`Comptime*Ptr` 的记号同步（或加注“旧记号”）。

### 3.7 `NON_AGGREGATE` 使用点的迁移

旧 `NON_AGGREGATE` 携带两个含义：「标量（以及 tuple/dict/option/union）可内联，用于承接没有运行期表示的编译期量」与「聚合（struct/array/complex）落内存」。新模型把两者拆开：前者由 `is_comptime` 的推断给出（`allow_inline or not _is_aggregate(type)`，见 2.3），后者由 `allow_inline=False` 给出。因此**所有 `NON_AGGREGATE` 点都迁到「默认」= `(is_comptime=None, allow_inline=False)`**，语义与旧 `NON_AGGREGATE` 逐条等价，无需单独特判。

| 位置 | 说明 | 旧 | 新 |
|---|---|---|---|
| `astgen._gen_stmt` 的 `Raise` 值槽（`astgen.py:497`） | `raise e` 的值临时量，`e` 可能是标量也可能是聚合 | `NON_AGGREGATE` | 默认 |
| `astgen._gen_for` 的 `it`（`astgen.py:700`） | 迭代器临时量（非 `unroll` 循环）；迭代器通常是聚合 | `NON_AGGREGATE` | 默认 |
| `astgen._as_ref`（`astgen.py:1156`） | 需要地址的任意表达式结果的临时量 | `NON_AGGREGATE` | 默认 |
| `astgen._gen_expr` 的 retloc 兜底（`astgen.py:1337`） | 任意表达式（含函数调用结果）的临时量 | `NON_AGGREGATE` | 默认 |
| `interp._operand_place`（`interp.py:5856`） | 方法接收者（struct）的地址 | `NON_AGGREGATE` | 默认 |
| `interp._cmp_overload`（`interp.py:5971`） | 比较运算符的 struct 操作数 | `NON_AGGREGATE` | 默认 |
| `interp.as_bool`（`interp.py:8196`） | `__bool__` 的 struct 接收者 | `NON_AGGREGATE` | 默认 |
| `interp.exec_len`（`interp.py:8242`） | `__len__` 的 struct 接收者 | `NON_AGGREGATE` | 默认 |

为什么「默认」能等价于旧 `NON_AGGREGATE`（对照 `interp.py:476-480` 的 `is_inline`，在 store 全可内联时）：

| 类型 | 旧 `is_inline` | 新 `resolve_storage` |
|---|---|---|
| struct/array/complex | `False`（内存） | `allow = allow_inline or not _is_aggregate = False` → 内存 |
| 标量 | `True`（`ComptimeBox`） | `ic=True` → `ComptimeBox` |
| tuple/dict/option/union | `True`（place 树） | `ic=True` → place 树 |
| 任一类型（有跨块 store） | `False` | `ic=False` → 内存 |

关键点：`foo() == i64` 这类「临时量承接编译期标量」的场景正是靠标量那一行保住——`foo()` 的结果槽来自 `_gen_expr` 的 retloc 兜底（`astgen.py:1337`），`is_comptime=None`，标量分支仍会推断出 Box，不会因为 `allow_inline=False` 而误落内存。所以推断式**必须**保留 `allow_inline or not _is_aggregate(type)` 这一项，而不能无条件 `(None, False) → False`。

配套约束：

- `alloca` 的 `is_comptime` 默认值**保持 `None`**（而不是 `False`）；改成 `False` 会让上表退化为「一律内存」，破坏标量/tuple 内联。真正需要显式 `is_comptime=False` 的只有 `_init_arg_node`（旧默认 `NONE`，接运行期参数，见 2.4、3.4）。
- `_gen_for` 的 `it` 是 `FULL if is_inline else NON_AGGREGATE`，迁为 `allow_inline=is_inline`（`is_comptime=None`）：`unroll` 循环取 `allow_inline=True`，其余取默认。
- `_gen_stmt` 的 `Raise` 值槽、`_as_ref`、`_gen_expr` retloc 均为默认；它们的旧 `NON_AGGREGATE` 行为由 2.3 的推断式原样重现。

---

## 4. 建议实施顺序

1. `syntax.py` 加 `Inline`/`inline()`；`astgen.py` 加 `_INLINE` 与 `_split_storage_marker`（先与旧 `Comptime` 语义并存，`Inline` 暂映射到 `NON_AGGREGATE`，保证可编译）。
2. `hir.py`：`Alloca` 换成两字段；`interp.py`：`PendingSlot` 换成两字段、`is_inline` → `resolve_storage`、`alloca` 直接返回指针、`_commit_pending_slot` 放宽。此时 `Comptime` 仍走旧推断（先不改行为），跑全量确认无回归。
3. 把 `Comptime` 标记改为 `is_comptime=True` / `Inline` 改为 `allow_inline=True`，迁移 `std`/`tests`，跑全量。
4. 纯机械重命名（`Comptime*` → `Inline*` + 加 `is_comptime` 字段），跑全量。
5. 文档（README、issues、archive）。

---

## 5. 待定 / 开放问题

- **`Inline` 参数**：本次只为局部变量引入 `Inline`/`syntax.inline()`；形式参数上的 `Comptime[T]` 仍是「深编译时参数」。是否需要 `Inline` 形参（浅 place 参数）留待后续任务。
- **`is_comptime=True` 的检查**：暂不加“必须深编译期”的显式检查，依赖 `ComptimeBox` 既有报错；后续若要更早/更清晰的诊断可补。
- **`InlineBox` 命名**：本次不引入（标量 inline 存储仍叫 `ComptimeBox`）。

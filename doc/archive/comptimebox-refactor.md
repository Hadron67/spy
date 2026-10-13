# ComptimeBox 重构：让运行时值一律落内存

本文记录「禁止任何寄存器写入 `ComptimeBox`」这条规则下**类别 B**（既有的、先建好再写入的编译期 place）的根因分析、已验证的解决方案与影响，作为后续实现的设计依据。相关既有结论见 `doc/issues.md`。

## 0. 背景与现状

目标规则：**`ComptimeBox` 只装编译期值**——任何引用了寄存器（`mir.Inst`/`mir.Param`）的运行时值都不进 box，而落内存。

已经落地的部分：

- **核心规则（`interp.HirRunner._record_pending_store`）**：`is_inline = 同块 and _is_comptime_val(value)`。于是 `PendingSlot.is_inline` 里那句 `all(store.data.info()[1] ...)` 会把「装了运行时值的 place」判成落内存。配合 `_is_comptime_val` 把「不引用寄存器的 `RuntimeVal`」（全局常量、字符串、函数、null）仍视为编译期值（否则 `gstr`/`sstr` → `as_static_ptr` 会挂）。
- **声明式 `Comptime[T]` 变量**：`hir.Alloca` 传 `commit_declared=False`，由 `HirRunner._stores_decide_storage` 决定哪些类型把存储交给 store（标量、或「所有字段都可如此处理」的 struct）；`PendingSlot.declared` + `committed_type()` 承载「类型已定、存储待定」；`HirRunner._commit_place_tree`（`_commit_tuple_places` 的一般化）提交 place 树里未提交的叶 place。（`_stores_decide_storage` 是**按类型**判断「存储能否交给 store」的临时代理，§3.3 会把它收敛成「可寻址性」判据。）
- **跨块的 box 写入守卫（`interp.HirRunner.store` 的叶子 `case ComptimeBox()`）**：往 box 写「引用了寄存器、且该 box 是在另一个运行期块创建」的值时报 `CompileError`。这是「运行时值不进 box」的**部分**实现（只拒绝真正会崩的跨块写）。
- **类别 A（已修）**：`interp.HirRunner.field_index_addr` 对 tagged-union 构造返回的字段 place 改用 `InlineMode.FULL`，避免编译期字段被物化成内存再读回成寄存器。修完后，在**无条件**守卫下只剩类别 B 的用例失败（见 §1）。

`ComptimeBox` 现在还有一个 `runtime_pos`（创建它的运行期块）字段，供上面的跨块守卫判断。

## 1. 类别 B 的用例

在**无条件**守卫（任何寄存器都不进 box）下仍失败的 3 个（全部是 option payload）：

| 用例 | 写入的 place | 值来源 |
|---|---|---|
| `options.test_a_pointer_to_compile_time_option_storage` | `Comptime[Option[T]]` 的 payload box | 运行期调用结果的字段（`ExtractValue(Load(struct),1)`） |
| `options.test_a_pointer_to_compile_time_pointer_option_storage` | 同上（payload 为 `ptr(i32)`） | 同上 |
| `options.test_an_absent_pointer_option_in_compile_time_storage` | 同上 | `pass_declared_ptr_mixed` 里 `ch: Comptime[Option[PtrMixed]] = pm`，`pm` 运行期 |

## 2. 根因：一批「先建好、后写入」的 eager 叶 place

这些 place 都在**知道值之前**就建成了编译期 place（`ComptimeBox` 或聚合指针里的 field），随后被写入运行时寄存器：

- **Option payload**：`HirRunner.init_comptime_option` → `_fresh_place(child, Undefined)`（eager box）；写入在 `store` 的 `ComptimeOptionPtr` 分支 `self.store(ptr.payload_ptr, option.value)`。
- **Tagged union payload / variant field**：`init_comptime_tagged_union` / `_write_tagged_union_tag` / `_store_comptime_tagged_union` 用 `_fresh_place(variant, Undefined)`；`_finish_struct_into_tagged_union` 走 `store(variant_field, load(place))`。
- **编译期指针 pointee**：`_init_ptr_target` / `_declared_comptime_place` 建 eager box，随后写运行时叶子。
- **声明式 `Comptime[T]`（T 为 tuple/dict/array/option/union）**：`init_comptime_tuple` / `init_comptime_dict` / `init_inline_aggregate` 的 `_fresh_place`。

共用入口就是两处 eager 造 place：`_fresh_place`（叶子）与 `_declared_comptime_place`。

### 2.1 为什么不能「简单延迟提交」（实测约束）

- 把 `_fresh_place` 叶子改成未提交 `PendingSlot`、并延迟 `alloca(declared)` → 打挂 **26–49** 个用例：这些 place 在构造成型 / `CommitSlot` 之前就会被**读、取地址、查类型**（`_type_of` / `_place_type` 对未提交 slot 返回 `None`），报 `cannot load from a slot before it is committed` / `unsupported MIR instruction Insertion` / `... has no runtime representation`。
- 把 Option/TaggedUnion 纳入 `_stores_decide_storage`（延迟声明式变量的提交）→ 立刻 9 个用例报 `cannot take a field or element address of ComptimeTaggedUnionPtr`：**option/union 的构造路径需要 place 形式先建好**。

结论：不能靠「延迟到 `CommitSlot`」。正确的机制是**在存储确定处（store 序列闭合处）提交**，并用「可寻址性」判据把成型前就要取地址的 place 排除在延迟之外（见 §3.2、§3.3）；「首次使用时提交」不能当通用触发点（它只在「永不赋值」的叶子上才需要，见 §3.2）。

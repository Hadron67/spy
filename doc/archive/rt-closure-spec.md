# 运行时闭包（Runtime Closure）实现方案

本文是实现「运行时闭包」这一特性的设计文档。

状态：**已实现**（44972c481ab40b5caaf1c0cc246417e9799c1747）。

---

## 1. 功能描述

### 1.1 背景

现在闭包（嵌套 `def` / `lambda`）**只存在于编译期**：

- `fn.ClosureValue` 持有 `ClosureFunction` + 捕获 place（`interp.InterpVal`），
- `interp.HirRunner._call_closure` 要么把它**内联**（`_start_inline`），要么编译成**隐藏指针参数**的运行时函数，
- `sval.ClosureType.to_mir_type()` 返回 `None`，所以闭包值进不了运行时位置。

`interp.HirRunner.call` 对一个 struct 值也没有 `__call__` 分派（会报 `cannot compile a call to ...`）。

### 1.2 新增功能

1. **函数调用语法的 `__call__` 重载（通用）**
   表达式 `x(args, ...)` 中，若 `x` 的静态类型是「指向某个 `sval.StructType` 的指针」，且该 struct 有 `__call__` 方法，则按普通方法调用处理：`x` 的地址作为 `self` 传入。

2. **`std.core.as_runtime_closure(closure, as_copy: bool = False)`**
   把一个编译期闭包 `closure` 变成一个**普通的 spy struct 值**，因此它可以被存进变量 / 结构体字段、传参、返回、调用（通过 `__call__`）。

   - struct 的**字段** = 闭包各捕获变量的 `ArgNode` 树里**递归**收集到的运行时叶子（`RuntimeArgNode`）。
   - struct 的 **`__call__` 方法** = 一个**普通生成的函数**：接收闭包声明的形参，从 struct 字段重建出捕获 place，调用闭包，并把结果作为自己的返回值。
   - `as_copy=False`（默认）：顶层运行时捕获**按引用**——字段里存「指向被捕获变量存储的指针」。
   - `as_copy=True`：顶层运行时捕获**按值**——把捕获指针指向的内容复制进字段，闭包通过字段的地址访问它（因此内容随 struct 存活）。

### 1.3 语义与行为

- 闭包**内联与非内联都支持**：两者都由 `_call_function_entry` 统一处理（原 `_call_closure` 并入它，见 2.6/3.2；内联体直接进 `__call__`，非内联则编译成隐藏捕获参数的运行时函数）。
- 闭包**泛型参数**支持：`__call__` 镜像闭包 signature 的泛型参数，调用时正常求解。
- **`*args` / `**kwargs`** 支持：`__call__` 镜像闭包的这两个形参。
- **多返回值 / 抛异常 / panic** 都继承闭包自身，不需要额外处理（`_call_function_entry` + `_make_runtime_call` 已经覆盖）。
- **ZST**：字段类型为零尺寸时该字段无存储；整个 struct 可能成为 ZST。按现有 ZST 逻辑自洽，不报错。
- 生成的 struct 类型是**匿名**的：只能由 `as_runtime_closure` 的返回值直接持有（局部变量、字段等），不能写进源码注解。

### 1.4 对捕获变量没有任何限制

捕获变量**不加任何限制**：编译期 / 运行时，标量 / 聚合 / 数组 / complex / option / plain union / tagged union，以及它们任意嵌套，全部支持。原因是收集与重建都严格镜像 `_val_to_node` / `_init_ptr_target` / `_init_container_target`，而这几支已经覆盖了 `ArgNode` 的全部形状（`_val_to_node(val, is_ref)` 是 `InterpVal → ArgNode` 的唯一入口，见 §3.2）：

| 捕获 place（`InterpVal`） | `_val_to_node(place, False)` 得到的 `ArgNode` | 处理方式 |
| --- | --- | --- |
| `ComptimeVal(obj)`（标量 / 类型值 / 函数值） | 编译期叶子（`sval.AnyValue`） | 直接烘进 `__call__` |
| `ComptimeBox` | `ComptimePtrArg`（内容为编译期叶子或运行时叶子） | 递归 |
| `ComptimeAggregatePtr`（结构体 / 数组 / complex） | `ComptimePtrArg` + `CompoundArgNode` | 递归 |
| `ComptimeOptionPtr`（编译期 tag） | `ComptimePtrArg` + `CompoundArgNode(OptionType, …)` | 递归 |
| `ComptimeOptionPtr`（运行时 tag） | `RuntimeArgNode(Ptr[OptionType])`（整体物化） | 字段（顶层） |
| `ComptimeTaggedUnionPtr`（编译期 tag） | `ComptimePtrArg` + `CompoundArgNode(TaggedUnionType, …)` | 递归 |
| `ComptimeTaggedUnionPtr`（运行时 tag） | `RuntimeArgNode(Ptr[TaggedUnionType])`（整体物化） | 字段（顶层） |
| `ComptimeTuplePtr` | `ComptimePtrArg` + `tuple(...)` | 递归 |
| `ComptimeDictPtr` | `ComptimePtrArg` + `frozendict(...)` | 递归 |
| `RuntimeVal(ptr, Ptr[T])` | `RuntimeArgNode(Ptr[T])` | 字段（顶层） |

唯一「不落字段」的是**完全没有运行时表示**的捕获（纯编译期值 / 类型），它直接烘进 `__call__` 的特化——这不是限制，而是它本来就没有运行时状态可存。

要标注的边界：`as_copy=True` 时顶层捕获会 `load` 被捕获的存储，所以被捕获的**存储**不能是动态大小类型（DST）——但 DST 本身无法作为变量存储存在，因此不会出现。

「没有任何限制」针对的是标量 / 聚合 / 数组 / complex / option / plain union / tagged union 及其任意嵌套。**唯一的例外**：捕获一个**闭包值**（`ClosureValue`）并把它当作**实参传给非内联被调者**，仍受既有的 `ClosureType` 限制（见 `pending-problems.md` #3）——这与「闭包只能传给 inline 函数」的既有限制一致，与运行时闭包本身无关。

---

## 2. 整体逻辑

### 2.1 `x(...)` → `__call__`

`x`（被调用者）本身的类型是 `sval.StructType`（一个 struct 值）；但 `astgen` 把 callee 按 **place** 生成，所以 `HirRunner.call` 里 `callee = self._auto_deref(callee)` 之后，callee 的类型是**指向该 struct 的指针** `PointerType(StructType s)`——它就是 `x` 的存贮地址。因为 callee 一定是 place，不可能是「非 place 的寄存器值」，所以判据只看 `_auto_deref` 之后 `_type_of(callee)` 这一支即可。

`HirRunner.call` 在 `_callee_object(...)` 处理之后、函数指针分支之前插入一步：

```
type = _type_of(callee)（已 _auto_deref）；
若 type 是 PointerType(elem=StructType s) 且 s 有 __call__ 方法：
    return self.call_method(callee, '__call__', args, ret, on_return)
```

注意：这一步只影响普通调用 `x(...)`（`hir.CallInplace`）；`x.m(...)`（`hir.CallMethodInplace`）走的仍是原来的 `call_method` 分派，不受影响。

`call_method` 会把 callee（即指向 struct 的地址）作为第一个实参（`self: Ptr[Struct]`）传入，之后就是正常的方法特化 / 编译 / 调用。

为了能解析到生成的 `__call__`：`_method_of` 在 `struct.get_method(name)` 拿到一个已经是 spy 值（`fn.FunctionValue`，例如生成的 `__call__`）的条目时**直接返回它**，不再走 `resolve_global`（后者只认 Python 对象）；拿到 Python 对象（普通方法）时维持原逻辑（`resolve_global`，泛型 struct 再包一层 `BoundMethod`）。`is_static_method`、`_resolve_method`、`call_method` 的既有分支都不改。

### 2.2 `as_runtime_closure` 的执行流程

`_call_builtin` 按名字分派到新增的 `HirRunner._builtin_as_runtime_closure`：

1. 校验 / 规范化参数（`@builtin_func` 不产生 `Signature`，`_call_builtin` 也不绑定实参，所以**全部自己来**）：`closure` 必须是 `ClosureValue`；`as_copy` 必须是**编译期常量 bool**（决定字段存指针还是存值，必须在编译期定）。
   - 取 `closure`：第 1 个位置实参（也接受关键字 `closure=`）；`_arg_value` 后在 `ComptimeVal`/`ConstRef` 里，用 `_callee_object(...)` 解开得到 `ClosureValue`。
   - 取 `as_copy`：第 2 个位置实参**或**关键字 `as_copy=`；`_to_comptime(_shallow_normalize(...))` 后必须是 Python `bool`（spy 里只有裸 `bool` 这个值，没有 `sval.Bool`），否则报错。
   - 其余位置实参、未知关键字、重复给出 `as_copy` 一律报错。
   - **默认值要自己补**：`@builtin_func` 只把名字绑成 `sval.BuiltinFn`，**不产生 `Signature`**，所以缺省的 `as_copy` 不会由 `bind_arg_pos` 自动填上——本内置在「未提供 `as_copy`」时自行当作 `False`。
2. 物化捕获 place：未 commit 的 `PendingSlot` 先 commit（复用 `_inline_capture_values` 的做法），得到 `capture_places`。
3. 调 `_build_runtime_closure(closure, capture_places, as_copy)`，得到 `RuntimeClosurePlan`（struct 类型 + 字段表 + 每个捕获的 `ArgNode` 与物化 place + 生成的 `__call__`）。
4. 让结果直接落在调用结果位置 `ret`，**既不再另开临时槽，也不做整 struct 拷贝**：struct 的每个运行时叶子都经 `ret` 的地址直接写入（见 2.8）。当 `ret` 还是未提交的 `PendingSlot` 时，它可能还没有自己的地址——既可能是变量槽（随后由 `CommitSlot` 提交），也可能是某个正在构造的结构体的字段（随后由 `finish_struct` 绑到字段地址）。两种情况都统一用 `_defer_ptr_convertion(ret, plan.struct_type)` 取得一个**延迟地址**：它记录一个 `_PendingPtrConvertion`，在槽提交 / 绑定时解析成真正的地址。该 action 的 `info()` 恒为 never-inline，因此会把槽强制落到 `alloca`（内存）分支——生成的 `__call__` 需要的 `self` 是 struct 本身的可寻址地址（否则 `self` 会经 `_val_to_node` → `ComptimePtrArg` 在 callee 内被复制，`as_copy=True` 的字段写回会丢）。`ret` 不是未提交的 `PendingSlot` 时（已有地址，或已提交的 inline 聚合），直接以它为目的地址。
5. 按捕获 `ArgNode` 树**递归**把每个运行时叶子的值直接写进该地址的对应字段（`field_index_addr(dest, i)`，`dest` 为第 4 步的地址；字段值从 `plan.capture_places` 读，见 2.7）。
6. 结果已在 `ret` 中——**不再 `load`/`store`**，省掉一次整 struct 拷贝。

### 2.3 struct 类型、字段与 `RuntimeClosurePlan`

**不新建模块**：builder / plan 都放在 `interp.py`（`_builtin_as_runtime_closure` 本来就在那）。**不引入 `Recipe` 类型**：直接复用 `fn.ArgNode`——它已经完整描述结构，且**编译期叶子本身就是 `sval.AnyValue`**（可直接嵌进 `hir.Const` / 原样复用），所以「结构 + 编译期常量」全都由 `ArgNode` 承载。唯一需要额外记的是「第 i 个运行时叶子落在第几个字段」，这由**规范遍历顺序**（见 2.6）确定，不需要额外的 map。

```
@dataclass(slots=True)
class RuntimeClosurePlan:
    struct_type: sval.StructType       # head.specialize(())
    call_fn: FunctionValue             # 生成的 __call__
    as_copy: bool
    field_names: tuple[str, ...]       # '_cap0', '_cap1', ...
    field_types: tuple[sval.Type, ...]
    captures: tuple[ArgNode, ...]      # 原样复用 _val_to_node 的结果，按捕获顺序
    capture_places: tuple[InterpVal, ...]  # 物化后的捕获 place，按捕获顺序；构造字段值时读它
    leaf_bases: tuple[int, ...]        # 第 i 个 capture 第一个运行时叶子的全局字段下标（处理它之前的累计计数）
```

- `sval.StructTypeHead(name_base)`（非泛型），`name_base` 由闭包名派生（`f'{closure.name_base}$closure#{uid}'`）。`uid` 是 **`HirRunner` 上的一个计数器**（例如 `self._runtime_closure_counter`，每次自增），**不用全局变量**；`closure.name_base` 已包含创建帧的函数名（即特化名），再加该计数器即可保证同一 `HirRunner` 生成的符号唯一。
- **不去重**：每执行一次 `as_runtime_closure` 就新建一个 head（即使同一闭包被转换多次），所以多次转换得到的 struct 类型互不相同，不能互相赋值——这是有意的。
- 递归遍历每个捕获的 `ArgNode`，为每个 `RuntimeArgNode` 叶子 `head.add_field(f'_cap{i}', type)`（字段类型见 2.7）。**只数运行时叶子**：编译期叶子不占字段。字段名用合法标识符 `_cap{i}`（而非 `$cap{i}`）：字段名会被 `lower.struct_ctype` 用作 ctypes `_fields_` 的键，`$` 开头无法用属性访问。
- `leaf_bases[i]` = 处理第 `i` 个 capture **之前**的全局字段累计计数（用一个运行计数器累加每个 capture 的运行时叶子数）。一个 capture 有 **0 个运行时叶子**时，`leaf_bases[i]` 取当时的计数，且**必然不被使用**：纯编译期叶子走 `hir.Const` 而不发 `RebuildCapture`；「含 0 个运行时叶子的 `ComptimePtrArg`」虽会发 `RebuildCapture`，但其 `leaf_source` 从不被调用。
- **构造顺序**（生成的 `__call__` 的 `self` 类型需要 struct 类型，而 struct 类型又要能 `get_method('__call__')`）：
  1. `head = StructTypeHead(name_base)`；
  2. 递归 `add_field` 收集字段；
  3. `struct_type = head.specialize(())`；
  4. 生成 `__call__`（其 `self: Ptr[struct_type]`）；
  5. `head.methods['__call__'] = call_fn`。
  （`StructType.get_method` 直接读 `head.methods`，所以「先 `specialize` 再挂方法」没有问题。）

### 2.4 `__call__` 的生成

`__call__` 是一个**普通 `fn.FunctionValue`**，`hir` 由程序化生成：

- **signature** 完整镜像闭包（`positional` 是 `IndexedMap[str, SignatureFormalArg]`，需要重新构造）：
  - `positional[0]` = `self`（名字取 `'self'`），类型 `Ptr[Struct]`；
  - 其余 = 闭包 `hir.signature.positional`（类型 / `is_comptime` / `default_value` / `is_type_value` 原样拷贝）；
  - `varargs` / `kwargs` / `ret_type` / `exceptions` / `callconv` / `may_panic` 原样拷贝；
  - `generic_args` = 闭包的泛型参数。
- `arg_is_ref = (True,) + closure.hir.arg_is_ref`。
- `body` = 生成的 HIR（见 2.5）。
- `name_base` = `f'{closure.name_base}.__call__#{uid}'`。
- `force_inline=False`。

由于它是普通 `FunctionValue`：标准 `HirRunner` 跑它、编译成普通 mir 函数，`_method_of` 能解析它，调用 / 特化 / 内联 / 误差传播全部走既有机制，`Analyser._request_function` **无需特判**，也不需要 `HirRunner` 子类。

### 2.5 `__call__` 的 HIR body

对每个捕获生成一个「重建捕获 place」的 operand：

- 捕获 `ArgNode` 是**纯编译期叶子**（`sval.AnyValue`）→ `hir.Const(value)`。
- 其余（`RuntimeArgNode` / `ComptimePtrArg` / `tuple` 等）→ 构造一条 `hir.RebuildCapture(node, as_copy, field_base, hir.Arg(0))` 指令，`field_base = plan.leaf_bases[i]`（`_generate_call_body` 把生成的指令收进一个 list，末尾并上 `CallInplace` / 收尾后返回 tuple；`HirRunner` 上没有 `_Builder.add`）。

然后：

```
hir.CallInplace(
    callee   = hir.Const(closure),                 # operand 经 as_value 得到 ClosureValue
    args     = hir.CallArgs(
        # 声明形参的 ArgEntry.is_ref 恒为 True：hir.Arg(i) 是该形参的 place（见下面的说明）
        positional=(
            ArgEntry(hir.Arg(1), True), ..., ArgEntry(hir.Arg(npos), True),
            # 闭包声明 *args 时，转发 __call__ 帧的 varargs 槽（第 1+npos 位）
            hir.Spread(hir.Arg(1 + npos)),
        ),
        # 闭包声明 **kwargs 时，槽下标 = 1 + npos + (1 if 闭包有 *args else 0)
        kwargs=(hir.Spread(hir.Arg(kw_index)),),
    ),
    ret      = hir.ResultLoc(),
    captures = (operand_0, ..., operand_n),        # 非 None（即使为空元组；见 2.6）
)
# 收尾（详见下）：有值 / void 都只补 hir.Ret()；Never 不补，让 body 自然走空
```

**收尾只看两态**：有值 / void 都只补 `hir.Ret()`——`Ret` 在函数体里发出 `mir.Ret`：有值时加载 `ResultLoc` 的值得以返回，void 因 `ret_by_value_index` 为 `None` 而返回空；仅当 `ret_type` 是 `EmptyType`（`-> Never`）时才不补 `Ret`（对 Never 闭包的调用已把块 `_cut()` 掉）。**void 不需要补 `hir.StoreVoidRetloc()`**：astgen 的无条件追加之所以安全，是因为它落在 `return` 的 `mir.Ret` 终结符之后、不可达；而生成的 `__call__` 体在 `CallInplace` 之后并无终结符，强行插入会把 `ResultLoc` 覆盖成 `Void`。何况结果位置本就由「对 void 闭包的调用把 `Void` 单元交付进 `ResultLoc`」定型为 `Void`，`Ret()` 已经正确（`rtc_write_back` 等 void 用例——含未标注 / 非内联 / 空体——均已覆盖）。

注意 `args` 的元素是 `ArgEntry[Value]`，而 `hir.Arg(i)` 只是 `Value`，必须用 `ArgEntry(...)` 包一层，且**每个声明形参都用 `ArgEntry(hir.Arg(i), True)`**。理由：`__call__` 的声明形参 `arg_is_ref = closure.hir.arg_is_ref[i-1]`（闭包声明形参一律按值，即 `False`），所以 `hir.Arg(i)` 是该形参的 *place*（`Ptr[T]` 形态）；而 `ArgEntry.is_ref = True` 的语义正是「逻辑上是值 `T`，`value` 为指向它的 `Ptr[T]`」，`_val_to_node` 据此用 `_arg_type_of` 把指针剥回 `T`，与闭包按值 `T` 的形参声明吻合。若误写成 `False`，`_val_to_node` 走值分支会得到 `RuntimeArgNode(Ptr[T])`，形参就会被解成 `Ptr[T]`，与声明不符。（`self` 不是这里的实参——它是 `RebuildCapture` 的 `base`；`__call__` 签名里 `self` 的 `arg_is_ref = True` 是为了让 `hir.Arg(0)` 直接就是接收者指针、在体内当左值用。）

结果直接落进 `__call__` 自己的 result location，由 `finish()` / `_finish_function` 生成对应的 `mir.Ret`——**不需要额外的 `on_return` 写回**。**多返回值**（`tuple[...]` 返回注解）时，`__call__` 往 `ResultLoc` 的写入与 `_call_function_entry`/`_make_runtime_call` 的 result-location 语义一致，沿用即可；实现时用一个多返回值闭包用例钉住（见 3.5）。

### 2.6 调用实参 `CallArgs`、`CallInplace.captures`、`RebuildCapture` 与规范遍历顺序

**调用实参表 `hir.CallArgs` / `hir.Spread`（新增，问题 2 的结构）**：一次调用的实参在 HIR 里是 `CallArgs`——位置项与关键字项各可以是一个普通实参，或一个 `*` / `**` **转发项** `Spread`（对应 `f(a, *v, **kv)`）。splat 贡献可变数量的实参，所以 HIR 不能直接携带 `RawArgList`；只有解释器求值时才摊平成 `RawArgList`：

```python
@dataclass(frozen=True, slots=True)
class Spread:
    # 一个 * / ** 转发项：value 求值得到 ComptimeTuplePtr（*）或 ComptimeDictPtr（**）
    value: Value

@dataclass(frozen=True, slots=True)
class CallArgs:
    positional: tuple[ArgEntry[Value] | Spread, ...]
    kwargs: tuple[tuple[str, ArgEntry[Value]] | Spread, ...]
```

`CallInplace.args` 与 `CallMethodInplace.args` 的类型由 `RawArgList[ArgEntry[Value]]` 改成 `CallArgs`（`RawArgList` 仍是**求值后**的形态）。`astgen` 目前**不产生** `Spread`（不实现 `*`/`**` 调用语法，见 3.2）；只有生成的 `__call__` 体用它。

`eval_call_args` 摊平 `CallArgs`：`Spread` 的 operand 必须是 `ComptimeTuplePtr`（`*`）/ `ComptimeDictPtr`（`**`）——**不兼容值形态** `ComptimeTuple`/`ComptimeDict`（一个是指针一个是值，只支持指针形态，因为 `__call__` 帧里存的就是指针形态）——，其元素 place 一律包成 `ArgEntry(place, True)`。

**`hir.CallInplace` 新增字段** `captures: tuple[Value, ...] | None = None`：与特化层的 `CallSignature.captures` 对齐，补全 HIR 里「调用点显式提供的捕获实参」这一缺口。`None`（`astgen` 构造点的默认值）表示「调用点没有提供捕获」；只有 `as_runtime_closure` 生成的 `__call__` 体把它设成一个元组（可能为空）。只有 `interp._exec_inst` 消费它，`astgen` 构造点加默认字段即不受影响。

`HirRunner._exec_inst` 的 `CallInplace` 分支：

```
captures = None if inst.captures is None else tuple(self.operand(c) for c in inst.captures)
return self.call(self.operand(inst.callee), self.eval_call_args(inst.args),
                 self.operand(inst.ret), captures=captures)
```

`HirRunner.call` 新增 `captures: tuple[InterpVal, ...] | None = None`，原样转发给 `_call_function_entry`（原 `_call_closure` 已并入它，见 3.2）。**`captures` 的统一语义**（对任何 `FunctionValue` 一致）：

- `captures is None` → 调用点没有提供捕获，用被调者自身的 `fn.captures`（见 3.2；普通函数值为 `()`）；
- `captures is not None` → 调用点提供捕获，**覆盖** `fn.captures`（对 `ClosureValue` 也合法——`as_runtime_closure` 的 `__call__` 正是如此）；
- 唯一报错：`captures is not None` 而 callee 不是 `FunctionValue`（builtin / 函数指针）。

因此 `captures is not None` 就是「这次调用由 `as_runtime_closure` 的 `__call__` 发起」的可靠判据（`None` 与空元组的区别），capture-free 的 RTC 也能被识别。

**`*args` / `**kwargs` 转发（问题 2 的解法）**：`__call__` 的形参布局与闭包**逐一对应**，所以 `__call__` 帧的 `arg_values` 里，声明的位置形参在 `[1, 1+npos)`；闭包的 `*args`（`ComptimeTuplePtr`）在第 `1+npos` 位，`**kwargs`（`ComptimeDictPtr`）紧随其后（下标 `1+npos + (1 if 有 *args else 0)`；`_init_args_from_signature` 的顺序：位置形参 → varargs → kwargs）。

生成的 `__call__` 体**直接**用一个 `hir.Spread(hir.Arg(1+npos))`（`**` 用对应下标）把这两个槽转发出去（见 2.5）：`eval_call_args` 摊平后，多余的位置实参经 `bind_arg_pos` 重新绑回闭包自己的 `*args`，关键字实参绑回它的 `**kwargs`。因此**不再需要**在 `HirRunner.call` 里窥探当前帧、也不依赖 captures 触发转发——转发完全表达在 `CallArgs` 里，capture-free 的 RTC 也自然正确。

**新指令 `hir.RebuildCapture`**（**不使用 `Any` 标注**）：

```python
class RebuildCapture(Inst):
    """从运行时闭包 struct 的字段重建一个捕获 place
    （见 std.core.as_runtime_closure）。base 是 struct 的地址。"""
    node: ArgNode          # 该捕获的 ArgNode 树，原样来自 _val_to_node
    as_copy: bool
    field_base: int        # 该捕获第一个叶子在 struct 字段里的下标
    base: Value            # hir.Arg(0)
```

（`ArgNode` 从 `.fn` 导入；`hir.py` 已有 `from .fn import ...`，追加 `ArgNode` 即可。）

解释器执行时调 `HirRunner._rebuild_capture(node, as_copy, field_base, base)`，其实现**镜像 `_init_ptr_target` / `_init_container_target`**，但运行时叶子的来源从「MIR 形参」换成「struct 字段 place」。为复用既有结构逻辑，给 `_init_arg_node` / `_init_ptr_target` / `_init_container_target` / `_init_value` 各加一个可选回调
`leaf_source: Callable[[ArgNode], InterpVal] | None`：

- `None`（默认）= 现有行为（消费 MIR 形参、`mir.Param`）；
- 非空 = 在每个 `RuntimeArgNode` 处直接返回 `leaf_source(node)`（该 place 即捕获/字段的 place，不再包槽）。

`_rebuild_capture` 用一个按规范顺序推进的 `leaf_source`（按 `field_base + 序号` 取字段 place），对每个运行时叶子返回一个 place：

- 顶层捕获、`as_copy=False` → `self.load(field)`（取回指针，作为指向被捕获存储的 place）；
- 顶层捕获、`as_copy=True` → `field`（字段地址，即持有该值的 place）；
- 嵌套叶子 → 用与该叶子**原本**的重建方式一致的方式，把字段值还原成一个持有该值的 place（镜像 `_init_ptr_target` 里对应分支产出的 place 形态）。

```
def _rebuild_capture(self, node: ArgNode, as_copy: bool, field_base: int, base: InterpVal) -> InterpVal:
    index = field_base
    def leaf_source(leaf: ArgNode) -> InterpVal:
        nonlocal index
        i = index; index += 1
        field = self.field_index_addr(base, _index_value(i))
        if leaf is node:
            return field if as_copy else self.load(field)
        return self._leaf_place_from_field(leaf, field)
    return self._init_arg_node(node, [], True, leaf_source)
```

**正确性判据**：重建出的 place 必须让 `_val_to_node(place, False)` 还原出与原始捕获**等价**的 `ArgNode` 形状（这样 callee 侧 `_init_ptr_target` 得到的 place 与原捕获 place 等价）。实现时用一个 round-trip 断言/测试钉住（见 3.5），`_leaf_place_from_field` 的每支形态都按这条判据定。

**`_rebuild_capture` 需要在 `leaf_source` 非空时直接返回它给的 place**（不再包槽）：即把 `_init_arg_node` 的 `RuntimeArgNode` 分支写成「`leaf_source is not None` → `return leaf_source(node)`」。

**`as_copy` 在哪处理**：`_init_arg_node` / `_init_ptr_target` / `_init_container_target` / `_init_value` **本身不感知 `as_copy`**——它们只在遇到每个 `RuntimeArgNode` 叶子时调用一次 `leaf_source`。`as_copy` 被折叠进 `_rebuild_capture` 构造的那一个闭包 `leaf_source` 里：

- 叶子是**顶层捕获节点**（`leaf is node`）时按 `as_copy` 二选一：`as_copy=False` → `self.load(field)`（取回指针，作为指向被捕获存贮的 place）；`as_copy=True` → `field`（字段地址，即持有该值的 place）。
- 其余（嵌套在 `ComptimePtrArg` 内容里的）叶子一律 `_leaf_place_from_field(leaf, field)`，与 `as_copy` 无关（内容本来就是每次调用复制的）。

用 `is` 而不是 `==` 判定「顶层」是刻意的：`ArgNode` 是 frozen dataclass，值相等的两个节点可能是不同对象。

辅助 `HirRunner._leaf_place_from_field(self, leaf: RuntimeArgNode, field: InterpVal) -> InterpVal`：把字段里存的值还原成「持有该值的 place」。它**只会**对 `RuntimeArgNode` 叶子被调用（`leaf_source` 只在 `RuntimeArgNode` 分支触发），嵌套叶子恒为按值（`is_ref=False`），所以实现就是（与 `_init_ptr_target` 的编译期叶子分支同一形态）：

```
place = self._declared_comptime_place(leaf.type)
self.store(place, self.load(field))
return place
```

**规范遍历顺序**（收集 / 构造 / 重建三处必须一致；修正后与 `_init_ptr_target`（重建侧）和 `convert_content`（发射侧）都一致）：

| `ArgNode` | 叶子顺序 |
| --- | --- |
| `RuntimeArgNode` | 1 个叶子 |
| `ComptimePtrArg(_, content)` | 递归 `content`（指针本身不占叶子） |
| `tuple(children)` | 依序 |
| `frozendict(items)` | 依序 |
| `CompoundArgNode(OptionType, (tag, payload))` | 先 tag 后 payload |
| `CompoundArgNode(TaggedUnionType, (tag, payload))` | 先 tag 后 payload |
| `CompoundArgNode(聚合 / complex, elems)` | 依序 |
| 编译期叶子 | 0 个叶子 |

> 注意（`Option` 的 tag/payload 顺序）：`_init_container_target` 的 Option 分支目前是**先算 payload、再把 tag 交给构造实参**（`payload_place = ...` 的语句在 `return ComptimeOptionPtr(self._init_value(tag, ...), payload_place)` 之前），于是消费顺序是 payload→tag；而 `convert_content` 的 Option 分支是 tag→payload，**同一个 `_init_container_target` 的 tagged union 分支也是 tag→payload**。也就是说 Option 分支是唯一不一致的一支，属于既有实现的疏漏。
>
> 本方案统一采用 **先 tag 后 payload**（与 union 分支、`convert_content`、以及 `elems` 元组 `(tag, payload)` 的书写顺序一致），并**顺手修正 `_init_container_target` 的 Option 分支**（见 3.2），使其先算 tag 再算 payload。对**有效程序**该修正行为不变：现有测试里 Option 的 tag 恒为编译期叶子（不占 MIR 实参、也不占字段），两种顺序得到完全相同的字段/实参下标。需要说明的是，**嵌套的运行时-tag option/union 由 `_val_to_node` 的 place 分支整体物化成单个 `RuntimeArgNode`**（tag 只在运行时可知时不拆 tag+payload，见 3.2），因此 `convert_content` 的 Option/union 分支只在编译期 tag 时才被走到，它那里的 tag 恒为编译期值。收集与构造两处的递归必须照此（tag→payload）实现。

### 2.7 `as_copy` 的取值规则（load / 不 load）

**核心不变量**：一个 `RuntimeArgNode` 对应的字段，存的是「它作为 MIR 实参时会被传出去的那个值」，只有**顶层捕获**在 `as_copy=True` 时额外下探一层：

| 位置 | `as_copy=False` | `as_copy=True` |
| --- | --- | --- |
| 顶层捕获（`node.type = Ptr[T]`，是一个 *place*） | 字段类型 `Ptr[T]`，直接存该 place 本身（那个**按值传的运行时指针**），**不 load** | 字段类型 `T`，存 `load(place)`（被复制的内容） |
| `ComptimePtrArg` 内容里的叶子 | 字段存 `load(leaf_place)`；若该叶子本身就是按值传的运行时值（甚至是指针值），照存其值，**不再 deref** | 同左（`ComptimePtrArg` 内容本来就是每次调用复制） |
| 纯编译期叶子 | 不落字段，烘进 `__call__` | 同左 |

重建侧相应地（见 2.6 的 `leaf_source`）：
- 顶层 `as_copy=False` → 操作数 = `self.load(field)`（取回指针，作为指向被捕获存储的 place）；
- 顶层 `as_copy=True` 与所有嵌套叶子 → 操作数 = `field`（字段地址，即持有该值的 place）。

> 判定「是否需要 load」不看类型像不像指针，而是**跟着 place 的形态走**：顶层捕获的 `InterpVal` 是被捕获存储的 place，`ComptimePtrArg` 内容则沿 `ComptimeBox.value` / `ComptimeAggregatePtr.ptrs` / `ComptimeTuplePtr.values` / `ComptimeDictPtr.values` / option-union 的 tag+payload 下降，与 `_init_ptr_target` 的往返严格对应。

### 2.8 构造侧的取叶：镜像 `convert_content`

构造 struct 时的「按 ArgNode 树递归、把叶子值写进字段」不是一个新发明的遍历，而是 **`convert_content`（`_make_runtime_call` 内，发射侧）的镜像**——`convert_content` 的 docstring 已声明它是 `_init_ptr_target` 的镜像、「runtime leaves are consumed in the same depth-first order」。因此**只要构造侧也镜像 `convert_content`，发射 / 构造 / 重建三方顺序就由「互相镜像」自动一致**（2.6 的规范遍历顺序表只是这条镜像关系的注脚，不再是需要人工对齐的三份实现）。

新增两个私有生成器（`HirRunner`）：

```python
def _capture_leaf_values(self, place: InterpVal, node: ArgNode, as_copy: bool) -> Iterator[InterpVal]:
    # place：该 capture 的 place；node：它的 ArgNode。顶层 node 只可能是
    # RuntimeArgNode（运行时捕获）/ CompPtrArg / 编译期叶子（捕获 place 永远是
    # 指针形态或编译期值，见 2.9）。
    if isinstance(node, RuntimeArgNode):
        # as_copy=False：存该 place 本身（按值传的运行时指针）；True：存它指向的内容
        yield self.load(place) if as_copy else place
        return
    if isinstance(node, ComptimePtrArg):
        yield from self._content_leaf_values(place, node.content)
        return
    # 编译期叶子：0 个叶子
    assert not isinstance(node, (tuple, frozendict, CompoundArgNode))

def _content_leaf_values(self, place: InterpVal, node: ArgNode) -> Iterator[InterpVal]:
    """逐条镜像 ``convert_content``（运行时叶子处产 ``self.load(place)``）。"""
    if isinstance(node, RuntimeArgNode):
        yield self.load(place)                       # convert_content: self.load(arg.value)
        return
    if isinstance(node, ComptimePtrArg):
        yield from self._content_leaf_values(self.load(place), node.content)
        return
    if isinstance(node, tuple):                      # place: ComptimeTuplePtr
        assert isinstance(place, ComptimeTuplePtr)
        for child, sub in zip(node, place.values):
            yield from self._content_leaf_values(sub, child)
        return
    if isinstance(node, frozendict):                 # place: ComptimeDictPtr
        assert isinstance(place, ComptimeDictPtr)
        for key, child in node.items():
            yield from self._content_leaf_values(place.values[key], child)
        return
    if isinstance(node, CompoundArgNode):
        if isinstance(node.container_type, sval.OptionType):
            assert isinstance(place, ComptimeOptionPtr)
            # tag 恒为编译期叶子（不产），故略过 tag；零尺寸 child 没有 payload 存储
            if not node.container_type.child.is_zst():
                yield from self._content_leaf_values(place.payload_ptr, node.elems[1])
            return
        if isinstance(node.container_type, sval.TaggedUnionType):
            assert isinstance(place, ComptimeTaggedUnionPtr)
            if _union_variant_type(node.container_type, place.tag).get_unit_value() is None:
                yield from self._content_leaf_values(place.payload_ptr, node.elems[1])
            return
        assert isinstance(place, ComptimeAggregatePtr)
        for child, sub in zip(node.elems, place.ptrs):
            yield from self._content_leaf_values(sub, child)
        return
    # 编译期叶子：不产
```

`_fill_runtime_closure(plan, dest)` 就是：对每个 capture 调 `_capture_leaf_values(capture_place, capture_node, plan.as_copy)`，把依次得到的值 `self.store(field_index_addr(dest, _index_value(i)), value)`，`i` 从 0 全局递增（顺序即 2.6 的规范遍历顺序）。收集侧 `_collect_capture_fields` 与它**逐支镜像**（含「略过 option 的编译期 tag、零尺寸 child 不产 payload」这一支），因此二者产出的字段数必然相等；万一不等（只可能是捕获了嵌套的运行时 tag，见 `pending-problems` #1），`_fill_runtime_closure` 会报一条带计数的 `CompileError`，而不是裸断言失败。

### 2.9 显式不变量

两条把规格里隐含的假设写死、防止未来回归：

1. **顶层运行时捕获一定是指针 place。** 捕获 place 只可能来自 `operand`（`hir.Arg` / `hir.Closure` / 槽），而 `_init_args_from_signature` 等产出的运行时参数值都是 `RuntimeVal(_, Ptr[_])` 或 `Comptime*Ptr` 形态，故 `_val_to_node` 值分支末尾「值形态 → `RuntimeArgNode(type)`」的兜底分支对**捕获**不可达。于是在收集字段、构造、`_rebuild_capture` 三处对顶层 `RuntimeArgNode` 断言：
   ```python
   assert isinstance(node.type, sval.PointerType), f'a runtime capture must be a place, got {node.type}'
   ```
   字段类型据此为 `Ptr[T]`（`as_copy=False`）或 `T = node.type.elem`（`as_copy=True`，见 2.7）。若将来真出现非指针顶层节点，会在断言处立刻暴露，而不是在重建侧静默错位。

2. **RTC struct 必须落在运行内存。** `as_runtime_closure` 落 `ret` 时，若 `ret` 是未提交的 `PendingSlot`，一律改用 `_defer_ptr_convertion(ret, plan.struct_type)` 写出（见 2.2 步 4）：该 action 的 `info()` 恒为 never-inline，从而把槽强制到 `alloca`（内存）分支。目的地**不一定**是新 `Alloca`——变量槽由其后的 `CommitSlot` 解析，而结构体构造的字段由外层 `finish_struct` 绑到字段地址；因此**不能**假定 `ret.committed is None` 且 `len(ret.stores) == 0`（后者在「同一结果槽被多个分支/构造写入」时也不成立）。只要 `self` 拿到的是 struct 本身的可寻址地址，`as_copy=True` 的字段写回就能随 struct 存活。补一条用例（同一 struct 连续调用两次、`as_copy=True`、断言状态保留）钉住。

---

## 3. 详细修改清单

### 3.1 `spy/compiler/hir.py`

**修改导入**：`from .fn import ArgEntry, ArgNode, ClosureFunction, frozendict`（追加 `ArgNode`；`RawArgList` 不再使用，移除）。

**新增 `Spread` / `CallArgs`**（放在 `CallInplace` 附近）：
```python
@dataclass(frozen=True, slots=True)
class Spread:
    # 一个 * / ** 转发项：value 求值得到 ComptimeTuplePtr（*）或 ComptimeDictPtr（**）
    value: Value

@dataclass(frozen=True, slots=True)
class CallArgs:
    # 按源码顺序：位置项（实参或 * splat）与关键字项（name=arg 或 ** splat）
    positional: tuple[ArgEntry[Value] | Spread, ...]
    kwargs: tuple[tuple[str, ArgEntry[Value]] | Spread, ...]
```

**修改 `CallInplace`**（现 L318-329）：`args` 改类型 + 新增 `captures`
```python
class CallInplace(Inst):
    callee: Value
    args: CallArgs          # was RawArgList[ArgEntry[Value]]
    ret: Value
    # 调用点显式提供的捕获实参（与 fn.CallSignature.captures 对齐）：
    # 每个元素求值得到一个捕获 place。None（astgen 构造点的默认值）表示
    # 调用点没有提供捕获；只有 as_runtime_closure 生成的 __call__ 体设为元组。
    captures: tuple[Value, ...] | None = None
```
同步更新 docstring（`args` 是 `CallArgs`；`captures` 是调用点提供的捕获实参，`None` 表示没有提供）。

**修改 `CallMethodInplace`**（现 L300-315）：`args: RawArgList[ArgEntry[Value]]` → `args: CallArgs`。

**边界约定（`CallArgs` 只存在于 HIR 层）**：`hir.CallArgs` 只出现在 `hir.CallInplace.args` / `hir.CallMethodInplace.args` 和 `astgen._gen_arglist` 的返回；一旦进入解释器，`eval_call_args` 是**唯一桥**，之后 `call` / `call_method` / `_call_class_method` / `_make_runtime_call` / `_provided_types` / `bind_arg_pos` **继续用 `RawArgList`，签名一律不变**。`hir.py` 中 `RawArgList` 仅这两处引用，可删导入。

**新增 `RebuildCapture`**（放在 `MakeClosure` 附近）：
```python
class RebuildCapture(Inst):
    node: ArgNode
    as_copy: bool
    field_base: int
    base: Value
```

### 3.2 `spy/compiler/fn.py`、`spy/compiler/astgen.py` 与 `spy/compiler/interp.py`

**`InterpVal → ArgNode` 的唯一入口 `HirRunner._val_to_node(val, is_ref)`**（`is_ref` 表明 `val` 是值还是一个 *place*）

原来分散在 `_provided_node`（一个实参）/ `_content_node`（一个编译期指针的内容，带一个冗余的 `type` 参数）/ `_node_of`（一个值，或「当作值用的 place」）里的编码逻辑合并为一个函数，`_node_of` 被删除：

- `is_ref=False` 分支就是原 `_node_of`：编译期值取自身；编译期指针 → `ComptimePtrArg`（其内容递归 `is_ref=True`）；运行时值 → `RuntimeArgNode`；tuple/dict → 元素树；`ComptimeOptionPtr`/`ComptimeTaggedUnionPtr` 的 tag 只在运行时可知时整体物化成 `RuntimeArgNode(ptr_type)`。调用点由 `_provided_node(arg)` 改为 `_val_to_node(arg.value, arg.is_ref)`。
- `is_ref=True` 分支是原 `_content_node` 的结构（聚合逐字段摊平成 `CompoundArgNode`、tuple/dict 逐元素、option/union 先 tag 后 payload）加上原 `_provided_node(is_ref=True)` 的叶子处理（编译期 place 取其值、运行时 place 剥一层指针后 `RuntimeArgNode`）；`type` 参数删除，类型由 `_place_type(place)` 现场取。option/union 的 tag 只在运行时可知时，这里同样整体物化成 `RuntimeArgNode(OptionType/TaggedUnionType)`（即原 `_provided_node(is_ref=True)` 对它的行为，也是发射侧不会去 `load` 一个布尔/整数 tag 值的关键，见 2.6 的注）。

发射 / 重建 / 收集三方的镜像关系不变：`_init_ptr_target` 的 place 与原捕获 place 等价、`convert_content` 与 `_content_leaf_values` 与 `_collect_capture_fields` 逐支镜像。

**修改 `spy/compiler/astgen.py`：`_gen_arglist` 返回 `hir.CallArgs`**
- `_gen_arglist`（现 L1632-1638）返回类型改为 `hir.CallArgs`：`positional` 是各实参的 `ArgEntry`，`kwargs` 为 `tuple((kw.arg, entry) for kw in keywords if kw.arg is not None)`（**不产生 `Spread`**；`*x` 仍走 `_gen_expr` 报错、`**kw` 仍被忽略——既有行为，本任务不改）。
- `_gen_for` 两处 `RawArgList((), frozendict())`（现 L668 / L681）改为 `hir.CallArgs((), ())`。
- `astgen.py` 的 `RawArgList` 导入若不再使用则移除。

**修改 `spy/compiler/fn.py`：`FunctionValue` 新增 `captures`**
- `FunctionValue.__init__(self, name_base, hir, force_inline: bool = False, captures: tuple[Any, ...] = ())`，存 `self.captures`（普通函数值为空元组）。
- `ClosureValue.__init__` 改为 `super().__init__(name_base, hir_ir, force_inline=fn.force_inline, captures=captures)`，不再自己存 `captures`；`closure_fn` 仍保留。
- 身份语义不变（`__eq__`/`__hash__` 仍按对象身份）。

**在 `HirRunner` 上新增计数器**：`self._runtime_closure_counter`（在 `__init__` 里初始化为 0），为生成的匿名 struct / `__call__` 的 `name_base` 提供 uid（见 2.3）；**不用全局变量**。

**新增 `RuntimeClosurePlan`（dataclass）**：字段见 2.3。

**新增 `HirRunner._build_runtime_closure(self, closure: ClosureValue, capture_places: tuple[InterpVal, ...], as_copy: bool) -> RuntimeClosurePlan`**
- 按 2.3 的构造顺序：用 `capture_places` 经 `_val_to_node` 收集每个捕获的 `ArgNode` 与字段（`captures`/`capture_places` 一并存进 plan）→ 建 head → `head.specialize(())` 得 `struct_type` → 建 `__call__`（其 `self: Ptr[struct_type]`）→ 挂到 `head.methods['__call__']`。

**新增 `HirRunner._collect_capture_fields(self, node: ArgNode, as_copy: bool, top: bool, out: list[sval.Type]) -> None`**
- 按 2.6 的规范顺序递归，给出每个运行时叶子的字段类型（见 2.7）。与 `_content_leaf_values`（2.8）逐支镜像（含略过 option 的编译期 tag 与零尺寸 option payload），保证字段数与取叶数一致。

**新增 `HirRunner._build_call_function(self, closure: ClosureValue, struct_type: sval.StructType, plan: RuntimeClosurePlan) -> FunctionValue`**
- 建镜像 signature（2.4）+ 生成 body（2.5）→ `FunctionIR` → `FunctionValue`。

**新增 `HirRunner._generate_call_body(self, plan: RuntimeClosurePlan, closure: ClosureValue) -> tuple[hir.Inst, ...]`**
- 每个捕获一个 operand（`hir.Const` 或 `hir.RebuildCapture`），末尾 `hir.CallInplace(args=hir.CallArgs(...), captures=...)` + 收尾（2.5）。
- `args` 的构造（见 2.5）：声明形参用 `ArgEntry(hir.Arg(i), True)`；闭包声明 `*args` 时追加 `hir.Spread(hir.Arg(1 + npos))`；声明 `**kwargs` 时 `kwargs=(hir.Spread(hir.Arg(1 + npos + (1 if 有 *args else 0))),)`。

**新增 `HirRunner._builtin_as_runtime_closure(self, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult`**
- 2.2 的步骤 1~6。

**新增 `HirRunner._capture_leaf_values(self, place: InterpVal, node: ArgNode, as_copy: bool) -> Iterator[InterpVal]` 与 `HirRunner._content_leaf_values(self, place: InterpVal, node: ArgNode) -> Iterator[InterpVal]`**
- 2.8 的取叶器：`convert_content`（发射侧）的镜像。顺序与 `convert_content` / `_init_ptr_target` 完全一致（由镜像关系保证），无需人工对齐。

**新增 `HirRunner._fill_runtime_closure(self, plan: RuntimeClosurePlan, dest: InterpVal) -> None`**
- 对每个 capture 调 `_capture_leaf_values(capture_place, capture_node, plan.as_copy)`，依次 `store(field_index_addr(dest, _index_value(i)), value)`，`i` 全局递增（见 2.8）。

**新增 `HirRunner._builtin_as_runtime_closure(self, args: RawArgList[ArgEntry[InterpVal]], ret: InterpVal) -> PollResult`**
- 2.2 的步骤 1~6。其中步骤 1 自行规范化参数：`closure` 取第 1 个位置实参或关键字 `closure=`；`as_copy` 取第 2 个位置实参或关键字 `as_copy=`，未提供时默认 `False`；其余位置实参 / 未知关键字 / 重复给 `as_copy` 一律报错。步骤 4 用 `_defer_ptr_convertion` 把 struct 直接写进目的地（见 2.2/2.9）。

**新增 `HirRunner._rebuild_capture(self, node: ArgNode, as_copy: bool, field_base: int, base: InterpVal) -> InterpVal`**
- 2.6 的实现（用按规范顺序推进的 `leaf_source` 调用 `_init_arg_node`）。对顶层 `RuntimeArgNode` 断言 `isinstance(node.type, sval.PointerType)`（见 2.9）。

**新增 `HirRunner._leaf_place_from_field(self, leaf: RuntimeArgNode, field: InterpVal) -> InterpVal`**
- 把字段里存的值还原成「持有该值的 place」（嵌套叶子用）；实现见 2.6（`_declared_comptime_place(leaf.type)` + `store(..., load(field))`）。只会收到 `RuntimeArgNode`。

**修改 `HirRunner._init_arg_node` / `_init_ptr_target` / `_init_container_target` / `_init_value`**（现 L1801 / L1874 / L1923 / L1957）
- 各加可选参数 `leaf_source: Callable[[ArgNode], InterpVal] | None = None`，并在递归中透传；`_init_value` 把它转交给它内部的 `_init_arg_node` 调用。
- 在 `RuntimeArgNode` 分支：`leaf_source is not None` 时直接 `return leaf_source(node)`；否则维持现状。
- 默认 `None` 时行为完全不变（所有既有调用点不受影响）。
- **顺带修正 `_init_container_target` 的 Option 分支**：把 tag 提到 payload 之前计算（先 `tag_value = self._init_value(tag, mir_args, leaf_source)`，再算 `payload_place`），与同函数的 tagged union 分支、以及 `convert_content` 的 Option 分支一致（见 2.6 的说明）。对有效程序此改动行为不变（现有测试里 Option 的 tag 恒为编译期叶子）；嵌套的运行时-tag option 目前不被支持（发射侧 `convert_content` 就会失败），完整支持不在本任务范围（见 `pending-problems.md`）。

**修改 `HirRunner.call`**（现 L5749）
- 签名加 `captures: tuple[InterpVal, ...] | None = None`（`_exec_inst` 传入；普通调用为 `None`，生成的 `__call__` 为元组）。
- `ClosureValue` / `FunctionValue` 分支都改调 `_call_function_entry(..., captures=captures)`（`ClosureValue` 是 `FunctionValue` 子类，可合并判定）；`BoundMethod` 分支同它、额外带 `generic_var_values`。
- 在函数指针分支前插入 struct `__call__` 分派（见 2.1）。
- 唯一报错：`captures is not None` 而 callee 不是 `FunctionValue`（builtin / 函数指针）。
- **不再有 `*args`/`**kwargs` 转发逻辑**：转发已表达在 `CallArgs` 的 `Spread` 里（见 2.6），本函数不再窥探当前帧。

**修改 `HirRunner._method_of`**（现 L7727）
- `struct.get_method(name)` 返回的条目**已经是 spy 值**（`fn.FunctionValue`，即生成的 `__call__`）时直接返回它，跳过 `resolve_global`；返回 Python 对象（普通方法）时维持原逻辑（`resolve_global`，泛型 struct 再包 `BoundMethod`）。
- 不改 `is_static_method`、`_resolve_method`、`call_method` 的既有分支。

**修改 `HirRunner._call_function_entry`（并入原 `_call_closure`）**（现 L8018；原 `_call_closure` 在 L8085）
- 合并后签名为 `(self, fn, args, ret, generic_var_values=None, on_return=None, captures=None, catch_unwind=False)`；调用点一律**关键字**传 `on_return`/`captures`/`catch_unwind`，避免位置错位。
- `_builtin_catch_unwind` 改调本函数（`catch_unwind=True`）；`call()` 里 `ClosureValue`/`FunctionValue` 两支合并到本函数；`BoundMethod` 支额外带 `generic_var_values`。
- 统一 captures（见 2.6）：`effective = captures if captures is not None else fn.captures`；`fn.captures` 见 3.2 的 `FunctionValue.captures`（默认 `()`）。
- 内联路径用 `closure_values = self._inline_capture_values(fn, effective)`；编译路径用 `effective` 构造 `capture_args`，并在非空时 `replace(call_sig, captures=...)`。
- `ClosureType` 实参检查**仅在 `isinstance(fn, ClosureValue)` 时**做（行为不变）。
- 保留 `generic_var_values`/`substitute_type_vars`（BoundMethod 用）与 `_request_function` 的第 5 参。
- 删除独立的 `_call_closure`。

**修改 `HirRunner._inline_capture_values`**（现 L8136）
- 签名改为 `(self, fn: FunctionValue, captures: tuple[InterpVal, ...])`，按 `captures` 物化（commit pending slot）。

**`HirRunner._init_closure_captures` 改名 `_init_capture_values`**（现 L1981；纯改名 + 文档更新，行为不变）——捕获已是调用路径的通用概念，不再专属闭包。

**修改 `HirRunner._call_builtin`**（现 L5797）
- 增加分派：`if fn.name == 'as_runtime_closure': return self._builtin_as_runtime_closure(args, ret)`。

**`HirRunner.operand_arglist` → `HirRunner.eval_call_args(self, args: hir.CallArgs) -> RawArgList[ArgEntry[InterpVal]]`**（现 L7856）
- 摊平 `CallArgs`：位置项逐个求值（`Spread` 走 `_spread_positional` 展开），关键字项并入（`Spread` 走 `_spread_kwargs`；键重复报错）。
- 新增 `HirRunner._spread_positional(self, place: InterpVal) -> tuple[ArgEntry[InterpVal], ...]`：`place`（经 `_shallow_normalize`）必须是 `ComptimeTuplePtr`，元素包成 `ArgEntry(place, True)`；否则报错（**不兼容** `ComptimeTuple`）。
- 新增 `HirRunner._spread_kwargs(self, place: InterpVal) -> frozendict[str, ArgEntry[InterpVal]]`：`place` 必须是 `ComptimeDictPtr`，条目包成 `ArgEntry(place, True)`；否则报错（**不兼容** `ComptimeDict`）。

**修改 `HirRunner._exec_inst`**（现 L2632）
- `case hir.CallInplace()`：`captures = None if inst.captures is None else tuple(self.operand(c) for c in inst.captures)`；`call(self.operand(inst.callee), self.eval_call_args(inst.args), self.operand(inst.ret), captures=captures)`。
- `case hir.CallMethodInplace()`：`call_method(self.operand(inst.base), inst.name, self.eval_call_args(inst.args), self.operand(inst.ret))`。
- 新增 `case hir.RebuildCapture(): regs[inst] = self._rebuild_capture(inst.node, inst.as_copy, inst.field_base, self.operand(inst.base))`。

**导入**：追加 `FunctionValue`（若未导入）、`replace`（补 args 时用）。

### 3.3 `spy/std/core.py`

新增内置函数声明：
```python
@builtin_func
def as_runtime_closure[T: Callable](closure: T, as_copy: bool = False) -> T: ...
```
（`Callable` 即 `collections.abc.Callable`，已由 `std/core.py` 顶部 `from ..compiler.dsl import Callable` 引入，无需新增 import；`T: Callable` 把传入的闭包类型原样带出，便于 Python 侧类型检查。**该注解完全不参与编译**：`@builtin_func` 只把名字绑成 `sval.BuiltinFn(name)`，不产生 `Signature`，`astgen` / 解释器都不会读它——返回的是**匿名 struct 值**，`-> T` 只是注解层面的近似。也正因为没有 `Signature`，默认参数不会自动填充（见 2.2）。）

### 3.4 `spy/std/__init__.py`

- 在 `from .core import (...)` 与 `__all__` 中加入 `as_runtime_closure`（保持 `std` 的导出习惯）。

### 3.5 `spy/tests/closures.py`

新增用例（沿用现有 `SpyClosureTest` 风格）：
- `as_copy=False`：捕获参数 / 局部量；运行时 struct 值可被调用，闭包对捕获量的写回可见（按引用）。
- `as_copy=True`：捕获量的值被复制，闭包写回不改变原变量。
- `as_copy=True` 的**跨调用持久性**：同一 RTC struct 连续调用两次，第二次能看到第一次对字段的写回（钉住 2.9 的「struct 必落内存」不变量）。
- `as_copy` 的**关键字形式** `as_runtime_closure(f, as_copy=True)` 与位置形式等价。
- 编译期捕获（`Comptime` 变量 / 编译期聚合 / option / tagged union）被烘进 `__call__`。
- 运行时捕获为标量 / 聚合 / option / tagged union 各一例。
- 非内联闭包（`@syntax.closure(inline=False)`）与内联闭包各一例。
- 泛型闭包、多返回值、抛异常各一例。
- 把 struct 值存进变量 / 结构体字段后再调用。
- **round-trip 测试**：对一个捕获 `x`，构造 `plan` 后模拟「构造 → `_rebuild_capture` → `_val_to_node`」，断言还原出的 `ArgNode` 与该捕获原始 `_val_to_node` 结果等价（覆盖标量 / 聚合 / option / tagged union / tuple / 编译期叶子）。
- 错误用例：`as_copy` 非编译期 bool、参数不是闭包、非闭包 callee 带 captures。
- **单运行时叶子 / ZST**：闭包恰好只有 1 个运行时捕获（该 struct 的 MIR 镜像成字段自身，`mirror_is_a_field`）与 0 个运行时捕获（整个 struct 为 ZST，`to_mir_type` 为 `None`）各一例。
- **带 `*args` / `**kwargs` 的闭包**：转换后再调用，验证 2.5/2.6 的 `Spread` 转发——覆盖内联与非内联、capture-free 带 `*args`（ZST struct）、只有 `**kwargs`（槽在 `1+npos`）、以及 `*args`+`**kwargs` 同时。
- **运行时 tag 的 option（对照）**：确认嵌套的运行时 tag option 在当前实现里被判为不可无损搬运、整体物化成单个运行时指针（因此走不到 `CompoundArgNode(OptionType)` 重建路径）。

### 3.6 `spy/README.md`（可选）

- 在「函数与调用 / 闭包」一节补一段运行时闭包说明，并更新「尚未实现」里「运行期的函数值调用……尚未实现」一条。

---

## 4. 已验证的关键约束

- `HirRunner` 不继承、不新增子类；`as_runtime_closure` 返回 struct 值，调用走通用 `__call__` 重载。
- `Analyser._request_function` 不做任何特判：生成的 `__call__` 是普通 `FunctionValue`，由标准 `HirRunner` 编译。
- **不新建模块、不引入 `Recipe`**：结构复用 `fn.ArgNode`；plan 放 `interp.py`。
- `hir.CallInplace` / `hir.CallMethodInplace` 的构造点是 `astgen._gen_call`（及 `_gen_for` 的空实参），消费点是 `interp._exec_inst`；`args` 由 `RawArgList` 改为 `CallArgs`，`captures` 加默认字段，`astgen` 不产 `Spread`。
- `_init_arg_node` / `_init_ptr_target` / `_init_container_target` / `_init_value` 新增参数默认 `None`，既有调用点零改动；`_init_container_target` 的 Option 分支改为 tag 先（见 2.6/3.2）对有效程序行为不变（tag 恒为编译期叶子）。
- 生成的 `__call__` 体内调用闭包时，每个声明形参的实参恒为 `ArgEntry(hir.Arg(i), True)`：`hir.Arg(i)` 是该形参的 place（`arg_is_ref=False`），`ArgEntry.is_ref=True` 表示「逻辑上是值、`value` 为指针」，与闭包按值形参吻合（见 2.5）；`*args`/`**kwargs` 用 `hir.Spread(hir.Arg(...))` 转发帧里的 varargs/kwargs 槽。
- `as_runtime_closure` 的 struct 直接构造在调用结果位置 `ret` 上（未提交的槽经 `_defer_ptr_convertion` 取延迟地址 + 直接写字段），既不另开临时槽，也不做整 struct 的 `load`/`store` 拷贝；目的地可能是变量槽，也可能是结构体构造的字段（见 2.2/2.9/3.2）。
- **`captures` 是调用路径的一等参数**（`hir.CallInplace.captures` / `HirRunner.call` / `_call_function_entry`）：`None` = 调用点未提供，用被调者自身的 `fn.captures`；非 `None` = 调用点提供并覆盖（对 `ClosureValue` 也合法——`__call__` 正是如此）；只有非 `FunctionValue` 被调者带非 `None` captures 才报错。`_call_closure` 已并入 `_call_function_entry`，`captures` 对任何 `FunctionValue` 统一处理（`is_closure` 只留给 `ClosureType` 检查）。
- `fn.FunctionValue` 新增 `captures: tuple[Any, ...] = ()`（`ClosureValue` 经 `super().__init__` 传入）；普通函数值与既有行为零变化。
- **`*args`/`**kwargs` 转发表达在 `CallArgs` 里**：`eval_call_args` 把 `Spread` 摊平进 `RawArgList`，`bind_arg_pos` 再绑回闭包自己的 `*args`/`**kwargs`；capture-free 的 RTC 自然正确。`Spread` **只支持指针形态**（`ComptimeTuplePtr`/`ComptimeDictPtr`，正是帧里的形态），不支持值形态；转发时元素包成 `ArgEntry(place, True)`，不产生额外拷贝。
- 构造侧的取叶器 `_capture_leaf_values` 严格镜像发射侧 `convert_content`（2.8），构造侧 `_fill_runtime_closure`、发射侧 `convert_content`、重建侧 `_init_ptr_target` 三者顺序由「互相镜像」保证一致（不必人工对齐三份实现）。
- **两条显式不变量**（2.9）：顶层运行时捕获的 `node.type` 恒为指针 `Ptr[T]`（否则断言报错）；RTC struct 恒落在运行内存（未提交的槽经 `_defer_ptr_convertion` 强制落内存）。
- `hir.CallArgs` 只存在于 HIR 层，`eval_call_args` 是进入解释器的唯一桥；`call`/`call_method` 等仍用 `RawArgList`（3.1）。
- 字段名用合法标识符 `_cap{i}`；`as_copy` 同时支持位置实参和关键字 `as_copy=`。
- 新增代码的类型注释一律用具体类型（含向后引用）；`Any` 只在确实必要时使用。

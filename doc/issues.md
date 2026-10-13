# 待解决问题（pending problems）

这里记录在实现各项特性时发现、但**本任务不处理**的顺带问题（多为既有实现里已存在的问题）。每节给出原状、最小复现/后果与（若有）规避办法。

> 说明：此前记录的若干问题（嵌套运行时-tag option/union 的 `ArgNode` 编码、`as_copy=True` 的 ZST 捕获边界、运行期控制流之后的闭包/编译期声明、`comptime()` 标记的元组解包紧跟内联调用后再跟声明、`if <编译期bool> and <编译期bool>:` 不折叠、`Comptime` 元组的元素 place 在运行期控制流里不可写、无类型标注的 `const_array(undefined(), N)` 再赋指针元素挂死、反射类型用于 `ptr_cast(..., Ptr[T])` 命中 `exec_ptr_cast` 断言、编译期聚合目的地上的 `store` 缺少隐式转换、内联 tuple 形参槽被 commit 成编译期 box、内联函数的 `type[T]` 形参在实参以值传入（如指针类型 `Ptr[...]`）时形参槽被定成它的内层类型）已随相关重构或修复解决，已从本文件移除。

---

## 1. 调用语法里的 `*` / `**` 转发尚未实现（astgen）

运行时闭包任务把 HIR 的调用实参改成了 `hir.CallArgs`（支持 `hir.Spread`），但**没有**实现 astgen 的转发语法：

- `astgen._gen_arglist` 现在返回 `hir.CallArgs`，但仍只处理普通参数：`**kw`（`kw.arg is None`）被**静默忽略**（既有行为），`*x`（`ast.Starred`）走 `_gen_expr` 报错。
- 待做：`_gen_arglist` 把 `ast.Starred` 转成 `hir.Spread(<tuple place>)`、把 `**kw` 转成 `hir.Spread(<dict place>)`。
- 注意 `hir.Spread` 只支持**指针形态**（`ComptimeTuplePtr` / `ComptimeDictPtr`），所以 splat 操作数应经 `_as_ref` 归一成一个 place（这样元组/字典表达式也落到 `ComptimeTuplePtr`/`ComptimeDictPtr`，而不是值的 `ComptimeTuple`/`ComptimeDict`）。

---

## 2. 捕获「闭包值」（`ClosureValue`）仍不被支持

`rt-closure-spec.md` 的 1.4 表格把「函数值」列为可捕获并「直接烘进 `__call__`」。但若捕获的是另一个**闭包**（`ClosureValue`）：

- `sval.ClosureType.to_mir_type()` 返回 `None`（闭包值进不了运行时位置）；
- `_call_function_entry` 里的 `_arg_node_types` 检查会抛 `a closure can only be passed to an inline function`。

这与「闭包只能传给 inline 函数」的既有限制一致，但和「对捕获变量没有任何限制」的表述放在一起容易误解。若要支持，需要给被捕获的闭包一条可运行时表示的路径（例如把它包装成带 `__call__` 的运行时 struct，正是本特性对顶层闭包做的事）。

---

## 3. 数组注解的长度不能是（裸名的）编译期局部变量

数组注解里的长度按**名字**解析，不取作用域里的编译期变量，因此用反射得到的长度 `count` 写 `Array[Ptr[Opaque], count]` 会报 `name 'count' is not defined ...`（元素类型为裸名编译期类型 `Array[T, 1]` 同理）。

最小复现：

```python
@func()
def len_bare_name() -> i32:
    comptime()
    count = 2
    slots: Array[Ptr[Opaque], count] = const_array(undefined(), count)  # 报 name 'count'
    return 1
```

规避：改用 `const_array(<运行期值>, count)` 让长度经推断得到（结果是 `Array[T, count]`，元素为运行期 place），不要把它写进注解。

---

## 4. 已经定成编译期的 place 不能再被赋运行时值

一个 `Comptime` place 的存储由写进它的值决定，并在其 `CommitSlot` 时固定：若初值是编译期值（如 `c: Comptime[i32] = 5`），它落成 `ComptimeBox`，之后再在**另一个运行期块**里给它赋一个引用了寄存器的运行时值会被拒绝（否则 box 会持有不支配后续读取的寄存器）。

最小复现：

```python
@func()
def f(x: i32) -> i32:
    c: Comptime[i32] = 5
    y: i32 = 0
    while y < 3:
        c = x + y      # CompileError: cannot store a runtime value into a
                       # compile-time place created in another block
        y = y + 1
    return c
```

规避：让初值就是运行时值（`c: Comptime[i32] = x`，落内存后循环可写），或直接用普通（非 `Comptime`）局部变量。

---

## 5. 前置的编译期循环会让带类型的 `comptime()` 局部变量落成内存，从而不能做编译期元组下标

对一个编译期 `Comptime` 元组，用 `comptime()` 声明的 `while` 循环变量做下标会报 `a tuple element index has to be a compile-time integer`；改用 `unroll()` 的 `for k in range(n)` 则正常。

真正的触发条件**不是** `+=`：只要下标变量 `k` 的声明之前存在一个 `unroll()` 的编译期循环就会失败——用多值调用的结果元组建元组、只在前面补一个（哪怕什么都不做的）编译期循环同样失败；反之，`+=` 元组若不写循环（手写两次 `+=`）则正常。原描述「多值调用没问题、只有 `+=` 元组如此」并不准确。

最小复现：

```python
@func()
def via_while(x: i32) -> i32:
    t: Comptime = ()
    comptime()
    j: usize = 0
    unroll()
    while j < 2:
        t += (x + j,)
        j = j + 1
    comptime()
    k: usize = 0
    s: i32 = 0
    unroll()
    while k < 2:
        s = s + t[k]     # a tuple element index has to be a compile-time integer
        k = k + 1
    return s
```

等价的 `unroll(); for k in range(2): s = s + t[k]` 正常。

根因：`astgen._predeclare` 会把一个**带类型的** `comptime` 局部变量（`comptime(); k: u64 = 0`、`k: Comptime[u64] = 0`）的 `Alloca` 预扫描提升到子块头部（裸 `Comptime` 已被此前「就地声明」的修复绕过，带类型的没有）。于是 `k` 的 `Alloca` 在入口块运行、`PendingSlot.runtime_pos` 记为入口块；而它前面的那个编译期循环结束时，`_exec_break_loop`/`_cut` 会新建一个退出块并让 walk 继续在其中，初始化 `k = 0` 的那条 `Store` 便落在这个退出块里。`_record_pending_store` 里的 `is_inline=self._runtime_position() == slot.runtime_pos and _is_comptime_val(value)` 因两块不等而判为 not inline，而 `PendingSlot.is_inline` 要求所有 store 都 inline，于是 `k` 被物化成内存（`RuntimeVal`/`Alloca`）。到 `t[k]` 时，`_subscript_comptime_tuple` 经 `_arg_value` 读到的是一个运行期 `Load` 而不是编译期整数，遂报错。（`_runtime_position` 的 docstring 声称编译期 `if`/`loop` 不改变运行期位置，但实现直接返回 `self._cur_block`，而编译期循环的退出块确实改变了它——意图与实现有缺口。这与 #4 是同一处 `PendingSlot` 存储判定的两个面。）

规避：编译期下标循环统一用 `for ... in range()` + `unroll()`，不要在 `+=` 元组上用 `while` 下标。

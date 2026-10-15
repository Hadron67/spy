# 待解决问题（pending problems）

这里记录在实现各项特性时发现、但**本任务不处理**的顺带问题（多为既有实现里已存在的问题）。每节给出原状、最小复现/后果与（若有）规避办法。

> 说明：此前记录的若干问题（嵌套运行时-tag option/union 的 `ArgNode` 编码、`as_copy=True` 的 ZST 捕获边界、运行期控制流之后的闭包/编译期声明、`comptime()` 标记的元组解包紧跟内联调用后再跟声明、`if <编译期bool> and <编译期bool>:` 不折叠、`Comptime` 元组的元素 place 在运行期控制流里不可写、无类型标注的 `const_array(undefined(), N)` 再赋指针元素挂死、反射类型用于 `ptr_cast(..., Ptr[T])` 命中 `exec_ptr_cast` 断言、编译期聚合目的地上的 `store` 缺少隐式转换、内联 tuple 形参槽被 commit 成编译期 box、内联函数的 `type[T]` 形参在实参以值传入（如指针类型 `Ptr[...]`）时形参槽被定成它的内层类型、前置声明把带类型的 `comptime` 局部变量的 `Alloca` 提升到入口块而无法做编译期下标、往 `Option[零尺寸类型]` 的构造不置 present、前置声明把同一 block 内被就地声明的编译期变量抢成运行时槽）已随相关重构或修复解决（后二者随「前置声明只覆盖连续 `def`」的改动移除），已从本文件移除。

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

## 6. 互递归 closure 的类型循环依赖（comptime 重构引入的回归）

两个兄弟闭包互相引用（互相捕获）时不再能编译：

```python
@func()
def f(n: i32) -> i32:
    @syntax.closure(inline=False)
    def even(x: i32) -> i32:
        if x == 0:
            return 1
        return odd(x - 1)      # 捕获 odd

    @syntax.closure(inline=False)
    def odd(x: i32) -> i32:
        if x == 0:
            return 0
        return even(x - 1)     # 捕获 even

    return even(n)
```

- **原状**（comptime 重构前）：closure 的值类型由**签名**决定（与捕获无关），互相引用的签名在解析期就已知，因此没有类型层面的循环；`spy.tests.closures.SpyClosureTest.test_mutually_recursive_closures` 通过。
- **现状**（重构后）：closure 的值类型是**含捕获字段的匿名 struct**（见 behaviour §9、task §9）。建 `even$closure` 时要把 `odd`（兄弟 `def` 的槽）作为按引用捕获的字段，字段类型 = `odd$closure` 的 struct 类型；而此时 `odd` 还没建出 → `interp.exec_make_closure` 报
  `cannot capture a value whose type is not known`（`_place_type(capture)` 为 None，因为那个槽还没 `CommitSlot`）。于是该用例失败。
- **可能的修复方向**（尚未决定）：
  - **(a)** 按引用捕获（`move=False`）的字段类型用**指向「前向声明的 struct」的指针**：先建一个空的 `StructTypeHead` 绑在闭包上，等该兄弟闭包建好后再补字段/特化（spy 已有「head 先绑定、字段后加」的自引用机制，见 `dsl._RegisteredClass.get_entry`）。
  - **(b)** 捕获另一个 closure 时，字段类型用它**签名对应的函数类型**（而非它的捕获 struct 类型），读取时再收敛成实际 struct 类型。
  - **(c)** 接受为新限制：互递归 closure 不再支持，改为报错并更新用例。
- 与 #2（捕获闭包值）相关：两者都是「闭包值」的类型/可表示性问题。

---

## 7. 声明的「编译期专属」返回类型无法编译（`-> bytes` / `-> type` 等）

```python
@func()
def f() -> bytes:
    return b'hi'
```

报 `bytes is a compile-time-only type: a runtime location cannot hold it and must declare its type`（`_no_runtime_type`）。

- 触发链：`run_function` 在跑函数体**之前**调 `_materialize_ret_sig` → `_commit_ret_places` → 对 `RetComptime` 叶子调 `_commit_pending_slot(place, node.type)`。该 place 是 `_ret_spec_place` 给的 `alloca(True)`，但此刻它**既没有 store 也没有 declared 类型**，于是 `_commit_pending_slot` 的
  `if val.use_comptime_storage() and (len(val.stores) > 0 or val.declared is not None)` 不成立 → 落到运行时路径 →
  `to_mir_type()` 为 `None` → `_no_runtime_type`。
- 对比：**不写**返回注解时结果位置是**推断**的（`alloca()`），`return 1` / `return (1, 2)` 反而能编译，值也能交付给 Python 调用方（behaviour §7.3）。
- 待做：让 `_commit_ret_places` 的 `RetComptime` 分支直接把它提交成编译期 box（该 place 按构造就是编译期存储）——例如带上声明类型，或走专门的分支。
- 波及：`-> bytes` 目前无用例（`grep` 不到），修好前不要依赖它。要返回类型值可以绕开：`t: Comptime = i32; return t`（不写返回注解）能编译并交给 Python。

---

## 8. 嵌套字段位置的 union 两变体构造仍不被支持

**已修**：`u: U = A(1) if c else B(2)`（以及 `Option[U]` 存储）现在可以了——`interp` 的 `_is_union_delivery` 让**运行时存储**里的
tagged union 每次构造记自己的 `_PendingAggregate`（`field_index_addr` 不复用已记录的 place），各按自己的变体在各自位置交付。

**仍未修**：把一个 union 字段**在构造实参里**用 `if` 表达式填两个不同变体时，字段 place 由 `field_index_addr` 的 `self.alloca()`
建出、**不带声明类型**（尽管 `_type_of(h_slot) = ptr(Holder)` 已经知道字段类型是 `U`），于是构造收尾（`finish_struct`）看不到“存储是 union”：

```python
@struct()
class Holder:
    u: TV_A | TV_B

@func()
def f(c: bool) -> i32:
    h = Holder(TV_A(1) if c else TV_B(2))   # CompileError: a slot is stored with
    ...                                     # incompatible types TV_A and TV_B
```

- 待做：构造的字段 place 应带上基类型声明的字段类型（`field_index_addr` 未提交分支的 `self.alloca()` → `alloca(declared=...)`）。
  这会让**所有**聚合构造的字段 place 提前定型（`committed_type`、编译期存储推断、ZST/DST 处理都受影响），需要单独一轮验证。
- 规避：先构造进一个显式声明类型的变量再传入（`u: TV_A | TV_B = TV_A(1) if c else TV_B(2)`；`h = Holder(u)`）。
- 相关（有意为之）：**编译期存储**的 union 装不下两个变体——一个 `Box` 只有一个值，所以
  `u: Comptime[TV_A | TV_B] = TV_A(1) if c else TV_B(2)`（运行时条件）仍报 `incompatible types`，与 struct 的
  `s: Comptime[S] = S(1) if c else S(2)` 报 `cannot store a runtime value into a compile-time variable` 一致；
  编译期条件会折叠成单次构造，正常。

---

## 9. 编译期 place 的视图可以丢 const（`*const T` → `*T` 被放行）

`sval.coerce_const` 对 `ComptimePtr` 的目标判定只要求 **pointee 相同**（`value.type.elem == target.elem`），**不检查 const 性**，
因此一个 const 视图的编译期 place 可以赋给声明为非 const 的指针位置：

```python
@func()
def take_i32(p: Ptr[i32]) -> i32: ...

@func()
def f() -> i32:
    const_n: ConstComptime[i32] = 5
    return take_i32(ref(const_n))    # ref(ConstComptime) 得到 *const i32，这里丢了 const
```

- **原状**（本轮之前就有）：`interp.HirRunner._coerce` 对编译期指针只检查「目标是某个指针类型」，`sval.coerce_const` 沿用同一判定；
  `spy/tests/` 里有 12 处用例依赖它（`comptime_ptrs.py`、`static.py`、`options.py`、`tagged_unions.py`）。
  运行时路径反而更严：`_convert_inst` 用 `PointerType.is_subtype_of`，`*const T -> *T` 会被拒。
- **后果**：视图类型只影响类型层面，写入仍由 `ComptimePtr.is_writable()`（同时看 box 的 `is_const`）拦住，所以不会产生错误代码；
  但把 const place 的地址交给一个声明为可写的 `*T` 位置，读起来像可以写。
- **更合理的规则**：pointee 相同的前提下只允许「加 const」（即运行时路径的 `is_subtype_of`）；那样上述 12 处用例需要改写
  （把 `Ptr[T]` 形参改成 `ConstPtr[T]` 等）。因为 `ref(ConstComptime)` 传给 `Ptr[T]` 形参是惯用写法，改动会波及用户代码，本轮不动。


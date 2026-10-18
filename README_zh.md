# spy - System Python

`spy` 是一个把 Python 函数 **JIT 编译成机器码**的包。写法和普通 Python 一样——用 `@spy.func()` 装饰一个函数即可注册；首次调用时按实参的类型编译出一个**特化**，之后的每次调用都是原生调用，不再经过 Python。函数体在编译期以具体类型「运行」，因此支持类似 Zig 的 **comptime** 语义（`spy.typeof`、编译期 `if`、泛型、内联普通 Python 函数等）。

函数必须定义在真实源码文件中（编译需要函数的源码，交互式环境里无法使用）。

## 示例

```python
import spy


@spy.func()                     # 注册；首次调用时编译
def add[T](a: T, b: T) -> T:
    return a + b


@spy.func()
def add_u64(a: spy.u64, b: spy.u64) -> spy.u64:
    return a + b


print(add(1, 2))                                  # 编译 add(i64, i64)
print(add(1.5, 2.25))                             # 编译 add(f64, f64)
print(add_u64(spy.as_(2**63 - 1, spy.u64), spy.as_(2, spy.u64)))
```

详细的特性文档见[doc/zh/features.md](doc/zh/features.md)。

## 项目结构

- [doc/](doc/) — 文档
- [spy/](spy/) — Python 包
- [spy/compiler/](spy/compiler/) — 编译器：Python 源码 → 机器码
- [spy/std/](spy/std/) — 标准库
- [spy/tests/](spy/tests/) — 集成测试

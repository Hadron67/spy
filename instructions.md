# 说明
spy - System Python

这是一个将python函数jit编译成机器码的包，通过注解实现。注解有两种：jit模式和aot模式，前者是在函数被调用时编译，后者是立即编译，这种情况下要求函数有类型注释，并且类型是可jit编译的。同时该包还有类似于zig的comptime语义，即可以在编译时进行一些计算。

一个简单例子：
```python
import spy

cache = spy.JitContext()

@cache.jit()
def add[T](a: T, b: T) -> T: # spy.jit注解中，类型注释对jit无影响
    return a + b

@cache.aot() # AOT: 立即用类型注释的类型编译
def add_aot(a: spy.u64, b: spy.u64) -> spy.u64:
    return a + b

@cache.jit()
def add_default[T](a: T, b: T = 0) -> T: # 支持默认参数
    return a + b

def add_inline[T](a: T, b: T) -> T:
    spy.compile_log("add_inline was compiled") # 编译时打印日志，运行时不会有动作
    return a + b

@cache.jit()
def foo(a, b):
    # 编译时运行的代码
    if spy.type(a) == spy.u64 and spy.type(b) == spy.u64:
        # 一个函数中可以调用其他jit函数，最终会被lower成llvm的call指令
        return add_aot(a, b)
    else:
        # 调用非jit函数的时候则将该函数内联并继续分析
        return add_inline(a, b)

print(add(1, 2)) # 将触发编译add(spy.i32, spy.i32)，因为python的int默认映射为`spy.i32`
print(add(1.0, 2.0)) # 触发编译add(spy.f64, spy.f64)
print(add('', '')) # 编译时报错：字符串类型（映射为spy.u8数组）不支持'+'算符
print(add_aot(1, 2)) # ok
print(add_aot(1.0, 2.0)) # 报错：类型不匹配

print(add_default(1, 2))
print(add_default(1)) # 插入默认参数
print(add_default(b = 45, a = 12)) # 支持kwarg

foo(spy.as(1, spy.u64), spy.as(2, spy.u64)) # 被编译为if分支，即add_aot
foo(1, 2) # 打印出"add_inline was compiled"
```

## 一些实现细节
基本流程是先将python ast通过astgen变为无类型的hir，然后运行这个hir得到有类型的mir，然后再lower成llvm。

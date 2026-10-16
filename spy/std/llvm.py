from spy.compiler import decl_func, f32, f64


@decl_func("llvm.sin.f32")
def sin_f32(val: f32) -> f32: ...

@decl_func("llvm.sin.f64")
def sin_f64(val: f64) -> f64: ...

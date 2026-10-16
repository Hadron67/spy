from spy.compiler import decl_func, f32, f64


@decl_func("llvm.sin.f32")
def sin_f32(val: f32) -> f32: ...

@decl_func("llvm.sin.f64")
def sin_f64(val: f64) -> f64: ...

@decl_func("llvm.cos.f32")
def cos_f32(val: f32) -> f32: ...

@decl_func("llvm.cos.f64")
def cos_f64(val: f64) -> f64: ...

@decl_func("llvm.tan.f32")
def tan_f32(val: f32) -> f32: ...

@decl_func("llvm.tan.f64")
def tan_f64(val: f64) -> f64: ...

#ifndef LAI_BACKEND_CPU_BACKEND_H
#define LAI_BACKEND_CPU_BACKEND_H

#include "backend.h"
#include "../core/tensor.h"
#include "../core/simd.h"

namespace lai {

// CPU backend: wraps existing ops:: and simd:: functions
class CpuBackend : public Backend {
public:
    const char* name() const override { return "CPU"; }

    void matvec(TensorView& y, const TensorView& A, const TensorView& x) override {
        ops::matvec_dispatch(y, A, x);
    }

    void rmsnorm(TensorView& y, const TensorView& x, const TensorView& w, f32 eps) override {
        ops::rmsnorm(y, x, w, eps);
    }

    void softmax(TensorView& y, const TensorView& x) override {
        ops::softmax(y, x);
    }

    void rope(TensorView& q, TensorView& k, i32 pos, i32 head_dim, f32 theta) override {
        ops::rope(q, k, pos, head_dim, theta);
    }

    void silu(TensorView& y, const TensorView& x) override {
        ops::silu(y, x);
    }

    void add(TensorView& y, const TensorView& a, const TensorView& b) override {
        ops::add(y, a, b);
    }

    void mul(TensorView& y, const TensorView& a, const TensorView& b) override {
        ops::mul(y, a, b);
    }

    void copy(TensorView& dst, const TensorView& src) override {
        simd::copy_f32(dst.data_f32(), src.data_f32(), dst.numel());
    }
};

// Factory implementation
inline Backend* Backend::create_cpu() {
    return new CpuBackend();
}

} // namespace lai

#endif // LAI_BACKEND_CPU_BACKEND_H

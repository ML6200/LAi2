#ifndef LAI_BACKEND_METAL_BACKEND_H
#define LAI_BACKEND_METAL_BACKEND_H

#ifdef LAI_METAL

#include "backend.h"

namespace lai {

// Metal GPU backend - declaration only
// Implementation in metal_backend.mm (Objective-C++)
class MetalBackend : public Backend {
public:
    MetalBackend();
    ~MetalBackend() override;

    bool init();

    const char* name() const override { return "Metal"; }

    void matvec(TensorView& y, const TensorView& A, const TensorView& x) override;
    void rmsnorm(TensorView& y, const TensorView& x, const TensorView& w, f32 eps) override;
    void softmax(TensorView& y, const TensorView& x) override;
    void rope(TensorView& q, TensorView& k, i32 pos, i32 head_dim, f32 theta) override;
    void silu(TensorView& y, const TensorView& x) override;
    void add(TensorView& y, const TensorView& a, const TensorView& b) override;
    void mul(TensorView& y, const TensorView& a, const TensorView& b) override;
    void copy(TensorView& dst, const TensorView& src) override;
    void sync() override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace lai

#endif // LAI_METAL
#endif // LAI_BACKEND_METAL_BACKEND_H

#ifndef LAI_BACKEND_BACKEND_H
#define LAI_BACKEND_BACKEND_H

#include "../core/types.h"
#include "../core/tensor.h"

namespace lai {

// Abstract compute backend interface
class Backend {
public:
    virtual ~Backend() = default;

    virtual const char* name() const = 0;

    // Core tensor operations
    virtual void matvec(TensorView& y, const TensorView& A, const TensorView& x) = 0;
    virtual void rmsnorm(TensorView& y, const TensorView& x, const TensorView& w, f32 eps) = 0;
    virtual void softmax(TensorView& y, const TensorView& x) = 0;
    virtual void rope(TensorView& q, TensorView& k, i32 pos, i32 head_dim, f32 theta) = 0;
    virtual void silu(TensorView& y, const TensorView& x) = 0;
    virtual void add(TensorView& y, const TensorView& a, const TensorView& b) = 0;
    virtual void mul(TensorView& y, const TensorView& a, const TensorView& b) = 0;
    virtual void copy(TensorView& dst, const TensorView& src) = 0;

    // Synchronize (wait for all async GPU ops to complete)
    virtual void sync() {}

    // Factory methods
    static Backend* create_cpu();

#ifdef LAI_METAL
    static Backend* create_metal();
#endif

    // Auto-detect best available backend
    static Backend* create_best() {
#ifdef LAI_METAL
        Backend* metal = create_metal();
        if (metal) return metal;
#endif
        return create_cpu();
    }
};

} // namespace lai

#endif // LAI_BACKEND_BACKEND_H

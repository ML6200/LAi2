#ifndef LAI_CORE_TENSOR_H
#define LAI_CORE_TENSOR_H

#include "types.h"
#include "allocator.h"
#include "simd.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace lai {

// Forward declaration
class Tensor;

// Tensor view - non-owning reference to tensor data
class TensorView {
public:
    TensorView() : data_(nullptr), shape_(), dtype_(DType::F32) {}

    TensorView(void* data, Shape shape, DType dtype)
        : data_(data), shape_(shape), dtype_(dtype) {}

    void* data() { return data_; }
    const void* data() const { return data_; }

    f32* data_f32() { return static_cast<f32*>(data_); }
    const f32* data_f32() const { return static_cast<const f32*>(data_); }

    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }

    i64 numel() const { return shape_.numel(); }
    i64 dim(i32 i) const { return shape_[i]; }
    i32 ndim() const { return shape_.ndim; }

    // Element access (F32 only)
    f32& at(i64 i) {
        return data_f32()[i];
    }

    const f32& at(i64 i) const {
        return data_f32()[i];
    }

    f32& at(i64 i, i64 j) {
        return data_f32()[i * shape_[1] + j];
    }

    const f32& at(i64 i, i64 j) const {
        return data_f32()[i * shape_[1] + j];
    }

    // Row slice
    TensorView row(i64 i) const {
        return TensorView(
            const_cast<f32*>(data_f32() + i * shape_[1]),
            Shape(shape_[1]),
            dtype_
        );
    }

protected:
    void* data_;
    Shape shape_;
    DType dtype_;
};

// Tensor - owns its data
class Tensor : public TensorView {
public:
    Tensor() : TensorView(), owned_(false) {}

    // Create tensor with shape (allocates memory)
    explicit Tensor(Shape shape, DType dtype = DType::F32)
        : TensorView(nullptr, shape, dtype), owned_(true) {
        size_t bytes = storage_bytes(numel(), dtype);
        // aligned_alloc requires size to be multiple of alignment
        size_t aligned_bytes = ((bytes + 63) / 64) * 64;
        if (aligned_bytes == 0) aligned_bytes = 64;  // Minimum allocation
        data_ = std::aligned_alloc(64, aligned_bytes);
        if (!data_) throw std::bad_alloc();
    }

    // Create from existing data (non-owning)
    Tensor(void* data, Shape shape, DType dtype)
        : TensorView(data, shape, dtype), owned_(false) {}

    ~Tensor() {
        if (owned_ && data_) {
            std::free(data_);
        }
    }

    // Move only
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : TensorView(other.data_, other.shape_, other.dtype_), owned_(other.owned_) {
        other.data_ = nullptr;
        other.owned_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owned_ && data_) std::free(data_);
            data_ = other.data_;
            shape_ = other.shape_;
            dtype_ = other.dtype_;
            owned_ = other.owned_;
            other.data_ = nullptr;
            other.owned_ = false;
        }
        return *this;
    }

    // Allocate in arena (non-owning after creation)
    static Tensor from_arena(Arena& arena, Shape shape, DType dtype = DType::F32) {
        size_t bytes = storage_bytes(shape.numel(), dtype);
        void* data = arena.alloc(bytes);
        if (!data) throw std::bad_alloc();
        return Tensor(data, shape, dtype);
    }

    // Fill with value
    void fill(f32 val) {
        simd::fill_f32(data_f32(), val, numel());
    }

    // Fill with zeros
    void zero() {
        std::memset(data_, 0, storage_bytes(numel(), dtype_));
    }

    bool owns_data() const { return owned_; }

private:
    bool owned_;
};

// ============================================================================
// Tensor Operations
// ============================================================================
namespace ops {

// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
inline void matmul(TensorView& C, const TensorView& A, const TensorView& B) {
    const i64 M = A.dim(0);
    const i64 K = A.dim(1);
    const i64 N = B.dim(1);

    const f32* a = A.data_f32();
    const f32* b = B.data_f32();
    f32* c = C.data_f32();

    // Simple blocked matmul
    constexpr i64 BLOCK = 64;

    // Zero output
    std::memset(c, 0, M * N * sizeof(f32));

    for (i64 i = 0; i < M; ++i) {
        for (i64 k = 0; k < K; ++k) {
            f32 aik = a[i * K + k];
            simd::fma_f32(c + i * N, b + k * N, aik, N);
        }
    }
}

// Matrix-vector multiplication: y = A @ x
// A: [M, K], x: [K], y: [M]
inline void matvec(TensorView& y, const TensorView& A, const TensorView& x) {
    const i64 M = A.dim(0);
    const i64 K = A.dim(1);

    const f32* a = A.data_f32();
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();

    for (i64 i = 0; i < M; ++i) {
        yp[i] = simd::dot_f32(a + i * K, xp, K);
    }
}

// Matrix-vector multiplication: y = A @ x (A is Q4_0, x is f32)
inline void matvec_q4(TensorView& y, const TensorView& A, const TensorView& x) {
    const i64 M = A.dim(0);
    const i64 K = A.dim(1);
    const Q4_0* a = static_cast<const Q4_0*>(A.data());
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();
    const i64 blocks_per_row = K / 32;

    for (i64 i = 0; i < M; ++i) {
        yp[i] = simd::dot_q4_f32(a + i * blocks_per_row, xp, K);
    }
}

// Matrix-vector multiplication: y = A @ x (A is Q8_0, x is f32)
inline void matvec_q8(TensorView& y, const TensorView& A, const TensorView& x) {
    const i64 M = A.dim(0);
    const i64 K = A.dim(1);
    const Q8_0* a = static_cast<const Q8_0*>(A.data());
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();
    const i64 blocks_per_row = K / 32;

    for (i64 i = 0; i < M; ++i) {
        yp[i] = simd::dot_q8_f32(a + i * blocks_per_row, xp, K);
    }
}

// Dispatch matvec based on weight dtype
inline void matvec_dispatch(TensorView& y, const TensorView& A, const TensorView& x) {
    switch (A.dtype()) {
        case DType::Q4_0: matvec_q4(y, A, x); break;
        case DType::Q8_0: matvec_q8(y, A, x); break;
        default:          matvec(y, A, x);     break;
    }
}

// Element-wise addition: y = a + b
inline void add(TensorView& y, const TensorView& a, const TensorView& b) {
    simd::add_f32(y.data_f32(), a.data_f32(), b.data_f32(), y.numel());
}

// Element-wise multiplication: y = a * b
inline void mul(TensorView& y, const TensorView& a, const TensorView& b) {
    simd::mul_f32(y.data_f32(), a.data_f32(), b.data_f32(), y.numel());
}

// Scale: y = a * scale
inline void scale(TensorView& y, const TensorView& a, f32 scale) {
    simd::scale_f32(y.data_f32(), a.data_f32(), scale, y.numel());
}

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
inline void rmsnorm(TensorView& y, const TensorView& x, const TensorView& weight, f32 eps = 1e-5f) {
    const i64 n = x.numel();
    const f32* xp = x.data_f32();
    const f32* wp = weight.data_f32();
    f32* yp = y.data_f32();

    // Compute RMS
    f32 sum_sq = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        sum_sq += xp[i] * xp[i];
    }
    f32 rms = 1.0f / std::sqrt(sum_sq / static_cast<f32>(n) + eps);

    // Normalize and scale
    for (i64 i = 0; i < n; ++i) {
        yp[i] = xp[i] * rms * wp[i];
    }
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
inline void layernorm(TensorView& y, const TensorView& x, const TensorView& weight,
                      const TensorView& bias, f32 eps = 1e-5f) {
    const i64 n = x.numel();
    const f32* xp = x.data_f32();
    const f32* wp = weight.data_f32();
    const f32* bp = bias.data_f32();
    f32* yp = y.data_f32();

    f32 mean = simd::mean_f32(xp, n);
    f32 var = simd::var_f32(xp, n, mean);
    f32 rstd = 1.0f / std::sqrt(var + eps);

    for (i64 i = 0; i < n; ++i) {
        yp[i] = (xp[i] - mean) * rstd * wp[i] + bp[i];
    }
}

// Softmax: y = exp(x - max(x)) / sum(exp(x - max(x)))
inline void softmax(TensorView& y, const TensorView& x) {
    const i64 n = x.numel();
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();

    f32 max_val = simd::max_f32(xp, n);

    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        yp[i] = std::exp(xp[i] - max_val);
        sum += yp[i];
    }

    f32 inv_sum = 1.0f / sum;
    simd::scale_f32(yp, yp, inv_sum, n);
}

// SiLU (Swish): y = x * sigmoid(x) = x / (1 + exp(-x))
inline void silu(TensorView& y, const TensorView& x) {
    const i64 n = x.numel();
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();

    for (i64 i = 0; i < n; ++i) {
        yp[i] = xp[i] / (1.0f + std::exp(-xp[i]));
    }
}

// GELU (approximate): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline void gelu(TensorView& y, const TensorView& x) {
    const i64 n = x.numel();
    const f32* xp = x.data_f32();
    f32* yp = y.data_f32();

    constexpr f32 c1 = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr f32 c2 = 0.044715f;

    for (i64 i = 0; i < n; ++i) {
        f32 x3 = xp[i] * xp[i] * xp[i];
        yp[i] = 0.5f * xp[i] * (1.0f + std::tanh(c1 * (xp[i] + c2 * x3)));
    }
}

// RoPE (Rotary Position Embedding)
inline void rope(TensorView& q, TensorView& k, i32 pos, i32 head_dim, f32 theta = 10000.0f) {
    const i64 half = head_dim / 2;

    for (i64 i = 0; i < half; ++i) {
        f32 freq = 1.0f / std::pow(theta, static_cast<f32>(2 * i) / static_cast<f32>(head_dim));
        f32 angle = static_cast<f32>(pos) * freq;
        f32 cos_val = std::cos(angle);
        f32 sin_val = std::sin(angle);

        // Rotate Q
        f32 q0 = q.at(i);
        f32 q1 = q.at(i + half);
        q.at(i) = q0 * cos_val - q1 * sin_val;
        q.at(i + half) = q0 * sin_val + q1 * cos_val;

        // Rotate K
        f32 k0 = k.at(i);
        f32 k1 = k.at(i + half);
        k.at(i) = k0 * cos_val - k1 * sin_val;
        k.at(i + half) = k0 * sin_val + k1 * cos_val;
    }
}

} // namespace ops
} // namespace lai

#endif // LAI_CORE_TENSOR_H

#ifndef LAI_CORE_SIMD_H
#define LAI_CORE_SIMD_H

#include "types.h"

// Platform detection
#if defined(__AVX2__)
    #define LAI_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define LAI_NEON 1
    #include <arm_neon.h>
#endif

namespace lai {
namespace simd {

// Vector width in floats
#if defined(LAI_AVX2)
    constexpr i32 VECTOR_WIDTH = 8;
#elif defined(LAI_NEON)
    constexpr i32 VECTOR_WIDTH = 4;
#else
    constexpr i32 VECTOR_WIDTH = 1;
#endif

// ============================================================================
// AVX2 Implementation
// ============================================================================
#if defined(LAI_AVX2)

// Dot product of two f32 vectors
inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    i64 i = 0;
    for (; i + 32 <= n; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0),  _mm256_loadu_ps(b + i + 0),  sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    }

    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    for (; i + 8 <= n; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    f32 result = _mm_cvtss_f32(sum128);

    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// Vector-scalar multiply and add: y = a * x + y
inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    __m256 va = _mm256_set1_ps(a);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(y + i, vy);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// Vector add: y = a + b
inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

// Vector multiply: y = a * b
inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

// Scale vector: y = a * scale
inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    __m256 vs = _mm256_set1_ps(scale);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, vs));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

// Sum of vector elements
inline f32 sum_f32(const f32* a, i64 n) {
    __m256 sum = _mm256_setzero_ps();

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(a + i));
    }

    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    f32 result = _mm_cvtss_f32(sum128);

    for (; i < n; ++i) {
        result += a[i];
    }

    return result;
}

// Max of vector elements
inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;

    __m256 vmax = _mm256_set1_ps(a[0]);

    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(a + i));
    }

    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));

    f32 result = _mm_cvtss_f32(max128);

    for (; i < n; ++i) {
        if (a[i] > result) result = a[i];
    }

    return result;
}

// ============================================================================
// NEON Implementation (ARM)
// ============================================================================
#elif defined(LAI_NEON)

inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }

    f32 result = vaddvq_f32(sum);

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    float32x4_t va = vdupq_n_f32(a);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_f32(vy, va, vx);
        vst1q_f32(y + i, vy);
    }

    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(y + i, vaddq_f32(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(y + i, vmulq_f32(va, vb));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    float32x4_t vs = vdupq_n_f32(scale);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(y + i, vmulq_f32(va, vs));
    }

    for (; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

inline f32 sum_f32(const f32* a, i64 n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        sum = vaddq_f32(sum, vld1q_f32(a + i));
    }

    f32 result = vaddvq_f32(sum);

    for (; i < n; ++i) {
        result += a[i];
    }

    return result;
}

inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;

    float32x4_t vmax = vdupq_n_f32(a[0]);

    i64 i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(a + i));
    }

    f32 result = vmaxvq_f32(vmax);

    for (; i < n; ++i) {
        if (a[i] > result) result = a[i];
    }

    return result;
}

// ============================================================================
// Scalar Fallback
// ============================================================================
#else

inline f32 dot_f32(const f32* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void fma_f32(f32* y, const f32* x, f32 a, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

inline void add_f32(f32* y, const f32* a, const f32* b, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

inline void mul_f32(f32* y, const f32* a, const f32* b, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

inline void scale_f32(f32* y, const f32* a, f32 scale, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        y[i] = a[i] * scale;
    }
}

inline f32 sum_f32(const f32* a, i64 n) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

inline f32 max_f32(const f32* a, i64 n) {
    if (n == 0) return 0.0f;
    f32 m = a[0];
    for (i64 i = 1; i < n; ++i) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

#endif

// ============================================================================
// Quantized operations (platform-independent, use scalar with SIMD F32 where possible)
// ============================================================================
// Dequantize a Q1_0 block (32 elements) to f32
inline void dequantize_q1_block(const Q1_0* block, f32* out) {
    const f32 d = f16_to_f32(block->d);
    for (i32 j = 0; j < 8; ++j) {
        for (i32 k = 0; k < 4; ++k) {
            i32 val = ((block->qs[j] >> (k * 2)) & 0x03) - 1;
            out[j * 4 + k] = d * static_cast<f32>(val);
        }
    }
}

// Dequantize a Q4_0 block (32 elements) to f32
inline void dequantize_q4_block(const Q4_0* block, f32* out) {
    const f32 d = f16_to_f32(block->d);
    for (i32 j = 0; j < 16; ++j) {
        i32 lo = (block->qs[j] & 0x0F) - 8;
        i32 hi = ((block->qs[j] >> 4) & 0x0F) - 8;
        out[j * 2]     = d * static_cast<f32>(lo);
        out[j * 2 + 1] = d * static_cast<f32>(hi);
    }
}

// Dequantize a Q8_0 block (32 elements) to f32
inline void dequantize_q8_block(const Q8_0* block, f32* out) {
    const f32 d = f16_to_f32(block->d);
    for (i32 j = 0; j < 32; ++j) {
        out[j] = d * static_cast<f32>(block->qs[j]);
    }
}

// Dot product: Q1_0 weights @ f32 activations
// n is the number of logical elements (must be multiple of 32)
inline f32 dot_q1_f32(const Q1_0* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    i64 num_blocks = n / 32;

    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        f32 block_sum = 0.0f;
        for (i32 j = 0; j < 8; ++j) {
            for (i32 k = 0; k < 4; ++k) {
                i32 val = ((a[i].qs[j] >> (k * 2)) & 0x03) - 1;
                block_sum += static_cast<f32>(val) * b[i * 32 + j * 4 + k];
            }
        }
        sum += block_sum * d;
    }

    return sum;
}

// Dot product: Q4_0 weights @ f32 activations
// n is the number of logical elements (must be multiple of 32)
inline f32 dot_q4_f32(const Q4_0* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    i64 num_blocks = n / 32;

#if defined(LAI_NEON)
    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        float32x4_t block_sum = vdupq_n_f32(0.0f);

        const f32* bp = b + i * 32;
        for (i32 j = 0; j < 16; j += 2) {
            // Process 4 elements at a time (2 bytes = 4 nibbles)
            i32 lo0 = (a[i].qs[j] & 0x0F) - 8;
            i32 hi0 = ((a[i].qs[j] >> 4) & 0x0F) - 8;
            i32 lo1 = (a[i].qs[j + 1] & 0x0F) - 8;
            i32 hi1 = ((a[i].qs[j + 1] >> 4) & 0x0F) - 8;

            float32x4_t vq = {static_cast<f32>(lo0), static_cast<f32>(hi0),
                              static_cast<f32>(lo1), static_cast<f32>(hi1)};
            float32x4_t vb = vld1q_f32(bp + j * 2);
            block_sum = vfmaq_f32(block_sum, vq, vb);
        }
        sum += vaddvq_f32(block_sum) * d;
    }
#elif defined(LAI_AVX2)
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    const __m256i offset = _mm256_set1_epi32(8);

    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        __m256 acc = _mm256_setzero_ps();

        const f32* bp = b + i * 32;
        const u8* qs = a[i].qs;

        for (i32 j = 0; j < 16; j += 4) {
            // Unpack 4 bytes -> 8 nibbles
            __m128i raw = _mm_loadu_si128((const __m128i*)(qs + j));
            
            // Get low nibbles
            __m128i lo = _mm_and_si128(raw, low_mask);
            // Get high nibbles
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);

            // Interleave lo/hi to get 8 nibbles in order
            __m128i lo_hi = _mm_unpacklo_epi8(lo, hi);
            
            // Convert i8 -> i32 -> f32
            __m256i w_i32 = _mm256_cvtepi8_epi32(lo_hi);
            // Subtract offset 8
            w_i32 = _mm256_sub_epi32(w_i32, offset);
            __m256 w_f32 = _mm256_cvtepi32_ps(w_i32);

            __m256 b_f32 = _mm256_loadu_ps(bp + j * 2);
            acc = _mm256_fmadd_ps(w_f32, b_f32, acc);
        }

        __m128 h_hi = _mm256_extractf128_ps(acc, 1);
        __m128 h_lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(h_lo, h_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum += _mm_cvtss_f32(sum128) * d;
    }
#else
    // Scalar fallback
    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        f32 block_sum = 0.0f;
        for (i32 j = 0; j < 16; ++j) {
            i32 lo = (a[i].qs[j] & 0x0F) - 8;
            i32 hi = ((a[i].qs[j] >> 4) & 0x0F) - 8;
            block_sum += static_cast<f32>(lo) * b[i * 32 + j * 2];
            block_sum += static_cast<f32>(hi) * b[i * 32 + j * 2 + 1];
        }
        sum += block_sum * d;
    }
#endif

    return sum;
}

// Dot product: Q8_0 weights @ f32 activations
// n is the number of logical elements (must be multiple of 32)
inline f32 dot_q8_f32(const Q8_0* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    i64 num_blocks = n / 32;

#if defined(LAI_NEON)
    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        float32x4_t block_sum = vdupq_n_f32(0.0f);

        const f32* bp = b + i * 32;
        for (i32 j = 0; j < 32; j += 4) {
            float32x4_t vq = {static_cast<f32>(a[i].qs[j]),
                              static_cast<f32>(a[i].qs[j + 1]),
                              static_cast<f32>(a[i].qs[j + 2]),
                              static_cast<f32>(a[i].qs[j + 3])};
            float32x4_t vb = vld1q_f32(bp + j);
            block_sum = vfmaq_f32(block_sum, vq, vb);
        }
        sum += vaddvq_f32(block_sum) * d;
    }
#elif defined(LAI_AVX2)
    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        __m256 acc = _mm256_setzero_ps();

        const f32* bp = b + i * 32;
        const i8* weight_ptr = a[i].qs;

        for (i32 j = 0; j < 32; j += 8) {
            // Load 8 bytes of weights
            __m128i w128 = _mm_loadu_si128((const __m128i*)(weight_ptr + j));
            // Convert first 8 bytes from i8 to i32
            __m256i w256 = _mm256_cvtepi8_epi32(w128);
            // Convert i32 to f32
            __m256 wf = _mm256_cvtepi32_ps(w256);
            // Load 8 floats of activations
            __m256 bf = _mm256_loadu_ps(bp + j);
            // FMA
            acc = _mm256_fmadd_ps(wf, bf, acc);
        }

        // Horizontal sum of acc
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum += _mm_cvtss_f32(sum128) * d;
    }
#else
    // Scalar fallback
    for (i64 i = 0; i < num_blocks; ++i) {
        const f32 d = f16_to_f32(a[i].d);
        f32 block_sum = 0.0f;
        for (i32 j = 0; j < 32; ++j) {
            block_sum += static_cast<f32>(a[i].qs[j]) * b[i * 32 + j];
        }
        sum += block_sum * d;
    }
#endif

    return sum;
}

// Dot product: BitNet ternary weights (stored as i8) @ f32 activations
// n must be multiple of 8 for AVX2
inline f32 dot_bitnet_i8_f32(const i8* a, const f32* b, i64 n) {
    f32 sum = 0.0f;
    i64 i = 0;

#if defined(LAI_AVX2)
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(b + i);
        // Load 8 weights and convert to float
        __m128i w_low = _mm_loadu_si128((const __m128i*)(a + i));
        __m256i w_i32 = _mm256_cvtepi8_epi32(w_low);
        __m256 w_f32 = _mm256_cvtepi32_ps(w_i32);
        
        sum_vec = _mm256_fmadd_ps(in_vec, w_f32, sum_vec);
    }
    
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#endif

    for (; i < n; ++i) {
        sum += static_cast<f32>(a[i]) * b[i];
    }
    return sum;
}

// ============================================================================
// Common operations (built on primitives above)
// ============================================================================

// Copy vector
inline void copy_f32(f32* dst, const f32* src, i64 n) {
    std::memcpy(dst, src, n * sizeof(f32));
}

// Fill vector with constant
inline void fill_f32(f32* a, f32 val, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        a[i] = val;
    }
}

// Compute mean
inline f32 mean_f32(const f32* a, i64 n) {
    return sum_f32(a, n) / static_cast<f32>(n);
}

// Compute variance
inline f32 var_f32(const f32* a, i64 n, f32 mean) {
    f32 sum = 0.0f;
    for (i64 i = 0; i < n; ++i) {
        f32 diff = a[i] - mean;
        sum += diff * diff;
    }
    return sum / static_cast<f32>(n);
}

} // namespace simd
} // namespace lai

#endif // LAI_CORE_SIMD_H

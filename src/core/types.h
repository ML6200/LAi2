#ifndef LAI_CORE_TYPES_H
#define LAI_CORE_TYPES_H

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace lai {

// Floating point types
using f32 = float;
using f64 = double;
using f16 = uint16_t;  // Half precision storage

// Integer types
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// Quantized types
struct Q1_0 {
    f16 d;           // Delta (scale)
    u8 qs[8];        // Quantized values (32 x 2-bit packed into 8 bytes)
};

struct Q4_0 {
    f16 d;           // Delta (scale)
    u8 qs[16];       // Quantized values (32 x 4-bit packed into 16 bytes)
};

struct Q8_0 {
    f16 d;           // Delta (scale)
    i8 qs[32];       // Quantized values
};

// Token type
using TokenId = i32;

// Shape type (max 4 dimensions)
struct Shape {
    i64 dims[4] = {1, 1, 1, 1};
    i32 ndim = 0;

    Shape() = default;
    Shape(i64 d0) : dims{d0, 1, 1, 1}, ndim(1) {}
    Shape(i64 d0, i64 d1) : dims{d0, d1, 1, 1}, ndim(2) {}
    Shape(i64 d0, i64 d1, i64 d2) : dims{d0, d1, d2, 1}, ndim(3) {}
    Shape(i64 d0, i64 d1, i64 d2, i64 d3) : dims{d0, d1, d2, d3}, ndim(4) {}

    i64 operator[](i32 i) const { return dims[i]; }
    i64& operator[](i32 i) { return dims[i]; }

    i64 numel() const {
        i64 n = 1;
        for (i32 i = 0; i < ndim; ++i) n *= dims[i];
        return n;
    }

    bool operator==(const Shape& other) const {
        if (ndim != other.ndim) return false;
        for (i32 i = 0; i < ndim; ++i) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

// Data types enum
enum class DType : u8 {
    F32,
    F16,
    Q1_0,
    Q4_0,
    Q8_0,
    I32,
    I8
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:  return sizeof(f32);
        case DType::F16:  return sizeof(f16);
        case DType::Q1_0: return sizeof(Q1_0);
        case DType::Q4_0: return sizeof(Q4_0);
        case DType::Q8_0: return sizeof(Q8_0);
        case DType::I32:  return sizeof(i32);
        case DType::I8:   return sizeof(i8);
    }
    return 0;
}

// Block size for quantized types
inline i32 dtype_block_size(DType dtype) {
    switch (dtype) {
        case DType::Q1_0: return 32;
        case DType::Q4_0: return 32;
        case DType::Q8_0: return 32;
        default: return 1;
    }
}

// Storage bytes for a given number of logical elements
inline size_t storage_bytes(i64 numel, DType dtype) {
    i32 bs = dtype_block_size(dtype);
    if (bs > 1) {
        // Quantized: (numel / block_size) blocks, each dtype_size bytes
        return static_cast<size_t>((numel / bs) * static_cast<i64>(dtype_size(dtype)));
    }
    return static_cast<size_t>(numel) * dtype_size(dtype);
}

// IEEE 754 half-precision (f16) to float conversion
inline f32 f16_to_f32(f16 h) {
    u32 sign = (h >> 15) & 0x1;
    u32 exp  = (h >> 10) & 0x1f;
    u32 mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            u32 result = sign << 31;
            f32 f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        }
        // Subnormal: normalize
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3ff;
    } else if (exp == 0x1f) {
        // Inf/NaN
        u32 result = (sign << 31) | 0x7f800000 | (mant << 13);
        f32 f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }

    u32 result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    f32 f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
}

// Float to IEEE 754 half-precision (f16) conversion
inline f16 f32_to_f16(f32 val) {
    u32 bits;
    std::memcpy(&bits, &val, sizeof(bits));

    u32 sign = (bits >> 16) & 0x8000;
    i32 exp  = ((bits >> 23) & 0xff) - 127;
    u32 mant = bits & 0x7fffff;

    if (exp > 15) {
        // Overflow -> infinity
        return static_cast<f16>(sign | 0x7c00);
    }
    if (exp < -14) {
        // Underflow -> zero (or subnormal, simplified)
        return static_cast<f16>(sign);
    }

    return static_cast<f16>(sign | ((exp + 15) << 10) | (mant >> 13));
}

// DType name string
inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::Q1_0: return "Q1_0";
        case DType::Q4_0: return "Q4_0";
        case DType::Q8_0: return "Q8_0";
        case DType::I32:  return "I32";
        case DType::I8:   return "I8";
    }
    return "unknown";
}

} // namespace lai

#endif // LAI_CORE_TYPES_H

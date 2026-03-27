// metal_backend.mm - Metal GPU compute backend for LAi
// Accelerates matrix-vector multiply on Apple Silicon via MSL compute shaders.
// Small element-wise ops delegate to CPU (dispatch overhead > compute time).

#ifdef LAI_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.h"
#include "../core/tensor.h"
#include <unordered_map>
#include <iostream>

namespace lai {

// ============================================================================
// MSL Compute Shaders
// ============================================================================

static const char* kMetalShaders = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Must match C++ Q4_0: { f16 d; u8 qs[16]; } = 18 bytes
struct Q4Block {
    ushort d;
    uchar  qs[16];
};

// Must match C++ Q8_0: { f16 d; i8 qs[32]; } = 34 bytes
struct Q8Block {
    ushort d;
    char   qs[32];
};

struct MatvecParams {
    uint M;
    uint K;
};

// F32 matvec: y[row] = dot(A[row,:], x)
kernel void matvec_f32(
    device const float*       A      [[buffer(0)]],
    device const float*       x      [[buffer(1)]],
    device float*             y      [[buffer(2)]],
    constant MatvecParams&    params [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.M) return;

    uint K = params.K;
    device const float* row_ptr = A + row * K;
    float sum = 0.0f;

    for (uint j = 0; j < K; ++j) {
        sum += row_ptr[j] * x[j];
    }
    y[row] = sum;
}

// Q4_0 matvec: y[row] = dot(dequant(A[row,:]), x)
// Nibble convention: lo = (qs[j] & 0x0F) - 8, hi = (qs[j] >> 4) - 8
kernel void matvec_q4(
    device const Q4Block*     A      [[buffer(0)]],
    device const float*       x      [[buffer(1)]],
    device float*             y      [[buffer(2)]],
    constant MatvecParams&    params [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.M) return;

    uint blocks_per_row = params.K / 32;
    float sum = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        device const Q4Block& block = A[row * blocks_per_row + b];
        float scale = float(as_type<half>(block.d));
        uint base = b * 32;

        for (uint j = 0; j < 16; ++j) {
            uchar byte_val = block.qs[j];
            float lo = float(int(byte_val & 0x0F) - 8);
            float hi = float(int(byte_val >> 4)    - 8);
            sum += scale * lo * x[base + j * 2];
            sum += scale * hi * x[base + j * 2 + 1];
        }
    }
    y[row] = sum;
}

// Q8_0 matvec: y[row] = dot(dequant(A[row,:]), x)
kernel void matvec_q8(
    device const Q8Block*     A      [[buffer(0)]],
    device const float*       x      [[buffer(1)]],
    device float*             y      [[buffer(2)]],
    constant MatvecParams&    params [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.M) return;

    uint blocks_per_row = params.K / 32;
    float sum = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        device const Q8Block& block = A[row * blocks_per_row + b];
        float scale = float(as_type<half>(block.d));
        uint base = b * 32;

        for (uint j = 0; j < 32; ++j) {
            sum += scale * float(block.qs[j]) * x[base + j];
        }
    }
    y[row] = sum;
}
)MSL";

// ============================================================================
// Pimpl implementation
// ============================================================================

struct MetalBackend::Impl {
    id<MTLDevice>       device  = nil;
    id<MTLCommandQueue> queue   = nil;
    id<MTLLibrary>      library = nil;

    // Compute pipelines
    id<MTLComputePipelineState> pipe_matvec_f32 = nil;
    id<MTLComputePipelineState> pipe_matvec_q4  = nil;
    id<MTLComputePipelineState> pipe_matvec_q8  = nil;

    // Weight buffer cache (CPU data ptr -> MTLBuffer, created once per weight)
    std::unordered_map<const void*, id<MTLBuffer>> buf_cache;

    // Reusable output scratch buffer
    id<MTLBuffer> out_buf      = nil;
    size_t        out_buf_size = 0;

    // Reusable input scratch buffer (for inputs > 4 KB that can't use setBytes)
    id<MTLBuffer> in_buf      = nil;
    size_t        in_buf_size = 0;

    // -- helpers --

    id<MTLComputePipelineState> make_pipeline(NSString* name) {
        id<MTLFunction> func = [library newFunctionWithName:name];
        if (!func) return nil;
        NSError* error = nil;
        return [device newComputePipelineStateWithFunction:func error:&error];
    }

    id<MTLBuffer> get_weight_buffer(const void* ptr, size_t bytes) {
        auto it = buf_cache.find(ptr);
        if (it != buf_cache.end()) return it->second;

        id<MTLBuffer> buf = [device newBufferWithBytes:ptr
                                                length:bytes
                                               options:MTLResourceStorageModeShared];
        if (buf) buf_cache[ptr] = buf;
        return buf;
    }

    void ensure_out(size_t bytes) {
        if (bytes <= out_buf_size) return;
        out_buf_size = std::max(bytes, size_t(4096));
        out_buf = [device newBufferWithLength:out_buf_size
                                     options:MTLResourceStorageModeShared];
    }

    void ensure_in(size_t bytes) {
        if (bytes <= in_buf_size) return;
        in_buf_size = std::max(bytes, size_t(4096));
        in_buf = [device newBufferWithLength:in_buf_size
                                    options:MTLResourceStorageModeShared];
    }
};

// ============================================================================
// MetalBackend lifetime
// ============================================================================

MetalBackend::MetalBackend() : impl_(nullptr) {}

MetalBackend::~MetalBackend() {
    delete impl_;
}

bool MetalBackend::init() {
    @autoreleasepool {
        impl_ = new Impl();

        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) { delete impl_; impl_ = nullptr; return false; }

        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) { delete impl_; impl_ = nullptr; return false; }

        // Compile MSL source
        NSError* error = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }

        impl_->library = [impl_->device newLibraryWithSource:@(kMetalShaders)
                                                     options:opts
                                                       error:&error];
        if (!impl_->library) {
            if (error) {
                std::cerr << "[Metal] shader compile error: "
                          << [[error localizedDescription] UTF8String] << "\n";
            }
            delete impl_; impl_ = nullptr;
            return false;
        }

        // Build pipelines
        impl_->pipe_matvec_f32 = impl_->make_pipeline(@"matvec_f32");
        impl_->pipe_matvec_q4  = impl_->make_pipeline(@"matvec_q4");
        impl_->pipe_matvec_q8  = impl_->make_pipeline(@"matvec_q8");

        if (!impl_->pipe_matvec_f32 || !impl_->pipe_matvec_q4 ||
            !impl_->pipe_matvec_q8) {
            std::cerr << "[Metal] failed to create compute pipelines\n";
            delete impl_; impl_ = nullptr;
            return false;
        }

        return true;
    }
}

// ============================================================================
// GPU-accelerated matrix-vector multiply
// ============================================================================

void MetalBackend::matvec(TensorView& y, const TensorView& A, const TensorView& x) {
    @autoreleasepool {
        const i64 M = A.dim(0);
        const i64 K = A.dim(1);
        const size_t x_bytes = static_cast<size_t>(K) * sizeof(f32);
        const size_t y_bytes = static_cast<size_t>(M) * sizeof(f32);

        // 1. Weight buffer (cached -- created once per unique weight pointer)
        size_t w_bytes = storage_bytes(A.numel(), A.dtype());
        id<MTLBuffer> a_buf = impl_->get_weight_buffer(A.data(), w_bytes);
        if (!a_buf) { ops::matvec_dispatch(y, A, x); return; }

        // 2. Output scratch buffer
        impl_->ensure_out(y_bytes);

        // 3. Select pipeline
        id<MTLComputePipelineState> pipe;
        switch (A.dtype()) {
            case DType::Q4_0: pipe = impl_->pipe_matvec_q4;  break;
            case DType::Q8_0: pipe = impl_->pipe_matvec_q8;  break;
            default:          pipe = impl_->pipe_matvec_f32;  break;
        }

        // 4. Encode compute command
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipe];
        [enc setBuffer:a_buf offset:0 atIndex:0];

        // Input vector: use setBytes for data <= 4 KB (avoids buffer alloc)
        if (x_bytes <= 4096) {
            [enc setBytes:x.data_f32() length:x_bytes atIndex:1];
        } else {
            impl_->ensure_in(x_bytes);
            memcpy([impl_->in_buf contents], x.data_f32(), x_bytes);
            [enc setBuffer:impl_->in_buf offset:0 atIndex:1];
        }

        [enc setBuffer:impl_->out_buf offset:0 atIndex:2];

        struct { uint32_t M; uint32_t K; } params = {
            static_cast<uint32_t>(M), static_cast<uint32_t>(K)
        };
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        // 5. Dispatch (one thread per output row)
        NSUInteger tg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        NSUInteger groups = (static_cast<NSUInteger>(M) + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // 6. Read output back to caller's buffer
        memcpy(y.data_f32(), [impl_->out_buf contents], y_bytes);
    }
}

// ============================================================================
// Small ops: CPU fallback (GPU dispatch overhead > compute for dim-sized vecs)
// ============================================================================

void MetalBackend::rmsnorm(TensorView& y, const TensorView& x,
                           const TensorView& w, f32 eps) {
    ops::rmsnorm(y, x, w, eps);
}

void MetalBackend::softmax(TensorView& y, const TensorView& x) {
    ops::softmax(y, x);
}

void MetalBackend::rope(TensorView& q, TensorView& k,
                        i32 pos, i32 head_dim, f32 theta) {
    ops::rope(q, k, pos, head_dim, theta);
}

void MetalBackend::silu(TensorView& y, const TensorView& x) {
    ops::silu(y, x);
}

void MetalBackend::add(TensorView& y, const TensorView& a, const TensorView& b) {
    ops::add(y, a, b);
}

void MetalBackend::mul(TensorView& y, const TensorView& a, const TensorView& b) {
    ops::mul(y, a, b);
}

void MetalBackend::copy(TensorView& dst, const TensorView& src) {
    simd::copy_f32(dst.data_f32(), src.data_f32(), dst.numel());
}

void MetalBackend::sync() {
    // All operations are currently synchronous (waitUntilCompleted in matvec)
}

// ============================================================================
// Factory
// ============================================================================

Backend* Backend::create_metal() {
    auto* backend = new MetalBackend();
    if (!backend->init()) {
        delete backend;
        return nullptr;
    }
    return backend;
}

} // namespace lai

#endif // LAI_METAL

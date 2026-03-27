#ifndef LAI_MODEL_CONFIG_H
#define LAI_MODEL_CONFIG_H

#include "../core/types.h"
#include <string>
#include <cstring>
#include <iomanip>

namespace lai {

#pragma pack(push, 1)
// Model configuration
struct ModelConfig {
    // Architecture
    i32 dim = 512;              // Embedding dimension
    i32 hidden_dim = 2048;      // FFN hidden dimension
    i32 n_layers = 12;          // Number of transformer layers
    i32 n_heads = 8;            // Number of attention heads
    i32 n_kv_heads = 8;         // Number of KV heads (for GQA)
    i32 vocab_size = 32000;     // Vocabulary size
    i32 max_seq_len = 1024;     // Maximum sequence length

    // Derived
    i32 head_dim() const { return dim / n_heads; }
    i32 kv_dim() const { return (dim / n_heads) * n_kv_heads; }

    // RoPE parameters
    f32 rope_theta = 10000.0f;

    // Normalization
    f32 norm_eps = 1e-5f;

    // Activation (0 = SiLU, 1 = GELU)
    i32 activation = 0;

    // Compute parameter count (approximate)
    i64 param_count() const {
        i64 embed = static_cast<i64>(vocab_size) * dim * 2;  // token + output
        i64 attn = n_layers * (4 * dim * dim);               // Q, K, V, O
        i64 ffn = n_layers * (3 * dim * hidden_dim);         // gate, up, down
        i64 norm = n_layers * dim * 2 + dim;                 // layer norms + final
        return embed + attn + ffn + norm;
    }

    // Estimate memory usage
    i64 memory_bytes(DType dtype = DType::F32) const {
        i64 params = param_count();
        i64 bytes_per_param = dtype == DType::Q4_0 ? 1 : (dtype == DType::Q8_0 ? 1 : 4);
        return params * bytes_per_param / (dtype == DType::Q4_0 ? 2 : 1);
    }
};
#pragma pack(pop)

// Preset configurations
namespace presets {

// LAi-Micro: ~10M params, ultra-low-end hardware (64MB RAM)
inline ModelConfig lai_micro() {
    ModelConfig cfg;
    cfg.dim = 288;
    cfg.hidden_dim = 768;
    cfg.n_layers = 6;
    cfg.n_heads = 6;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 16000;
    cfg.max_seq_len = 256;
    return cfg;
}

// LAi-Tiny: ~30M params, low-end hardware (128MB RAM)
inline ModelConfig lai_tiny() {
    ModelConfig cfg;
    cfg.dim = 384;
    cfg.hidden_dim = 1152;
    cfg.n_layers = 8;
    cfg.n_heads = 6;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 32000;
    cfg.max_seq_len = 512;
    return cfg;
}

// LAi-Mini: ~100M params, fits in 4GB RAM with Q4
inline ModelConfig lai_mini() {
    ModelConfig cfg;
    cfg.dim = 512;
    cfg.hidden_dim = 1536;
    cfg.n_layers = 12;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 4;
    cfg.vocab_size = 32000;
    cfg.max_seq_len = 1024;
    return cfg;
}

// LAi-Small: ~350M params, better quality
inline ModelConfig lai_small() {
    ModelConfig cfg;
    cfg.dim = 768;
    cfg.hidden_dim = 2304;
    cfg.n_layers = 16;
    cfg.n_heads = 12;
    cfg.n_kv_heads = 4;
    cfg.vocab_size = 32000;
    cfg.max_seq_len = 2048;
    return cfg;
}

} // namespace presets

// Generation parameters
struct GenerationConfig {
    i32 max_tokens = 256;       // Maximum tokens to generate
    f32 temperature = 0.7f;     // Sampling temperature
    f32 top_p = 0.9f;           // Nucleus sampling threshold
    i32 top_k = 40;             // Top-k sampling
    f32 repeat_penalty = 1.1f;  // Repetition penalty
    f32 min_p = 0.0f;              // Min-p threshold (0 = disabled)
    f32 frequency_penalty = 0.0f;  // Count-based frequency penalty
    f32 presence_penalty = 0.0f;   // Binary presence penalty
    f32 dry_multiplier = 0.0f;     // DRY n-gram penalty (0 = disabled)
    i32 dry_allowed_length = 2;    // Minimum n-gram length for DRY
    f32 mirostat_tau = 0.0f;       // Mirostat v2 target surprise (0 = disabled)
    f32 mirostat_eta = 0.1f;       // Mirostat v2 learning rate
    i32 seed = -1;              // Random seed (-1 = random)

    // Stop tokens
    static constexpr i32 MAX_STOP_TOKENS = 8;
    i32 stop_tokens[MAX_STOP_TOKENS] = {0};
    i32 n_stop_tokens = 0;

    void add_stop_token(i32 token) {
        if (n_stop_tokens < MAX_STOP_TOKENS) {
            stop_tokens[n_stop_tokens++] = token;
        }
    }

    bool is_stop_token(i32 token) const {
        for (i32 i = 0; i < n_stop_tokens; ++i) {
            if (stop_tokens[i] == token) return true;
        }
        return false;
    }
};

#pragma pack(push, 1)
// Model file header (binary format)
struct ModelHeader {
    char magic[4] = {'L', 'A', 'i', '1'};  // Magic number
    u32 version = 1;                        // Format version
    ModelConfig config;                     // Model configuration
    DType weight_dtype = DType::F32;       // Weight data type
    u64 vocab_offset = 0;                   // Offset to vocabulary
    u64 weights_offset = 0;                 // Offset to weights

    bool is_valid() const {
        return magic[0] == 'L' && magic[1] == 'A' &&
               magic[2] == 'i' && magic[3] == '1';
    }
};
#pragma pack(pop)

} // namespace lai

#endif // LAI_MODEL_CONFIG_H

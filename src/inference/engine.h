#ifndef LAI_INFERENCE_ENGINE_H
#define LAI_INFERENCE_ENGINE_H

#include "../core/types.h"
#include "../core/tensor.h"
#include "../model/transformer.h"
#include "../model/config.h"
#include "../tokenizer/bpe.h"
#include "../backend/cpu_backend.h"
#include "sampler.h"
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iostream>
#include <memory>

namespace lai {

// Callback for streaming tokens
using StreamCallback = std::function<bool(const std::string& token, i32 token_id)>;

// Generation statistics
struct GenerationStats {
    i32 prompt_tokens = 0;
    i32 generated_tokens = 0;
    f64 prefill_time_ms = 0.0;
    f64 generation_time_ms = 0.0;
    f64 total_time_ms = 0.0;

    f64 tokens_per_second() const {
        return generation_time_ms > 0 ? (generated_tokens * 1000.0 / generation_time_ms) : 0.0;
    }

    f64 prefill_tokens_per_second() const {
        return prefill_time_ms > 0 ? (prompt_tokens * 1000.0 / prefill_time_ms) : 0.0;
    }
};

// Inference engine
class Engine {
public:
    Engine() = default;

    // Initialize with model path (vocab is embedded in model)
    bool init(const std::string& model_path, const std::string& vocab_path = "",
              bool use_mmap = true, const std::string& backend_name = "auto") {
        // Try mmap first (zero-copy, instant startup)
        bool loaded = false;
        if (use_mmap) {
            loaded = model_.load_mmap(model_path, &tokenizer_);
        }
        // Fall back to fread-based loading
        if (!loaded) {
            loaded = model_.load(model_path, &tokenizer_);
        }
        if (!loaded) {
            return false;
        }

        // Fall back to separate vocab file if provided and vocab wasn't embedded
        if (!vocab_path.empty() && tokenizer_.vocab_size() == 0) {
            if (!tokenizer_.load(vocab_path)) {
                return false;
            }
        }

        // Check if we got a vocabulary
        if (tokenizer_.vocab_size() == 0) {
            return false;
        }

        logits_ = Tensor(Shape(model_.config().vocab_size));
        sampler_ = Sampler();

        // Register chat special tokens as stop token IDs
        const char* stop_strs[] = {"<user>", "<system>", "</user>", "</system>", "</assistant>"};
        for (const char* s : stop_strs) {
            i32 id = tokenizer_.get_id(s);
            if (id != SpecialTokens::UNK) {
                stop_token_ids_.push_back(id);
            }
        }

        // Create compute backend
        if (backend_name == "cpu") {
            backend_.reset(Backend::create_cpu());
        }
#ifdef LAI_METAL
        else if (backend_name == "metal") {
            backend_.reset(Backend::create_metal());
            if (!backend_) backend_.reset(Backend::create_cpu());
        }
#endif
        else { // "auto"
            backend_.reset(Backend::create_best());
        }
        model_.set_backend(backend_.get());

        return true;
    }

    // Initialize with pre-loaded model (for testing)
    void init(const ModelConfig& config) {
        model_.init(config);
        logits_ = Tensor(Shape(config.vocab_size));
        sampler_ = Sampler();
    }

    // Generate text from prompt
    std::string generate(const std::string& prompt,
                        const GenerationConfig& gen_config = GenerationConfig(),
                        StreamCallback callback = nullptr,
                        GenerationStats* stats = nullptr) {

        auto start_time = std::chrono::high_resolution_clock::now();

        // Encode prompt
        std::vector<i32> tokens = tokenizer_.encode(prompt, true, false);

        if (stats) {
            stats->prompt_tokens = static_cast<i32>(tokens.size());
        }

        // Prefill: process prompt tokens
        auto prefill_start = std::chrono::high_resolution_clock::now();

        model_.reset();
        for (size_t i = 0; i < tokens.size(); ++i) {
            model_.forward(logits_, tokens[i], static_cast<i32>(i));
        }

        auto prefill_end = std::chrono::high_resolution_clock::now();

        if (stats) {
            stats->prefill_time_ms = std::chrono::duration<f64, std::milli>(
                prefill_end - prefill_start).count();
        }

        // Generation loop
        auto gen_start = std::chrono::high_resolution_clock::now();

        std::vector<i32> generated;
        std::string output;
        i32 pos = static_cast<i32>(tokens.size());

        // Track recent tokens for repetition penalty
        std::vector<i32> recent_tokens;
        const i32 recent_window = 64;

        for (i32 i = 0; i < gen_config.max_tokens; ++i) {
            // Sample next token
            i32 next_token = sampler_.sample(logits_, gen_config, recent_tokens);

            // Check for stop tokens (EOS, user-configured, and chat special tokens)
            if (next_token == SpecialTokens::EOS || gen_config.is_stop_token(next_token)) {
                break;
            }
            bool is_chat_stop = false;
            for (i32 sid : stop_token_ids_) {
                if (next_token == sid) { is_chat_stop = true; break; }
            }
            if (is_chat_stop) break;

            // Decode and output token
            std::string token_str = tokenizer_.decode_token(next_token);

            if (callback) {
                if (!callback(token_str, next_token)) {
                    break;  // User requested stop
                }
            }

            output += token_str;
            generated.push_back(next_token);

            // Update recent tokens
            recent_tokens.push_back(next_token);
            if (static_cast<i32>(recent_tokens.size()) > recent_window) {
                recent_tokens.erase(recent_tokens.begin());
            }

            // Forward pass for next token
            model_.forward(logits_, next_token, pos);
            ++pos;

            // Check context length
            if (pos >= model_.config().max_seq_len) {
                break;
            }
        }

        auto gen_end = std::chrono::high_resolution_clock::now();

        if (stats) {
            stats->generated_tokens = static_cast<i32>(generated.size());
            stats->generation_time_ms = std::chrono::duration<f64, std::milli>(
                gen_end - gen_start).count();
            stats->total_time_ms = std::chrono::duration<f64, std::milli>(
                gen_end - start_time).count();
        }

        return output;
    }

    // Chat completion
    std::string chat(const std::string& user_message,
                    const std::string& system_prompt = "",
                    const GenerationConfig& gen_config = GenerationConfig(),
                    StreamCallback callback = nullptr,
                    GenerationStats* stats = nullptr) {

        std::string prompt;

        if (!system_prompt.empty()) {
            prompt += tokenizer_.format_chat("system", system_prompt);
        }

        prompt += tokenizer_.format_chat("user", user_message);
        prompt += "<assistant>";  // Start assistant response

        return generate(prompt, gen_config, callback, stats);
    }

    // Translation (Hungarian <-> English)
    std::string translate(const std::string& text, bool to_hungarian = true,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         StreamCallback callback = nullptr) {

        std::string prompt;

        if (to_hungarian) {
            prompt = "<system>Translate the following English text to Hungarian. "
                    "Only output the translation, nothing else.</system>"
                    "<user>" + text + "</user><assistant>";
        } else {
            prompt = "<system>Fordítsd le a következő magyar szöveget angolra. "
                    "Csak a fordítást írd ki, semmi mást.</system>"
                    "<user>" + text + "</user><assistant>";
        }

        return generate(prompt, gen_config, callback);
    }

    // Code assistance
    std::string code_assist(const std::string& query,
                           const std::string& code_context = "",
                           const GenerationConfig& gen_config = GenerationConfig(),
                           StreamCallback callback = nullptr,
                           GenerationStats* stats = nullptr) {

        std::string prompt = "<system>You are a helpful coding assistant. "
                            "Provide clear, concise code solutions.</system>";

        if (!code_context.empty()) {
            prompt += "<user>Here is my code:\n```\n" + code_context + "\n```\n\n" + query + "</user>";
        } else {
            prompt += "<user>" + query + "</user>";
        }

        prompt += "<assistant>";

        return generate(prompt, gen_config, callback, stats);
    }

    // Text processing (summarize, rewrite, etc.)
    std::string process_text(const std::string& instruction,
                            const std::string& text,
                            const GenerationConfig& gen_config = GenerationConfig(),
                            StreamCallback callback = nullptr,
                            GenerationStats* stats = nullptr) {

        std::string prompt = "<system>Follow the user's instruction precisely.</system>"
                            "<user>" + instruction + "\n\nText:\n" + text + "</user>"
                            "<assistant>";

        return generate(prompt, gen_config, callback, stats);
    }

    // Accessors
    const ModelConfig& config() const { return model_.config(); }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    Tokenizer& tokenizer() { return tokenizer_; }

    // Memory usage
    i64 memory_bytes() const {
        return model_.memory_bytes() + logits_.numel() * sizeof(f32);
    }

    // Reset model state
    void reset() {
        model_.reset();
    }

    // Set random seed
    void set_seed(u32 seed) {
        sampler_.set_seed(seed);
    }

    const char* backend_name() const {
        return backend_ ? backend_->name() : "none";
    }

private:
    Transformer model_;
    Tokenizer tokenizer_;
    Sampler sampler_;
    Tensor logits_;
    std::unique_ptr<Backend> backend_;
    std::vector<i32> stop_token_ids_;
};

} // namespace lai

#endif // LAI_INFERENCE_ENGINE_H

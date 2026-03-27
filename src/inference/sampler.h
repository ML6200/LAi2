#ifndef LAI_INFERENCE_SAMPLER_H
#define LAI_INFERENCE_SAMPLER_H

#include "../core/types.h"
#include "../core/tensor.h"
#include "../model/config.h"
#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_map>

namespace lai {

// Token sampler with various strategies
class Sampler {
public:
    Sampler() : rng_(std::random_device{}()) {}

    explicit Sampler(u32 seed) : rng_(seed) {}

    void set_seed(u32 seed) {
        rng_.seed(seed);
    }

    void reset_mirostat() {
        mirostat_mu_ = 0.0f;
    }

    // Sample next token from logits
    i32 sample(const TensorView& logits, const GenerationConfig& config,
               const std::vector<i32>& recent_tokens = {},
               const f32* logit_bias = nullptr) {
        const i64 vocab_size = logits.numel();
        const f32* logits_data = logits.data_f32();

        // Copy logits to work buffer
        std::vector<f32> probs(vocab_size);
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] = logits_data[i];
        }

        // Apply logit bias if provided
        if (logit_bias) {
            for (i64 i = 0; i < vocab_size; ++i) {
                probs[i] += logit_bias[i];
            }
        }

        // Apply frequency/presence penalty (on logits, before temperature)
        if ((config.frequency_penalty != 0.0f || config.presence_penalty != 0.0f)
            && !recent_tokens.empty()) {
            apply_freq_presence_penalty(probs, recent_tokens,
                                        config.frequency_penalty, config.presence_penalty);
        }

        // Apply DRY penalty (on logits, before temperature)
        if (config.dry_multiplier > 0.0f && !recent_tokens.empty()) {
            apply_dry(probs, recent_tokens, config.dry_multiplier,
                      config.dry_allowed_length, vocab_size);
        }

        // Apply repetition penalty (legacy, on logits)
        if (config.repeat_penalty != 1.0f && !recent_tokens.empty()) {
            for (i32 token : recent_tokens) {
                if (token >= 0 && token < vocab_size) {
                    if (probs[token] > 0) {
                        probs[token] /= config.repeat_penalty;
                    } else {
                        probs[token] *= config.repeat_penalty;
                    }
                }
            }
        }

        // Mirostat v2: separate sampling path
        if (config.mirostat_tau > 0.0f) {
            return sample_mirostat(probs, config.mirostat_tau, config.mirostat_eta,
                                   config.temperature);
        }

        // Temperature scaling
        if (config.temperature != 1.0f && config.temperature > 0.0f) {
            f32 inv_temp = 1.0f / config.temperature;
            for (i64 i = 0; i < vocab_size; ++i) {
                probs[i] *= inv_temp;
            }
        }

        // Greedy decoding (temperature = 0)
        if (config.temperature <= 0.0f) {
            return static_cast<i32>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        }

        // Softmax
        f32 max_val = *std::max_element(probs.begin(), probs.end());
        f32 sum = 0.0f;
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(probs[i] - max_val);
            sum += probs[i];
        }
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
        }

        // Top-k filtering
        if (config.top_k > 0 && config.top_k < vocab_size) {
            apply_top_k(probs, config.top_k);
        }

        // Top-p (nucleus) filtering
        if (config.top_p < 1.0f && config.top_p > 0.0f) {
            apply_top_p(probs, config.top_p);
        }

        // Min-p filtering
        if (config.min_p > 0.0f && config.min_p < 1.0f) {
            apply_min_p(probs, config.min_p);
        }

        // Sample from distribution
        return sample_from_probs(probs);
    }

    // Argmax (greedy) sampling
    i32 argmax(const TensorView& logits) {
        const i64 n = logits.numel();
        const f32* data = logits.data_f32();

        i32 best = 0;
        f32 best_val = data[0];
        for (i64 i = 1; i < n; ++i) {
            if (data[i] > best_val) {
                best_val = data[i];
                best = static_cast<i32>(i);
            }
        }
        return best;
    }

private:
    // ========================================================================
    // Filtering methods
    // ========================================================================

    void apply_top_k(std::vector<f32>& probs, i32 k) {
        // Find k-th largest value
        std::vector<f32> sorted = probs;
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<f32>());
        f32 threshold = sorted[k - 1];

        // Zero out values below threshold
        f32 sum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            } else {
                sum += probs[i];
            }
        }

        // Renormalize
        if (sum > 0.0f) {
            for (f32& p : probs) {
                p /= sum;
            }
        }
    }

    void apply_top_p(std::vector<f32>& probs, f32 top_p) {
        // Sort by probability
        std::vector<std::pair<f32, i32>> sorted;
        sorted.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0.0f) {
                sorted.push_back({probs[i], static_cast<i32>(i)});
            }
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<std::pair<f32, i32>>());

        // Find cutoff
        f32 cumsum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i].first;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        std::fill(probs.begin(), probs.end(), 0.0f);
        f32 sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            probs[sorted[i].second] = sorted[i].first;
            sum += sorted[i].first;
        }

        // Renormalize
        if (sum > 0.0f) {
            for (f32& p : probs) {
                p /= sum;
            }
        }
    }

    // Min-p: keep only tokens with prob >= min_p * max_prob
    void apply_min_p(std::vector<f32>& probs, f32 min_p) {
        f32 max_prob = *std::max_element(probs.begin(), probs.end());
        f32 threshold = min_p * max_prob;

        f32 sum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            } else {
                sum += probs[i];
            }
        }

        if (sum > 0.0f) {
            for (f32& p : probs) {
                p /= sum;
            }
        }
    }

    // ========================================================================
    // Penalty methods (applied on logits before softmax)
    // ========================================================================

    // Frequency + presence penalty: penalize based on token occurrence counts
    void apply_freq_presence_penalty(std::vector<f32>& logits,
                                     const std::vector<i32>& recent_tokens,
                                     f32 freq_penalty, f32 pres_penalty) {
        std::unordered_map<i32, i32> counts;
        for (i32 token : recent_tokens) {
            counts[token]++;
        }

        i64 vocab_size = static_cast<i64>(logits.size());
        for (auto& [token, count] : counts) {
            if (token >= 0 && token < vocab_size) {
                logits[token] -= freq_penalty * static_cast<f32>(count)
                              + pres_penalty;
            }
        }
    }

    // DRY (Don't Repeat Yourself): penalize tokens that would extend repeated n-grams
    void apply_dry(std::vector<f32>& logits,
                   const std::vector<i32>& recent_tokens,
                   f32 multiplier, i32 allowed_length, i64 vocab_size) {
        const i32 n = static_cast<i32>(recent_tokens.size());
        if (n < allowed_length) return;

        // For each possible next token, check if it would create a repeated n-gram
        // by looking at the suffix of recent_tokens.
        //
        // Strategy: find the longest match between the end of the sequence
        // and any earlier position. The token following that earlier match
        // gets penalized proportionally to match length.

        // Look at the last `allowed_length-1` to `n-1` tokens as the suffix
        // Try to find matches earlier in the sequence
        for (i32 suffix_len = allowed_length - 1; suffix_len < n && suffix_len < 64; ++suffix_len) {
            // The suffix is recent_tokens[n - suffix_len .. n-1]
            const i32 suffix_start = n - suffix_len;

            // Search for this suffix earlier in the sequence
            for (i32 pos = 0; pos <= n - suffix_len - 1; ++pos) {
                bool match = true;
                for (i32 k = 0; k < suffix_len; ++k) {
                    if (recent_tokens[pos + k] != recent_tokens[suffix_start + k]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    // The token that followed this earlier occurrence
                    i32 next_pos = pos + suffix_len;
                    if (next_pos < n) {
                        i32 penalized_token = recent_tokens[next_pos];
                        if (penalized_token >= 0 && penalized_token < vocab_size) {
                            // Penalty grows exponentially with match length
                            f32 penalty = multiplier * static_cast<f32>(suffix_len);
                            logits[penalized_token] -= penalty;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Mirostat v2: adaptive sampling to target perplexity
    // ========================================================================

    i32 sample_mirostat(std::vector<f32>& logits, f32 tau, f32 eta, f32 temperature) {
        // Initialize mu on first call
        if (mirostat_mu_ == 0.0f) {
            mirostat_mu_ = 2.0f * tau;
        }

        // Apply temperature
        if (temperature > 0.0f && temperature != 1.0f) {
            f32 inv_temp = 1.0f / temperature;
            for (f32& l : logits) {
                l *= inv_temp;
            }
        }

        // Softmax
        f32 max_val = *std::max_element(logits.begin(), logits.end());
        f32 sum = 0.0f;
        for (f32& l : logits) {
            l = std::exp(l - max_val);
            sum += l;
        }
        for (f32& l : logits) {
            l /= sum;
        }

        // Sort tokens by probability (descending)
        std::vector<std::pair<f32, i32>> sorted;
        sorted.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > 0.0f) {
                sorted.push_back({logits[i], static_cast<i32>(i)});
            }
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<std::pair<f32, i32>>());

        // Find the token set where surprise (-log2(p)) is closest to mu
        // Accumulate tokens until their surprise exceeds mu
        std::vector<f32> kept_probs;
        std::vector<i32> kept_indices;
        for (auto& [prob, idx] : sorted) {
            f32 surprise = -std::log2(prob);
            if (surprise > mirostat_mu_ && !kept_probs.empty()) {
                break;
            }
            kept_probs.push_back(prob);
            kept_indices.push_back(idx);
        }

        // Ensure at least one token
        if (kept_probs.empty()) {
            kept_probs.push_back(sorted[0].first);
            kept_indices.push_back(sorted[0].second);
        }

        // Renormalize kept set
        f32 kept_sum = 0.0f;
        for (f32 p : kept_probs) kept_sum += p;
        for (f32& p : kept_probs) p /= kept_sum;

        // Sample from kept set
        std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
        f32 r = dist(rng_);
        f32 cumsum = 0.0f;
        i32 chosen = kept_indices.back();
        f32 chosen_prob = kept_probs.back();
        for (size_t i = 0; i < kept_probs.size(); ++i) {
            cumsum += kept_probs[i];
            if (r < cumsum) {
                chosen = kept_indices[i];
                chosen_prob = kept_probs[i];
                break;
            }
        }

        // Update mu based on the actual surprise of the chosen token
        f32 surprise = -std::log2(chosen_prob);
        mirostat_mu_ -= eta * (surprise - tau);

        return chosen;
    }

    // ========================================================================
    // Base sampling
    // ========================================================================

    i32 sample_from_probs(const std::vector<f32>& probs) {
        std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
        f32 r = dist(rng_);

        f32 cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<i32>(i);
            }
        }

        // Fallback to last token (shouldn't happen)
        return static_cast<i32>(probs.size() - 1);
    }

    std::mt19937 rng_;
    f32 mirostat_mu_ = 0.0f;
};

} // namespace lai

#endif // LAI_INFERENCE_SAMPLER_H

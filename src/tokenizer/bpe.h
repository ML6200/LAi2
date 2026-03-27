#ifndef LAI_TOKENIZER_BPE_H
#define LAI_TOKENIZER_BPE_H

#include "../core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>

namespace lai {

// Special tokens
struct SpecialTokens {
    static constexpr i32 PAD = 0;
    static constexpr i32 BOS = 1;       // Beginning of sequence
    static constexpr i32 EOS = 2;       // End of sequence
    static constexpr i32 UNK = 3;       // Unknown token

    static constexpr i32 BYTE_OFFSET = 4;  // Byte tokens start at index 4
};

// BPE Tokenizer optimized for Hungarian + English
class Tokenizer {
public:
    Tokenizer() = default;

    // Load vocabulary from file
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        // Read header
        u32 magic, version, vocab_size;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        f.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

        if (magic != 0x4C414956) return false;  // "LAIV"

        vocab_.resize(vocab_size);
        token_to_id_.clear();
        scores_.resize(vocab_size);

        // Read vocabulary
        for (u32 i = 0; i < vocab_size; ++i) {
            u32 len;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));

            vocab_[i].resize(len);
            f.read(vocab_[i].data(), len);

            f.read(reinterpret_cast<char*>(&scores_[i]), sizeof(f32));

            token_to_id_[vocab_[i]] = i;
        }

        return true;
    }

    // Load vocabulary from C FILE* at current position
    bool load_from_file(FILE* f) {
        if (!f) return false;

        // Read header
        u32 magic, version, vocab_size;
        fread(&magic, sizeof(magic), 1, f);
        fread(&version, sizeof(version), 1, f);
        fread(&vocab_size, sizeof(vocab_size), 1, f);

        if (magic != 0x4C414956) return false;  // "LAIV"

        vocab_.resize(vocab_size);
        token_to_id_.clear();
        scores_.resize(vocab_size);

        // Read vocabulary
        for (u32 i = 0; i < vocab_size; ++i) {
            u32 len;
            fread(&len, sizeof(len), 1, f);

            vocab_[i].resize(len);
            fread(vocab_[i].data(), 1, len, f);

            fread(&scores_[i], sizeof(f32), 1, f);

            token_to_id_[vocab_[i]] = i;
        }

        return true;
    }

    // Load vocabulary from memory buffer (for mmap)
    bool load_from_memory(const u8* data, size_t len) {
        if (!data || len < 12) return false;

        size_t pos = 0;
        auto read = [&](void* dst, size_t n) -> bool {
            if (pos + n > len) return false;
            std::memcpy(dst, data + pos, n);
            pos += n;
            return true;
        };

        u32 magic, version, vocab_size;
        if (!read(&magic, 4) || !read(&version, 4) || !read(&vocab_size, 4))
            return false;

        if (magic != 0x4C414956) return false;  // "LAIV"

        vocab_.resize(vocab_size);
        token_to_id_.clear();
        scores_.resize(vocab_size);

        for (u32 i = 0; i < vocab_size; ++i) {
            u32 token_len;
            if (!read(&token_len, 4)) return false;

            vocab_[i].resize(token_len);
            if (!read(vocab_[i].data(), token_len)) return false;

            if (!read(&scores_[i], 4)) return false;

            token_to_id_[vocab_[i]] = i;
        }

        return true;
    }

    // Save vocabulary to file
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;

        u32 magic = 0x4C414956;  // "LAIV"
        u32 version = 1;
        u32 vocab_size = static_cast<u32>(vocab_.size());

        f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));
        f.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

        for (u32 i = 0; i < vocab_size; ++i) {
            u32 len = static_cast<u32>(vocab_[i].size());
            f.write(reinterpret_cast<const char*>(&len), sizeof(len));
            f.write(vocab_[i].data(), len);
            f.write(reinterpret_cast<const char*>(&scores_[i]), sizeof(f32));
        }

        return true;
    }

    // Encode text to tokens
    std::vector<i32> encode(const std::string& text, bool add_bos = true, bool add_eos = false) const {
        std::vector<i32> tokens;

        if (add_bos) {
            tokens.push_back(SpecialTokens::BOS);
        }

        if (text.empty()) {
            if (add_eos) tokens.push_back(SpecialTokens::EOS);
            return tokens;
        }

        // UTF-8 aware character splitting
        std::vector<std::string> chars;
        for (size_t i = 0; i < text.size();) {
            size_t char_len = utf8_char_len(text[i]);
            chars.push_back(text.substr(i, char_len));
            i += char_len;
        }

        // Initial tokenization: each char becomes a token
        std::vector<std::string> pieces = chars;

        // BPE merge loop
        while (pieces.size() > 1) {
            // Find best merge
            f32 best_score = -1e10f;
            size_t best_idx = 0;
            std::string best_merge;

            for (size_t i = 0; i < pieces.size() - 1; ++i) {
                std::string merged = pieces[i] + pieces[i + 1];
                auto it = token_to_id_.find(merged);
                if (it != token_to_id_.end()) {
                    f32 score = scores_[it->second];
                    if (score > best_score) {
                        best_score = score;
                        best_idx = i;
                        best_merge = merged;
                    }
                }
            }

            if (best_score == -1e10f) break;  // No more merges possible

            // Apply merge
            pieces[best_idx] = best_merge;
            pieces.erase(pieces.begin() + best_idx + 1);
        }

        // Convert pieces to token IDs
        for (const auto& piece : pieces) {
            auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                // Handle unknown: encode as byte tokens
                for (unsigned char c : piece) {
                    tokens.push_back(static_cast<i32>(c) + SpecialTokens::BYTE_OFFSET);
                }
            }
        }

        if (add_eos) {
            tokens.push_back(SpecialTokens::EOS);
        }

        return tokens;
    }

    // Decode tokens to text
    std::string decode(const std::vector<i32>& tokens) const {
        std::string text;
        for (i32 token : tokens) {
            if (token < 0 || token >= static_cast<i32>(vocab_.size())) continue;

            // Skip special tokens in output
            if (token == SpecialTokens::BOS || token == SpecialTokens::EOS ||
                token == SpecialTokens::PAD) continue;

            text += vocab_[token];
        }
        return text;
    }

    // Decode single token
    std::string decode_token(i32 token) const {
        if (token < 0 || token >= static_cast<i32>(vocab_.size())) {
            return "<token:" + std::to_string(token) + ">";
        }
        return vocab_[token];
    }

    // Vocabulary size
    i32 vocab_size() const { return static_cast<i32>(vocab_.size()); }

    // Get token string
    const std::string& get_token(i32 id) const {
        static const std::string empty;
        if (id < 0 || id >= static_cast<i32>(vocab_.size())) return empty;
        return vocab_[id];
    }

    // Get token ID
    i32 get_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        return it != token_to_id_.end() ? it->second : SpecialTokens::UNK;
    }

    // Build vocabulary from text corpus (for training)
    void train(const std::vector<std::string>& texts, i32 vocab_size, i32 min_freq = 2) {
        // Initialize with special tokens and byte tokens
        vocab_.clear();
        scores_.clear();
        token_to_id_.clear();

        // Add special tokens
        add_token("<pad>", 0.0f);   // 0
        add_token("<bos>", 0.0f);   // 1
        add_token("<eos>", 0.0f);   // 2
        add_token("<unk>", 0.0f);   // 3

        // Add byte tokens (256 tokens for raw bytes)
        for (i32 i = 0; i < 256; ++i) {
            std::string s(1, static_cast<char>(i));
            add_token(s, -100.0f);  // Low priority
        }

        // Count character frequencies
        std::unordered_map<std::string, i64> char_freq;
        for (const auto& text : texts) {
            for (size_t i = 0; i < text.size();) {
                size_t len = utf8_char_len(text[i]);
                char_freq[text.substr(i, len)]++;
                i += len;
            }
        }

        // Add frequent characters to vocab
        for (const auto& [ch, freq] : char_freq) {
            if (freq >= min_freq && token_to_id_.find(ch) == token_to_id_.end()) {
                add_token(ch, static_cast<f32>(std::log(freq)));
            }
        }

        // BPE training: iteratively find and add most frequent pairs
        std::unordered_map<std::string, i64> pair_freq;
        std::vector<std::vector<std::string>> tokenized_texts;

        // Initial tokenization
        for (const auto& text : texts) {
            std::vector<std::string> chars;
            for (size_t i = 0; i < text.size();) {
                size_t len = utf8_char_len(text[i]);
                chars.push_back(text.substr(i, len));
                i += len;
            }
            tokenized_texts.push_back(chars);
        }

        // BPE iterations
        while (static_cast<i32>(vocab_.size()) < vocab_size) {
            pair_freq.clear();

            // Count pair frequencies
            for (const auto& tokens : tokenized_texts) {
                for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                    std::string pair = tokens[i] + tokens[i + 1];
                    pair_freq[pair]++;
                }
            }

            if (pair_freq.empty()) break;

            // Find most frequent pair
            auto best = std::max_element(pair_freq.begin(), pair_freq.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            if (best->second < min_freq) break;

            // Add new token
            std::string new_token = best->first;
            f32 score = static_cast<f32>(std::log(best->second));
            add_token(new_token, score);

            // Merge in all texts
            for (auto& tokens : tokenized_texts) {
                for (size_t i = 0; i + 1 < tokens.size();) {
                    if (tokens[i] + tokens[i + 1] == new_token) {
                        tokens[i] = new_token;
                        tokens.erase(tokens.begin() + i + 1);
                    } else {
                        ++i;
                    }
                }
            }
        }
    }

    // Format chat message
    std::string format_chat(const std::string& role, const std::string& content) const {
        if (role == "user") {
            return "<user>" + content + "</user>";
        } else if (role == "assistant") {
            return "<assistant>" + content + "</assistant>";
        } else if (role == "system") {
            return "<system>" + content + "</system>";
        }
        return content;
    }

private:
    void add_token(const std::string& token, f32 score) {
        i32 id = static_cast<i32>(vocab_.size());
        vocab_.push_back(token);
        scores_.push_back(score);
        token_to_id_[token] = id;
    }

    static size_t utf8_char_len(unsigned char c) {
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1;
    }

    std::vector<std::string> vocab_;
    std::vector<f32> scores_;
    std::unordered_map<std::string, i32> token_to_id_;
};

} // namespace lai

#endif // LAI_TOKENIZER_BPE_H

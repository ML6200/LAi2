#ifndef LAI_CLI_REPL_H
#define LAI_CLI_REPL_H

#include "../inference/engine.h"
#include "../model/config.h"
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace lai {

// ANSI color codes
namespace color {
    constexpr const char* RESET = "\033[0m";
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* YELLOW = "\033[33m";
    constexpr const char* BLUE = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN = "\033[36m";
}

// Operating mode
enum class Mode {
    CHAT,
    TRANSLATE,
    CODE,
    TEXT
};

inline const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::CHAT: return "chat";
        case Mode::TRANSLATE: return "translate";
        case Mode::CODE: return "code";
        case Mode::TEXT: return "text";
    }
    return "unknown";
}

// REPL (Read-Eval-Print Loop) for interactive use
class REPL {
public:
    REPL() : mode_(Mode::CHAT), running_(false), verbose_(false) {}

    // Initialize with model (vocab is embedded in model)
    bool init(const std::string& model_path, const std::string& vocab_path = "",
              bool use_mmap = true, const std::string& backend_name = "auto") {
        std::cout << color::CYAN << "Loading model..." << color::RESET << std::endl;

        if (!engine_.init(model_path, vocab_path, use_mmap, backend_name)) {
            std::cout << color::RED << "Failed to load model!" << color::RESET << std::endl;
            return false;
        }

        print_info();
        return true;
    }

    // Run interactive loop
    void run() {
        running_ = true;
        print_welcome();

        std::string line;
        while (running_) {
            print_prompt();

            if (!std::getline(std::cin, line)) {
                break;  // EOF
            }

            line = trim(line);
            if (line.empty()) continue;

            if (line[0] == '/') {
                handle_command(line);
            } else {
                handle_input(line);
            }
        }

        std::cout << "\nGoodbye!\n";
    }

    // Process single input (non-interactive)
    void process(const std::string& input) {
        handle_input(input);
    }

private:
    void print_welcome() {
        std::cout << "\n";
        std::cout << color::BOLD << color::CYAN;
        std::cout << "╔═══════════════════════════════════════════════╗\n";
        std::cout << "║     LAi - Lightweight AI Assistant            ║\n";
        std::cout << "║     Hungarian + English | CPU Optimized       ║\n";
        std::cout << "╚═══════════════════════════════════════════════╝\n";
        std::cout << color::RESET << "\n";

        std::cout << color::DIM;
        std::cout << "Type /help for commands, /quit to exit\n";
        std::cout << "Current mode: " << mode_name(mode_) << "\n";
        std::cout << color::RESET << "\n";
    }

    void print_info() {
        const auto& cfg = engine_.config();
        std::cout << color::DIM;
        std::cout << "Model: " << cfg.n_layers << " layers, "
                  << cfg.dim << " dim, "
                  << cfg.n_heads << " heads\n";
        std::cout << "Memory: " << (engine_.memory_bytes() / (1024 * 1024)) << " MB\n";
        std::cout << "Vocab: " << engine_.tokenizer().vocab_size() << " tokens\n";
        std::cout << "Backend: " << engine_.backend_name() << "\n";
        std::cout << color::RESET;
    }

    void print_prompt() {
        std::cout << color::GREEN << "[" << mode_name(mode_) << "] "
                  << color::BOLD << "> " << color::RESET;
        std::cout.flush();
    }

    void print_help() {
        std::cout << color::BOLD << "\nCommands:\n" << color::RESET;
        std::cout << "  /chat      - Switch to chat mode\n";
        std::cout << "  /translate - Switch to translation mode\n";
        std::cout << "  /code      - Switch to code assistance mode\n";
        std::cout << "  /text      - Switch to text processing mode\n";
        std::cout << "  /hu <text> - Translate English to Hungarian\n";
        std::cout << "  /en <text> - Translate Hungarian to English\n";
        std::cout << "  /temp <n>  - Set temperature (0.0-2.0)\n";
        std::cout << "  /tokens <n>- Set max tokens\n";
        std::cout << "  /top_k <n> - Set top-k sampling (0 = disabled)\n";
        std::cout << "  /top_p <n> - Set top-p/nucleus (0.0-1.0)\n";
        std::cout << "  /min_p <n> - Set min-p threshold (0.0-1.0, 0 = disabled)\n";
        std::cout << "  /repeat <n>- Set repetition penalty (1.0 = disabled)\n";
        std::cout << "  /dry <n>   - Set DRY multiplier (0 = disabled)\n";
        std::cout << "  /mirostat <tau> - Enable Mirostat v2 (0 = disabled)\n";
        std::cout << "  /freq <n>  - Set frequency penalty (0 = disabled)\n";
        std::cout << "  /pres <n>  - Set presence penalty (0 = disabled)\n";
        std::cout << "  /reset     - Reset conversation context\n";
        std::cout << "  /stats     - Toggle statistics display\n";
        std::cout << "  /info      - Show model information\n";
        std::cout << "  /help      - Show this help\n";
        std::cout << "  /quit      - Exit\n";
        std::cout << "\n";
    }

    void handle_command(const std::string& cmd) {
        std::istringstream iss(cmd);
        std::string command;
        iss >> command;

        if (command == "/quit" || command == "/exit" || command == "/q") {
            running_ = false;
        }
        else if (command == "/help" || command == "/?") {
            print_help();
        }
        else if (command == "/chat") {
            mode_ = Mode::CHAT;
            std::cout << "Switched to chat mode\n";
        }
        else if (command == "/translate") {
            mode_ = Mode::TRANSLATE;
            std::cout << "Switched to translation mode\n";
        }
        else if (command == "/code") {
            mode_ = Mode::CODE;
            std::cout << "Switched to code assistance mode\n";
        }
        else if (command == "/text") {
            mode_ = Mode::TEXT;
            std::cout << "Switched to text processing mode\n";
        }
        else if (command == "/hu") {
            std::string text;
            std::getline(iss >> std::ws, text);
            if (!text.empty()) {
                translate_text(text, true);
            } else {
                std::cout << "Usage: /hu <english text to translate>\n";
            }
        }
        else if (command == "/en") {
            std::string text;
            std::getline(iss >> std::ws, text);
            if (!text.empty()) {
                translate_text(text, false);
            } else {
                std::cout << "Usage: /en <hungarian text to translate>\n";
            }
        }
        else if (command == "/temp") {
            f32 temp;
            if (iss >> temp && temp >= 0.0f && temp <= 2.0f) {
                gen_config_.temperature = temp;
                std::cout << "Temperature set to " << temp << "\n";
            } else {
                std::cout << "Usage: /temp <0.0-2.0>\n";
            }
        }
        else if (command == "/tokens") {
            i32 tokens;
            if (iss >> tokens && tokens > 0 && tokens <= 2048) {
                gen_config_.max_tokens = tokens;
                std::cout << "Max tokens set to " << tokens << "\n";
            } else {
                std::cout << "Usage: /tokens <1-2048>\n";
            }
        }
        else if (command == "/top_k") {
            i32 k;
            if (iss >> k && k >= 0) {
                gen_config_.top_k = k;
                std::cout << "Top-k set to " << k << (k == 0 ? " (disabled)" : "") << "\n";
            } else {
                std::cout << "Usage: /top_k <0+> (0 = disabled)\n";
            }
        }
        else if (command == "/top_p") {
            f32 p;
            if (iss >> p && p >= 0.0f && p <= 1.0f) {
                gen_config_.top_p = p;
                std::cout << "Top-p set to " << p << "\n";
            } else {
                std::cout << "Usage: /top_p <0.0-1.0>\n";
            }
        }
        else if (command == "/min_p") {
            f32 p;
            if (iss >> p && p >= 0.0f && p <= 1.0f) {
                gen_config_.min_p = p;
                std::cout << "Min-p set to " << p << (p == 0.0f ? " (disabled)" : "") << "\n";
            } else {
                std::cout << "Usage: /min_p <0.0-1.0> (0 = disabled)\n";
            }
        }
        else if (command == "/repeat") {
            f32 r;
            if (iss >> r && r >= 1.0f) {
                gen_config_.repeat_penalty = r;
                std::cout << "Repetition penalty set to " << r << "\n";
            } else {
                std::cout << "Usage: /repeat <1.0+> (1.0 = disabled)\n";
            }
        }
        else if (command == "/dry") {
            f32 d;
            if (iss >> d && d >= 0.0f) {
                gen_config_.dry_multiplier = d;
                std::cout << "DRY multiplier set to " << d << (d == 0.0f ? " (disabled)" : "") << "\n";
            } else {
                std::cout << "Usage: /dry <0.0+> (0 = disabled)\n";
            }
        }
        else if (command == "/mirostat") {
            f32 tau;
            if (iss >> tau && tau >= 0.0f) {
                gen_config_.mirostat_tau = tau;
                std::cout << "Mirostat v2 tau set to " << tau << (tau == 0.0f ? " (disabled)" : "") << "\n";
            } else {
                std::cout << "Usage: /mirostat <tau> (0 = disabled, typical: 3.0-5.0)\n";
            }
        }
        else if (command == "/freq") {
            f32 f;
            if (iss >> f) {
                gen_config_.frequency_penalty = f;
                std::cout << "Frequency penalty set to " << f << "\n";
            } else {
                std::cout << "Usage: /freq <penalty> (0 = disabled)\n";
            }
        }
        else if (command == "/pres") {
            f32 p;
            if (iss >> p) {
                gen_config_.presence_penalty = p;
                std::cout << "Presence penalty set to " << p << "\n";
            } else {
                std::cout << "Usage: /pres <penalty> (0 = disabled)\n";
            }
        }
        else if (command == "/reset") {
            engine_.reset();
            conversation_.clear();
            std::cout << "Context reset\n";
        }
        else if (command == "/stats") {
            verbose_ = !verbose_;
            std::cout << "Statistics " << (verbose_ ? "enabled" : "disabled") << "\n";
        }
        else if (command == "/info") {
            print_info();
        }
        else {
            std::cout << color::YELLOW << "Unknown command. Type /help for commands.\n"
                      << color::RESET;
        }
    }

    void handle_input(const std::string& input) {
        GenerationStats stats;
        std::string response;

        // Stream callback - print tokens as they're generated
        auto stream_cb = [](const std::string& token, i32 /*token_id*/) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;  // Continue generation
        };

        std::cout << color::CYAN;

        switch (mode_) {
            case Mode::CHAT:
                response = engine_.chat(input, system_prompt_, gen_config_,
                                       stream_cb, verbose_ ? &stats : nullptr);
                conversation_.push_back({"user", input});
                conversation_.push_back({"assistant", response});
                break;

            case Mode::TRANSLATE:
                translate_text(input, true);  // Default: to Hungarian
                return;

            case Mode::CODE:
                response = engine_.code_assist(input, "", gen_config_,
                                              stream_cb, verbose_ ? &stats : nullptr);
                break;

            case Mode::TEXT:
                response = engine_.process_text(input, "", gen_config_,
                                               stream_cb, verbose_ ? &stats : nullptr);
                break;
        }

        std::cout << color::RESET << "\n";

        if (verbose_ && stats.generated_tokens > 0) {
            print_stats(stats);
        }
    }

    void translate_text(const std::string& text, bool to_hungarian) {
        GenerationStats stats;

        auto stream_cb = [](const std::string& token, i32 /*token_id*/) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;
        };

        std::cout << color::CYAN;

        engine_.translate(text, to_hungarian, gen_config_,
                         stream_cb);

        std::cout << color::RESET << "\n";
    }

    void print_stats(const GenerationStats& stats) {
        std::cout << color::DIM;
        std::cout << "\n[Stats: "
                  << stats.prompt_tokens << " prompt, "
                  << stats.generated_tokens << " generated, "
                  << std::fixed << std::setprecision(1)
                  << stats.tokens_per_second() << " tok/s, "
                  << stats.total_time_ms << " ms total]\n";
        std::cout << color::RESET;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    Engine engine_;
    Mode mode_;
    GenerationConfig gen_config_;
    bool running_;
    bool verbose_;

    std::string system_prompt_ = "You are a helpful AI assistant that speaks Hungarian and English fluently. "
                                 "Respond in the same language as the user's message.";

    std::vector<std::pair<std::string, std::string>> conversation_;
};

} // namespace lai

#endif // LAI_CLI_REPL_H

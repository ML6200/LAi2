#include "cli/repl.h"
#include "model/config.h"
#include <iostream>
#include <cstring>
#include <iomanip>

void print_usage(const char* prog) {
    std::cout << "LAi - Lightweight AI Assistant\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <path>    Path to model file (default: models/lai-mini.bin)\n";
    std::cout << "  -v, --vocab <path>    Path to vocabulary file (optional, uses embedded vocab)\n";
    std::cout << "  -p, --prompt <text>   Single prompt mode (non-interactive)\n";
    std::cout << "  -t, --translate <text> Translate text (EN->HU)\n";
    std::cout << "  --to-en <text>        Translate text (HU->EN)\n";
    std::cout << "  --temp <float>        Temperature (default: 0.7)\n";
    std::cout << "  --tokens <int>        Max tokens to generate (default: 256)\n";
    std::cout << "  --top-k <int>         Top-k sampling (default: 40, 0 = disabled)\n";
    std::cout << "  --top-p <float>       Top-p/nucleus sampling (default: 0.9)\n";
    std::cout << "  --min-p <float>       Min-p threshold (default: 0, 0 = disabled)\n";
    std::cout << "  --repeat <float>      Repetition penalty (default: 1.1)\n";
    std::cout << "  --freq-penalty <float> Frequency penalty (default: 0)\n";
    std::cout << "  --presence-penalty <float> Presence penalty (default: 0)\n";
    std::cout << "  --dry <float>         DRY n-gram penalty multiplier (default: 0)\n";
    std::cout << "  --mirostat-tau <float> Mirostat v2 target surprise (default: 0, 0 = disabled)\n";
    std::cout << "  --no-mmap             Disable memory-mapped loading\n";
    std::cout << "  --backend <name>      Compute backend: auto, cpu, metal (default: auto)\n";
    std::cout << "  --test                Run self-tests\n";
    std::cout << "  --bench               Run benchmarks\n";
    std::cout << "  --info                Show model info and exit\n";
    std::cout << "  -h, --help            Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "                      # Interactive mode\n";
    std::cout << "  " << prog << " -p \"Hello!\"          # Single prompt\n";
    std::cout << "  " << prog << " -t \"Hello world\"     # Translate to Hungarian\n";
    std::cout << "  " << prog << " --to-en \"Szia világ\" # Translate to English\n";
}

void run_tests() {
    std::cout << "Running tests...\n\n";

    // Test 1: Types
    std::cout << "Test 1: Types... ";
    {
        lai::Shape s(2, 3, 4);
        if (s.numel() != 24 || s.ndim != 3) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 2: Tensor
    std::cout << "Test 2: Tensor... ";
    {
        lai::Tensor t(lai::Shape(4, 4));
        t.fill(1.0f);
        if (t.at(0, 0) != 1.0f || t.numel() != 16) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 3: SIMD dot product
    std::cout << "Test 3: SIMD dot product... ";
    {
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        float result = lai::simd::dot_f32(a, b, 8);
        if (std::abs(result - 36.0f) > 0.001f) {
            std::cout << "FAILED (got " << result << ", expected 36)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 4: Arena allocator
    std::cout << "Test 4: Arena allocator... ";
    {
        lai::Arena arena(1024);
        void* p1 = arena.alloc(100);
        void* p2 = arena.alloc(200);
        if (!p1 || !p2 || arena.used() < 300) {
            std::cout << "FAILED\n";
            return;
        }
        arena.reset();
        if (arena.used() != 0) {
            std::cout << "FAILED (reset)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 5: Model config
    std::cout << "Test 5: Model config... ";
    {
        auto cfg = lai::presets::lai_mini();
        if (cfg.dim != 512 || cfg.n_layers != 12 || cfg.head_dim() != 64) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 6: Softmax
    std::cout << "Test 6: Softmax... ";
    {
        lai::Tensor x{lai::Shape{4}};
        lai::Tensor y{lai::Shape{4}};
        x.at(0) = 1.0f; x.at(1) = 2.0f; x.at(2) = 3.0f; x.at(3) = 4.0f;
        lai::ops::softmax(y, x);
        float sum = y.at(0) + y.at(1) + y.at(2) + y.at(3);
        if (std::abs(sum - 1.0f) > 0.001f) {
            std::cout << "FAILED (sum = " << sum << ")\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 7: Min-p filtering
    std::cout << "Test 7: Min-p filtering... ";
    {
        lai::Tensor logits{lai::Shape{4}};
        logits.at(0) = 10.0f; logits.at(1) = 5.0f; logits.at(2) = 1.0f; logits.at(3) = 0.1f;

        lai::GenerationConfig cfg;
        cfg.temperature = 1.0f;
        cfg.min_p = 0.1f;  // Keep tokens with prob >= 10% of max
        cfg.top_k = 0;
        cfg.top_p = 1.0f;
        cfg.repeat_penalty = 1.0f;

        lai::Sampler sampler(42);
        // Sample many times - token 3 (logit=0.1) should rarely/never appear
        int counts[4] = {0};
        for (int trial = 0; trial < 1000; ++trial) {
            lai::i32 token = sampler.sample(logits, cfg);
            if (token >= 0 && token < 4) counts[token]++;
        }
        // Token 0 (highest logit) should be most common
        if (counts[0] < counts[3]) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 8: DRY penalty
    std::cout << "Test 8: DRY penalty... ";
    {
        lai::Tensor logits{lai::Shape{8}};
        for (int i = 0; i < 8; ++i) logits.at(i) = 5.0f;  // Equal logits

        lai::GenerationConfig cfg;
        cfg.temperature = 0.001f;  // Near-greedy
        cfg.dry_multiplier = 10.0f;
        cfg.dry_allowed_length = 2;
        cfg.top_k = 0;
        cfg.top_p = 1.0f;
        cfg.repeat_penalty = 1.0f;
        cfg.min_p = 0.0f;

        // Simulate repeated pattern: [1, 2, 3, 1, 2, 3, 1, 2]
        // DRY should penalize token 3 (would complete the repeated n-gram)
        std::vector<lai::i32> recent = {1, 2, 3, 1, 2, 3, 1, 2};

        lai::Sampler sampler(42);
        lai::i32 token = sampler.sample(logits, cfg, recent);
        // Token 3 should be heavily penalized, so we should NOT get 3
        if (token == 3) {
            std::cout << "FAILED (got penalized token)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 9: Mirostat v2
    std::cout << "Test 9: Mirostat v2... ";
    {
        lai::Tensor logits{lai::Shape{16}};
        // Create a distribution with one dominant token
        for (int i = 0; i < 16; ++i) logits.at(i) = 0.0f;
        logits.at(0) = 10.0f;

        lai::GenerationConfig cfg;
        cfg.temperature = 1.0f;
        cfg.mirostat_tau = 5.0f;  // Target ~5 bits of surprise
        cfg.mirostat_eta = 0.1f;
        cfg.repeat_penalty = 1.0f;

        lai::Sampler sampler(42);
        // Should produce valid tokens without crashing
        bool valid = true;
        for (int trial = 0; trial < 100; ++trial) {
            lai::i32 token = sampler.sample(logits, cfg);
            if (token < 0 || token >= 16) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 10: F16 conversion round-trip
    std::cout << "Test 10: F16 conversion... ";
    {
        float values[] = {1.0f, -1.0f, 0.5f, 0.0f, 65504.0f, -0.001f};
        bool ok = true;
        for (float v : values) {
            lai::f16 h = lai::f32_to_f16(v);
            float r = lai::f16_to_f32(h);
            float err = std::abs(r - v) / std::max(1.0f, std::abs(v));
            if (err > 0.01f) { ok = false; break; }
        }
        if (!ok) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 11: Q8 quantize/dequantize round-trip
    std::cout << "Test 11: Q8 round-trip... ";
    {
        float data[32];
        for (int i = 0; i < 32; ++i) data[i] = (i - 16) * 0.1f;

        // Quantize
        float amax = 0;
        for (int i = 0; i < 32; ++i) {
            float a = std::abs(data[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        lai::Q8_0 block;
        block.d = lai::f32_to_f16(scale);
        for (int i = 0; i < 32; ++i) {
            float v = data[i] / scale;
            block.qs[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(v))));
        }

        // Dequantize
        float out[32];
        lai::simd::dequantize_q8_block(&block, out);

        float max_err = 0;
        for (int i = 0; i < 32; ++i) {
            float err = std::abs(out[i] - data[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 0.02f) {
            std::cout << "FAILED (max_err=" << max_err << ")\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 12: Q4 quantize/dequantize round-trip
    std::cout << "Test 12: Q4 round-trip... ";
    {
        float data[32];
        for (int i = 0; i < 32; ++i) data[i] = (i - 16) * 0.1f;

        // Quantize to Q4
        float amax = 0;
        for (int i = 0; i < 32; ++i) {
            float a = std::abs(data[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 7.0f;
        lai::Q4_0 block;
        block.d = lai::f32_to_f16(scale);
        for (int j = 0; j < 16; ++j) {
            int lo = static_cast<int>(std::max(-8.0f, std::min(7.0f, std::round(data[j * 2] / scale))));
            int hi = static_cast<int>(std::max(-8.0f, std::min(7.0f, std::round(data[j * 2 + 1] / scale))));
            block.qs[j] = static_cast<uint8_t>(((lo + 8) & 0x0F) | (((hi + 8) & 0x0F) << 4));
        }

        // Dequantize
        float out[32];
        lai::simd::dequantize_q4_block(&block, out);

        float max_err = 0;
        for (int i = 0; i < 32; ++i) {
            float err = std::abs(out[i] - data[i]);
            if (err > max_err) max_err = err;
        }
        // Q4 has lower precision, allow more error
        if (max_err > 0.3f) {
            std::cout << "FAILED (max_err=" << max_err << ")\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 13: Q4 dot product
    std::cout << "Test 13: Q4 dot product... ";
    {
        // Create a Q4 block with known values and compute dot product
        lai::Q4_0 block;
        float scale = 1.0f;
        block.d = lai::f32_to_f16(scale);
        // Set all quantized values to 1 (stored as 1+8=9 in unsigned nibble)
        for (int j = 0; j < 16; ++j) {
            block.qs[j] = ((9) & 0x0F) | ((9 & 0x0F) << 4);  // both nibbles = 1
        }

        float b[32];
        for (int i = 0; i < 32; ++i) b[i] = 1.0f;

        float result = lai::simd::dot_q4_f32(&block, b, 32);
        // All 32 values are 1.0 * 1.0 = 1.0, sum = 32
        if (std::abs(result - 32.0f) > 0.1f) {
            std::cout << "FAILED (got " << result << ", expected 32)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 14: Metal backend matvec
#ifdef LAI_METAL
    std::cout << "Test 14: Metal backend... ";
    {
        auto* backend = lai::Backend::create_metal();
        if (!backend) {
            std::cout << "SKIPPED (no Metal device)\n";
        } else {
            // Create a small matrix and vector
            const int M = 64, K = 64;
            lai::Tensor A{lai::Shape(M, K)};
            lai::Tensor x{lai::Shape(K)};
            lai::Tensor y_gpu{lai::Shape(M)};
            lai::Tensor y_cpu{lai::Shape(M)};

            // Fill with known values
            for (int i = 0; i < M * K; ++i) A.data_f32()[i] = static_cast<float>(i % 7) * 0.1f;
            for (int i = 0; i < K; ++i) x.data_f32()[i] = static_cast<float>(i % 5) * 0.2f;

            // CPU reference
            lai::ops::matvec(y_cpu, A, x);

            // GPU
            backend->matvec(y_gpu, A, x);

            // Compare
            float max_err = 0;
            for (int i = 0; i < M; ++i) {
                float err = std::abs(y_gpu.data_f32()[i] - y_cpu.data_f32()[i]);
                if (err > max_err) max_err = err;
            }

            delete backend;

            if (max_err > 0.01f) {
                std::cout << "FAILED (max_err=" << max_err << ")\n";
                return;
            }
            std::cout << "OK (max_err=" << max_err << ")\n";
        }
    }
#else
    std::cout << "Test 14: Metal backend... SKIPPED (not macOS)\n";
#endif

    std::cout << "\nAll tests passed!\n";
}

void run_benchmarks() {
    std::cout << "Running benchmarks...\n\n";

    const int N = 512;
    const int K = 512;
    const int iters = 100;

    // Benchmark: Matrix-vector multiply
    std::cout << "Benchmark: Matrix-vector multiply (" << N << "x" << K << ")...\n";
    {
        lai::Tensor A{lai::Shape(N, K)};
        lai::Tensor x{lai::Shape(K)};
        lai::Tensor y{lai::Shape(N)};

        A.fill(0.01f);
        x.fill(1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            lai::ops::matvec(y, A, x);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_iter = 2.0 * N * K;  // multiply + add
        double gflops = (ops_per_iter * iters) / (ms * 1e6);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << ms << " ms (" << iters << " iters)\n";
        std::cout << "  Performance: " << gflops << " GFLOP/s\n";
    }

    // Benchmark: Dot product
    std::cout << "\nBenchmark: Dot product (" << N * K << " elements)...\n";
    {
        std::vector<float> a(N * K, 0.01f);
        std::vector<float> b(N * K, 0.01f);

        auto start = std::chrono::high_resolution_clock::now();
        float result = 0;
        for (int i = 0; i < iters * 10; ++i) {
            result += lai::simd::dot_f32(a.data(), b.data(), N * K);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_iter = 2.0 * N * K;
        double gflops = (ops_per_iter * iters * 10) / (ms * 1e6);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << ms << " ms (" << iters * 10 << " iters)\n";
        std::cout << "  Performance: " << gflops << " GFLOP/s\n";
        (void)result;  // Prevent optimization
    }

    // Benchmark: Q4 matrix-vector multiply
    std::cout << "\nBenchmark: Q4 Matrix-vector multiply (" << N << "x" << K << ")...\n";
    {
        // Allocate Q4 weight matrix
        lai::i64 blocks_per_row = K / 32;
        lai::i64 total_blocks = N * blocks_per_row;
        std::vector<lai::Q4_0> q4_data(total_blocks);
        for (lai::i64 i = 0; i < total_blocks; ++i) {
            q4_data[i].d = lai::f32_to_f16(0.01f);
            for (int j = 0; j < 16; ++j) q4_data[i].qs[j] = 0x99; // value 1 in both nibbles
        }

        lai::Tensor x{lai::Shape(K)};
        lai::Tensor y{lai::Shape(N)};
        x.fill(1.0f);

        lai::TensorView A_q4(q4_data.data(), lai::Shape(N, K), lai::DType::Q4_0);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            lai::ops::matvec_q4(y, A_q4, x);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_iter = 2.0 * N * K;
        double gflops = (ops_per_iter * iters) / (ms * 1e6);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << ms << " ms (" << iters << " iters)\n";
        std::cout << "  Performance: " << gflops << " GFLOP/s\n";
    }

    // Benchmark: Metal GPU vs CPU (larger matrix for meaningful comparison)
#ifdef LAI_METAL
    std::cout << "\nBenchmark: Metal GPU matvec (2048x2048)...\n";
    {
        const int GM = 2048, GK = 2048;
        const int g_iters = 50;

        lai::Tensor gA{lai::Shape(GM, GK)};
        lai::Tensor gx{lai::Shape(GK)};
        lai::Tensor gy{lai::Shape(GM)};
        gA.fill(0.01f);
        gx.fill(1.0f);

        // CPU baseline
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < g_iters; ++i) {
            lai::ops::matvec(gy, gA, gx);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // Metal GPU
        auto* metal = lai::Backend::create_metal();
        if (metal) {
            // Warm-up
            metal->matvec(gy, gA, gx);

            auto gpu_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < g_iters; ++i) {
                metal->matvec(gy, gA, gx);
            }
            auto gpu_end = std::chrono::high_resolution_clock::now();
            double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

            double cpu_gflops = (2.0 * GM * GK * g_iters) / (cpu_ms * 1e6);
            double gpu_gflops = (2.0 * GM * GK * g_iters) / (gpu_ms * 1e6);

            std::cout << "  CPU: " << std::fixed << std::setprecision(2)
                      << cpu_ms << " ms (" << cpu_gflops << " GFLOP/s)\n";
            std::cout << "  GPU: " << gpu_ms << " ms (" << gpu_gflops << " GFLOP/s)\n";
            std::cout << "  Speedup: " << std::setprecision(1) << cpu_ms / gpu_ms << "x\n";

            delete metal;
        } else {
            std::cout << "  Metal not available, skipping GPU benchmark\n";
        }
    }
#endif

    std::cout << "\nBenchmarks complete!\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/lai-mini.bin";
    std::string vocab_path = "";  // Empty = use embedded vocab
    std::string prompt;
    std::string translate_text;
    bool to_hungarian = true;
    float temperature = 0.7f;
    int max_tokens = 256;
    int top_k = 40;
    float top_p = 0.9f;
    float min_p = 0.0f;
    float repeat_penalty = 1.1f;
    float freq_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float dry_multiplier = 0.0f;
    float mirostat_tau = 0.0f;
    std::string backend_name = "auto";
    bool no_mmap = false;
    bool run_test = false;
    bool run_bench = false;
    bool show_info = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_path = argv[i];
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0) {
            if (++i < argc) vocab_path = argv[i];
        }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i < argc) prompt = argv[i];
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--translate") == 0) {
            if (++i < argc) {
                translate_text = argv[i];
                to_hungarian = true;
            }
        }
        else if (strcmp(argv[i], "--to-en") == 0) {
            if (++i < argc) {
                translate_text = argv[i];
                to_hungarian = false;
            }
        }
        else if (strcmp(argv[i], "--temp") == 0) {
            if (++i < argc) temperature = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--tokens") == 0) {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        }
        else if (strcmp(argv[i], "--top-k") == 0) {
            if (++i < argc) top_k = std::stoi(argv[i]);
        }
        else if (strcmp(argv[i], "--top-p") == 0) {
            if (++i < argc) top_p = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--min-p") == 0) {
            if (++i < argc) min_p = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--repeat") == 0) {
            if (++i < argc) repeat_penalty = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--freq-penalty") == 0) {
            if (++i < argc) freq_penalty = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--presence-penalty") == 0) {
            if (++i < argc) presence_penalty = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--dry") == 0) {
            if (++i < argc) dry_multiplier = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--mirostat-tau") == 0) {
            if (++i < argc) mirostat_tau = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--backend") == 0) {
            if (++i < argc) backend_name = argv[i];
        }
        else if (strcmp(argv[i], "--no-mmap") == 0) {
            no_mmap = true;
        }
        else if (strcmp(argv[i], "--test") == 0) {
            run_test = true;
        }
        else if (strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        }
        else if (strcmp(argv[i], "--info") == 0) {
            show_info = true;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Run tests
    if (run_test) {
        run_tests();
        return 0;
    }

    // Run benchmarks
    if (run_bench) {
        run_benchmarks();
        return 0;
    }

    // Show info
    if (show_info) {
        // Try loading actual model for real info
        lai::Transformer model;
        if (model.load(model_path)) {
            auto cfg = model.config();
            std::cout << "Model: " << model_path << "\n";
            std::cout << "  Layers: " << cfg.n_layers << "\n";
            std::cout << "  Dimension: " << cfg.dim << "\n";
            std::cout << "  Heads: " << cfg.n_heads << "\n";
            std::cout << "  Head dim: " << cfg.head_dim() << "\n";
            std::cout << "  FFN hidden: " << cfg.hidden_dim << "\n";
            std::cout << "  Vocab size: " << cfg.vocab_size << "\n";
            std::cout << "  Max seq len: " << cfg.max_seq_len << "\n";
            std::cout << "  Parameters: ~" << (cfg.param_count() / 1000000) << "M\n";
            std::cout << "  Weight dtype: " << lai::dtype_name(model.weight_dtype()) << "\n";
            std::cout << "  Memory: ~" << (model.memory_bytes() / (1024 * 1024)) << " MB\n";
#ifdef LAI_METAL
            std::cout << "  Metal GPU: available\n";
#else
            std::cout << "  Metal GPU: not available\n";
#endif
        } else {
            // Fall back to preset info
            auto cfg = lai::presets::lai_mini();
            std::cout << "LAi-Mini Model Configuration (preset):\n";
            std::cout << "  Layers: " << cfg.n_layers << "\n";
            std::cout << "  Dimension: " << cfg.dim << "\n";
            std::cout << "  Heads: " << cfg.n_heads << "\n";
            std::cout << "  Head dim: " << cfg.head_dim() << "\n";
            std::cout << "  FFN hidden: " << cfg.hidden_dim << "\n";
            std::cout << "  Vocab size: " << cfg.vocab_size << "\n";
            std::cout << "  Max seq len: " << cfg.max_seq_len << "\n";
            std::cout << "  Parameters: ~" << (cfg.param_count() / 1000000) << "M\n";
            std::cout << "  Memory (F32): ~" << (cfg.memory_bytes() / (1024 * 1024)) << " MB\n";
            std::cout << "  Memory (Q4): ~" << (cfg.memory_bytes(lai::DType::Q4_0) / (1024 * 1024)) << " MB\n";
        }
        return 0;
    }

    // Interactive mode or single prompt
    lai::REPL repl;

    // Check if model exists, if not, print helpful message
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cout << "\n";
        std::cout << lai::color::YELLOW;
        std::cout << "Model not found at: " << model_path << "\n\n";
        std::cout << "To use LAi, you need to:\n";
        std::cout << "1. Train a model using the training scripts in training/\n";
        std::cout << "2. Or download a pre-trained model\n\n";
        std::cout << "For now, you can run:\n";
        std::cout << "  ./lai --test   # Run self-tests\n";
        std::cout << "  ./lai --bench  # Run benchmarks\n";
        std::cout << "  ./lai --info   # Show model specs\n";
        std::cout << lai::color::RESET << "\n";
        return 1;
    }
    fclose(f);

    if (!repl.init(model_path, vocab_path, !no_mmap, backend_name)) {
        return 1;
    }

    // Non-interactive modes
    if (!translate_text.empty()) {
        lai::Engine engine;
        if (!engine.init(model_path, vocab_path, !no_mmap, backend_name)) {
            std::cerr << "Failed to load model or vocabulary!\n";
            return 1;
        }

        lai::GenerationConfig cfg;
        cfg.temperature = temperature;
        cfg.max_tokens = max_tokens;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
        cfg.min_p = min_p;
        cfg.repeat_penalty = repeat_penalty;
        cfg.frequency_penalty = freq_penalty;
        cfg.presence_penalty = presence_penalty;
        cfg.dry_multiplier = dry_multiplier;
        cfg.mirostat_tau = mirostat_tau;

        auto stream_cb = [](const std::string& token, lai::i32) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;
        };

        engine.translate(translate_text, to_hungarian, cfg, stream_cb);
        std::cout << "\n";
        return 0;
    }

    if (!prompt.empty()) {
        repl.process(prompt);
        return 0;
    }

    // Interactive mode
    repl.run();

    return 0;
}

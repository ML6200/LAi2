# LAi - Lightweight AI Assistant

# ⚠️ Warning

**This repository is under active development and currently not functional.**

---

A from-scratch efficient LLM assistant written in pure C++ for CPU-only inference on low-end hardware created with Claude Code by experiment. Optimized for Hungarian and English. 

## Features

- **Pure C++ inference** - No Python dependencies for running
- **Extreme efficiency** - Runs on 4GB RAM with CPU-only
- **Bilingual** - Hungarian and English support
- **SIMD optimized** - AVX2 (x86) and NEON (ARM) acceleration
- **Multiple modes** - Chat, Translation, Code, Text processing
- **Embedded vocabulary** - Model file includes vocab, no separate files needed

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Copy-paste commands to get started in 5 minutes
- **[TRAINING.md](TRAINING.md)** - Complete training guide with troubleshooting
- **[CLAUDE.md](CLAUDE.md)** - Architecture details and development guide

## Quick Start

### Build

```bash
# Release build (optimized)
make release

# Debug build (with sanitizers)
make debug

# Run tests
make test

# Run benchmarks
make bench
```

### Run

```bash
# Interactive mode
./lai

# Single prompt
./lai -p "Hello, how are you?"

# Translate English to Hungarian
./lai -t "Hello world"

# Translate Hungarian to English
./lai --to-en "Szia világ"

# Show model info
./lai --info
```

## Training

### Quick Start (Synthetic Data)

1. **Install dependencies:**
```bash
pip install -r training/requirements.txt
```

2. **Generate training data:**
```bash
# Generate diverse synthetic data (25K examples)
python training/generate_data.py --output data/train.txt \
    --sentences 10000 --stories 5000 --translations 3000 --instructions 2000
```

3. **Build BPE vocabulary:**
```bash
# Build 8K token vocabulary
python training/build_vocab.py --data data/train.txt \
    --vocab_size 8000 --output data/vocab.bin
```

4. **Train the model:**
```bash
# On Apple Silicon (MPS)
python training/train.py --config tiny --epochs 10 --batch_size 32 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device mps

# On NVIDIA GPU (CUDA)
python training/train.py --config tiny --epochs 10 --batch_size 64 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device cuda

# On CPU (slower)
python training/train.py --config tiny --epochs 10 --batch_size 16 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device cpu
```

5. **Test the model:**
```bash
./lai -m models/lai-tiny.bin -p "Hello, how are you?"
```

### Using Real Datasets (Better Quality)

For production-quality models, use real datasets:

```bash
# Requires HuggingFace datasets (pip install datasets)
python training/data.py --output data/train.txt --size small
```

This downloads:
- Hungarian Wikipedia articles
- English TinyStories dataset
- Translation pairs from OPUS-100

### Model Sizes

| Config | Parameters | Training Time (MPS) | RAM Usage |
|--------|-----------|---------------------|-----------|
| `tiny` | 19M | ~15 min (10 epochs) | 2GB |
| `mini` | 83M | ~1 hour (10 epochs) | 4GB |
| `small` | 350M | ~4 hours (10 epochs) | 8GB |

### Important Notes

- **Vocabulary is embedded** in model files - no separate vocab file needed for inference
- **MPS users**: The script automatically disables DataLoader multiprocessing
- **Progress tracking**: Loss should decrease from ~8-10 to <1.0 over 10 epochs
- **Checkpoint saving**: Models are saved only at the end of training

### Google Colab Notebook

See `training/colab_notebook.ipynb` for a complete training setup that works on free Colab GPUs.

## Model Specifications

### LAi-Mini (Default)
- Parameters: ~150M
- Layers: 12
- Dimension: 512
- Heads: 8
- Context: 1024 tokens
- Memory (Q4): ~75MB
- Memory (F32): ~600MB

### LAi-Tiny (Ultra-low memory)
- Parameters: ~50M
- Layers: 8
- Dimension: 384
- Heads: 6
- Context: 512 tokens
- Memory (Q4): ~25MB

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Memory | <500MB | With Q4 quantization |
| Speed | 15+ tok/s | On modern CPU |
| Startup | <1 sec | Cold start |

## CLI Commands

When running in interactive mode:

| Command | Description |
|---------|-------------|
| `/chat` | Switch to chat mode |
| `/translate` | Switch to translation mode |
| `/code` | Switch to code assistance |
| `/text` | Switch to text processing |
| `/hu <text>` | Translate English → Hungarian |
| `/en <text>` | Translate Hungarian → English |
| `/temp <n>` | Set temperature (0.0-2.0) |
| `/tokens <n>` | Set max tokens |
| `/reset` | Reset conversation context |
| `/stats` | Toggle statistics display |
| `/info` | Show model information |
| `/help` | Show help |
| `/quit` | Exit |

## Project Structure

```
LAi/
├── src/
│   ├── core/           # Tensor ops, SIMD, memory allocator
│   ├── model/          # Transformer architecture
│   ├── tokenizer/      # BPE tokenizer
│   ├── inference/      # Engine, sampler, KV-cache
│   └── cli/            # REPL interface
├── training/           # PyTorch training scripts
├── data/               # Vocabulary, prompts
└── models/             # Trained model weights
```

## Building from Source

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Make or CMake

### Platform Support
- Linux (x86-64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (with MinGW or MSVC)

### Build Options

```bash
# Debug with AddressSanitizer
make debug

# Release with LTO
make release

# Check memory leaks
make valgrind

# Run performance benchmarks
make bench
```

## Architecture

The model uses a modern transformer architecture:

- **RMSNorm** - Efficient layer normalization
- **RoPE** - Rotary position embeddings
- **SwiGLU** - Improved FFN activation
- **GQA** - Grouped query attention (optional)

### Memory Optimizations

1. **Q4 Quantization** - 4-bit weights reduce memory 4x
2. **KV-Cache** - Efficient autoregressive generation
3. **Arena Allocator** - Zero-allocation inference
4. **SIMD** - Vectorized operations

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Acknowledgments

- Inspired by llama.cpp, but built from scratch
- Hungarian NLP community for language resources
- TinyStories dataset for English training data

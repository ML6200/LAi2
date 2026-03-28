#!/bin/bash
set -e

# LAi Micro Training Pipeline
# Optimized for low-end hardware and EN/HU support

# 1. Data Preparation
if [ ! -f "data/train_micro.txt" ]; then
    echo "--- Step 1: Preparing Micro Dataset (EN + HU) ---"
    python3 training/data.py --output data/train_micro.txt --size micro
else
    echo "--- Step 1: Dataset already exists, skipping ---"
fi

# 2. Build Vocabulary (Optimized for 16K tokens)
if [ ! -f "data/vocab_micro.bin" ]; then
    echo "--- Step 2: Building Vocabulary (16K tokens) ---"
    python3 training/build_vocab.py \
        --data data/train_micro.txt \
        --vocab_size 16000 \
        --output data/vocab_micro.bin
else
    echo "--- Step 2: Vocabulary already exists, skipping ---"
fi

# 3. Train Micro Model
echo "--- Step 3: Training Micro Model (10M parameters) ---"
# Detect device
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)"; then
    DEVICE="mps"
fi
echo "Using device: $DEVICE"

python3 training/train.py \
    --config micro \
    --data data/train_micro.txt \
    --vocab data/vocab_micro.bin \
    --output models/lai-micro.bin \
    --epochs 10 \
    --batch_size 128 \
    --device $DEVICE

echo "--- Training Complete! ---"
echo "Model saved to: models/lai-micro.bin"
echo "To test the model, run:"
echo "./lai -m models/lai-micro.bin -p \"Szia, hogy vagy?\""

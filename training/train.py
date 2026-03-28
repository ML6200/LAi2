#!/usr/bin/env python3
"""
LAi Training Script
Train a small transformer model for Hungarian + English

Usage:
    python train.py --config mini --epochs 10 --batch_size 32
"""

import os
import math
import json
import struct
import argparse
import psutil
import multiprocessing
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class ModelConfig:
    """Model configuration matching C++ implementation"""
    dim: int = 512
    hidden_dim: int = 2048
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @staticmethod
    def micro():
        return ModelConfig(dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=2, vocab_size=16000, max_seq_len=256)

    @staticmethod
    def tiny():
        return ModelConfig(dim=384, hidden_dim=1152, n_layers=8, n_heads=6, n_kv_heads=2, vocab_size=32000, max_seq_len=512)

    @staticmethod
    def mini():
        return ModelConfig(dim=512, hidden_dim=1536, n_layers=12, n_heads=8, n_kv_heads=4, vocab_size=32000, max_seq_len=1024)

    @staticmethod
    def small():
        return ModelConfig(dim=768, hidden_dim=2304, n_layers=16, n_heads=12, n_kv_heads=4, vocab_size=32000, max_seq_len=2048)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary embeddings"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to Q and K
    xq, xk: [batch, seq_len, n_heads, head_dim]
    freqs_cis: [seq_len, head_dim//2] complex
    """
    # Reshape to complex: [batch, seq_len, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis is [seq_len, head_dim//2], need to broadcast to [1, seq_len, 1, head_dim//2]
    freqs_cis = freqs_cis[:xq_.shape[1]]  # Trim to actual seq_len
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Attention
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Efficient attention (FlashAttention)
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                is_causal=True if mask is None else False
            )
        else:
            # Fallback for older PyTorch
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Transformer layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Full transformer model"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie embeddings
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len, config.rope_theta)
        )

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, use_checkpoint: bool = False):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Mask is handled implicitly by is_causal=True in Attention
        mask = None 

        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            # Gradient checkpointing only if requested and training
            if self.training and use_checkpoint:
                h = checkpoint(layer, h, freqs_cis, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, mask)

        h = self.norm(h)

        if targets is not None:
            # Full parallel loss calculation for CPU/GPU efficiency
            logits = self.output(h)
            # Flatten to [batch*seq, vocab]
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-100)
        else:
            # Inference mode: only compute last token logits usually
            logits = self.output(h)
            loss = None

        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def tokenize_worker(args):
    text, tokenizer_path, max_len = args
    # Re-load tokenizer in each worker because it's not easily picklable if it has large state
    # but since it's small we can just pass it or load it once
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(text)
    
    results = []
    if len(tokens) <= max_len:
        if len(tokens) > 2:
            results.append(tokens)
    else:
        for i in range(0, len(tokens) - 1, max_len):
            chunk = tokens[i:i + max_len + 1]
            if len(chunk) > 2:
                results.append(chunk)
    return results

class TextDataset(Dataset):
    """Simple text dataset for training (Parallel Optimized)"""
    def __init__(self, texts: List[str], tokenizer_path: str, max_len: int = 512):
        print(f"  Tokenizing {len(texts)} lines using {multiprocessing.cpu_count()} cores...")
        
        # Parallel tokenization
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as pool:
            # Map tokenize_worker across texts
            worker_args = [(text, tokenizer_path, max_len) for text in texts]
            
            if tqdm:
                results = list(tqdm(pool.imap(tokenize_worker, worker_args), total=len(texts), desc="  Tokenizing"))
            else:
                results = pool.map(tokenize_worker, worker_args)

        # Flatten results and track slices
        all_tokens = []
        self.slices = []
        current_offset = 0
        
        for text_chunks in results:
            for chunk in text_chunks:
                all_tokens.extend(chunk)
                self.slices.append((current_offset, len(chunk)))
                current_offset += len(chunk)

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        print(f"  Dataset: {len(self.slices)} samples, {len(self.data)} tokens loaded.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        start, length = self.slices[idx]
        chunk = self.data[start : start + length]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def collate_fn(batch):
    """Pad batch to same length"""
    max_len = max(len(x) for x, y in batch)
    xs = torch.full((len(batch), max_len), 0, dtype=torch.long)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)] = x
        ys[i, :len(y)] = y

    return xs, ys


class BPETokenizer:
    """BPE tokenizer that loads from binary vocab file"""
    def __init__(self):
        self.vocab = []
        self.vocab_to_id = {}
        self.scores = []

        # Special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def load(self, path: str):
        """Load vocabulary from binary file"""
        with open(path, 'rb') as f:
            magic = struct.unpack('I', f.read(4))[0]
            version = struct.unpack('I', f.read(4))[0]
            vocab_size = struct.unpack('I', f.read(4))[0]

            if magic != 0x4C414956:
                raise ValueError("Invalid vocab file magic")

            self.vocab = []
            self.scores = []
            self.vocab_to_id = {}

            for i in range(vocab_size):
                token_len = struct.unpack('I', f.read(4))[0]
                token = f.read(token_len).decode('utf-8')
                score = struct.unpack('f', f.read(4))[0]

                self.vocab.append(token)
                self.scores.append(score)
                self.vocab_to_id[token] = i

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text using BPE"""
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        if not text:
            if add_eos:
                tokens.append(self.eos_id)
            return tokens

        # UTF-8 aware character splitting
        chars = []
        i = 0
        while i < len(text):
            char_len = self._utf8_char_len(ord(text[i]))
            chars.append(text[i:i+char_len])
            i += char_len

        # Initial tokenization
        pieces = chars[:]

        # BPE merge loop
        while len(pieces) > 1:
            # Find best merge
            best_score = -1e10
            best_idx = -1
            best_merge = None

            for i in range(len(pieces) - 1):
                merged = pieces[i] + pieces[i + 1]
                if merged in self.vocab_to_id:
                    score = self.scores[self.vocab_to_id[merged]]
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_merge = merged

            if best_idx == -1:
                break

            # Apply merge
            pieces[best_idx] = best_merge
            pieces.pop(best_idx + 1)

        # Convert pieces to token IDs
        for piece in pieces:
            if piece in self.vocab_to_id:
                tokens.append(self.vocab_to_id[piece])
            else:
                # Encode as byte tokens
                for byte in piece.encode('utf-8'):
                    tokens.append(byte + 4)  # BYTE_OFFSET = 4

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        text = ""
        for token in tokens:
            if token < 0 or token >= len(self.vocab):
                continue
            if token in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            text += self.vocab[token]
        return text

    @staticmethod
    def _utf8_char_len(first_byte: int) -> int:
        if (first_byte & 0x80) == 0:
            return 1
        elif (first_byte & 0xE0) == 0xC0:
            return 2
        elif (first_byte & 0xF0) == 0xE0:
            return 3
        elif (first_byte & 0xF8) == 0xF0:
            return 4
        return 1


def export_model(model: Transformer, tokenizer: BPETokenizer, path: str):
    """Export model to C++ binary format with embedded vocabulary"""
    config = model.config

    with open(path, 'wb') as f:
        # Magic and version
        f.write(b'LAi1')
        f.write(struct.pack('I', 1))

        # Config (must match C++ struct layout)
        f.write(struct.pack('i', config.dim))
        f.write(struct.pack('i', config.hidden_dim))
        f.write(struct.pack('i', config.n_layers))
        f.write(struct.pack('i', config.n_heads))
        f.write(struct.pack('i', config.n_kv_heads))
        f.write(struct.pack('i', config.vocab_size))
        f.write(struct.pack('i', config.max_seq_len))
        f.write(struct.pack('f', config.rope_theta))
        f.write(struct.pack('f', config.norm_eps))
        f.write(struct.pack('i', 0))  # activation type

        # Weight dtype
        f.write(struct.pack('B', 0))  # F32

        # Offsets (placeholder)
        vocab_offset_pos = f.tell()
        f.write(struct.pack('Q', 0))  # vocab_offset
        weights_offset_pos = f.tell()
        f.write(struct.pack('Q', 0))  # weights_offset

        # Padding to 256 bytes for alignment
        header_size = f.tell()
        if header_size < 256:
            f.write(b'\x00' * (256 - header_size))

        # Write vocabulary
        vocab_offset = f.tell()
        f.seek(vocab_offset_pos)
        f.write(struct.pack('Q', vocab_offset))
        f.seek(vocab_offset)

        # Vocab format: magic, version, size, then tokens
        f.write(struct.pack('I', 0x4C414956))  # Magic: "LAIV"
        f.write(struct.pack('I', 1))            # Version
        f.write(struct.pack('I', len(tokenizer.vocab)))  # Vocab size

        for token, score in zip(tokenizer.vocab, tokenizer.scores):
            token_bytes = token.encode('utf-8')
            f.write(struct.pack('I', len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack('f', score))

        # Record weights offset
        weights_offset = f.tell()
        f.seek(weights_offset_pos)
        f.write(struct.pack('Q', weights_offset))
        f.seek(weights_offset)

        # Write weights in order expected by C++
        def write_tensor(tensor):
            data = tensor.detach().cpu().float().numpy()
            f.write(data.tobytes())

        # Embeddings
        write_tensor(model.tok_embeddings.weight)
        write_tensor(model.output.weight)
        write_tensor(model.norm.weight)

        # Layers
        for layer in model.layers:
            write_tensor(layer.attention.wq.weight)
            write_tensor(layer.attention.wk.weight)
            write_tensor(layer.attention.wv.weight)
            write_tensor(layer.attention.wo.weight)
            write_tensor(layer.feed_forward.w_gate.weight)
            write_tensor(layer.feed_forward.w_up.weight)
            write_tensor(layer.feed_forward.w_down.weight)
            write_tensor(layer.attention_norm.weight)
            write_tensor(layer.ffn_norm.weight)

    print(f"Model exported to {path}")
    print(f"  Config: {config.dim}d, {config.n_layers}L, {config.n_heads}H")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Parameters: {model.num_params() / 1e6:.1f}M")
    print(f"  File size: {os.path.getsize(path) / 1e6:.1f} MB")


def train(config: ModelConfig, train_texts: List[str], tokenizer: BPETokenizer,
          epochs: int, batch_size: int, lr: float, device: str,
          output_path: str, vocab_path: str):
    """Train the model"""
    # CPU Optimization for Xeon
    if device == "cpu":
        num_threads = psutil.cpu_count(logical=False)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1) # Usually best for single-node CPU training
        print(f"  CPU Training: Using {num_threads} threads")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Training LAi model:")
    print(f"  Config: {config.dim}d, {config.n_layers}L, {config.n_heads}H")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    if device == "cuda" and torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"  GPU Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
        if free_mem < 2 * 1024**3:
            print("  WARNING: Low GPU memory! Consider restarting runtime or reducing batch size further.")

    # Adjust config vocab size to match actual tokenizer size
    config.vocab_size = len(tokenizer.vocab)
    print(f"  Vocab Size: {config.vocab_size}")

    model = Transformer(config).to(device)
    print(f"  Parameters: {model.num_params() / 1e6:.1f}M")

    # Dataset and dataloader
    dataset = TextDataset(train_texts, vocab_path, config.max_seq_len)

    # Use more workers for Xeon CPU or MPS
    if device in ["cpu", "mps"]:
        num_workers = min(psutil.cpu_count(logical=False), 8)
    else:
        num_workers = 2
        
    pin_memory = device == 'cuda'

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=num_workers,
                           pin_memory=pin_memory)
    print(f"  Training samples: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Learning rate scheduler
    warmup_steps = 100
    total_steps = len(dataloader) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Mixed precision
    use_amp = device != 'cpu'
    device_type = 'cuda' if 'cuda' in device else ('cpu' if 'cpu' in device else 'mps')
    
    # Handle MPS (Apple Silicon)
    if device == 'mps':
        use_amp = True # Newer PyTorch supports MPS autocast
        # Disable GradScaler for MPS due to float64 bugs in some torch versions
        scaler = None
    else:
        scaler = GradScaler(device_type) if use_amp else None

    # Training loop
    model.train()
    global_step = 0
    process = psutil.Process()
    
    # Only use checkpointing for large models or if explicitly enabled
    use_checkpoint = config.n_layers >= 16

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type):
                    _, loss = model(x, y, use_checkpoint=use_checkpoint)
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                _, loss = model(x, y, use_checkpoint=use_checkpoint)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if batch_idx % 100 == 0:
                mem_rss = process.memory_info().rss / 1024 / 1024
                print(f"  Epoch {epoch+1}/{epochs}, Step {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}, "
                      f"RSS: {mem_rss:.0f}MB")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.bin"
        export_model(model, tokenizer, str(checkpoint_path))

    return model


def main():
    parser = argparse.ArgumentParser(description="Train LAi model")
    parser.add_argument("--config", choices=["micro", "tiny", "mini", "small"], default="mini")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data", type=str, help="Path to training data (text file)")
    parser.add_argument("--vocab", type=str, default="data/vocab.bin",
                       help="Path to vocabulary file (must exist)")
    parser.add_argument("--output", type=str, default="models/lai-mini.bin")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Get config
    if args.config == "micro":
        config = ModelConfig.micro()
    elif args.config == "tiny":
        config = ModelConfig.tiny()
    elif args.config == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig.mini()

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}...")
    if not os.path.exists(args.vocab):
        print(f"ERROR: Vocabulary file not found: {args.vocab}")
        print(f"Please run: python training/build_vocab.py --data <data.txt> --output {args.vocab}")
        return

    tokenizer = BPETokenizer()
    tokenizer.load(args.vocab)
    print(f"  Loaded {len(tokenizer.vocab)} tokens")

    # Load training data
    if args.data and os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Sample data for testing
        print("No training data provided. Using sample data for testing.")
        texts = [
            "Hello, how are you? I am fine, thank you!",
            "Szia! Hogy vagy? Köszönöm, jól vagyok.",
            "The quick brown fox jumps over the lazy dog.",
            "A gyors barna róka átugrik a lusta kutya felett.",
            "Machine learning is a subset of artificial intelligence.",
            "A gépi tanulás a mesterséges intelligencia része.",
            "Budapest is the capital of Hungary.",
            "Budapest Magyarország fővárosa.",
        ] * 1000  # Repeat for more training data

    # Train
    model = train(config, texts, tokenizer, args.epochs, args.batch_size, args.lr, args.device, args.output, args.vocab)

    # Export (vocabulary is embedded in model file now)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    export_model(model, tokenizer, args.output)

    print(f"\nTraining complete!")
    print(f"  Model saved to: {args.output}")
    print(f"  Vocabulary is embedded in the model file")


if __name__ == "__main__":
    main()

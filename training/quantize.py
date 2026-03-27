#!/usr/bin/env python3
"""Quantize an existing LAi F32 model to Q4_0 or Q8_0 format."""

import struct
import numpy as np
import argparse
import os
import sys


# DType enum values matching C++ DType
DTYPE_F32  = 0
DTYPE_F16  = 1
DTYPE_Q4_0 = 2
DTYPE_Q8_0 = 3

BLOCK_SIZE = 32


def read_header(f):
    """Read and parse LAi model header."""
    magic = f.read(4)
    if magic != b'LAi1':
        raise ValueError(f"Invalid magic: {magic}")

    version = struct.unpack('I', f.read(4))[0]

    config = {}
    config['dim'] = struct.unpack('i', f.read(4))[0]
    config['hidden_dim'] = struct.unpack('i', f.read(4))[0]
    config['n_layers'] = struct.unpack('i', f.read(4))[0]
    config['n_heads'] = struct.unpack('i', f.read(4))[0]
    config['n_kv_heads'] = struct.unpack('i', f.read(4))[0]
    config['vocab_size'] = struct.unpack('i', f.read(4))[0]
    config['max_seq_len'] = struct.unpack('i', f.read(4))[0]
    config['rope_theta'] = struct.unpack('f', f.read(4))[0]
    config['norm_eps'] = struct.unpack('f', f.read(4))[0]
    config['activation'] = struct.unpack('i', f.read(4))[0]

    weight_dtype = struct.unpack('B', f.read(1))[0]
    # Packed struct: no padding between fields
    vocab_offset = struct.unpack('Q', f.read(8))[0]
    weights_offset = struct.unpack('Q', f.read(8))[0]

    return {
        'version': version,
        'config': config,
        'weight_dtype': weight_dtype,
        'vocab_offset': vocab_offset,
        'weights_offset': weights_offset,
    }


def quantize_q4(data: np.ndarray) -> bytes:
    """Quantize f32 data to Q4_0 blocks. Input length must be multiple of 32."""
    assert len(data) % BLOCK_SIZE == 0
    result = bytearray()

    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i:i + BLOCK_SIZE]
        amax = np.max(np.abs(block))
        d = amax / 7.0 if amax > 0 else 1.0

        # Quantize to [-8, 7] range (stored as unsigned [0, 15])
        quantized = np.clip(np.round(block / d), -8, 7).astype(np.int8)

        # Pack into f16 scale + 16 bytes (2 nibbles per byte)
        d_f16 = np.float16(d)
        result += d_f16.tobytes()

        for j in range(16):
            lo = int(quantized[j * 2] + 8) & 0x0F
            hi = int(quantized[j * 2 + 1] + 8) & 0x0F
            result += struct.pack('B', lo | (hi << 4))

    return bytes(result)


def quantize_q8(data: np.ndarray) -> bytes:
    """Quantize f32 data to Q8_0 blocks. Input length must be multiple of 32."""
    assert len(data) % BLOCK_SIZE == 0
    result = bytearray()

    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i:i + BLOCK_SIZE]
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 1.0

        quantized = np.clip(np.round(block / d), -128, 127).astype(np.int8)

        d_f16 = np.float16(d)
        result += d_f16.tobytes()
        result += quantized.tobytes()

    return bytes(result)


def quantize_model(input_path: str, output_path: str, dtype: str):
    """Quantize an F32 model file to Q4_0 or Q8_0."""
    quantize_fn = quantize_q4 if dtype == 'q4' else quantize_q8
    dtype_byte = DTYPE_Q4_0 if dtype == 'q4' else DTYPE_Q8_0

    with open(input_path, 'rb') as fin:
        # Read full header (256 bytes)
        header_bytes = fin.read(256)
        header_data = fin.seek(0)
        header = read_header(fin)

        if header['weight_dtype'] != DTYPE_F32:
            print(f"Error: Model is already quantized (dtype={header['weight_dtype']})")
            sys.exit(1)

        config = header['config']
        dim = config['dim']
        kv_dim = (dim // config['n_heads']) * config['n_kv_heads']
        hidden = config['hidden_dim']
        vocab = config['vocab_size']
        n_layers = config['n_layers']

        # Read vocabulary section (copy as-is), if present
        vocab_data = b''
        if header['vocab_offset'] > 0:
            fin.seek(header['vocab_offset'])
            vocab_end = header['weights_offset']
            vocab_data = fin.read(vocab_end - header['vocab_offset'])

        # Read weights
        fin.seek(header['weights_offset'])

        # Embeddings stay F32
        embed_size = vocab * dim
        token_embed = np.frombuffer(fin.read(embed_size * 4), dtype=np.float32)
        output_weight = np.frombuffer(fin.read(embed_size * 4), dtype=np.float32)
        final_norm = np.frombuffer(fin.read(dim * 4), dtype=np.float32)

        # Layer weights to quantize
        layer_weights = []
        weight_shapes = [
            ('wq', dim * dim),
            ('wk', kv_dim * dim),
            ('wv', kv_dim * dim),
            ('wo', dim * dim),
            ('w_gate', hidden * dim),
            ('w_up', hidden * dim),
            ('w_down', dim * hidden),
        ]

        for l in range(n_layers):
            layer = {}
            for name, size in weight_shapes:
                layer[name] = np.frombuffer(fin.read(size * 4), dtype=np.float32).copy()
            # Norms stay F32
            layer['attn_norm'] = np.frombuffer(fin.read(dim * 4), dtype=np.float32)
            layer['ffn_norm'] = np.frombuffer(fin.read(dim * 4), dtype=np.float32)
            layer_weights.append(layer)

    # Write quantized model
    with open(output_path, 'wb') as fout:
        # Write header with updated dtype
        fout.write(b'LAi1')
        fout.write(struct.pack('I', 1))  # version

        fout.write(struct.pack('i', config['dim']))
        fout.write(struct.pack('i', config['hidden_dim']))
        fout.write(struct.pack('i', config['n_layers']))
        fout.write(struct.pack('i', config['n_heads']))
        fout.write(struct.pack('i', config['n_kv_heads']))
        fout.write(struct.pack('i', config['vocab_size']))
        fout.write(struct.pack('i', config['max_seq_len']))
        fout.write(struct.pack('f', config['rope_theta']))
        fout.write(struct.pack('f', config['norm_eps']))
        fout.write(struct.pack('i', config['activation']))

        fout.write(struct.pack('B', dtype_byte))

        # Offset placeholders (packed, no padding)
        vocab_offset_pos = fout.tell()
        fout.write(struct.pack('Q', 0))  # vocab_offset
        weights_offset_pos = fout.tell()
        fout.write(struct.pack('Q', 0))  # weights_offset

        # Pad header to 256 bytes
        header_size = fout.tell()
        if header_size < 256:
            fout.write(b'\x00' * (256 - header_size))

        # Write vocabulary (if present in source)
        if vocab_data:
            vocab_offset = fout.tell()
            fout.seek(vocab_offset_pos)
            fout.write(struct.pack('Q', vocab_offset))
            fout.seek(vocab_offset)
            fout.write(vocab_data)
        else:
            # No vocab: leave offset as 0
            pass

        # Write weights
        weights_offset = fout.tell()
        fout.seek(weights_offset_pos)
        fout.write(struct.pack('Q', weights_offset))
        fout.seek(weights_offset)

        # Embeddings: always F32
        fout.write(token_embed.tobytes())
        fout.write(output_weight.tobytes())
        fout.write(final_norm.tobytes())

        # Layer weights: quantized
        for l, layer in enumerate(layer_weights):
            for name, _ in weight_shapes:
                q_data = quantize_fn(layer[name])
                fout.write(q_data)
            # Norms: always F32
            fout.write(layer['attn_norm'].tobytes())
            fout.write(layer['ffn_norm'].tobytes())

            if (l + 1) % 4 == 0 or l == n_layers - 1:
                print(f"  Quantized layer {l + 1}/{n_layers}")

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    ratio = output_size / input_size

    print(f"\nQuantization complete:")
    print(f"  Input:  {input_path} ({input_size / 1e6:.1f} MB, F32)")
    print(f"  Output: {output_path} ({output_size / 1e6:.1f} MB, {dtype.upper()})")
    print(f"  Compression: {ratio:.2f}x ({(1 - ratio) * 100:.0f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description='Quantize LAi model')
    parser.add_argument('input', help='Input F32 model file')
    parser.add_argument('output', help='Output quantized model file')
    parser.add_argument('--dtype', choices=['q4', 'q8'], default='q4',
                        help='Quantization type (default: q4)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Quantizing {args.input} to {args.dtype.upper()}...")
    quantize_model(args.input, args.output, args.dtype)


if __name__ == '__main__':
    main()

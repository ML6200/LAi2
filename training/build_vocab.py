#!/usr/bin/env python3
"""
Build proper BPE vocabulary for LAi
Uses byte-pair encoding to create 32K subword tokens
"""

import struct
import argparse
import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from pathlib import Path
import re


class BPEVocabBuilder:
    """Build BPE vocabulary from text corpus"""

    def __init__(self, vocab_size: int = 32000, checkpoint_path: str = "data/vocab_checkpoint.pkl"):
        self.vocab_size = vocab_size
        self.vocab = []
        self.scores = []
        self.checkpoint_path = checkpoint_path

    def save_checkpoint(self, words, stats, word_freqs, iteration):
        """Save current training state to a pickle file"""
        state = {
            'vocab': self.vocab,
            'scores': self.scores,
            'words': words,
            'stats': stats,
            'word_freqs': word_freqs,
            'iteration': iteration
        }
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(state, f)
        os.replace(temp_path, self.checkpoint_path)

    def load_checkpoint(self):
        """Load training state from pickle file if it exists"""
        if os.path.exists(self.checkpoint_path):
            print(f"  Found checkpoint at {self.checkpoint_path}. Resuming...")
            with open(self.checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train(self, texts: List[str], min_freq: int = 2):
        """Train BPE on text corpus (Optimized with Checkpoints)"""
        print(f"Training BPE vocabulary (target size: {self.vocab_size})...")

        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.vocab = checkpoint['vocab']
            self.scores = checkpoint['scores']
            words = checkpoint['words']
            stats = checkpoint['stats']
            word_freqs = checkpoint['word_freqs']
            start_iteration = checkpoint['iteration']
            print(f"  Resuming from iteration {start_iteration} (vocab_size={len(self.vocab)})")
        else:
            # Initialize with special tokens
            self.vocab = ['<pad>', '<bos>', '<eos>', '<unk>']
            self.scores = [0.0, 0.0, 0.0, 0.0]

            # Add byte tokens (256 tokens for raw bytes)
            for i in range(256):
                self.vocab.append(chr(i))
                self.scores.append(-100.0)

            # Count frequencies of unique "words"
            word_freqs = Counter(texts)
            stats = {} 
            words = {} 
            
            for word_str, freq in word_freqs.items():
                symbols = []
                i = 0
                while i < len(word_str):
                    char_len = self._utf8_char_len(ord(word_str[i]))
                    symbols.append(word_str[i:i+char_len])
                    i += char_len
                
                words[word_str] = symbols
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i+1])
                    stats[pair] = stats.get(pair, 0) + freq
            start_iteration = 0

        # BPE iterations
        max_token_len = 32
        iteration = start_iteration
        
        while len(self.vocab) < self.vocab_size:
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get)
            best_freq = stats[best_pair]
            
            if best_freq < min_freq:
                break
                
            new_token = best_pair[0] + best_pair[1]
            if len(new_token) > max_token_len:
                del stats[best_pair]
                continue
                
            self.vocab.append(new_token)
            self.scores.append(float(best_freq))
            
            # Update 'words' and 'stats'
            for word_str, symbols in words.items():
                freq = word_freqs[word_str]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
                        if i > 0:
                            stats[(symbols[i-1], symbols[i])] -= freq
                        if i < len(symbols) - 2:
                            if not (symbols[i+1] == best_pair[0] and symbols[i+2] == best_pair[1]):
                                stats[(symbols[i+1], symbols[i+2])] -= freq
                        
                        new_symbols.append(new_token)
                        if i > 0:
                            stats[(symbols[i-1], new_token)] = stats.get((symbols[i-1], new_token), 0) + freq
                        
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                words[word_str] = new_symbols
            
            if iteration % 100 == 0:
                # Cleanup zero stats
                stats = {k: v for k, v in stats.items() if v > 0}
                print(f"  Iteration {iteration}: vocab_size={len(self.vocab)}, "
                      f"best_pair={repr(new_token[:20])}, freq={best_freq}")
                
                # Save checkpoint every 500 iterations
                if iteration % 500 == 0 and iteration > start_iteration:
                    self.save_checkpoint(words, stats, word_freqs, iteration)

            iteration += 1

        # Clean up checkpoint on success
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            
        print(f"  Final vocabulary size: {len(self.vocab)}")



    def save(self, path: str):
        """Save vocabulary to binary file (C++ compatible format)"""
        print(f"Saving vocabulary to {path}...")

        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('I', 0x4C414956))  # Magic: "LAIV"
            f.write(struct.pack('I', 1))            # Version
            f.write(struct.pack('I', len(self.vocab)))  # Vocab size

            # Tokens
            for token, score in zip(self.vocab, self.scores):
                token_bytes = token.encode('utf-8')
                f.write(struct.pack('I', len(token_bytes)))
                f.write(token_bytes)
                f.write(struct.pack('f', score))

        print(f"  Saved {len(self.vocab)} tokens")

    @staticmethod
    def _utf8_char_len(first_byte: int) -> int:
        """Get UTF-8 character length from first byte"""
        if (first_byte & 0x80) == 0:
            return 1
        elif (first_byte & 0xE0) == 0xC0:
            return 2
        elif (first_byte & 0xF0) == 0xE0:
            return 3
        elif (first_byte & 0xF8) == 0xF0:
            return 4
        return 1


def main():
    parser = argparse.ArgumentParser(description="Build BPE vocabulary")
    parser.add_argument("--data", type=str, default="data/train.txt",
                       help="Path to training data")
    parser.add_argument("--vocab_size", type=int, default=32000,
                       help="Target vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2,
                       help="Minimum frequency for tokens")
    parser.add_argument("--output", type=str, default="data/vocab.bin",
                       help="Output vocabulary file")
    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(texts)} lines")

    # Build vocabulary
    builder = BPEVocabBuilder(vocab_size=args.vocab_size)
    builder.train(texts, min_freq=args.min_freq)

    # Save
    builder.save(args.output)
    print("Done!")


if __name__ == "__main__":
    main()

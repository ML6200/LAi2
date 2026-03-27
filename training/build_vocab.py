#!/usr/bin/env python3
"""
Build proper BPE vocabulary for LAi
Uses byte-pair encoding to create 32K subword tokens
"""

import struct
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import re


class BPEVocabBuilder:
    """Build BPE vocabulary from text corpus"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.scores = []

    def train(self, texts: List[str], min_freq: int = 2):
        """Train BPE on text corpus (Optimized)"""
        print(f"Training BPE vocabulary (target size: {self.vocab_size})...")

        # Initialize with special tokens
        self.vocab = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.scores = [0.0, 0.0, 0.0, 0.0]

        # Add byte tokens (256 tokens for raw bytes)
        for i in range(256):
            self.vocab.append(chr(i))
            self.scores.append(-100.0)

        # Count frequencies of unique "words" (lines/sequences) to avoid redundant processing
        word_freqs = Counter(texts)
        
        # Initial tokenization into characters
        # We store each unique word as a list of symbols
        stats = {} # (symbol1, symbol2) -> frequency
        words = {} # unique_word_string -> [list, of, symbols]
        
        for word_str, freq in word_freqs.items():
            # Initial split into characters (UTF-8 aware)
            symbols = []
            i = 0
            while i < len(word_str):
                char_len = self._utf8_char_len(ord(word_str[i]))
                symbols.append(word_str[i:i+char_len])
                i += char_len
            
            words[word_str] = symbols
            # Initial pair counts
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                stats[pair] = stats.get(pair, 0) + freq

        print(f"  Processed {len(word_freqs)} unique sequences.")
        print(f"  Initial unique symbols: {len(set(s for syms in words.values() for s in syms))}")

        # BPE iterations
        max_token_len = 32
        iteration = 0
        target_merges = self.vocab_size - len(self.vocab)
        
        while len(self.vocab) < self.vocab_size:
            if not stats:
                break
                
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            best_freq = stats[best_pair]
            
            if best_freq < min_freq:
                break
                
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            if len(new_token) > max_token_len:
                # Skip this pair but remove it from stats so we don't pick it again
                del stats[best_pair]
                continue
                
            self.vocab.append(new_token)
            self.scores.append(float(best_freq))
            
            # Update 'words' and 'stats'
            new_words = {}
            for word_str, symbols in words.items():
                freq = word_freqs[word_str]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
                        # Perform merge
                        # Update stats for pairs that will be broken
                        if i > 0:
                            stats[(symbols[i-1], symbols[i])] -= freq
                        if i < len(symbols) - 2:
                            if not (symbols[i+1] == best_pair[0] and symbols[i+2] == best_pair[1]):
                                stats[(symbols[i+1], symbols[i+2])] -= freq
                        
                        new_symbols.append(new_token)
                        # Update stats for the new pair created
                        if i > 0:
                            stats[(symbols[i-1], new_token)] = stats.get((symbols[i-1], new_token), 0) + freq
                        
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                
                # Update stats for potential new pairs at merge points
                # (The logic above handles most, but let's re-verify pairs in the new sequence)
                words[word_str] = new_symbols
            
            # Re-calculating stats periodically to fix drift or complex merge overlaps
            if iteration % 100 == 0:
                stats = {}
                for word_str, symbols in words.items():
                    freq = word_freqs[word_str]
                    for i in range(len(symbols) - 1):
                        pair = (symbols[i], symbols[i+1])
                        stats[pair] = stats.get(pair, 0) + freq
                
                print(f"  Iteration {iteration}: vocab_size={len(self.vocab)}, "
                      f"best_pair={repr(new_token[:20])}, freq={best_freq}")
            
            iteration += 1

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

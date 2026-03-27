#!/usr/bin/env python3
"""
Data preparation for LAi training

Downloads and prepares Hungarian + English training data:
- Hungarian Wikipedia
- OSCAR Hungarian corpus
- English TinyStories
- Translation pairs

Usage:
    python data.py --output data/train.txt --size small
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import urllib.request
import gzip
import random


def download_file(url: str, path: str) -> bool:
    """Download file with progress"""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def load_hungarian_wiki(max_articles: int = 10000) -> List[str]:
    """Load Hungarian Wikipedia articles"""
    texts = []

    # Try to load from HuggingFace datasets if available
    try:
        from datasets import load_dataset
        print("Loading Hungarian Wikipedia from HuggingFace...")
        # Use the new wikimedia/wikipedia dataset format
        try:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.hu", split="train", streaming=True)
        except Exception:
            # Fallback to older format or alternative
            try:
                dataset = load_dataset("graelo/wikipedia", "hu", split="train", streaming=True)
            except Exception:
                print("Wikipedia dataset not available, using sample data.")
                return get_sample_hungarian_texts()

        for i, example in enumerate(dataset):
            if i >= max_articles:
                break
            text = example.get("text", "")
            if len(text) > 100:
                texts.append(text)

            if i % 1000 == 0:
                print(f"  Loaded {i} articles...")

    except ImportError:
        print("HuggingFace datasets not available. Using sample data.")
        texts = get_sample_hungarian_texts()
    except Exception as e:
        print(f"Error loading Wikipedia: {e}. Using sample data.")
        texts = get_sample_hungarian_texts()

    if not texts:
        texts = get_sample_hungarian_texts()

    print(f"Loaded {len(texts)} Hungarian Wikipedia articles")
    return texts


def load_hungarian_oscar(max_docs: int = 10000) -> List[str]:
    """Load Hungarian OSCAR corpus"""
    texts = []

    try:
        from datasets import load_dataset
        print("Loading Hungarian OSCAR corpus...")
        # Use the new OSCAR corpus format
        try:
            dataset = load_dataset("oscar-corpus/OSCAR-2301", "hu", split="train", streaming=True)
        except Exception:
            try:
                # Alternative: CulturaX which includes Hungarian
                dataset = load_dataset("uonlp/CulturaX", "hu", split="train", streaming=True)
            except Exception:
                print("OSCAR dataset not available, skipping.")
                return []

        for i, example in enumerate(dataset):
            if i >= max_docs:
                break
            text = example.get("text", "")
            if len(text) > 50:
                texts.append(text)

            if i % 1000 == 0:
                print(f"  Loaded {i} documents...")

    except ImportError:
        print("HuggingFace datasets not available.")
        texts = []
    except Exception as e:
        print(f"Error loading OSCAR: {e}")
        texts = []

    print(f"Loaded {len(texts)} Hungarian OSCAR documents")
    return texts


def load_english_tinystories(max_stories: int = 50000) -> List[str]:
    """Load TinyStories dataset for English"""
    texts = []

    try:
        from datasets import load_dataset
        print("Loading TinyStories...")
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        except Exception:
            print("TinyStories not available, using sample data.")
            return get_sample_english_texts()

        for i, example in enumerate(dataset):
            if i >= max_stories:
                break
            text = example.get("text", "")
            if len(text) > 50:
                texts.append(text)

            if i % 5000 == 0:
                print(f"  Loaded {i} stories...")

    except ImportError:
        print("HuggingFace datasets not available. Using sample data.")
        texts = get_sample_english_texts()
    except Exception as e:
        print(f"Error loading TinyStories: {e}. Using sample data.")
        texts = get_sample_english_texts()

    if not texts:
        texts = get_sample_english_texts()

    print(f"Loaded {len(texts)} English stories")
    return texts


def load_translation_pairs(max_pairs: int = 10000) -> List[str]:
    """Load Hungarian-English translation pairs"""
    pairs = []

    try:
        from datasets import load_dataset
        print("Loading translation pairs...")
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", "en-hu", split="train", streaming=True)
        except Exception:
            try:
                dataset = load_dataset("opus100", "en-hu", split="train", streaming=True)
            except Exception:
                print("OPUS-100 not available, using sample translation pairs.")
                return get_sample_translation_pairs()

        for i, example in enumerate(dataset):
            if i >= max_pairs:
                break
            translation = example.get("translation", {})
            en = translation.get("en", "")
            hu = translation.get("hu", "")

            if en and hu:
                # Format as translation examples
                pairs.append(f"<system>Translate English to Hungarian.</system><user>{en}</user><assistant>{hu}</assistant>")
                pairs.append(f"<system>Fordítsd angolról magyarra.</system><user>{en}</user><assistant>{hu}</assistant>")
                pairs.append(f"<system>Translate Hungarian to English.</system><user>{hu}</user><assistant>{en}</assistant>")

            if i % 2000 == 0:
                print(f"  Loaded {i} pairs...")

    except ImportError:
        print("HuggingFace datasets not available.")
        pairs = get_sample_translation_pairs()
    except Exception as e:
        print(f"Error loading translation pairs: {e}")
        pairs = get_sample_translation_pairs()

    if not pairs:
        pairs = get_sample_translation_pairs()

    print(f"Created {len(pairs)} translation examples")
    return pairs


def get_sample_hungarian_texts() -> List[str]:
    """Sample Hungarian texts for testing"""
    return [
        "Budapest Magyarország fővárosa és legnagyobb városa. A Duna két partján fekszik, amely ketté osztja a várost Budára és Pestre.",
        "A magyar nyelv a finnugor nyelvcsaládhoz tartozik. Egyedülálló nyelv Európában, amely nem rokon a szomszédos országok nyelveivel.",
        "A Balaton Közép-Európa legnagyobb tava. A magyar tengernek is nevezik, és népszerű nyaralóhely mind a magyarok, mind a külföldiek körében.",
        "Magyarország híres a termálfürdőiről. Budapest több mint 120 természetes termálforrással rendelkezik.",
        "A gulyás a magyar konyha egyik leghíresebb étele. Eredetileg a pásztorok készítették a pusztán.",
        "A Rubik-kockát Rubik Ernő magyar feltaláló alkotta meg 1974-ben. A világ egyik legnépszerűbb játéka lett.",
        "A tokaji bor világhírű magyar termék. A Tokaji borvidék az UNESCO világörökség része.",
        "Liszt Ferenc a 19. század egyik legnagyobb zongoravirtuóza és zeneszerzője volt. Magyarországon született.",
    ] * 100


def get_sample_english_texts() -> List[str]:
    """Sample English texts for testing"""
    return [
        "Once upon a time, there was a little girl who loved to read books. She would spend hours in the library.",
        "The sun was shining brightly on the meadow. Birds were singing in the trees, and flowers bloomed everywhere.",
        "Tom had a small dog named Max. They played together every day in the park near their house.",
        "Learning to code is like learning a new language. It takes practice, but anyone can do it with dedication.",
        "The ocean is vast and mysterious. Scientists are still discovering new species in its depths.",
        "Friendship is one of life's greatest treasures. Good friends support each other through thick and thin.",
        "Technology has changed the way we live. We can now connect with people all around the world instantly.",
        "Exercise is important for staying healthy. Even a short walk each day can make a big difference.",
    ] * 100


def get_sample_translation_pairs() -> List[str]:
    """Sample translation pairs for testing"""
    pairs = [
        ("Hello, how are you?", "Szia, hogy vagy?"),
        ("Good morning!", "Jó reggelt!"),
        ("Thank you very much.", "Köszönöm szépen."),
        ("What is your name?", "Mi a neved?"),
        ("I love learning languages.", "Szeretem a nyelvtanulást."),
        ("The weather is nice today.", "Ma szép az idő."),
        ("Where is the train station?", "Hol van a vasútállomás?"),
        ("I would like a coffee, please.", "Kérek egy kávét."),
    ]

    formatted = []
    for en, hu in pairs * 50:
        formatted.append(f"<system>Translate English to Hungarian.</system><user>{en}</user><assistant>{hu}</assistant>")
        formatted.append(f"<system>Translate Hungarian to English.</system><user>{hu}</user><assistant>{en}</assistant>")

    return formatted


def create_instruction_data(texts: List[str]) -> List[str]:
    """Create instruction-following examples from texts"""
    instructions = []

    templates = [
        ("Summarize the following text:", "Foglald össze a következő szöveget:"),
        ("What is the main idea of this text?", "Mi a fő gondolata ennek a szövegnek?"),
        ("Explain this in simple terms:", "Magyarázd el egyszerűen:"),
        ("Continue this text:", "Folytasd ezt a szöveget:"),
        ("Translate this to Hungarian:", "Fordítsd le magyarra:"),
        ("Translate this to English:", "Fordítsd le angolra:"),
        ("Rewrite this text in a formal tone:", "Írd át ezt a szöveget hivatalos stílusban:"),
        ("Extract the key entities from this text:", "Gyűjtsd ki a legfontosabb entitásokat ebből a szövegből:"),
    ]

    for text in texts[:len(texts)//2]:
        if 100 < len(text) < 2000:
            template_en, template_hu = random.choice(templates)
            
            is_hu = any(c in text for c in 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ')

            if is_hu:
                # Hungarian instruction for Hungarian text
                instructions.append(f"<user>{template_hu}\n\n{text}</user><assistant>")
            else:
                # English instruction for English text
                instructions.append(f"<user>{template_en}\n\n{text}</user><assistant>")

    return instructions


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LAi")
    parser.add_argument("--output", type=str, default="data/train.txt", help="Output file path")
    parser.add_argument("--size", choices=["micro", "tiny", "small", "medium", "large"], default="small")
    parser.add_argument("--hu_wiki", type=int, default=None, help="Number of Hungarian Wikipedia articles")
    parser.add_argument("--hu_oscar", type=int, default=None, help="Number of Hungarian OSCAR documents")
    parser.add_argument("--en_stories", type=int, default=None, help="Number of English stories")
    parser.add_argument("--translations", type=int, default=None, help="Number of translation pairs")
    args = parser.parse_args()

    # Set sizes based on preset
    size_presets = {
        "micro": {"hu_wiki": 500, "hu_oscar": 500, "en_stories": 2000, "translations": 500},
        "tiny": {"hu_wiki": 2000, "hu_oscar": 2000, "en_stories": 10000, "translations": 2000},
        "small": {"hu_wiki": 10000, "hu_oscar": 10000, "en_stories": 40000, "translations": 10000},
        "medium": {"hu_wiki": 40000, "hu_oscar": 40000, "en_stories": 100000, "translations": 25000},
        "large": {"hu_wiki": 100000, "hu_oscar": 100000, "en_stories": 250000, "translations": 50000},
    }

    preset = size_presets[args.size]

    hu_wiki_count = args.hu_wiki or preset["hu_wiki"]
    hu_oscar_count = args.hu_oscar or preset["hu_oscar"]
    en_stories_count = args.en_stories or preset["en_stories"]
    translations_count = args.translations or preset["translations"]

    print(f"Preparing {args.size} dataset:")
    print(f"  Hungarian Wikipedia: {hu_wiki_count}")
    print(f"  Hungarian OSCAR: {hu_oscar_count}")
    print(f"  English stories: {en_stories_count}")
    print(f"  Translation pairs: {translations_count}")
    print()

    # Collect data
    all_texts = []

    # Hungarian data
    hu_wiki = load_hungarian_wiki(hu_wiki_count)
    all_texts.extend(hu_wiki)

    hu_oscar = load_hungarian_oscar(hu_oscar_count)
    all_texts.extend(hu_oscar)

    # English data
    en_stories = load_english_tinystories(en_stories_count)
    all_texts.extend(en_stories)

    # Translation pairs
    translations = load_translation_pairs(translations_count)
    all_texts.extend(translations)

    # Create instruction data
    instructions = create_instruction_data(all_texts[:10000])
    all_texts.extend(instructions)

    # Shuffle
    random.shuffle(all_texts)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in all_texts:
            # Clean and write
            text = text.strip().replace('\n\n\n', '\n\n')
            if text:
                f.write(text + '\n')

    total_chars = sum(len(t) for t in all_texts)
    print(f"\nDataset created:")
    print(f"  Total examples: {len(all_texts)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

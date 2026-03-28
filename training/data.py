#!/usr/bin/env python3
"""
Data preparation for LAi training

Downloads and prepares Hungarian + English training data in chat format.
All examples use <system>/<user>/<assistant> tags matching inference format.

Usage:
    python data.py --output data/train.txt --size small
"""

import os
import json
import argparse
import random
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================================
# Chat formatting helpers
# ============================================================================

def chat(user: str, assistant: str, system: str = "") -> str:
    """Format a single chat example with proper tags"""
    parts = []
    if system:
        parts.append(f"<system>{system}</system>")
    parts.append(f"<user>{user}</user>")
    parts.append(f"<assistant>{assistant}</assistant>")
    return "".join(parts)


def translation_example(source: str, target: str, src_lang: str, tgt_lang: str) -> str:
    """Format a translation pair as chat"""
    lang_names = {"en": "English", "hu": "Hungarian"}
    system = f"Translate {lang_names[src_lang]} to {lang_names[tgt_lang]}."
    return chat(source, target, system)


# ============================================================================
# HuggingFace dataset loaders
# ============================================================================

def try_load_hf_dataset(name: str, config: str = None, split: str = "train",
                        max_items: int = 1000) -> list:
    """Try loading a HuggingFace dataset, return empty list on failure"""
    try:
        from datasets import load_dataset
        args = [name]
        if config:
            args.append(config)
        ds = load_dataset(*args, split=split, streaming=True)
        items = []
        for i, item in enumerate(ds):
            if i >= max_items:
                break
            items.append(item)
            if i > 0 and i % 2000 == 0:
                print(f"    {i} items loaded...")
        return items
    except Exception as e:
        print(f"  Could not load {name}: {e}")
        return []


# ============================================================================
# Translation data (highest value for bilingual model)
# ============================================================================

def load_translation_pairs(max_pairs: int = 5000) -> List[str]:
    """Load HU-EN translation pairs from OPUS-100"""
    print("Loading translation pairs...")
    examples = []

    items = try_load_hf_dataset("Helsinki-NLP/opus-100", "en-hu", max_items=max_pairs)

    for item in items:
        tr = item.get("translation", {})
        en = tr.get("en", "").strip()
        hu = tr.get("hu", "").strip()
        if not en or not hu:
            continue
        # Skip very long pairs (won't fit in context)
        if len(en) > 300 or len(hu) > 300:
            continue

        # Both directions
        examples.append(translation_example(en, hu, "en", "hu"))
        examples.append(translation_example(hu, en, "hu", "en"))

    if not examples:
        examples = get_builtin_translation_pairs()

    print(f"  {len(examples)} translation examples")
    return examples


def get_builtin_translation_pairs() -> List[str]:
    """Fallback translation pairs covering common patterns"""
    pairs = [
        # Greetings & basics
        ("Hello!", "Szia!"),
        ("Good morning!", "Jó reggelt!"),
        ("Good evening!", "Jó estét!"),
        ("Good night!", "Jó éjszakát!"),
        ("Goodbye!", "Viszlát!"),
        ("See you later!", "Később találkozunk!"),
        ("Thank you very much.", "Köszönöm szépen."),
        ("You're welcome.", "Szívesen."),
        ("Please help me.", "Kérem, segítsen."),
        ("Excuse me.", "Elnézést."),
        ("I'm sorry.", "Sajnálom."),
        ("Yes, of course.", "Igen, természetesen."),
        ("No, thank you.", "Nem, köszönöm."),

        # Questions
        ("How are you?", "Hogy vagy?"),
        ("What is your name?", "Mi a neved?"),
        ("Where is the train station?", "Hol van a vasútállomás?"),
        ("How much does this cost?", "Mennyibe kerül ez?"),
        ("Do you speak English?", "Beszélsz angolul?"),
        ("What time is it?", "Hány óra van?"),
        ("Where are you from?", "Honnan származol?"),
        ("Can you help me?", "Tudsz segíteni?"),
        ("What do you recommend?", "Mit ajánlasz?"),
        ("Where is the bathroom?", "Hol van a mosdó?"),

        # Statements
        ("I am learning Hungarian.", "Magyarul tanulok."),
        ("I love this city.", "Imádom ezt a várost."),
        ("The weather is beautiful today.", "Ma gyönyörű az idő."),
        ("I would like a coffee.", "Kérnék egy kávét."),
        ("I don't understand.", "Nem értem."),
        ("I speak a little Hungarian.", "Beszélek egy kicsit magyarul."),
        ("This food is delicious.", "Ez az étel nagyon finom."),
        ("I live in Budapest.", "Budapesten élek."),
        ("She is a doctor.", "Ő orvos."),
        ("The children are playing in the park.", "A gyerekek a parkban játszanak."),
        ("I have two brothers and one sister.", "Két bátyám és egy húgom van."),
        ("We arrived at the hotel.", "Megérkeztünk a szállodába."),
        ("The book is on the table.", "A könyv az asztalon van."),
        ("I need to buy some groceries.", "Vásárolnom kell néhány élelmiszert."),
        ("The train leaves at five o'clock.", "A vonat öt órakor indul."),
        ("It is raining outside.", "Kint esik az eső."),
        ("My favorite color is blue.", "A kedvenc színem a kék."),
        ("We went to the museum yesterday.", "Tegnap múzeumba mentünk."),
        ("He works at the university.", "Az egyetemen dolgozik."),
        ("The river flows through the city.", "A folyó a városon keresztül folyik."),

        # Longer sentences
        ("Budapest is the capital of Hungary and one of the most beautiful cities in Europe.",
         "Budapest Magyarország fővárosa és Európa egyik legszebb városa."),
        ("The Hungarian language is unique because it is not related to most other European languages.",
         "A magyar nyelv egyedülálló, mert nem rokon a legtöbb más európai nyelvvel."),
        ("Lake Balaton is the largest lake in Central Europe and a popular vacation spot.",
         "A Balaton Közép-Európa legnagyobb tava és népszerű nyaralóhely."),
        ("I would like to visit the thermal baths in Budapest.",
         "Szeretném meglátogatni a budapesti termálfürdőket."),
        ("Could you please give me directions to the nearest pharmacy?",
         "Meg tudná mondani, merre van a legközelebbi gyógyszertár?"),
    ]

    examples = []
    for en, hu in pairs:
        examples.append(translation_example(en, hu, "en", "hu"))
        examples.append(translation_example(hu, en, "hu", "en"))
    return examples


# ============================================================================
# Conversational / Q&A data
# ============================================================================

def load_english_conversations(max_items: int = 5000) -> List[str]:
    """Load English instruction/conversation data"""
    print("Loading English conversation data...")
    examples = []

    # Try Alpaca (instruction following)
    items = try_load_hf_dataset("tatsu-lab/alpaca", max_items=max_items)

    for item in items:
        instruction = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue
        # Skip long examples that won't fit in 256-token context
        if len(instruction) + len(inp) + len(output) > 600:
            continue

        user_msg = instruction
        if inp:
            user_msg += f"\n\n{inp}"

        examples.append(chat(user_msg, output))

    if not examples:
        examples = get_builtin_conversations()

    print(f"  {len(examples)} English conversation examples")
    return examples


def get_builtin_conversations() -> List[str]:
    """Fallback conversational examples"""
    convos = [
        # General knowledge
        ("What is the capital of Hungary?",
         "The capital of Hungary is Budapest. It is situated on both banks of the Danube river."),
        ("Mi Magyarország fővárosa?",
         "Magyarország fővárosa Budapest. A Duna két partján helyezkedik el."),
        ("Who invented the Rubik's Cube?",
         "The Rubik's Cube was invented by Ernő Rubik, a Hungarian architect, in 1974."),
        ("Ki találta fel a Rubik-kockát?",
         "A Rubik-kockát Rubik Ernő magyar építész találta fel 1974-ben."),

        # Helpful assistant
        ("Tell me a fun fact.",
         "Hungary has over 1,000 natural hot springs, making it one of the richest countries in thermal water in the world."),
        ("Mondj egy érdekességet.",
         "Magyarországon több mint 1000 természetes hévízforrás található, ami a világon az egyik leggazdagabb termálvízkészletet jelenti."),
        ("What should I see in Budapest?",
         "You should visit the Parliament building, Buda Castle, the thermal baths, and take a walk along the Danube. The ruin bars in the Jewish Quarter are also very popular."),
        ("Mit érdemes megnézni Budapesten?",
         "Érdemes meglátogatni az Országházat, a Budai Várat, a termálfürdőket, és sétálni a Duna-parton. A zsidó negyed romkocsmái is nagyon népszerűek."),

        # Simple tasks
        ("Count from one to five in Hungarian.",
         "Egy, kettő, három, négy, öt."),
        ("List the days of the week.",
         "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday."),
        ("Sorold fel a hét napjait.",
         "Hétfő, kedd, szerda, csütörtök, péntek, szombat, vasárnap."),
        ("What are the seasons?",
         "The four seasons are spring, summer, autumn, and winter."),
        ("Mik az évszakok?",
         "A négy évszak a tavasz, a nyár, az ősz és a tél."),

        # Conversational
        ("Hi! How are you?",
         "Hello! I'm doing well, thank you for asking. How can I help you today?"),
        ("Szia! Hogy vagy?",
         "Szia! Jól vagyok, köszönöm hogy kérdezed. Miben segíthetek ma?"),
        ("What can you do?",
         "I can help you with conversations in English and Hungarian, translate between the two languages, and answer questions."),
        ("Miben tudsz segíteni?",
         "Tudok segíteni angol és magyar nyelvű beszélgetésekben, fordítani a két nyelv között, és kérdésekre válaszolni."),
        ("Tell me about yourself.",
         "I am LAi, a lightweight AI assistant that supports both Hungarian and English. I was designed to run efficiently on low-end hardware."),

        # Math & logic
        ("What is 15 + 27?",
         "15 + 27 = 42."),
        ("What is the square root of 144?",
         "The square root of 144 is 12."),

        # Explanations
        ("Explain what a computer is in simple terms.",
         "A computer is an electronic device that processes information. It takes input, follows instructions called programs, and produces output. Think of it as a very fast calculator that can also handle text, images, and more."),
        ("Mi az a mesterséges intelligencia?",
         "A mesterséges intelligencia olyan számítógépes rendszer, amely emberi gondolkodást utánzó feladatokat végez. Képes tanulni adatokból, felismerni mintákat és döntéseket hozni."),
    ]

    examples = []
    for user, assistant in convos:
        examples.append(chat(user, assistant))
    return examples


# ============================================================================
# Hungarian text data (short, diverse passages)
# ============================================================================

def load_hungarian_texts(max_articles: int = 5000) -> List[str]:
    """Load Hungarian text formatted as reading comprehension / knowledge"""
    print("Loading Hungarian text data...")
    examples = []

    items = try_load_hf_dataset("wikimedia/wikipedia", "20231101.hu", max_items=max_articles)
    if not items:
        items = try_load_hf_dataset("graelo/wikipedia", "hu", max_items=max_articles)

    for item in items:
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        # Extract first paragraph (short, self-contained)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        if not paragraphs:
            continue

        first_para = paragraphs[0]
        # Truncate to reasonable length for context window
        if len(first_para) > 500:
            # Cut at sentence boundary
            sentences = first_para.split(". ")
            first_para = ". ".join(sentences[:3]) + "."

        # Format as Q&A about the topic
        if title:
            examples.append(chat(
                f"Mi az a {title}?" if random.random() < 0.5 else f"Mesélj a következőről: {title}",
                first_para
            ))

            # Also create a "summarize" example from a longer paragraph
            if len(paragraphs) > 1 and len(paragraphs[1]) > 100:
                long_text = paragraphs[1][:400]
                short = paragraphs[0][:200]
                examples.append(chat(
                    f"Foglald össze röviden:\n\n{long_text}",
                    short,
                    "Foglald össze a megadott szöveget egy-két mondatban."
                ))

    if not examples:
        examples = get_builtin_hungarian_qa()

    print(f"  {len(examples)} Hungarian text examples")
    return examples


def get_builtin_hungarian_qa() -> List[str]:
    """Fallback Hungarian Q&A"""
    qa_pairs = [
        ("Mi az a Balaton?",
         "A Balaton Közép-Európa legnagyobb tava, Magyarország nyugati részén található. A magyar tengernek is nevezik, és népszerű nyaralóhely."),
        ("Meséld el röviden Budapest történetét.",
         "Budapest 1873-ban jött létre Buda, Pest és Óbuda egyesítéséből. A város a Duna két partján fekszik, és Magyarország fővárosa."),
        ("Mi a gulyás?",
         "A gulyás a magyar konyha egyik leghíresebb étele. Eredetileg a pásztorok készítették. Marhahúsból, hagymából, paprikából és burgonyából készül."),
        ("Ki volt Liszt Ferenc?",
         "Liszt Ferenc a 19. század egyik legnagyobb zongoravirtuóza és zeneszerzője volt. Magyarországon született és világhírű lett."),
        ("Mi a tokaji bor?",
         "A tokaji bor világhírű magyar borkülönlegesség. A Tokaji borvidéken készül, amely az UNESCO világörökség része."),
        ("Milyen nyelv a magyar?",
         "A magyar nyelv a finnugor nyelvcsaládhoz tartozik. Egyedülálló Európában, mert nem rokon a legtöbb szomszédos ország nyelvével."),
        ("Hány lakosa van Budapestnek?",
         "Budapestnek körülbelül 1,7 millió lakosa van, ami az ország lakosságának mintegy hatoda."),
        ("Mi az Országház?",
         "Az Országház Budapest egyik leghíresebb épülete, a Duna partján áll. Neogótikus stílusban épült, és itt ülésezik a magyar parlament."),
    ]
    return [chat(q, a) for q, a in qa_pairs]


# ============================================================================
# English story data (short, coherent)
# ============================================================================

def load_english_stories(max_stories: int = 5000) -> List[str]:
    """Load TinyStories formatted as chat examples"""
    print("Loading English story data...")
    examples = []

    items = try_load_hf_dataset("roneneldan/TinyStories", max_items=max_stories)

    for item in items:
        text = item.get("text", "").strip()
        if not text or len(text) < 50:
            continue
        # Skip very long stories
        if len(text) > 500:
            # Truncate at sentence boundary
            sentences = text.split(". ")
            text = ". ".join(sentences[:4]) + "."

        # Format as story generation
        prompts = [
            "Tell me a short story.",
            "Write a children's story.",
            "Tell me a bedtime story.",
            "Write a short tale.",
        ]
        examples.append(chat(random.choice(prompts), text,
                            "You are a storyteller. Write short, engaging stories."))

    if not examples:
        examples = get_builtin_stories()

    print(f"  {len(examples)} English story examples")
    return examples


def get_builtin_stories() -> List[str]:
    """Fallback English stories"""
    stories = [
        "Once upon a time, there was a little girl who loved to read books. She would spend hours in the library, discovering new worlds through stories.",
        "Tom had a small dog named Max. They played together every day in the park. One day, Max found a shiny coin buried in the sand.",
        "The sun was shining brightly on the meadow. Birds were singing in the trees. A little rabbit hopped through the flowers, looking for breakfast.",
        "Sarah wanted to learn to paint. She practiced every day after school. Soon, her paintings filled the walls of her room with color.",
        "A wise old owl lived in the tallest tree in the forest. Animals would come to ask for advice. The owl always listened carefully before answering.",
        "Ben built a small boat out of wood. He sailed it on the pond behind his house. The boat floated perfectly, and Ben smiled with pride.",
        "On a cold winter night, the stars sparkled like diamonds. A little boy looked up at the sky and made a wish. He wished for snow.",
        "The baker woke up early every morning to make fresh bread. The whole neighborhood could smell the delicious aroma from his shop.",
    ]
    sys = "You are a storyteller. Write short, engaging stories."
    return [chat("Tell me a short story.", s, sys) for s in stories]


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LAi")
    parser.add_argument("--output", type=str, default="data/train.txt", help="Output file path")
    parser.add_argument("--size", choices=["micro", "tiny", "small", "medium", "large"], default="small")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace downloads, use builtins only")
    args = parser.parse_args()

    # Dataset sizes tuned per model config
    # Micro: 256 context, 10M params — needs short, focused examples
    # Tiny:  512 context, ~50M params
    # Small: 1024 context, ~150M params
    size_presets = {
        "micro":  {"translations": 2000,  "conversations": 2000,  "hu_texts": 1000,  "stories": 1000},
        "tiny":   {"translations": 5000,  "conversations": 5000,  "hu_texts": 3000,  "stories": 3000},
        "small":  {"translations": 10000, "conversations": 10000, "hu_texts": 8000,  "stories": 8000},
        "medium": {"translations": 25000, "conversations": 25000, "hu_texts": 20000, "stories": 20000},
        "large":  {"translations": 50000, "conversations": 50000, "hu_texts": 40000, "stories": 40000},
    }

    preset = size_presets[args.size]
    print(f"Preparing {args.size} dataset:")
    for k, v in preset.items():
        print(f"  {k}: {v}")
    print()

    all_examples = []

    if args.no_hf:
        # Builtin-only mode (no internet required)
        all_examples.extend(get_builtin_translation_pairs())
        all_examples.extend(get_builtin_conversations())
        all_examples.extend(get_builtin_hungarian_qa())
        all_examples.extend(get_builtin_stories())
        # Repeat to get more variety via shuffling
        all_examples = all_examples * max(1, preset["translations"] // len(all_examples))
    else:
        # HuggingFace datasets (higher quality, needs internet)
        all_examples.extend(load_translation_pairs(preset["translations"]))
        all_examples.extend(load_english_conversations(preset["conversations"]))
        all_examples.extend(load_hungarian_texts(preset["hu_texts"]))
        all_examples.extend(load_english_stories(preset["stories"]))

    # Deduplicate
    unique_examples = list(set(all_examples))
    print(f"\nDeduplication: {len(all_examples)} -> {len(unique_examples)} unique examples")
    all_examples = unique_examples

    # Shuffle
    random.shuffle(all_examples)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in all_examples:
            f.write(text.strip() + '\n')

    total_chars = sum(len(t) for t in all_examples)
    print(f"\nDataset created: {output_path}")
    print(f"  Total examples: {len(all_examples):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Sanity check: show a few examples
    print(f"\nSample examples:")
    for ex in random.sample(all_examples, min(3, len(all_examples))):
        print(f"  {ex[:120]}...")


if __name__ == "__main__":
    main()

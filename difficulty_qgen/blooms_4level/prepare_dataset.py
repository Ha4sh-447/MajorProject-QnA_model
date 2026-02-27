#!/usr/bin/env python3
"""
Option B: 4-Level Bloom's Taxonomy — Consolidated & Balanced Dataset

AUGMENTATION STRATEGY:
Instead of template-wrapping existing questions (which produces gibberish),
we extract KEY SENTENCES from the chapter text and build clean questions
around them using Bloom's verb starters. This produces coherent, educational
questions that a student would actually encounter.

Levels:
  L1: Remember & Understand
  L2: Apply
  L3: Analyze
  L4: Evaluate & Create
"""

import json
import re
import random
import os
from collections import Counter

INPUT_PATH = "../../dataset/qna.json"
OUTPUT_DIR = "../../dataset"
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "blooms_4level_train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "blooms_4level_val.json")

VAL_SPLIT = 0.15
SEED = 42

LEVEL_TAGS = {
    1: "REMEMBER & UNDERSTAND",
    2: "APPLY",
    3: "ANALYZE",
    4: "EVALUATE & CREATE",
}

QUESTION_TYPE_TO_LEVEL = {
    "MCQ": 1,
    "Matching Questions": 1,
    "Very Short Answer Questions": 1,
    "Short Answer Questions": [2, 3],
    "Assertion and Reason Questions": [3, 4],
    "Long Answer Questions": 4,
}

PROMPT_TEMPLATES = {
    1: [
        "generate a remember and understand level question: {text}",
        "create a recall or comprehension question from: {text}",
        "generate a level 1 question testing factual knowledge: {text}",
    ],
    2: [
        "generate an apply level question: {text}",
        "create a question requiring application of concepts from: {text}",
        "generate a level 2 question testing problem solving: {text}",
    ],
    3: [
        "generate an analyze level question: {text}",
        "create a question requiring analysis from: {text}",
        "generate a level 3 question testing analytical thinking: {text}",
    ],
    4: [
        "generate an evaluate and create level question: {text}",
        "create a question requiring evaluation or design from: {text}",
        "generate a level 4 question testing critical and creative thinking: {text}",
    ],
}

# ─── AUGMENTATION: Clean Bloom's question generators ─────────────────────────
# Each function takes a KEY SENTENCE from the chapter and builds a proper question.

def make_analyze_questions(key_sentence):
    """Generate proper Analyze-level questions from a key sentence."""
    # Clean the sentence
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[ANALYZE] What is the relationship between the factors described in: '{sent}'?",
        f"[ANALYZE] Compare and contrast the concepts presented in the following: {sent}.",
        f"[ANALYZE] Break down the process described here and identify the key components: {sent}.",
        f"[ANALYZE] Examine how the concept of '{sent[:60]}' relates to the broader topic discussed.",
        f"[ANALYZE] What patterns can you identify from the fact that {sent.lower()}?",
        f"[ANALYZE] Differentiate between the terms and concepts mentioned in: {sent}.",
    ]
    return random.choice(templates)

def make_evaluate_create_questions(key_sentence):
    """Generate proper Evaluate & Create level questions from a key sentence."""
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[EVALUATE & CREATE] Critically assess the statement: '{sent}'. Do you agree? Justify your answer.",
        f"[EVALUATE & CREATE] Design an experiment that could demonstrate the principle described in: {sent}.",
        f"[EVALUATE & CREATE] Propose an alternative explanation for the phenomenon described: {sent}.",
        f"[EVALUATE & CREATE] Evaluate the significance of the fact that {sent.lower()} in the context of this topic.",
        f"[EVALUATE & CREATE] If '{sent[:60]}' were not true, what consequences would follow? Justify your reasoning.",
        f"[EVALUATE & CREATE] Create a real-world scenario where the concept described in '{sent[:60]}' would be practically useful.",
    ]
    return random.choice(templates)


def extract_key_sentences(chapter_text, min_len=40, max_len=200):
    """Extract informative key sentences from chapter text for augmentation."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chapter_text)
    key_sents = []
    for sent in sentences:
        sent = sent.strip()
        # Filter: must be informative (not headers, not too short, not just numbers)
        if len(sent) < min_len or len(sent) > max_len:
            continue
        if re.match(r'^\d+\.?\s*$', sent):  # Just a number
            continue
        if sent.isupper():  # All caps header
            continue
        if re.match(r'^(Fig|Figure|Table|Note|Hint|Example)\b', sent, re.IGNORECASE):
            continue
        if sent.count(' ') < 5:  # Too few words
            continue
        key_sents.append(sent)
    return key_sents


# ─── Text utilities ──────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def clean_question(text):
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    text = re.sub(r'BIOLOGY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHEMISTRY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PHYSICS,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHAPTER\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'UNIT\s*[IVX]+:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\b\d{1,3}\b)(\s+\1){2,}', '', text)
    text = re.sub(r'\b[A-Z ]{15,}\b', '', text)
    text = re.sub(r'^\d+\.\s*\n?', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()

def chunk_text(text, max_chars=800, overlap_chars=100):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            sent = sent[:max_chars]
        if current_len + len(sent) > max_chars and current:
            chunks.append(' '.join(current))
            overlap = current[-1] if len(current[-1]) <= overlap_chars else ""
            current = [overlap, sent] if overlap else [sent]
            current_len = sum(len(s) for s in current)
        else:
            current.append(sent)
            current_len += len(sent)
    if current:
        chunks.append(' '.join(current))
    return chunks if chunks else [text[:max_chars]]

def classify_level(question_text, levels):
    if not isinstance(levels, list):
        return levels
    q_lower = question_text.lower()
    higher = ['explain', 'describe', 'discuss', 'compare', 'contrast', 'differentiate',
              'analyze', 'evaluate', 'justify', 'design', 'propose', 'how would you']
    lower = ['what is', 'define', 'name', 'list', 'identify', 'state',
             'which of the following', 'select', 'choose', 'match']
    h = sum(1 for kw in higher if kw in q_lower)
    l = sum(1 for kw in lower if kw in q_lower)
    return max(levels) if h > l else min(levels)

def find_best_chunk(question, chunks):
    q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
    best, best_score = chunks[0], 0
    for chunk in chunks:
        score = len(q_words & set(re.findall(r'\b\w{4,}\b', chunk.lower())))
        if score > best_score:
            best_score, best = score, chunk
    return best


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("=" * 60)
    print("OPTION B: 4-LEVEL BLOOM'S — CONSOLIDATED DATASET")
    print("=" * 60 + "\n")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 {len(data)} chapters loaded\n")

    examples_by_level = {i: [] for i in range(1, 5)}
    chapter_info = []  # Store chapter text + chunks for augmentation

    for chapter in data:
        chapter_text = clean_text(chapter["chapter_text"])
        chapter_name = chapter.get("chapter_name", "")
        subject = chapter.get("subject", "")
        chunks = chunk_text(chapter_text)
        key_sents = extract_key_sentences(chapter_text)

        chapter_info.append({
            "chunks": chunks, "key_sents": key_sents,
            "chapter": chapter_name, "subject": subject,
        })

        for qtype, questions in chapter["questions"].items():
            if qtype not in QUESTION_TYPE_TO_LEVEL:
                continue
            level_spec = QUESTION_TYPE_TO_LEVEL[qtype]

            for q_text in questions:
                q_cleaned = clean_question(q_text)
                if not q_cleaned or len(q_cleaned) < 10:
                    continue
                level = classify_level(q_cleaned, level_spec)
                best_chunk = find_best_chunk(q_cleaned, chunks)
                tag = LEVEL_TAGS[level]
                target = f"[{tag}] {q_cleaned}"

                examples_by_level[level].append({
                    "chunk": best_chunk,
                    "target": target,
                    "level": level,
                })

    print("📊 Base distribution:")
    for lv in range(1, 5):
        print(f"   L{lv} ({LEVEL_TAGS[lv]:25s}): {len(examples_by_level[lv])}")

    # Balance: downsample L1, augment L3/L4
    TARGET_PER_LEVEL = 700

    if len(examples_by_level[1]) > TARGET_PER_LEVEL:
        examples_by_level[1] = random.sample(examples_by_level[1], TARGET_PER_LEVEL)
        print(f"\n✂️  L1 downsampled to {TARGET_PER_LEVEL}")

    # Augment L3 and L4 using KEY SENTENCES from chapters (not template wrapping!)
    print("\n🔄 Augmenting with key sentences from chapter text...")

    # Collect all key sentences with their chunks
    all_key_sent_data = []
    for ch in chapter_info:
        for sent in ch["key_sents"]:
            # Find the chunk this sentence belongs to
            best_chunk = ch["chunks"][0]
            for chunk in ch["chunks"]:
                if sent[:50] in chunk:
                    best_chunk = chunk
                    break
            all_key_sent_data.append({
                "sent": sent,
                "chunk": best_chunk,
                "chapter": ch["chapter"],
                "subject": ch["subject"],
            })

    random.shuffle(all_key_sent_data)
    print(f"   Available key sentences: {len(all_key_sent_data)}")

    for target_level in [3, 4]:
        current = len(examples_by_level[target_level])
        needed = TARGET_PER_LEVEL - current
        if needed <= 0:
            continue

        gen_func = make_analyze_questions if target_level == 3 else make_evaluate_create_questions
        augmented = 0

        for ks in all_key_sent_data:
            if augmented >= needed:
                break
            new_target = gen_func(ks["sent"])
            examples_by_level[target_level].append({
                "chunk": ks["chunk"],
                "target": new_target,
                "level": target_level,
            })
            augmented += 1

        print(f"   L{target_level} ({LEVEL_TAGS[target_level]}): {current} → {len(examples_by_level[target_level])} (+{augmented})")

    # Create final examples with diverse prompts
    print("\n📝 Creating training examples...")
    final_examples = []

    for level in range(1, 5):
        templates = PROMPT_TEMPLATES[level]
        for ex in examples_by_level[level]:
            selected = random.sample(templates, min(2, len(templates)))
            for tmpl in selected:
                prompt = tmpl.format(text=ex["chunk"])
                final_examples.append({
                    "input_text": prompt,
                    "target_text": ex["target"],
                })

    print(f"\n📊 Final distribution:")
    level_counts = Counter()
    for ex in final_examples:
        tag = ex["target_text"].split("]")[0] + "]" if ex["target_text"].startswith("[") else "?"
        level_counts[tag] += 1
    for tag in sorted(level_counts):
        print(f"   {tag}: {level_counts[tag]}")
    print(f"   Total: {len(final_examples)}")

    # Split and save
    random.shuffle(final_examples)
    split_idx = int(len(final_examples) * (1 - VAL_SPLIT))
    train = final_examples[:split_idx]
    val = final_examples[split_idx:]

    def save(data, path):
        out = [{"input_text": ex["input_text"], "target_text": ex["target_text"]} for ex in data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {len(out)} → {path}")

    print(f"\n💾 Saving ({len(train)} train / {len(val)} val)...")
    save(train, OUTPUT_TRAIN)
    save(val, OUTPUT_VAL)

    # Show augmented samples for quality check
    print(f"\n📝 AUGMENTED SAMPLE QUALITY CHECK:")
    for level in [3, 4]:
        level_ex = [ex for ex in final_examples if f"[{LEVEL_TAGS[level]}]" in ex["target_text"]]
        augmented_ex = [ex for ex in level_ex if any(p in ex["target_text"] for p in
            ["relationship between", "Compare and contrast", "Break down", "patterns can",
             "Critically assess", "Design an experiment", "Propose an alternative",
             "Evaluate the significance", "consequences would", "real-world scenario"])]
        if augmented_ex:
            print(f"\n  L{level} ({LEVEL_TAGS[level]}) — Augmented samples:")
            for s in random.sample(augmented_ex, min(3, len(augmented_ex))):
                print(f"    → {s['target_text'][:200]}")

    print(f"\n✅ Done. Next: python train.py")


if __name__ == "__main__":
    main()

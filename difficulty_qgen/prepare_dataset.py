#!/usr/bin/env python3
"""
Enhanced dataset preparation for difficulty-conditioned question generation.

Key design choices to promote CREATIVE generation (not memorization):
1. Diverse prompt templates — model learns the intent, not one phrasing
2. Complexity-specific instructions — each level gets guidance about expected depth
3. Context variation — chunks are paired with multiple questions to learn patterns
4. Question text is cleaned of numbering/artifacts so model learns structure, not formatting

Difficulty Mapping:
  Level 1: MCQ, Matching, Assertion/Reason  → recognition/recall
  Level 2: Very Short Answer                → 1-2 sentence recall
  Level 3: Short Answer                     → multi-sentence explanation
  Level 4: Long Answer                      → detailed, analytical response
"""

import json
import re
import random
import os
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────────────────

INPUT_PATH = "../dataset/qna.json"
OUTPUT_DIR = "../dataset"
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "difficulty_dataset_train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "difficulty_dataset_val.json")
OUTPUT_FULL = os.path.join(OUTPUT_DIR, "difficulty_dataset_full.json")

VAL_SPLIT = 0.15
SEED = 42

DIFFICULTY_MAP = {
    "MCQ": 1,
    "Matching Questions": 1,
    "Assertion and Reason Questions": 1,
    "Very Short Answer Questions": 2,
    "Short Answer Questions": 3,
    "Long Answer Questions": 4,
}

# ─── Diverse prompt templates per difficulty level ────────────────────────────
# Using many templates forces the model to learn the INTENT (difficulty level)
# rather than memorizing a single prompt-to-output mapping.

PROMPT_TEMPLATES = {
    1: [
        "generate level 1 question: {text}",
        "create a multiple choice question from this text: {text}",
        "generate an objective type question based on: {text}",
        "produce an MCQ-style question that tests factual recall: {text}",
        "write a recognition-based question from: {text}",
        "formulate a level 1 (easy) question: {text}",
    ],
    2: [
        "generate level 2 question: {text}",
        "create a very short answer question from this text: {text}",
        "generate a question requiring a brief 1-2 sentence answer: {text}",
        "produce a concise recall question based on: {text}",
        "write a short factual question from: {text}",
        "formulate a level 2 (moderate) question: {text}",
    ],
    3: [
        "generate level 3 question: {text}",
        "create a short answer question from this text: {text}",
        "generate a question requiring a multi-sentence explanation: {text}",
        "produce a question that tests understanding and reasoning: {text}",
        "write an analytical question from: {text}",
        "formulate a level 3 (challenging) question: {text}",
    ],
    4: [
        "generate level 4 question: {text}",
        "create a long answer question from this text: {text}",
        "generate a question requiring a detailed paragraph-length response: {text}",
        "produce a comprehensive question that tests deep understanding: {text}",
        "write a question requiring critical analysis and synthesis: {text}",
        "formulate a level 4 (advanced) question: {text}",
    ],
}

# The canonical prompt used at inference time (first template for each level)
CANONICAL_PROMPT = "generate level {level} question: {text}"


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text):
    """Clean chapter text: normalize whitespace, remove page artifacts."""
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    text = re.sub(r'\b\d+\s*\n\s*\d+\b', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def clean_question(text):
    """Clean question text — remove numbering, page headers, artifacts."""
    # Remove year stamps like "2025-26"
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    # Remove subject headers
    text = re.sub(r'BIOLOGY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHEMISTRY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PHYSICS,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHAPTER\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'UNIT\s*[IVX]+:.*', '', text, flags=re.IGNORECASE)
    # Remove repeated page numbers like "29 29 29"
    text = re.sub(r'(\b\d{1,3}\b)(\s+\1){2,}', '', text)
    # Remove all-caps headers (>15 chars)
    text = re.sub(r'\b[A-Z ]{15,}\b', '', text)
    # Strip leading question number like "1.\n" or "12.\n"
    text = re.sub(r'^\d+\.\s*\n?', '', text)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text, max_chars=1800, overlap_chars=200):
    """Split long text into overlapping chunks at paragraph/sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if para_len > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_len = len(sent)
                if current_len + sent_len > max_chars and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    overlap = current_chunk[-1] if current_chunk else ""
                    if len(overlap) <= overlap_chars:
                        current_chunk = [overlap, sent]
                        current_len = len(overlap) + sent_len
                    else:
                        current_chunk = [sent]
                        current_len = sent_len
                else:
                    current_chunk.append(sent)
                    current_len += sent_len
        elif current_len + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            overlap = current_chunk[-1] if current_chunk else ""
            if len(overlap) <= overlap_chars:
                current_chunk = [overlap, para]
                current_len = len(overlap) + para_len
            else:
                current_chunk = [para]
                current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


# ─── Dataset Creation ────────────────────────────────────────────────────────

def create_training_examples(data):
    """
    Create training examples with diverse prompts to promote creative generation.

    For each (question, chunk) pair we create MULTIPLE training examples using
    different prompt templates. This teaches the model that "generate level 2
    question" and "create a very short answer question" mean the same thing,
    preventing it from memorizing a single input→output mapping.
    """
    examples = []
    stats = Counter()

    for chapter in data:
        chapter_text = clean_text(chapter["chapter_text"])
        chapter_name = chapter.get("chapter_name", "")
        subject = chapter.get("subject", "")

        chunks = chunk_text(chapter_text)

        for qtype, questions in chapter["questions"].items():
            if qtype not in DIFFICULTY_MAP:
                print(f"  ⚠️  Unknown question type: {qtype}, skipping")
                continue

            level = DIFFICULTY_MAP[qtype]
            templates = PROMPT_TEMPLATES[level]

            for q_text in questions:
                q_cleaned = clean_question(q_text)
                if not q_cleaned or len(q_cleaned) < 10:
                    continue

                # Find best matching chunk via keyword overlap
                q_words = set(re.findall(r'\b\w{4,}\b', q_cleaned.lower()))
                best_chunk = chunks[0]
                best_score = 0
                for chunk in chunks:
                    chunk_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
                    overlap = len(q_words & chunk_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_chunk = chunk

                # Create examples with DIVERSE prompt templates
                # Use 2 random templates per question (+ the canonical one)
                # This 3x multiplier helps the model generalize
                used_templates = set()

                # Always include the canonical prompt
                canonical = CANONICAL_PROMPT.format(level=level, text=best_chunk)
                examples.append({
                    "input_text": canonical,
                    "target_text": q_cleaned,
                    "difficulty": level,
                    "question_type": qtype,
                    "subject": subject,
                    "chapter": chapter_name,
                })
                used_templates.add(0)
                stats[f"level_{level}"] += 1
                stats["total"] += 1

                # Add 2 random alternative templates
                available = [i for i in range(len(templates)) if i not in used_templates]
                for idx in random.sample(available, min(2, len(available))):
                    alt_prompt = templates[idx].format(text=best_chunk)
                    examples.append({
                        "input_text": alt_prompt,
                        "target_text": q_cleaned,
                        "difficulty": level,
                        "question_type": qtype,
                        "subject": subject,
                        "chapter": chapter_name,
                    })
                    stats[f"level_{level}"] += 1
                    stats["total"] += 1

    return examples, stats


def main():
    random.seed(SEED)

    print("=" * 60)
    print("ENHANCED DIFFICULTY DATASET PREPARATION")
    print("=" * 60)
    print("Diverse prompts for creative generation\n")

    # Load
    print(f"📂 Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   {len(data)} chapters loaded\n")

    # Create examples
    print("🔄 Creating training examples (3x augmented with diverse prompts)...")
    examples, stats = create_training_examples(data)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {stats['total']}")
    for level in range(1, 5):
        count = stats.get(f"level_{level}", 0)
        pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"   Level {level}: {count} ({pct:.1f}%)")

    # Shuffle
    random.shuffle(examples)

    # Split
    split_idx = int(len(examples) * (1 - VAL_SPLIT))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"\n📏 Split:")
    print(f"   Train: {len(train_examples)}")
    print(f"   Val:   {len(val_examples)}")

    # Length stats
    input_lens = [len(ex["input_text"]) for ex in examples]
    target_lens = [len(ex["target_text"]) for ex in examples]
    print(f"\n📐 Length Stats (chars):")
    print(f"   Input  — mean: {sum(input_lens)/len(input_lens):.0f}, max: {max(input_lens)}")
    print(f"   Target — mean: {sum(target_lens)/len(target_lens):.0f}, max: {max(target_lens)}")

    # Show one sample per level
    print(f"\n📝 Samples:")
    print("-" * 60)
    for level in range(1, 5):
        level_ex = [ex for ex in examples if ex["difficulty"] == level]
        if level_ex:
            s = level_ex[0]
            print(f"\n  Level {level} | {s['subject']} — {s['chapter']}")
            print(f"  Prompt:  {s['input_text'][:130]}...")
            print(f"  Target:  {s['target_text'][:130]}...")

    # Save (training format: only input_text + target_text)
    def save(data, path):
        out = [{"input_text": ex["input_text"], "target_text": ex["target_text"]} for ex in data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {len(out)} examples → {path}")

    print(f"\n💾 Saving...")
    save(train_examples, OUTPUT_TRAIN)
    save(val_examples, OUTPUT_VAL)

    # Full dataset with metadata
    with open(OUTPUT_FULL, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Full metadata → {OUTPUT_FULL}")

    print(f"\n✅ Done. Next: python train.py")


if __name__ == "__main__":
    main()

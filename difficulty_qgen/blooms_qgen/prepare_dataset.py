#!/usr/bin/env python3
"""
Bloom's Taxonomy Question Generation — Dataset Preparation

Maps NCERT questions to 6 Bloom's levels and creates training data with
STRUCTURED output format (MCQs include options, long answers include
sub-parts, etc.)

Bloom's Levels:
  1. Remember  — "What is the definition of...?"    → Detects: Recall failure
  2. Understand — "Explain in your own words why..." → Detects: Comprehension failure
  3. Apply     — "How would you use this concept..." → Detects: Transfer failure
  4. Analyze   — "What is the relationship between X and Y?" → Detects: Structural gaps
  5. Evaluate  — "Which approach is better and why?" → Detects: Critical thinking deficits
  6. Create    — "Design a solution that uses..."   → Detects: Synthesis capability

Mapping from NCERT question types:
  MCQ                         → Level 1 (Remember) and Level 2 (Understand)
  Very Short Answer Questions → Level 1 (Remember) and Level 2 (Understand)
  Short Answer Questions      → Level 3 (Apply) and Level 4 (Analyze)
  Assertion and Reason        → Level 4 (Analyze) and Level 5 (Evaluate)
  Long Answer Questions       → Level 5 (Evaluate) and Level 6 (Create)
  Matching Questions          → Level 1 (Remember)
"""

import json
import re
import random
import os
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────────────────

INPUT_PATH = "../../dataset/qna.json"
OUTPUT_DIR = "../../dataset"
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "blooms_dataset_train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "blooms_dataset_val.json")

VAL_SPLIT = 0.15
SEED = 42

BLOOMS_LEVELS = {
    1: "remember",
    2: "understand",
    3: "apply",
    4: "analyze",
    5: "evaluate",
    6: "create",
}

BLOOMS_DESCRIPTIONS = {
    1: "Recall facts, definitions, and basic concepts",
    2: "Explain ideas, interpret meaning, compare concepts",
    3: "Use knowledge in new situations, solve problems",
    4: "Examine relationships, draw connections, find patterns",
    5: "Judge, justify, critique, and defend positions",
    6: "Design, construct, produce, or propose new ideas",
}

# Map NCERT question types → Bloom's level(s)
# Some types span two levels — we assign based on the question's characteristics
QUESTION_TYPE_TO_BLOOMS = {
    "MCQ": [1, 2],                              # Remember + Understand
    "Matching Questions": [1],                   # Remember
    "Very Short Answer Questions": [1, 2],       # Remember + Understand
    "Short Answer Questions": [3, 4],            # Apply + Analyze
    "Assertion and Reason Questions": [4, 5],    # Analyze + Evaluate
    "Long Answer Questions": [5, 6],             # Evaluate + Create
}

# Diverse prompt templates per Bloom's level
PROMPT_TEMPLATES = {
    1: [
        "generate a remember level question: {text}",
        "create a recall-based question from: {text}",
        "generate a Bloom's level 1 question testing factual recall: {text}",
        "write a question asking what, who, when, or where based on: {text}",
    ],
    2: [
        "generate an understand level question: {text}",
        "create a comprehension question from: {text}",
        "generate a Bloom's level 2 question testing understanding: {text}",
        "write a question asking to explain or describe based on: {text}",
    ],
    3: [
        "generate an apply level question: {text}",
        "create a question requiring application of concepts from: {text}",
        "generate a Bloom's level 3 question testing knowledge transfer: {text}",
        "write a question asking how to use or solve based on: {text}",
    ],
    4: [
        "generate an analyze level question: {text}",
        "create a question requiring analysis of relationships from: {text}",
        "generate a Bloom's level 4 question testing analytical thinking: {text}",
        "write a question asking to compare, contrast, or relate based on: {text}",
    ],
    5: [
        "generate an evaluate level question: {text}",
        "create a question requiring critical evaluation from: {text}",
        "generate a Bloom's level 5 question testing judgement: {text}",
        "write a question asking to judge, justify, or critique based on: {text}",
    ],
    6: [
        "generate a create level question: {text}",
        "create a question requiring synthesis and design from: {text}",
        "generate a Bloom's level 6 question testing creativity: {text}",
        "write a question asking to design, propose, or construct based on: {text}",
    ],
}


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text):
    """Clean chapter text."""
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    text = re.sub(r'\b\d+\s*\n\s*\d+\b', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def clean_question(text):
    """Clean question text — keep structure (options, sub-parts) intact."""
    text = re.sub(r'\b\d{4}-\d{2}\b', '', text)
    text = re.sub(r'BIOLOGY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHEMISTRY,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PHYSICS,?\s*EXEMPLAR\s*PROBLEMS?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHAPTER\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'UNIT\s*[IVX]+:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\b\d{1,3}\b)(\s+\1){2,}', '', text)
    text = re.sub(r'\b[A-Z ]{15,}\b', '', text)
    # Strip leading question number but keep everything else (options etc.)
    text = re.sub(r'^\d+\.\s*\n?', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()


def has_mcq_options(text):
    """Check if question text includes MCQ options (a. b. c. d.)."""
    return bool(re.search(r'\b[a-d]\.\s', text))


def classify_blooms_level(question_text, question_type, available_levels):
    """
    Heuristically assign a Bloom's level based on question text keywords.
    For types that span two levels, use keyword analysis to pick one.
    """
    if len(available_levels) == 1:
        return available_levels[0]

    q_lower = question_text.lower()

    # Keywords that suggest higher-order thinking
    higher_keywords = [
        'explain', 'describe', 'discuss', 'compare', 'contrast', 'differentiate',
        'analyze', 'evaluate', 'justify', 'critique', 'design', 'propose',
        'how would you', 'what would happen', 'why do you think', 'suggest',
        'draw', 'illustrate', 'construct', 'create', 'develop'
    ]
    lower_keywords = [
        'what is', 'define', 'name', 'list', 'identify', 'state',
        'which of the following', 'select', 'choose', 'match', 'fill'
    ]

    higher_score = sum(1 for kw in higher_keywords if kw in q_lower)
    lower_score = sum(1 for kw in lower_keywords if kw in q_lower)

    if higher_score > lower_score:
        return max(available_levels)
    else:
        return min(available_levels)


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text, max_chars=1800, overlap_chars=200):
    """Split long text into overlapping chunks."""
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


# ─── Format targets with Bloom's structure ───────────────────────────────────

def format_target(question_text, blooms_level, question_type):
    """
    Format the target text to make the output structure explicit.
    This teaches the model what each Bloom's level LOOKS LIKE.
    """
    level_name = BLOOMS_LEVELS[blooms_level]

    # For MCQs - ensure options are clearly formatted
    if has_mcq_options(question_text):
        return f"[{level_name.upper()}] {question_text}"

    # For Assertion/Reason - keep the full structured format
    if question_type == "Assertion and Reason Questions":
        return f"[{level_name.upper()}] {question_text}"

    # For all other types - prefix with level tag
    return f"[{level_name.upper()}] {question_text}"


# ─── Dataset Creation ────────────────────────────────────────────────────────

def create_training_examples(data):
    """Create Bloom's-aligned training examples."""
    examples = []
    stats = Counter()

    for chapter in data:
        chapter_text = clean_text(chapter["chapter_text"])
        chapter_name = chapter.get("chapter_name", "")
        subject = chapter.get("subject", "")
        chunks = chunk_text(chapter_text)

        for qtype, questions in chapter["questions"].items():
            if qtype not in QUESTION_TYPE_TO_BLOOMS:
                print(f"  ⚠️  Unknown: {qtype}")
                continue

            available_levels = QUESTION_TYPE_TO_BLOOMS[qtype]

            for q_text in questions:
                q_cleaned = clean_question(q_text)
                if not q_cleaned or len(q_cleaned) < 10:
                    continue

                # Assign Bloom's level
                level = classify_blooms_level(q_cleaned, qtype, available_levels)

                # Find best chunk
                q_words = set(re.findall(r'\b\w{4,}\b', q_cleaned.lower()))
                best_chunk = chunks[0]
                best_score = 0
                for chunk in chunks:
                    chunk_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
                    overlap = len(q_words & chunk_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_chunk = chunk

                # Format target with Bloom's level tag
                target = format_target(q_cleaned, level, qtype)

                # Create examples with diverse prompts (2 per question)
                templates = PROMPT_TEMPLATES[level]
                selected = random.sample(templates, min(2, len(templates)))

                for template in selected:
                    prompt = template.format(text=best_chunk)
                    examples.append({
                        "input_text": prompt,
                        "target_text": target,
                        "blooms_level": level,
                        "blooms_name": BLOOMS_LEVELS[level],
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
    print("BLOOM'S TAXONOMY DATASET PREPARATION")
    print("=" * 60 + "\n")

    print(f"📂 Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   {len(data)} chapters\n")

    print("🔄 Creating Bloom's-aligned training examples...")
    examples, stats = create_training_examples(data)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total: {stats['total']} examples")
    for level in range(1, 7):
        count = stats.get(f"level_{level}", 0)
        pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
        name = BLOOMS_LEVELS[level]
        print(f"   Level {level} ({name:10s}): {count:5d} ({pct:5.1f}%)")

    random.shuffle(examples)

    split_idx = int(len(examples) * (1 - VAL_SPLIT))
    train = examples[:split_idx]
    val = examples[split_idx:]

    print(f"\n📏 Split: {len(train)} train / {len(val)} val")

    # Length stats
    t_lens = [len(ex["target_text"]) for ex in examples]
    print(f"\n📐 Target length: mean={sum(t_lens)/len(t_lens):.0f}, max={max(t_lens)}, min={min(t_lens)}")

    # Samples
    print(f"\n📝 Samples:")
    print("-" * 60)
    for level in range(1, 7):
        level_ex = [ex for ex in examples if ex["blooms_level"] == level]
        if level_ex:
            s = level_ex[0]
            print(f"\n  Level {level} ({s['blooms_name']}) | {s['question_type']}")
            print(f"  Prompt: {s['input_text'][:120]}...")
            print(f"  Target: {s['target_text'][:200]}...")

    # Save
    def save(data, path):
        out = [{"input_text": ex["input_text"], "target_text": ex["target_text"]} for ex in data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {len(out)} → {path}")

    print(f"\n💾 Saving...")
    save(train, OUTPUT_TRAIN)
    save(val, OUTPUT_VAL)

    print(f"\n✅ Done. Next: python train.py")


if __name__ == "__main__":
    main()

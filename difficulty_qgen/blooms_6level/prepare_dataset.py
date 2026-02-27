#!/usr/bin/env python3
"""
Option A: 6-Level Bloom's Taxonomy — Augmented & Balanced Dataset

AUGMENTATION STRATEGY:
Extracts KEY SENTENCES from chapter text and builds proper questions
using Bloom's verb starters. This produces coherent, educational questions.

Levels: Remember, Understand, Apply, Analyze, Evaluate, Create
"""

import json
import re
import random
import os
from collections import Counter

INPUT_PATH = "../../dataset/qna.json"
OUTPUT_DIR = "../../dataset"
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "blooms_6level_train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "blooms_6level_val.json")

VAL_SPLIT = 0.15
SEED = 42

BLOOMS = {1: "remember", 2: "understand", 3: "apply", 4: "analyze", 5: "evaluate", 6: "create"}

QUESTION_TYPE_TO_BLOOMS = {
    "MCQ": [1, 2],
    "Matching Questions": [1],
    "Very Short Answer Questions": [1, 2],
    "Short Answer Questions": [3, 4],
    "Assertion and Reason Questions": [4, 5],
    "Long Answer Questions": [5, 6],
}

PROMPT_TEMPLATES = {
    1: [
        "generate a remember level question: {text}",
        "create a recall-based question from: {text}",
        "generate a Bloom's level 1 question testing factual recall: {text}",
    ],
    2: [
        "generate an understand level question: {text}",
        "create a comprehension question from: {text}",
        "generate a Bloom's level 2 question testing understanding: {text}",
    ],
    3: [
        "generate an apply level question: {text}",
        "create a question requiring application of concepts from: {text}",
        "generate a Bloom's level 3 question testing knowledge transfer: {text}",
    ],
    4: [
        "generate an analyze level question: {text}",
        "create a question requiring analysis from: {text}",
        "generate a Bloom's level 4 question testing analytical thinking: {text}",
    ],
    5: [
        "generate an evaluate level question: {text}",
        "create a question requiring critical evaluation from: {text}",
        "generate a Bloom's level 5 question testing judgement: {text}",
    ],
    6: [
        "generate a create level question: {text}",
        "create a question requiring synthesis and design from: {text}",
        "generate a Bloom's level 6 question testing creativity: {text}",
    ],
}


# ─── AUGMENTATION: Clean question generators from key sentences ──────────────

def make_understand_questions(key_sentence):
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[UNDERSTAND] Explain in your own words why {sent.lower()}.",
        f"[UNDERSTAND] Describe the significance of the following: {sent}.",
        f"[UNDERSTAND] Summarize the main idea presented in: '{sent}'.",
        f"[UNDERSTAND] What does it mean when we say that {sent.lower()}?",
        f"[UNDERSTAND] Interpret the concept described here: {sent}.",
        f"[UNDERSTAND] Paraphrase the following statement: '{sent}'.",
    ]
    return random.choice(templates)

def make_analyze_questions(key_sentence):
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[ANALYZE] What is the relationship between the factors described in: '{sent}'?",
        f"[ANALYZE] Compare and contrast the concepts presented in: {sent}.",
        f"[ANALYZE] Break down the process described and identify key components: {sent}.",
        f"[ANALYZE] What patterns can you identify from the fact that {sent.lower()}?",
        f"[ANALYZE] Differentiate between the terms and concepts mentioned in: {sent}.",
        f"[ANALYZE] Examine how '{sent[:60]}' connects to the broader topic.",
    ]
    return random.choice(templates)

def make_evaluate_questions(key_sentence):
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[EVALUATE] Critically assess the statement: '{sent}'. Do you agree? Justify your answer.",
        f"[EVALUATE] What are the strengths and limitations of the concept that {sent.lower()}?",
        f"[EVALUATE] Evaluate the significance of the following in this context: {sent}.",
        f"[EVALUATE] If '{sent[:60]}' were not true, what consequences would follow?",
        f"[EVALUATE] Judge whether the following claim is well-supported: {sent}.",
        f"[EVALUATE] Which aspects of '{sent[:60]}' are most important and why?",
    ]
    return random.choice(templates)

def make_create_questions(key_sentence):
    sent = key_sentence.strip().rstrip('.')
    templates = [
        f"[CREATE] Design an experiment to demonstrate: {sent}.",
        f"[CREATE] Propose an alternative explanation for: {sent}.",
        f"[CREATE] Create a real-world scenario where the concept '{sent[:60]}' would be practically useful.",
        f"[CREATE] Develop a hypothesis related to: {sent}. Outline how you would test it.",
        f"[CREATE] Suggest a new method to investigate: {sent}.",
        f"[CREATE] How would you modify the concept of '{sent[:60]}' to apply it in a different context?",
    ]
    return random.choice(templates)


AUGMENT_FUNCS = {2: make_understand_questions, 4: make_analyze_questions,
                 5: make_evaluate_questions, 6: make_create_questions}


def extract_key_sentences(chapter_text, min_len=40, max_len=200):
    """Extract informative key sentences from chapter text for augmentation."""
    sentences = re.split(r'(?<=[.!?])\s+', chapter_text)
    key_sents = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < min_len or len(sent) > max_len:
            continue
        if re.match(r'^\d+\.?\s*$', sent):
            continue
        if sent.isupper():
            continue
        if re.match(r'^(Fig|Figure|Table|Note|Hint|Example)\b', sent, re.IGNORECASE):
            continue
        if sent.count(' ') < 5:
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
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current, current_len = [], 0
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

def classify_blooms_level(question_text, available_levels):
    if len(available_levels) == 1:
        return available_levels[0]
    q_lower = question_text.lower()
    higher = ['explain', 'describe', 'discuss', 'compare', 'contrast', 'differentiate',
              'analyze', 'evaluate', 'justify', 'design', 'propose', 'how would you',
              'what would happen', 'why do you think', 'suggest', 'draw', 'illustrate']
    lower = ['what is', 'define', 'name', 'list', 'identify', 'state',
             'which of the following', 'select', 'choose', 'match', 'fill']
    h = sum(1 for kw in higher if kw in q_lower)
    l = sum(1 for kw in lower if kw in q_lower)
    return max(available_levels) if h > l else min(available_levels)

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
    print("OPTION A: 6-LEVEL BLOOM'S — AUGMENTED DATASET")
    print("=" * 60 + "\n")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 {len(data)} chapters loaded\n")

    examples_by_level = {i: [] for i in range(1, 7)}
    chapter_info = []

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
            if qtype not in QUESTION_TYPE_TO_BLOOMS:
                continue
            available_levels = QUESTION_TYPE_TO_BLOOMS[qtype]
            for q_text in questions:
                q_cleaned = clean_question(q_text)
                if not q_cleaned or len(q_cleaned) < 10:
                    continue
                level = classify_blooms_level(q_cleaned, available_levels)
                best_chunk = find_best_chunk(q_cleaned, chunks)
                target = f"[{BLOOMS[level].upper()}] {q_cleaned}"
                examples_by_level[level].append({
                    "chunk": best_chunk, "target": target, "level": level,
                })

    print("📊 Base distribution:")
    for lv in range(1, 7):
        print(f"   L{lv} ({BLOOMS[lv]:10s}): {len(examples_by_level[lv])}")

    TARGET_PER_LEVEL = 700

    if len(examples_by_level[1]) > TARGET_PER_LEVEL:
        examples_by_level[1] = random.sample(examples_by_level[1], TARGET_PER_LEVEL)
        print(f"\n✂️  L1 downsampled to {TARGET_PER_LEVEL}")

    # Collect all key sentences with their chunks
    all_key_sent_data = []
    for ch in chapter_info:
        for sent in ch["key_sents"]:
            best_chunk = ch["chunks"][0]
            for chunk in ch["chunks"]:
                if sent[:50] in chunk:
                    best_chunk = chunk
                    break
            all_key_sent_data.append({
                "sent": sent, "chunk": best_chunk,
                "chapter": ch["chapter"], "subject": ch["subject"],
            })
    random.shuffle(all_key_sent_data)
    print(f"\n   Available key sentences: {len(all_key_sent_data)}")

    # Augment L2, L4, L5, L6
    print("\n🔄 Augmenting with key sentences from chapter text...")
    for target_level in [2, 4, 5, 6]:
        current = len(examples_by_level[target_level])
        needed = TARGET_PER_LEVEL - current
        if needed <= 0:
            continue

        gen_func = AUGMENT_FUNCS[target_level]
        augmented = 0
        for ks in all_key_sent_data:
            if augmented >= needed:
                break
            new_target = gen_func(ks["sent"])
            examples_by_level[target_level].append({
                "chunk": ks["chunk"], "target": new_target, "level": target_level,
            })
            augmented += 1

        print(f"   L{target_level} ({BLOOMS[target_level]}): {current} → {len(examples_by_level[target_level])} (+{augmented})")

    # Create final examples
    print("\n📝 Creating training examples...")
    final_examples = []

    for level in range(1, 7):
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
    for level in [2, 5, 6]:
        level_ex = [ex for ex in final_examples if f"[{BLOOMS[level].upper()}]" in ex["target_text"]]
        augmented_ex = [ex for ex in level_ex if any(p in ex["target_text"] for p in
            ["Explain in your own", "Describe the significance", "Summarize the main",
             "Critically assess", "strengths and limitations", "Design an experiment",
             "Propose an alternative", "real-world scenario", "Develop a hypothesis"])]
        if augmented_ex:
            print(f"\n  L{level} ({BLOOMS[level]}) — Augmented samples:")
            for s in random.sample(augmented_ex, min(3, len(augmented_ex))):
                print(f"    → {s['target_text'][:200]}")

    print(f"\n✅ Done. Next: python train.py")


if __name__ == "__main__":
    main()

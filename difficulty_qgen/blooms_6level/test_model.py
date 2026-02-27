#!/usr/bin/env python3
"""Option A: 6-Level Bloom's — Test Script"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "../../question_gen_blooms_6level_model"
BASE_MODEL = "google/flan-t5-small"

BLOOMS = {
    1: ("Remember",    "Recall facts, definitions"),
    2: ("Understand",  "Explain ideas, interpret meaning"),
    3: ("Apply",       "Use knowledge in new situations"),
    4: ("Analyze",     "Examine relationships, find patterns"),
    5: ("Evaluate",    "Judge, justify, critique positions"),
    6: ("Create",      "Design, propose new ideas"),
}

TEST_TEXTS = [
    {"name": "Biology — Cell Division",
     "text": "Mitosis is a type of cell division in which a single cell divides to produce two genetically identical daughter cells. The process consists of four stages: prophase, metaphase, anaphase, and telophase. During prophase, chromosomes condense and become visible. In metaphase, chromosomes align at the cell's equator. Anaphase involves the separation of sister chromatids, and in telophase, nuclear envelopes reform around the separated chromosomes."},
    {"name": "Physics — Thermodynamics",
     "text": "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another. The internal energy of a system changes when heat is added or work is done. Mathematically, ΔU = Q - W, where ΔU is change in internal energy, Q is heat added, and W is work done by the system."},
    {"name": "Chemistry — Periodic Table",
     "text": "Elements in the periodic table are arranged by increasing atomic number. Elements in the same group have similar chemical properties because they have the same number of valence electrons. Ionization energy generally increases across a period from left to right and decreases down a group."},
]


def load_model():
    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    print("   ✅ Loaded\n")
    return model, tokenizer


def generate(model, tokenizer, text, level, creative=True):
    name = BLOOMS[level][0].lower()
    prompt = f"generate a {name} level question: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    with torch.no_grad():
        if creative:
            out = model.generate(**inputs, max_length=384, do_sample=True, temperature=0.8, top_p=0.92, top_k=50, no_repeat_ngram_size=3, repetition_penalty=1.3)
        else:
            out = model.generate(**inputs, max_length=384, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def test_all_levels(model, tokenizer):
    print("=" * 80)
    print("6-LEVEL BLOOM'S — ALL LEVELS TEST")
    print("=" * 80 + "\n")
    for test in TEST_TEXTS:
        print(f"📖 {test['name']}")
        print("-" * 80)
        for level in range(1, 7):
            name, desc = BLOOMS[level]
            output = generate(model, tokenizer, test["text"], level)
            print(f"  L{level} {name} ({desc}):")
            print(f"  → {output}\n")
        print("=" * 80 + "\n")


def test_format(model, tokenizer):
    print("=" * 80)
    print("FORMAT VALIDATION")
    print("=" * 80 + "\n")
    text = TEST_TEXTS[0]["text"]
    tags = {1:"[REMEMBER]", 2:"[UNDERSTAND]", 3:"[APPLY]", 4:"[ANALYZE]", 5:"[EVALUATE]", 6:"[CREATE]"}
    score = 0
    for level in range(1, 7):
        output = generate(model, tokenizer, text, level, creative=False)
        tag = tags[level]
        found = tag in output.upper()
        if found: score += 1
        print(f"  {'✅' if found else '⚠️'} L{level} ({BLOOMS[level][0]}): {'found' if found else 'MISSING'}")
        print(f"     → {output[:150]}\n")
    print(f"  Score: {score}/6 tags correct\n")


def test_novelty(model, tokenizer):
    print("=" * 80)
    print("NOVELTY TEST — 3 runs per level")
    print("=" * 80 + "\n")
    text = TEST_TEXTS[0]["text"]
    total_unique = 0
    for level in range(1, 7):
        outputs = [generate(model, tokenizer, text, level) for _ in range(3)]
        for i, o in enumerate(outputs):
            print(f"  L{level} Run {i+1}: {o[:100]}...")
        unique = len(set(outputs))
        total_unique += unique
        print(f"  → {unique}/3 unique\n")
    print(f"  Total: {total_unique}/18 unique\n")


def interactive(model, tokenizer):
    print("=" * 80)
    print("INTERACTIVE (type 'quit' to exit)")
    print("=" * 80 + "\n")
    while True:
        text = input("📝 Text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        try:
            level = int(input("🎯 Level (1-6): ").strip())
            level = max(1, min(6, level))
        except ValueError:
            level = 1
        print(f"🤖 → {generate(model, tokenizer, text, level)}\n")


def main():
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ Cannot load: {e}\n   Run: python train.py first")
        return
    print("1. All levels  2. Format  3. Novelty  4. Interactive  5. All")
    c = input("Choice: ").strip()
    if c == "1": test_all_levels(model, tokenizer)
    elif c == "2": test_format(model, tokenizer)
    elif c == "3": test_novelty(model, tokenizer)
    elif c == "4": interactive(model, tokenizer)
    else:
        test_all_levels(model, tokenizer)
        test_format(model, tokenizer)
        test_novelty(model, tokenizer)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bloom's Taxonomy Question Generation — Test Script

Tests the model across all 6 Bloom's levels and validates that:
1. Output format matches the level (MCQ has options, higher levels have depth)
2. Questions are diverse (not memorized from training data)
3. Level tag in output matches the requested level
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "../../question_gen_blooms_model"
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
    {
        "name": "Biology — Cell Division",
        "text": (
            "Mitosis is a type of cell division in which a single cell divides to produce "
            "two genetically identical daughter cells. The process consists of four stages: "
            "prophase, metaphase, anaphase, and telophase. During prophase, chromosomes "
            "condense and become visible. In metaphase, chromosomes align at the cell's "
            "equator. Anaphase involves the separation of sister chromatids, and in "
            "telophase, nuclear envelopes reform around the separated chromosomes."
        ),
    },
    {
        "name": "Physics — Thermodynamics",
        "text": (
            "The first law of thermodynamics states that energy cannot be created or "
            "destroyed, only transformed from one form to another. The internal energy "
            "of a system changes when heat is added or work is done. Mathematically, "
            "ΔU = Q - W, where ΔU is change in internal energy, Q is heat added, and "
            "W is work done by the system. An adiabatic process occurs when no heat "
            "is exchanged with the surroundings."
        ),
    },
    {
        "name": "Chemistry — Periodic Table",
        "text": (
            "Elements in the periodic table are arranged by increasing atomic number. "
            "Elements in the same group have similar chemical properties because they "
            "have the same number of valence electrons. Ionization energy generally "
            "increases across a period from left to right and decreases down a group. "
            "Electronegativity follows a similar trend. Noble gases have complete "
            "outer electron shells and are generally unreactive."
        ),
    },
]


def load_model():
    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    print("   ✅ Loaded\n")
    return model, tokenizer


def generate(model, tokenizer, text, level, creative=True):
    """Generate a question at a specific Bloom's level."""
    level_name = BLOOMS[level][0].lower()
    prompt = f"generate a {level_name} level question: {text}"

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True,
    ).to(model.device)

    with torch.no_grad():
        if creative:
            out = model.generate(
                **inputs, max_length=384,
                do_sample=True, temperature=0.8, top_p=0.92, top_k=50,
                no_repeat_ngram_size=3, repetition_penalty=1.3,
            )
        else:
            out = model.generate(
                **inputs, max_length=384,
                num_beams=4, early_stopping=True,
                no_repeat_ngram_size=3, repetition_penalty=1.2,
            )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def test_all_levels(model, tokenizer):
    """Test all 6 Bloom's levels on each test text."""
    print("=" * 80)
    print("BLOOM'S TAXONOMY — ALL LEVELS TEST")
    print("=" * 80 + "\n")

    for test in TEST_TEXTS:
        print(f"📖 {test['name']}")
        print(f"   {test['text'][:90]}...")
        print("-" * 80)

        for level in range(1, 7):
            name, desc = BLOOMS[level]
            output = generate(model, tokenizer, test["text"], level)
            print(f"\n  L{level} {name} ({desc}):")
            print(f"  → {output}")

        print("\n" + "=" * 80 + "\n")


def test_format_validation(model, tokenizer):
    """Check if outputs contain the expected level tags."""
    print("=" * 80)
    print("FORMAT VALIDATION — Checking [LEVEL_TAG] in output")
    print("=" * 80 + "\n")

    text = TEST_TEXTS[0]["text"]
    expected_tags = {
        1: "[REMEMBER]", 2: "[UNDERSTAND]", 3: "[APPLY]",
        4: "[ANALYZE]", 5: "[EVALUATE]", 6: "[CREATE]",
    }

    for level in range(1, 7):
        output = generate(model, tokenizer, text, level, creative=False)
        tag = expected_tags[level]
        has_tag = tag in output.upper()
        status = "✅" if has_tag else "⚠️"
        print(f"  {status} L{level} ({BLOOMS[level][0]}): {'tag found' if has_tag else 'tag missing'}")
        print(f"     Output: {output[:150]}...")
        print()


def test_novelty(model, tokenizer):
    """Generate multiple outputs per level to verify diversity."""
    print("=" * 80)
    print("NOVELTY TEST — 3 runs per level, same input")
    print("=" * 80 + "\n")

    text = TEST_TEXTS[0]["text"]

    for level in range(1, 7):
        name = BLOOMS[level][0]
        outputs = []
        for i in range(3):
            out = generate(model, tokenizer, text, level, creative=True)
            outputs.append(out)
            print(f"  L{level} Run {i+1}: {out[:100]}...")

        unique = len(set(outputs))
        avg_len = sum(len(o) for o in outputs) / len(outputs)
        print(f"  → {unique}/3 unique, avg {avg_len:.0f} chars\n")


def interactive(model, tokenizer):
    """Interactive mode."""
    print("=" * 80)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("Bloom's levels: 1=Remember 2=Understand 3=Apply 4=Analyze 5=Evaluate 6=Create")
    print("=" * 80 + "\n")

    while True:
        text = input("📝 Text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        try:
            level = int(input("🎯 Bloom's level (1-6): ").strip())
            level = max(1, min(6, level))
        except ValueError:
            level = 1
        output = generate(model, tokenizer, text, level)
        print(f"\n🤖 L{level} ({BLOOMS[level][0]}): {output}")
        print("-" * 80 + "\n")


def main():
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ Cannot load model: {e}")
        print(f"   Run: python train.py  first")
        return

    print("Choose mode:")
    print("  1. Test all 6 levels")
    print("  2. Format validation")
    print("  3. Novelty test")
    print("  4. Interactive")
    print("  5. All")

    choice = input("\nChoice (1-5): ").strip()
    if choice == "1":
        test_all_levels(model, tokenizer)
    elif choice == "2":
        test_format_validation(model, tokenizer)
    elif choice == "3":
        test_novelty(model, tokenizer)
    elif choice == "4":
        interactive(model, tokenizer)
    elif choice == "5":
        test_all_levels(model, tokenizer)
        test_format_validation(model, tokenizer)
        test_novelty(model, tokenizer)
        interactive(model, tokenizer)
    else:
        test_all_levels(model, tokenizer)


if __name__ == "__main__":
    main()

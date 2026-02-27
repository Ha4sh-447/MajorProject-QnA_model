#!/usr/bin/env python3
"""
Test the difficulty-conditioned question generation model.
Tests at each difficulty level, measures quality, and provides interactive mode.

Uses sampling-based generation (temperature + top-p) to ensure novel output
rather than deterministic beam search that might reproduce training data.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_DIR = "../question_gen_difficulty_model"
BASE_MODEL = "google/flan-t5-small"

DIFFICULTY_LABELS = {
    1: "Multiple Choice (MCQ)",
    2: "Very Short Answer",
    3: "Short Answer",
    4: "Long Answer",
}

TEST_TEXTS = [
    {
        "name": "Biology — Photosynthesis",
        "text": (
            "Photosynthesis is the process by which green plants and certain other "
            "organisms transform light energy into chemical energy. During "
            "photosynthesis in green plants, light energy is captured and used to "
            "convert water, carbon dioxide, and minerals into oxygen and energy-rich "
            "organic compounds. The light reactions occur in the thylakoid membranes "
            "and the Calvin cycle takes place in the stroma of the chloroplasts."
        ),
    },
    {
        "name": "Physics — Newton's Laws",
        "text": (
            "Newton's first law of motion states that an object at rest stays at "
            "rest and an object in motion stays in motion with the same speed and "
            "in the same direction unless acted upon by an unbalanced force. "
            "Newton's second law states that the acceleration of an object is "
            "directly proportional to the net force acting on the object and "
            "inversely proportional to its mass. The mathematical expression is "
            "F = ma. Newton's third law states that for every action, there is "
            "an equal and opposite reaction."
        ),
    },
    {
        "name": "Chemistry — Chemical Bonding",
        "text": (
            "Chemical bonds are the forces that hold atoms together in molecules "
            "and compounds. There are three main types of chemical bonds: ionic "
            "bonds, covalent bonds, and metallic bonds. Ionic bonds form when "
            "electrons are transferred from one atom to another, creating ions "
            "that attract each other. Covalent bonds form when atoms share "
            "electrons. The octet rule states that atoms tend to form bonds "
            "until they have eight electrons in their valence shell."
        ),
    },
]


def load_model():
    """Load the fine-tuned LoRA model."""
    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    print("   ✅ Loaded\n")
    return model, tokenizer


def generate_question(model, tokenizer, text, difficulty_level,
                      creative=True, max_length=256):
    """
    Generate a question at a specific difficulty level.

    When creative=True, uses sampling (temperature + top-p) for novel output.
    When creative=False, uses beam search (more deterministic, closer to training data).
    """
    prompt = f"generate level {difficulty_level} question: {text}"

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True,
    ).to(model.device)

    with torch.no_grad():
        if creative:
            # Sampling-based: produces novel, diverse questions
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
        else:
            # Beam search: more deterministic, higher quality but less diverse
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_all_levels(model, tokenizer):
    """Test generation at all difficulty levels."""
    print("=" * 80)
    print("TESTING ALL DIFFICULTY LEVELS")
    print("=" * 80 + "\n")

    for test in TEST_TEXTS:
        print(f"📖 {test['name']}")
        print(f"   {test['text'][:100]}...")
        print("-" * 80)

        for level in range(1, 5):
            label = DIFFICULTY_LABELS[level]
            # Generate with creative sampling
            output = generate_question(model, tokenizer, test["text"], level, creative=True)
            print(f"\n  Level {level} ({label}):")
            print(f"  → {output}")

        print("\n" + "=" * 80 + "\n")


def test_novelty(model, tokenizer):
    """Generate multiple outputs for the same input to verify diversity."""
    print("=" * 80)
    print("NOVELTY & DIVERSITY TEST")
    print("Generating 3 questions per level from the same text")
    print("=" * 80 + "\n")

    text = TEST_TEXTS[0]["text"]

    for level in range(1, 5):
        print(f"  Level {level} ({DIFFICULTY_LABELS[level]}):")
        outputs = []
        for i in range(3):
            out = generate_question(model, tokenizer, text, level, creative=True)
            outputs.append(out)
            print(f"    Run {i+1}: {out[:120]}...")

        # Check diversity
        unique = len(set(outputs))
        avg_len = sum(len(o) for o in outputs) / len(outputs)
        print(f"    → {unique}/3 unique outputs, avg {avg_len:.0f} chars")
        print()


def interactive_mode(model, tokenizer):
    """Interactive testing."""
    print("=" * 80)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 80 + "\n")

    while True:
        text = input("📝 Text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            level = int(input("🎯 Level (1-4): ").strip())
            level = max(1, min(4, level))
        except ValueError:
            level = 1

        output = generate_question(model, tokenizer, text, level, creative=True)
        print(f"\n🤖 Level {level} ({DIFFICULTY_LABELS[level]}):")
        print(f"   {output}")
        print("-" * 80 + "\n")


def main():
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ Cannot load model: {e}")
        print(f"   Run: python train.py  first")
        return

    print("Choose mode:")
    print("  1. Test all levels")
    print("  2. Novelty/diversity test")
    print("  3. Interactive")
    print("  4. All")

    choice = input("\nChoice (1-4): ").strip()
    if choice == "1":
        test_all_levels(model, tokenizer)
    elif choice == "2":
        test_novelty(model, tokenizer)
    elif choice == "3":
        interactive_mode(model, tokenizer)
    elif choice == "4":
        test_all_levels(model, tokenizer)
        test_novelty(model, tokenizer)
        interactive_mode(model, tokenizer)
    else:
        test_all_levels(model, tokenizer)


if __name__ == "__main__":
    main()

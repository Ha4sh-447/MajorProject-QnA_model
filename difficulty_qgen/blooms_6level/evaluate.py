#!/usr/bin/env python3
"""
Bloom's Taxonomy Model Evaluation — Standard NLP Metrics

Metrics computed:
1. BLEU (1-4 gram) — n-gram overlap with reference questions
2. ROUGE-L — longest common subsequence overlap
3. BERTScore — semantic similarity using BERT embeddings
4. Self-BLEU — diversity: lower = more diverse outputs
5. Tag Accuracy — does output contain correct [LEVEL_TAG]?
6. Level Distinctness — are outputs for different levels actually different?

Usage:
    python evaluate.py               # 4-level model
    python evaluate.py --model-dir ../../question_gen_blooms_6level_model --levels 6
"""

import argparse
import json
import random
import re
import sys
import warnings
from collections import defaultdict

import torch
import numpy as np

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

FOUR_LEVEL = {
    "model_dir": "../../question_gen_blooms_4level_model",
    "val_path": "../../dataset/blooms_4level_val.json",
    "base_model": "google/flan-t5-small",
    "levels": {
        1: ("Remember & Understand", "remember and understand"),
        2: ("Apply", "apply"),
        3: ("Analyze", "analyze"),
        4: ("Evaluate & Create", "evaluate and create"),
    },
    "tags": {
        1: "[REMEMBER & UNDERSTAND]",
        2: "[APPLY]",
        3: "[ANALYZE]",
        4: "[EVALUATE & CREATE]",
    },
}

SIX_LEVEL = {
    "model_dir": "../../question_gen_blooms_6level_model",
    "val_path": "../../dataset/blooms_6level_val.json",
    "base_model": "google/flan-t5-small",
    "levels": {
        1: ("Remember", "remember"),
        2: ("Understand", "understand"),
        3: ("Apply", "apply"),
        4: ("Analyze", "analyze"),
        5: ("Evaluate", "evaluate"),
        6: ("Create", "create"),
    },
    "tags": {
        1: "[REMEMBER]", 2: "[UNDERSTAND]", 3: "[APPLY]",
        4: "[ANALYZE]", 5: "[EVALUATE]", 6: "[CREATE]",
    },
}

SAMPLE_TEXTS = [
    "Mitosis is a type of cell division in which a single cell divides to produce two genetically identical daughter cells. The process consists of four stages: prophase, metaphase, anaphase, and telophase. During prophase, chromosomes condense and become visible. In metaphase, chromosomes align at the cell's equator.",
    "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another. The internal energy of a system changes when heat is added or work is done by the system.",
    "Elements in the periodic table are arranged by increasing atomic number. Elements in the same group have similar chemical properties because they have the same number of valence electrons.",
]


def load_model(config):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    base = AutoModelForSeq2SeqLM.from_pretrained(
        config["base_model"], torch_dtype=torch.float16
    ).to("cuda")
    model = PeftModel.from_pretrained(base, config["model_dir"])
    model.eval()
    print("   ✅ Loaded\n")
    return model, tokenizer


def generate(model, tokenizer, text, level_name, creative=True):
    prompt = f"generate a {level_name} level question: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    with torch.no_grad():
        if creative:
            out = model.generate(
                **inputs, max_length=384, do_sample=True, temperature=0.8,
                top_p=0.92, top_k=50, no_repeat_ngram_size=3, repetition_penalty=1.3,
            )
        else:
            out = model.generate(
                **inputs, max_length=384, num_beams=4, early_stopping=True,
                no_repeat_ngram_size=3,
            )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─── Metric functions ────────────────────────────────────────────────────────

def compute_bleu(references, hypotheses):
    """Compute BLEU-1 through BLEU-4."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1

    scores = {f"BLEU-{n}": [] for n in range(1, 5)}
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        if not hyp_tokens:
            for n in range(1, 5):
                scores[f"BLEU-{n}"].append(0)
            continue
        for n in range(1, 5):
            weights = tuple(1.0/n if i < n else 0 for i in range(4))
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smooth)
            scores[f"BLEU-{n}"].append(score)

    return {k: np.mean(v) for k, v in scores.items()}


def compute_rouge(references, hypotheses):
    """Compute ROUGE-L."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    for ref, hyp in zip(references, hypotheses):
        result = scorer.score(ref, hyp)
        scores.append(result['rougeL'].fmeasure)

    return {"ROUGE-L": np.mean(scores)}


def compute_bertscore(references, hypotheses):
    """Compute BERTScore."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False, device="cuda")
    return {
        "BERTScore-P": P.mean().item(),
        "BERTScore-R": R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }


def compute_self_bleu(outputs_per_level):
    """Compute Self-BLEU: average BLEU of each output against all others at the same level.
    Lower = more diverse."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1

    all_scores = []
    for level, outputs in outputs_per_level.items():
        if len(outputs) < 2:
            continue
        for i, hyp in enumerate(outputs):
            refs = [o.lower().split() for j, o in enumerate(outputs) if j != i]
            hyp_tokens = hyp.lower().split()
            if hyp_tokens and refs:
                score = sentence_bleu(refs, hyp_tokens, smoothing_function=smooth)
                all_scores.append(score)

    return {"Self-BLEU": np.mean(all_scores) if all_scores else 0}


def compute_tag_accuracy(outputs_per_level, tags):
    """Check if output contains the correct level tag."""
    correct, total = 0, 0
    per_level = {}
    for level, outputs in outputs_per_level.items():
        expected_tag = tags[level].upper()
        level_correct = sum(1 for o in outputs if expected_tag in o.upper())
        per_level[level] = f"{level_correct}/{len(outputs)}"
        correct += level_correct
        total += len(outputs)

    return {
        "Tag Accuracy": correct / total if total > 0 else 0,
        "Tag Per-Level": per_level,
    }


def compute_level_distinctness(outputs_per_level):
    """Measure how distinct outputs are across different levels.
    Uses pairwise string similarity — lower similarity = more distinct."""
    from difflib import SequenceMatcher

    levels = sorted(outputs_per_level.keys())
    if len(levels) < 2:
        return {"Level Distinctness": 1.0}

    similarities = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            for o1 in outputs_per_level[levels[i]][:5]:
                for o2 in outputs_per_level[levels[j]][:5]:
                    # Remove tags for comparison
                    c1 = re.sub(r'^\[.*?\]\s*', '', o1).lower()
                    c2 = re.sub(r'^\[.*?\]\s*', '', o2).lower()
                    sim = SequenceMatcher(None, c1, c2).ratio()
                    similarities.append(sim)

    avg_sim = np.mean(similarities) if similarities else 0
    return {"Level Distinctness": 1.0 - avg_sim}


# ─── Main evaluation pipeline ───────────────────────────────────────────────

def run_evaluation(config, n_val_samples=50, n_gen_per_level=10):
    model, tokenizer = load_model(config)

    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70 + "\n")

    # Load validation data
    print("📂 Loading validation data...")
    val_data = json.load(open(config["val_path"]))
    random.seed(42)
    random.shuffle(val_data)

    # 1. Reference-based metrics (BLEU, ROUGE, BERTScore)
    print(f"\n📊 Metric 1-3: Reference-based (on {n_val_samples} val samples)...\n")
    references = []
    hypotheses = []

    for ex in val_data[:n_val_samples]:
        ref = ex["target_text"]
        # Extract the level name from the input prompt
        match = re.search(r'generate (?:a |an )?(.+?) level question:', ex["input_text"])
        if not match:
            continue
        level_name = match.group(1)
        # Get the chunk text
        chunk = ex["input_text"].split(": ", 1)[-1] if ": " in ex["input_text"] else ex["input_text"]
        hyp = generate(model, tokenizer, chunk, level_name, creative=False)
        references.append(ref)
        hypotheses.append(hyp)

    if references:
        bleu = compute_bleu(references, hypotheses)
        rouge = compute_rouge(references, hypotheses)
        bertscore = compute_bertscore(references, hypotheses)

        print("  ┌──────────────────────────────────┐")
        print("  │     Reference-Based Metrics       │")
        print("  ├──────────────────────────────────┤")
        for k, v in {**bleu, **rouge, **bertscore}.items():
            print(f"  │  {k:20s}  {v:.4f}    │")
        print("  └──────────────────────────────────┘\n")

    # 2. Tag accuracy & Level distinctness
    print(f"📊 Metric 4-5: Tag Accuracy & Level Distinctness ({n_gen_per_level}/level)...\n")
    outputs_per_level = defaultdict(list)

    for level_id, (name, prompt_name) in config["levels"].items():
        for text in SAMPLE_TEXTS:
            for _ in range(n_gen_per_level // len(SAMPLE_TEXTS) + 1):
                out = generate(model, tokenizer, text, prompt_name, creative=True)
                outputs_per_level[level_id].append(out)
        outputs_per_level[level_id] = outputs_per_level[level_id][:n_gen_per_level]

    tag_acc = compute_tag_accuracy(outputs_per_level, config["tags"])
    distinctness = compute_level_distinctness(outputs_per_level)

    print("  ┌──────────────────────────────────┐")
    print("  │     Generation Quality Metrics    │")
    print("  ├──────────────────────────────────┤")
    print(f"  │  {'Tag Accuracy':20s}  {tag_acc['Tag Accuracy']:.4f}    │")
    print(f"  │  {'Level Distinctness':20s}  {distinctness['Level Distinctness']:.4f}    │")
    print("  └──────────────────────────────────┘\n")

    print("  Tag Accuracy per level:")
    for level_id, (name, _) in config["levels"].items():
        print(f"    L{level_id} ({name:25s}): {tag_acc['Tag Per-Level'].get(level_id, 'N/A')}")

    # 3. Self-BLEU (diversity)
    print(f"\n📊 Metric 6: Self-BLEU (Diversity)...\n")
    self_bleu = compute_self_bleu(outputs_per_level)
    print("  ┌──────────────────────────────────┐")
    print(f"  │  {'Self-BLEU':20s}  {self_bleu['Self-BLEU']:.4f}    │")
    print(f"  │  {'(lower = more diverse)':34s}│")
    print("  └──────────────────────────────────┘\n")

    # 4. Sample outputs
    print("=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70 + "\n")
    text = SAMPLE_TEXTS[0]
    for level_id, (name, prompt_name) in config["levels"].items():
        out = generate(model, tokenizer, text, prompt_name, creative=True)
        print(f"  L{level_id} ({name}): {out[:200]}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_metrics = {**bleu, **rouge, **bertscore, **tag_acc, **distinctness, **self_bleu}
    del all_metrics["Tag Per-Level"]
    for k, v in all_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bloom's Taxonomy QGen Model")
    parser.add_argument("--levels", type=int, default=4, choices=[4, 6], help="4-level or 6-level model")
    parser.add_argument("--n-val", type=int, default=50, help="Number of val samples for reference metrics")
    parser.add_argument("--n-gen", type=int, default=10, help="Generations per level for tag/diversity metrics")
    args = parser.parse_args()

    config = FOUR_LEVEL if args.levels == 4 else SIX_LEVEL

    # Check dependencies
    missing = []
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
    except ImportError:
        missing.append("nltk")
    try:
        import rouge_score
    except ImportError:
        missing.append("rouge-score")
    try:
        import bert_score
    except ImportError:
        missing.append("bert-score")

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        sys.exit(1)

    run_evaluation(config, n_val_samples=args.n_val, n_gen_per_level=args.n_gen)

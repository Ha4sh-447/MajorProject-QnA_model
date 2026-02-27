#!/usr/bin/env python3
"""
Bloom's Taxonomy Question Generation — Training Script

Fine-tunes flan-t5-small with LoRA.

Key findings from debugging:
- flan-t5-small is only 60M params → uses ~304 MB in FP16
- 8-bit quantization is UNNECESSARY and causes NaN gradients in batched training
- Solution: plain FP16 + LoRA, no quantization needed for 3.7GB VRAM
"""

import os
import gc
import shutil
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "google/flan-t5-small"
TRAIN_PATH = "../../dataset/blooms_dataset_train.json"
VAL_PATH = "../../dataset/blooms_dataset_val.json"
OUTPUT_DIR = "../../question_gen_blooms_model"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 384
BATCH_SIZE = 4             # Can afford larger batch without 8-bit overhead
GRADIENT_ACCUMULATION = 4  # Effective batch = 16
LEARNING_RATE = 5e-5
EPOCHS = 8
WARMUP_RATIO = 0.1
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Setup ───────────────────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()

print("=" * 70)
print("BLOOM'S TAXONOMY QUESTION GENERATION — TRAINING")
print("=" * 70)
print("Mode: FP32 + LoRA (model is only ~900 MB, no quantization/fp16 needed)\n")

# Clean old checkpoints to prevent resuming broken state
if os.path.exists(OUTPUT_DIR):
    print(f"🗑️  Removing old checkpoints from {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)
    print("   Done\n")

# ─── Tokenizer ───────────────────────────────────────────────────────────────

print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"   Vocab: {len(tokenizer)}\n")

# ─── Data Loading & Cleaning ────────────────────────────────────────────────

print("📂 Loading datasets...")
train_df = pd.read_json(TRAIN_PATH)
val_df = pd.read_json(VAL_PATH)

print(f"   Raw: {len(train_df)} train, {len(val_df)} val")

# Clean: drop empty, too short, or too long examples
train_df = train_df.dropna(subset=["input_text", "target_text"])
val_df = val_df.dropna(subset=["input_text", "target_text"])
train_df = train_df[train_df["target_text"].str.len() >= 15].reset_index(drop=True)
val_df = val_df[val_df["target_text"].str.len() >= 15].reset_index(drop=True)

# Pre-tokenize to filter out examples that are too long
# (avoids truncation artifacts during training)
print("   Filtering by token length...")
def filter_by_length(df, max_input, max_target):
    keep = []
    for _, row in df.iterrows():
        inp_len = len(tokenizer(row["input_text"])["input_ids"])
        tgt_len = len(tokenizer(row["target_text"])["input_ids"])
        if inp_len <= max_input and tgt_len <= max_target:
            keep.append(True)
        else:
            keep.append(False)
    return df[keep].reset_index(drop=True)

train_df = filter_by_length(train_df, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
val_df = filter_by_length(val_df, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

print(f"   Clean: {len(train_df)} train, {len(val_df)} val\n")

train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

# ─── Model + LoRA (NO quantization) ─────────────────────────────────────────

print("🤖 Loading model (FP32)...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
).to("cuda")
# NOTE: No gradient_checkpointing — model is tiny, not worth the overhead

print("🔧 Applying LoRA (r=32)...")
model = get_peft_model(model, LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q", "v", "k", "o"],   # More target modules for richer adaptation
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
))

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

vram = torch.cuda.memory_allocated() / 1024**2
print(f"   VRAM: {vram:.0f} MB\n")

if trainable == 0:
    raise ValueError("No trainable params!")

# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_function(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("🔄 Tokenizing...")
tokenized_train = train_data.map(preprocess_function, batched=True,
                                  remove_columns=train_data.column_names, desc="Train")
tokenized_val = val_data.map(preprocess_function, batched=True,
                              remove_columns=val_data.column_names, desc="Val")
print(f"   Done: {len(tokenized_train)} train, {len(tokenized_val)} val\n")

# ─── Training ────────────────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    fp16=False,                     # FP16 AMP produces NaN on this GPU — use pure FP32
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=25,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

steps_per_epoch = len(tokenized_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
print("=" * 70)
print("🚀 TRAINING")
print(f"   Model:  {MODEL_NAME} (FP16, no quantization)")
print(f"   LoRA:   r=32, targets=[q,v,k,o]")
print(f"   Batch:  {BATCH_SIZE}×{GRADIENT_ACCUMULATION}={BATCH_SIZE*GRADIENT_ACCUMULATION}")
print(f"   Steps:  ~{steps_per_epoch}/epoch, ~{steps_per_epoch*EPOCHS} total")
print(f"   LR:     {LEARNING_RATE}, warmup: {WARMUP_RATIO*100:.0f}%")
print("=" * 70 + "\n")

try:
    trainer.train()
    print("\n✅ TRAINING COMPLETE")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Saved to {OUTPUT_DIR}")

    # Sanity check
    print("\n🧪 Sanity test...")
    model.eval()
    test_text = "Photosynthesis converts light energy into chemical energy in chloroplasts. The light reactions occur in thylakoid membranes."
    for level, name in [(1, "remember"), (3, "apply"), (5, "evaluate"), (6, "create")]:
        prompt = f"generate a {name} level question: {test_text}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=MAX_TARGET_LENGTH,
                                  do_sample=True, temperature=0.8, top_p=0.92,
                                  no_repeat_ngram_size=3, repetition_penalty=1.3)
        print(f"  L{level} ({name}): {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — saving...")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   ✅ Saved to {OUTPUT_DIR}")
    except Exception:
        print("   ❌ Could not save")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    torch.cuda.empty_cache()
    gc.collect()

print("\n✅ Next: python test_model.py")

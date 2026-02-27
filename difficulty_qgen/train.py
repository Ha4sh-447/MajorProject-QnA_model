#!/usr/bin/env python3
"""
Training script for difficulty-conditioned question generation.
Fine-tunes flan-t5-small with LoRA + 8-bit quantization.

Uses the enhanced dataset from prepare_dataset.py which has diverse
prompt templates to encourage creative generation.

Hardware target: Nvidia 3050 4GB VRAM, 16GB RAM
"""

import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "google/flan-t5-small"
TRAIN_PATH = "../dataset/difficulty_dataset_train.json"
VAL_PATH = "../dataset/difficulty_dataset_val.json"
OUTPUT_DIR = "../question_gen_difficulty_model"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8   # Effective batch = 16
LEARNING_RATE = 3e-4
EPOCHS = 5
WARMUP_STEPS = 100
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Setup ───────────────────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()

print("=" * 70)
print("DIFFICULTY-CONDITIONED QUESTION GENERATION — TRAINING")
print("=" * 70 + "\n")

# ─── Data ────────────────────────────────────────────────────────────────────

print("📂 Loading datasets...")
train_df = pd.read_json(TRAIN_PATH)
val_df = pd.read_json(VAL_PATH)

train_df = train_df.dropna(subset=["input_text", "target_text"])
val_df = val_df.dropna(subset=["input_text", "target_text"])
train_df = train_df[train_df["target_text"].str.len() >= 10].reset_index(drop=True)
val_df = val_df[val_df["target_text"].str.len() >= 10].reset_index(drop=True)

print(f"   Train: {len(train_df)} | Val: {len(val_df)}\n")

train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

# ─── Tokenizer ───────────────────────────────────────────────────────────────

print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"   Vocab: {len(tokenizer)}\n")

# ─── Model + LoRA ────────────────────────────────────────────────────────────

print("🤖 Loading model (8-bit)...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

print("🔧 Applying LoRA...")
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
))

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)\n")

if trainable == 0:
    raise ValueError("No trainable params — LoRA failed!")

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

# ─── NaN-safe Trainer ────────────────────────────────────────────────────────

class SafeTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.max_nan = 10

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
            if torch.isnan(loss):
                self.nan_count += 1
                print(f"⚠️  NaN loss ({self.nan_count}/{self.max_nan})")
                if self.nan_count >= self.max_nan:
                    raise ValueError("Max NaN tolerance exceeded")
                return torch.tensor(0.0, device=model.device, requires_grad=True)
            self.nan_count = 0
            return loss
        except ValueError:
            raise

# ─── Training ────────────────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
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

trainer = SafeTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

steps_per_epoch = len(tokenized_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
print("=" * 70)
print("🚀 TRAINING")
print(f"   Model: {MODEL_NAME} | Epochs: {EPOCHS}")
print(f"   Batch: {BATCH_SIZE}×{GRADIENT_ACCUMULATION}={BATCH_SIZE*GRADIENT_ACCUMULATION}")
print(f"   Steps: ~{steps_per_epoch}/epoch, ~{steps_per_epoch*EPOCHS} total")
print("=" * 70 + "\n")

try:
    trainer.train()
    print("\n✅ TRAINING COMPLETE")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Model saved to {OUTPUT_DIR}")

    # Quick sanity check
    print("\n🧪 Sanity test...")
    model.eval()
    for level in [1, 4]:
        prompt = f"generate level {level} question: Photosynthesis converts light energy into chemical energy in chloroplasts."
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=MAX_TARGET_LENGTH, num_beams=4,
                                  early_stopping=True, no_repeat_ngram_size=3)
        print(f"  Level {level}: {tokenizer.decode(out[0], skip_special_tokens=True)[:150]}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — saving checkpoint...")
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

print("\n✅ Done. Next: python test_model.py")

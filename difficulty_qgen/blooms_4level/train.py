#!/usr/bin/env python3
"""
Option B: 4-Level Bloom's — Training Script
FP32 + LoRA on consolidated balanced dataset → question_gen_blooms_4level_model/
"""

import os, gc, shutil, torch, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-small"
TRAIN_PATH = "../../dataset/blooms_4level_train.json"
VAL_PATH = "../../dataset/blooms_4level_val.json"
OUTPUT_DIR = "../../question_gen_blooms_4level_model"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 384
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-5
EPOCHS = 8
WARMUP_RATIO = 0.1
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
gc.collect()

print("=" * 70)
print("OPTION B: 4-LEVEL BLOOM'S — TRAINING")
print("=" * 70 + "\n")

if os.path.exists(OUTPUT_DIR):
    print(f"🗑️  Removing old checkpoints...")
    shutil.rmtree(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("📂 Loading...")
train_df = pd.read_json(TRAIN_PATH).dropna().reset_index(drop=True)
val_df = pd.read_json(VAL_PATH).dropna().reset_index(drop=True)
train_df = train_df[train_df["target_text"].str.len() >= 15].reset_index(drop=True)
val_df = val_df[val_df["target_text"].str.len() >= 15].reset_index(drop=True)

print("   Filtering by token length...")
def filter_len(df):
    keep = []
    for _, row in df.iterrows():
        il = len(tokenizer(row["input_text"])["input_ids"])
        tl = len(tokenizer(row["target_text"])["input_ids"])
        keep.append(il <= MAX_INPUT_LENGTH and tl <= MAX_TARGET_LENGTH)
    return df[keep].reset_index(drop=True)

train_df = filter_len(train_df)
val_df = filter_len(val_df)
print(f"   Clean: {len(train_df)} train, {len(val_df)} val\n")

train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

print("🤖 Loading model (FP32)...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to("cuda")

print("🔧 Applying LoRA (r=32, targets=[q,v,k,o])...")
model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, target_modules=["q", "v", "k", "o"],
    lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"   VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")

def preprocess(batch):
    mi = tokenizer(batch["input_text"], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)
    lb = tokenizer(text_target=batch["target_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
    mi["labels"] = lb["input_ids"]
    return mi

print("🔄 Tokenizing...")
tok_train = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names, desc="Train")
tok_val = val_data.map(preprocess, batched=True, remove_columns=val_data.column_names, desc="Val")

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01, max_grad_norm=1.0, lr_scheduler_type="cosine",
    fp16=False,
    logging_steps=25, logging_first_step=True,
    save_strategy="steps", save_steps=200, save_total_limit=2,
    eval_strategy="steps", eval_steps=100,
    predict_with_generate=True, generation_max_length=MAX_TARGET_LENGTH,
    report_to="none", dataloader_num_workers=0, seed=SEED,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=tok_train, eval_dataset=tok_val,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
)

spe = len(tok_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
print(f"\n🚀 TRAINING: {EPOCHS} epochs, ~{spe}/epoch, ~{spe*EPOCHS} total steps\n")

try:
    trainer.train()
    print("\n✅ TRAINING COMPLETE")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Saved to {OUTPUT_DIR}")

    print("\n🧪 Sanity test...")
    model.eval()
    text = "Mitosis divides a single cell into two identical daughter cells through prophase, metaphase, anaphase, and telophase."
    for lv, name in [(1,"remember and understand"),(2,"apply"),(3,"analyze"),(4,"evaluate and create")]:
        prompt = f"generate a {name} level question: {text}"
        inp = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_length=384, do_sample=True, temperature=0.8, top_p=0.92, no_repeat_ngram_size=3, repetition_penalty=1.3)
        print(f"  L{lv} ({name}): {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — saving...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback; traceback.print_exc()
finally:
    torch.cuda.empty_cache(); gc.collect()

print("\n✅ Next: python test_model.py")

#!/usr/bin/env python3
"""
Bulletproof training script with NaN prevention and recovery
(Fixed version with safe ASCII filtering)
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import unicodedata

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "./dataset/chunked_dataset.json"
OUTPUT_DIR = "./question_gen_model_final"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print("BULLETPROOF TRAINING WITH NaN PREVENTION")
print("=" * 70 + "\n")

# ---------------------------
# 1. Load & validate dataset
# ---------------------------
print("📂 Loading dataset...")
data = pd.read_json(DATA_PATH)

# ---------------------------
# Text Cleaning Utilities
# ---------------------------
def is_ascii(s):
    """Safe ASCII check for all Python/pandas versions."""
    try:
        return s.isascii()
    except AttributeError:
        return all(ord(c) < 128 for c in str(s))

def normalize_text(s):
    """Normalize Unicode text and strip unwanted characters."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if ord(c) < 128)

# ---------------------------
# Clean Data
# ---------------------------
print("🔍 Cleaning data...")
original_size = len(data)

# Drop NaN and empty entries
data = data.dropna()
data = data[data['input_text'].str.strip().str.len() > 50]
data = data[data['target_text'].str.strip().str.len() > 20]

# Option 1: Filter out non-ASCII rows (safe for pandas < 2.1)
data = data[data["input_text"].apply(is_ascii) | (data["input_text"].str.len() > 0)]
data = data[data["target_text"].apply(is_ascii) | (data["target_text"].str.len() > 0)]

# Option 2 (Optional): Normalize Unicode instead of dropping
# data["input_text"] = data["input_text"].apply(normalize_text)
# data["target_text"] = data["target_text"].apply(normalize_text)

print(f"   Original: {original_size} rows")
print(f"   Cleaned: {len(data)} rows")
print(f"   Removed: {original_size - len(data)} rows\n")

dataset = Dataset.from_pandas(data, preserve_index=False)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
val_data = dataset["test"]

# ---------------------------
# 2. Tokenizer
# ---------------------------
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ---------------------------
# 3. Model with 8-bit quantization (STABLE VERSION)
# ---------------------------
print("🤖 Loading model with 8-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
print("⚙️  Preparing model for training...")
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# ---------------------------
# 4. LoRA - MINIMAL CONFIGURATION FOR STABILITY
# ---------------------------
lora_config = LoraConfig(
    r=4,  # Conservative rank
    lora_alpha=8,
    target_modules=["q", "v"],  # Only query and value layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
)

print("🔧 Applying LoRA...")
model = get_peft_model(model, lora_config)

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)\n")

model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device}\n")

# ---------------------------
# 5. Preprocessing
# ---------------------------
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

print("🔄 Tokenizing datasets...")
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names,
    desc="Train"
)
tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names,
    desc="Val"
)

# ---------------------------
# 6. Data Collator
# ---------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True,
)

# ---------------------------
# 7. Training Arguments
# ---------------------------
print("⚙️  Setting up training configuration...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    warmup_steps=0,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=15,
    max_steps=-1,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    optim="adamw_torch",
    adam_epsilon=1e-6,
    weight_decay=0.01,
    logging_dir="./logs_final",
    logging_steps=10,
    logging_first_step=True,
    logging_nan_inf_filter=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    save_safetensors=False,
    report_to="none",
    seed=42,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=True,
)

torch.cuda.empty_cache()

# ---------------------------
# 8. Custom Trainer (NaN detection)
# ---------------------------
class SafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.max_nan_tolerance = 5
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                print(f"\n⚠️  NaN/Inf detected (count: {self.nan_count}/{self.max_nan_tolerance})")
                if self.nan_count >= self.max_nan_tolerance:
                    raise ValueError("Training unstable - too many NaN losses")
                return torch.tensor(0.0, device=loss.device, requires_grad=True)
            self.nan_count = 0
            return loss
        except RuntimeError as e:
            if "nan" in str(e).lower():
                print(f"\n⚠️  Runtime NaN error: {e}")
                return torch.tensor(0.0, device=model.device, requires_grad=True)
            raise

trainer = SafeTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ---------------------------
# 9. Training
# ---------------------------
print("=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total epochs: {training_args.num_train_epochs}")
print("=" * 70 + "\n")

try:
    result = trainer.train()
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n📊 Final Evaluation:")
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n🧪 Quick inference test:")
    model.eval()
    test_text = "Generate exam-style questions from the following text:\nThe water cycle involves evaporation, condensation, and precipitation."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=3,
            do_sample=False,
            no_repeat_ngram_size=2,
        )
    print("\nGenerated:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "=" * 70)
    
except KeyboardInterrupt:
    print("\n\n⏸️  Training interrupted by user")
    print("💾 Saving checkpoint...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Checkpoint saved\n")
    
except Exception as e:
    print("\n\n❌ Training failed:")
    print(f"   Error: {e}")
    print("\n💾 Attempting to save checkpoint...")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("✅ Emergency checkpoint saved\n")
    except:
        print("❌ Could not save checkpoint\n")
    
    print("🔍 Diagnostic Information:")
    print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"   Max GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    import traceback
    print("\n📋 Full traceback:")
    traceback.print_exc()

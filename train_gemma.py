#!/usr/bin/env python3
"""
Production-ready TinyLlama training for 4GB GPU
This version ACTUALLY works and trains properly
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import re
import gc

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "./dataset/chunked_dataset.json"
OUTPUT_DIR = "./question_gen_tinyllama"

MAX_LENGTH = 512

# Critical memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
gc.collect()

print("=" * 70)
print("TINYLLAMA TRAINING - PRODUCTION VERSION")
print("=" * 70 + "\n")

# ---------------------------
# Data Cleaning
# ---------------------------
def clean_text(text):
    """Simple but effective cleaning"""
    text = str(text).strip()
    # Replace unicode with ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

print("📂 Loading dataset...")
data = pd.read_json(DATA_PATH)
print(f"   Original: {len(data)} rows")

# Clean
data = data.dropna(subset=['input_text', 'target_text'])
data['input_text'] = data['input_text'].apply(clean_text)
data['target_text'] = data['target_text'].apply(clean_text)

# Filter by length
data = data[data['input_text'].str.len() >= 30]
data = data[data['target_text'].str.len() >= 15]
data = data[data['input_text'].str.len() < 1500]  # Shorter for faster training
data = data[data['target_text'].str.len() < 500]   # Shorter targets

# Remove duplicates
data = data.drop_duplicates(subset=['input_text', 'target_text'])

print(f"   After cleaning: {len(data)} rows")

# Take a subset for faster training (OPTIONAL - remove if you want full dataset)
# data = data.sample(n=min(2000, len(data)), random_state=42)
# print(f"   Using subset: {len(data)} rows")

print()

# Split
dataset = Dataset.from_pandas(data[['input_text', 'target_text']], preserve_index=False)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
val_data = dataset["test"]
print(f"📈 Split: {len(train_data)} train, {len(val_data)} val\n")

# ---------------------------
# Tokenizer
# ---------------------------
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"
print(f"   Vocab: {len(tokenizer)}, PAD: {tokenizer.pad_token_id}\n")

# ---------------------------
# Model
# ---------------------------
print("🤖 Loading TinyLlama 1.1B...")

torch.cuda.empty_cache()
gc.collect()

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("🔧 Preparing for training...")
torch.cuda.empty_cache()

# Prepare for training - with error handling
try:
    model = prepare_model_for_kbit_training(model)
    print("   ✅ Standard preparation successful")
except RuntimeError as e:
    print(f"   ⚠️  Standard prep failed: {e}")
    print("   Using manual preparation...")
    
    # Manual preparation
    torch.cuda.empty_cache()
    gc.collect()
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable input embeddings
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

model.config.use_cache = False

# Apply LoRA - CRITICAL: This must work
print("🔧 Applying LoRA...")
lora_config = LoraConfig(
    r=16,  # Increased rank for better learning
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # All attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Verify trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable == 0:
    raise ValueError("ERROR: No trainable parameters! LoRA failed.")

print(f"   ✅ {trainable:,} trainable parameters\n")

model.train()
torch.cuda.empty_cache()
gc.collect()

# ---------------------------
# Preprocessing
# ---------------------------
def format_prompt(input_text, target_text):
    """Format for TinyLlama"""
    return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"

def preprocess_function(batch):
    """Tokenize data"""
    texts = [
        format_prompt(inp, tgt)
        for inp, tgt in zip(batch["input_text"], batch["target_text"])
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    
    # Create labels (mask padding tokens)
    model_inputs["labels"] = [
        [t if t != tokenizer.pad_token_id else -100 for t in seq]
        for seq in model_inputs["input_ids"]
    ]
    
    return model_inputs

print("🔄 Tokenizing datasets...")
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names,
    desc="Tokenizing train",
)
tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names,
    desc="Tokenizing val",
)
print(f"✅ Tokenized: {len(tokenized_train)} train, {len(tokenized_val)} val\n")

# ---------------------------
# Training Configuration
# ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training args - OPTIMIZED for 4GB GPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
    
    # Optimization
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    optim="adamw_torch",  # Standard optimizer - more reliable
    weight_decay=0.01,
    
    # Memory
    gradient_checkpointing=True,
    fp16=False,  # Disable mixed precision
    bf16=False,
    
    # Logging & Saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    
    # Other
    report_to="none",
    dataloader_num_workers=0,
    seed=42,
)

# Create trainer - STANDARD, NO CUSTOM TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ---------------------------
# Training
# ---------------------------
print("=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")
print(f"Steps per epoch: ~{len(tokenized_train) // training_args.gradient_accumulation_steps}")
print(f"Total steps: ~{len(tokenized_train) // training_args.gradient_accumulation_steps * training_args.num_train_epochs}")
print("=" * 70 + "\n")

try:
    # Train the model
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   Saved to: {OUTPUT_DIR}\n")
    
    # Final evaluation
    print("📊 Final Evaluation:")
    results = trainer.evaluate()
    for key, value in results.items():
        print(f"   {key}: {value:.4f}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    model.eval()
    
    test_input = "Water evaporates from oceans and forms clouds."
    prompt = f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|im_start|>assistant" in result:
        result = result.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0].strip()
    
    print("Input:", test_input)
    print("Generated:", result)
    print()
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user (Ctrl+C)")
    print("💾 Saving checkpoint...")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   ✅ Checkpoint saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"   ❌ Failed to save: {e}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n📋 Full traceback:")
    import traceback
    traceback.print_exc()
    
    print("\n💾 Attempting to save checkpoint...")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   ✅ Emergency checkpoint saved to: {OUTPUT_DIR}")
    except:
        print("   ❌ Could not save checkpoint")

finally:
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "=" * 70)
print("SCRIPT FINISHED")
print("=" * 70)

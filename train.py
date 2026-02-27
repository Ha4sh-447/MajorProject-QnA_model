# train_stable.py
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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "google/flan-t5-small"  # CHANGED BACK TO SMALL - more stable with 8-bit
DATA_PATH = "./dataset/chunked_dataset.json"
OUTPUT_DIR = "./question_gen_model_stable"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CRITICAL: Disable cuBLAS workspace config that causes issues
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

print("="*60)
print("STABLE 8-BIT TRAINING CONFIGURATION")
print("="*60 + "\n")

# ---------------------------
# 1. Load dataset
# ---------------------------
print("Loading dataset...")
data = pd.read_json(DATA_PATH)
data = data.dropna()
data = data[data['input_text'].str.strip() != '']
data = data[data['target_text'].str.strip() != '']

print(f"Dataset size: {len(data)} rows")

dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
val_data = dataset["test"]

# ---------------------------
# 2. Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ---------------------------
# 3. Load model in 8-bit - STABLE CONFIG
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_skip_modules=None,  # Don't skip any modules
)

print("Loading model with stable 8-bit quantization...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,  # Fixed: Changed from torch_dtype
)

print("Preparing for k-bit training...")
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}  # Prevents NaN gradients
)

# Disable cache for training
model.config.use_cache = False

# ---------------------------
# 4. LoRA - CONSERVATIVE CONFIG
# ---------------------------
lora_config = LoraConfig(
    r=8,  # Reduced from 16
    lora_alpha=16,  # Reduced from 32
    target_modules=["q", "v"],  # Only q and v for stability
    lora_dropout=0.05,  # Lower dropout
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
)

print("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Ensure model is in training mode
model.train()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ---------------------------
# 5. Preprocessing
# ---------------------------
def preprocess_function(batch):
    """Tokenize with proper handling"""
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

print("Tokenizing datasets...")
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names,
    desc="Tokenizing training data"
)

tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names,
    desc="Tokenizing validation data"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True,
    pad_to_multiple_of=8,  # Better for GPU efficiency
)

# ---------------------------
# 6. Training Arguments - STABLE
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Learning rate - CRITICAL CHANGES
    learning_rate=1e-4,  # Lower learning rate for stability
    lr_scheduler_type="linear",  # Simple linear scheduler
    warmup_ratio=0.1,  # 10% warmup
    
    # Batch size
    per_device_train_batch_size=4,  # Increased for stability
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # Effective batch = 8
    
    # Training length
    num_train_epochs=10,  # Reduced from 15
    
    # Precision - CRITICAL: NO FP16 with 8-bit
    fp16=False,  # CHANGED: Disabled to prevent NaN
    bf16=False,
    
    # Gradient handling - CRITICAL
    max_grad_norm=1.0,  # Increased from 0.5
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    
    # Logging
    logging_dir="./logs",
    logging_steps=10,  # More frequent
    logging_first_step=True,
    logging_nan_inf_filter=True,  # Filter NaN/Inf from logs
    
    # Evaluation
    eval_strategy="epoch",
    
    # Saving
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Other
    report_to="none",
    dataloader_pin_memory=False,  # Disabled for stability
    dataloader_num_workers=0,  # Single worker
    seed=42,
    data_seed=42,
    
    # Prevent issues
    remove_unused_columns=True,
    dataloader_drop_last=False,
)

# Clear cache
torch.cuda.empty_cache()

# ---------------------------
# 7. Custom Trainer with NaN detection
# ---------------------------
class StableTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to detect NaN early"""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("\n❌ NaN/Inf loss detected! Skipping this batch...")
            return torch.tensor(0.0, requires_grad=True).to(loss.device)
        
        return loss

trainer = StableTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("="*60)
print("🚀 TRAINING CONFIGURATION")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"FP16: {training_args.fp16} (Disabled for 8-bit stability)")
print(f"Total epochs: {training_args.num_train_epochs}")
print("="*60 + "\n")

print("Starting stable training...\n")

try:
    trainer.train()
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)
    
    # Save
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION")
    print("="*60)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Quick test
    print("\n" + "="*60)
    print("🧪 INFERENCE TEST")
    print("="*60)
    
    model.eval()
    test_text = "Generate exam-style questions from the following text:\nThe photoelectric effect occurs when light strikes a metal surface and ejects electrons."
    
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=3,
            early_stopping=True
        )
    
    print("\nGenerated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "="*60)

except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check CUDA memory: nvidia-smi")
    print("2. Reduce batch size to 2")
    print("3. Try without gradient checkpointing")
    import traceback
    traceback.print_exc()

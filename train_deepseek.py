import os
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth import UnslothVisionDataCollator
from transformers import AutoTokenizer, AutoModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------
# Paths to your CSV files
TRAIN_FILE = "dataset/train.csv"
VAL_FILE   = "dataset/val.csv"

# Base folder for images. 
# If your CSV says "90/121.jpg", this joins to "dataset/90/121.jpg"
IMAGE_ROOT = "dataset"  

# ✅ MATCHING YOUR CSV HEADERS:
COL_IMAGE = "filepath"  # Matches your "filepath" header
COL_TEXT  = "text"      # Matches your "text" header

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

# ------------------------------------------------------------------------
# 2. Download DeepSeek-OCR locally
# ------------------------------------------------------------------------
snapshot_download("unsloth/DeepSeek-OCR", local_dir="deepseek_ocr")

# ------------------------------------------------------------------------
# 3. Load model & tokenizer
# ------------------------------------------------------------------------
model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=True,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
    auto_model=AutoModel, 
)

# Use ChatML for DeepSeek
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# ------------------------------------------------------------------------
# 4. Load & Format Dataset
# ------------------------------------------------------------------------
dataset = load_dataset("csv", data_files={"train": TRAIN_FILE, "val": VAL_FILE})

def format_data(example):
    # Construct absolute image path
    img_path = os.path.join(IMAGE_ROOT, example[COL_IMAGE])
    
    # Create the conversation structure
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this image to text."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(example[COL_TEXT])}
            ]
        }
    ]
    
    return {
        "image": img_path,
        "conversations": conversation
    }

# Apply formatting
train_dataset = dataset["train"].map(format_data, remove_columns=dataset["train"].column_names)
val_dataset   = dataset["val"].map(format_data, remove_columns=dataset["val"].column_names)

# Debug: Print the first image path to verify it's correct
print(f"DEBUG: Checking first image path: {train_dataset[0]['image']}")

# ------------------------------------------------------------------------
# 5. Configure LoRA
# ------------------------------------------------------------------------
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_rslora=False,
)

# ------------------------------------------------------------------------
# 6. Trainer setup
# ------------------------------------------------------------------------
data_collator = UnslothVisionDataCollator(model, tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="", 
    max_seq_length=4096,
    dataset_num_proc=2,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="deepseek_ocr_finetuned",
        save_strategy="steps",
        save_steps=20,
        eval_strategy="steps",
        eval_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False,
    ),
)

# ------------------------------------------------------------------------
# 7. Start training
# ------------------------------------------------------------------------
print("Starting training...")
trainer.train()

# ------------------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------------------
print(f"Saving fine-tuned adapter...")
model.save_pretrained("deepseek_ocr_finetuned")
tokenizer.save_pretrained("deepseek_ocr_finetuned")
print("Done!")
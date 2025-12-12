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
TRAIN_FILE = "train.csv"   # Expecting your CSV file here
VAL_FILE   = "val.csv"     # Expecting your CSV file here
OUTPUT_DIR = "deepseek_ocr_finetuned"
IMAGE_ROOT = os.getcwd()   # Current directory (where 90/, 91/ folders are)

# CSV Column Names (Verify these match your CSV header!)
COL_IMAGE = "image_path"   # Column containing "99/121.jpg"
COL_TEXT  = "text"         # Column containing the OCR text

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

# ✅ FIX: Use "chatml" for DeepSeek, not "llava"
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# ------------------------------------------------------------------------
# 4. Load & Format Dataset (CSV Version)
# ------------------------------------------------------------------------
dataset = load_dataset("csv", data_files={"train": TRAIN_FILE, "val": VAL_FILE})

def format_data(example):
    # 1. Construct absolute image path
    # Combines current folder + image path from CSV (e.g. "99/121.jpg")
    img_path = os.path.join(IMAGE_ROOT, example[COL_IMAGE])
    
    # 2. Create the conversation structure
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
    dataset_text_field="", # Not used because we pre-formatted "conversations"
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
        output_dir=OUTPUT_DIR,
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
print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
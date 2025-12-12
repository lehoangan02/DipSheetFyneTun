import os
import torch
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModel
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------------
# 1. Paths & output
# ------------------------------------------------------------------------
TRAIN_FILE = "dataset/deepseek_train.json"
VAL_FILE   = "dataset/deepseek_val.json"
OUTPUT_DIR = "deepseek_ocr_finetuned"

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

# ------------------------------------------------------------------------
# 2. Download DeepSeek-OCR locally
# ------------------------------------------------------------------------
snapshot_download("unsloth/DeepSeek-OCR", local_dir="deepseek_ocr")

# ------------------------------------------------------------------------
# 3. Load model & tokenizer
# ------------------------------------------------------------------------
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/DeepSeek-OCR",
    load_in_4bit=True,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
    vision_tower_device_map="auto",
)

# REQUIRED – set LLaVA chat template
tokenizer = get_chat_template(tokenizer, chat_template="llava")

# ------------------------------------------------------------------------
# 4. Load & format dataset
# ------------------------------------------------------------------------
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "val": VAL_FILE})

def format_data(example):
    for turn in example["conversations"]:
        turn["content"] = turn["content"].replace("<image_placeholder>", "<image>")
    return example

train_dataset = dataset["train"].map(format_data)
val_dataset   = dataset["val"].map(format_data)

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
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    dataset_text_field=None,  # Required for vision inputs
    max_seq_length=4096,      # Supports long context
    dataset_num_proc=2,

    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,               # 60 steps is sufficient for a small dataset
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
    ),
)

# ------------------------------------------------------------------------
# 7. Start training
# ------------------------------------------------------------------------
print("Starting training...")
trainer.train()

# ------------------------------------------------------------------------
# 8. Save LoRA adapter
# ------------------------------------------------------------------------
print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! You can now run inference with your fine-tuned adapter.")

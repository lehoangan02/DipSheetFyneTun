import torch
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------
# Path to the JSON files you created
TRAIN_FILE = "dataset/deepseek_train.json"
VAL_FILE   = "dataset/deepseek_val.json"

# Output directory for the fine-tuned model
OUTPUT_DIR = "deepseek_ocr_finetuned"

# ------------------------------------------------------------------------
# 2. Load Model & Tokenizer
# ------------------------------------------------------------------------
# We load unsloth/DeepSeek-OCR. 
# load_in_4bit=True significantly reduces VRAM usage (recommended).
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/DeepSeek-OCR",
    load_in_4bit = True, 
    use_gradient_checkpointing = "unsloth", # optimize for long context
)

# ------------------------------------------------------------------------
# 3. Load and Format Dataset
# ------------------------------------------------------------------------
# Load your local JSON dataset
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "val": VAL_FILE})

# The model strictly expects the token "<image>" for vision inputs.
# Your preprocessing script used "<image_placeholder>", so we must fix it here.
def format_data(example):
    conversations = example["conversations"]
    # Replace the placeholder with the correct token
    for turn in conversations:
        turn["content"] = turn["content"].replace("<image_placeholder>", "<image>")
    return {"conversations": conversations}

train_dataset = dataset["train"].map(format_data)
val_dataset   = dataset["val"].map(format_data)

# ------------------------------------------------------------------------
# 4. Configure LoRA (Low-Rank Adaptation)
# ------------------------------------------------------------------------
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # Enable vision layer fine-tuning
    finetune_language_layers   = True, # Enable language layer fine-tuning
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,           # LoRA rank (16 is standard)
    lora_alpha = 16,  # LoRA alpha
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ------------------------------------------------------------------------
# 5. Training Setup
# ------------------------------------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text", # Dummy field, data collator handles LLaVA format
    max_seq_length = 2048,       # DeepSeek-OCR handles long context
    dataset_num_proc = 2,
    
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # The doc mentions 60 steps was enough for great results
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = OUTPUT_DIR,
        save_strategy = "steps",
        save_steps = 20,
        eval_strategy = "steps",
        eval_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)

# ------------------------------------------------------------------------
# 6. Start Training
# ------------------------------------------------------------------------
print("Starting training...")
trainer_stats = trainer.train()

# ------------------------------------------------------------------------
# 7. Save the Model
# ------------------------------------------------------------------------
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save merged 16-bit model (optional, for easier inference later)
# model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "merged_16bit")

print("Done! You can now run inference using the saved adapter.")
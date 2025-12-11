import os
# --- MEMORY OPTIMIZATION ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import types
import argparse
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms # <--- NEW: Manual Processing
from transformers import (
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# 1. HARDCODED FORWARD PATCH
# ---------------------------
def deepseek_vl_forward(
    self,
    input_ids=None,
    images=None,
    attention_mask=None,
    labels=None,
    **kwargs
):
    """
    Manually defined forward pass for DeepSeek VL.
    """
    vision_model = self.vision_model
    aligner = self.aligner
    language_model = self.language_model
    
    # 1. Process Images
    if images is not None and len(images.shape) > 1:
        # Check if we have batch dimension [B, N, C, H, W] or just [B, C, H, W]
        # DeepSeek often expects [B, N, C, H, W] where N=1 for single image
        if images.dim() == 4:
            images = images.unsqueeze(1) # Add 'N' dim

        if images.dim() == 5:
            b, n, c, h, w = images.shape
            # Flatten B*N for the vision encoder
            images = images.view(b * n, c, h, w)
        
        # Extract features
        v_out = vision_model(images)
        if hasattr(v_out, "last_hidden_state"):
            v_embeds = v_out.last_hidden_state
        else:
            v_embeds = v_out
        
        # Project to LLM space
        v_embeds = aligner(v_embeds) 
    else:
        v_embeds = None

    # 2. Process Text
    inputs_embeds = language_model.get_input_embeddings()(input_ids)

    # 3. Merge (Simplified Strategy)
    if hasattr(self, "prepare_inputs_embeds"):
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=images,
            images_seq_mask=kwargs.get("images_seq_mask"),
            images_spatial_crop=kwargs.get("images_spatial_crop")
        )
    
    # 4. Forward LLM
    return language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        **kwargs
    )

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = "deepseek-ai/deepseek-vl-1.3b-chat"
TRAIN_DATA = "dataset/deepseek_train.json"
VAL_DATA = "dataset/deepseek_val.json"
OUTPUT_DIR = "checkpoints/deepseek_ocr_lora"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Low memory mode")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    if "ipykernel" in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()

# ---------------------------
# DATASET (MANUAL TRANSFORMS)
# ---------------------------
class DeepSeekOCRDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- MANUAL IMAGE TRANSFORM (Replaces AutoImageProcessor) ---
        # Standard DeepSeek / CLIP Normalization
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)), # Fixed size for 1.3b model stability
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        # ------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_content = item['conversations'][0]['content']
        assistant_content = item['conversations'][1]['content']
        image_path = item['images'][0]
        
        # 1. Load & Transform Image
        try:
            image = Image.open(image_path).convert("RGB")
            # Apply Manual Transform
            pixel_values = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            pixel_values = torch.zeros((3, 384, 384))

        # 2. Format Prompt
        # Simplified manual formatting that works with tokenizer
        prompt = f"User: <image_placeholder> {user_content}\n\nAssistant: {assistant_content}<｜end of sentence｜>"

        # 3. Tokenize
        token_out = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = token_out.input_ids.squeeze(0)
        attention_mask = token_out.attention_mask.squeeze(0)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
            "labels": labels
        }

def collate_fn(batch):
    images = torch.stack([x['images'] for x in batch])
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "labels": labels
    }

# ---------------------------
# TRAINING LOOP
# ---------------------------
def train():
    args = parse_args()
    
    # 1. Load Tokenizer Only (We do images manually now)
    print(">>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if args.local:
        optimizer = "paged_adamw_8bit"
        use_bf16 = False
        use_fp16 = True
        lora_rank = 8
        lora_targets = ["q_proj", "v_proj"]
    else:
        optimizer = "adamw_torch"
        use_bf16 = supports_bf16
        use_fp16 = not supports_bf16
        lora_rank = 16
        lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(">>> Loading Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )

    # ---------------------------------------------------------
    # APPLY PATCHES
    # ---------------------------------------------------------
    print(">>> Applying Model Patches...")
    
    if not hasattr(model, "vision_model") and hasattr(model, "model"):
         print(">>> Adjusting model structure (unwrapping)...")
         model = model.model

    # Always inject custom forward to be safe against "unimplemented" errors
    print(">>> Injecting Custom DeepSeek Forward Method...")
    model.__class__.forward = deepseek_vl_forward

    # Patch Helper Methods
    model.__class__.get_input_embeddings = lambda self: self.language_model.get_input_embeddings()
    model.__class__.gradient_checkpointing_enable = lambda self, **kwargs: self.language_model.gradient_checkpointing_enable(**kwargs)
    model.__class__.gradient_checkpointing_disable = lambda self: self.language_model.gradient_checkpointing_disable()
    model.__class__.prepare_inputs_for_generation = lambda self, *args, **kwargs: self.language_model.prepare_inputs_for_generation(*args, **kwargs)
    # ---------------------------------------------------------

    print(">>> Preparing PEFT...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank*2,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Create Datasets (Passing Tokenizer Only)
    train_dataset = DeepSeekOCRDataset(TRAIN_DATA, tokenizer)
    val_dataset = DeepSeekOCRDataset(VAL_DATA, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim=optimizer,
        remove_unused_columns=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    print(">>> Starting Training...")
    trainer.train()

    print(">>> Saving...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
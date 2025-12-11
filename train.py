import os
# --- MEMORY OPTIMIZATION ENV VARS ---
# Helps prevent fragmentation on both Colab and Local
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import types
import argparse
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoProcessor
from deepseek_vl.utils.io import load_pil_images
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# CONFIGURATION & ARGS
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-VL for OCR")
    parser.add_argument("--local", action="store_true", help="Enable ultra-low memory mode for local 4GB GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    return parser.parse_args()

# Global Constants
MODEL_PATH = "deepseek-ai/deepseek-vl-1.3b-chat"
TRAIN_DATA = "dataset/deepseek_train.json"
VAL_DATA = "dataset/deepseek_val.json"
OUTPUT_DIR = "checkpoints/deepseek_ocr_lora"

# ---------------------------
# DATASET CLASS
# ---------------------------
class DeepSeekOCRDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        conversation = [
            {
                "role": "User",
                "content": item['conversations'][0]['content'],
                "images": item['images'] 
            },
            {
                "role": "Assistant",
                "content": item['conversations'][1]['content']
            }
        ]
        
        pil_images = load_pil_images(conversation)
        
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        
        return {
            "input_ids": prepare_inputs.input_ids.squeeze(0),
            "attention_mask": prepare_inputs.attention_mask.squeeze(0),
            "images": prepare_inputs.pixel_values.squeeze(0),
            "labels": prepare_inputs.input_ids.squeeze(0)
        }

# ---------------------------
# DATA COLLATOR
# ---------------------------
def collate_fn(batch):
    images = torch.stack([x['images'] for x in batch])
    input_ids = [x['input_ids'] for x in batch]
    labels = [x['labels'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "labels": labels
    }

# ---------------------------
# MAIN TRAINING LOOP
# ---------------------------
def train():
    args = parse_args()
    
    # --- AUTO-DETECT BF16 SUPPORT ---
    # Colab T4 GPUs = False. A100 GPUs = True.
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # --- PROFILE SETTINGS ---
    if args.local:
        print(">>> MODE: LOCAL (Low Memory Optimization)")
        lora_rank = 8
        lora_targets = ["q_proj", "v_proj"] 
        optimizer = "paged_adamw_8bit"      
        grad_accum = 8
        use_bf16 = False                    
        use_fp16 = True
        eval_strat = "no"                   
        num_workers = 0                     
    else:
        print(f">>> MODE: COLAB / SERVER (Performance) | BF16 Support: {supports_bf16}")
        lora_rank = 16
        # Target all linear layers for better accuracy
        lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        optimizer = "adamw_torch"
        grad_accum = 16
        
        # FIX: Automatically use correct precision based on hardware
        use_bf16 = supports_bf16            
        use_fp16 = not supports_bf16        
        
        eval_strat = "epoch"
        num_workers = 2

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(">>> Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(">>> Loading Model (4-bit)...")
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
    # MONKEY PATCH (Required for both modes)
    # ---------------------------------------------------------
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
    model.gradient_checkpointing_enable = types.MethodType(gradient_checkpointing_enable, model)
    model.gradient_checkpointing_disable = types.MethodType(gradient_checkpointing_disable, model)
    # ---------------------------------------------------------

    print(">>> Preparing Model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = DeepSeekOCRDataset(TRAIN_DATA, processor)
    val_dataset = DeepSeekOCRDataset(VAL_DATA, processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        eval_strategy=eval_strat,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=num_workers,
        gradient_checkpointing=True,
        optim=optimizer
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

    print(f">>> Saving Adapter to {OUTPUT_DIR}/final_adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
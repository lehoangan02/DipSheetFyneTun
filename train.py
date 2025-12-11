import os
import torch
import json
import types  # <--- Added this import
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig # <--- Added BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoProcessor
from deepseek_vl.utils.io import load_pil_images
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = "deepseek-ai/deepseek-vl-1.3b-chat" 
TRAIN_DATA = "dataset/deepseek_train.json"
VAL_DATA = "dataset/deepseek_val.json"
OUTPUT_DIR = "checkpoints/deepseek_ocr_lora"

# Hyperparameters
BATCH_SIZE = 1          
GRAD_ACCUMULATION = 16  
LEARNING_RATE = 2e-4
EPOCHS = 3

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
    print(">>> Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(">>> Loading Model (4-bit)...")
    
    # 1. Configuration for 4-bit loading (Fixes the deprecation warning)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config, # Use the new config object
        trust_remote_code=True,
        device_map="auto"
    )

    # ---------------------------------------------------------
    # ### FIX FOR PEFT ERROR: Monkey patch get_input_embeddings
    # ---------------------------------------------------------
    # The DeepSeek wrapper hides the language model, so we point 
    # the function to the internal language model's embedding layer.
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
    # ---------------------------------------------------------

    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = DeepSeekOCRDataset(TRAIN_DATA, processor)
    val_dataset = DeepSeekOCRDataset(VAL_DATA, processor)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True, 
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=args,
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
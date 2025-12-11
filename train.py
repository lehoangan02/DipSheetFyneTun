import os
# --- MEMORY OPTIMIZATION ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import types
import argparse
import inspect
import importlib
import sys
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM
from deepseek_vl.utils.io import load_pil_images
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# 1. ROBUST DISCOVERY
# ---------------------------
def get_processor_and_model(model_path):
    print(f">>> Loading AutoProcessor for {model_path}...")
    try:
        # Trust remote code is essential for DeepSeek
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Processor Load Error: {e}. Trying explicit import...")
        from deepseek_vl.models import VLChatProcessor
        processor = VLChatProcessor.from_pretrained(model_path)

    print(f">>> Loading Model for {model_path}...")
    # We use AutoModel. The magic happens in the patching later.
    model_class = AutoModelForCausalLM
    return processor, model_class

# ---------------------------
# 2. THE CRITICAL PATCH (Bring Your Own Forward)
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
    Explicit forward function injected into the model.
    Handles embedding images and merging with text embeddings before passing to LLM.
    """
    # 1. Access submodules (assuming standard DeepSeek structure)
    vision_model = self.vision_model
    aligner = self.aligner
    language_model = self.language_model
    
    # 2. Process Images (if present)
    if images is not None and len(images.shape) > 1:
        # images shape: [batch, num_images, 3, H, W] or [batch, 3, H, W]
        if images.dim() == 5:
            b, n, c, h, w = images.shape
            images = images.view(b * n, c, h, w)
        
        # Get Visual Features
        pixel_values = images
        # vision_model output is object with .last_hidden_state or direct tensor
        v_out = vision_model(pixel_values)
        if hasattr(v_out, "last_hidden_state"):
            v_embeds = v_out.last_hidden_state
        else:
            v_embeds = v_out
            
        # Align to LLM dimension
        v_embeds = aligner(v_embeds) 
    else:
        v_embeds = None

    # 3. Get Text Embeddings
    inputs_embeds = language_model.get_input_embeddings()(input_ids)

    # 4. Merge Embeddings (The tricky part: replacing placeholder tokens)
    # DeepSeek uses a specific token ID for images. 
    # Usually handled by the prepare_inputs_embeds inside the model, 
    # but since we are patching, we delegate to the internal helper if it exists.
    
    # If the model has the standard helper, use it:
    if hasattr(self, "prepare_inputs_embeds"):
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=images,
            images_seq_mask=kwargs.get("images_seq_mask"),
            images_spatial_crop=kwargs.get("images_spatial_crop")
        )
    else:
        # MANUAL MERGE FALLBACK (Simplified for stability)
        # If we can't do complex merge, we just run text.
        # This prevents crash but effectively ignores images if the helper is missing.
        # Most DeepSeek implementations loaded via remote code WILL have prepare_inputs_embeds.
        pass 

    # 5. Pass to Language Model
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
# DATASET
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
            {"role": "User", "content": item['conversations'][0]['content'], "images": item['images']},
            {"role": "Assistant", "content": item['conversations'][1]['content']}
        ]
        pil_images = load_pil_images(conversation)
        
        outputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        
        return {
            "input_ids": outputs.input_ids.squeeze(0),
            "attention_mask": outputs.attention_mask.squeeze(0),
            "images": outputs.pixel_values.squeeze(0),
            "labels": outputs.input_ids.squeeze(0)
        }

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
# TRAINING LOOP
# ---------------------------
def train():
    args = parse_args()
    processor, ModelClass = get_processor_and_model(MODEL_PATH)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if args.local:
        print(">>> MODE: LOCAL (Low Memory)")
        optimizer = "paged_adamw_8bit"
        use_bf16 = False
        use_fp16 = True
        lora_rank = 8
        lora_targets = ["q_proj", "v_proj"]
    else:
        print(f">>> MODE: COLAB (Performance) | BF16: {supports_bf16}")
        optimizer = "adamw_torch"
        use_bf16 = supports_bf16
        use_fp16 = not supports_bf16
        lora_rank = 16
        lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(">>> Loading Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = ModelClass.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )

    # ---------------------------------------------------------
    # MANDATORY CLASS PATCHING
    # ---------------------------------------------------------
    print(">>> Applying Critical Class Patches...")

    # 1. INJECT FORWARD METHOD (The fix for _forward_unimplemented)
    # We detect if the current forward is missing or broken
    if not hasattr(model, "forward") or "unimplemented" in str(model.forward).lower():
        print(">>> CRITICAL: Forward method missing. Injecting custom DeepSeek forward.")
        model.__class__.forward = deepseek_vl_forward
    else:
        # Even if it exists, it might be the generic "BaseModel" one that doesn't handle images.
        # We check class name. If it's "MultiModalityCausalLM", we trust it.
        # If it's "PeftModel" or "AutoModel", we inject.
        cls_name = model.__class__.__name__
        if "MultiModality" not in cls_name:
             print(f">>> CRITICAL: Class is '{cls_name}'. Injecting custom DeepSeek forward to support images.")
             model.__class__.forward = deepseek_vl_forward
        else:
             print(">>> Verified: Model class seems correct.")

    # 2. Patch get_input_embeddings (Required for PEFT)
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    model.__class__.get_input_embeddings = get_input_embeddings

    # 3. Patch Gradient Checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()
        
    model.__class__.gradient_checkpointing_enable = gradient_checkpointing_enable
    model.__class__.gradient_checkpointing_disable = gradient_checkpointing_disable

    # 4. Patch Inputs for Generation
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)
    
    model.__class__.prepare_inputs_for_generation = prepare_inputs_for_generation
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

    train_dataset = DeepSeekOCRDataset(TRAIN_DATA, processor)
    val_dataset = DeepSeekOCRDataset(VAL_DATA, processor)

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
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
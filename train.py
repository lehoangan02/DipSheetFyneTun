import os
# --- MEMORY OPTIMIZATION ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import types
import argparse
import inspect
import importlib
import pkgutil
import sys
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM
from deepseek_vl.utils.io import load_pil_images
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# 1. DYNAMIC DISCOVERY UTILS
# ---------------------------
def find_class_in_package(package_name, substring_match):
    """Scans a package to find a class name containing the substring."""
    print(f">>> Scanning {package_name} for class matching '{substring_match}'...")
    found_classes = []
    try:
        pkg = importlib.import_module(package_name)
        if hasattr(pkg, "__path__"):
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                try:
                    mod = importlib.import_module(name)
                    for cls_name, cls_obj in inspect.getmembers(mod):
                        if inspect.isclass(cls_obj) and substring_match in cls_name:
                            # --- FILTERS ---
                            if any(x in cls_name for x in ["PreTrained", "Config", "Output", "AutoModel", "BaseImage"]):
                                continue
                            found_classes.append(cls_obj)
                except Exception:
                    continue
    except Exception as e:
        print(f"    Scan error: {e}")
    
    if found_classes:
        # Prefer exact match or shortest relevant name
        best_match = sorted(list(set(found_classes)), key=lambda x: len(x.__name__), reverse=True)[0]
        print(f"    FOUND: {best_match.__name__}")
        return best_match
    return None

# ---------------------------
# 2. LOADERS
# ---------------------------
def load_components(model_path):
    # --- FIND PROCESSOR ---
    ProcessorClass = find_class_in_package("deepseek_vl.models", "VLChatProcessor")
    if not ProcessorClass:
        ProcessorClass = find_class_in_package("deepseek_vl.models", "Processor")
    
    if ProcessorClass:
        print(f">>> Using Processor: {ProcessorClass.__name__}")
        processor = ProcessorClass.from_pretrained(model_path, trust_remote_code=True)
    else:
        print(">>> WARNING: Official Processor not found. Using AutoProcessor.")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # --- FIND MODEL CLASS ---
    ModelClass = find_class_in_package("deepseek_vl.models", "MultiModality")
    
    if ModelClass is None:
        print(">>> CRITICAL WARNING: 'MultiModality' class not found. Trying 'ForCausalLM'.")
        ModelClass = find_class_in_package("deepseek_vl.models", "ForCausalLM")

    if ModelClass is None:
        print(">>> FALLBACK: Using AutoModelForCausalLM.")
        ModelClass = AutoModelForCausalLM
    else:
        print(f">>> Using Model Class: {ModelClass.__name__}")

    return processor, ModelClass

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
    processor, ModelClass = load_components(MODEL_PATH)

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
    # MANDATORY MONKEY PATCHES
    # ---------------------------------------------------------
    print(">>> Applying Mandatory Monkey Patches for DeepSeek VL...")

    # 1. EXPLICIT FORWARD BINDING (Fixes _forward_unimplemented)
    if hasattr(ModelClass, "forward"):
        print(f">>> Binding {ModelClass.__name__}.forward to model instance...")
        # We force the instance to use the CLASS method
        model.forward = types.MethodType(ModelClass.forward, model)
    else:
        print(">>> WARNING: ModelClass has no forward method. Trying fallback...")
    
    # 2. Patch get_input_embeddings (Required for PEFT)
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)

    # 3. Patch Gradient Checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()
        
    model.gradient_checkpointing_enable = types.MethodType(gradient_checkpointing_enable, model)
    model.gradient_checkpointing_disable = types.MethodType(gradient_checkpointing_disable, model)

    # 4. Patch Inputs for Generation
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)
    
    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)
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
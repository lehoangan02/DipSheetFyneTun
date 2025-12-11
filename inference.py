import sys
import torch
from transformers import AutoModelForCausalLM
# --- FIX: CORRECT IMPORT NAMES ---
from deepseek_vl.models import DeepseekVLProcessor, DeepseekVLForCausalLM
from deepseek_vl.utils.io import load_pil_images
from peft import PeftModel

# 1. Load Base Model
base_model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
processor = DeepseekVLProcessor.from_pretrained(base_model_path)
model = DeepseekVLForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Load Your Fine-Tuned Adapter
adapter_path = "checkpoints/deepseek_ocr_lora/final_adapter"
# Only load adapter if it exists (allows testing base model before training finishes)
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded adapter from {adapter_path}")
except Exception as e:
    print(f"Could not load adapter (maybe not trained yet?): {e}")

# 3. Inference
# Get image from command line arg, or default
image_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/images/test/default.jpg"

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Convert the text in the image to string.",
        "images": [image_path]
    },
    {"role": "Assistant", "content": ""}
]

pil_images = load_pil_images(conversation)
prepare_inputs = processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        prepare_inputs.input_ids,
        images=prepare_inputs.pixel_values,
        max_new_tokens=100,
        do_sample=False, 
        use_cache=True
    )

answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print("\n--- OUTPUT ---")
print(answer)
print("--------------")
from unsloth import FastVisionModel
from transformers import AutoModel
import os

# Load your fine-tuned model
model, tokenizer = FastVisionModel.from_pretrained(
    "deepseek_ocr_finetuned", # Point to your saved directory
    load_in_4bit = True,
    auto_model = AutoModel, # Required for custom architecture
)
FastVisionModel.for_inference(model)

# Define prompt
prompt = "<image>\nConvert the text in the image to string."
image_file = "dataset/images/test_data/some_test_image.jpg" # Pick a real image

# Run Inference
res = model.infer(
    tokenizer,
    prompt = prompt,
    image_file = image_file,
    base_size = 1024,
    image_size = 640,
    crop_mode = True,
    save_results = True
)

print("\nPrediction:", res[0])
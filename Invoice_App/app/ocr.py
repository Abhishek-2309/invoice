from PIL import Image
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
import torch

ocr_model_id = "nanonets/Nanonets-OCR-s"

ocr_model = AutoModelForImageTextToText.from_pretrained(
    ocr_model_id, torch_dtype="auto", device_map="auto"
).eval()
ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_model_id)
ocr_processor = AutoProcessor.from_pretrained(ocr_model_id)

def ocr_page_with_nanonets(image_path: str, max_new_tokens=15000) -> str:
    image = Image.open(image_path)
    prompt = "Extract the text... (same as your current one)"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": f"file://{image_path}"}, {"type": "text", "text": prompt}]}
    ]

    text = ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ocr_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(ocr_model.device)

    try:
        outputs = ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        markdown = ocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return markdown
    finally:
        torch.cuda.empty_cache()
        import gc; gc.collect()

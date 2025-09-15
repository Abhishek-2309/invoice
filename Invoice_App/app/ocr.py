from PIL import Image
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
import torch
import re

ocr_model_id = "nanonets/Nanonets-OCR-s"

ocr_model = AutoModelForImageTextToText.from_pretrained(
    ocr_model_id, torch_dtype="auto", device_map="auto"
).eval()

ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_model_id)
ocr_processor = AutoProcessor.from_pretrained(ocr_model_id)
ocr_model = torch.compile(ocr_model)

def strip_prompt_from_output(text: str) -> str:
    split_pattern = r"(?:^|\n)assistant\s*\n"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()

def ocr_page_with_nanonets(image_path: str, max_new_tokens=4000) -> str:
    image = Image.open(image_path)
    prompt2 = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": f"file://{image_path}"}, {"type": "text", "text": prompt2}]}
    ]
    text = ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ocr_processor(text=[text], images=[image], return_tensors="pt", padding=False).to(ocr_model.device)
    with torch.no_grad():
        outputs = ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    markdown = ocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    del inputs
    del outputs
    return strip_prompt_from_output(markdown)

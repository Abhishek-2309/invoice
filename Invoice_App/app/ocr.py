import os
import base64
import httpx
from PIL import Image
import re

OCR_VLLM_BASE_URL = os.getenv("OCR_VLLM_BASE_URL", "http://vllm-ocr:8002/v1")
OCR_VLLM_MODEL = os.getenv("OCR_VLLM_MODEL", "nanonets/Nanonets-OCR-s")
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "300"))

def _encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Return (mime, base64_str) for the given image file."""
    # Infer MIME from actual file
    try:
        with Image.open(image_path) as im:
            fmt = (im.format or "").lower()
    except Exception:
        fmt = ""
    # Fallback to extension
    ext = os.path.splitext(image_path)[1].lower()
    if fmt in {"png", "jpeg", "jpg", "webp", "tiff", "bmp"}:
        mime = "image/" + ("jpeg" if fmt == "jpg" else fmt)
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
        mime = "image/" + ("jpeg" if ext in {".jpg", ".jpeg"} else ext[1:])
    else:
        mime = "image/png"  # safe default

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64

def strip_prompt_from_output(text: str) -> str:
    split_pattern = r"(?:^|\n)assistant\s*\n"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()

def ocr_page_with_nanonets(image_path: str, max_new_tokens: int = 2000) -> str:
    """
    Calls the OCR vLLM server (nanonets/Nanonets-OCR-s) with a base64 image and returns markdown.
    """
    mime, img_b64 = _encode_image_to_base64(image_path)

    prompt = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in html format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; "
        "otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. "
        "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
        "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
    )

    payload = {
        "model": OCR_VLLM_MODEL,
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        # Disable any "thinking" mode if template supports it
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}

    r = httpx.post(
        f"{OCR_VLLM_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
        timeout=OCR_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return strip_prompt_from_output(text)



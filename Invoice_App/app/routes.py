import os
import tempfile
import uuid
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from typing import Dict, Callable

from app.ocr import ocr_page_with_nanonets
from app.Folder_Processing import process_zip
from app.llm_processing import (
    process_invoice_items,
    process_invoice_compact,
)

router = APIRouter()

UPLOAD_DIR = "uploads"
JSON_OUTPUT_DIR = os.path.join(UPLOAD_DIR, "json_results")
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
OCR_VLLM_BASE_URL = os.getenv("OCR_VLLM_BASE_URL", "http://vllm-ocr:8002/v1")


@router.get("/healthz")
async def healthz():
    """
    Checks both vLLM servers:
      - Text LLM (Qwen3) on VLLM_BASE_URL
      - OCR LLM (Nanonets-OCR-s) on OCR_VLLM_BASE_URL
    """
    async def ping(url: str) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(url.rstrip("/") + "/models")
            ok = r.status_code == 200
            return {"ok": ok, "models": r.json() if ok else None}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    text_llm = await ping(VLLM_BASE_URL)
    ocr_llm = await ping(OCR_VLLM_BASE_URL)

    ok = text_llm.get("ok", False) and ocr_llm.get("ok", False)
    return {"ok": ok, "text_llm": text_llm, "ocr_llm": ocr_llm}


def _img_to_tmp(img, tmpdir: str) -> str:
    p = os.path.join(tmpdir, f"{uuid.uuid4().hex}.png")
    img.save(p)
    return p


async def _process_single_file(file: UploadFile, process_fn: Callable[[str], Dict]):
    """
    Common single-file processing:
      - Image: OCR once → markdown → process_fn(markdown)
      - PDF: split to images, OCR page-wise → join markdown → process_fn(full_markdown)
    """
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            if suffix in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
                tmp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(await file.read())
                md = ocr_page_with_nanonets(tmp_path)
                return process_fn(md)

            elif suffix == ".pdf":
                tmp_pdf = os.path.join(tmpdir, f"{uuid.uuid4().hex}.pdf")
                with open(tmp_pdf, "wb") as f:
                    f.write(await file.read())
                images = convert_from_path(tmp_pdf, dpi=300)
                parts = []
                for img in images:
                    img_path = _img_to_tmp(img, tmpdir)
                    parts.append(ocr_page_with_nanonets(img_path))
                full_markdown = "\n".join(parts)
                return process_fn(full_markdown)

            else:
                raise HTTPException(status_code=400, detail="Only image/PDF supported")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")



@router.post("/upload_items")
async def upload_items(file: UploadFile = File(...)):
    """
    Uses the 'items' prompt/schema (Main_Table.items = list[dict]).
    """
    result = await _process_single_file(file, process_invoice_items)
    return {"mode": "items", "result": result}


@router.post("/upload_compact")
async def upload_compact(file: UploadFile = File(...)):
    """
    Uses the 'compact' prompt/schema (Main_Table.columns + rows).
    """
    result = await _process_single_file(file, process_invoice_compact)
    return {"mode": "compact", "result": result}


@router.post("/upload_zip")
async def upload_zip(
    file: UploadFile = File(...),
    mode: str = Query("items", regex="^(items|compact)$"),
) -> Dict[str, dict]:
    """
    Upload a .zip containing PDFs/images.
    mode = items | compact (defaults to 'items').
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")

    process_fn = process_invoice_items if mode == "items" else process_invoice_compact
    try:
        result = process_zip(file, JSON_OUTPUT_DIR, process_fn=process_fn)
        return {"mode": mode, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing failed: {e}")

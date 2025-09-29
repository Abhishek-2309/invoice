import os
import shutil
import tempfile
import uuid
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from pdf2image import convert_from_path
from fastapi.responses import JSONResponse
from app.llm_processing import process_invoice_dir
from app.ocr import ocr_page_with_nanonets
from app.Folder_Processing import process_zip
from typing import Dict

router = APIRouter()

UPLOAD_DIR = "uploads"
JSON_OUTPUT_DIR = os.path.join(UPLOAD_DIR, "json_results")

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

@router.get("/healthz")
async def healthz():
    base = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(base.replace("/v1", "/v1/models"))
        ok = r.status_code == 200
        return {"ok": ok, "vllm_models": r.json() if ok else None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            if suffix in [".png", ".jpg", ".jpeg"]:
                tmp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(await file.read())
                md = ocr_page_with_nanonets(tmp_path)
                return process_invoice_dir(md)

            elif suffix == ".pdf":
                tmp_pdf = os.path.join(tmpdir, f"{uuid.uuid4().hex}.pdf")
                with open(tmp_pdf, "wb") as f:
                    f.write(await file.read())
                images = convert_from_path(tmp_pdf)
                full_markdown = "\n".join(ocr_page_with_nanonets(_img_to_tmp(img, tmpdir)) for img in images)
                return process_invoice_dir(full_markdown)

            else:
                raise HTTPException(status_code=400, detail="Only image/PDF supported")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

def _img_to_tmp(img, tmpdir: str) -> str:
    p = os.path.join(tmpdir, f"{uuid.uuid4().hex}.png")
    img.save(p)
    return p

@router.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)) -> Dict[str, dict]:
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")
    try:
        result = process_zip(file, JSON_OUTPUT_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing failed: {e}")

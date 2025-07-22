import os
import shutil
import tempfile
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.llm_processing import process_invoice_dir
from app.ocr import ocr_page_with_nanonets
from app.Folder_Processing import process_zip
from typing import Dict

UPLOAD_DIR = "uploads"
JSON_OUTPUT_DIR = os.path.join(UPLOAD_DIR, "json_results")

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

router = APIRouter()

@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            full_markdown = ""
            if ext == ".pdf":
                images = convert_from_path(temp_path, dpi=300)
                for i, img in enumerate(images):
                    img_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.png")
                    img.save(img_path)
                    full_markdown += ocr_page_with_nanonets(img_path) + "\n"
            else:
                full_markdown = ocr_page_with_nanonets(temp_path)

            structured_json = process_invoice_dir(full_markdown)
            return structured_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@router.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)) -> Dict[str, dict]:
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")
    try:
        result = process_zip(file, JSON_OUTPUT_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing failed: {e}")

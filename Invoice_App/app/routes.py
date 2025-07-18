import os, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.llm_processing import process_invoice_dir
from app.ocr import ocr_page_with_nanonets
from app.Folder_Processing import process_zip


UPLOAD_DIR = "uploads"
JSON_OUTPUT_DIR = os.path.join(UPLOAD_DIR, "json_results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

router = APIRouter()


@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1].lower()
    temp_path = os.path.join(UPLOAD_DIR, f"temp{suffix}")
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        markdown = ocr_page_with_nanonets(temp_path)
        print('Router-markdown', markdown)
        structured_json = process_invoice_dir(markdown)
        print('Router-json', structured_json)
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



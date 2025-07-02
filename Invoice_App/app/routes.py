import os, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.ocr import ocr_page_with_nanonets
from app.llm_processing import process_invoice, llm

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()
    
@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1].lower()
    temp_path = os.path.join(UPLOAD_DIR, f"temp{suffix}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        markdown = ocr_page_with_nanonets(temp_path)
        structured_json = process_invoice(markdown, llm)
        return JSONResponse(content=structured_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

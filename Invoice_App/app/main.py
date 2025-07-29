import os
import unsloth
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI
from app.routes import router
from app.llm_engine import load_llm

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("Loading Qwen LLM...")
    load_llm()
    print("Qwen model loaded successfully.")

app.include_router(router)
@app.get("/")
def root():
    return {"status": "FastAPI is running!"}

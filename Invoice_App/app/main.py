from fastapi import FastAPI
from app.routes import router
from app.llm_processing import set_llm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch

app = FastAPI()
app.include_router(router)

@app.get("/")
def root():
    return {"status": "FastAPI is running!"}

@app.on_event("startup")
def load_llm():
    model_id = "Qwen/Qwen2.5-7B"
    print("[INFO] Loading LLM...")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096, return_full_text=False)
    llm = HuggingFacePipeline(pipeline=pipe)

    set_llm(llm)
    print("[INFO] LLM loaded and ready.")

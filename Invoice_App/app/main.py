import os
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Invoice")

@app.get("/")
def root():
    return {"status": "FastAPI is running!"}

app.include_router(router)

Invoice-Handler

This project provides a FastAPI-based service for automatically extracting structured data from invoices, with the help of Nanonets-ocr-s and a quantized large language model (LLM) - Qwen3:8B to parse invoice text and tables into a  JSON schema.

LLMs Used:

1) nanonets-ocr-s - First LLM used to get the OCR of an Image
2) Qwen3:8B  - used to get a JSON output.


The pipeline works as follows:

1) On startup of the application, both the models - Qwen3 and nanonets-ocr are loaded onto memory.
2) Once started up, there are 2 routes/endpoints available to the user to either upload a single PDF/Image or a zip file containing multiple such files.
3) If a zip file is sent, using the os.walk function, we walk through every file that has either image or pdf related extension and run the pipeline for those files alone.
4) For each file, we run the OCR LLM with a pre-specified prompt, once the output is obtained, we remove the unnecessary text present and give the required table htmls + text to the next parsing step
5) The OCR output is sent into a Qwen3-8B model, quantized via Unsloth, for fast inference on GPUs, as instantiated in llm_engine.py
6) The Output obtained previously, is passed through to this model, it gives the outputs for the fields based on a structured prompt in prompts.py, including keys for buyer/seller details, table items, summary details, payment details and others
7) Once an output is obtained, it is further parsed through to remove unnecessary details not part of the JSON structure and its schema is verified with pydantic models present in the schema.py file.
8) The final result is given as a structured JSON for either a single file or a directory of files.


Tech Stack

Core

FastAPI: High-performance API framework
Uvicorn/Gunicorn: ASGI server for deployment
Pydantic v2: Request/response validation

OCR

nanonets-ocr-s
pdf2image: PDF rendering
Pillow / OpenCV: Image handling

LLM Inference

Unsloth: For Qwen3-8B quantization and faster inference
Transformers >= 4.38.0: Model loading/inference
Torch >= 2.0.0: GPU/CPU execution
Accelerate: Efficient model distribution
Safetensors: Faster weight loading

Utilities

httpx: Async HTTP client
tenacity: Retry logic
uuid: Unique IDs for tracking/debugging

Project Structure

main.py - FastAPI entrypoint

routes.py - API endpoints 

schemas.py - Pydantic models for JSON responses

prompts.py - Invoice prompt 

ocr.py - nanonets-ocr-s extrction  

llm_engine.py - Model loading for Qwen LLM

llm_processing.py - Prompt builder + JSON postprocessing

Folder_Processing.py - Utility for bulk folder-based processing

requirements.txt - Dependencies

How It Works
Flow

Upload → Invoice uploaded as image or PDF.

OCR → Nanonets extracts text and HTML tables.

Prompting → Text is wrapped into a structured prompt.

Inference → Qwen3-8B generates JSON.

Validation → JSON parsed & validated via Pydantic.

Response → Clean structured JSON returned to client.




Deployment
Local (development)
uvicorn main:app --reload

Production (example with Gunicorn + Uvicorn workers)
gunicorn -k uvicorn.workers.UvicornWorker -w 1 main:app --bind 0.0.0.0:8000

Performance Considerations

GPU Required: Best performance on x86 with CUDA-enabled NVIDIA GPUs.

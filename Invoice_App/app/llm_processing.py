import re
import json
from typing import Dict
from app.schemas import InvoiceSchema
from app.prompts import kv_prompt
from app.llm_engine import chat

JSON_FENCE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
BRACE_GRAB = re.compile(r"(\{[\s\S]*\})")

def _extract_json_from_output(text: str) -> Dict:
    m = JSON_FENCE.search(text) or BRACE_GRAB.search(text)
    if not m:
        raise ValueError("Model returned no JSON.")
    raw = m.group(1)
    raw = re.sub(r"(?<!\\)\\(?![\\/\"bfnrtu])", r"\\\\", raw)
    return json.loads(raw)

def process_invoice_dir(full_markdown: str) -> Dict:
    user_prompt = kv_prompt.format(doc_body=full_markdown)
    messages = [
        {"role": "system", "content": "You convert invoice markdown into a single strict JSON object that matches the schema. Return only JSON."},
        {"role": "user", "content": user_prompt},
    ]
    model_out = chat(messages, temperature=0.0, max_tokens=2000)
    data = _extract_json_from_output(model_out)

    try:
        obj = InvoiceSchema.model_validate(data)
        return obj.model_dump(by_alias=True, exclude_none=True)
    except Exception:
        return data

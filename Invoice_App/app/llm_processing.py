import re
import json
from typing import Dict, Type
from pydantic import BaseModel

from app.llm_engine import chat
from app.prompts import kv_prompt_items, kv_prompt_compact
from app.schemas import InvoiceSchemaItems, InvoiceSchemaCompact

JSON_FENCE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
BRACE_GRAB = re.compile(r"(\{[\s\S]*\})")

def _extract_json_from_output(text: str) -> Dict:
    m = JSON_FENCE.search(text) or BRACE_GRAB.search(text)
    if not m:
        raise ValueError("Model returned no JSON.")
    raw = m.group(1)
    raw = re.sub(r"(?<!\\)\\(?![\\/\"bfnrtu])", r"\\\\", raw)
    return json.loads(raw)

def _run_llm(full_markdown: str, prompt_template: str, schema_cls: Type[BaseModel]) -> Dict:
    user_prompt = prompt_template.format(doc_body=full_markdown)
    messages = [
        {"role": "system", "content": "You convert invoice markdown into a single strict JSON object that matches the schema. Return only JSON."},
        {"role": "user", "content": user_prompt},
    ]
    model_out = chat(messages, temperature=0.0, max_tokens=6000)
    data = _extract_json_from_output(model_out)

    # Validate to ensure shape conformity
    obj = schema_cls.model_validate(data)
    return obj.model_dump(by_alias=True, exclude_none=True)

# Public helpers the routes can import:
def process_invoice_items(full_markdown: str) -> Dict:
    """Verbose: Main_Table.items = list[dict]"""
    return _run_llm(full_markdown, kv_prompt_items, InvoiceSchemaItems)

def process_invoice_compact(full_markdown: str) -> Dict:
    """Compact: Main_Table.columns + rows"""
    return _run_llm(full_markdown, kv_prompt_compact, InvoiceSchemaCompact)

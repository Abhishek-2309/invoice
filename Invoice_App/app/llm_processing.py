import re
import json
from typing import Dict
from bs4 import BeautifulSoup
from app.llm_engine import load_llm
from app.prompts import kv2_prompt

def extract_json_from_output(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        json_str = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error: {e}")
    raise ValueError("No valid JSON object found in LLM output.")

def process_invoice_dir(markdown: str):
    model, tokenizer = load_llm()
    return process_invoice(markdown, tokenizer, model)

def process_invoice(markdown_html: str, tokenizer, model) -> dict:
    filled_prompt = kv2_prompt.replace("{doc_body}", markdown_html)
    messages = [{"role": "user", "content": filled_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=4096,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.1
    )
    output_ids = generated_ids[0][len(input_ids[0]):]
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(full_output)
    fields_json = extract_json_from_output(full_output)
    return fields_json

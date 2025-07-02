# app/llm_processing.py

from bs4 import BeautifulSoup
from app.schemas import TableResult, KVResult, InvoiceSchema
from app.prompts import identify_prompt, kv_prompt
import json, re, torch, gc

_llm = None  # Private LLM holder

def set_llm(llm_instance):
    global _llm
    _llm = llm_instance

def get_llm():
    if _llm is None:
        raise RuntimeError("LLM not initialized. Call set_llm on startup.")
    return _llm

def extract_json_from_output(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No valid JSON found")

def extract_tables(html: str):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    table_str = "\n\n".join(f"[Table {i}]\n{str(t)}" for i, t in enumerate(tables))
    return tables, table_str, soup

def process_invoice_dir(markdown_html: str):
    return process_invoice(markdown_html, get_llm())

def process_invoice(markdown_html: str, llm) -> dict:
    tables, table_str, soup = extract_tables(markdown_html)
    identify_chain = identify_prompt | llm

    try:
        raw_table = identify_chain.invoke({"tables": table_str})
        parsed_table = extract_json_from_output(raw_table)
        table_result = TableResult(**parsed_table)
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    main_idx = table_result.main_table_index
    tables[main_idx].extract()
    remaining_html = str(soup)

    kv_chain = kv_prompt | llm
    try:
        raw_kv = kv_chain.invoke({"doc_body": remaining_html})
        parsed_kv = extract_json_from_output(raw_kv)
        kv_result = KVResult(**parsed_kv)
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    invoice_data = InvoiceSchema(
        Header=kv_result.Header,
        Items=table_result.items,
        Payment_Terms=kv_result.Payment_Terms,
        Summary=kv_result.Summary,
        Other_Important_Sections=kv_result.Other_Important_Sections,
    )
    return invoice_data.model_dump()

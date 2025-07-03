from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bs4 import BeautifulSoup
from app.schemas import TableResult, KVResult, InvoiceSchema
from app.prompts import identify_prompt, kv_prompt
import json, re, torch, gc

# Model setup
model_id = "Qwen/Qwen2.5-7B"
llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=4096, return_full_text=False)


def extract_json_from_output(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON object found in LLM output.")


def extract_tables(html: str):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    table_str = "\n\n".join(f"[Table {i}]\n{str(t)}" for i, t in enumerate(tables))
    print(table_str)
    return tables, table_str, soup


def process_invoice(markdown_html: str, llm) -> dict:
    tables, table_str, soup = extract_tables(markdown_html)

    if not tables:
        raise ValueError("No <table> elements found in the document.")
    
    full_identify_prompt = identify_prompt.format(tables=table_str)
    print(full_identify_prompt)

    try:
        raw_table = llm(full_identify_prompt, do_sample=False)[0]["generated_text"]
        print("RAW TABLE:", raw_table)
        parsed_table = extract_json_from_output(raw_table)
        print("PARSED TABLE:", parsed_table)
        table_result = TableResult(**parsed_table)
    except Exception as e:
        raise ValueError(f"Failed to parse main table JSON output: {e}") from e


    # Remove the main table from the document before feeding to kv_prompt
    main_idx = table_result.main_table_index
    try:
        tables[main_idx].extract()
    except IndexError:
        raise ValueError(f"main_table_index {main_idx} out of range. Only found {len(tables)} tables.")
    remaining_html = str(soup)

    full_kv_prompt = kv_prompt.format(doc_body=remaining_html)
    print(full_kv_prompt)

    try:
        raw_kv = llm(full_kv_prompt, do_sample=False)[0]["generated_text"]
        print("RAW KV:", raw_kv)
        parsed_kv = extract_json_from_output(raw_kv)
        print("PARSED KV:", parsed_kv)
        kv_result = KVResult(**parsed_kv)
    except Exception as e:
        raise ValueError(f"Failed to parse KV JSON output: {e}") from e


    invoice_data = InvoiceSchema(
        Header=kv_result.Header,
        Items=table_result.items,
        Payment_Terms=kv_result.Payment_Terms,
        Summary=kv_result.Summary,
        Other_Important_Sections=kv_result.Other_Important_Sections,
    )
    print("INVOICE DATA:", invoice_data)
    return invoice_data.model_dump()

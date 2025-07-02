from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from bs4 import BeautifulSoup
from app.schemas import TableResult, KVResult, InvoiceSchema
from app.prompts import identify_prompt, kv_prompt
import json, re, torch, gc

model_id = "Qwen/Qwen2.5-7B"

llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

hf_pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=4096, return_full_text=False)
llm = HuggingFacePipeline(pipeline=hf_pipe)

def extract_json_from_output(text: str) -> dict:
    match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\\{.*\\})", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No valid JSON")

def extract_tables(html: str):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    table_str = "\\n\\n".join(f"[Table {i}]\\n{str(t)}" for i, t in enumerate(tables))
    return tables, table_str, soup

def process_invoice(markdown_html: str, llm) -> dict:
    tables, table_str, soup = extract_tables(markdown_html)
    identify_chain = identify_prompt | llm

    try:
        raw_table = identify_chain.invoke({"tables": table_str})
        parsed_table = extract_json_from_output(raw_table)
        table_result = TableResult(**parsed_table)
    except Exception as e:
        raise ValueError(f"Failed to parse main table JSON output: {e}") from e
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
    except Exception as e:
        raise ValueError(f"Failed to parse KV JSON output: {e}") from e
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

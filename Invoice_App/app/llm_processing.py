from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bs4 import BeautifulSoup
from app.schemas import TableResult, KVResult, InvoiceSchema
from app.prompts import identify_prompt, kv_prompt
import json, re, torch, gc
from typing import Any

def process_invoice_dir(markdown):
    # Model setup
    model_id = "Qwen/Qwen2.5-7B"
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=4096, return_full_text=False)
    return process_invoice(markdown, llm)
    
def flatten_html_table_no_repeat(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    for wm in table.find_all("watermark"):
        wm.replace_with(f"[{wm.get_text(strip=True)}]")

    table_matrix = []
    rowspan_track = {}

    rows = table.find_all("tr")
    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        current_row = []
        col_idx = 0
        while col_idx < 100:
            if (row_idx, col_idx) in rowspan_track:
                remaining, value = rowspan_track[(row_idx, col_idx)]
                current_row.append("")
                if remaining > 1:
                    rowspan_track[(row_idx + 1, col_idx)] = (remaining - 1, value)
                del rowspan_track[(row_idx, col_idx)]
                col_idx += 1
                continue

            if not cells:
                break

            cell = cells.pop(0)
            text = cell.get_text(strip=True).replace("\n", " ")
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for i in range(colspan):
                current_row.append(text)

            if rowspan > 1:
                for i in range(colspan):
                    rowspan_track[(row_idx + 1, col_idx + i)] = (rowspan - 1, text)

            col_idx += colspan

        table_matrix.append(current_row)

    max_cols = max(len(row) for row in table_matrix)
    padded = [row + [""] * (max_cols - len(row)) for row in table_matrix]

    def fmt(row): return "| " + " | ".join(cell if cell else " " for cell in row) + " |"
    sep = "| " + " | ".join(["---"] * max_cols) + " |"

    return "\n".join([fmt(padded[0]), sep] + [fmt(r) for r in padded[1:]])

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


def process_invoice(markdown_html: str, llm: Any) -> dict:
    tables, _, soup = extract_tables(markdown_html)

    if not tables:
        raise ValueError("No <table> elements found in the document.")

    # Step 1: Flatten all tables to markdown and create identify prompt
    table_markdowns = []
    for i, table in enumerate(tables):
        try:
            flattened = flatten_html_table_no_repeat(str(table))
            table_markdowns.append(f"[Table {i}]\n{flattened}")
        except Exception as e:
            table_markdowns.append(f"[Table {i}] (error flattening): {e}")
    table_str = "\n\n".join(table_markdowns)

    full_identify_prompt = identify_prompt.format(tables=table_str)
    print(full_identify_prompt)

    # Step 2: Run table identification prompt
    try:
        raw_table = llm(full_identify_prompt, do_sample=False)[0]["generated_text"]
        print("RAW TABLE:", raw_table)
        parsed_table = extract_json_from_output(raw_table)
        print("PARSED TABLE:", parsed_table)
        table_result = TableResult(**parsed_table)
    except Exception as e:
        raise ValueError(f"Failed to parse main table JSON output: {e}") from e

    # Step 3: Replace main table with Markdown + markers, don't remove
    main_idx = table_result.main_table_index
    try:
        main_table_md = flatten_html_table_no_repeat(str(tables[main_idx]))
        pre_tag = soup.new_tag("pre")
        pre_tag.string = f"[Main Table Start]\n{main_table_md}\n[Main Table End]"
        tables[main_idx].replace_with(pre_tag)
    except IndexError:
        raise ValueError(f"main_table_index {main_idx} out of range. Only found {len(tables)} tables.")

    # Step 4: Continue with KV extraction on full modified HTML
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

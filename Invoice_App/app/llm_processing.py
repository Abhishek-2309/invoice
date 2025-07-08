from bs4 import BeautifulSoup
import json
import re
import os
import csv
import numpy as np
import pandas as pd
import torch
import spacy
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from app.schemas import KVResult, InvoiceSchema
from app.prompts import kv_prompt, kv2_prompt
from app.ocr import ocr_model, ocr_processor
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load spaCy model once
nlp = spacy.load("en_core_web_md")

INVOICE_HEADER_KEYWORDS = [
    "item", "product", "description", "details", "part number", "sku", "goods", "service", "article", "line item",
    "quantity", "qty", "unit", "units", "uom", "nos", "pcs", "pieces", "kg", "litre", "liter",
    "rate", "unit price", "price", "cost", "mrp", "list price", "selling price",
    "total", "total amount", "net amount", "gross amount", "amount", "value", "line total",
    "discount", "discount%", "rebate", "adjustment", "charges", "other charges",
    "hsn", "sac", "code", "hsn code", "sac code",
    "serial", "no", "sr. no", "sl no", "line no", "remarks", "batch no", "expiry date"
]

def process_invoice_dir(markdown: str):
    model_id = "Qwen/Qwen2.5-3b"  # Instruction-tuned version
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,  # or bfloat16 if using Ampere+
        trust_remote_code=True
    )
    return process_invoice(markdown, model, tokenizer)


def extract_json_from_output(text: str) -> dict:
    print(text)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON object found in LLM output.")


def flatten_html_table_smart_span(html: str):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    for wm in table.find_all("watermark"):
        wm.replace_with(f"[{wm.get_text(strip=True)}]")

    table_matrix = []
    cell_map = {}
    rows = table.find_all("tr")
    max_cols = 0

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        current_row = []
        col_idx = 0

        while col_idx < 100:
            while (row_idx, col_idx) in cell_map:
                current_row.append("")
                col_idx += 1

            if not cells:
                break

            cell = cells.pop(0)
            text = cell.get_text(strip=True).replace("\n", " ")
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            current_row.append(text)

            for i in range(1, colspan):
                current_row.append("")

            for i in range(1, rowspan):
                for j in range(colspan):
                    cell_map[(row_idx + i, col_idx + j)] = text

            col_idx += colspan

        table_matrix.append(current_row)
        max_cols = max(max_cols, len(current_row))

    for row in table_matrix:
        while len(row) < max_cols:
            row.append("")

    return table_matrix


def convert_html_to_csv(html: str, output_csv_path: str) -> str:
    matrix = flatten_html_table_smart_span(html)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)
    return output_csv_path


def extract_and_merge_thead_headers_with_span(html: str):
    from collections import defaultdict

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    thead = table.find("thead") if table else None

    if not thead:
        return [], 0

    rows = thead.find_all("tr")
    header_matrix = []
    col_occupancy = defaultdict(int)
    total_cols = 0

    for r_idx, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cur_row = []
        col_idx = 0

        while len(cur_row) < total_cols or len(cur_row) < len(cells):
            if col_occupancy[(r_idx, col_idx)]:
                cur_row.append("")
                col_idx += 1
            else:
                break

        for cell in cells:
            text = cell.get_text(strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for i in range(colspan):
                cur_row.append(text)

            for i in range(1, rowspan):
                for j in range(colspan):
                    col_occupancy[(r_idx + i, col_idx + j)] = 1

            col_idx += colspan

        total_cols = max(total_cols, len(cur_row))
        header_matrix.append(cur_row)

    for row in header_matrix:
        while len(row) < total_cols:
            row.append("")

    final_headers = []
    for col_cells in zip(*header_matrix):
        merged = " ".join(cell for cell in col_cells if cell.strip())
        final_headers.append(merged.strip())

    return final_headers, len(header_matrix)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

REFERENCE_HEADER = ["product", "description", "qty", "unit price", "amount", "tax"]
def score_header_similarity(headers: list[str]) -> int:
    headers = [h.lower() for h in headers]

    # Define weighted keyword groups
    product_keywords = {"item", "product", "description", "details", "part number", "sku", "goods", "service", "article", "line item"}
    quantity_keywords = {"quantity", "qty", "unit", "units", "uom", "nos", "pcs", "pieces", "kg", "litre", "liter"}
    price_keywords = {"rate", "unit price", "price", "cost", "mrp", "list price", "selling price"}
    amount_keywords = {"total", "total amount", "net amount", "gross amount", "amount", "value", "line total"}
    discount_keywords = {"discount", "discount%", "rebate", "adjustment", "charges", "other charges"}
    tax_keywords = {"hsn", "sac", "code", "hsn code", "sac code"}
    misc_keywords = {"serial", "no", "sr. no", "sl no", "line no", "remarks", "batch no", "expiry date"}

    # Penalize if any of these are found
    payment_only_keywords = {"voucher", "payment", "received", "mode", "reference", "receipt"}

    score = 0

    for header in headers:
        if any(kw in header for kw in product_keywords):
            score += 5
        if any(kw in header for kw in quantity_keywords):
            score += 4
        if any(kw in header for kw in price_keywords):
            score += 4
        if any(kw in header for kw in amount_keywords):
            score += 2
        if any(kw in header for kw in discount_keywords):
            score += 1
        if any(kw in header for kw in tax_keywords):
            score += 3
        if any(kw in header for kw in misc_keywords):
            score += 1
        if any(kw in header for kw in payment_only_keywords):
            score -= 4  # Penalize voucher/payment tables

    # Extra boost if multiple major groups are matched
    groups_matched = sum([
        any(kw in h for h in headers for kw in g)
        for g in [product_keywords, quantity_keywords, price_keywords, amount_keywords]
    ])
    if groups_matched >= 2:
        score += 6
    elif groups_matched == 1:
        score += 2

    return score

def score_with_spacy(headers: list[str]) -> float:
    header_vecs = [nlp(h).vector for h in headers if h.strip()]
    ref_vecs = [nlp(ref).vector for ref in REFERENCE_HEADER]

    if not header_vecs:
        return 0.0

    similarities = []
    for hv in header_vecs:
        row_sim = [cosine_similarity([hv], [rv])[0][0] for rv in ref_vecs]
        similarities.append(max(row_sim))  # take max sim to any ref

    return float(np.mean(similarities))

def table_csv_to_dicts(csv_path: str, headers: list[str], skiprows=0) -> list[dict]:
    df = pd.read_csv(csv_path, header=None, skiprows=skiprows)
    df.fillna(" ", inplace=True)

    expected_cols = len(headers)
    data_dicts = []

    for _, row in df.iterrows():
        row_values = row.tolist()
        non_empty_cells = [str(cell).strip() for cell in row_values if str(cell).strip()]
        if len(non_empty_cells) < max(2, expected_cols // 2):
            continue
        if len(non_empty_cells) <= 2 and expected_cols > 4:
            continue
        row_values = row_values[:expected_cols] + [""] * max(0, expected_cols - len(row_values))
        row_dict = {header: str(row_values[i]).strip() for i, header in enumerate(headers)}
        data_dicts.append(row_dict)

    return data_dicts


def detect_summary_rows(
    rows: list[dict],
    summary_keywords=None,
    summary_region_ratio: float = 0.3,
    min_filled_threshold: float = 0.4,
):
    if summary_keywords is None:
        summary_keywords = [
            "total", "subtotal", "tax", "vat", "amount",
            "grand", "net payable", "payable", "balance"
        ]

    item_rows = []
    summary_rows = []
    total_rows = len(rows)

    def is_summary_row(row: dict, idx: int) -> bool:
        is_bottom_section = idx >= int((1 - summary_region_ratio) * total_rows)

        row_text = " ".join(str(v).lower() for v in row.values())
        keyword_match = any(kw in row_text for kw in summary_keywords)

        total_cells = len(row)
        non_empty_cells = sum(1 for v in row.values() if str(v).strip())
        fill_ratio = non_empty_cells / total_cells if total_cells > 0 else 0

        return is_bottom_section and (keyword_match or fill_ratio < min_filled_threshold)

    for idx, row in enumerate(rows):
        if is_summary_row(row, idx):
            summary_rows.append(row)
        else:
            item_rows.append(row)

    return item_rows, summary_rows


def extract_best_table_and_headers(html_tables: list[str]) -> tuple[str, list[str], int]:
    best_score = -1
    best_table = None
    best_headers = []
    best_header_row_index = 0

    for html_table in html_tables:
        soup = BeautifulSoup(html_table, "html.parser")
        table = soup.find("table")
        if not table:
            continue

        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            candidate_headers = [cell.get_text(strip=True) for cell in cells]

            # Skip if too few cells to be a header
            if len(candidate_headers) < 2:
                continue

            keyword_score = score_header_similarity(candidate_headers)
            semantic_score = score_with_spacy(candidate_headers)
            combined_score = 0.6 * keyword_score + 0.4 * semantic_score
            if combined_score > best_score:
                best_score = combined_score
                best_table = html_table
                best_headers = candidate_headers
                best_header_row_index = i

    return best_table, best_headers, best_header_row_index # +1 to skip header

def strip_prompt_from_output(text: str) -> str:
    split_pattern = r"(?:^|\n)assistant\s*\n"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()  # fallback: return everything

def extract_invoice_kv_fields(markdown: str, prompt, max_new_tokens = 4096) -> dict:
    filled_prompt = prompt.replace("{doc_body}", markdown)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": filled_prompt}
    ]

    prompt_text = ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ocr_processor(text=prompt_text, return_tensors="pt").to(ocr_model.device)

    outputs = ocr_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = ocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    markdown_res = strip_prompt_from_output(result)
    return extract_json_from_output(markdown_res)

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def process_invoice(markdown_html: str, model: Any, tokenizer:Any) -> dict:
    soup = BeautifulSoup(markdown_html, "html.parser")
    html_tables = [str(tbl) for tbl in soup.find_all("table")]

    best_table = None
    best_headers = []
    best_header_rows = 0

    best_table, best_headers, best_header_rows = extract_best_table_and_headers(html_tables)

    if not best_table:
        raise ValueError("No invoice-like table found.")

    csv_path = "main_invoice_table.csv"
    convert_html_to_csv(best_table, csv_path)
    rows = table_csv_to_dicts(csv_path, best_headers, skiprows=best_header_rows)
    item_rows, summary_rows = detect_summary_rows(rows)

    
    for table_tag in soup.find_all("table"):
        if str(table_tag) == best_table:
            # Prepare summary rows as plain text
            summary_text = []
            for row in summary_rows:
                # row is a dict
                words = [str(v).strip() for v in row.values()]
                summary_text.append(" ".join(words).strip())
    
            # Create NavigableString to inject in place of main table
            summary_string = soup.new_string("\n".join(summary_text))
    
            # Insert text *after* the table, then remove table
            table_tag.insert_after(summary_string)
            table_tag.decompose()
    
        else:
            continue
            """
            # Convert other tables to text, even if some cells are empty
            plain_text = []
            for row in table_tag.find_all("tr"):
                row_cells = []
                for cell in row.find_all(["td", "th"]):
                    text = cell.get_text(strip=True)
                    colspan = int(cell.get("colspan", 1))
                    row_cells.extend([text] + [""] * (colspan - 1))
                plain_text.append(",".join(row_cells))

            replacement_string = soup.new_string("\n".join(plain_text))
            table_tag.replace_with(replacement_string)
            """

    print(str(soup))
    kv_data = extract_invoice_kv_fields(str(soup), kv_prompt)
    flat_data = flatten_dict(kv_data)
    formatted = "\n".join(f"- {k}: {v}" for k, v in flat_data.items())    
    filled_prompt = kv2_prompt.replace("{doc_body}", formatted)
    inputs = tokenizer(filled_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return raw_kv
    """
    raw_kv = llm(full_kv_prompt, do_sample=False)[0]["generated_text"]
    print(raw_kv)
    parsed_kv = extract_json_from_output(raw_kv)
    return parsed_kv
    
    
    kv_result = KVResult(**kv_data)
    
    return InvoiceSchema(
        Header=kv_result.Header,
        Items=item_rows,
        Payment_Terms=kv_result.Payment_Terms,
        Summary=kv_result.Summary,
        Other_Important_Sections=kv_result.Other_Important_Sections,
    ).model_dump()
    """
    

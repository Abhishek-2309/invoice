from bs4 import BeautifulSoup
import json
import re
import os
import csv
import pandas as pd
import torch
import spacy
from typing import Any
from app.schemas import KVResult, InvoiceSchema
from app.prompts import kv_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Canonical invoice headers
INVOICE_HEADER_KEYWORDS = [
    "item", "description", "product", "hsn", "code", "quantity", "qty", "rate",
    "unit price", "amount", "total", "value", "tax", "price", "serial", "no", "mrp"
]

def process_invoice_dir(markdown):
    # Model setup
    model_id = "Qwen/Qwen2.5-7B"
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=4096, return_full_text=False)
    return process_invoice(markdown, llm)


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


def score_header_similarity(headers: list[str]) -> float:
    if not headers:
        return 0.0
    invoice_keywords = [nlp(k.lower()) for k in INVOICE_HEADER_KEYWORDS]
    score = 0
    count = 0
    for h in headers:
        h_doc = nlp(h.lower())
        best_sim = max((h_doc.similarity(k) for k in invoice_keywords), default=0)
        score += best_sim
        count += 1
    return score / count if count else 0


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


def detect_summary_rows(rows: list[dict], summary_keywords=None, min_keywords=1):
    if summary_keywords is None:
        summary_keywords = ["total", "subtotal", "tax", "vat", "gst", "amount", "grand", "net payable"]

    item_rows = []
    summary_rows = []

    def row_contains_keyword(row: dict):
        row_text = " ".join(str(v).lower() for v in row.values())
        hits = [kw for kw in summary_keywords if kw in row_text]
        return len(hits) >= min_keywords

    for row in rows:
        if row_contains_keyword(row):
            summary_rows.append(row)
        else:
            item_rows.append(row)

    return item_rows, summary_rows


def process_invoice(markdown_html: str, llm: Any) -> dict:
    soup = BeautifulSoup(markdown_html, "html.parser")
    html_tables = [str(tbl) for tbl in soup.find_all("table")]

    best_score = -1
    best_table = None
    best_headers = []
    best_header_rows = 0

    for html_table in html_tables:
        headers, header_row_count = extract_and_merge_thead_headers_with_span(html_table)
        score = score_header_similarity(headers)
        if score > best_score:
            best_score = score
            best_table = html_table
            best_headers = headers
            best_header_rows = header_row_count

    if not best_table:
        raise ValueError("No invoice-like table found.")

    csv_path = "main_invoice_table.csv"
    convert_html_to_csv(best_table, csv_path)
    rows = table_csv_to_dicts(csv_path, best_headers, skiprows=best_header_rows)
    item_rows, summary_rows = detect_summary_rows(rows)

    for table_tag in soup.find_all("table"):
        table_tag.decompose()

    print(str(soup))
    # Use LLM for KV metadata
    full_kv_prompt = kv_prompt.format(doc_body=str(soup))
    raw_kv = llm(full_kv_prompt, do_sample=False)[0]["generated_text"]
    parsed_kv = extract_json_from_output(raw_kv)
    kv_result = KVResult(**parsed_kv)

    return InvoiceSchema(
        Header=kv_result.Header,
        Items=item_rows,
        Payment_Terms=kv_result.Payment_Terms,
        Summary=kv_result.Summary,
        Other_Important_Sections=kv_result.Other_Important_Sections,
    ).model_dump()

from langchain_core.prompts import PromptTemplate

identify_prompt = PromptTemplate(
    input_variables=["tables"],
    template="""
You are an expert table reader from HTML.

You are given multiple HTML tables extracted from an invoice. Read every table and its corresponding HTML code carefully.

Identify the table containing line items (such as product/service, quantity, price, etc.).

Your task is to extract:
1. `main_table_index`: index of the main line item table
2. `items`: each row of item data from that table (e.g., product, quantity, rate, etc.)
3. `summary_row`: footer/summary row data like total, tax, etc.

Return **ONLY JSON** inside triple backticks like this:
- Do NOT use markdown formatting or YAML
- Do NOT include any explanation, comments, or notes
- Your entire output must be a valid JSON object like:
```json
{{
  "main_table_index": 0,
  "items": [
    {{ "Sr. No.": "1", "Description": "Soap", "Rate": "50.00" }}
  ],
  "summary_row": {{
    "Total": "1000.00",
    "Tax": "180.00"
  }}
}}
```

Tables:
{tables}
"""
)


kv_prompt = PromptTemplate(
    input_variables=["doc_body"],
    template="""
You are given the body of an invoice (in HTML/Markdown) excluding the main line item table.

Extract all invoice metadata as structured key-value pairs. An example JSON schema is as follows:
~~~json
{{
  "Header": {{
    "Unique Invoice Number": "...",
    "Invoice Date": "...",
    "Seller's Information": {{
      "Company Name": "...",
      "Address": "...",
      "Contact": "...",
      "GSTIN": "..."
    }},
    "Buyer's Information": {{
      "Company Name": "...",
      "Address": "...",
      "Contact": "...",
      "GSTIN": "..."
    }}
  }},
  "Payment Terms": {{
    "Bank_details": {{
      "Bank Name": "...",
      "IFSC_code": "...",
      "bank_account_no": "..."
    }},
    "Payment Due Date": "...",
    "Payment Methods": "..."
  }},
  "Summary": {{
    "Subtotal": "...",
    "Taxes": "...",
    "Discounts": "...",
    "Total Amount Due": "..."
  }},
  "Other Important Sections": {{
    "Terms and conditions": "...",
    "Notes/Comments": "...",
    "Signature": "..."
  }}
}}
~~~

Make sure all values are strings and do not contain newlines.
Extract only from document contents, If relevant values are not found, leave empty.
Return **ONLY JSON** inside triple backticks like this:
```json
{{
  "Header": ...,
  ...
}}
Text:
{doc_body}
"""
)

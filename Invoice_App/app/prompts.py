from langchain_core.prompts import PromptTemplate

identify_prompt = PromptTemplate(
    input_variables=["tables"],
    template="""
You are an expert table reader from HTML. You are given multiple HTML tables extracted from an invoice.
Read every table and its corresponding html code, and find the main table along with the corresponding values in each row
Identify the table containing line items (such as product/service, quantity, price, etc.).
Items should only contain the itemized rows. For other rows that are towards the footer of table and don't fit items, include in summary_row
Return strictly a JSON output with these fields alone in this format:
```
{{
  "main_table_index": <index>,
  "items": [
    {{ "<column1>": "value", ... }},
    ...
  ],
  "summary_row": {{
    "<total_field1>": "value"
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
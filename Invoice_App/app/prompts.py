identify_prompt = """
You are an expert table reader. Read one or multiple of the following tables extracted from an invoice and given in markdown string format.
Each table is given in a row by row format with table cells separated by '|'. An example is as follows:

| SI No. | Description of Goods | Quantity | Rate | per | Amount |
| 1 | Soyabean Oil 14.6kg Tin | 25 | 680.00 | tin | 17000.00 |
. 
.

If only a single table is given, asssume it to be the main table
Else, Read every table and its corresponding data, and find the main table along with the corresponding values in each row
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


kv_prompt = """
You are given the body of an invoice (in Markdown) 
[Main Table Start] and [Main Table End] mark the start and end of the main line table
Read the entire markdown body and extract all the following details except the line items in the main line table.
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

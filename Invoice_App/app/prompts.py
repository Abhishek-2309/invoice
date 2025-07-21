kv2_prompt = """
You are given the markdown of an Invoice document with tables enclosed within '<table>' tags.
Extract and map the following fields into JSON:
- Invoice Number
- Invoice Date
- Seller's Information
- Buyer's Information
- Main Table (items and summary_row)
- Payment Terms
- Summary (Subtotal, Taxes, Total Amount Due)
- Other Important Sections

Return a JSON object with these fields. If a field is not present, leave it empty.
Markdown:
{doc_body}
"""

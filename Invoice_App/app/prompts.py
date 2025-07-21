kv2_prompt = """
You are given the markdown of an Invoice document with tables enclosed within '<table>' tags.
Extract and map the following fields alone into a strict JSON format:
- Invoice Number
- Invoice Date
- Seller's Info (containing Company_Name, Address, Contact_Details, GSTIN)
- Buyer's Information (containing Buyer's Company Name, Address, Contact_Details, GSTIN)
- Main Table (containing all the rows of the main table containing the line items of the invoice)
- Payment Terms (containing Bank Details such as Bank Name, Bank IFSC Code, Bank Account No, and other payment details such as Payment due date and payment methods)
- Summary (Subtotal, Taxes, Discounts,  Total Amount Due)
- Other Important Sections

Return a JSON object with these fields. If a field is not present, leave it empty.
Markdown:
{doc_body}
"""

kv2_prompt = """
You are given the markdown of an Invoice document with tables enclosed within '<table>' tags.
Extract and map the following fields alone into a strict JSON format:
- Invoice_Number
- Invoice_Date
- Seller's_Info (containing Company_Name, Address, Contact_Details, GSTIN)
- Buyer's_Info (containing Buyer's Company_Name(can even be Buyer name), Address, Contact_Details, GSTIN)
- Main_Table (Refers to the main table containing all the line items of the invoice, It should only contain only those rows of the main table having line items, do NOT include Totals and others inside.)
- Payment_Terms (containing Bank_Details(Bank_Name, Bank_IFSC_Code, Bank_Account_No), and other payment details such as payment_due_date and payment_methods)
- Summary (containing Subtotal(refers to Total before taxes), Taxes, Discounts, Total_Amount_Due(refers to the Final amount to be paid))
- Other_Important_Sections(containing Terms_and_Conditions, Notes/Comments, Signature)

Return a JSON object with these fields. If a field is not present, leave it empty.
Markdown:
{doc_body}
"""

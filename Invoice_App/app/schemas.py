from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional


class PartyInfo(BaseModel):
    model_config = ConfigDict(extra='allow')
    Company_Name: Optional[str] = Field(None, alias="Company Name")
    Address: Optional[str] = None
    Contact: Optional[str] = None
    GSTIN: Optional[str] = None


class HeaderSection(BaseModel):
    model_config = ConfigDict(extra='allow')
    Unique_Invoice_Number: Optional[str] = Field(None, alias="Unique Invoice Number")
    Invoice_Date: Optional[str] = Field(None, alias="Invoice Date")
    Seller_Info: Optional[PartyInfo] = Field(None, alias="Seller's Information")
    Buyer_Info: Optional[PartyInfo] = Field(None, alias="Buyer's Information")


class PaymentTerms(BaseModel):
    model_config = ConfigDict(extra='allow')
    Bank_details: Optional[Dict[str, str]] = None
    Payment_Due_Date: Optional[str] = Field(None, alias="Payment Due Date")
    Payment_Methods: Optional[str] = Field(None, alias="Payment Methods")


class SummarySection(BaseModel):
    model_config = ConfigDict(extra='allow')
    Subtotal: Optional[str] = None
    Taxes: Optional[str] = None
    Discounts: Optional[str] = None
    Total_Amount_Due: Optional[str] = Field(None, alias="Total Amount Due")


class OtherImportantSections(BaseModel):
    model_config = ConfigDict(extra='allow')
    Terms_and_conditions: Optional[str] = Field(None, alias="Terms and conditions")
    Notes_or_Comments: Optional[str] = Field(None, alias="Notes/Comments")
    Signature: Optional[str] = None


class TableResult(BaseModel):
    main_table_index: int
    items: List[Dict[str, Any]]
    summary_row: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra='allow')


class KVResult(BaseModel):
    Header: HeaderSection
    Payment_Terms: PaymentTerms = Field(..., alias="Payment Terms")
    Summary: SummarySection
    Other_Important_Sections: OtherImportantSections = Field(..., alias="Other Important Sections")
    model_config = ConfigDict(extra='allow')


class InvoiceSchema(BaseModel):
    model_config = ConfigDict(extra='allow')
    Header: Optional[HeaderSection]
    Items: Optional[List[Dict[str, Any]]]
    Payment_Terms: Optional[PaymentTerms] = Field(None, alias="Payment Terms")
    Summary: Optional[SummarySection]
    Other_Important_Sections: Optional[OtherImportantSections] = Field(None, alias="Other Important Sections")

"""LangChain tools for OpenFDA data search."""
from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import pandas as pd

from biodsa.tools.openfda.complete_response_letters import (
    search_complete_response_letters,
    LetterType,
    ApprovalStatus,
    SortOrder as CRLSortOrder,
)
from biodsa.tools.openfda.drugs_at_fda import (
    search_drugs_at_fda,
    MarketingStatus,
    SubmissionStatus,
    SubmissionType,
    SortOrder as DrugsFDASortOrder,
)
from biodsa.tools.openfda.product_labeling import (
    search_drug_labels,
    ProductType,
    RouteType,
    SortOrder as LabelSortOrder,
)
from biodsa.sandbox.sandbox_interface import ExecutionSandboxWrapper

__all__ = [
    "SearchCompleteResponseLettersTool",
    "SearchDrugsAtFDATool",
    "SearchDrugLabelsTool",
    "SearchCompleteResponseLettersToolInput",
    "SearchDrugsAtFDAToolInput",
    "SearchDrugLabelsToolInput",
]


# =====================================================
# Tool 1: Search Complete Response Letters
# =====================================================
class SearchCompleteResponseLettersToolInput(BaseModel):
    """Input schema for SearchCompleteResponseLettersTool."""
    
    search: Optional[str] = Field(
        default=None,
        description="General search term to query across all fields"
    )
    application_number: Optional[str] = Field(
        default=None,
        description="Filter by specific application number. Formats: 'NDA 123456', 'BLA 761215', 'BL 103324'. Space between prefix and number is optional."
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Filter by company name"
    )
    letter_type: Optional[str] = Field(
        default=None,
        description="Filter by letter type (e.g., 'COMPLETE RESPONSE', 'TENTATIVE APPROVAL')"
    )
    letter_date: Optional[str] = Field(
        default=None,
        description="Filter by letter date (MM/DD/YYYY format, e.g., '02/18/2022')"
    )
    approval_status: Optional[str] = Field(
        default=None,
        description="Filter by approval status ('Approved' or 'Unapproved')"
    )
    approver_center: Optional[str] = Field(
        default=None,
        description="Filter by FDA center responsible for approval"
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort order: 'relevance', 'letter_date:desc', 'letter_date:asc', or 'company_name'"
    )
    limit: int = Field(
        default=100,
        description="Maximum number of results to return (1-1000)",
        ge=1,
        le=1000
    )
    skip: int = Field(
        default=0,
        description="Number of results to skip for pagination",
        ge=0
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Path to save the results as CSV"
    )


class SearchCompleteResponseLettersTool(BaseTool):
    """
    Tool to search FDA Complete Response Letters (CRLs) database.
    
    Complete Response Letters are sent by FDA when an application will not be approved
    in its current form. This database provides public access to redacted CRLs for
    transparency in FDA decision-making.
    
    Use this to find information about:
    - Why drug applications were not approved
    - Deficiencies identified by FDA
    - Company responses to FDA feedback
    - Historical regulatory decisions
    """
    
    name: str = "search_complete_response_letters"
    description: str = (
        "Search FDA Complete Response Letters (CRLs). CRLs are issued when a drug application "
        "will not be approved in its current form. Returns letter details including application number, "
        "company name, letter type, date, approval status, and letter text. "
        "CRITICAL USE: When investigating why specific drug applications were rejected or not approved, "
        "or when analyzing FDA regulatory decisions and deficiency patterns."
    )
    args_schema: Type[BaseModel] = SearchCompleteResponseLettersToolInput
    sandbox: ExecutionSandboxWrapper = None
    
    def __init__(self, sandbox: ExecutionSandboxWrapper = None):
        super().__init__()
        self.sandbox = sandbox
    
    def _run(
        self,
        search: Optional[str] = None,
        application_number: Optional[str] = None,
        company_name: Optional[str] = None,
        letter_type: Optional[str] = None,
        letter_date: Optional[str] = None,
        approval_status: Optional[str] = None,
        approver_center: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
        save_path: Optional[str] = None,
    ) -> str:
        """Execute the tool to search Complete Response Letters."""
        
        df, md_str = search_complete_response_letters(
            search=search,
            application_number=application_number,
            company_name=company_name,
            letter_type=letter_type,
            letter_date=letter_date,
            approval_status=approval_status,
            approver_center=approver_center,
            sort=sort,
            limit=limit,
            skip=skip,
            save_path=save_path,
        )
        
        if df is None or len(df) == 0:
            return "No Complete Response Letters found matching the search criteria."
        
        return md_str


# =====================================================
# Tool 2: Search Drugs@FDA
# =====================================================
class SearchDrugsAtFDAToolInput(BaseModel):
    """Input schema for SearchDrugsAtFDATool."""
    
    search: Optional[str] = Field(
        default=None,
        description="General search term to query across all fields"
    )
    application_number: Optional[str] = Field(
        default=None,
        description="Filter by specific application number. Formats: 'NDA 123456', 'BLA 761215', 'ANDA 123456'."
    )
    sponsor_name: Optional[str] = Field(
        default=None,
        description="Filter by sponsor/company name (case-insensitive)"
    )
    brand_name: Optional[str] = Field(
        default=None,
        description="Filter by brand/trade name of the drug (case-insensitive)"
    )
    generic_name: Optional[str] = Field(
        default=None,
        description="Filter by generic name of the drug"
    )
    active_ingredient: Optional[str] = Field(
        default=None,
        description="Filter by active ingredient name (case-insensitive)"
    )
    marketing_status: Optional[str] = Field(
        default=None,
        description="Filter by marketing status (e.g., 'Prescription', 'Over-the-counter', 'Discontinued')"
    )
    submission_status: Optional[str] = Field(
        default=None,
        description="Filter by submission status ('AP' for Approved, 'TA' for Tentative Approval)"
    )
    submission_type: Optional[str] = Field(
        default=None,
        description="Filter by submission type ('ORIG' for Original, 'SUPPL' for Supplement)"
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort order: 'relevance', 'application_number', or 'sponsor_name'"
    )
    limit: int = Field(
        default=100,
        description="Maximum number of results to return (1-1000)",
        ge=1,
        le=1000
    )
    skip: int = Field(
        default=0,
        description="Number of results to skip for pagination",
        ge=0
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Path to save the results as CSV"
    )


class SearchDrugsAtFDATool(BaseTool):
    """
    Tool to search the FDA Drugs@FDA database.
    
    The Drugs@FDA database contains information about drug products approved by FDA.
    It includes application information, product details, submission history, and
    approval documentation.
    
    Use this to find:
    - Drug approval information (NDA, ANDA, BLA numbers)
    - Product details (brand names, dosage forms, active ingredients)
    - Submission history and status
    - Sponsor/manufacturer information
    - Marketing status
    """
    
    name: str = "search_drugs_at_fda"
    description: str = (
        "Search FDA Drugs@FDA database for approved drug products. Returns comprehensive information "
        "including application numbers, sponsors, brand names, active ingredients, dosage forms, "
        "marketing status, and submission history. "
        "CRITICAL USE: When researching FDA-approved drugs, finding approval dates and application numbers, "
        "or investigating drug product details and regulatory status."
    )
    args_schema: Type[BaseModel] = SearchDrugsAtFDAToolInput
    sandbox: ExecutionSandboxWrapper = None
    
    def __init__(self, sandbox: ExecutionSandboxWrapper = None):
        super().__init__()
        self.sandbox = sandbox
    
    def _run(
        self,
        search: Optional[str] = None,
        application_number: Optional[str] = None,
        sponsor_name: Optional[str] = None,
        brand_name: Optional[str] = None,
        generic_name: Optional[str] = None,
        active_ingredient: Optional[str] = None,
        marketing_status: Optional[str] = None,
        submission_status: Optional[str] = None,
        submission_type: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
        save_path: Optional[str] = None,
    ) -> str:
        """Execute the tool to search Drugs@FDA."""
        
        df, md_str = search_drugs_at_fda(
            search=search,
            application_number=application_number,
            sponsor_name=sponsor_name,
            brand_name=brand_name,
            generic_name=generic_name,
            active_ingredient=active_ingredient,
            marketing_status=marketing_status,
            submission_status=submission_status,
            submission_type=submission_type,
            sort=sort,
            limit=limit,
            skip=skip,
            save_path=save_path,
        )
        
        if df is None or len(df) == 0:
            return "No drugs found matching the search criteria."
        
        return md_str


# =====================================================
# Tool 3: Search Drug Labels
# =====================================================
class SearchDrugLabelsToolInput(BaseModel):
    """Input schema for SearchDrugLabelsTool."""
    
    search: Optional[str] = Field(
        default=None,
        description="General search term to query across all fields"
    )
    brand_name: Optional[str] = Field(
        default=None,
        description="Filter by brand/trade name"
    )
    generic_name: Optional[str] = Field(
        default=None,
        description="Filter by generic name"
    )
    substance_name: Optional[str] = Field(
        default=None,
        description="Filter by active substance/ingredient name"
    )
    manufacturer_name: Optional[str] = Field(
        default=None,
        description="Filter by manufacturer name"
    )
    application_number: Optional[str] = Field(
        default=None,
        description="Filter by application number (NDA, ANDA, BLA)"
    )
    product_type: Optional[str] = Field(
        default=None,
        description="Filter by product type (e.g., 'HUMAN PRESCRIPTION DRUG', 'HUMAN OTC DRUG')"
    )
    route: Optional[str] = Field(
        default=None,
        description="Filter by route of administration (e.g., 'ORAL', 'INTRAVENOUS')"
    )
    indication: Optional[str] = Field(
        default=None,
        description="Search within indications and usage section"
    )
    adverse_reaction: Optional[str] = Field(
        default=None,
        description="Search within adverse reactions section"
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort order: 'relevance', 'effective_time:desc', or 'effective_time:asc'"
    )
    limit: int = Field(
        default=100,
        description="Maximum number of results to return (1-1000)",
        ge=1,
        le=1000
    )
    skip: int = Field(
        default=0,
        description="Number of results to skip for pagination",
        ge=0
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Path to save the results as CSV"
    )


class SearchDrugLabelsTool(BaseTool):
    """
    Tool to search FDA Drug Product Labeling database.
    
    The Drug Label database contains comprehensive labeling information for approved
    drug products, including prescription and over-the-counter drugs. Labels include
    detailed information about:
    - Indications and usage
    - Dosage and administration
    - Contraindications
    - Warnings and precautions
    - Adverse reactions
    - Drug interactions
    - Clinical pharmacology
    
    This is the most detailed source of FDA-approved drug information.
    """
    
    name: str = "search_drug_labels"
    description: str = (
        "Search FDA Drug Product Labeling database for comprehensive drug information. "
        "Returns detailed label sections including indications, dosage, contraindications, warnings, "
        "adverse reactions, drug interactions, and clinical pharmacology. "
        "CRITICAL USE: When you need detailed, FDA-approved information about drug indications, "
        "dosing, safety warnings, side effects, or drug interactions. This is the authoritative source "
        "for drug labeling information."
    )
    args_schema: Type[BaseModel] = SearchDrugLabelsToolInput
    sandbox: ExecutionSandboxWrapper = None
    
    def __init__(self, sandbox: ExecutionSandboxWrapper = None):
        super().__init__()
        self.sandbox = sandbox
    
    def _run(
        self,
        search: Optional[str] = None,
        brand_name: Optional[str] = None,
        generic_name: Optional[str] = None,
        substance_name: Optional[str] = None,
        manufacturer_name: Optional[str] = None,
        application_number: Optional[str] = None,
        product_type: Optional[str] = None,
        route: Optional[str] = None,
        indication: Optional[str] = None,
        adverse_reaction: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
        save_path: Optional[str] = None,
    ) -> str:
        """Execute the tool to search drug labels."""
        
        df, md_str = search_drug_labels(
            search=search,
            brand_name=brand_name,
            generic_name=generic_name,
            substance_name=substance_name,
            manufacturer_name=manufacturer_name,
            application_number=application_number,
            product_type=product_type,
            route=route,
            indication=indication,
            adverse_reaction=adverse_reaction,
            sort=sort,
            limit=limit,
            skip=skip,
            save_path=save_path,
        )
        
        if df is None or len(df) == 0:
            return "No drug labels found matching the search criteria."
        
        return md_str


"""Tools for OpenFDA Complete Response Letters search."""
import logging
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from enum import StrEnum
import requests
import re

# ================================================
# Constants
# ================================================
OPENFDA_CRL_URL = "https://api.fda.gov/transparency/crl.json"

# ================================================
# Enums
# ================================================

class LetterType(StrEnum):
    """Complete Response Letter types based on actual FDA data."""
    COMPLETE_RESPONSE = "COMPLETE RESPONSE"
    CORRECTED_PROVISIONAL_DETERMINATION = "CORRECTED PROVISIONAL DETERMINATION"
    PROVISIONAL_DETERMINATION = "PROVISIONAL DETERMINATION"
    REFUSAL_TO_FILE = "REFUSAL TO FILE"
    RESCIND_COMPLETE_RESPONSE = "RESCIND COMPLETE RESPONSE"
    TENTATIVE_APPROVAL = "TENTATIVE APPROVAL"


class ApprovalStatus(StrEnum):
    """Approval status for applications based on actual FDA data."""
    APPROVED = "Approved"
    UNAPPROVED = "Unapproved"


class ApplicationType(StrEnum):
    """Application type prefixes."""
    NDA = "NDA"  # New Drug Application
    BLA = "BLA"  # Biologics License Application
    BL = "BL"    # Biologics License (legacy)


class SortOrder(StrEnum):
    """Sort order options for search results."""
    RELEVANCE = "relevance"
    DATE_DESC = "letter_date:desc"
    DATE_ASC = "letter_date:asc"
    COMPANY_NAME = "company_name"


class CountField(StrEnum):
    """Fields available for aggregation counting."""
    LETTER_TYPE = "letter_type"
    APPROVAL_STATUS = "approval_status"
    COMPANY_NAME = "company_name"
    APPROVER_CENTER = "approver_center"
    LETTER_YEAR = "letter_year"


# ================================================
# Schemas
# ================================================

class CompleteResponseLetterItem(BaseModel):
    """Individual Complete Response Letter item."""
    file_name: Optional[str] = Field(default=None, description="The name of the file")
    application_number: Optional[str] = Field(
        default=None, 
        description="The NDA, ANDA, or BLA number reported by the labeler"
    )
    letter_type: Optional[str] = Field(default=None, description="The type of letter")
    letter_date: Optional[str] = Field(default=None, description="The date the letter was signed")
    letter_year: Optional[str] = Field(default=None, description="The year the letter was signed")
    approval_status: Optional[str] = Field(default=None, description="The approval status")
    company_name: Optional[str] = Field(default=None, description="The name of the company")
    company_rep: Optional[str] = Field(default=None, description="The representative of the company")
    company_address: Optional[str] = Field(default=None, description="The contact address of the company")
    approver_name: Optional[str] = Field(default=None, description="The name of the person responsible for approval")
    approver_title: Optional[str] = Field(default=None, description="The professional title of the person responsible for approval")
    approver_center: Optional[str] = Field(default=None, description="The FDA center responsible for approval")
    text: Optional[str] = Field(default=None, description="The text of the file")


class CRLSearchResponse(BaseModel):
    """Response for Complete Response Letters search."""
    results: list[CompleteResponseLetterItem]
    meta: dict[str, Any] = Field(default_factory=dict)


# ================================================
# Helper Functions
# ================================================

def _normalize_application_number(app_number: str) -> list[str]:
    """
    Normalize application number to handle different formats.
    
    Args:
        app_number: Application number like "NDA123456" or "NDA 123456"
        
    Returns:
        List of normalized formats to search for
    """
    if not app_number:
        return []
    
    app_number = app_number.strip()
    
    # Pattern: PREFIX + optional space + numbers
    pattern = r'^(NDA|BLA|BL)\s*(\d+)(.*)$'
    match = re.match(pattern, app_number.upper())
    
    if match:
        prefix = match.group(1)
        number = match.group(2)
        suffix = match.group(3)
        
        # Return both formats: with space and without space
        base_with_space = f"{prefix} {number}{suffix}"
        base_without_space = f"{prefix}{number}{suffix}"
        
        return [base_with_space, base_without_space]
    else:
        # If it doesn't match pattern, return as-is
        return [app_number]


def _build_search_query(
    search: Optional[str] = None,
    application_number: Optional[str] = None,
    company_name: Optional[str] = None,
    letter_type: Optional[LetterType | str] = None,
    letter_date: Optional[str] = None,
    approval_status: Optional[ApprovalStatus | str] = None,
    approver_center: Optional[str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
) -> dict[str, Any]:
    """Build the search query parameters for OpenFDA API."""
    params = {}
    
    # Build search terms
    search_terms = []
    
    if search:
        search_terms.append(search)
    
    if application_number:
        # Handle different application number formats
        normalized_numbers = _normalize_application_number(application_number)
        if len(normalized_numbers) == 1:
            search_terms.append(f"application_number:{normalized_numbers[0]}")
        else:
            # Create OR query for different formats
            app_num_terms = [f"application_number:{num}" for num in normalized_numbers]
            search_terms.append(f"({' OR '.join(app_num_terms)})")
    
    if company_name:
        search_terms.append(f"company_name:\"{company_name}\"")
    
    if letter_type:
        letter_type_value = letter_type.value if isinstance(letter_type, LetterType) else letter_type
        search_terms.append(f'letter_type:"{letter_type_value}"')
    
    if letter_date:
        search_terms.append(f'letter_date:"{letter_date}"')
    
    if approval_status:
        approval_status_value = approval_status.value if isinstance(approval_status, ApprovalStatus) else approval_status
        search_terms.append(f'approval_status:"{approval_status_value}"')
    
    if approver_center:
        search_terms.append(f'approver_center:"{approver_center}"')
    
    if search_terms:
        params["search"] = " AND ".join(search_terms)
    
    # Add pagination parameters
    params["limit"] = limit
    params["skip"] = skip
    
    # Add sorting if specified
    if sort:
        sort_value = sort.value if isinstance(sort, SortOrder) else sort
        params["sort"] = sort_value
    
    # Add count parameter if specified
    if count:
        count_value = count.value if isinstance(count, CountField) else count
        params["count"] = count_value
    
    return params


def _clean_result(record: dict) -> dict:
    """Clean and standardize a single CRL record."""
    # Extract relevant fields and handle nested structures
    cleaned = {}
    
    # Map standard fields from OpenFDA API response to our model
    field_mapping = {
        "file_name": "file_name",
        "application_number": "application_number", 
        "letter_type": "letter_type",
        "letter_date": "letter_date",
        "letter_year": "letter_year",
        "approval_status": "approval_status",
        "company_name": "company_name",
        "company_rep": "company_rep", 
        "company_address": "company_address",
        "approver_name": "approver_name",
        "approver_title": "approver_title",
        "approver_center": "approver_center",
        "text": "text"
    }
    
    for api_field, output_field in field_mapping.items():
        value = record.get(api_field)
        if value:
            # Handle list values
            if isinstance(value, list) and value:
                if api_field == "approver_center":
                    # Join multiple centers with ", "
                    cleaned[output_field] = ", ".join(value)
                elif api_field == "application_number":
                    # Take the first application number
                    cleaned[output_field] = value[0]
                else:
                    # For other fields, take the first element
                    cleaned[output_field] = value[0]
            else:
                cleaned[output_field] = value
    
    return cleaned


def _to_markdown(data: list[dict]) -> str:
    """Convert list of dictionaries to markdown table."""
    if not data:
        return ""
    
    df = pd.DataFrame(data)
    return df.to_markdown(index=False)


# ================================================
# Main Search Function
# ================================================

def search_complete_response_letters(
    search: Optional[str] = None,
    application_number: Optional[str] = None,
    company_name: Optional[str] = None,
    letter_type: Optional[LetterType | str] = None,
    letter_date: Optional[str] = None,
    approval_status: Optional[ApprovalStatus | str] = None,
    approver_center: Optional[str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
    save_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Search FDA Complete Response Letters (CRLs) database.

    Complete Response Letters are sent by FDA when an application will not be approved
    in its current form. This database provides public access to redacted CRLs for
    transparency in FDA decision-making.

    Args:
        search: General search term to query across all fields
        application_number: Filter by specific application number. Formats: 'NDA 123456', 'BLA 761215', 'BL 103324'. Space between prefix and number is optional (e.g., 'NDA123456' works too).
        company_name: Filter by company name
        letter_type: Filter by letter type (use LetterType enum or string)
        letter_date: Filter by letter date (MM/DD/YYYY format, e.g., '02/18/2022')
        approval_status: Filter by approval status (use ApprovalStatus enum or string)
        approver_center: Filter by FDA center responsible for approval
        sort: Sort order for results (use SortOrder enum or string)
        limit: Maximum number of results to return (1-1000, default: 100)
        skip: Number of results to skip for pagination (default: 0)
        count: Field to count by for aggregation (use CountField enum or string)
        save_path: Path to save the results as CSV

    Returns:
        A tuple with the dataframe and the string with the search results
    """
    
    # Build query parameters
    params = _build_search_query(
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
        count=count,
    )
    
    # Make API request
    try:
        response = requests.get(OPENFDA_CRL_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"OpenFDA API error: {e}")
        return pd.DataFrame(), f"# Complete Response Letters Search Results\n\nError: {str(e)}"
    
    # Parse the response
    try:
        results_data = data.get("results", [])
        meta_data = data.get("meta", {})
        
        # Clean and parse each result
        cleaned_results = []
        for result in results_data:
            cleaned = _clean_result(result)
            cleaned_results.append(CompleteResponseLetterItem(**cleaned))
        
    except Exception as e:
        logging.error(f"Error parsing OpenFDA response: {e}")
        return pd.DataFrame(), f"# Complete Response Letters Search Results\n\nError: Failed to parse response: {e}"
    
    # Prepare results for display
    results_data = []
    for result in cleaned_results:
        result_dict = result.model_dump(exclude_none=True)
        # Limit text field for display
        if result_dict.get("text") and len(result_dict["text"]) > 200:
            result_dict["text"] = result_dict["text"][:200] + "..."
        results_data.append(result_dict)
    
    # Generate summary
    total_results = meta_data.get("results", {}).get("total", len(results_data))
    summary = f"# Complete Response Letters Search Results\n\nFound {total_results} letters"
    
    if skip > 0:
        summary += f" (showing results {skip + 1}-{skip + len(results_data)})"
    
    summary += "\n\n"
    
    # Format results as markdown
    if results_data:
        results_markdown = _to_markdown(results_data)
        output = f"{summary}{results_markdown}"
    else:
        output = f"{summary}No results found."

    # Create dataframe
    # Prepare full data for CSV (without text truncation)
    full_results_data = []
    for result in cleaned_results:
        full_results_data.append(result.model_dump(exclude_none=True))
    output_dataframe = pd.DataFrame(full_results_data)

    # Save to file if requested
    if save_path:
        try:
            if not save_path.endswith(".csv"):
                save_path = save_path + ".csv"

            if full_results_data:
                output_dataframe.to_csv(save_path, index=False)
                save_msg = f"Full results saved to {save_path}"
            else:
                save_msg = "No results to save"
            
            output += f"\n\n---\n{save_msg}"
            
        except Exception as e:
            logging.error(f"Error saving results to {save_path}: {e}")
            output += f"\n\n---\nError saving results to {save_path}: {e}"
    
    return output_dataframe, output


"""Tools for OpenFDA Drugs@FDA database search."""
import logging
import re
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from enum import StrEnum
import requests

# ================================================
# Constants
# ================================================
OPENFDA_DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"

# ================================================
# Enums
# ================================================

class MarketingStatus(StrEnum):
    """Marketing status for drug products based on actual FDA data."""
    PRESCRIPTION = "Prescription"
    DISCONTINUED = "Discontinued"
    TENTATIVE_APPROVAL = "None (Tentative Approval)"
    OVER_THE_COUNTER = "Over-the-counter"


class SubmissionStatus(StrEnum):
    """Submission status codes based on actual FDA data."""
    APPROVED = "AP"  # Approved
    TENTATIVE_APPROVAL = "TA"  # Tentative Approval


class SubmissionType(StrEnum):
    """Submission type codes based on actual FDA data."""
    ORIGINAL = "ORIG"  # Original application
    SUPPLEMENT = "SUPPL"  # Supplement


class SortOrder(StrEnum):
    """Sort order options for search results."""
    RELEVANCE = "relevance"
    APPLICATION_NUMBER = "application_number"
    SPONSOR_NAME = "sponsor_name"


class CountField(StrEnum):
    """Fields available for aggregation counting."""
    SPONSOR_NAME = "sponsor_name"
    MARKETING_STATUS = "products.marketing_status"
    SUBMISSION_STATUS = "submissions.submission_status"
    SUBMISSION_TYPE = "submissions.submission_type"
    APPLICATION_NUMBER = "application_number"


# ================================================
# Schemas
# ================================================

class ActiveIngredient(BaseModel):
    """Active ingredient information."""
    name: Optional[str] = Field(default=None, description="Name of the active ingredient")
    strength: Optional[str] = Field(default=None, description="Strength of the active ingredient")


class Product(BaseModel):
    """Drug product information."""
    product_number: Optional[str] = Field(default=None, description="Product number")
    brand_name: Optional[str] = Field(default=None, description="Brand or trade name")
    active_ingredients: list[ActiveIngredient] = Field(default_factory=list, description="List of active ingredients")
    dosage_form: Optional[str] = Field(default=None, description="Dosage form (e.g., tablet, injection)")
    route: Optional[str] = Field(default=None, description="Route of administration")
    marketing_status: Optional[str] = Field(default=None, description="Marketing status")
    reference_drug: Optional[str] = Field(default=None, description="Whether this is a reference drug")
    reference_standard: Optional[str] = Field(default=None, description="Whether this is a reference standard")
    te_code: Optional[str] = Field(default=None, description="Therapeutic equivalence code")


class ApplicationDoc(BaseModel):
    """Application document information."""
    id: Optional[str] = Field(default=None, description="Document ID")
    url: Optional[str] = Field(default=None, description="Document URL")
    date: Optional[str] = Field(default=None, description="Document date")
    type: Optional[str] = Field(default=None, description="Document type (e.g., Label, Letter)")


class Submission(BaseModel):
    """Submission information."""
    submission_type: Optional[str] = Field(default=None, description="Type of submission")
    submission_number: Optional[str] = Field(default=None, description="Submission number")
    submission_status: Optional[str] = Field(default=None, description="Submission status")
    submission_status_date: Optional[str] = Field(default=None, description="Status date")
    submission_class_code: Optional[str] = Field(default=None, description="Classification code")
    submission_class_code_description: Optional[str] = Field(default=None, description="Classification description")
    review_priority: Optional[str] = Field(default=None, description="Review priority")
    application_docs: list[ApplicationDoc] = Field(default_factory=list, description="List of application documents")


class DrugApplicationItem(BaseModel):
    """Individual drug application item."""
    application_number: Optional[str] = Field(default=None, description="Application number (NDA, ANDA, BLA)")
    sponsor_name: Optional[str] = Field(default=None, description="Sponsor company name")
    products: list[Product] = Field(default_factory=list, description="List of drug products")
    submissions: list[Submission] = Field(default_factory=list, description="List of submissions")


class DrugsAtFDASearchResponse(BaseModel):
    """Response for Drugs@FDA search."""
    results: list[DrugApplicationItem]
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
    pattern = r'^(NDA|BLA|ANDA)\s*(\d+)(.*)$'
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


def _normalize_text_for_search(text: str, field_name: str) -> list[str]:
    """
    Normalize text for search to handle case sensitivity.
    
    Args:
        text: The text to normalize
        field_name: The field name to determine normalization strategy
        
    Returns:
        List of normalized text variations to try
    """
    if not text:
        return []
    
    text = text.strip()
    variations = []
    
    # For sponsor names, try common variations
    if field_name == "sponsor_name":
        variations.extend([
            text.upper(),  # Most common in FDA data
            text,  # As provided
            text.title(),  # Title Case
            text.lower(),  # lowercase
        ])
    
    # For brand names, try variations
    elif field_name == "brand_name":
        variations.extend([
            text.upper(),  # Most common
            text,  # As provided
            text.title(),  # Title Case
        ])
    
    # For active ingredients, try variations
    elif field_name == "active_ingredient":
        variations.extend([
            text.upper(),  # Most common
            text,  # As provided
            text.title(),  # Title Case
        ])
    
    else:
        # Default: try as-is and uppercase
        variations.extend([text, text.upper()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations


def _build_search_query(
    search: Optional[str] = None,
    application_number: Optional[str] = None,
    sponsor_name: Optional[str] = None,
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    active_ingredient: Optional[str] = None,
    marketing_status: Optional[MarketingStatus | str] = None,
    submission_status: Optional[SubmissionStatus | str] = None,
    submission_type: Optional[SubmissionType | str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
) -> dict[str, Any]:
    """Build the search query parameters for OpenFDA Drugs@FDA API."""
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
    
    if sponsor_name:
        # Handle case variations for sponsor name
        sponsor_variations = _normalize_text_for_search(sponsor_name, "sponsor_name")
        if len(sponsor_variations) == 1:
            search_terms.append(f'sponsor_name:"{sponsor_variations[0]}"')
        else:
            # Create OR query for different case variations
            sponsor_terms = [f'sponsor_name:"{var}"' for var in sponsor_variations]
            search_terms.append(f"({' OR '.join(sponsor_terms)})")
    
    if brand_name:
        # Handle case variations for brand name
        brand_variations = _normalize_text_for_search(brand_name, "brand_name")
        if len(brand_variations) == 1:
            search_terms.append(f'products.brand_name:"{brand_variations[0]}"')
        else:
            brand_terms = [f'products.brand_name:"{var}"' for var in brand_variations]
            search_terms.append(f"({' OR '.join(brand_terms)})")
    
    if generic_name:
        search_terms.append(f'openfda.generic_name:"{generic_name}"')
    
    if active_ingredient:
        # Handle case variations for active ingredient
        ingredient_variations = _normalize_text_for_search(active_ingredient, "active_ingredient")
        if len(ingredient_variations) == 1:
            search_terms.append(f'products.active_ingredients.name:"{ingredient_variations[0]}"')
        else:
            ingredient_terms = [f'products.active_ingredients.name:"{var}"' for var in ingredient_variations]
            search_terms.append(f"({' OR '.join(ingredient_terms)})")
    
    if marketing_status:
        marketing_status_value = marketing_status.value if isinstance(marketing_status, MarketingStatus) else marketing_status
        search_terms.append(f'products.marketing_status:"{marketing_status_value}"')
    
    if submission_status:
        submission_status_value = submission_status.value if isinstance(submission_status, SubmissionStatus) else submission_status
        search_terms.append(f'submissions.submission_status:"{submission_status_value}"')
    
    if submission_type:
        submission_type_value = submission_type.value if isinstance(submission_type, SubmissionType) else submission_type
        search_terms.append(f'submissions.submission_type:"{submission_type_value}"')
    
    if search_terms:
        params["search"] = " AND ".join(search_terms)
    
    # Add pagination parameters
    params["limit"] = limit
    params["skip"] = skip
    
    # Add sorting if specified (skip "relevance" as it's not supported by OpenFDA API)
    if sort:
        sort_value = sort.value if isinstance(sort, SortOrder) else sort
        # OpenFDA API doesn't support sort="relevance" - skip it to use default sorting
        if sort_value != "relevance":
            params["sort"] = sort_value
    
    # Add count parameter if specified
    if count:
        count_value = count.value if isinstance(count, CountField) else count
        params["count"] = count_value
    
    return params


def _clean_result(record: dict) -> dict:
    """Clean and standardize a single drug application record."""
    cleaned = {}
    
    # Map main fields
    cleaned["application_number"] = record.get("application_number")
    cleaned["sponsor_name"] = record.get("sponsor_name")
    
    # Process products
    products = []
    for product in record.get("products", []):
        clean_product = {
            "product_number": product.get("product_number"),
            "brand_name": product.get("brand_name"),
            "dosage_form": product.get("dosage_form"),
            "route": product.get("route"),
            "marketing_status": product.get("marketing_status"),
            "reference_drug": product.get("reference_drug"),
            "reference_standard": product.get("reference_standard"),
            "te_code": product.get("te_code")
        }
        
        # Process active ingredients
        active_ingredients = []
        for ingredient in product.get("active_ingredients", []):
            active_ingredients.append({
                "name": ingredient.get("name"),
                "strength": ingredient.get("strength")
            })
        clean_product["active_ingredients"] = active_ingredients
        products.append(clean_product)
    
    cleaned["products"] = products
    
    # Process submissions
    submissions = []
    for submission in record.get("submissions", []):
        clean_submission = {
            "submission_type": submission.get("submission_type"),
            "submission_number": submission.get("submission_number"),
            "submission_status": submission.get("submission_status"),
            "submission_status_date": submission.get("submission_status_date"),
            "submission_class_code": submission.get("submission_class_code"),
            "submission_class_code_description": submission.get("submission_class_code_description"),
            "review_priority": submission.get("review_priority")
        }
        
        # Process application documents
        application_docs = []
        for doc in submission.get("application_docs", []):
            application_docs.append({
                "id": doc.get("id"),
                "url": doc.get("url"),
                "date": doc.get("date"),
                "type": doc.get("type")
            })
        clean_submission["application_docs"] = application_docs
        
        submissions.append(clean_submission)
    
    cleaned["submissions"] = submissions
    
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

def search_drugs_at_fda(
    search: Optional[str] = None,
    application_number: Optional[str] = None,
    sponsor_name: Optional[str] = None,
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    active_ingredient: Optional[str] = None,
    marketing_status: Optional[MarketingStatus | str] = None,
    submission_status: Optional[SubmissionStatus | str] = None,
    submission_type: Optional[SubmissionType | str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
    save_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Search FDA Drugs@FDA database.

    The Drugs@FDA database contains information about drug products approved by FDA.
    It includes application information, product details, and submission history.

    Args:
        search: General search term to query across all fields
        application_number: Filter by specific application number. Formats: 'NDA 123456', 'BLA 761215', 'ANDA 123456'. Space between prefix and number is optional.
        sponsor_name: Filter by sponsor/company name. Case-insensitive - will try multiple variations (e.g., 'Pfizer', 'PFIZER', 'pfizer')
        brand_name: Filter by brand/trade name of the drug. Case-insensitive - will try multiple variations
        generic_name: Filter by generic name of the drug
        active_ingredient: Filter by active ingredient name. Case-insensitive - will try multiple variations
        marketing_status: Filter by marketing status (use MarketingStatus enum or string)
        submission_status: Filter by submission status (use SubmissionStatus enum or string)
        submission_type: Filter by submission type (use SubmissionType enum or string)
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
        count=count,
    )
    
    # Make API request
    try:
        response = requests.get(OPENFDA_DRUGSFDA_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"OpenFDA Drugs@FDA API error: {e}")
        return pd.DataFrame(), f"# Drugs@FDA Search Results\n\nError: {str(e)}"
    
    # Parse the response
    try:
        results_data = data.get("results", [])
        meta_data = data.get("meta", {})
        
        # Clean and parse each result
        cleaned_results = []
        for result in results_data:
            cleaned = _clean_result(result)
            cleaned_results.append(DrugApplicationItem(**cleaned))
        
    except Exception as e:
        logging.error(f"Error parsing OpenFDA Drugs@FDA response: {e}")
        return pd.DataFrame(), f"# Drugs@FDA Search Results\n\nError: Failed to parse response: {e}"
    
    # Prepare results for display
    results_data = []
    for result in cleaned_results:
        result_dict = result.model_dump(exclude_none=True)
        
        # Simplify for display - show key product info
        if result_dict.get("products"):
            # Take first product for brief display
            first_product = result_dict["products"][0]
            result_dict["brand_name"] = first_product.get("brand_name")
            result_dict["dosage_form"] = first_product.get("dosage_form")
            result_dict["marketing_status"] = first_product.get("marketing_status")
            
            # Simplify active ingredients
            if first_product.get("active_ingredients"):
                ingredients = [f"{ing.get('name', '')} {ing.get('strength', '')}".strip() 
                             for ing in first_product["active_ingredients"]]
                result_dict["active_ingredients"] = ", ".join(ingredients)
        
        # Simplify submissions - show latest
        if result_dict.get("submissions"):
            latest_submission = result_dict["submissions"][0]  # Assume first is latest
            result_dict["latest_submission_status"] = latest_submission.get("submission_status")
            result_dict["latest_submission_type"] = latest_submission.get("submission_type")
        
        # Remove complex nested data for brief display
        result_dict.pop("products", None)
        result_dict.pop("submissions", None)
        
        results_data.append(result_dict)
    
    # Generate summary
    total_results = meta_data.get("results", {}).get("total", len(results_data))
    summary = f"# Drugs@FDA Search Results\n\nFound {total_results} applications"
    
    if skip > 0:
        summary += f" (showing results {skip + 1}-{skip + len(results_data)})"
    
    summary += "\n\n"

    # Prepare flattened data for CSV
    flattened_data = []
    try:
        for result in cleaned_results:
            base_data = {
                "application_number": result.application_number,
                "sponsor_name": result.sponsor_name,
            }
            
            # Add latest submission info to base data
            if result.submissions:
                latest_submission = result.submissions[0]  # Assume first is latest
                base_data.update({
                    "latest_submission_type": latest_submission.submission_type,
                    "latest_submission_number": latest_submission.submission_number,
                    "latest_submission_status": latest_submission.submission_status,
                    "latest_submission_status_date": latest_submission.submission_status_date,
                    "latest_submission_class_code": latest_submission.submission_class_code,
                    "latest_submission_class_code_description": latest_submission.submission_class_code_description,
                    "latest_submission_review_priority": latest_submission.review_priority,
                })
            
            # Create one row per product
            if result.products:
                for product in result.products:
                    # Create row with specific column order
                    row = {
                        "application_number": result.application_number,
                        "sponsor_name": result.sponsor_name,
                        # Product information immediately after application info
                        "product_number": product.product_number,
                        "brand_name": product.brand_name,
                        "dosage_form": product.dosage_form,
                        "route": product.route,
                        "marketing_status": product.marketing_status,
                        "reference_drug": product.reference_drug,
                        "reference_standard": product.reference_standard,
                        "te_code": product.te_code,
                    }
                    
                    # Add active ingredients as a combined string
                    if product.active_ingredients:
                        ingredients = []
                        for ing in product.active_ingredients:
                            ing_str = f"{ing.name or ''} {ing.strength or ''}".strip()
                            if ing_str:
                                ingredients.append(ing_str)
                        row["active_ingredients"] = "; ".join(ingredients)
                    else:
                        row["active_ingredients"] = ""
                    
                    # Add submission information last
                    if result.submissions:
                        latest_submission = result.submissions[0]
                        row.update({
                            "latest_submission_type": latest_submission.submission_type,
                            "latest_submission_number": latest_submission.submission_number,
                            "latest_submission_status": latest_submission.submission_status,
                            "latest_submission_status_date": latest_submission.submission_status_date,
                            "latest_submission_class_code": latest_submission.submission_class_code,
                            "latest_submission_class_code_description": latest_submission.submission_class_code_description,
                            "latest_submission_review_priority": latest_submission.review_priority,
                        })
                        
                        # Collect application_docs from ALL submissions (not just latest)
                        all_docs_list = []
                        for submission in result.submissions:
                            if submission.application_docs:
                                for doc in submission.application_docs:
                                    doc_info = f"{doc.type}: {doc.url}"
                                    all_docs_list.append(doc_info)
                        
                        if all_docs_list:
                            row["application_docs"] = " | ".join(all_docs_list)
                        else:
                            row["application_docs"] = ""
                    else:
                        row.update({
                            "latest_submission_type": "",
                            "latest_submission_number": "",
                            "latest_submission_status": "",
                            "latest_submission_status_date": "",
                            "latest_submission_class_code": "",
                            "latest_submission_class_code_description": "",
                            "latest_submission_review_priority": "",
                            "application_docs": "",
                        })
                    
                    flattened_data.append(row)
            else:
                # No products, just add base data
                row = {
                    "application_number": result.application_number,
                    "sponsor_name": result.sponsor_name,
                    # Empty product information
                    "product_number": "",
                    "brand_name": "",
                    "dosage_form": "",
                    "route": "",
                    "marketing_status": "",
                    "reference_drug": "",
                    "reference_standard": "",
                    "te_code": "",
                    "active_ingredients": "",
                }
                
                # Add submission information
                if result.submissions:
                    latest_submission = result.submissions[0]
                    row.update({
                        "latest_submission_type": latest_submission.submission_type,
                        "latest_submission_number": latest_submission.submission_number,
                        "latest_submission_status": latest_submission.submission_status,
                        "latest_submission_status_date": latest_submission.submission_status_date,
                        "latest_submission_class_code": latest_submission.submission_class_code,
                        "latest_submission_class_code_description": latest_submission.submission_class_code_description,
                        "latest_submission_review_priority": latest_submission.review_priority,
                    })
                    
                    # Collect application_docs from ALL submissions (not just latest)
                    all_docs_list = []
                    for submission in result.submissions:
                        if submission.application_docs:
                            for doc in submission.application_docs:
                                doc_info = f"{doc.type}: {doc.url}"
                                all_docs_list.append(doc_info)
                    
                    if all_docs_list:
                        row["application_docs"] = " | ".join(all_docs_list)
                    else:
                        row["application_docs"] = ""
                else:
                    row.update({
                        "latest_submission_type": "",
                        "latest_submission_number": "",
                        "latest_submission_status": "",
                        "latest_submission_status_date": "",
                        "latest_submission_class_code": "",
                        "latest_submission_class_code_description": "",
                        "latest_submission_review_priority": "",
                        "application_docs": "",
                    })
                
                flattened_data.append(row)
    except Exception as e:
        flattened_data = []
    
    # Format results as markdown
    if results_data:
        results_markdown = _to_markdown(results_data)
        output = f"{summary}{results_markdown}"
        output_dataframe = pd.DataFrame(flattened_data)
    else:
        output = f"{summary}No results found."
        output_dataframe = pd.DataFrame()
    
    # Save to file if requested
    if save_path:
        if flattened_data:
            output_dataframe.to_csv(save_path, index=False)
            save_msg = f"Flattened results saved to {save_path}"
        else:
            save_msg = "No results to save"
        
        output += f"\n\n---\n{save_msg}"
        
    return output_dataframe, output


"""Tools for OpenFDA Drug Product Labeling search."""
import logging
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from enum import StrEnum
import requests

# ================================================
# Constants
# ================================================
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# ================================================
# Enums
# ================================================

class ProductType(StrEnum):
    """Product type options based on FDA data."""
    HUMAN_PRESCRIPTION = "HUMAN PRESCRIPTION DRUG"
    HUMAN_OTC = "HUMAN OTC DRUG"
    VACCINE = "VACCINE"
    PLASMA_DERIVATIVE = "PLASMA DERIVATIVE"
    CELLULAR_THERAPY = "CELLULAR THERAPY"


class RouteType(StrEnum):
    """Common route of administration types."""
    ORAL = "ORAL"
    INTRAVENOUS = "INTRAVENOUS"
    INTRAMUSCULAR = "INTRAMUSCULAR"
    SUBCUTANEOUS = "SUBCUTANEOUS"
    TOPICAL = "TOPICAL"
    OPHTHALMIC = "OPHTHALMIC"
    INHALATION = "RESPIRATORY (INHALATION)"


class SortOrder(StrEnum):
    """Sort order options for search results."""
    RELEVANCE = "relevance"
    EFFECTIVE_TIME_DESC = "effective_time:desc"
    EFFECTIVE_TIME_ASC = "effective_time:asc"


class CountField(StrEnum):
    """Fields available for aggregation counting."""
    BRAND_NAME = "openfda.brand_name"
    GENERIC_NAME = "openfda.generic_name"
    MANUFACTURER_NAME = "openfda.manufacturer_name"
    PRODUCT_TYPE = "openfda.product_type"
    ROUTE = "openfda.route"
    SUBSTANCE_NAME = "openfda.substance_name"


# ================================================
# Schemas
# ================================================

class OpenFDAInfo(BaseModel):
    """OpenFDA harmonized information."""
    application_number: list[str] = Field(default_factory=list, description="Application numbers (NDA, ANDA, BLA)")
    brand_name: list[str] = Field(default_factory=list, description="Brand or trade names")
    generic_name: list[str] = Field(default_factory=list, description="Generic names")
    manufacturer_name: list[str] = Field(default_factory=list, description="Manufacturer names")
    product_type: list[str] = Field(default_factory=list, description="Product types")
    route: list[str] = Field(default_factory=list, description="Routes of administration")
    substance_name: list[str] = Field(default_factory=list, description="Active substance names")
    rxcui: list[str] = Field(default_factory=list, description="RxNorm concept unique identifiers")
    spl_id: list[str] = Field(default_factory=list, description="SPL document IDs")
    spl_set_id: list[str] = Field(default_factory=list, description="SPL set IDs")
    package_ndc: list[str] = Field(default_factory=list, description="Package NDC codes")
    unii: list[str] = Field(default_factory=list, description="Unique Ingredient Identifiers")


class ProductLabelItem(BaseModel):
    """Individual drug product label item."""
    id: Optional[str] = Field(default=None, description="Label document ID")
    set_id: Optional[str] = Field(default=None, description="Label set ID")
    effective_time: Optional[str] = Field(default=None, description="Effective date of the label")
    version: Optional[str] = Field(default=None, description="Version number")
    
    # Label sections (commonly used)
    indications_and_usage: list[str] = Field(default_factory=list, description="Indications and usage information")
    dosage_and_administration: list[str] = Field(default_factory=list, description="Dosage and administration")
    contraindications: list[str] = Field(default_factory=list, description="Contraindications")
    warnings: list[str] = Field(default_factory=list, description="Warnings and precautions")
    adverse_reactions: list[str] = Field(default_factory=list, description="Adverse reactions")
    drug_interactions: list[str] = Field(default_factory=list, description="Drug interactions")
    description: list[str] = Field(default_factory=list, description="Description")
    clinical_pharmacology: list[str] = Field(default_factory=list, description="Clinical pharmacology")
    how_supplied: list[str] = Field(default_factory=list, description="How supplied")
    
    # OpenFDA harmonized data
    openfda: Optional[OpenFDAInfo] = Field(default=None, description="OpenFDA harmonized information")


class ProductLabelSearchResponse(BaseModel):
    """Response for Product Label search."""
    results: list[ProductLabelItem]
    meta: dict[str, Any] = Field(default_factory=dict)


# ================================================
# Helper Functions
# ================================================

def _build_search_query(
    search: Optional[str] = None,
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    substance_name: Optional[str] = None,
    manufacturer_name: Optional[str] = None,
    application_number: Optional[str] = None,
    product_type: Optional[ProductType | str] = None,
    route: Optional[RouteType | str] = None,
    indication: Optional[str] = None,
    adverse_reaction: Optional[str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
) -> dict[str, Any]:
    """Build the search query parameters for OpenFDA Drug Label API."""
    params = {}
    
    # Build search terms
    search_terms = []
    
    if search:
        search_terms.append(search)
    
    if brand_name:
        search_terms.append(f'openfda.brand_name:"{brand_name}"')
    
    if generic_name:
        search_terms.append(f'openfda.generic_name:"{generic_name}"')
    
    if substance_name:
        search_terms.append(f'openfda.substance_name:"{substance_name}"')
    
    if manufacturer_name:
        search_terms.append(f'openfda.manufacturer_name:"{manufacturer_name}"')
    
    if application_number:
        search_terms.append(f'openfda.application_number:"{application_number}"')
    
    if product_type:
        product_type_value = product_type.value if isinstance(product_type, ProductType) else product_type
        search_terms.append(f'openfda.product_type:"{product_type_value}"')
    
    if route:
        route_value = route.value if isinstance(route, RouteType) else route
        search_terms.append(f'openfda.route:"{route_value}"')
    
    if indication:
        search_terms.append(f'indications_and_usage:"{indication}"')
    
    if adverse_reaction:
        search_terms.append(f'adverse_reactions:"{adverse_reaction}"')
    
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
    """Clean and standardize a single product label record."""
    cleaned = {}
    
    # Map main fields
    cleaned["id"] = record.get("id")
    cleaned["set_id"] = record.get("set_id")
    cleaned["effective_time"] = record.get("effective_time")
    cleaned["version"] = record.get("version")
    
    # Map label sections
    label_sections = [
        "indications_and_usage",
        "dosage_and_administration",
        "contraindications",
        "warnings",
        "adverse_reactions",
        "drug_interactions",
        "description",
        "clinical_pharmacology",
        "how_supplied",
    ]
    
    for section in label_sections:
        value = record.get(section)
        if value:
            if isinstance(value, list):
                cleaned[section] = value
            else:
                cleaned[section] = [value]
        else:
            cleaned[section] = []
    
    # Process OpenFDA data
    openfda_data = record.get("openfda", {})
    if openfda_data:
        openfda_cleaned = {}
        openfda_fields = [
            "application_number",
            "brand_name",
            "generic_name",
            "manufacturer_name",
            "product_type",
            "route",
            "substance_name",
            "rxcui",
            "spl_id",
            "spl_set_id",
            "package_ndc",
            "unii",
        ]
        
        for field in openfda_fields:
            value = openfda_data.get(field, [])
            if not isinstance(value, list):
                value = [value] if value else []
            openfda_cleaned[field] = value
        
        cleaned["openfda"] = openfda_cleaned
    
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

def search_drug_labels(
    search: Optional[str] = None,
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    substance_name: Optional[str] = None,
    manufacturer_name: Optional[str] = None,
    application_number: Optional[str] = None,
    product_type: Optional[ProductType | str] = None,
    route: Optional[RouteType | str] = None,
    indication: Optional[str] = None,
    adverse_reaction: Optional[str] = None,
    sort: Optional[SortOrder | str] = None,
    limit: int = 100,
    skip: int = 0,
    count: Optional[CountField | str] = None,
    save_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Search FDA Drug Product Labeling database.

    The Drug Label database contains labeling information for approved drug products,
    including prescription and over-the-counter drugs. Labels provide comprehensive
    information about the drug product including indications, dosage, warnings, and more.

    Args:
        search: General search term to query across all fields
        brand_name: Filter by brand/trade name
        generic_name: Filter by generic name
        substance_name: Filter by active substance/ingredient name
        manufacturer_name: Filter by manufacturer name
        application_number: Filter by application number (NDA, ANDA, BLA)
        product_type: Filter by product type (use ProductType enum or string)
        route: Filter by route of administration (use RouteType enum or string)
        indication: Search within indications and usage section
        adverse_reaction: Search within adverse reactions section
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
        count=count,
    )
    
    # Make API request
    try:
        response = requests.get(OPENFDA_LABEL_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"OpenFDA Drug Label API error: {e}")
        return pd.DataFrame(), f"# Drug Product Labeling Search Results\n\nError: {str(e)}"
    
    # Parse the response
    try:
        results_data = data.get("results", [])
        meta_data = data.get("meta", {})
        
        # Clean and parse each result
        cleaned_results = []
        for result in results_data:
            cleaned = _clean_result(result)
            cleaned_results.append(ProductLabelItem(**cleaned))
        
    except Exception as e:
        logging.error(f"Error parsing OpenFDA Drug Label response: {e}")
        return pd.DataFrame(), f"# Drug Product Labeling Search Results\n\nError: Failed to parse response: {e}"
    
    # Prepare results for display
    results_data = []
    for result in cleaned_results:
        result_dict = {
            "id": result.id,
            "set_id": result.set_id,
            "effective_time": result.effective_time,
            "version": result.version,
        }
        
        # Add OpenFDA info if available
        if result.openfda:
            result_dict["brand_name"] = ", ".join(result.openfda.brand_name[:3]) if result.openfda.brand_name else ""
            result_dict["generic_name"] = ", ".join(result.openfda.generic_name[:3]) if result.openfda.generic_name else ""
            result_dict["manufacturer"] = ", ".join(result.openfda.manufacturer_name[:2]) if result.openfda.manufacturer_name else ""
            result_dict["product_type"] = ", ".join(result.openfda.product_type[:2]) if result.openfda.product_type else ""
            result_dict["route"] = ", ".join(result.openfda.route[:3]) if result.openfda.route else ""
            result_dict["substance_name"] = ", ".join(result.openfda.substance_name[:3]) if result.openfda.substance_name else ""
        
        # Add brief snippets of label sections
        if result.indications_and_usage:
            snippet = result.indications_and_usage[0][:200] if result.indications_and_usage[0] else ""
            result_dict["indications_snippet"] = snippet + "..." if len(snippet) == 200 else snippet
        
        results_data.append(result_dict)
    
    # Generate summary
    total_results = meta_data.get("results", {}).get("total", len(results_data))
    summary = f"# Drug Product Labeling Search Results\n\nFound {total_results} labels"
    
    if skip > 0:
        summary += f" (showing results {skip + 1}-{skip + len(results_data)})"
    
    summary += "\n\n"
    
    # Prepare flattened data for CSV
    flattened_data = []
    for result in cleaned_results:
        row = {
            "id": result.id,
            "set_id": result.set_id,
            "effective_time": result.effective_time,
            "version": result.version,
        }
        
        # Add OpenFDA fields
        if result.openfda:
            row["brand_name"] = "; ".join(result.openfda.brand_name) if result.openfda.brand_name else ""
            row["generic_name"] = "; ".join(result.openfda.generic_name) if result.openfda.generic_name else ""
            row["manufacturer_name"] = "; ".join(result.openfda.manufacturer_name) if result.openfda.manufacturer_name else ""
            row["product_type"] = "; ".join(result.openfda.product_type) if result.openfda.product_type else ""
            row["route"] = "; ".join(result.openfda.route) if result.openfda.route else ""
            row["substance_name"] = "; ".join(result.openfda.substance_name) if result.openfda.substance_name else ""
            row["application_number"] = "; ".join(result.openfda.application_number) if result.openfda.application_number else ""
            row["package_ndc"] = "; ".join(result.openfda.package_ndc) if result.openfda.package_ndc else ""
        else:
            row.update({
                "brand_name": "",
                "generic_name": "",
                "manufacturer_name": "",
                "product_type": "",
                "route": "",
                "substance_name": "",
                "application_number": "",
                "package_ndc": "",
            })
        
        # Add label sections (join multiple paragraphs)
        row["indications_and_usage"] = " ".join(result.indications_and_usage) if result.indications_and_usage else ""
        row["dosage_and_administration"] = " ".join(result.dosage_and_administration) if result.dosage_and_administration else ""
        row["contraindications"] = " ".join(result.contraindications) if result.contraindications else ""
        row["warnings"] = " ".join(result.warnings) if result.warnings else ""
        row["adverse_reactions"] = " ".join(result.adverse_reactions) if result.adverse_reactions else ""
        row["drug_interactions"] = " ".join(result.drug_interactions) if result.drug_interactions else ""
        row["description"] = " ".join(result.description) if result.description else ""
        row["clinical_pharmacology"] = " ".join(result.clinical_pharmacology) if result.clinical_pharmacology else ""
        row["how_supplied"] = " ".join(result.how_supplied) if result.how_supplied else ""
        
        flattened_data.append(row)
    
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
        try:
            if not save_path.endswith(".csv"):
                save_path = save_path + ".csv"
            
            if flattened_data:
                output_dataframe.to_csv(save_path, index=False)
                save_msg = f"Results saved to {save_path}"
            else:
                save_msg = "No results to save"
            
            output += f"\n\n---\n{save_msg}"
            
        except Exception as e:
            logging.error(f"Error saving results to {save_path}: {e}")
            output += f"\n\n---\nError saving results to {save_path}: {e}"
    
    return output_dataframe, output


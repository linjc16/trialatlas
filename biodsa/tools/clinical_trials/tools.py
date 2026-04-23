from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import pandas as pd

from biodsa.tools.clinical_trials.trials import (
    search_trials,
    fetch_trial_details_by_ids,
    RecruitingStatus,
    StudyType,
    SponsorType,
    TrialPhase,
    AgeGroup,
    PrimaryPurpose,
    InterventionType,
    SortOrder,
    DateField,
    LineOfTherapy,
)
from biodsa.sandbox.sandbox_interface import ExecutionSandboxWrapper

__all__ = [
    "SearchTrialsTool",
    "FetchTrialDetailsTool",
    "SearchTrialsToolInput",
    "FetchTrialDetailsToolInput",
]


# =====================================================
# Tool 1: Search Clinical Trials
# =====================================================
class SearchTrialsToolInput(BaseModel):
    """Input schema for SearchTrialsTool."""
    
    conditions: Optional[List[str]] = Field(
        default=None,
        description="List of medical conditions or diseases to search for (e.g., ['breast cancer', 'diabetes'])"
    )
    terms: Optional[List[str]] = Field(
        default=None,
        description="General search terms or keywords"
    )
    interventions: Optional[List[str]] = Field(
        default=None,
        description="Intervention or drug names (e.g., ['pembrolizumab', 'chemotherapy'])"
    )
    recruiting_status: Optional[str] = Field(
        default=None,
        description="Recruitment status: 'OPEN', 'CLOSED', or 'ANY'"
    )
    study_type: Optional[str] = Field(
        default=None,
        description="Type of study: 'INTERVENTIONAL', 'OBSERVATIONAL', or 'EXPANDED_ACCESS'"
    )
    sponsor: Optional[str] = Field(
        default=None,
        description="Sponsor name of the trial"
    )
    sponsor_type: Optional[str] = Field(
        default=None,
        description="Sponsor type: 'INDUSTRY', 'FED', 'NETWORK', 'NIH', 'OTHER', 'OTHER_GOV', 'INDIV'"
    )
    nct_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific NCT IDs to search for (e.g., ['NCT04567890'])"
    )
    lat: Optional[float] = Field(
        default=None,
        description="Latitude for location-based search"
    )
    long: Optional[float] = Field(
        default=None,
        description="Longitude for location-based search"
    )
    distance: Optional[float] = Field(
        default=None,
        description="Distance in miles from lat/long (default: 50 miles if location provided)"
    )
    min_date: Optional[str] = Field(
        default=None,
        description="Minimum date filter (YYYY-MM-DD)"
    )
    max_date: Optional[str] = Field(
        default=None,
        description="Maximum date filter (YYYY-MM-DD)"
    )
    date_field: Optional[str] = Field(
        default=None,
        description="Date field to filter: 'POSTED', 'UPDATE', 'START', 'PRIMARY_COMPLETION', 'COMPLETION'"
    )
    phase: Optional[str] = Field(
        default=None,
        description="Trial phase: 'EARLY_PHASE1', 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4', 'NOT_APPLICABLE'"
    )
    age_group: Optional[str] = Field(
        default=None,
        description="Age group: 'CHILD', 'ADULT', 'OLDER_ADULT'"
    )
    primary_purpose: Optional[str] = Field(
        default=None,
        description="Primary purpose: 'TREATMENT', 'PREVENTION', 'DIAGNOSTIC', 'SUPPORTIVE_CARE', etc."
    )
    intervention_type: Optional[str] = Field(
        default=None,
        description="Intervention type: 'DRUG', 'DEVICE', 'BIOLOGICAL', 'PROCEDURE', etc."
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort order: 'RELEVANCE', 'LAST_UPDATE', 'ENROLLMENT', 'START_DATE', etc."
    )
    next_page_hash: Optional[str] = Field(
        default=None,
        description="Token for retrieving the next page of results"
    )
    prior_therapies: Optional[List[str]] = Field(
        default=None,
        description="Prior therapies to search in eligibility criteria"
    )
    progression_on: Optional[List[str]] = Field(
        default=None,
        description="Therapies the patient has progressed on"
    )
    required_mutations: Optional[List[str]] = Field(
        default=None,
        description="Required genetic mutations in eligibility criteria"
    )
    excluded_mutations: Optional[List[str]] = Field(
        default=None,
        description="Excluded genetic mutations in eligibility criteria"
    )
    biomarker_expression: Optional[Dict[str, str]] = Field(
        default=None,
        description="Biomarker expression requirements (e.g., {'PD-L1': '≥50%'})"
    )
    line_of_therapy: Optional[str] = Field(
        default=None,
        description="Line of therapy: '1L' (first-line), '2L' (second-line), '3L+' (third-line or later)"
    )
    allow_brain_mets: Optional[bool] = Field(
        default=None,
        description="Whether to include trials that accept brain metastases"
    )
    page_size: Optional[int] = Field(
        default=None,
        description="Number of results per page (1-1000)"
    )
    expand_synonyms: bool = Field(
        default=True,
        description="Expand condition searches with disease synonyms"
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Path to save the search results"
    )


class SearchTrialsTool(BaseTool):
    """
    Tool to search for clinical trials on ClinicalTrials.gov.
    
    This comprehensive search tool allows filtering by:
    - Medical conditions and diseases
    - Interventions and drugs
    - Trial phase, status, and type
    - Geographic location
    - Patient eligibility criteria (biomarkers, mutations, prior therapies)
    - Sponsor information
    - Dates and enrollment
    
    Returns detailed trial information including title, status, conditions, interventions,
    eligibility criteria, locations, and more.
    """
    
    name: str = "search_clinical_trials"
    description: str = (
        "Search for clinical trials on ClinicalTrials.gov with comprehensive filtering options. "
        "Filter by conditions (diseases), interventions (drugs), trial phase, recruitment status, "
        "location, patient eligibility (mutations, biomarkers, prior therapies), and more. "
        "Returns trial metadata including NCT ID, title, status, phase, conditions, interventions, "
        "eligibility criteria, study design, locations, and contact information. "
        "CRITICAL USE: When looking for clinical trial options for specific diseases, patient populations, "
        "or investigating trial designs and outcomes."
    )
    args_schema: Type[BaseModel] = SearchTrialsToolInput
    sandbox: ExecutionSandboxWrapper = None
    
    def __init__(self, sandbox: ExecutionSandboxWrapper = None):
        super().__init__()
        self.sandbox = sandbox
    
    def _run(
        self,
        conditions: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
        interventions: Optional[List[str]] = None,
        recruiting_status: Optional[str] = None,
        study_type: Optional[str] = None,
        sponsor: Optional[str] = None,
        sponsor_type: Optional[str] = None,
        nct_ids: Optional[List[str]] = None,
        lat: Optional[float] = None,
        long: Optional[float] = None,
        distance: Optional[float] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        date_field: Optional[str] = None,
        phase: Optional[str] = None,
        age_group: Optional[str] = None,
        primary_purpose: Optional[str] = None,
        intervention_type: Optional[str] = None,
        sort: Optional[str] = None,
        next_page_hash: Optional[str] = None,
        prior_therapies: Optional[List[str]] = None,
        progression_on: Optional[List[str]] = None,
        required_mutations: Optional[List[str]] = None,
        excluded_mutations: Optional[List[str]] = None,
        biomarker_expression: Optional[Dict[str, str]] = None,
        line_of_therapy: Optional[str] = None,
        allow_brain_mets: Optional[bool] = None,
        page_size: Optional[int] = None,
        expand_synonyms: bool = True,
        save_path: Optional[str] = None,
    ) -> str:
        """Execute the tool to search clinical trials."""

        # Sanitize single-value enum fields: take only the first value if
        # the model passes a comma-separated string (e.g. "PHASE1,PHASE2")
        if phase and ',' in phase:
            phase = phase.split(',')[0].strip()
        if intervention_type and ',' in intervention_type:
            intervention_type = intervention_type.split(',')[0].strip()
        if recruiting_status and ',' in recruiting_status:
            recruiting_status = recruiting_status.split(',')[0].strip()

        # Call the search_trials function
        df, md_str, total_count, next_page_token = search_trials(
            conditions=conditions,
            terms=terms,
            interventions=interventions,
            recruiting_status=recruiting_status,
            study_type=study_type,
            sponsor=sponsor,
            sponsor_type=sponsor_type,
            nct_ids=nct_ids,
            lat=lat,
            long=long,
            distance=distance,
            min_date=min_date,
            max_date=max_date,
            date_field=date_field,
            phase=phase,
            age_group=age_group,
            primary_purpose=primary_purpose,
            intervention_type=intervention_type,
            sort=sort,
            next_page_hash=next_page_hash,
            prior_therapies=prior_therapies,
            progression_on=progression_on,
            required_mutations=required_mutations,
            excluded_mutations=excluded_mutations,
            biomarker_expression=biomarker_expression,
            line_of_therapy=line_of_therapy,
            allow_brain_mets=allow_brain_mets,
            page_size=page_size,
            expand_synonyms=expand_synonyms,
            save_path=save_path,
        )
        
        # TODO: Optionally upload DataFrame to sandbox if needed
        # if self.sandbox and df is not None:
        #     self.sandbox.upload_dataframe(df, "search_results.csv")
        
        if df is None or len(df) == 0:
            return "No clinical trials found matching the search criteria."
        
        # Add pagination info if there's a next page
        result = md_str
        if next_page_token:
            result += f"\n\n---\n**Pagination:** Use `next_page_hash='{next_page_token}'` to retrieve the next page of results."
        
        return result


# =====================================================
# Tool 2: Fetch Trial Details by IDs
# =====================================================
class FetchTrialDetailsToolInput(BaseModel):
    """Input schema for FetchTrialDetailsTool."""
    
    trial_ids: List[str] = Field(
        ...,
        description="List of NCT IDs to fetch detailed information for (e.g., ['NCT04567890', 'NCT03456789'])"
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the results as CSV"
    )


class FetchTrialDetailsTool(BaseTool):
    """
    Tool to fetch detailed information for specific clinical trials by their NCT IDs.
    
    This tool retrieves comprehensive information about clinical trials including:
    - Study design and methodology
    - Detailed eligibility criteria
    - Outcome measures and endpoints
    - Contact and location information
    - Study results and publications (if available)
    
    Use this when you have specific NCT IDs and need complete trial details.
    """
    
    name: str = "fetch_trial_details"
    description: str = (
        "Fetch detailed information for specific clinical trials using their NCT IDs. "
        "Returns comprehensive trial data including full study design, detailed eligibility criteria, "
        "outcome measures, contacts, locations, arms/groups, and published results if available. "
        "CRITICAL USE: When you have identified relevant trials (by NCT ID) and need complete details "
        "for analysis, comparison, or extracting specific protocol information."
    )
    args_schema: Type[BaseModel] = FetchTrialDetailsToolInput
    sandbox: ExecutionSandboxWrapper = None
    
    def __init__(self, sandbox: ExecutionSandboxWrapper = None):
        super().__init__()
        self.sandbox = sandbox
    
    def _run(
        self,
        trial_ids: List[str],
        save_path: Optional[str] = None,
    ) -> str:
        """Execute the tool to fetch trial details."""
        
        if not trial_ids or len(trial_ids) == 0:
            return "Error: No trial IDs provided. Please provide at least one NCT ID."
        
        # Call the fetch function
        df, md_str, total_count, next_page_token = fetch_trial_details_by_ids(
            trial_ids=trial_ids,
            save_path=save_path,
        )
        
        # TODO: Optionally upload DataFrame to sandbox if needed
        # if self.sandbox and df is not None:
        #     self.sandbox.upload_dataframe(df, "trial_details.csv")
        
        if df is None or len(df) == 0:
            return f"No details found for the provided trial IDs: {', '.join(trial_ids)}"
        
        return md_str

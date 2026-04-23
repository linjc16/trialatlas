"""
This file contains the schema for the trial atlas agent and its sub-agents.
"""

from typing import List, Type, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class GoDrugBenchmarkingInput(BaseModel):
    task_description: str = Field(
        ...,
        description="The task description for the drug benchmarking."
    )

class GoDrugBenchmarkingTool(BaseTool):
    name: str = "drug_benchmarking"
    description: str = "Call this tool to start a round of drug benchmarking."
    args_schema: Type[BaseModel] = GoDrugBenchmarkingInput

    def _run(self, task_description: str) -> str:
        """
        Start a round of drug benchmarking.
        """
        return f"Drug benchmarking started."

class GoCompetingTrialAnalysisInput(BaseModel):
    task_description: str = Field(
        ...,
        description="The task description for the competing trial analysis."
    )

class GoCompetingTrialAnalysisTool(BaseTool):
    name: str = "competing_trial_analysis"
    description: str = "Call this tool to start a round of competing trial analysis."
    args_schema: Type[BaseModel] = GoCompetingTrialAnalysisInput

    def _run(self, task_description: str) -> str:
        """
        Start a round of competing trial analysis.
        """
        return f"Competing trial analysis started."

class GoFeasibilityAnalysisInput(BaseModel):

    task_description: str = Field(
        ...,
        description="The task description for the feasibility analysis."
    )

class GoFeasibilityAnalysisTool(BaseTool):
    name: str = "feasibility_analysis"
    description: str = "Call this tool to start a round of feasibility analysis."
    args_schema: Type[BaseModel] = GoFeasibilityAnalysisInput

    def _run(self, task_description: str) -> str:
        """
        Start a round of feasibility analysis.
        """
        return f"Feasibility analysis started."

class GoLiteratureEvidenceInput(BaseModel):
    task_description: str = Field(
        ...,
        description="The task description for the literature evidence."
    )

class GoLiteratureEvidenceTool(BaseTool):   
    name: str = "literature_evidence"
    description: str = "Call this tool to start a round of literature evidence."
    args_schema: Type[BaseModel] = GoLiteratureEvidenceInput

    def _run(self, task_description: str) -> str:
        """
        Start a round of literature evidence.
        """
        return f"Literature evidence started."

class GoTrialOutcomePredictionInput(BaseModel):
    task_description: str = Field(
        ...,
        description="The task description for the trial outcome prediction."
    )

class GoTrialOutcomePredictorTool(BaseTool):
    name: str = "trial_outcome_prediction"
    description: str = "Call this tool to start a round of trial outcome prediction."
    args_schema: Type[BaseModel] = GoTrialOutcomePredictionInput

    def _run(self, task_description: str) -> str:
        return f"Trial outcome prediction started."

AVAILABLE_SUB_AGENT_TOOL_MAPPING = {
    "drug_benchmarking": GoDrugBenchmarkingTool,
    "competing_trial_analysis": GoCompetingTrialAnalysisTool,
    # "feasibility_analysis": GoFeasibilityAnalysisTool,
    "literature_evidence": GoLiteratureEvidenceTool,
    "trial_outcome_prediction": GoTrialOutcomePredictorTool,
}


def get_module_tool(module: str, module_kwargs: Dict[str, Any] = {}) -> BaseTool:
    module_cls = AVAILABLE_SUB_AGENT_TOOL_MAPPING.get(module)
    if module_cls is None:
        raise ValueError(f"Module {module} not found. Must be one of: {AVAILABLE_SUB_AGENT_TOOL_MAPPING.keys()}")
    return module_cls(**module_kwargs)

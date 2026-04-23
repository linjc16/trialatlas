from pydantic import BaseModel
from typing import List, Annotated, Sequence
from langgraph.graph.message import add_messages, BaseMessage

from biodsa.agents.state import CodeExecutionResult

class TrialAtlasAgentState(BaseModel):
    """State for the trial atlas agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class DrugBenchmarkingAgentState(BaseModel):
    """State for the drug benchmarking agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class CompetingTrialAnalysisAgentState(BaseModel):
    """State for the competing trial analysis agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class FeasibilityAnalysisAgentState(BaseModel):
    """State for the feasibility analysis agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class LiteratureEvidenceAgentState(BaseModel):
    """State for the literature evidence agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class TrialOutcomePredictorAgentState(BaseModel):
    """State for the trial outcome predictor agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_execution_results: List[CodeExecutionResult] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
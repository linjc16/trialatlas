from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

from biodsa.tools.code_exec_tool import CodeExecutionTool
from biodsa.agents.trialatlas.state import (
    FeasibilityAnalysisAgentState,
    TrialAtlasAgentState
)
from biodsa.agents.trialatlas.subagents.base_subagent import BaseReActAgent

FEASIBILITY_ANALYSIS_AGENT_SYSTEM_PROMPT = """
You are a helpful assistant that helps with feasibility analysis by assessing the feasibility of the trial design.
"""

class FeasibilityAnalysisAgent(BaseReActAgent):
    name = "feasibility_analysis_agent"
    tool_name = "feasibility_analysis"
    state_class = FeasibilityAnalysisAgentState
    parent_state_class = TrialAtlasAgentState
    
    def _build_system_prompt(self) -> str:
        return FEASIBILITY_ANALYSIS_AGENT_SYSTEM_PROMPT

    def _get_tools(self) -> List[BaseTool]:
        return [
            CodeExecutionTool(sandbox=self.sandbox)
        ]

    def _prepare_subagent_inputs(self, tool_call_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the input state for the subagent from tool call arguments.
        
        Default implementation expects a 'task_description' argument and creates
        a simple message-based input. Override this method if your agent needs
        different arguments or a different input structure.
        
        Args:
            tool_call_args: The arguments from the parent graph's tool call
            
        Returns:
            A dictionary containing the initial state for the subagent
            (typically includes 'messages' key)
        """
        task_description = tool_call_args.get("task_description")
        if task_description is None:
            raise ValueError(
                f"Tool call for '{self.tool_name}' must have 'task_description' argument. "
                f"Override _prepare_subagent_inputs() if you need different arguments."
            )
        
        return {
            "messages": [HumanMessage(content=task_description)],
        }
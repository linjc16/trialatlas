from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

from biodsa.tools.code_exec_tool import CodeExecutionTool
from biodsa.agents.trialatlas.state import (
    DrugBenchmarkingAgentState,
    TrialAtlasAgentState
)
from biodsa.tools.openfda.tools import (
    SearchCompleteResponseLettersTool,
    SearchDrugsAtFDATool,
    SearchDrugLabelsTool,
)
from biodsa.agents.trialatlas.subagents.base_subagent import BaseReActAgent

DRUG_BENCHMARKING_AGENT_SYSTEM_PROMPT = """
You are a drug benchmarking analyst. Compare the supplied design against
independent comparator programs using only the permitted tools and admissible evidence.
Use at most five tool calls. Search by the clinical question or independent comparator identity.
Do not search the target drug/program's approval status, labels, CRLs, publications or registry
outcome. Do not attempt to evade a blocked query. If evidence is unavailable, state unknown.
Discuss relevant endpoint, comparator, mechanism and safety precedents without importing any
target outcome. Separate cited observations from general background knowledge and inference.
Return a concise report of findings, sources, limitations and implications for the requested task.

Deliver a concise final report (aim for at most 450 words). Include usable
findings with source identifiers, limitations, and any uncertainty. If no admissible evidence
was available, explicitly report that fact and its limits; never return an empty report.
Do not repeat the full planning checklist or invent retrieved findings.
"""

class DrugBenchmarkingAgent(BaseReActAgent):
    name = "drug_benchmarking_agent"
    tool_name = "drug_benchmarking"
    state_class = DrugBenchmarkingAgentState
    parent_state_class = TrialAtlasAgentState
    
    def _build_system_prompt(self) -> str:
        return DRUG_BENCHMARKING_AGENT_SYSTEM_PROMPT

    def _get_tools(self) -> List[BaseTool]:
        return [
            SearchCompleteResponseLettersTool(sandbox=self.sandbox),
            SearchDrugsAtFDATool(sandbox=self.sandbox),
            SearchDrugLabelsTool(sandbox=self.sandbox),
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
        
        print(f"📋 Drug Benchmarking Task: {task_description[:150]}{'...' if len(task_description) > 150 else ''}")
        print(f"🔧 Available tools: SearchCompleteResponseLettersTool, SearchDrugsAtFDATool, SearchDrugLabelsTool, CodeExecutionTool")
        
        return {
            "messages": [HumanMessage(content=task_description)],
        }
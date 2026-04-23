from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from biodsa.tools.code_exec_tool import CodeExecutionTool
from biodsa.agents.trialatlas.state import (
    TrialAtlasAgentState,
    TrialOutcomePredictorAgentState
)
from biodsa.agents.trialatlas.subagents.base_subagent import BaseReActAgent

TRIAL_OUTCOME_PREDICTOR_AGENT_SYSTEM_PROMPT = """
You are a helpful assistant that helps with trial outcome prediction.
"""

class TrialOutcomePredictorAgent(BaseReActAgent):
    name = "trial_outcome_predictor_agent"
    tool_name = "trial_outcome_prediction"
    state_class = TrialOutcomePredictorAgentState
    parent_state_class = TrialAtlasAgentState
    
    def _build_system_prompt(self) -> str:
        return TRIAL_OUTCOME_PREDICTOR_AGENT_SYSTEM_PROMPT

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


    def invoke_as_a_subagent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Invoke this agent as a subagent within a parent agent graph.
        
        This method:
        1. Extracts the tool call from parent graph that matches this agent's tool_name
        2. Extracts tool call arguments and prepares subagent inputs via _prepare_subagent_inputs()
        3. Runs the subagent with the prepared inputs
        4. Returns the response as a ToolMessage for the parent graph
        5. Tracks and accumulates token usage
        
        To customize how tool call arguments are processed, override _prepare_subagent_inputs().
        
        Note: This method expects the parent state to have:
        - messages: List of messages (last one should contain tool calls)
        - total_input_tokens: Token tracking (optional, defaults to 0)
        - total_output_tokens: Token tracking (optional, defaults to 0)
        """
        # NOTE: This agent currently uses the default invoke_as_a_subagent behavior.
        # When implementing actual trial outcome prediction logic, you may want to:
        # 1. Override _prepare_subagent_inputs() if you need custom input processing
        # 2. Override invoke_as_a_subagent() if you need to bypass the standard flow entirely
        # Access parent state properties - use getattr for flexibility
        parent_messages = getattr(state, 'messages', None)
        if parent_messages is None or len(parent_messages) == 0:
            raise ValueError("Parent state must have 'messages' attribute with at least one message")
        
        parent_graph_message = parent_messages[-1]
        parent_graph_message_tool_calls = parent_graph_message.tool_calls
        
        # Find the tool call that matches this agent's tool name
        subgraph_tool_call_id = None
        tool_call_args = None
        for tool_call in parent_graph_message_tool_calls:
            if tool_call["name"] == self.tool_name:
                subgraph_tool_call_id = tool_call["id"]
                tool_call_args = tool_call.get("args", {})
                break
        
        if subgraph_tool_call_id is None:
            raise ValueError(
                f"No '{self.tool_name}' tool call found in the parent graph message"
            )
        
        # Prepare inputs using the customizable method
        inputs = self._prepare_subagent_inputs(tool_call_args)

        # Run the subagent
        # TODO: Implement the trial outcome prediction logic here
        response = ToolMessage(
            content=f"The input trial design has 50% chance of success.",
            name=self.tool_name,
            tool_call_id=subgraph_tool_call_id
        )

        return {
            "messages": [response],
        }



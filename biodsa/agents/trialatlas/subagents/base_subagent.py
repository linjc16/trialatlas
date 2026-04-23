from abc import abstractmethod
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing import List, Dict, Any, Literal, Type, TypeVar

from biodsa.agents.base_agent import BaseAgent

# Generic type variables for react agent states
TAgentState = TypeVar('TAgentState')
TParentState = TypeVar('TParentState')


class BaseReActAgent(BaseAgent):
    """
    Abstract base class for ReAct-style agents.
    
    This class provides common functionality for ReAct-style sub-agents,
    including graph creation, tool execution, and token tracking.
    
    Subclasses must define:
    - tool_name: Class attribute for the tool name used in parent graph
    - state_class: Class attribute for this agent's internal state type
    - parent_state_class: Class attribute for the parent agent's state type
    - _build_system_prompt(): Return the system prompt for the agent
    - _get_tools(): Return the list of tools available to the agent
    """
    
    # Subclasses must define these class attributes
    tool_name: str = None
    state_class: Type = None
    parent_state_class: Type = None
    
    def __init__(self,
        model_name: str,
        api_type: str,
        api_key: str,
        endpoint: str = None,
        container_id: str = None,
        model_kwargs: Dict[str, Any] = None,    
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_type=api_type,
            api_key=api_key,
            endpoint=endpoint,
            container_id=container_id,
            model_kwargs=model_kwargs,
        )
        if self.tool_name is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define 'tool_name' class attribute")
        if self.state_class is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define 'state_class' class attribute")
        if self.parent_state_class is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define 'parent_state_class' class attribute")
        
        self.agent_graph = self._create_agent_graph()

    def _create_agent_graph(self) -> StateGraph:
        """Creates the standard agent graph with agent_node and tool_node."""
        workflow = StateGraph(
            self.state_class,
            input=self.state_class,
            output=self.state_class
        )
        workflow.add_node("agent_node", self._agent_node)
        workflow.add_node("tool_node", self._tool_node)
        workflow.add_conditional_edges(
            "agent_node",
            self._should_continue_agent,
            {
                "tool_node": "tool_node",
                "end": END
            }
        )
        workflow.add_edge("tool_node", "agent_node")
        workflow.set_entry_point("agent_node")
        workflow = workflow.compile(name=self.name)
        return workflow

    def _agent_node(self, state: TAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Standard agent node that calls the model with system prompt and tools."""
        print("-" * 100)
        print(f"🤖 {self.name} - Agent Node Processing...")
        print("-" * 100)
        
        messages = state.messages
        system_prompt = self._build_system_prompt()
        tools = self._get_tools()
        
        print(f"📋 Available tools: {[tool.name for tool in tools]}")
        
        messages = [
            SystemMessage(content=system_prompt),
        ] + messages
        response = self._call_model(
            model_name=self.model_name,
            messages=messages,
            tools=tools,
            parallel_tool_calls=False,
        )
        
        # Print tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔨 Model requested {len(response.tool_calls)} tool call(s)")
            for tool_call in response.tool_calls:
                print(f"   - {tool_call['name']}")
        
        return {
            "messages": [response],
        }
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        Build and return the system prompt for this agent.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _get_tools(self) -> List[BaseTool]:
        """
        Return the list of tools available to this agent.
        
        Must be implemented by subclasses.
        """
        pass

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

    def _tool_node(self, state: TAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Standard tool node that executes all tool calls from the last message."""
        messages = state.messages
        all_tool_calls = messages[-1].tool_calls
        responses = []
        
        # Create a mapping of tool names to tool instances for efficient lookup
        tools_map = {tool.name: tool for tool in self._get_tools()}
        
        for tool_call in all_tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            print("-" * 100)
            print(f"🔧 {self.name} - Executing tool: {tool_name}")
            if tool_name == "code_execution":
                # Truncate code for display if too long
                code_preview = tool_input.get("code", "")[:200]
                if len(tool_input.get("code", "")) > 200:
                    code_preview += "..."
                print(f"   Code preview: {code_preview}")
            else:
                # Show input preview (truncate if too long)
                input_preview = str(tool_input)[:200]
                if len(str(tool_input)) > 200:
                    input_preview += "..."
                print(f"   Input: {input_preview}")
            print("-" * 100)
            
            tool = tools_map.get(tool_name)
            if tool is None:
                error_msg = f"Tool '{tool_name}' not found in available tools: {list(tools_map.keys())}"
                print(f"❌ Error: {error_msg}")
                raise ValueError(error_msg)

            try:
                tool_output = tool._run(**tool_input)
            except TypeError as e:
                tool_output = {"error": str(e), "stdout": f"Tool call error: {e}. Please retry with correct arguments.", "exit_code": 1}
            
            # Print output summary
            if tool_name == "code_execution" and isinstance(tool_output, dict):
                stdout_preview = tool_output.get("stdout", "")[:300]
                if len(tool_output.get("stdout", "")) > 300:
                    stdout_preview += "..."
                print(f"✅ Tool execution completed")
                print(f"   Output preview: {stdout_preview}")
            else:
                output_preview = str(tool_output)[:300]
                if len(str(tool_output)) > 300:
                    output_preview += "..."
                print(f"✅ Tool execution completed")
                print(f"   Output preview: {output_preview}")
            
            response = ToolMessage(
                content=tool_output,
                name=tool_name,
                tool_call_id=tool_call["id"]
            )   
            responses.append(response)
        return {
            "messages": responses,
        }

    def _should_continue_agent(self, state: TAgentState) -> Literal["tool_node", "end"]:
        """Determines whether to continue to tool execution or end."""
        messages = state.messages
        last_message = messages[-1]
        tool_calls = getattr(last_message, 'tool_calls', None)
        decision = "end"
        if tool_calls is not None:
            if len(tool_calls) > 0:
                decision = "tool_node"
                print(f"📊 {self.name} - Routing to: Tool Node ({len(tool_calls)} tool call(s))")
            else:
                print(f"✅ {self.name} - Routing to: END (no tool calls)")
        else:
            print(f"✅ {self.name} - Routing to: END (no tool calls)")
        return decision

    def invoke_as_a_subagent(self, state: TParentState, config: RunnableConfig) -> Dict[str, Any]:
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
        
        # Print subagent invocation info
        task_description = tool_call_args.get("task_description", "")
        task_preview = task_description[:200] + "..." if len(task_description) > 200 else task_description
        print("=" * 100)
        print(f"🚀 Starting {self.name}")
        print(f"   Task: {task_preview}")
        print("=" * 100)

        # Run the subagent
        agent_outputs = self.agent_graph.invoke(inputs, config=config)
        all_messages = agent_outputs['messages']
        final_response = all_messages[-1].content
        
        print("=" * 100)
        print(f"✅ {self.name} completed")
        response_preview = final_response[:200] + "..." if len(final_response) > 200 else final_response
        print(f"   Response preview: {response_preview}")
        print("=" * 100)

        # Create response as ToolMessage for parent graph
        response = ToolMessage(
            content=final_response,
            name=self.tool_name,
            tool_call_id=subgraph_tool_call_id
        )

        # Track token usage (with fallback to 0 if not present)
        agent_total_input_tokens = agent_outputs.get('total_input_tokens', 0)
        agent_total_output_tokens = agent_outputs.get('total_output_tokens', 0)
        parent_input_tokens = getattr(state, 'total_input_tokens', 0)
        parent_output_tokens = getattr(state, 'total_output_tokens', 0)
        total_input_tokens = parent_input_tokens + agent_total_input_tokens
        total_output_tokens = parent_output_tokens + agent_total_output_tokens

        return {
            "messages": [response],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }


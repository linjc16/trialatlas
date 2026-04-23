"""
Proposed by:

TBD. TrialAtlas: Building AI Agents Team for Clinical Trial Design and Optimization. In submission.
"""

from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from biodsa.utils.render_utils import render_message_colored
from biodsa.agents.trialatlas.execution import TrialAtlasExecutionResults
from biodsa.agents.base_agent import BaseAgent
from biodsa.tools.code_exec_tool import CodeExecutionTool
from biodsa.agents.trialatlas.state import (
    TrialAtlasAgentState, 
)
from biodsa.agents.trialatlas.prompt import TRIAL_ATLAS_AGENT_SYSTEM_PROMPT
from biodsa.agents.trialatlas.prompt import get_module_prompt_guidance

# sub-agents
from biodsa.agents.trialatlas.subagents import (
    DrugBenchmarkingAgent,
    CompetingTrialAnalysisAgent,
    FeasibilityAnalysisAgent,
    LiteratureEvidenceAgent,
    TrialOutcomePredictorAgent,
)
from biodsa.agents.trialatlas.schema import get_module_tool

# main agent
class TrialAtlasAgent(BaseAgent):
    name = "trialatlas"
    def __init__(
        self,
        model_name: str,
        api_type: str,
        api_key: str,
        endpoint: str = None,
        container_id: str = None,
        model_kwargs: Dict[str, Any] = None,
        small_model_name: str = None,
        small_model_kwargs: Dict[str, Any] = None,
        small_model_api_type: str = None,
        small_model_api_key: str = None,
        small_model_endpoint: str = None,
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

        # create the sub-agents
        sub_agent_model_kwargs = {
            "model_name": small_model_name,
            "api_type": small_model_api_type,
            "api_key": small_model_api_key,
            "endpoint": small_model_endpoint,
            "model_kwargs": small_model_kwargs,
            "container_id": container_id,
        }
        self.drug_benchmarking_agent = DrugBenchmarkingAgent(**sub_agent_model_kwargs)
        self.competing_trial_analysis_agent = CompetingTrialAnalysisAgent(**sub_agent_model_kwargs)
        # self.feasibility_analysis_agent = FeasibilityAnalysisAgent(**sub_agent_model_kwargs)
        self.literature_evidence_agent = LiteratureEvidenceAgent(**sub_agent_model_kwargs)
        self.trial_outcome_predictor_agent = TrialOutcomePredictorAgent(**sub_agent_model_kwargs)

        # create the main agent graph
        self.agent_graph = self._create_agent_graph()

    def _call_drug_benchmarking_agent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
        print("-" * 100)
        print("🔍 Calling Drug Benchmarking Agent...")
        print("-" * 100)
        return self.drug_benchmarking_agent.invoke_as_a_subagent(state, config)

    def _call_competing_trial_analysis_agent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
        print("-" * 100)
        print("🔍 Calling Competing Trial Analysis Agent...")
        print("-" * 100)
        return self.competing_trial_analysis_agent.invoke_as_a_subagent(state, config)

    # def _call_feasibility_analysis_agent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
    #     return self.feasibility_analysis_agent.invoke_as_a_subagent(state, config)

    def _call_literature_evidence_agent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
        print("-" * 100)
        print("🔍 Calling Literature Evidence Agent...")
        print("-" * 100)
        return self.literature_evidence_agent.invoke_as_a_subagent(state, config)

    def _call_trial_outcome_predictor_agent(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
        print("-" * 100)
        print("🔍 Calling Trial Outcome Predictor Agent...")
        print("-" * 100)
        return self.trial_outcome_predictor_agent.invoke_as_a_subagent(state, config)

    def _should_go_which_sub_agent(self, state: TrialAtlasAgentState) -> Literal["drug_benchmarking_agent", "competing_trial_analysis_agent", "feasibility_analysis_agent", "literature_evidence_agent", "trial_outcome_predictor_agent", "end"]:
        last_message = state.messages[-1]
        tool_calls = last_message.tool_calls
        if tool_calls is not None:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                if tool_name == "drug_benchmarking":
                    print(f"📊 Routing to: Drug Benchmarking Agent")
                    return "drug_benchmarking_agent"
                elif tool_name == "competing_trial_analysis":
                    print(f"📊 Routing to: Competing Trial Analysis Agent")
                    return "competing_trial_analysis_agent"
                # elif tool_name == "feasibility_analysis":
                #     print(f"📊 Routing to: Feasibility Analysis Agent")
                #     return "feasibility_analysis_agent"
                elif tool_name == "literature_evidence":
                    print(f"📊 Routing to: Literature Evidence Agent")
                    return "literature_evidence_agent"
                elif tool_name == "trial_outcome_prediction":
                    print(f"📊 Routing to: Trial Outcome Predictor Agent")
                    return "trial_outcome_predictor_agent"
                else:
                    # common tool call, not a sub-agent tool call
                    print(f"🔧 Routing to: Tool Node (tool: {tool_name})")
                    return "tool_node"
        print("✅ Routing to: END (no more tool calls)")
        return "end"

    def _orchestrator_agent_node(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:        
        print("-" * 100)
        print("🎯 Orchestrator Agent Node - Processing...")
        print("-" * 100)
        
        # get the common tools
        tools = self._get_tools()
        # get the module tools
        module_tools = self._get_modules_as_tools(config.get("configurable", {}).get("modules", []))
        tools.extend(module_tools)
        
        if module_tools:
            print(f"📦 Available module tools: {[tool.name for tool in module_tools]}")

        # build the system prompt
        messages = state.messages
        system_prompt = self._build_system_prompt(config)
        messages = [
            SystemMessage(content=system_prompt),
        ] + messages
        response = self._call_model(
            model_name=self.model_name,
            messages=messages,
            tools=tools,
            parallel_tool_calls=False,
        )

        # track the token usage
        total_input_tokens, total_output_tokens = self._get_input_output_tokens(response)
        existing_total_input_tokens = state.total_input_tokens
        existing_total_output_tokens = state.total_output_tokens
        total_input_tokens = existing_total_input_tokens + total_input_tokens
        total_output_tokens = existing_total_output_tokens + total_output_tokens
        
        # Print tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔨 Model requested {len(response.tool_calls)} tool call(s)")
            for tool_call in response.tool_calls:
                print(f"   - {tool_call['name']}")
        
        return {
            "messages": [response],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

    def _build_system_prompt(self, config: RunnableConfig) -> str:
        modules = config.get("configurable", {}).get("modules", [])
        module_prompt_guidances = [get_module_prompt_guidance(module) for module in modules]
        module_prompt_guidance = "\n".join(module_prompt_guidances)
        system_prompt = TRIAL_ATLAS_AGENT_SYSTEM_PROMPT + "\n" + module_prompt_guidance
        return system_prompt

    def _get_tools(self) -> List[BaseTool]:
        tools = [
            CodeExecutionTool(sandbox=self.sandbox)
        ]
        return tools
        
    def _get_modules_as_tools(self, modules: List[str]) -> List[BaseTool]:
        tools = []
        for module in modules:
            target_module_tool = get_module_tool(module)
            if target_module_tool is not None:
                tools.append(target_module_tool)
        return tools


    def _tool_node(self, state: TrialAtlasAgentState, config: RunnableConfig) -> TrialAtlasAgentState:
        """
        A function to execute the tool node for the orchestrator agent.
        """
        messages = state.messages
        all_tool_calls = messages[-1].tool_calls
        responses = []
        
        # Get all available tools (common tools + module tools)
        available_tools = self._get_tools()
        module_tools = self._get_modules_as_tools(config.get("configurable", {}).get("modules", []))
        available_tools.extend(module_tools)
        available_tools_dict = {tool.name: tool for tool in available_tools}
        
        for tool_call in all_tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            print("-" * 100)
            print(f"🔧 Executing tool: {tool_name}")
            if tool_name == "code_execution":
                # Truncate code for display if too long
                code_preview = tool_input.get("code", "")[:200]
                if len(tool_input.get("code", "")) > 200:
                    code_preview += "..."
                print(f"   Code preview: {code_preview}")
            else:
                print(f"   Input: {tool_input}")
            print("-" * 100)
            
            if tool_name not in available_tools_dict:
                error_msg = f"Tool '{tool_name}' not found in available tools: {list(available_tools_dict.keys())}"
                print(f"❌ Error: {error_msg}")
                raise ValueError(error_msg)
            
            called_tool = available_tools_dict[tool_name]
            try:
                tool_output = called_tool._run(**tool_input)
            except TypeError as e:
                tool_output = {"error": str(e), "stdout": f"Tool call error: {e}. Please retry with correct arguments.", "exit_code": 1}
            
            # Print output summary for code execution
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

    def _create_agent_graph(self, debug: bool = False):
        # orchestrator agent workflow (main agent)
        orchestrator_agent_workflow = StateGraph(
            TrialAtlasAgentState,
            input=TrialAtlasAgentState,
            output=TrialAtlasAgentState
        )
        orchestrator_agent_workflow.add_node("orchestrator_agent_node", self._orchestrator_agent_node)
        orchestrator_agent_workflow.add_node("drug_benchmarking_agent", self._call_drug_benchmarking_agent)
        orchestrator_agent_workflow.add_node("competing_trial_analysis_agent", self._call_competing_trial_analysis_agent)
        # orchestrator_agent_workflow.add_node("feasibility_analysis_agent", self._call_feasibility_analysis_agent)
        orchestrator_agent_workflow.add_node("literature_evidence_agent", self._call_literature_evidence_agent)
        orchestrator_agent_workflow.add_node("trial_outcome_predictor_agent", self._call_trial_outcome_predictor_agent)
        orchestrator_agent_workflow.add_node("tool_node", self._tool_node)
        orchestrator_agent_workflow.add_conditional_edges(
            "orchestrator_agent_node",
            self._should_go_which_sub_agent,
            {
                "drug_benchmarking_agent": "drug_benchmarking_agent",
                "competing_trial_analysis_agent": "competing_trial_analysis_agent",
                # "feasibility_analysis_agent": "feasibility_analysis_agent",
                "literature_evidence_agent": "literature_evidence_agent",
                "trial_outcome_predictor_agent": "trial_outcome_predictor_agent",
                "tool_node": "tool_node",
                "end": END
            }
        )
        orchestrator_agent_workflow.add_edge("drug_benchmarking_agent", "orchestrator_agent_node")
        orchestrator_agent_workflow.add_edge("competing_trial_analysis_agent", "orchestrator_agent_node")
        # orchestrator_agent_workflow.add_edge("feasibility_analysis_agent", "orchestrator_agent_node")
        orchestrator_agent_workflow.add_edge("literature_evidence_agent", "orchestrator_agent_node")
        orchestrator_agent_workflow.add_edge("trial_outcome_predictor_agent", "orchestrator_agent_node")
        orchestrator_agent_workflow.add_edge("tool_node", "orchestrator_agent_node")
        orchestrator_agent_workflow.set_entry_point("orchestrator_agent_node")
        orchestrator_agent_workflow = orchestrator_agent_workflow.compile(
            debug=debug,
            name="orchestrator_agent_workflow"
        )
        return orchestrator_agent_workflow

    def generate(self, input_query: str, modules: List[str] = None, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        A function to generate the response for the agent.

        Args:
            input_query: The user query to process
            modules: List of sub-agent modules to make available for the agent. If None, uses all available.
            verbose: Whether to print the verbose output
        Returns:
            List[Dict[str, Any]]: The result from the agent graph or an error dict
        """
        assert self.agent_graph is not None, "Agent graph is not set"

        # Extract input_query from kwargs
        if input_query is None:
            return [{"error": "input_query is required"}]

        # Set default if not provided
        if modules is None:
            modules = ["drug_benchmarking", "competing_trial_analysis", "literature_evidence", "trial_outcome_prediction"]

        try:
            print("=" * 100)
            print(f"🚀 Starting TrialAtlas Agent")
            print(f"   Input query: {input_query[:200]}{'...' if len(input_query) > 200 else ''}")
            print(f"   Modules: {modules}")
            print("=" * 100)
            
            all_results = []
            inputs = {
                "messages": [("user", input_query)],
            }

            # Invoke the agent graph and return the result
            for streamed_chunk in self.agent_graph.stream(
                inputs,
                stream_mode = ["values"],
                subgraphs=True,
                config={
                    "recursion_limit": 100,
                    "configurable": {
                        "modules": modules
                    },
                }
            ):
                chunk = streamed_chunk[-1]
                if verbose:
                    last_message = chunk['messages'][-1]
                    # Use colored rendering for better visualization
                    print(render_message_colored(last_message, show_tool_calls=True))
                all_results.append(chunk)
            
            print("=" * 100)
            print("✅ TrialAtlas Agent execution completed")
            print("=" * 100)
            return all_results

        except Exception as e:
            print("=" * 100)
            print(f"❌ Error streaming response: {e}")
            print("=" * 100)
            raise e

    def go(
        self,
        input_query: str,
        modules: List[str] = None,
        verbose: bool = True,
    ) -> TrialAtlasExecutionResults:
        """
        A function to execute the agent and return the execution results.

        Args:
            input_query: The user query to process
            modules: List of sub-agent modules to make available for the agent. If None, uses all available.
                Must be a subset of: ["drug_benchmarking", "competing_trial_analysis", "feasibility_analysis", "literature_evidence", "trial_outcome_prediction"]
                If None, uses all available.
            verbose: Whether to print the verbose output
        Returns:
            TrialAtlasExecutionResults: The execution results from the agent
        """

        results = self.generate(input_query, modules=modules, verbose=verbose)
        final_state = results[-1]
        message_history = self._format_messages(final_state['messages'])
        code_execution_results = self._format_code_execution_results(final_state.get('code_execution_results', []))
        total_input_tokens = final_state['total_input_tokens']
        total_output_tokens = final_state['total_output_tokens']
        final_response = final_state['messages'][-1].content
        if isinstance(final_response, list):
            final_response = ' '.join(
                c.get('text', str(c)) if isinstance(c, dict) else str(c)
                for c in final_response
            )

        return TrialAtlasExecutionResults(
            sandbox=self.sandbox,
            message_history=message_history,
            code_execution_results=code_execution_results,
            final_response=final_response,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens
        )
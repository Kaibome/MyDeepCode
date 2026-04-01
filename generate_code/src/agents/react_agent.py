"""
Core LangGraph-based ReAct agent implementation.

This module implements the ReAct (Reasoning + Acting) loop using LangGraph's StateGraph.
It provides the agent state model, node functions, and conditional logic for building
a tool-using agent that can reason and act in a loop.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from src.config.settings import Settings
from src.config.llm_config import create_deepseek_llm, format_messages_for_llm
from src.tools.tool_registry import ToolRegistry, ToolExecutionResult

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """
    Represents the state of a ReAct agent throughout its execution.

    This state is passed between nodes in the LangGraph workflow and contains
    all necessary information for the agent to reason, act, and maintain context.

    Attributes:
        messages: List of messages in the conversation (human, AI, and tool responses).
        tool_calls: List of tool calls requested by the LLM in the current step.
        intermediate_steps: List of (action, observation) tuples from previous steps.
        iteration_count: Number of iterations completed in the current reasoning loop.
        max_iterations: Maximum number of iterations allowed before termination.
        conversation_id: Unique identifier for the conversation session.
        metadata: Additional metadata for the conversation.
    """
    
    messages: List[BaseMessage] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    intermediate_steps: List[Tuple[Dict[str, Any], str]] = Field(default_factory=list)
    iteration_count: int = Field(default=0)
    max_iterations: int = Field(default=10)
    conversation_id: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration for AgentState."""
        arbitrary_types_allowed = True
    
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message: The message to add (HumanMessage, AIMessage, or ToolMessage).
            
        Raises:
            TypeError: If message is not a BaseMessage instance.
        """
        if not isinstance(message, BaseMessage):
            raise TypeError(f"Expected BaseMessage, got {type(message)}")
        self.messages.append(message)
    
    def add_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """
        Add a tool call to the current step's tool calls.
        
        Args:
            tool_call: Dictionary containing tool call information.
        """
        self.tool_calls.append(tool_call)
    
    def add_intermediate_step(self, action: Dict[str, Any], observation: str) -> None:
        """
        Add an intermediate step (action, observation) to the history.
        
        Args:
            action: Dictionary describing the action taken (tool call).
            observation: Result or observation from executing the action.
        """
        self.intermediate_steps.append((action, observation))
    
    def increment_iteration(self) -> None:
        """Increment the iteration count by 1."""
        self.iteration_count += 1
    
    def has_exceeded_max_iterations(self) -> bool:
        """
        Check if the agent has exceeded the maximum allowed iterations.
        
        Returns:
            True if iteration_count >= max_iterations, False otherwise.
        """
        return self.iteration_count >= self.max_iterations
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation state.
        
        Returns:
            Dictionary containing conversation metadata and statistics.
        """
        return {
            "conversation_id": self.conversation_id,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "intermediate_step_count": len(self.intermediate_steps),
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }


def model_node(
    state: AgentState,
    llm: Any,
    tool_registry: ToolRegistry,
    system_prompt: Optional[str] = None
) -> AgentState:
    """
    Node function that calls the LLM to generate a response or tool calls.
    
    This is the "reasoning" part of the ReAct loop. The LLM analyzes the current
    state and decides whether to respond directly or request tool execution.
    
    Args:
        state: Current agent state containing conversation history.
        llm: Configured LLM client (e.g., ChatOpenAI instance).
        tool_registry: Tool registry for available tools.
        system_prompt: Optional system prompt to guide the LLM's behavior.
        
    Returns:
        Updated AgentState with LLM response and parsed tool calls.
        
    Raises:
        ValueError: If LLM response cannot be parsed or is invalid.
        RuntimeError: If LLM call fails.
    """
    logger.debug(f"model_node called with state: {state.get_conversation_summary()}")
    
    try:
        # Prepare messages for the LLM
        messages = format_messages_for_llm(
            [{"role": msg.type, "content": msg.content} for msg in state.messages],
            system_prompt=system_prompt
        )
        
        # Get available tools for the LLM
        available_tools = []
        tool_descriptions = []
        
        for tool_info in tool_registry.list_tools(enabled_only=True):
            tool_def = tool_registry.get_tool(tool_info["name"])
            if tool_def and tool_def.input_schema:
                # Create a simplified tool description for the LLM
                tool_desc = {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": tool_def.input_schema.schema() if hasattr(tool_def.input_schema, "schema") else {}
                }
                tool_descriptions.append(tool_desc)
                
                # Create LangChain tool wrapper
                tool_wrapper = BaseTool(
                    name=tool_info["name"],
                    description=tool_info["description"],
                    func=lambda **kwargs: tool_registry.execute_tool(tool_info["name"], **kwargs).result
                )
                available_tools.append(tool_wrapper)
        
        logger.debug(f"Available tools for LLM: {[t.name for t in available_tools]}")
        
        # Bind tools to the LLM if any are available
        if available_tools:
            llm_with_tools = llm.bind_tools(available_tools)
        else:
            llm_with_tools = llm
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        # Parse the response
        if hasattr(response, "tool_calls") and response.tool_calls:
            # LLM requested tool execution
            tool_calls = []
            for tool_call in response.tool_calls:
                tool_call_dict = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}),
                    "id": tool_call.get("id", f"tool_call_{len(state.tool_calls)}"),
                    "type": "tool_call"
                }
                tool_calls.append(tool_call_dict)
            
            # Update state with tool calls
            state.tool_calls = tool_calls
            
            # Add the LLM's message (which contains tool calls) to history
            state.add_message(response)
            
            logger.info(f"LLM requested {len(tool_calls)} tool calls: {[tc['name'] for tc in tool_calls]}")
            
        else:
            # LLM provided a direct response
            state.tool_calls = []  # Clear any previous tool calls
            
            # Add the LLM's response to history
            state.add_message(response)
            
            logger.info(f"LLM provided direct response: {response.content[:100]}...")
        
        # Increment iteration count
        state.increment_iteration()
        
        logger.debug(f"model_node completed. New iteration count: {state.iteration_count}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in model_node: {str(e)}", exc_info=True)
        raise RuntimeError(f"LLM call failed: {str(e)}")


def tool_node(
    state: AgentState,
    tool_registry: ToolRegistry
) -> AgentState:
    """
    Node function that executes requested tools and records results.
    
    This is the "acting" part of the ReAct loop. Tools requested by the LLM
    are executed here, and their results are added to the conversation history.
    
    Args:
        state: Current agent state containing tool calls to execute.
        tool_registry: Tool registry for executing tools.
        
    Returns:
        Updated AgentState with tool execution results.
        
    Raises:
        ValueError: If a requested tool is not available or arguments are invalid.
        RuntimeError: If tool execution fails.
    """
    logger.debug(f"tool_node called with {len(state.tool_calls)} tool calls")
    
    if not state.tool_calls:
        logger.warning("tool_node called with no tool calls")
        return state
    
    try:
        # Execute each tool call
        for tool_call in state.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"tool_call_{len(state.intermediate_steps)}")
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Check if tool is available
            if not tool_registry.is_tool_available(tool_name):
                error_msg = f"Tool '{tool_name}' is not available or disabled"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Execute the tool
            result: ToolExecutionResult = tool_registry.execute_tool(tool_name, **tool_args)
            
            # Record the intermediate step
            state.add_intermediate_step(
                action={"tool": tool_name, "args": tool_args, "id": tool_id},
                observation=result.result if result.success else f"Error: {result.error}"
            )
            
            # Create a tool message for the conversation history
            tool_message = ToolMessage(
                content=result.result if result.success else f"Error: {result.error}",
                tool_call_id=tool_id,
                name=tool_name
            )
            
            # Add tool message to conversation history
            state.add_message(tool_message)
            
            logger.info(f"Tool '{tool_name}' executed successfully: {result.success}")
            
        # Clear tool calls after execution
        state.tool_calls = []
        
        logger.debug(f"tool_node completed. Added {len(state.tool_calls)} tool results")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in tool_node: {str(e)}", exc_info=True)
        
        # Create error tool message
        error_message = ToolMessage(
            content=f"Tool execution failed: {str(e)}",
            tool_call_id="error",
            name="error_handler"
        )
        state.add_message(error_message)
        
        # Clear tool calls to prevent infinite loops
        state.tool_calls = []
        
        return state


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Conditional function that determines whether the agent should continue.
    
    This function decides whether to continue the ReAct loop or end execution.
    It checks for termination conditions like maximum iterations reached,
    no tool calls requested, or explicit completion signals.
    
    Args:
        state: Current agent state to evaluate.
        
    Returns:
        "continue" if the agent should continue to the next iteration,
        "end" if the agent should terminate.
    """
    logger.debug(f"should_continue evaluating state: {state.get_conversation_summary()}")
    
    # Check if maximum iterations exceeded
    if state.has_exceeded_max_iterations():
        logger.warning(f"Maximum iterations ({state.max_iterations}) exceeded. Ending agent.")
        return "end"
    
    # Check if there are pending tool calls (should continue to tool_node)
    if state.tool_calls:
        logger.debug(f"Tool calls pending: {len(state.tool_calls)}. Continuing to tool_node.")
        return "continue"
    
    # Check the last message to see if it indicates completion
    if state.messages:
        last_message = state.messages[-1]
        
        # If last message is from AI and contains completion indicators
        if isinstance(last_message, AIMessage):
            content = last_message.content.lower() if last_message.content else ""
            
            # Check for completion phrases
            completion_phrases = [
                "final answer",
                "i have answered",
                "that's all",
                "no more tools needed",
                "conclusion:",
                "in summary",
                "to summarize"
            ]
            
            for phrase in completion_phrases:
                if phrase in content:
                    logger.debug(f"Completion phrase '{phrase}' found. Ending agent.")
                    return "end"
    
    # Default: continue if we haven't determined to end
    logger.debug("No termination conditions met. Continuing agent.")
    return "continue"


def create_react_graph(
    llm: Any,
    tool_registry: ToolRegistry,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10
) -> Any:
    """
    Create and compile a LangGraph StateGraph for the ReAct agent.
    
    This function builds the complete workflow graph with model_node and tool_node,
    connected by conditional edges based on the should_continue function.
    
    Args:
        llm: Configured LLM client.
        tool_registry: Tool registry for available tools.
        system_prompt: Optional system prompt for the LLM.
        max_iterations: Maximum number of iterations allowed.
        
    Returns:
        Compiled LangGraph that can be invoked with an initial state.
        
    Raises:
        ValueError: If LLM or tool_registry is not provided.
    """
    if not llm:
        raise ValueError("LLM must be provided")
    if not tool_registry:
        raise ValueError("ToolRegistry must be provided")
    
    logger.info(f"Creating ReAct graph with max_iterations={max_iterations}")
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node(
        "model",
        lambda state: model_node(state, llm, tool_registry, system_prompt)
    )
    workflow.add_node(
        "tools",
        lambda state: tool_node(state, tool_registry)
    )
    
    # Set entry point
    workflow.set_entry_point("model")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to model (for next iteration)
    workflow.add_edge("tools", "model")
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    logger.info("ReAct graph compiled successfully")
    
    return compiled_graph


def get_default_system_prompt() -> str:
    """
    Get the default system prompt for the ReAct agent.
    
    Returns:
        String containing the default system prompt that guides the agent's behavior.
    """
    return """You are a helpful assistant that can use tools to answer questions.

You have access to various tools that can help you gather information or perform tasks.
When you need to use a tool, you should:
1. Think about what information you need
2. Choose the appropriate tool
3. Provide the required arguments
4. Wait for the tool's response

After receiving tool results, analyze them and decide if you need more information
or if you can provide a final answer.

Always be concise and helpful in your responses. If you're not sure about something,
use the available tools to find accurate information.

When you have a complete answer, provide it clearly and indicate that you're done."""

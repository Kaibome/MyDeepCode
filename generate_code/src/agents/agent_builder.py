"""
Agent builder module for constructing ReAct agents with tool integration.

This module provides factory functions to build LangGraph-based ReAct agents
with DeepSeek LLM integration and tool support.
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from langgraph.graph import END, StateGraph

from src.config.settings import Settings
from src.config.llm_config import create_deepseek_llm
from src.tools.tool_registry import ToolRegistry
from src.tools.weather_tool import create_weather_tool
from src.tools.travel_tool import create_travel_tool
from src.agents.react_agent import AgentState, model_node, tool_node, should_continue

logger = logging.getLogger(__name__)


def build_react_agent(
    settings: Settings,
    tool_registry: Optional[ToolRegistry] = None,
    system_prompt: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> Any:
    """
    Build a complete ReAct agent with tool integration.

    This function creates a LangGraph-based ReAct agent with DeepSeek LLM
    integration and registered tools. The agent follows the ReAct pattern:
    Reasoning (LLM) -> Acting (Tools) -> Repeat until completion.

    Args:
        settings: Application settings containing LLM configuration.
        tool_registry: Optional pre-configured tool registry. If None,
            a new registry will be created with default tools.
        system_prompt: Optional system prompt to guide the agent's behavior.
            If None, a default prompt will be used.
        max_iterations: Maximum number of ReAct iterations before termination.
            If None, uses the value from settings.

    Returns:
        A compiled LangGraph agent that can be invoked with user messages.

    Raises:
        ValueError: If LLM configuration is invalid or tools cannot be created.
        RuntimeError: If agent graph construction fails.

    Example:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> agent = build_react_agent(settings)
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    """
    logger.info("Building ReAct agent with DeepSeek LLM integration")

    # Validate inputs
    if not settings:
        raise ValueError("Settings object is required to build agent")

    # Create LLM client
    try:
        llm = create_deepseek_llm(settings)
        logger.debug("Created DeepSeek LLM client")
    except Exception as e:
        logger.error(f"Failed to create DeepSeek LLM: {e}")
        raise ValueError(f"LLM configuration failed: {e}") from e

    # Create or validate tool registry
    if tool_registry is None:
        tool_registry = _create_default_tool_registry()
        logger.debug("Created default tool registry")
    elif not isinstance(tool_registry, ToolRegistry):
        raise TypeError(f"Expected ToolRegistry, got {type(tool_registry)}")

    # Set system prompt if not provided
    if system_prompt is None:
        system_prompt = _get_default_system_prompt(tool_registry)

    # Set max iterations
    if max_iterations is None:
        max_iterations = settings.AGENT_MAX_ITERATIONS

    logger.info(
        f"Building agent with {len(tool_registry.get_available_tool_names())} tools "
        f"and max {max_iterations} iterations"
    )

    # Build the LangGraph workflow
    try:
        agent = _build_langgraph_agent(
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )
        logger.info("Successfully built ReAct agent")
        return agent
    except Exception as e:
        logger.error(f"Failed to build LangGraph agent: {e}")
        raise RuntimeError(f"Agent construction failed: {e}") from e


def _create_default_tool_registry() -> ToolRegistry:
    """
    Create a tool registry with default tools.

    Returns:
        A ToolRegistry instance with weather and travel tools registered.

    Raises:
        RuntimeError: If any tool creation fails.
    """
    from src.tools.tool_registry import ToolRegistry

    registry = ToolRegistry()
    logger.debug("Created empty tool registry")

    try:
        # Create and register weather tool
        weather_tool = create_weather_tool()
        registry.register_tool(
            name="weather_query",
            tool_instance=weather_tool,
            description="Get weather information for a city on a specific date",
            input_schema=weather_tool.args_schema,
            metadata={"category": "information", "mock": True},
        )
        logger.debug("Registered weather tool")
    except Exception as e:
        logger.warning(f"Failed to register weather tool: {e}")
        # Continue without weather tool

    try:
        # Create and register travel tool
        travel_tool = create_travel_tool()
        registry.register_tool(
            name="travel_query",
            tool_instance=travel_tool,
            description="Find travel options between two cities",
            input_schema=travel_tool.args_schema,
            metadata={"category": "travel", "mock": True},
        )
        logger.debug("Registered travel tool")
    except Exception as e:
        logger.warning(f"Failed to register travel tool: {e}")
        # Continue without travel tool

    if len(registry.get_available_tool_names()) == 0:
        logger.warning("No tools were registered successfully")

    return registry


def _get_default_system_prompt(tool_registry: ToolRegistry) -> str:
    """
    Generate a default system prompt based on available tools.

    Args:
        tool_registry: Tool registry to get tool information from.

    Returns:
        A system prompt string that guides the agent's behavior.
    """
    tool_list = tool_registry.list_tools(enabled_only=True)
    tool_descriptions = []

    for tool_info in tool_list:
        name = tool_info.get("name", "unknown")
        description = tool_info.get("description", "No description")
        tool_descriptions.append(f"- {name}: {description}")

    tools_section = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"

    return f"""You are a helpful assistant with access to tools. Follow the ReAct pattern:

1. **Reason**: Think step by step about what you need to do
2. **Act**: Use tools when needed to get information
3. **Observe**: Consider the tool results
4. **Repeat**: Continue until you can provide a complete answer

**Available Tools:**
{tools_section}

**Guidelines:**
- Always think before acting
- Use tools when you need specific information
- Be concise but thorough in your reasoning
- If a tool fails, try to work around it or explain the limitation
- When providing final answers, summarize key information clearly

**Response Format:**
For tool calls, use the exact format: TOOL_CALL: tool_name {{"arg1": "value1", "arg2": "value2"}}
For final answers, provide a clear, helpful response.
"""


def _build_langgraph_agent(
    llm: Any,
    tool_registry: ToolRegistry,
    system_prompt: str,
    max_iterations: int,
) -> Any:
    """
    Build the LangGraph workflow for the ReAct agent.

    Args:
        llm: Configured LLM client.
        tool_registry: Tool registry with registered tools.
        system_prompt: System prompt for the agent.
        max_iterations: Maximum number of iterations.

    Returns:
        A compiled LangGraph agent.

    Raises:
        ValueError: If graph construction fails.
    """
    logger.debug(f"Building LangGraph workflow with max {max_iterations} iterations")

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes with bound parameters
    workflow.add_node(
        "model",
        lambda state: model_node(
            state=state,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
        ),
    )

    workflow.add_node(
        "tools",
        lambda state: tool_node(state=state, tool_registry=tool_registry),
    )

    # Set entry point
    workflow.set_entry_point("model")

    # Add conditional edges
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "model")

    # Compile the graph
    try:
        compiled_agent = workflow.compile()
        logger.debug("Successfully compiled LangGraph workflow")
        return compiled_agent
    except Exception as e:
        logger.error(f"Failed to compile LangGraph workflow: {e}")
        raise ValueError(f"Graph compilation failed: {e}") from e


def build_agent_with_custom_tools(
    settings: Settings,
    tools: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> Any:
    """
    Build a ReAct agent with custom tools.

    Args:
        settings: Application settings.
        tools: List of tool definitions. Each tool should be a dict with:
            - name: Tool identifier
            - instance: Tool instance with a run() method
            - description: Tool description
            - input_schema: Optional Pydantic model for input validation
            - metadata: Optional metadata dict
        system_prompt: Optional custom system prompt.
        max_iterations: Maximum iterations.

    Returns:
        A compiled LangGraph agent.

    Raises:
        ValueError: If tool definitions are invalid.
    """
    from src.tools.tool_registry import ToolRegistry

    logger.info(f"Building agent with {len(tools)} custom tools")

    # Create tool registry
    registry = ToolRegistry()

    # Register each tool
    for tool_def in tools:
        required_fields = {"name", "instance", "description"}
        missing_fields = required_fields - set(tool_def.keys())
        if missing_fields:
            raise ValueError(f"Tool definition missing fields: {missing_fields}")

        name = tool_def["name"]
        instance = tool_def["instance"]
        description = tool_def["description"]
        input_schema = tool_def.get("input_schema")
        metadata = tool_def.get("metadata", {})

        # Validate tool instance has run method
        if not hasattr(instance, "run") or not callable(instance.run):
            raise ValueError(f"Tool '{name}' must have a run() method")

        success = registry.register_tool(
            name=name,
            tool_instance=instance,
            description=description,
            input_schema=input_schema,
            metadata=metadata,
            overwrite=True,
        )

        if not success:
            logger.warning(f"Failed to register tool: {name}")

    # Build agent with custom registry
    return build_react_agent(
        settings=settings,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def get_agent_summary(agent: Any) -> Dict[str, Any]:
    """
    Get a summary of the agent's configuration and capabilities.

    Args:
        agent: Compiled LangGraph agent.

    Returns:
        Dictionary with agent summary information.
    """
    if not hasattr(agent, "nodes") or not hasattr(agent, "edges"):
        return {"error": "Not a valid LangGraph agent"}

    summary = {
        "type": "ReAct Agent",
        "nodes": list(agent.nodes.keys()),
        "edges": [],
        "has_tools": False,
        "iteration_limit": "Unknown",
    }

    # Count edges
    if hasattr(agent, "edges"):
        summary["edges"] = [
            {"from": edge[0], "to": edge[1]} for edge in agent.edges if isinstance(edge, tuple)
        ]

    # Check for tool node
    summary["has_tools"] = "tools" in agent.nodes

    # Try to get iteration info from state
    try:
        state_schema = agent.config.get("configurable", {}).get("state_schema")
        if state_schema and hasattr(state_schema, "model_fields"):
            if "iteration_count" in state_schema.model_fields:
                summary["iteration_limit"] = "Tracked in state"
    except Exception:
        pass

    return summary


def validate_agent(agent: Any) -> bool:
    """
    Validate that an agent is properly constructed.

    Args:
        agent: Agent to validate.

    Returns:
        True if agent appears valid, False otherwise.
    """
    try:
        # Check basic structure
        if not hasattr(agent, "invoke"):
            logger.warning("Agent missing invoke method")
            return False

        if not hasattr(agent, "nodes"):
            logger.warning("Agent missing nodes attribute")
            return False

        # Check for required nodes
        nodes = agent.nodes
        if "model" not in nodes:
            logger.warning("Agent missing 'model' node")
            return False

        # Check node functions are callable
        model_node_func = nodes.get("model")
        if not callable(model_node_func):
            logger.warning("Model node is not callable")
            return False

        logger.debug("Agent validation passed")
        return True

    except Exception as e:
        logger.warning(f"Agent validation failed: {e}")
        return False
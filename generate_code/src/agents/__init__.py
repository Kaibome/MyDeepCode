"""
Agents module for the ReAct Agent system.

This module contains the core agent implementations, including the LangGraph-based
ReAct agent and agent builder factory.
"""

from typing import List

__all__: List[str] = [
    "react_agent",
    "agent_builder",
]

# Import statements for public API
# These will be populated when the modules are implemented
try:
    from .react_agent import AgentState, model_node, tool_node, should_continue
    from .agent_builder import build_react_agent
    
    __all__.extend([
        "AgentState",
        "model_node", 
        "tool_node",
        "should_continue",
        "build_react_agent",
    ])
except ImportError:
    # Modules not yet implemented
    pass

__version__ = "0.1.0"
__author__ = "ReAct Agent Implementation Team"
__description__ = "Core agent implementations for the ReAct pattern using LangGraph"
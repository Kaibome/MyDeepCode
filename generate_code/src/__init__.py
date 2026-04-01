"""
ReAct Agent with Tool Integration using LangGraph and DeepSeek.

This package implements a modular, tool-using agent system implementing the ReAct 
(Reasoning + Acting) pattern, integrated with a FastAPI web interface, using 
mock tools for weather and travel queries with a clear path for future MCP server integration.

Modules:
    agents: Core LangGraph ReAct agent implementation
    tools: Mock tool implementations and registry
    api: FastAPI web interface
    config: Configuration and settings management
"""

__version__ = "0.1.0"
__author__ = "ReAct Agent Implementation Team"
__description__ = "A modular ReAct agent with tool integration using LangGraph and DeepSeek"

from typing import List

__all__: List[str] = []  # Populated by submodules
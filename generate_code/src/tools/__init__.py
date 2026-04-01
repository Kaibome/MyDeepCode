"""
Tools module for the ReAct Agent system.

This module provides tool implementations and a central registry for managing
tools that can be used by the ReAct agent. Tools are designed to be modular
and easily extensible.

Public Classes:
    - WeatherQueryTool: Mock weather query tool
    - TravelQueryTool: Mock travel query tool
    - ToolRegistry: Central tool management system

Public Functions:
    - create_tool_registry: Factory function to create and register all tools
"""

from typing import List, Dict, Any, Optional
import warnings

__version__ = "0.1.0"
__author__ = "ReAct Agent Implementation Team"
__description__ = "Tool implementations and registry for the ReAct agent system"
__all__: List[str] = []

# Try to import tool classes and registry
try:
    from .weather_tool import WeatherQueryTool
    __all__.append("WeatherQueryTool")
except ImportError as e:
    warnings.warn(f"Could not import WeatherQueryTool: {e}")

try:
    from .travel_tool import TravelQueryTool
    __all__.append("TravelQueryTool")
except ImportError as e:
    warnings.warn(f"Could not import TravelQueryTool: {e}")

try:
    from .tool_registry import ToolRegistry, create_tool_registry
    __all__.extend(["ToolRegistry", "create_tool_registry"])
except ImportError as e:
    warnings.warn(f"Could not import ToolRegistry: {e}")

# Clean up namespace
del List, Dict, Any, Optional, warnings
"""
Configuration module for the ReAct Agent system.

This module provides configuration management, settings loading, and LLM client
configuration for the ReAct agent system using LangGraph and DeepSeek.
"""

from typing import List

__version__: str = "0.1.0"
__author__: str = "ReAct Agent Implementation Team"
__description__: str = "Configuration management for the ReAct agent system"

# Public API exports
__all__: List[str] = []

# Import configuration components with graceful fallbacks
try:
    from .settings import Settings
    __all__.append("Settings")
except ImportError:
    print("Warning: Could not import Settings from .settings")

try:
    from .llm_config import create_deepseek_llm
    __all__.append("create_deepseek_llm")
except ImportError:
    print("Warning: Could not import create_deepseek_llm from .llm_config")

# Re-export for cleaner imports
if "Settings" in __all__:
    Settings = Settings  # type: ignore

if "create_deepseek_llm" in __all__:
    create_deepseek_llm = create_deepseek_llm  # type: ignore
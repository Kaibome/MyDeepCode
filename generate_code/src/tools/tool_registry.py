"""
Tool Registry for the ReAct Agent System.

This module provides a centralized registry for managing and executing tools
in the ReAct agent system. It serves as a singleton manager that allows
tools to be registered, retrieved, and executed by name.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type, Callable, Set
from pydantic import BaseModel, Field, validator

from src.tools.weather_tool import WeatherQueryTool, create_weather_tool
from src.tools.travel_tool import TravelQueryTool, create_travel_tool

logger = logging.getLogger(__name__)


class ToolDefinition(BaseModel):
    """
    Definition of a tool for registration in the ToolRegistry.
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of the tool's purpose.
        tool_instance: The actual tool instance (e.g., WeatherQueryTool).
        input_schema: Pydantic model for validating tool inputs.
        is_enabled: Whether the tool is currently enabled.
        metadata: Additional metadata about the tool.
    """
    
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    tool_instance: Any = Field(..., description="The actual tool instance")
    input_schema: Optional[Type[BaseModel]] = Field(
        None, description="Pydantic model for input validation"
    )
    is_enabled: bool = Field(default=True, description="Whether the tool is enabled")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional tool metadata"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ToolExecutionResult(BaseModel):
    """
    Result of a tool execution.
    
    Attributes:
        success: Whether the execution was successful.
        result: The execution result (if successful).
        error: Error message (if unsuccessful).
        tool_name: Name of the executed tool.
        execution_time_ms: Execution time in milliseconds.
    """
    
    success: bool = Field(..., description="Whether execution was successful")
    result: Optional[Any] = Field(None, description="Execution result if successful")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    tool_name: str = Field(..., description="Name of the executed tool")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result.
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
        }


class ToolRegistry:
    """
    Central registry for managing and executing tools.
    
    This class implements a singleton pattern to provide a single point
    of access for tool registration, retrieval, and execution.
    
    Attributes:
        _tools: Dictionary mapping tool names to ToolDefinition objects.
        _initialized: Whether the registry has been initialized.
    """
    
    _instance: Optional['ToolRegistry'] = None
    _tools: Dict[str, ToolDefinition]
    _initialized: bool
    
    def __new__(cls) -> 'ToolRegistry':
        """
        Create or return the singleton instance.
        
        Returns:
            The singleton ToolRegistry instance.
        """
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._tools = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the tool registry.
        
        Note: This method is called on each instance access due to singleton
        pattern, but actual initialization only happens once.
        """
        if not self._initialized:
            self._initialized = True
            logger.debug("ToolRegistry initialized")
    
    def register_tool(
        self,
        name: str,
        tool_instance: Any,
        description: str,
        input_schema: Optional[Type[BaseModel]] = None,
        is_enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            name: Unique name for the tool.
            tool_instance: The tool instance to register.
            description: Description of what the tool does.
            input_schema: Optional Pydantic model for input validation.
            is_enabled: Whether the tool is enabled by default.
            metadata: Additional metadata about the tool.
            overwrite: Whether to overwrite an existing tool with the same name.
            
        Returns:
            True if registration was successful, False otherwise.
            
        Raises:
            ValueError: If a tool with the same name already exists and overwrite=False.
        """
        if name in self._tools and not overwrite:
            raise ValueError(
                f"Tool '{name}' already registered. Use overwrite=True to replace."
            )
        
        tool_def = ToolDefinition(
            name=name,
            description=description,
            tool_instance=tool_instance,
            input_schema=input_schema,
            is_enabled=is_enabled,
            metadata=metadata or {},
        )
        
        self._tools[name] = tool_def
        logger.info(f"Registered tool '{name}': {description}")
        return True
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            name: Name of the tool to retrieve.
            
        Returns:
            ToolDefinition if found, None otherwise.
        """
        return self._tools.get(name)
    
    def get_tool_instance(self, name: str) -> Optional[Any]:
        """
        Get the tool instance by name.
        
        Args:
            name: Name of the tool to retrieve.
            
        Returns:
            Tool instance if found, None otherwise.
        """
        tool_def = self.get_tool(name)
        return tool_def.tool_instance if tool_def else None
    
    def execute_tool(
        self,
        name: str,
        **kwargs: Any,
    ) -> ToolExecutionResult:
        """
        Execute a tool by name with the given arguments.
        
        Args:
            name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.
            
        Returns:
            ToolExecutionResult containing the execution outcome.
            
        Raises:
            ValueError: If the tool is not found or is disabled.
        """
        import time
        
        start_time = time.time()
        
        # Get the tool definition
        tool_def = self.get_tool(name)
        if not tool_def:
            error_msg = f"Tool '{name}' not found in registry"
            logger.error(error_msg)
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                tool_name=name,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Check if tool is enabled
        if not tool_def.is_enabled:
            error_msg = f"Tool '{name}' is disabled"
            logger.warning(error_msg)
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                tool_name=name,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Validate inputs if schema is provided
        if tool_def.input_schema:
            try:
                validated_inputs = tool_def.input_schema(**kwargs)
                # Convert validated inputs back to dict for tool execution
                kwargs = validated_inputs.dict()
            except Exception as e:
                error_msg = f"Input validation failed for tool '{name}': {str(e)}"
                logger.error(error_msg)
                return ToolExecutionResult(
                    success=False,
                    error=error_msg,
                    tool_name=name,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
        
        # Execute the tool
        try:
            # Check if tool has a run method
            if hasattr(tool_def.tool_instance, 'run'):
                result = tool_def.tool_instance.run(**kwargs)
            elif hasattr(tool_def.tool_instance, '__call__'):
                result = tool_def.tool_instance(**kwargs)
            else:
                error_msg = f"Tool '{name}' has no executable method (run or __call__)"
                logger.error(error_msg)
                return ToolExecutionResult(
                    success=False,
                    error=error_msg,
                    tool_name=name,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Executed tool '{name}' in {execution_time_ms:.2f}ms")
            
            return ToolExecutionResult(
                success=True,
                result=result,
                tool_name=name,
                execution_time_ms=execution_time_ms,
            )
            
        except Exception as e:
            error_msg = f"Tool execution failed for '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                tool_name=name,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
    
    def list_tools(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Args:
            enabled_only: If True, only return enabled tools.
            
        Returns:
            List of dictionaries containing tool information.
        """
        tools_list = []
        for name, tool_def in self._tools.items():
            if enabled_only and not tool_def.is_enabled:
                continue
            
            # Get tool schema if available
            tool_schema = None
            if hasattr(tool_def.tool_instance, 'get_tool_schema'):
                try:
                    tool_schema = tool_def.tool_instance.get_tool_schema()
                except Exception as e:
                    logger.warning(f"Failed to get schema for tool '{name}': {e}")
            
            tools_list.append({
                "name": name,
                "description": tool_def.description,
                "enabled": tool_def.is_enabled,
                "input_schema": tool_def.input_schema.__name__ if tool_def.input_schema else None,
                "metadata": tool_def.metadata,
                "schema": tool_schema,
            })
        
        return tools_list
    
    def enable_tool(self, name: str) -> bool:
        """
        Enable a tool.
        
        Args:
            name: Name of the tool to enable.
            
        Returns:
            True if tool was enabled, False if tool not found.
        """
        tool_def = self.get_tool(name)
        if not tool_def:
            logger.warning(f"Cannot enable tool '{name}': not found")
            return False
        
        tool_def.is_enabled = True
        logger.info(f"Enabled tool '{name}'")
        return True
    
    def disable_tool(self, name: str) -> bool:
        """
        Disable a tool.
        
        Args:
            name: Name of the tool to disable.
            
        Returns:
            True if tool was disabled, False if tool not found.
        """
        tool_def = self.get_tool(name)
        if not tool_def:
            logger.warning(f"Cannot disable tool '{name}': not found")
            return False
        
        tool_def.is_enabled = False
        logger.info(f"Disabled tool '{name}'")
        return True
    
    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove.
            
        Returns:
            True if tool was removed, False if tool not found.
        """
        if name not in self._tools:
            logger.warning(f"Cannot remove tool '{name}': not found")
            return False
        
        del self._tools[name]
        logger.info(f"Removed tool '{name}' from registry")
        return True
    
    def clear_tools(self) -> None:
        """
        Clear all tools from the registry.
        """
        tool_count = len(self._tools)
        self._tools.clear()
        logger.info(f"Cleared all {tool_count} tools from registry")
    
    def get_tool_names(self, enabled_only: bool = False) -> List[str]:
        """
        Get list of all registered tool names.
        
        Args:
            enabled_only: If True, only return names of enabled tools.
            
        Returns:
            List of tool names.
        """
        if enabled_only:
            return [name for name, tool_def in self._tools.items() if tool_def.is_enabled]
        return list(self._tools.keys())
    
    def tool_exists(self, name: str) -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            name: Name of the tool to check.
            
        Returns:
            True if tool exists, False otherwise.
        """
        return name in self._tools
    
    def get_tool_count(self, enabled_only: bool = False) -> int:
        """
        Get the number of registered tools.
        
        Args:
            enabled_only: If True, only count enabled tools.
            
        Returns:
            Number of tools.
        """
        if enabled_only:
            return sum(1 for tool_def in self._tools.values() if tool_def.is_enabled)
        return len(self._tools)
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a tool.
        
        Args:
            name: Name of the tool.
            
        Returns:
            Tool schema dictionary if available, None otherwise.
        """
        tool_def = self.get_tool(name)
        if not tool_def:
            return None
        
        if hasattr(tool_def.tool_instance, 'get_tool_schema'):
            try:
                return tool_def.tool_instance.get_tool_schema()
            except Exception as e:
                logger.warning(f"Failed to get schema for tool '{name}': {e}")
                return None
        
        # Fallback: create basic schema from tool definition
        return {
            "name": name,
            "description": tool_def.description,
            "args": {
                "type": "object",
                "properties": {},
            } if not tool_def.input_schema else {
                "type": "object",
                "properties": tool_def.input_schema.schema().get("properties", {}),
            },
        }


def create_tool_registry() -> ToolRegistry:
    """
    Create and populate a ToolRegistry with default tools.
    
    This function creates a ToolRegistry instance and registers the
    default tools (WeatherQueryTool and TravelQueryTool).
    
    Returns:
        A populated ToolRegistry instance.
    """
    registry = ToolRegistry()
    
    # Register weather tool
    try:
        weather_tool = create_weather_tool()
        registry.register_tool(
            name="weather_query",
            tool_instance=weather_tool,
            description="Query weather information for a city on a specific date",
            input_schema=getattr(weather_tool, 'args_schema', None),
            metadata={"category": "weather", "mock": True},
        )
        logger.info("Registered weather_query tool")
    except Exception as e:
        logger.error(f"Failed to register weather_query tool: {e}")
    
    # Register travel tool
    try:
        travel_tool = create_travel_tool()
        registry.register_tool(
            name="travel_query",
            tool_instance=travel_tool,
            description="Query travel options between two cities",
            input_schema=getattr(travel_tool, 'args_schema', None),
            metadata={"category": "travel", "mock": True},
        )
        logger.info("Registered travel_query tool")
    except Exception as e:
        logger.error(f"Failed to register travel_query tool: {e}")
    
    logger.info(f"ToolRegistry created with {registry.get_tool_count()} tools")
    return registry


# Export the singleton instance
tool_registry = create_tool_registry()
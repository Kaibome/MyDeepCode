"""
API route definitions for the ReAct Agent system.

This module defines the REST API endpoints for interacting with the ReAct agent,
including chat, health checks, and tool management.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from src.agents.agent_builder import build_react_agent
from src.config.settings import Settings, get_settings
from src.tools.tool_registry import ToolRegistry, create_tool_registry

# Import schemas if available, otherwise define placeholders
try:
    from src.api.schemas import (
        ChatRequest,
        ChatResponse,
        ErrorResponse,
        HealthResponse,
        ToolInfo,
        ToolListResponse,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    # Define placeholder schemas if not available
    SCHEMAS_AVAILABLE = False
    from pydantic import BaseModel, Field

    class ChatRequest(BaseModel):
        """Request model for chat endpoint."""
        message: str = Field(..., description="User message to process")
        conversation_id: Optional[str] = Field(
            None, description="Optional conversation ID for multi-turn conversations"
        )
        system_prompt: Optional[str] = Field(
            None, description="Optional system prompt to override default"
        )
        max_iterations: Optional[int] = Field(
            None, description="Maximum iterations for the agent (overrides settings)"
        )
        return_format: str = Field(
            "text", description="Response format: 'text' or 'structured'"
        )

    class ChatResponse(BaseModel):
        """Response model for chat endpoint."""
        response: str = Field(..., description="Agent's response message")
        conversation_id: str = Field(..., description="Conversation ID")
        tool_calls: List[Dict[str, Any]] = Field(
            default_factory=list, description="List of tool calls made during processing"
        )
        iteration_count: int = Field(
            ..., description="Number of iterations performed by the agent"
        )
        processing_time_ms: float = Field(
            ..., description="Time taken to process the request in milliseconds"
        )
        timestamp: datetime = Field(
            default_factory=datetime.now, description="Response timestamp"
        )

    class HealthResponse(BaseModel):
        """Response model for health check endpoint."""
        status: str = Field(..., description="Service status")
        timestamp: datetime = Field(
            default_factory=datetime.now, description="Health check timestamp"
        )
        version: str = Field(..., description="API version")
        uptime_seconds: Optional[float] = Field(
            None, description="Service uptime in seconds"
        )

    class ToolInfo(BaseModel):
        """Information about a registered tool."""
        name: str = Field(..., description="Tool name")
        description: str = Field(..., description="Tool description")
        enabled: bool = Field(..., description="Whether the tool is enabled")
        input_schema: Optional[Dict[str, Any]] = Field(
            None, description="Tool input schema if available"
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional tool metadata"
        )

    class ToolListResponse(BaseModel):
        """Response model for tools list endpoint."""
        tools: List[ToolInfo] = Field(..., description="List of available tools")
        count: int = Field(..., description="Total number of tools")

    class ErrorResponse(BaseModel):
        """Response model for error responses."""
        error: str = Field(..., description="Error message")
        detail: Optional[str] = Field(None, description="Detailed error information")
        timestamp: datetime = Field(
            default_factory=datetime.now, description="Error timestamp"
        )


# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["agent"])

# Global state for conversation tracking
_conversation_states: Dict[str, Dict[str, Any]] = {}
_app_start_time = datetime.now()


def get_tool_registry() -> ToolRegistry:
    """
    Dependency to get the tool registry instance.
    
    Returns:
        ToolRegistry: The singleton tool registry instance.
    """
    return create_tool_registry()


def get_agent(settings: Settings, tool_registry: ToolRegistry) -> Any:
    """
    Dependency to get the ReAct agent instance.
    
    Args:
        settings: Application settings.
        tool_registry: Tool registry instance.
        
    Returns:
        Any: The compiled LangGraph agent.
        
    Raises:
        HTTPException: If agent creation fails.
    """
    try:
        agent = build_react_agent(settings, tool_registry)
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring the API service.
    
    Returns:
        HealthResponse: Service health status and metadata.
    """
    uptime = (datetime.now() - _app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
    )


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    enabled_only: bool = True,
) -> ToolListResponse:
    """
    List all available tools in the system.
    
    Args:
        tool_registry: Tool registry instance.
        enabled_only: If True, only return enabled tools.
        
    Returns:
        ToolListResponse: List of available tools.
    """
    try:
        tools_data = tool_registry.list_tools(enabled_only=enabled_only)
        
        tools = []
        for tool_data in tools_data:
            tool_info = ToolInfo(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                enabled=tool_data.get("enabled", False),
                input_schema=tool_data.get("input_schema"),
                metadata=tool_data.get("metadata"),
            )
            tools.append(tool_info)
        
        return ToolListResponse(tools=tools, count=len(tools))
    except Exception as e:
        logger.error(f"Failed to list tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tools: {str(e)}",
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
) -> ChatResponse:
    """
    Process a chat message with the ReAct agent.
    
    This endpoint accepts a user message, processes it through the ReAct agent
    with tool integration, and returns the agent's response.
    
    Args:
        request: Chat request containing the user message and optional parameters.
        settings: Application settings.
        tool_registry: Tool registry instance.
        
    Returns:
        ChatResponse: Agent's response with conversation metadata.
        
    Raises:
        HTTPException: If processing fails or agent creation fails.
    """
    start_time = datetime.now()
    
    try:
        # Get or create conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation state if it exists
        conversation_state = _conversation_states.get(conversation_id, {})
        
        # Build agent with custom parameters if provided
        agent_kwargs = {}
        if request.system_prompt:
            agent_kwargs["system_prompt"] = request.system_prompt
        if request.max_iterations:
            agent_kwargs["max_iterations"] = request.max_iterations
        
        # Create agent
        agent = build_react_agent(settings, tool_registry, **agent_kwargs)
        
        # Prepare initial state
        initial_state = {
            "messages": [
                {"role": "user", "content": request.message}
            ],
            "tool_calls": [],
            "intermediate_steps": [],
            "iteration_count": 0,
            "conversation_id": conversation_id,
        }
        
        # Update with previous conversation state if available
        if conversation_state:
            # Add previous messages to maintain context
            previous_messages = conversation_state.get("messages", [])
            initial_state["messages"] = previous_messages + initial_state["messages"]
        
        # Invoke agent
        logger.info(f"Processing chat request for conversation {conversation_id}")
        result = agent.invoke(initial_state)
        
        # Extract response
        messages = result.get("messages", [])
        tool_calls = result.get("tool_calls", [])
        iteration_count = result.get("iteration_count", 0)
        
        # Find the last assistant message
        response_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break
        
        # Update conversation state
        _conversation_states[conversation_id] = {
            "messages": messages,
            "last_updated": datetime.now(),
            "iteration_count": iteration_count,
        }
        
        # Clean up old conversations (basic garbage collection)
        _cleanup_old_conversations()
        
        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format tool calls for response
        formatted_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                formatted_tool_calls.append(tool_call)
            else:
                # Convert to dict if it's not already
                formatted_tool_calls.append({
                    "name": getattr(tool_call, "name", "unknown"),
                    "args": getattr(tool_call, "args", {}),
                    "result": getattr(tool_call, "result", ""),
                })
        
        logger.info(
            f"Chat request completed for conversation {conversation_id}: "
            f"{iteration_count} iterations, {processing_time_ms:.2f}ms"
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            tool_calls=formatted_tool_calls,
            iteration_count=iteration_count,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
) -> Dict[str, Any]:
    """
    Get information about a specific conversation.
    
    Args:
        conversation_id: The conversation ID to retrieve.
        
    Returns:
        Dict containing conversation information.
        
    Raises:
        HTTPException: If conversation not found.
    """
    if conversation_id not in _conversation_states:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    
    conversation_state = _conversation_states[conversation_id]
    
    return {
        "conversation_id": conversation_id,
        "message_count": len(conversation_state.get("messages", [])),
        "iteration_count": conversation_state.get("iteration_count", 0),
        "last_updated": conversation_state.get("last_updated"),
        "active": True,
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
) -> Dict[str, Any]:
    """
    Delete a specific conversation.
    
    Args:
        conversation_id: The conversation ID to delete.
        
    Returns:
        Dict containing deletion status.
    """
    if conversation_id in _conversation_states:
        del _conversation_states[conversation_id]
        logger.info(f"Deleted conversation {conversation_id}")
        return {"status": "deleted", "conversation_id": conversation_id}
    else:
        return {"status": "not_found", "conversation_id": conversation_id}


@router.get("/conversations")
async def list_conversations(
    limit: int = 10,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List all active conversations.
    
    Args:
        limit: Maximum number of conversations to return.
        offset: Offset for pagination.
        
    Returns:
        Dict containing list of conversations and metadata.
    """
    conversation_ids = list(_conversation_states.keys())
    
    # Apply pagination
    start_idx = offset
    end_idx = offset + limit
    paginated_ids = conversation_ids[start_idx:end_idx]
    
    conversations = []
    for conv_id in paginated_ids:
        conv_state = _conversation_states[conv_id]
        conversations.append({
            "conversation_id": conv_id,
            "message_count": len(conv_state.get("messages", [])),
            "last_updated": conv_state.get("last_updated"),
        })
    
    return {
        "conversations": conversations,
        "total": len(conversation_ids),
        "limit": limit,
        "offset": offset,
        "has_more": end_idx < len(conversation_ids),
    }


@router.post("/tools/{tool_name}/execute")
async def execute_tool_directly(
    tool_name: str,
    parameters: Dict[str, Any],
    tool_registry: ToolRegistry = Depends(get_tool_registry),
) -> Dict[str, Any]:
    """
    Execute a tool directly with provided parameters.
    
    This endpoint allows direct execution of tools without going through the agent.
    Useful for testing and debugging.
    
    Args:
        tool_name: Name of the tool to execute.
        parameters: Tool parameters as key-value pairs.
        tool_registry: Tool registry instance.
        
    Returns:
        Dict containing tool execution result.
        
    Raises:
        HTTPException: If tool not found or execution fails.
    """
    try:
        if not tool_registry.is_tool_available(tool_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found or disabled",
            )
        
        result = tool_registry.execute_tool(tool_name, **parameters)
        
        return {
            "tool_name": tool_name,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": datetime.now(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}",
        )


def _cleanup_old_conversations(max_age_hours: int = 24) -> None:
    """
    Clean up old conversations from memory.
    
    Args:
        max_age_hours: Maximum age of conversations to keep (in hours).
    """
    current_time = datetime.now()
    to_delete = []
    
    for conv_id, conv_state in _conversation_states.items():
        last_updated = conv_state.get("last_updated")
        if last_updated:
            age_hours = (current_time - last_updated).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_delete.append(conv_id)
    
    for conv_id in to_delete:
        del _conversation_states[conv_id]
        logger.debug(f"Cleaned up old conversation: {conv_id}")
    
    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old conversations")


# Export router
__all__ = ["router"]
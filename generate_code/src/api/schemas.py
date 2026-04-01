"""
Pydantic schemas for the ReAct Agent API.

This module defines the request and response models for the FastAPI endpoints,
ensuring proper data validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    
    Represents a user's chat message and optional configuration for the ReAct agent.
    """
    
    message: str = Field(
        ...,
        description="The user's message to process with the ReAct agent.",
        min_length=1,
        max_length=5000,
        example="What's the weather in Tokyo tomorrow?"
    )
    
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for multi-turn conversations. "
                   "If not provided, a new conversation will be started.",
        example="conv_12345"
    )
    
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system prompt to override the default agent behavior.",
        example="You are a helpful assistant that can answer questions about weather and travel."
    )
    
    max_iterations: Optional[int] = Field(
        None,
        description="Maximum number of ReAct iterations (reasoning + acting cycles). "
                   "If not provided, uses the default from settings.",
        ge=1,
        le=50,
        example=10
    )
    
    return_format: str = Field(
        "text",
        description="Response format: 'text' for plain text or 'structured' for detailed JSON.",
        regex="^(text|structured)$",
        example="text"
    )
    
    @validator("conversation_id")
    def validate_conversation_id(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate conversation ID format.
        
        Args:
            v: The conversation ID to validate.
            
        Returns:
            The validated conversation ID.
            
        Raises:
            ValueError: If the conversation ID is invalid.
        """
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Conversation ID must be a string")
        
        if len(v) > 100:
            raise ValueError("Conversation ID must be 100 characters or less")
        
        return v
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "message": "What's the weather in Tokyo tomorrow?",
                "conversation_id": "conv_12345",
                "system_prompt": "You are a helpful assistant.",
                "max_iterations": 10,
                "return_format": "text"
            }
        }


class ToolCall(BaseModel):
    """
    Represents a tool call made by the agent.
    
    This model captures information about a specific tool invocation
    during the agent's execution.
    """
    
    tool_name: str = Field(
        ...,
        description="Name of the tool that was called.",
        example="weather_query"
    )
    
    arguments: Dict[str, Any] = Field(
        ...,
        description="Arguments passed to the tool.",
        example={"city": "Tokyo", "date": "2024-01-01"}
    )
    
    result: Optional[Any] = Field(
        None,
        description="Result returned by the tool execution.",
        example="Weather in Tokyo on 2024-01-01: Sunny, 22°C. (Mock Data)"
    )
    
    success: bool = Field(
        ...,
        description="Whether the tool execution was successful.",
        example=True
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if the tool execution failed.",
        example="Invalid date format"
    )
    
    execution_time_ms: Optional[float] = Field(
        None,
        description="Tool execution time in milliseconds.",
        example=15.5
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the tool call."
    )


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    
    Represents the agent's response to a user's chat message,
    including any tool calls made during processing.
    """
    
    response: str = Field(
        ...,
        description="The agent's response message.",
        example="The weather in Tokyo tomorrow is expected to be sunny with a temperature of 22°C."
    )
    
    conversation_id: str = Field(
        ...,
        description="Conversation ID for this interaction.",
        example="conv_12345"
    )
    
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls made during agent execution."
    )
    
    iteration_count: int = Field(
        ...,
        description="Number of ReAct iterations performed.",
        ge=0,
        example=2
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds.",
        ge=0.0,
        example=1250.5
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the response."
    )
    
    @validator("conversation_id")
    def ensure_conversation_id(cls, v: str) -> str:
        """
        Ensure conversation ID is present.
        
        Args:
            v: The conversation ID.
            
        Returns:
            The conversation ID, generating a new one if empty.
        """
        if not v:
            return f"conv_{uuid4().hex[:8]}"
        return v
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "response": "The weather in Tokyo tomorrow is expected to be sunny with a temperature of 22°C.",
                "conversation_id": "conv_12345",
                "tool_calls": [
                    {
                        "tool_name": "weather_query",
                        "arguments": {"city": "Tokyo", "date": "2024-01-01"},
                        "result": "Weather in Tokyo on 2024-01-01: Sunny, 22°C. (Mock Data)",
                        "success": True,
                        "error_message": None,
                        "execution_time_ms": 15.5,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                ],
                "iteration_count": 2,
                "processing_time_ms": 1250.5,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Provides information about the API service health and status.
    """
    
    status: str = Field(
        ...,
        description="Service status: 'healthy', 'degraded', or 'unhealthy'.",
        example="healthy"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the health check."
    )
    
    version: str = Field(
        ...,
        description="API version.",
        example="0.1.0"
    )
    
    uptime_seconds: Optional[float] = Field(
        None,
        description="Service uptime in seconds.",
        ge=0.0,
        example=3600.5
    )
    
    dependencies: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of external dependencies (e.g., LLM API, database)."
    )
    
    message: Optional[str] = Field(
        None,
        description="Additional status message.",
        example="All systems operational"
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "0.1.0",
                "uptime_seconds": 3600.5,
                "dependencies": {
                    "deepseek_api": True,
                    "tool_registry": True
                },
                "message": "All systems operational"
            }
        }


class ToolInputSchema(BaseModel):
    """
    Schema for tool input parameters.
    
    This is a dynamic model that can be used to validate tool inputs.
    """
    
    # This is a generic model that can be extended for specific tools
    # In practice, each tool would have its own input schema
    
    class Config:
        """Pydantic model configuration."""
        extra = "allow"  # Allow additional fields for dynamic schemas


class ToolInfo(BaseModel):
    """
    Information about a registered tool.
    
    Provides metadata about a tool available in the system.
    """
    
    name: str = Field(
        ...,
        description="Tool name.",
        example="weather_query"
    )
    
    description: str = Field(
        ...,
        description="Tool description.",
        example="Get weather information for a city on a specific date."
    )
    
    enabled: bool = Field(
        ...,
        description="Whether the tool is currently enabled.",
        example=True
    )
    
    input_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for tool input parameters.",
        example={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["city", "date"]
        }
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional tool metadata.",
        example={"category": "weather", "version": "1.0.0"}
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "name": "weather_query",
                "description": "Get weather information for a city on a specific date.",
                "enabled": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                    },
                    "required": ["city", "date"]
                },
                "metadata": {"category": "weather", "version": "1.0.0"}
            }
        }


class ToolListResponse(BaseModel):
    """
    Response model for tools listing endpoint.
    
    Provides a list of all available tools in the system.
    """
    
    tools: List[ToolInfo] = Field(
        ...,
        description="List of available tools."
    )
    
    count: int = Field(
        ...,
        description="Total number of tools.",
        ge=0,
        example=2
    )
    
    enabled_count: int = Field(
        ...,
        description="Number of enabled tools.",
        ge=0,
        example=2
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "tools": [
                    {
                        "name": "weather_query",
                        "description": "Get weather information for a city on a specific date.",
                        "enabled": True,
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                            },
                            "required": ["city", "date"]
                        },
                        "metadata": {"category": "weather", "version": "1.0.0"}
                    },
                    {
                        "name": "travel_query",
                        "description": "Get travel options between two cities.",
                        "enabled": True,
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "from_city": {"type": "string", "description": "Departure city"},
                                "to_city": {"type": "string", "description": "Destination city"},
                                "travel_date": {"type": "string", "description": "Optional travel date"},
                                "travel_type": {"type": "string", "description": "Type of travel: flight, train, bus, any"},
                                "max_results": {"type": "integer", "description": "Maximum number of results"}
                            },
                            "required": ["from_city", "to_city"]
                        },
                        "metadata": {"category": "travel", "version": "1.0.0"}
                    }
                ],
                "count": 2,
                "enabled_count": 2
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Provides consistent error responses across all API endpoints.
    """
    
    error: str = Field(
        ...,
        description="Error message.",
        example="Invalid request parameters"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Detailed error information.",
        example="The 'date' parameter must be in YYYY-MM-DD format"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the error."
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier for debugging.",
        example="req_12345"
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "error": "Invalid request parameters",
                "detail": "The 'date' parameter must be in YYYY-MM-DD format",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_12345"
            }
        }


class ConversationState(BaseModel):
    """
    State of a conversation for multi-turn interactions.
    
    Tracks the history and context of a conversation.
    """
    
    conversation_id: str = Field(
        ...,
        description="Unique conversation identifier.",
        example="conv_12345"
    )
    
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of messages in the conversation."
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Conversation creation timestamp."
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp."
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation metadata."
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "conversation_id": "conv_12345",
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                    {"role": "assistant", "content": "The weather in Tokyo is sunny."}
                ],
                "created_at": "2024-01-01T12:00:00Z",
                "last_updated": "2024-01-01T12:01:00Z",
                "metadata": {"user_id": "user_123", "topic": "weather"}
            }
        }


class ConversationResponse(BaseModel):
    """
    Response model for conversation retrieval.
    
    Provides information about a specific conversation.
    """
    
    conversation: ConversationState = Field(
        ...,
        description="Conversation state."
    )
    
    message_count: int = Field(
        ...,
        description="Number of messages in the conversation.",
        ge=0,
        example=5
    )
    
    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "conversation": {
                    "conversation_id": "conv_12345",
                    "messages": [
                        {"role": "user", "content": "What's the weather in Tokyo?"},
                        {"role": "assistant", "content": "The weather in Tokyo is sunny."}
                    ],
                    "created_at": "2024-01-01T12:00:00Z",
                    "last_updated": "2024-01-01T12:01:00Z",
                    "metadata": {"user_id": "user_123", "topic": "weather"}
                },
                "message_count": 2
            }
        }


# Export all schemas
__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ToolCall",
    "HealthResponse",
    "ToolInfo",
    "ToolListResponse",
    "ErrorResponse",
    "ConversationState",
    "ConversationResponse",
    "ToolInputSchema",
]
"""
DeepSeek LLM client configuration for the ReAct Agent system.

This module provides a factory function to create and configure a DeepSeek LLM client
using the OpenAI-compatible API interface through LangChain.
"""

import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .settings import Settings

logger = logging.getLogger(__name__)


def create_deepseek_llm(settings: Settings) -> ChatOpenAI:
    """
    Create and configure a DeepSeek LLM client using the OpenAI-compatible interface.
    
    Args:
        settings: Application settings containing DeepSeek API configuration.
        
    Returns:
        Configured ChatOpenAI instance for DeepSeek API.
        
    Raises:
        ValueError: If required API key is missing or invalid.
        
    Example:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> llm = create_deepseek_llm(settings)
        >>> response = llm.invoke("Hello, how are you?")
    """
    # Get DeepSeek configuration from settings
    deepseek_config = settings.get_deepseek_config()
    
    # Validate required configuration
    api_key = deepseek_config.get("api_key")
    if not api_key:
        raise ValueError(
            "DeepSeek API key is required. "
            "Please set the DEEPSEEK_API_KEY environment variable."
        )
    
    base_url = deepseek_config.get("base_url", "https://api.deepseek.com")
    temperature = deepseek_config.get("temperature", 0.7)
    max_tokens = deepseek_config.get("max_tokens", 1000)
    
    logger.info(
        f"Creating DeepSeek LLM client with base_url={base_url}, "
        f"temperature={temperature}, max_tokens={max_tokens}"
    )
    
    # Create the LLM client
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30.0,  # 30 second timeout
        max_retries=3,  # Retry up to 3 times on failure
    )
    
    # Configure additional settings
    llm.model_kwargs = {
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    
    return llm


def create_deepseek_llm_with_custom_config(
    api_key: str,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: str = "deepseek-chat",
    **kwargs: Any
) -> ChatOpenAI:
    """
    Create a DeepSeek LLM client with custom configuration.
    
    Args:
        api_key: DeepSeek API key (required).
        base_url: Base URL for the API (defaults to "https://api.deepseek.com").
        temperature: Sampling temperature (0.0 to 2.0, defaults to 0.7).
        max_tokens: Maximum tokens to generate (defaults to 1000).
        model: Model name (defaults to "deepseek-chat").
        **kwargs: Additional keyword arguments passed to ChatOpenAI.
        
    Returns:
        Configured ChatOpenAI instance.
        
    Raises:
        ValueError: If API key is missing.
        
    Example:
        >>> llm = create_deepseek_llm_with_custom_config(
        ...     api_key="your-api-key",
        ...     temperature=0.5,
        ...     max_tokens=500
        ... )
    """
    if not api_key:
        raise ValueError("API key is required for DeepSeek LLM client.")
    
    # Use defaults if not provided
    base_url = base_url or "https://api.deepseek.com"
    temperature = temperature or 0.7
    max_tokens = max_tokens or 1000
    
    logger.info(
        f"Creating custom DeepSeek LLM client with model={model}, "
        f"base_url={base_url}, temperature={temperature}, max_tokens={max_tokens}"
    )
    
    # Create the LLM client
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=kwargs.pop("timeout", 30.0),
        max_retries=kwargs.pop("max_retries", 3),
        **kwargs
    )
    
    # Set default model kwargs if not provided
    if "model_kwargs" not in kwargs:
        llm.model_kwargs = {
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    
    return llm


def format_messages_for_llm(
    messages: list,
    system_prompt: Optional[str] = None
) -> list[BaseMessage]:
    """
    Format messages for LLM consumption.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        system_prompt: Optional system prompt to prepend.
        
    Returns:
        List of formatted BaseMessage objects.
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> formatted = format_messages_for_llm(messages)
    """
    formatted_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_messages.append(SystemMessage(content=system_prompt))
    
    # Format each message
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))
        elif role == "system":
            formatted_messages.append(SystemMessage(content=content))
        else:
            logger.warning(f"Unknown message role: {role}. Treating as user message.")
            formatted_messages.append(HumanMessage(content=content))
    
    return formatted_messages


def get_llm_config_summary(llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Get a summary of LLM configuration.
    
    Args:
        llm: Configured ChatOpenAI instance.
        
    Returns:
        Dictionary with LLM configuration summary.
        
    Example:
        >>> llm = create_deepseek_llm(settings)
        >>> summary = get_llm_config_summary(llm)
        >>> print(summary["model"])  # "deepseek-chat"
    """
    return {
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "base_url": llm.base_url,
        "timeout": llm.timeout,
        "max_retries": llm.max_retries,
        "model_kwargs": llm.model_kwargs,
    }


def validate_llm_connection(llm: ChatOpenAI) -> bool:
    """
    Validate that the LLM client can connect to the API.
    
    Args:
        llm: Configured ChatOpenAI instance.
        
    Returns:
        True if connection is successful, False otherwise.
        
    Example:
        >>> llm = create_deepseek_llm(settings)
        >>> if validate_llm_connection(llm):
        ...     print("LLM connection successful")
        ... else:
        ...     print("LLM connection failed")
    """
    try:
        # Try a simple, low-cost request to validate connection
        test_response = llm.invoke("Hello")
        return test_response is not None and hasattr(test_response, "content")
    except Exception as e:
        logger.error(f"LLM connection validation failed: {e}")
        return False
"""
Configuration settings for the ReAct Agent system.

This module defines the Settings class that loads and validates environment
variables for the application using Pydantic Settings.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class manages all configuration for the ReAct Agent system,
    including API keys, server settings, and agent parameters.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # DeepSeek API Configuration
    DEEPSEEK_API_KEY: str = Field(
        ...,
        description="API key for DeepSeek LLM service. Required for agent operation.",
        min_length=1
    )
    
    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        description="Base URL for DeepSeek API. Defaults to official endpoint."
    )
    
    # Agent Configuration
    AGENT_MAX_ITERATIONS: int = Field(
        default=10,
        description="Maximum number of reasoning iterations for the ReAct agent.",
        ge=1,
        le=50
    )
    
    AGENT_TEMPERATURE: float = Field(
        default=0.7,
        description="Temperature parameter for LLM sampling.",
        ge=0.0,
        le=2.0
    )
    
    AGENT_MAX_TOKENS: int = Field(
        default=1000,
        description="Maximum tokens to generate per LLM call.",
        ge=100,
        le=4000
    )
    
    # Server Configuration
    SERVER_HOST: str = Field(
        default="0.0.0.0",
        description="Host address for the FastAPI server."
    )
    
    SERVER_PORT: int = Field(
        default=8000,
        description="Port for the FastAPI server.",
        ge=1024,
        le=65535
    )
    
    SERVER_RELOAD: bool = Field(
        default=False,
        description="Enable auto-reload for development."
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages."
    )
    
    # Tool Configuration
    WEATHER_TOOL_ENABLED: bool = Field(
        default=True,
        description="Enable the weather query tool."
    )
    
    TRAVEL_TOOL_ENABLED: bool = Field(
        default=True,
        description="Enable the travel query tool."
    )
    
    # Optional: Conversation persistence
    CONVERSATION_STORAGE_PATH: Optional[str] = Field(
        default=None,
        description="Optional path for storing conversation history."
    )
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate that LOG_LEVEL is a valid logging level.
        
        Args:
            v: The log level string to validate.
            
        Returns:
            The validated log level in uppercase.
            
        Raises:
            ValueError: If the log level is not valid.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {v}. Must be one of {valid_levels}"
            )
        return v_upper
    
    @field_validator("DEEPSEEK_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """
        Validate that DEEPSEEK_API_KEY is not empty.
        
        Args:
            v: The API key string to validate.
            
        Returns:
            The validated API key.
            
        Raises:
            ValueError: If the API key is empty.
        """
        if not v or v.strip() == "":
            raise ValueError("DEEPSEEK_API_KEY cannot be empty")
        return v.strip()
    
    @field_validator("CONVERSATION_STORAGE_PATH")
    @classmethod
    def validate_storage_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate conversation storage path if provided.
        
        Args:
            v: The storage path string to validate.
            
        Returns:
            The validated path, or None if not provided.
            
        Raises:
            ValueError: If the path contains invalid characters.
        """
        if v is None:
            return None
        
        v = v.strip()
        if not v:
            return None
        
        # Check for basic path validity
        if ".." in v or "//" in v:
            raise ValueError(
                "CONVERSATION_STORAGE_PATH cannot contain '..' or '//'"
            )
        
        return v
    
    def get_deepseek_config(self) -> dict:
        """
        Get DeepSeek API configuration as a dictionary.
        
        Returns:
            Dictionary with DeepSeek configuration parameters.
        """
        return {
            "api_key": self.DEEPSEEK_API_KEY,
            "base_url": self.DEEPSEEK_BASE_URL,
            "temperature": self.AGENT_TEMPERATURE,
            "max_tokens": self.AGENT_MAX_TOKENS
        }
    
    def get_server_config(self) -> dict:
        """
        Get server configuration as a dictionary.
        
        Returns:
            Dictionary with server configuration parameters.
        """
        return {
            "host": self.SERVER_HOST,
            "port": self.SERVER_PORT,
            "reload": self.SERVER_RELOAD
        }
    
    def get_agent_config(self) -> dict:
        """
        Get agent configuration as a dictionary.
        
        Returns:
            Dictionary with agent configuration parameters.
        """
        return {
            "max_iterations": self.AGENT_MAX_ITERATIONS,
            "temperature": self.AGENT_TEMPERATURE,
            "max_tokens": self.AGENT_MAX_TOKENS
        }


def get_settings() -> Settings:
    """
    Factory function to get application settings.
    
    Returns:
        Settings instance loaded from environment variables.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.DEEPSEEK_API_KEY)
    """
    return Settings()


# Create a global settings instance for easy access
settings = get_settings()


if __name__ == "__main__":
    # Test the settings loading
    test_settings = get_settings()
    print("Settings loaded successfully:")
    print(f"  DeepSeek API Key: {'*' * 8}{test_settings.DEEPSEEK_API_KEY[-4:]}")
    print(f"  DeepSeek Base URL: {test_settings.DEEPSEEK_BASE_URL}")
    print(f"  Agent Max Iterations: {test_settings.AGENT_MAX_ITERATIONS}")
    print(f"  Server Host: {test_settings.SERVER_HOST}")
    print(f"  Server Port: {test_settings.SERVER_PORT}")
    print(f"  Log Level: {test_settings.LOG_LEVEL}")
    print(f"  Weather Tool Enabled: {test_settings.WEATHER_TOOL_ENABLED}")
    print(f"  Travel Tool Enabled: {test_settings.TRAVEL_TOOL_ENABLED}")
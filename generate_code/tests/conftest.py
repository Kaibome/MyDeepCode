"""
Pytest configuration and shared fixtures for the ReAct Agent test suite.

This module provides shared fixtures and configuration for all test modules.
It includes fixtures for:
- Settings and configuration
- Tool instances and registry
- Agent components
- Test clients and utilities
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.config.settings import Settings, get_settings
from src.tools.tool_registry import ToolRegistry
from src.tools.weather_tool import WeatherQueryTool, create_weather_tool
from src.tools.travel_tool import TravelQueryTool, create_travel_tool
from src.agents.agent_builder import build_react_agent
from src.api.fastapi_app import get_app


# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the test session.
    
    Returns:
        An asyncio event loop instance.
        
    Yields:
        asyncio.AbstractEventLoop: The event loop for async tests.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Create test settings with environment variables for testing.
    
    Returns:
        Settings: A Settings instance configured for testing.
        
    Note:
        Uses a test API key and disables external API calls.
    """
    # Set test environment variables
    os.environ["DEEPSEEK_API_KEY"] = "test-api-key-1234567890"
    os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
    os.environ["AGENT_MAX_ITERATIONS"] = "3"
    os.environ["AGENT_TEMPERATURE"] = "0.1"
    os.environ["AGENT_MAX_TOKENS"] = "100"
    os.environ["SERVER_HOST"] = "127.0.0.1"
    os.environ["SERVER_PORT"] = "8000"
    os.environ["SERVER_RELOAD"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["LOG_FORMAT"] = "simple"
    os.environ["WEATHER_TOOL_ENABLED"] = "true"
    os.environ["TRAVEL_TOOL_ENABLED"] = "true"
    
    # Create a temporary directory for conversation storage
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["CONVERSATION_STORAGE_PATH"] = temp_dir
        settings = get_settings()
        yield settings
    
    # Clean up environment variables
    for key in [
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "AGENT_MAX_ITERATIONS",
        "AGENT_TEMPERATURE", "AGENT_MAX_TOKENS", "SERVER_HOST", "SERVER_PORT",
        "SERVER_RELOAD", "LOG_LEVEL", "LOG_FORMAT", "WEATHER_TOOL_ENABLED",
        "TRAVEL_TOOL_ENABLED", "CONVERSATION_STORAGE_PATH"
    ]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture(scope="function")
def mock_settings() -> Settings:
    """
    Create mock settings for unit tests.
    
    Returns:
        Settings: A Settings instance with minimal configuration.
        
    Note:
        Uses mock values and disables external dependencies.
    """
    return Settings(
        DEEPSEEK_API_KEY="mock-api-key",
        DEEPSEEK_BASE_URL="https://mock.deepseek.com",
        AGENT_MAX_ITERATIONS=2,
        AGENT_TEMPERATURE=0.1,
        AGENT_MAX_TOKENS=50,
        SERVER_HOST="127.0.0.1",
        SERVER_PORT=9999,
        SERVER_RELOAD=False,
        LOG_LEVEL="WARNING",
        LOG_FORMAT="simple",
        WEATHER_TOOL_ENABLED=True,
        TRAVEL_TOOL_ENABLED=True,
        CONVERSATION_STORAGE_PATH=tempfile.gettempdir(),
    )


@pytest.fixture(scope="function")
def weather_tool() -> WeatherQueryTool:
    """
    Create a WeatherQueryTool instance for testing.
    
    Returns:
        WeatherQueryTool: A configured weather tool instance.
    """
    return create_weather_tool()


@pytest.fixture(scope="function")
def travel_tool() -> TravelQueryTool:
    """
    Create a TravelQueryTool instance for testing.
    
    Returns:
        TravelQueryTool: A configured travel tool instance.
    """
    return create_travel_tool()


@pytest.fixture(scope="function")
def tool_registry(weather_tool: WeatherQueryTool, travel_tool: TravelQueryTool) -> ToolRegistry:
    """
    Create a ToolRegistry with test tools registered.
    
    Args:
        weather_tool: The weather tool fixture.
        travel_tool: The travel tool fixture.
        
    Returns:
        ToolRegistry: A tool registry with weather and travel tools registered.
    """
    registry = ToolRegistry()
    
    # Register weather tool
    registry.register_tool(
        name="weather_query",
        tool_instance=weather_tool,
        description="Get weather information for a city on a specific date",
        input_schema=weather_tool.input_schema if hasattr(weather_tool, 'input_schema') else None,
        is_enabled=True,
        metadata={"category": "weather", "mock": True}
    )
    
    # Register travel tool
    registry.register_tool(
        name="travel_query",
        tool_instance=travel_tool,
        description="Find travel options between two cities",
        input_schema=travel_tool.input_schema if hasattr(travel_tool, 'input_schema') else None,
        is_enabled=True,
        metadata={"category": "travel", "mock": True}
    )
    
    return registry


@pytest.fixture(scope="function")
def empty_tool_registry() -> ToolRegistry:
    """
    Create an empty ToolRegistry for testing.
    
    Returns:
        ToolRegistry: An empty tool registry instance.
    """
    return ToolRegistry()


@pytest.fixture(scope="function")
def mock_llm():
    """
    Create a mock LLM for testing.
    
    Returns:
        dict: A mock LLM configuration dictionary.
        
    Note:
        This fixture returns a simple mock that can be used in place of
        a real LLM for testing agent logic without API calls.
    """
    class MockLLM:
        """Mock LLM for testing."""
        
        def __init__(self):
            self.model_name = "mock-llm"
            self.temperature = 0.1
            self.max_tokens = 100
            self.api_key = "mock-key"
            self.base_url = "https://mock.api.com"
            
        def invoke(self, messages, **kwargs):
            """Mock invoke method."""
            # Return a simple response based on input
            last_message = messages[-1].content if messages else ""
            
            if "weather" in last_message.lower():
                return {
                    "content": "I'll use the weather_query tool to get weather information.",
                    "tool_calls": [{
                        "name": "weather_query",
                        "args": {"city": "Paris", "date": "2024-01-01"},
                        "id": "call_123"
                    }]
                }
            elif "travel" in last_message.lower():
                return {
                    "content": "I'll use the travel_query tool to find travel options.",
                    "tool_calls": [{
                        "name": "travel_query",
                        "args": {"from_city": "Berlin", "to_city": "Munich"},
                        "id": "call_456"
                    }]
                }
            else:
                return {
                    "content": "This is a mock response from the LLM.",
                    "tool_calls": []
                }
    
    return MockLLM()


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """
    Create a FastAPI TestClient for API testing.
    
    Returns:
        TestClient: A FastAPI test client instance.
    """
    app = get_app()
    return TestClient(app)


@pytest.fixture(scope="function")
def sample_chat_request() -> Dict[str, Any]:
    """
    Create a sample chat request for testing.
    
    Returns:
        Dict[str, Any]: A sample chat request dictionary.
    """
    return {
        "message": "What's the weather in Paris tomorrow?",
        "conversation_id": "test-conv-123",
        "system_prompt": "You are a helpful assistant.",
        "max_iterations": 3,
        "return_format": "text"
    }


@pytest.fixture(scope="function")
def sample_tool_execution_args() -> Dict[str, Any]:
    """
    Create sample tool execution arguments for testing.
    
    Returns:
        Dict[str, Any]: Sample arguments for tool execution.
    """
    return {
        "city": "London",
        "date": "2024-12-25"
    }


@pytest.fixture(scope="function")
def sample_travel_args() -> Dict[str, Any]:
    """
    Create sample travel tool arguments for testing.
    
    Returns:
        Dict[str, Any]: Sample arguments for travel tool.
    """
    return {
        "from_city": "New York",
        "to_city": "Los Angeles",
        "travel_date": "2024-07-15",
        "travel_type": "flight",
        "max_results": 3
    }


@pytest.fixture(scope="function")
def temp_conversation_file() -> Generator[Path, None, None]:
    """
    Create a temporary conversation storage file.
    
    Yields:
        Path: Path to the temporary conversation file.
        
    Note:
        The file is automatically cleaned up after the test.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Write empty conversation data
        json.dump({"conversations": {}}, f)
        temp_file = Path(f.name)
    
    yield temp_file
    
    # Clean up
    if temp_file.exists():
        temp_file.unlink()


@pytest.fixture(scope="function")
def agent_state_class():
    """
    Provide the AgentState class for testing.
    
    Returns:
        type: The AgentState class.
        
    Note:
        This fixture imports the class dynamically to avoid circular imports.
    """
    from src.agents.react_agent import AgentState
    return AgentState


@pytest.fixture(scope="function")
def sample_agent_state(agent_state_class) -> Any:
    """
    Create a sample AgentState instance for testing.
    
    Args:
        agent_state_class: The AgentState class fixture.
        
    Returns:
        AgentState: A sample AgentState instance.
    """
    return agent_state_class(
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "I'll check the weather for you."}
        ],
        tool_calls=[
            {
                "name": "weather_query",
                "args": {"city": "Tokyo", "date": "2024-01-01"},
                "id": "call_789"
            }
        ],
        intermediate_steps=[],
        iteration=1
    )


@pytest.fixture(scope="function")
def mock_tool_execution_result() -> Dict[str, Any]:
    """
    Create a mock tool execution result for testing.
    
    Returns:
        Dict[str, Any]: A mock tool execution result.
    """
    return {
        "success": True,
        "result": "Weather in Paris on 2024-01-01: Sunny, 22°C. (Mock Data)",
        "execution_time": 0.05,
        "error": None,
        "tool_name": "weather_query",
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Get the test data directory.
    
    Returns:
        Path: Path to the test data directory.
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="function")
def create_test_tool():
    """
    Create a factory function for test tools.
    
    Returns:
        Callable: A function that creates test tools.
    """
    def _create_tool(name: str, description: str, func: callable) -> Dict[str, Any]:
        """
        Create a test tool dictionary.
        
        Args:
            name: Tool name.
            description: Tool description.
            func: Tool function.
            
        Returns:
            Dict[str, Any]: Tool dictionary.
        """
        return {
            "name": name,
            "instance": func,
            "description": description,
            "input_schema": None,
            "metadata": {"test": True}
        }
    
    return _create_tool


@pytest.fixture(scope="function")
def validate_response_schema():
    """
    Create a response schema validation function.
    
    Returns:
        Callable: A function that validates response schemas.
    """
    def _validate(response_data: Dict[str, Any], expected_fields: List[str]) -> bool:
        """
        Validate that response contains expected fields.
        
        Args:
            response_data: Response data to validate.
            expected_fields: List of expected field names.
            
        Returns:
            bool: True if all expected fields are present.
            
        Raises:
            AssertionError: If validation fails.
        """
        missing_fields = [field for field in expected_fields if field not in response_data]
        if missing_fields:
            raise AssertionError(f"Missing fields in response: {missing_fields}")
        return True
    
    return _validate


# Custom pytest markers
def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires external components)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end test (requires full system)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test"
    )


# Skip tests that require external APIs by default
def pytest_collection_modifyitems(config, items):
    """
    Skip integration and e2e tests by default unless explicitly requested.
    
    Args:
        config: Pytest configuration object.
        items: List of test items.
    """
    skip_integration = pytest.mark.skip(reason="integration test - run with -m integration")
    skip_e2e = pytest.mark.skip(reason="e2e test - run with -m e2e")
    skip_slow = pytest.mark.skip(reason="slow test - run with -m slow")
    
    for item in items:
        if "integration" in item.keywords and not config.getoption("-m", "").startswith("integration"):
            item.add_marker(skip_integration)
        if "e2e" in item.keywords and not config.getoption("-m", "").startswith("e2e"):
            item.add_marker(skip_e2e)
        if "slow" in item.keywords and not config.getoption("-m", "").startswith("slow"):
            item.add_marker(skip_slow)
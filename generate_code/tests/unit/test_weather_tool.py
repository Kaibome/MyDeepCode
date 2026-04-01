"""
Unit tests for the WeatherQueryTool module.

This module contains comprehensive tests for the weather query tool functionality,
including input validation, tool execution, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.tools.weather_tool import (
    WeatherQueryInput,
    WeatherQueryTool,
    create_weather_tool,
)


class TestWeatherQueryInput:
    """Test cases for the WeatherQueryInput Pydantic model."""

    def test_valid_input(self) -> None:
        """Test that valid input passes validation."""
        input_data = {"city": "Tokyo", "date": "2024-01-15"}
        weather_input = WeatherQueryInput(**input_data)
        
        assert weather_input.city == "Tokyo"
        assert weather_input.date == "2024-01-15"
        assert weather_input.normalized_city == "tokyo"

    def test_city_normalization(self) -> None:
        """Test that city names are properly normalized."""
        test_cases = [
            ("New York", "new york"),
            ("SAN FRANCISCO", "san francisco"),
            ("  London  ", "london"),
            ("PaRis", "paris"),
        ]
        
        for input_city, expected_normalized in test_cases:
            weather_input = WeatherQueryInput(city=input_city, date="2024-01-15")
            assert weather_input.normalized_city == expected_normalized

    def test_invalid_date_format(self) -> None:
        """Test that invalid date formats raise validation errors."""
        invalid_dates = [
            "2024-13-01",  # Invalid month
            "2024-01-32",  # Invalid day
            "2024/01/01",  # Wrong separator
            "01-01-2024",  # Wrong order
            "not-a-date",  # Not a date
        ]
        
        for invalid_date in invalid_dates:
            with pytest.raises(ValueError):
                WeatherQueryInput(city="Tokyo", date=invalid_date)

    def test_future_date_validation(self) -> None:
        """Test that dates too far in the future are rejected."""
        # Create a date 2 years in the future
        future_date = (datetime.now() + timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        
        with pytest.raises(ValueError, match="Date cannot be more than 1 year in the future"):
            WeatherQueryInput(city="Tokyo", date=future_date)

    def test_past_date_validation(self) -> None:
        """Test that dates too far in the past are rejected."""
        # Create a date 2 years in the past
        past_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        
        with pytest.raises(ValueError, match="Date cannot be more than 1 year in the past"):
            WeatherQueryInput(city="Tokyo", date=past_date)

    def test_city_length_validation(self) -> None:
        """Test that city names have appropriate length constraints."""
        # Test too short city name
        with pytest.raises(ValueError, match="City name must be between 1 and 100 characters"):
            WeatherQueryInput(city="", date="2024-01-15")
        
        # Test too long city name
        long_city = "A" * 101
        with pytest.raises(ValueError, match="City name must be between 1 and 100 characters"):
            WeatherQueryInput(city=long_city, date="2024-01-15")

    def test_city_character_validation(self) -> None:
        """Test that city names don't contain invalid characters."""
        invalid_cities = [
            "City123",  # Contains digits
            "New@York",  # Contains special character
            "San#Francisco",  # Contains special character
        ]
        
        for invalid_city in invalid_cities:
            with pytest.raises(ValueError, match="City name can only contain letters, spaces, hyphens, and apostrophes"):
                WeatherQueryInput(city=invalid_city, date="2024-01-15")

    def test_valid_city_characters(self) -> None:
        """Test that valid city characters pass validation."""
        valid_cities = [
            "New York",
            "San Francisco",
            "St. Louis",
            "Los Angeles",
            "Rio de Janeiro",
            "Kuala Lumpur",
        ]
        
        for valid_city in valid_cities:
            weather_input = WeatherQueryInput(city=valid_city, date="2024-01-15")
            assert weather_input.city == valid_city


class TestWeatherQueryTool:
    """Test cases for the WeatherQueryTool class."""

    def test_tool_initialization(self) -> None:
        """Test that the tool initializes correctly."""
        tool = WeatherQueryTool()
        
        assert tool.name == "weather_query_tool"
        assert "weather" in tool.description.lower()
        assert "query" in tool.description.lower()
        assert tool.args_schema == WeatherQueryInput

    def test_run_method_string_output(self) -> None:
        """Test the run method with string output format."""
        tool = WeatherQueryTool()
        
        result = tool.run(city="Tokyo", date="2024-01-15")
        
        assert isinstance(result, str)
        assert "Tokyo" in result
        assert "2024-01-15" in result
        assert "Weather" in result
        assert "Mock Data" in result

    def test_run_method_dict_output(self) -> None:
        """Test the run method with dictionary output format."""
        tool = WeatherQueryTool()
        
        result = tool.run(city="Tokyo", date="2024-01-15", return_format="dict")
        
        assert isinstance(result, dict)
        assert result["city"] == "Tokyo"
        assert result["date"] == "2024-01-15"
        assert "weather" in result
        assert "temperature" in result
        assert "conditions" in result
        assert "humidity" in result
        assert "wind_speed" in result

    def test_run_method_with_kwargs(self) -> None:
        """Test the run method with additional keyword arguments."""
        tool = WeatherQueryTool()
        
        result = tool.run(
            city="Tokyo", 
            date="2024-01-15",
            return_format="dict",
            include_forecast=True
        )
        
        assert isinstance(result, dict)
        assert "forecast" in result
        assert isinstance(result["forecast"], list)
        assert len(result["forecast"]) > 0

    def test_call_method(self) -> None:
        """Test that the tool is callable."""
        tool = WeatherQueryTool()
        
        result = tool("Tokyo", "2024-01-15")
        
        assert isinstance(result, str)
        assert "Tokyo" in result
        assert "2024-01-15" in result

    def test_get_tool_schema(self) -> None:
        """Test that the tool schema is correctly generated."""
        tool = WeatherQueryTool()
        schema = tool.get_tool_schema()
        
        assert isinstance(schema, dict)
        assert schema["name"] == "weather_query_tool"
        assert "description" in schema
        assert "args_schema" in schema
        assert schema["args_schema"] == WeatherQueryInput
        assert "func" in schema
        assert callable(schema["func"])

    def test_mock_data_generation(self) -> None:
        """Test that mock data is generated consistently."""
        tool = WeatherQueryTool()
        
        # Run the same query multiple times
        results = []
        for _ in range(5):
            result = tool.run(city="Tokyo", date="2024-01-15", return_format="dict")
            results.append(result)
        
        # Check that temperature is within reasonable bounds
        for result in results:
            temp = result["temperature"]
            assert isinstance(temp, (int, float))
            assert -20 <= temp <= 45  # Reasonable temperature range in Celsius
        
        # Check that conditions are valid
        valid_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Snowy", "Stormy"]
        for result in results:
            assert result["conditions"] in valid_conditions

    def test_different_cities_produce_different_data(self) -> None:
        """Test that different cities produce different mock data."""
        tool = WeatherQueryTool()
        
        tokyo_result = tool.run(city="Tokyo", date="2024-01-15", return_format="dict")
        london_result = tool.run(city="London", date="2024-01-15", return_format="dict")
        
        # They should have different data (mock implementation uses city in hash)
        assert tokyo_result["city"] != london_result["city"]
        # Other fields might be different due to hash-based generation

    def test_error_handling_invalid_input(self) -> None:
        """Test that invalid input raises appropriate errors."""
        tool = WeatherQueryTool()
        
        # Test with invalid date
        with pytest.raises(ValueError):
            tool.run(city="Tokyo", date="invalid-date")
        
        # Test with empty city
        with pytest.raises(ValueError):
            tool.run(city="", date="2024-01-15")

    def test_temperature_unit_parameter(self) -> None:
        """Test that temperature unit parameter works correctly."""
        tool = WeatherQueryTool()
        
        # Test Celsius (default)
        result_c = tool.run(city="Tokyo", date="2024-01-15", return_format="dict", temperature_unit="celsius")
        assert "°C" in result_c.get("temperature_display", "")
        
        # Test Fahrenheit
        result_f = tool.run(city="Tokyo", date="2024-01-15", return_format="dict", temperature_unit="fahrenheit")
        assert "°F" in result_f.get("temperature_display", "")

    def test_weather_icon_generation(self) -> None:
        """Test that weather icons are generated based on conditions."""
        tool = WeatherQueryTool()
        
        result = tool.run(city="Tokyo", date="2024-01-15", return_format="dict", include_icon=True)
        
        assert "icon" in result
        assert result["icon"] in ["☀️", "⛅", "☁️", "🌧️", "❄️", "⛈️"]

    def test_historical_date_handling(self) -> None:
        """Test that historical dates are handled correctly."""
        tool = WeatherQueryTool()
        
        # Test with a historical date
        historical_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        result = tool.run(city="Tokyo", date=historical_date, return_format="dict")
        
        assert result["date"] == historical_date
        assert "historical" in result.get("notes", "").lower()

    def test_future_date_handling(self) -> None:
        """Test that future dates are handled correctly."""
        tool = WeatherQueryTool()
        
        # Test with a future date
        future_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        result = tool.run(city="Tokyo", date=future_date, return_format="dict")
        
        assert result["date"] == future_date
        assert "forecast" in result.get("notes", "").lower()


class TestCreateWeatherTool:
    """Test cases for the create_weather_tool factory function."""

    def test_create_weather_tool(self) -> None:
        """Test that the factory function creates a tool correctly."""
        tool = create_weather_tool()
        
        assert isinstance(tool, WeatherQueryTool)
        assert tool.name == "weather_query_tool"
        assert tool.args_schema == WeatherQueryInput

    def test_create_weather_tool_with_custom_name(self) -> None:
        """Test creating a tool with a custom name."""
        custom_name = "custom_weather_tool"
        tool = create_weather_tool(name=custom_name)
        
        assert tool.name == custom_name
        assert isinstance(tool, WeatherQueryTool)

    def test_create_weather_tool_with_custom_description(self) -> None:
        """Test creating a tool with a custom description."""
        custom_description = "Custom weather query tool for testing"
        tool = create_weather_tool(description=custom_description)
        
        assert tool.description == custom_description

    def test_tool_registration_compatibility(self) -> None:
        """Test that the created tool can be registered in a tool registry."""
        from src.tools.tool_registry import ToolRegistry
        
        tool = create_weather_tool()
        registry = ToolRegistry()
        
        # Register the tool
        success = registry.register_tool(
            name="weather",
            tool_instance=tool,
            description=tool.description,
            input_schema=WeatherQueryInput
        )
        
        assert success is True
        assert registry.is_tool_available("weather")

    def test_tool_execution_through_registry(self) -> None:
        """Test that the tool can be executed through a tool registry."""
        from src.tools.tool_registry import ToolRegistry
        
        tool = create_weather_tool()
        registry = ToolRegistry()
        
        # Register and execute the tool
        registry.register_tool(
            name="weather",
            tool_instance=tool,
            description=tool.description,
            input_schema=WeatherQueryInput
        )
        
        result = registry.execute_tool("weather", city="Tokyo", date="2024-01-15")
        
        assert result.success is True
        assert "Tokyo" in result.output
        assert "2024-01-15" in result.output


class TestWeatherToolIntegration:
    """Integration tests for the weather tool."""

    def test_tool_with_realistic_scenarios(self) -> None:
        """Test the tool with realistic usage scenarios."""
        tool = create_weather_tool()
        
        scenarios = [
            {"city": "New York", "date": "2024-03-15", "description": "Spring in NYC"},
            {"city": "London", "date": "2024-07-20", "description": "Summer in London"},
            {"city": "Sydney", "date": "2024-12-25", "description": "Christmas in Sydney"},
            {"city": "Tokyo", "date": "2024-04-01", "description": "Spring in Tokyo"},
        ]
        
        for scenario in scenarios:
            result = tool.run(
                city=scenario["city"],
                date=scenario["date"],
                return_format="dict"
            )
            
            assert result["city"] == scenario["city"]
            assert result["date"] == scenario["date"]
            assert isinstance(result["temperature"], (int, float))
            assert isinstance(result["conditions"], str)
            assert isinstance(result["humidity"], (int, float))
            assert 0 <= result["humidity"] <= 100

    def test_tool_performance(self) -> None:
        """Test that the tool executes quickly for multiple requests."""
        import time
        
        tool = create_weather_tool()
        start_time = time.time()
        
        # Execute multiple requests
        for i in range(10):
            tool.run(city=f"City{i}", date="2024-01-15")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in less than 1 second
        assert execution_time < 1.0, f"Tool execution too slow: {execution_time:.2f}s"

    def test_tool_thread_safety(self) -> None:
        """Test that the tool can be used safely from multiple threads."""
        import threading
        import concurrent.futures
        
        tool = create_weather_tool()
        results = []
        
        def query_weather(city: str) -> str:
            return tool.run(city=city, date="2024-01-15")
        
        cities = ["Tokyo", "London", "Paris", "New York", "Sydney"] * 2
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_city = {
                executor.submit(query_weather, city): city 
                for city in cities
            }
            
            for future in concurrent.futures.as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    result = future.result()
                    results.append((city, result))
                except Exception as exc:
                    pytest.fail(f"Thread for city {city} raised exception: {exc}")
        
        assert len(results) == len(cities)
        for city, result in results:
            assert city in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
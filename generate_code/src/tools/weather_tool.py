"""
Mock weather query tool for the ReAct agent system.

This module provides a WeatherQueryTool class that simulates weather queries
by returning formatted mock data. It includes validation for date formats
and parameter handling.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator

# Configure logger
logger = logging.getLogger(__name__)


class WeatherQueryInput(BaseModel):
    """
    Input schema for weather query tool.
    
    Attributes:
        city: Name of the city to query weather for.
        date: Date for the weather query in YYYY-MM-DD format.
    """
    
    city: str = Field(
        description="Name of the city to query weather for.",
        examples=["Tokyo", "New York", "London", "Paris"]
    )
    
    date: str = Field(
        description="Date for the weather query in YYYY-MM-DD format.",
        examples=["2024-01-15", "2024-12-25"]
    )
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """
        Validate that the date string is in YYYY-MM-DD format.
        
        Args:
            v: Date string to validate.
            
        Returns:
            Validated date string.
            
        Raises:
            ValueError: If date format is invalid.
        """
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD format.")
    
    @field_validator('city')
    @classmethod
    def validate_city(cls, v: str) -> str:
        """
        Validate and normalize city name.
        
        Args:
            v: City name to validate.
            
        Returns:
            Normalized city name (title case).
        """
        return v.strip().title()


class WeatherQueryTool:
    """
    Mock weather query tool that returns formatted weather information.
    
    This tool simulates weather queries by generating mock data based on
    city and date inputs. It's designed to be used within the ReAct agent
    system as a tool that the agent can call.
    
    Attributes:
        name: Tool name identifier.
        description: Tool description for the agent.
        args_schema: Input schema for tool arguments.
    """
    
    def __init__(self) -> None:
        """
        Initialize the weather query tool.
        """
        self.name: str = "weather_query"
        self.description: str = (
            "Query weather information for a specific city and date. "
            "Returns mock weather data including temperature, conditions, "
            "humidity, and wind speed."
        )
        self.args_schema: type = WeatherQueryInput
        
        logger.debug(f"Initialized {self.name} tool")
    
    def run(
        self,
        city: str,
        date: str,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute weather query and return mock weather data.
        
        Args:
            city: Name of the city to query weather for.
            date: Date for the weather query in YYYY-MM-DD format.
            **kwargs: Additional keyword arguments (ignored in mock implementation).
            
        Returns:
            Formatted weather information string or dictionary with detailed data.
            
        Raises:
            ValueError: If city is empty or date format is invalid.
            
        Example:
            >>> tool = WeatherQueryTool()
            >>> result = tool.run("Tokyo", "2024-01-15")
            >>> print(result)
            "Weather in Tokyo on 2024-01-15: Sunny, 22°C. (Mock Data)"
        """
        # Validate inputs using the schema
        try:
            validated_input = WeatherQueryInput(city=city, date=date)
            city = validated_input.city
            date = validated_input.date
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        logger.info(f"Querying weather for {city} on {date}")
        
        # Generate mock weather data based on city and date
        weather_data = self._generate_mock_weather(city, date)
        
        # Format the response
        formatted_response = self._format_weather_response(weather_data)
        
        logger.debug(f"Weather query completed for {city} on {date}")
        return formatted_response
    
    def _generate_mock_weather(self, city: str, date: str) -> Dict[str, Any]:
        """
        Generate mock weather data based on city and date.
        
        This method creates deterministic mock data by hashing the city
        and date to produce consistent results for the same inputs.
        
        Args:
            city: City name.
            date: Date string in YYYY-MM-DD format.
            
        Returns:
            Dictionary containing mock weather data.
        """
        # Parse the date
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            month = date_obj.month
            day = date_obj.day
        except ValueError:
            # Fallback to default values if date parsing fails
            month = 1
            day = 1
        
        # Create a simple hash from city and date for deterministic mock data
        city_hash = sum(ord(c) for c in city.lower()) % 100
        date_hash = (month * 31 + day) % 100
        combined_hash = (city_hash + date_hash) % 100
        
        # Determine weather conditions based on hash
        if combined_hash < 30:
            condition = "Sunny"
            temp_c = 20 + (combined_hash % 10)  # 20-29°C
        elif combined_hash < 60:
            condition = "Cloudy"
            temp_c = 15 + (combined_hash % 8)   # 15-22°C
        elif combined_hash < 80:
            condition = "Rainy"
            temp_c = 10 + (combined_hash % 7)   # 10-16°C
        else:
            condition = "Snowy"
            temp_c = -5 + (combined_hash % 10)  # -5 to 4°C
        
        # Generate mock data
        return {
            "city": city,
            "date": date,
            "condition": condition,
            "temperature_c": temp_c,
            "temperature_f": temp_c * 9/5 + 32,
            "humidity_percent": 40 + (combined_hash % 40),  # 40-79%
            "wind_speed_kmh": 5 + (combined_hash % 20),     # 5-24 km/h
            "wind_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][combined_hash % 8],
            "pressure_hpa": 1010 + (combined_hash % 20),    # 1010-1029 hPa
            "visibility_km": 5 + (combined_hash % 15),      # 5-19 km
            "uv_index": min(combined_hash % 12, 11),        # 0-11
            "sunrise": "06:30",
            "sunset": "18:45",
            "moon_phase": ["New", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                          "Full", "Waning Gibbous", "Last Quarter", "Waning Crescent"][combined_hash % 8],
            "is_mock_data": True
        }
    
    def _format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """
        Format weather data into a human-readable string.
        
        Args:
            weather_data: Dictionary containing weather information.
            
        Returns:
            Formatted weather report string.
        """
        city = weather_data["city"]
        date = weather_data["date"]
        condition = weather_data["condition"]
        temp_c = weather_data["temperature_c"]
        humidity = weather_data["humidity_percent"]
        wind_speed = weather_data["wind_speed_kmh"]
        wind_dir = weather_data["wind_direction"]
        
        return (
            f"Weather in {city} on {date}:\n"
            f"• Condition: {condition}\n"
            f"• Temperature: {temp_c:.1f}°C ({weather_data['temperature_f']:.1f}°F)\n"
            f"• Humidity: {humidity}%\n"
            f"• Wind: {wind_speed} km/h from {wind_dir}\n"
            f"• Pressure: {weather_data['pressure_hpa']} hPa\n"
            f"• Visibility: {weather_data['visibility_km']} km\n"
            f"• UV Index: {weather_data['uv_index']}\n"
            f"• Sunrise: {weather_data['sunrise']}, Sunset: {weather_data['sunset']}\n"
            f"• Moon Phase: {weather_data['moon_phase']}\n"
            f"(Mock Data - Generated for demonstration purposes)"
        )
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for registration in the agent system.
        
        Returns:
            Dictionary containing tool schema information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "func": self.run
        }
    
    def __call__(self, city: str, date: str, **kwargs: Any) -> str:
        """
        Make the tool callable for convenience.
        
        Args:
            city: Name of the city to query weather for.
            date: Date for the weather query in YYYY-MM-DD format.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Formatted weather information string.
        """
        return self.run(city, date, **kwargs)


def create_weather_tool() -> WeatherQueryTool:
    """
    Factory function to create a WeatherQueryTool instance.
    
    Returns:
        Configured WeatherQueryTool instance.
        
    Example:
        >>> weather_tool = create_weather_tool()
        >>> result = weather_tool("London", "2024-05-20")
        >>> print(result)
    """
    return WeatherQueryTool()


if __name__ == "__main__":
    # Example usage when run directly
    tool = WeatherQueryTool()
    
    # Test with valid input
    try:
        result = tool.run("Tokyo", "2024-01-15")
        print("Test 1 - Valid input:")
        print(result)
        print()
    except Exception as e:
        print(f"Test 1 failed: {e}")
    
    # Test with invalid date format
    try:
        result = tool.run("New York", "2024/01/15")
        print("Test 2 - Invalid date format (should fail):")
        print(result)
    except ValueError as e:
        print(f"Test 2 correctly failed with: {e}")
    
    # Test with empty city
    try:
        result = tool.run("", "2024-01-15")
        print("Test 3 - Empty city (should fail):")
        print(result)
    except Exception as e:
        print(f"Test 3 correctly failed with: {e}")
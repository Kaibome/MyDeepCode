"""
Mock travel query tool for the ReAct agent system.

This module provides a TravelQueryTool that returns mock travel options
between two cities, simulating flight, train, and bus options.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TravelQueryInput(BaseModel):
    """
    Pydantic model for validating and normalizing travel query inputs.

    Attributes:
        from_city: The departure city name.
        to_city: The destination city name.
        travel_date: Optional travel date in YYYY-MM-DD format.
        travel_type: Optional type of travel (flight, train, bus, or any).
        max_results: Maximum number of results to return (default: 5).
    """

    from_city: str = Field(
        ...,
        description="The departure city name (e.g., 'New York', 'London', 'Tokyo').",
        min_length=1,
        max_length=100,
    )
    to_city: str = Field(
        ...,
        description="The destination city name (e.g., 'Paris', 'Berlin', 'Sydney').",
        min_length=1,
        max_length=100,
    )
    travel_date: Optional[str] = Field(
        default=None,
        description="Travel date in YYYY-MM-DD format (e.g., '2024-12-25'). If not provided, defaults to tomorrow.",
    )
    travel_type: Optional[str] = Field(
        default="any",
        description="Type of travel: 'flight', 'train', 'bus', or 'any' (default).",
    )
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of travel options to return (default: 5, max: 10).",
        ge=1,
        le=10,
    )

    @field_validator("from_city", "to_city")
    @classmethod
    def validate_city_name(cls, v: str) -> str:
        """
        Validate and normalize city names.

        Args:
            v: City name string.

        Returns:
            Normalized city name with proper capitalization.

        Raises:
            ValueError: If city name contains invalid characters.
        """
        if not v:
            raise ValueError("City name cannot be empty")
        
        # Basic validation for city names
        if any(char.isdigit() for char in v):
            raise ValueError(f"City name '{v}' should not contain numbers")
        
        # Normalize capitalization
        return v.strip().title()

    @field_validator("travel_date")
    @classmethod
    def validate_travel_date(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate travel date format.

        Args:
            v: Date string in YYYY-MM-DD format.

        Returns:
            Validated date string.

        Raises:
            ValueError: If date format is invalid or date is in the past.
        """
        if v is None:
            return None
        
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
            if date_obj.date() < datetime.now().date():
                raise ValueError(f"Travel date '{v}' cannot be in the past")
            return v
        except ValueError as e:
            raise ValueError(
                f"Invalid date format '{v}'. Expected YYYY-MM-DD format."
            ) from e

    @field_validator("travel_type")
    @classmethod
    def validate_travel_type(cls, v: Optional[str]) -> str:
        """
        Validate travel type.

        Args:
            v: Travel type string.

        Returns:
            Validated travel type.

        Raises:
            ValueError: If travel type is not one of the allowed values.
        """
        if v is None:
            return "any"
        
        valid_types = ["flight", "train", "bus", "any"]
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid travel type '{v}'. Must be one of: {', '.join(valid_types)}"
            )
        return v.lower()


class TravelQueryTool:
    """
    Mock travel query tool that returns formatted mock travel data.

    This tool simulates travel queries by generating realistic-looking
    travel options (flights, trains, buses) between two cities.
    """

    def __init__(self, name: str = "travel_query_tool"):
        """
        Initialize the travel query tool.

        Args:
            name: Name of the tool for identification.
        """
        self.name = name
        self.description = "Query travel options between two cities"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run(
        self,
        from_city: str,
        to_city: str,
        travel_date: Optional[str] = None,
        travel_type: str = "any",
        max_results: int = 5,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute a travel query and return mock travel options.

        Args:
            from_city: The departure city name.
            to_city: The destination city name.
            travel_date: Optional travel date in YYYY-MM-DD format.
            travel_type: Type of travel (flight, train, bus, or any).
            max_results: Maximum number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            If return_format is 'string', returns a formatted string.
            If return_format is 'dict', returns a dictionary with travel options.

        Raises:
            ValueError: If input validation fails.
        """
        try:
            # Validate inputs using Pydantic model
            input_data = TravelQueryInput(
                from_city=from_city,
                to_city=to_city,
                travel_date=travel_date,
                travel_type=travel_type,
                max_results=max_results,
            )

            self.logger.info(
                f"Travel query: {input_data.from_city} -> {input_data.to_city}, "
                f"date: {input_data.travel_date or 'tomorrow'}, "
                f"type: {input_data.travel_type}"
            )

            # Generate mock travel options
            travel_options = self._generate_mock_travel_options(input_data)

            # Format response based on kwargs
            return_format = kwargs.get("return_format", "string")
            if return_format == "dict":
                return self._format_dict_response(input_data, travel_options)
            else:
                return self._format_string_response(input_data, travel_options)

        except Exception as e:
            self.logger.error(f"Error in travel query: {e}")
            raise

    def _generate_mock_travel_options(
        self, input_data: TravelQueryInput
    ) -> List[Dict[str, Any]]:
        """
        Generate mock travel options based on input parameters.

        Args:
            input_data: Validated travel query input.

        Returns:
            List of travel option dictionaries.
        """
        travel_options = []
        travel_types = []

        # Determine which travel types to generate
        if input_data.travel_type == "any":
            travel_types = ["flight", "train", "bus"]
        else:
            travel_types = [input_data.travel_type]

        # Generate options for each travel type
        for travel_type in travel_types:
            num_options = min(
                random.randint(2, 4), input_data.max_results - len(travel_options)
            )
            if num_options <= 0:
                break

            for i in range(num_options):
                option = self._generate_mock_travel_option(
                    input_data.from_city,
                    input_data.to_city,
                    travel_type,
                    i + 1,
                    input_data.travel_date,
                )
                travel_options.append(option)

                if len(travel_options) >= input_data.max_results:
                    break

        # Sort by price (cheapest first)
        travel_options.sort(key=lambda x: x["price"])

        return travel_options

    def _generate_mock_travel_option(
        self,
        from_city: str,
        to_city: str,
        travel_type: str,
        option_num: int,
        travel_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single mock travel option.

        Args:
            from_city: Departure city.
            to_city: Destination city.
            travel_type: Type of travel.
            option_num: Option number for variation.
            travel_date: Optional travel date.

        Returns:
            Dictionary representing a travel option.
        """
        # Set default travel date to tomorrow if not provided
        if travel_date:
            date_obj = datetime.strptime(travel_date, "%Y-%m-%d")
        else:
            date_obj = datetime.now() + timedelta(days=1)

        # Base prices by travel type
        base_prices = {"flight": 300, "train": 150, "bus": 80}

        # Generate departure and arrival times
        departure_hour = random.randint(6, 20)
        duration_hours = {
            "flight": random.randint(1, 6),
            "train": random.randint(2, 8),
            "bus": random.randint(4, 12),
        }

        # Calculate price with some variation
        base_price = base_prices.get(travel_type, 200)
        price_variation = random.uniform(0.8, 1.2)
        price = round(base_price * price_variation * (1 + option_num * 0.1))

        # Generate provider names
        providers = {
            "flight": ["SkyJet Airways", "Global Airlines", "QuickFly"],
            "train": ["RailExpress", "National Rail", "HighSpeed Trains"],
            "bus": ["CoachConnect", "BusBuddy", "TravelExpress"],
        }

        provider = random.choice(providers.get(travel_type, ["Travel Co."]))

        return {
            "type": travel_type,
            "provider": provider,
            "departure_city": from_city,
            "arrival_city": to_city,
            "departure_time": f"{departure_hour:02d}:00",
            "arrival_time": f"{(departure_hour + duration_hours[travel_type]) % 24:02d}:00",
            "duration_hours": duration_hours[travel_type],
            "price": price,
            "currency": "USD",
            "travel_date": date_obj.strftime("%Y-%m-%d"),
            "available_seats": random.randint(10, 50),
            "booking_reference": f"{travel_type.upper()}{random.randint(1000, 9999)}",
        }

    def _format_string_response(
        self, input_data: TravelQueryInput, travel_options: List[Dict[str, Any]]
    ) -> str:
        """
        Format travel options as a human-readable string.

        Args:
            input_data: Validated travel query input.
            travel_options: List of travel option dictionaries.

        Returns:
            Formatted string response.
        """
        if not travel_options:
            return (
                f"No travel options found from {input_data.from_city} "
                f"to {input_data.to_city} for the specified criteria."
            )

        # Build response header
        travel_date = input_data.travel_date or "tomorrow"
        response_lines = [
            f"Travel options from {input_data.from_city} to {input_data.to_city} "
            f"on {travel_date}:",
            "",
        ]

        # Add each travel option
        for i, option in enumerate(travel_options, 1):
            response_lines.append(f"{i}. {option['type'].title()} - {option['provider']}")
            response_lines.append(
                f"   Departure: {option['departure_time']} from {option['departure_city']}"
            )
            response_lines.append(
                f"   Arrival: {option['arrival_time']} in {option['arrival_city']}"
            )
            response_lines.append(
                f"   Duration: {option['duration_hours']} hours"
            )
            response_lines.append(
                f"   Price: ${option['price']} {option['currency']}"
            )
            response_lines.append(
                f"   Available seats: {option['available_seats']}"
            )
            response_lines.append(
                f"   Booking reference: {option['booking_reference']}"
            )
            response_lines.append("")

        # Add summary
        cheapest = min(travel_options, key=lambda x: x["price"])
        response_lines.append(
            f"Found {len(travel_options)} travel options. "
            f"Cheapest option: {cheapest['type']} at ${cheapest['price']}."
        )
        response_lines.append("(Note: This is mock data for demonstration purposes.)")

        return "\n".join(response_lines)

    def _format_dict_response(
        self, input_data: TravelQueryInput, travel_options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format travel options as a structured dictionary.

        Args:
            input_data: Validated travel query input.
            travel_options: List of travel option dictionaries.

        Returns:
            Structured dictionary response.
        """
        return {
            "query": {
                "from_city": input_data.from_city,
                "to_city": input_data.to_city,
                "travel_date": input_data.travel_date or "tomorrow",
                "travel_type": input_data.travel_type,
                "max_results": input_data.max_results,
            },
            "travel_options": travel_options,
            "summary": {
                "total_options": len(travel_options),
                "cheapest_option": (
                    min(travel_options, key=lambda x: x["price"]) if travel_options else None
                ),
                "price_range": (
                    {
                        "min": min(opt["price"] for opt in travel_options),
                        "max": max(opt["price"] for opt in travel_options),
                    }
                    if travel_options
                    else None
                ),
            },
            "metadata": {
                "source": "mock_travel_tool",
                "timestamp": datetime.now().isoformat(),
                "note": "Mock data for demonstration purposes",
            },
        }

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for registration in agent systems.

        Returns:
            Dictionary containing tool schema information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": TravelQueryInput,
            "func": self.run,
        }

    def __call__(
        self,
        from_city: str,
        to_city: str,
        travel_date: Optional[str] = None,
        travel_type: str = "any",
        max_results: int = 5,
        **kwargs: Any,
    ) -> str:
        """
        Make the tool callable like a function.

        Args:
            from_city: The departure city name.
            to_city: The destination city name.
            travel_date: Optional travel date in YYYY-MM-DD format.
            travel_type: Type of travel (flight, train, bus, or any).
            max_results: Maximum number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            Formatted string response with travel options.
        """
        return self.run(
            from_city=from_city,
            to_city=to_city,
            travel_date=travel_date,
            travel_type=travel_type,
            max_results=max_results,
            **kwargs,
        )


def create_travel_tool(name: str = "travel_query_tool") -> TravelQueryTool:
    """
    Factory function to create a configured TravelQueryTool instance.

    Args:
        name: Name for the tool instance.

    Returns:
        Configured TravelQueryTool instance.
    """
    return TravelQueryTool(name=name)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the tool
    tool = create_travel_tool()
    
    # Test with string output
    print("=== Testing Travel Query Tool (String Output) ===")
    result = tool(
        from_city="New York",
        to_city="Los Angeles",
        travel_date="2024-12-25",
        travel_type="flight",
        max_results=3,
    )
    print(result)
    
    print("\n=== Testing Travel Query Tool (Dict Output) ===")
    result_dict = tool.run(
        from_city="London",
        to_city="Paris",
        travel_type="train",
        max_results=2,
        return_format="dict",
    )
    print(f"Found {len(result_dict['travel_options'])} options")
    print(f"Cheapest: ${result_dict['summary']['cheapest_option']['price']}")
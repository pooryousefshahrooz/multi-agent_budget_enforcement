"""TravelPlanner's search tools as an MCP server: ReDel's TravelPlannerMixin (redel/tools/travelplanner/search.py,
branch demo/emnlp), same tool names, docstrings, parameters and implementations, over the official TravelPlanner
database (testbed/tasks/data/travelplanner/database).

    python testbed/mcp_servers/travelplanner_search.py --http 9351
"""
import enum
import os
import re
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field

PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else 9351
DB = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "tasks" / "data" / "travelplanner" / "database"
mcp = FastMCP("travelplanner", log_level="WARNING", host="127.0.0.1", port=PORT)


class GDMTransportationMethod(enum.Enum):
    self_driving = "self-driving"
    taxi = "taxi"


flight_data = pd.read_csv(DB / "flights/clean_Flights_2022.csv").dropna()[[
    "Flight Number", "Price", "DepTime", "ArrTime", "ActualElapsedTime", "FlightDate", "OriginCityName",
    "DestCityName", "Distance"]]
gdm_data = pd.read_csv(DB / "googleDistanceMatrix/distance.csv")
accommodation_data = pd.read_csv(DB / "accommodations/clean_accommodations_2022.csv").dropna()[[
    "NAME", "price", "room type", "house_rules", "minimum nights", "maximum occupancy", "review rate number", "city"]]
restaurant_data = pd.read_csv(DB / "restaurants/clean_restaurant_2022.csv").dropna()[
    ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"]]
attraction_data = pd.read_csv(DB / "attractions/attractions.csv").dropna()[
    ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"]]
city_data = {}
for unit in (DB / "background/citySet_with_states.txt").read_text().strip().split("\n"):
    city, state = unit.split("\t")
    city_data.setdefault(state, []).append(city)


@mcp.tool()
def flight_search(
    departure_city: Annotated[str, Field(description="The city you'll be flying out from.")],
    destination_city: Annotated[str, Field(description="The city you aim to reach.")],
    date: Annotated[str, Field(description="The date of your travel in YYYY-MM-DD format.")],
) -> str:
    """
    A flight information retrieval tool.
    Example: flight_search("New York", "London", "2022-10-01") would fetch flights from New York to London on October 1, 2022.
    """
    departure_city = clean_city_name(departure_city)
    destination_city = clean_city_name(destination_city)
    results = flight_data[flight_data["OriginCityName"] == departure_city]
    results = results[results["DestCityName"] == destination_city]
    results = results[results["FlightDate"] == date]
    if len(results) == 0:
        return "There is no flight from {} to {} on {}.".format(departure_city, destination_city, date)
    return results.to_string(index=False)


@mcp.tool()
def google_distance_matrix(
    origin: Annotated[str, Field(description="The departure city of your journey.")],
    destination: Annotated[str, Field(description="The destination city of your journey.")],
    mode: Annotated[GDMTransportationMethod, Field(description="The method of transportation.")],
) -> str:
    """
    Estimate the distance, time and cost between two cities.
    Example: google_distance_matrix("Paris", "Lyon", "self-driving") would provide driving distance, time and cost between Paris and Lyon.
    """
    origin = clean_city_name(extract_before_parenthesis(origin))
    destination = clean_city_name(extract_before_parenthesis(destination))
    info = {"origin": origin, "destination": destination, "cost": None, "duration": None, "distance": None}
    response = gdm_data[(gdm_data["origin"] == origin) & (gdm_data["destination"] == destination)]
    if len(response) > 0:
        if (
            response["duration"].values[0] is None
            or response["distance"].values[0] is None
            or response["duration"].values[0] is np.nan
            or response["distance"].values[0] is np.nan
        ):
            return "No valid information."
        info["duration"] = response["duration"].values[0]
        info["distance"] = response["distance"].values[0]
        if mode == GDMTransportationMethod.self_driving:
            info["cost"] = int(eval(info["distance"].replace("km", "").replace(",", "")) * 0.05)
        elif mode == GDMTransportationMethod.taxi:
            info["cost"] = int(eval(info["distance"].replace("km", "").replace(",", "")))
        if "day" in info["duration"]:
            return "No valid information."
        return (
            f"{mode.value}, from {origin} to {destination}, duration: {info['duration']}, distance:"
            f" {info['distance']}, cost: {info['cost']}"
        )
    return f"{mode.value}, from {origin} to {destination}, no valid information."


@mcp.tool()
def accommodation_search(
    city: Annotated[str, Field(description="The name of the city where you're seeking accommodation.")],
) -> str:
    """
    Discover accommodations in your desired city.
    Example: accommodation_search("Rome") would present a list of hotel rooms in Rome.
    """
    results = accommodation_data[accommodation_data["city"] == clean_city_name(city)]
    if len(results) == 0:
        return "There is no attraction in this city."
    return results.to_string(index=False)


@mcp.tool()
def restaurant_search(
    city: Annotated[str, Field(description="The name of the city where you're seeking restaurants.")],
) -> str:
    """
    Explore dining options in a city of your choice.
    Example: restaurant_search("Tokyo") would show a curated list of restaurants in Tokyo.
    """
    results = restaurant_data[restaurant_data["City"] == clean_city_name(city)]
    if len(results) == 0:
        return "There is no restaurant in this city."
    return results.to_string(index=False)


@mcp.tool()
def attraction_search(
    city: Annotated[str, Field(description="The name of the city where you're seeking attractions.")],
) -> str:
    """
    Find attractions in a city of your choice.
    Example: attraction_search("London") would return attractions in London.
    """
    results = attraction_data[attraction_data["City"] == clean_city_name(city)].reset_index(drop=True)
    if len(results) == 0:
        return "There is no attraction in this city."
    return results.to_string(index=False)


@mcp.tool()
def city_search(
    state: Annotated[str, Field(description="The name of the state where you're seeking cities.")],
) -> str:
    """
    Find cities in a state of your choice.
    Example: city_search("California") would return cities in California.
    """
    if state not in city_data:
        raise ValueError("Invalid State")
    return str(city_data[state])   # ReDel returns the list and kani sends str() of it; one text result for MCP too


def extract_before_parenthesis(s):
    match = re.search(r"^(.*?)\([^)]*\)", s)
    return match.group(1) if match else s


def clean_city_name(n):
    return n.split(",")[0].strip()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

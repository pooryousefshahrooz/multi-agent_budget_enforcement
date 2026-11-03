"""TravelPlanner's plan-saving tool as an MCP server: ReDel's TravelPlannerRootMixin.submit_day_plan
(redel/tools/travelplanner/planner.py, branch demo/emnlp), same name, docstring and parameters. In ReDel only the root
agent has it; in a testbed config, give it to the agent that should write the plan.

The saved plan belongs to the current task: bench.py clears it (DELETE /plan) before a task and reads it (GET /plan)
after.

    python testbed/mcp_servers/travelplanner_plan.py --http 9352
"""
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else 9352
mcp = FastMCP("travelplanner-plan", log_level="WARNING", host="127.0.0.1", port=PORT)
current_plan = []


@mcp.custom_route("/plan", methods=["GET", "DELETE"])
async def plan(request: Request):
    if request.method == "DELETE":
        current_plan.clear()
    return JSONResponse(current_plan)


@mcp.tool()
def submit_day_plan(
    day: int,
    current_city: Annotated[str, Field(description=(
        'The city for this day. If the plan is to travel between two cities, indicate it as "from A to B" (e.g.'
        ' "from Ithaca to Charlotte").'))],
    transportation: Annotated[str, Field(description=(
        'e.g. "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46"'))],
    breakfast: Annotated[str, Field(description='The name of a restaurant in the city, e.g. "Nagaland\'s Kitchen, Charlotte"')],
    attraction: Annotated[str, Field(description=(
        'Use a ";" to separate different attractions, with each attraction formatted as "Name, City". e.g. "The'
        ' Charlotte Museum of History, Charlotte;"'))],
    lunch: Annotated[str, Field(description='e.g. "Cafe Maple Street, Charlotte"')],
    dinner: Annotated[str, Field(description='e.g. "Bombay Vada Pav, Charlotte"')],
    accommodation: Annotated[str, Field(description='e.g. "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"')],
) -> str:
    """
    Once you have collected information about the trip, call this function once for each day of the trip, starting on day 1.
    If a field does not apply for a given day, use "-" to indicate this.
    You MUST call this function to save the plan for each query.
    """
    current_plan.append({
        "day": day, "current_city": current_city, "transportation": transportation, "breakfast": breakfast,
        "attraction": attraction, "lunch": lunch, "dinner": dinner, "accommodation": accommodation,
    })
    return f"Saved plan for day {day}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

"""MCP server for the propagation experiment.

  noop(step)        does nothing; lets a scripted agent make a counted tool call
  slow(seconds)     takes that long; shows which deadline governs the tool call (the user's, or the client's own)
  summarize(text)   asks the calling agent's client for a model call (MCP sampling, sampling/createMessage), so a
                    tool call triggers a model call; returns what the client's model produced, or the client's error

    python testbed/mcp_servers/probe_tools.py --http 9351
"""
import asyncio, sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

PORT = int(sys.argv[sys.argv.index("--http") + 1]) if "--http" in sys.argv else 9351
mcp = FastMCP("probe-tools", log_level="WARNING", host="127.0.0.1", port=PORT)


@mcp.tool()
def noop(step: int = 0) -> str:
    """noop: does nothing; returns ok."""
    return f"ok {step}"


@mcp.tool()
async def slow(seconds: float = 40) -> str:
    """slow: takes the given number of seconds, then returns ok."""
    await asyncio.sleep(seconds)
    return f"ok after {seconds}s"


@mcp.tool()
async def summarize(text: str, ctx: Context) -> str:
    """summarize: summarizes the text with the caller's model (MCP sampling)."""
    try:
        result = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=f"Summarize: {text}"))],
            max_tokens=50)
    except Exception as e:   # the client does not support sampling
        return f"SAMPLING FAILED: {type(e).__name__}: {str(e)[:200]}"
    content = result.content
    return f"SAMPLED: {getattr(content, 'text', content)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

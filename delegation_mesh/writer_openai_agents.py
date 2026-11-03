"""[4] Writer agent: OpenAI Agents SDK, bounded by its native `Runner.run(max_turns=...)`.

    python delegation_mesh/writer_openai_agents.py [--limit 5]
"""
from contextlib import AsyncExitStack

from mesh import AgentNode, MODEL, gateway_mcp, local_mcp, resolve_mcp

from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

set_tracing_disabled(True)   # no tracing export to OpenAI

CARD = {
    "name": "Writer", "framework": "OpenAI Agents SDK", "slug": "openai-agents",
    "accepts": ["analysis"], "produces": "report",
    "subtask": "Write the final short technical report from the plan, research and analysis.",
    "native_limit": {"name": "Runner.run(max_turns)", "value": 5, "counts": "agent turns per run"},
}
SYSTEM = ("You are the Writer. Write a concise technical report (max 350 words, markdown headings): "
          "overview, Reno under loss, CUBIC under loss, key differences, recommendation for a "
          "high-bandwidth network with justification.")

# MCP servers this agent may use (Agents SDK Agent(mcp_servers=...)). Empty = no tools.
MCP_SERVERS = [
    # gateway_mcp("tcp-reference"),       # paid tool behind the metered-resource gateway
    # local_mcp("tcp_reference.py"),
    # {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp"},
]


def mcp_server(s):
    from agents.mcp import MCPServerStdio, MCPServerStreamableHttp
    if s["transport"] == "stdio":
        return MCPServerStdio(name=s["name"], cache_tools_list=True,
                              params={"command": s["command"], "args": s.get("args", []), "env": s.get("env")})
    return MCPServerStreamableHttp(name=s["name"], cache_tools_list=True,
                                   params={"url": s["url"], "headers": s.get("headers") or {}})


async def run(system: str, message: str, hop) -> str:
    client = AsyncOpenAI(base_url=hop.llm_base_url, api_key="unused", max_retries=0, default_headers=hop.headers)
    async with AsyncExitStack() as stack:
        servers = [await stack.enter_async_context(mcp_server(s)) for s in resolve_mcp(MCP_SERVERS, hop)]
        writer = Agent(name="writer", instructions=system, mcp_servers=servers,
                       model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client))
        result = await Runner.run(writer, message, max_turns=hop.limit)
    return str(result.final_output)


if __name__ == "__main__":
    AgentNode(CARD, SYSTEM, run).serve_forever()

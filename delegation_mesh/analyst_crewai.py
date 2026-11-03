"""[3] Analyst agent: CrewAI, bounded by its native `Agent.max_iter`.

    python delegation_mesh/analyst_crewai.py [--limit 5]
"""
import asyncio

from mesh import AgentNode, MODEL, gateway_mcp, local_mcp, resolve_mcp

from crewai import Agent, Crew, LLM, Process, Task

CARD = {
    "name": "Analyst", "framework": "CrewAI", "slug": "crewai",
    "accepts": ["research"], "produces": "analysis",
    "subtask": "Analyse what the research means for a high-bandwidth, high-latency network and recommend one.",
    "native_limit": {"name": "Agent.max_iter", "value": 5, "counts": "reasoning iterations per task"},
}
SYSTEM = ("Analyse throughput recovery after loss on high bandwidth-delay-product links, fairness and "
          "loss sensitivity, then give a clear, justified recommendation (max 180 words).")

# MCP servers this agent may use (CrewAI's native MCP client). Empty = no tools.
MCP_SERVERS = [
    # gateway_mcp("tcp-reference"),       # paid tool behind the metered-resource gateway
    # local_mcp("tcp_reference.py"),
    # {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp"},
]


def mcp_tools(agent: Agent, specs: list) -> list:
    if not specs:
        return []
    from crewai.mcp import MCPServerHTTP, MCPServerStdio
    configs = [MCPServerStdio(command=s["command"], args=s.get("args", []), env=s.get("env"))
               if s["transport"] == "stdio" else MCPServerHTTP(url=s["url"], headers=s.get("headers"))
               for s in specs]
    tools = agent.get_mcp_tools(configs)
    # CrewAI prefixes each tool with the server's full command line; use the plain MCP tool name instead,
    # which a small model can actually reproduce in its "Action:" line
    for t in tools:
        t.name = t.original_tool_name
    return tools


async def run(system: str, message: str, hop) -> str:
    llm = LLM(model=f"openai/{MODEL}", base_url=hop.llm_base_url, api_key="unused", temperature=0.3,
              extra_headers=hop.headers)   # CrewAI routes openai/ models through LiteLLM here
    analyst = Agent(role="Network performance analyst", goal=system,
                    backstory="You analyze congestion control for high-bandwidth networks.",
                    llm=llm, max_iter=hop.limit, max_retry_limit=0, allow_delegation=False, verbose=False)
    analyst.tools = mcp_tools(analyst, resolve_mcp(MCP_SERVERS, hop))
    # CrewAI interpolates {placeholders} in task text, so neutralise braces
    safe = message.replace("{", "(").replace("}", ")")
    task = Task(description=safe, expected_output="A concise analysis with a justified recommendation.",
                agent=analyst)
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False, tracing=False)
    result = await asyncio.to_thread(crew.kickoff)
    return result.raw


if __name__ == "__main__":
    AgentNode(CARD, SYSTEM, run).serve_forever()

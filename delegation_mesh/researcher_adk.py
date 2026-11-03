"""[2] Researcher agent: Google ADK, bounded by its native `RunConfig.max_llm_calls`.

Runs on the VECTURAE fork of ADK (frameworks/adk-python, branch `vecturae`): the request's budget token is passed in
`RunConfig.vecturae_token`, and ADK itself verifies it at the start of the invocation, checks the deadline before every
LLM call, and charges one level of depth when it delegates (agent transfer, AgentTool).

    python delegation_mesh/researcher_adk.py [--limit 5]
"""
from mesh import AgentNode, MODEL, gateway_mcp, local_mcp, resolve_mcp

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

CARD = {
    "name": "Researcher", "framework": "Google ADK", "slug": "adk",
    "accepts": ["plan"], "produces": "research",
    "subtask": "Answer the planner's questions: how Reno and CUBIC grow and cut their window under packet loss.",
    "native_limit": {"name": "RunConfig.max_llm_calls", "value": 5, "counts": "LLM calls per run"},
}
SYSTEM = ("You are the Researcher. Explain the mechanics precisely (max 200 words): Reno's AIMD, fast "
          "retransmit/fast recovery and halving on loss; CUBIC's cubic window growth, beta=0.7 "
          "reduction and RTT-independence.")

# MCP servers this agent may use (ADK McpToolset). Empty = no tools, one LLM call per hop.
MCP_SERVERS = [
    # gateway_mcp("tcp-reference"),       # paid tool behind the metered-resource gateway
    # local_mcp("tcp_reference.py"),
    # {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp"},
]


def mcp_toolsets(specs):
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
    from mcp import StdioServerParameters
    sets = []
    for s in specs:
        if s["transport"] == "stdio":
            params = StdioConnectionParams(server_params=StdioServerParameters(
                command=s["command"], args=s.get("args", []), env=s.get("env")), timeout=30)
        else:
            params = StreamableHTTPConnectionParams(url=s["url"], headers=s.get("headers"))
        sets.append(McpToolset(connection_params=params))
    return sets


async def run(system: str, message: str, hop) -> str:
    toolsets = mcp_toolsets(resolve_mcp(MCP_SERVERS, hop))
    agent = LlmAgent(name="researcher", instruction=system, tools=toolsets,
                     model=LiteLlm(model=f"openai/{MODEL}", api_base=hop.llm_base_url, api_key="unused",
                                   temperature=0.3, extra_headers=hop.headers))
    runner = InMemoryRunner(agent=agent, app_name="chain")
    session = await runner.session_service.create_session(app_name="chain", user_id="user")
    text = ""
    try:
        async for event in runner.run_async(
                user_id="user", session_id=session.id,
                new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=message)]),
                run_config=RunConfig(max_llm_calls=hop.limit, vecturae_token=hop.token)):
            if event.error_message:
                raise RuntimeError(event.error_message)
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
    finally:
        for t in toolsets:
            await t.close()
    return text


if __name__ == "__main__":
    AgentNode(CARD, SYSTEM, run).serve_forever()

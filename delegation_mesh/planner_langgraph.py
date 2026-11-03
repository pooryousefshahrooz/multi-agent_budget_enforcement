"""[1] Planner agent: LangGraph, bounded by its native `recursion_limit`.

Runs on the VECTURAE fork of LangGraph (frameworks/langgraph, branch `vecturae`): the request's budget token is passed
in the run's config, and LangGraph itself verifies it, checks the deadline before every superstep, and charges one
level of depth for every subgraph it starts.

    python delegation_mesh/planner_langgraph.py [--limit 5]
"""
import vecturae
from mesh import AgentNode, MODEL, gateway_mcp, local_mcp, resolve_mcp

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

CARD = {
    "name": "Planner", "framework": "LangGraph", "slug": "langgraph",
    "accepts": ["request", "revision"], "produces": "plan",
    "subtask": "Break the request into an outline and the key technical questions to research.",
    "native_limit": {"name": "recursion_limit", "value": 5, "counts": "graph supersteps per run"},
}
SYSTEM = ("You are the Planner. Produce a short outline (max 120 words) and 3-5 key technical "
          "questions for the researcher. Do not write the report.")

# MCP servers this agent may use (loaded with langchain-mcp-adapters). Empty = no tools, one LLM call per hop.
MCP_SERVERS = [
    # gateway_mcp("tcp-reference"),       # paid tool behind the metered-resource gateway
    # local_mcp("tcp_reference.py"),
    # {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp"},
]


async def mcp_tools(specs):
    if not specs:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient
    conns = {}
    for s in specs:
        if s["transport"] == "stdio":
            conns[s["name"]] = {"transport": "stdio", "command": s["command"], "args": s.get("args", []),
                                "env": s.get("env")}
        else:
            conns[s["name"]] = {"transport": "streamable_http", "url": s["url"], "headers": s.get("headers")}
    return await MultiServerMCPClient(conns).get_tools()


async def run(system: str, message: str, hop) -> str:
    llm = ChatOpenAI(model=MODEL, base_url=hop.llm_base_url, api_key="unused", temperature=0.3, max_retries=0,
                     default_headers=hop.headers)
    tools = await mcp_tools(resolve_mcp(MCP_SERVERS, hop))
    model = llm.bind_tools(tools) if tools else llm

    async def planner(state: MessagesState):
        return {"messages": [await model.ainvoke([SystemMessage(content=system)] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("planner", planner)
    g.add_edge(START, "planner")
    if tools:
        g.add_node("tools", ToolNode(tools))
        g.add_conditional_edges("planner", tools_condition)
        g.add_edge("tools", "planner")
    else:
        g.add_edge("planner", END)
    config = {"recursion_limit": hop.limit}
    if hop.token:   # the patched LangGraph (frameworks/langgraph) enforces the request's budget inside the run
        config["configurable"] = {vecturae.CONFIG_KEY: hop.token}
    out = await g.compile().ainvoke({"messages": [HumanMessage(content=message)]}, config)
    return out["messages"][-1].content


if __name__ == "__main__":
    AgentNode(CARD, SYSTEM, run).serve_forever()

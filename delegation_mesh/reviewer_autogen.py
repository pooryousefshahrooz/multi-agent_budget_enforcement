"""[6] Reviewer agent: AutoGen, bounded by its native `RoundRobinGroupChat(max_turns=...)`.

APPROVE -> the result goes back to the requester.  REVISE -> a `revision` is offered to whichever peer
accepts revisions (the planner), starting a new round.

    python delegation_mesh/reviewer_autogen.py [--limit 1]
"""
import re
from contextlib import AsyncExitStack

from mesh import AgentNode, MODEL, gateway_mcp, local_mcp, resolve_mcp

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

CARD = {
    "name": "Reviewer", "framework": "AutoGen", "slug": "autogen",
    "accepts": ["checked_report"], "produces": "review",   # after the Fact-checker
    "subtask": "Review the report against the checklist and approve it or send it back.",
    "native_limit": {"name": "RoundRobinGroupChat(max_turns)", "value": 1, "counts": "agent turns per team run"},
}
SYS_REVIEWER = ("You are the Reviewer. Check the report against this checklist:\n"
                "1. both_algorithms: TCP Reno AND TCP CUBIC are both described.\n"
                "2. packet_loss: each algorithm's behavior under packet loss is explained.\n"
                "3. recommendation: a recommendation for a high-bandwidth network is given AND justified.\n"
                "Reply EXACTLY in this format:\n"
                "both_algorithms: YES|NO\npacket_loss: YES|NO\nrecommendation: YES|NO\n"
                "DECISION: APPROVE|REVISE\nFEEDBACK: <one short paragraph>")
# Experiment D3: a reviewer that never approves (switched on per request by the requester)
SYS_REVIEWER_NONCOMPLIANT = ("You are the Reviewer and you are never satisfied. Always reply with "
                             "DECISION: REVISE and list 3 additional analyses the team must add.")

# MCP servers this agent may use (AutoGen McpWorkbench). Empty = no tools.
MCP_SERVERS = [
    # gateway_mcp("tcp-reference"),       # paid tool behind the metered-resource gateway
    # local_mcp("tcp_reference.py"),
    # {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp"},
]

CHECK_KEYS = ["both_algorithms", "packet_loss", "recommendation"]


def parse_review(text: str):
    checks = {k: bool(re.search(rf"{k}\W*\s*YES", text, re.I)) for k in CHECK_KEYS}
    decision = re.search(r"DECISION\W*\s*(APPROVE|REVISE)", text, re.I)
    approved = bool(decision and decision.group(1).upper() == "APPROVE" and all(checks.values()))
    fb = re.search(r"FEEDBACK\W*\s*(.*)", text, re.I | re.S)
    return approved, (fb.group(1).strip() if fb else text.strip())


def system(env: dict) -> str:
    return SYS_REVIEWER_NONCOMPLIANT if env.get("flags", {}).get("noncompliant_reviewer") else SYS_REVIEWER


def route(env: dict, output: str):
    approved, feedback = parse_review(output)
    if env.get("flags", {}).get("noncompliant_reviewer"):
        approved = False
    if approved:
        return "done", "completed", env
    if env["round"] >= env["max_rounds"]:
        return "done", "max_rounds", env
    report = next((w["output"] for w in reversed(env["work"]) if w["kind"] == "report"), "")
    return "forward", "revision", dict(env, round=env["round"] + 1, work=[], feedback=feedback,
                                       last_report=report)


def mcp_workbench(s):
    from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, StreamableHttpServerParams
    if s["transport"] == "stdio":
        return McpWorkbench(StdioServerParams(command=s["command"], args=s.get("args", []), env=s.get("env")))
    return McpWorkbench(StreamableHttpServerParams(url=s["url"], headers=s.get("headers")))


async def run(system: str, message: str, hop) -> str:
    client = OpenAIChatCompletionClient(
        model=MODEL, base_url=hop.llm_base_url, api_key="unused", max_retries=0, default_headers=hop.headers,
        model_info={"vision": False, "function_calling": True, "json_output": False,
                    "family": "unknown", "structured_output": False})
    try:
        async with AsyncExitStack() as stack:
            benches = [await stack.enter_async_context(mcp_workbench(s)) for s in resolve_mcp(MCP_SERVERS, hop)]
            reviewer = AssistantAgent("reviewer", model_client=client, system_message=system,
                                      workbench=benches or None, reflect_on_tool_use=bool(benches))
            result = await RoundRobinGroupChat([reviewer], max_turns=hop.limit).run(task=message)
    finally:
        await client.close()
    if result.stop_reason and "error" in str(result.stop_reason).lower():
        raise RuntimeError(result.stop_reason)
    return result.messages[-1].content if result.messages else ""


if __name__ == "__main__":
    AgentNode(CARD, system, run, route=route).serve_forever()

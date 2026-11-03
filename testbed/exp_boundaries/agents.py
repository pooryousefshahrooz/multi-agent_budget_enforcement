"""The four agents of one LangGraph organization, each served by its own LangGraph Agent Server (native A2A endpoint).

    user -> A (planner) -> {B (Open Library), C (Wikipedia)} -> D (verifier) -> done, or back to A (next round)

Stock LangGraph throughout: each agent is a StateGraph whose work node is a prebuilt ReAct agent; MCP tools come from
langchain-mcp-adapters; the Agent Server receives A2A. LangGraph has no A2A client, so an agent delegates with a plain
A2A SendMessage (the work as text; the Agent Server gives the graph none of the incoming message's metadata, so there
is no limit an agent could pass on). Model, MCP and A2A traffic all go through the instrumentation proxy. Every node
reports LangGraph's own counter for this run (steps taken, steps left before GraphRecursionError) to the proxy.

The text between agents carries the round and the task: "ROUND: k", "TASK: ...", plus SUBTASK / RESULT / FEEDBACK.
"""
import asyncio, os, re, sys, uuid

import httpx
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.a2a_util import text_of
from common.config import PROXY, Config, a2a_url, llm_url, mcp_url

CFG = Config(os.environ["EXP_CONFIG"])
ORG = "orgA"
MAX_ROUNDS = int(CFG.raw.get("max_rounds", 3))


def field(text, name):
    m = re.search(rf"^{name}:\s*(.*?)(?=^\w+:|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


LIMIT_KEYS = ("max_depth", "deadline", "max_model_calls", "max_tool_calls", "max_tokens", "max_cost_usd")


async def report(agent, node, config):
    """LangGraph's own counter for this run (steps taken, steps left), and which of the user's limit kinds appear
    anywhere in the run's config (metadata or configurable)."""
    pad = config["configurable"].get("__pregel_scratchpad")
    meta = config.get("metadata") or {}
    seen = str(meta) + str({k: v for k, v in config["configurable"].items() if not k.startswith("__")})
    async with httpx.AsyncClient() as c:
        await c.post(f"{PROXY}/_report", json={
            "org": ORG, "agent": agent, "node": node, "run_id": meta.get("run_id"), "thread_id": meta.get("thread_id"),
            "langgraph_step": pad.step if pad else None, "steps_left": (pad.stop - pad.step) if pad else None,
            "limit_kinds_in_config": [k for k in LIMIT_KEYS if k in seen]})


async def send(sender, to, text, context=None):
    """Delegate over A2A: a plain SendMessage with the work as text."""
    message = {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": text}]}
    if context:
        message["contextId"] = context
    body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": message}}
    async with httpx.AsyncClient(timeout=3600) as c:
        r = await c.post(a2a_url(f"{ORG}.{sender}", ORG, to) + "/", json=body, headers={"A2A-Version": "1.0"})
    return text_of(r.json().get("result", {}).get("task", {}))


async def react(agent, prompt, text):
    """The agent's own work: LangGraph's prebuilt ReAct agent with the agent's MCP tools."""
    spec = CFG.agent(ORG, agent)
    tools = []
    if spec.get("mcp"):
        client = MultiServerMCPClient({s: {"transport": "streamable_http", "url": mcp_url(ORG, agent, s)}
                                       for s in spec["mcp"]})
        tools = await client.get_tools()
    model = ChatOpenAI(model="model", base_url=llm_url(ORG, agent), api_key="x", max_retries=0)
    out = await create_react_agent(model, tools, prompt=prompt).ainvoke({"messages": [("user", text)]})
    return out["messages"][-1].content


def last_text(state):
    return state["messages"][-1].content if state["messages"] else ""


# ---- A: planner -------------------------------------------------------------------------------------------------
async def a_plan(state, config):
    await report("A", "plan", config)
    text = last_text(state)
    task, round_ = field(text, "TASK") or text, int(field(text, "ROUND") or 1)
    feedback = field(text, "FEEDBACK")
    plan = await react("A", CFG.agent(ORG, "A")["role"],
                       f"TASK: {task}" + (f"\nFEEDBACK FROM THE VERIFIER: {feedback}" if feedback else ""))
    return {"messages": [AIMessage(content=f"ROUND: {round_}\nTASK: {task}\nPLAN: {plan}")]}


async def a_fanout(state, config):
    await report("A", "fanout", config)
    text = last_text(state)
    context = str(uuid.uuid4())   # one A2A context per round: D joins B's and C's results by it
    head = f"ROUND: {field(text, 'ROUND')}\nTASK: {field(text, 'TASK')}\nPLAN: {field(text, 'PLAN')}"
    replies = await asyncio.gather(send("A", "B", head, context), send("A", "C", head, context))
    final = next((r for r in replies if not r.startswith("WAITING")), replies[-1])
    return {"messages": [AIMessage(content=final)]}


# ---- B, C: workers ----------------------------------------------------------------------------------------------
def worker(name):
    async def work(state, config):
        await report(name, "work", config)
        text = last_text(state)
        result = await react(name, CFG.agent(ORG, name)["role"], text)
        return {"messages": [AIMessage(content=result)]}

    async def forward(state, config):
        await report(name, "forward", config)
        incoming = next(m.content for m in state["messages"] if m.type == "human")
        msg = (f"FROM: {name}\nROUND: {field(incoming, 'ROUND')}\nTASK: {field(incoming, 'TASK')}\n"
               f"RESULT: {last_text(state)}")
        reply = await send(name, "D", msg, config["metadata"]["thread_id"])
        return {"messages": [AIMessage(content=reply)]}
    g = StateGraph(MessagesState)
    g.add_node("work", work)
    g.add_node("forward", forward)
    g.add_edge(START, "work")
    g.add_edge("work", "forward")
    g.add_edge("forward", END)
    return g.compile()


# ---- D: verifier (joins B and C) ----------------------------------------------------------------------------------
async def d_verify(state, config):
    await report("D", "verify", config)
    received = {field(m.content, "FROM"): m.content for m in state["messages"] if m.type == "human"}
    if not {"B", "C"} <= set(received):   # the other branch has not arrived on this context yet
        return {"messages": [AIMessage(content=f"WAITING: have {sorted(received)}")]}
    task, round_ = field(received["B"], "TASK"), int(field(received["B"], "ROUND") or 1)
    findings = "\n\n".join(f"[{k}] {field(v, 'RESULT')}" for k, v in sorted(received.items()))
    verdict = await react("D", CFG.agent(ORG, "D")["role"], f"TASK: {task}\n\nFINDINGS:\n{findings}")
    if verdict.strip().upper().startswith("VERIFIED") or round_ >= MAX_ROUNDS:   # loop guard: application code
        return {"messages": [AIMessage(content=f"ROUND: {round_}\n{verdict}")]}
    back = f"ROUND: {round_ + 1}\nTASK: {task}\nFEEDBACK: {verdict}"
    return {"messages": [AIMessage(content=await send("D", "A", back))]}


def single(*nodes):
    g = StateGraph(MessagesState)
    for name, fn in nodes:
        g.add_node(name, fn)
    g.add_edge(START, nodes[0][0])
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        g.add_edge(a, b)
    g.add_edge(nodes[-1][0], END)
    return g.compile()


graph_A = single(("plan", a_plan), ("fanout", a_fanout))
graph_B = worker("B")
graph_C = worker("C")
graph_D = single(("verify", d_verify))

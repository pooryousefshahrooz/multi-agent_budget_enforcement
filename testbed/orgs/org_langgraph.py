"""LangGraph organization. Each agent is a LangGraph prebuilt ReAct agent (model + MCP tools from
langchain-mcp-adapters + delegate tools), served by LangGraph's own A2A server: one LangGraph Agent Server
(`langgraph dev`) for the organization. With delegation: a2a every agent is its own graph at its own
/a2a/{assistant_id}; with delegation: native the organization's agents are joined by LangGraph's own mechanism
(native_graph) and only the agents that receive work from outside are served.
The operator's per-agent bound maps to recursion_limit (2N - 1 supersteps for N model calls)."""
import json, os, subprocess, sys, time

import httpx

from common.config import agent_port, llm_url, mcp_url, upstreams_dir

HERE = os.path.dirname(os.path.abspath(__file__))


def mcp_connection(org, name, server, spec):
    """langchain-mcp-adapters connection to one MCP server; with the agent's `sampling: true`, the client answers the
    server's sampling requests through the developer's callback (common.sampling)."""
    conn = {"transport": "streamable_http", "url": mcp_url(org, name, server)}
    if spec.get("sampling"):
        from common.sampling import sampling_callback
        conn["session_kwargs"] = {"sampling_callback": sampling_callback(org, name)}
    return conn


def observer(cfg, org, name):
    """LangChain callback on the agent's model: at every model call it reports the run's metadata (where LangGraph keeps
    the request's data) and LangGraph's own step counter. It changes nothing about how the agent runs."""
    from langchain_core.callbacks import AsyncCallbackHandler

    from common.agent_logic import find_limit
    from common.config import PROXY

    class Observer(AsyncCallbackHandler):
        async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None,
                                      metadata=None, **kw):
            md = metadata or {}
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{PROXY}/_report", json={
                    "org": org, "agent": name, "limit": find_limit(md, cfg.marker),
                    "counter": {"langgraph_step": md.get("langgraph_step"),
                                "checkpoint_ns": md.get("langgraph_checkpoint_ns")}})
    return Observer()


async def react_agent(cfg, org, name, extra_tools=()):
    """One agent of a native organization: the prebuilt ReAct agent with its MCP tools, its cross-organization delegate
    tools, and any tools that reach the organization's other agents (mechanism: tool)."""
    from langchain_core.tools import StructuredTool
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from common.agent_logic import delegate_tools
    spec = cfg.agent(org, name)
    tools = list(extra_tools) + [StructuredTool.from_function(coroutine=fn, name=n, description=d)
                                 for n, d, fn in delegate_tools(cfg, org, name)]
    if spec.get("mcp"):
        tools += await MultiServerMCPClient({s: mcp_connection(org, name, s, spec) for s in spec["mcp"]}).get_tools()
    model = ChatOpenAI(model="model", base_url=llm_url(org, name), api_key="x", max_retries=0,
                       callbacks=[observer(cfg, org, name)])
    return create_react_agent(model, tools, prompt=spec.get("role", f"You are {name}."), name=name)


async def native_graph(cfg, org, ingress):
    """A native organization (delegation: native): its agents joined by LangGraph's own mechanism, from `ingress`.
      subgraph   one StateGraph, every agent a node (its ReAct agent as a subgraph), the organization's edges as the
                 graph's edges (a join waits for all its predecessors)
      tool       an agent reaches each successor through a tool that invokes the successor agent (no joins)"""
    from langchain_core.tools import StructuredTool
    from langgraph.graph import END, START, MessagesState, StateGraph
    nodes, edges = cfg.reachable(org, ingress)
    mech = cfg.orgs[org].get("mechanism", "subgraph")
    if mech == "subgraph":
        from langchain_core.messages import AIMessage

        def node(n, agent):
            # the agent (a ReAct subgraph) is called inside the node with the question and the work so far as a user
            # message (LangGraph's pattern for a subgraph with its own input); the shared message list would end in
            # the previous agent's AI message, which a model answers with nothing
            async def run(state, config):
                question = next(m.content for m in state["messages"] if m.type == "human")
                work = "\n\n".join(f"[{m.name}] {m.content}" for m in state["messages"] if m.type == "ai")
                text = question if not work else f"{question}\n\nWork so far:\n{work}"
                out = await agent.ainvoke({"messages": [("user", text)]}, config)
                return {"messages": [AIMessage(content=out["messages"][-1].content, name=n)]}
            return run
        g = StateGraph(MessagesState)
        for n in nodes:
            g.add_node(n, node(n, await react_agent(cfg, org, n)))
        g.add_edge(START, ingress)
        for n in nodes:
            preds = [a for a, b in edges if b == n]
            if preds:
                g.add_edge(preds if len(preds) > 1 else preds[0], n)
        for n in nodes:
            if not any(a == n for a, _ in edges):
                g.add_edge(n, END)
        return g.compile()
    if mech == "tool":
        if any(sum(1 for a, b in edges if b == n) > 1 for n in nodes):
            raise SystemExit(f"{org}: mechanism 'tool' cannot express a join")

        async def build(n):
            subs = []
            for s in [b for a, b in edges if a == n]:
                child = await build(s)

                async def call(task: str, _child=child) -> str:
                    out = await _child.ainvoke({"messages": [("user", task)]})   # stock: no config passed
                    return out["messages"][-1].content
                subs.append(StructuredTool.from_function(coroutine=call, name=f"ask_{s}",
                                                         description=f"Ask {s}: {cfg.agent(org, s).get('role', s)}"))
            return await react_agent(cfg, org, n, subs)
        return await build(ingress)
    raise SystemExit(f"{org}: unknown LangGraph mechanism {mech!r} (subgraph | tool)")


def make_work(cfg, org, name):
    from langchain_core.tools import StructuredTool
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from common.agent_logic import delegate_tools
    spec = cfg.agent(org, name)
    limit = cfg.orgs[org].get("agent_limit")
    delegates = [StructuredTool.from_function(coroutine=fn, name=n, description=d)
                 for n, d, fn in delegate_tools(cfg, org, name)]

    async def work(text, run_ctx=None):
        tools = list(delegates)
        if spec.get("mcp"):
            client = MultiServerMCPClient({s: mcp_connection(org, name, s, spec) for s in spec["mcp"]})
            tools += await client.get_tools()
        model = ChatOpenAI(model="model", base_url=llm_url(org, name), api_key="x", max_retries=0)
        agent = create_react_agent(model, tools, prompt=spec.get("role", f"You are {name}."))
        from common.agent_logic import ALLOWED_CALLS
        allowed = ALLOWED_CALLS.get()
        # N model calls: the agent/tools supersteps of a ReAct agent; the (N+1)-th model call exceeds the limit
        config = ({"recursion_limit": max(2 * allowed, 1)} if allowed is not None else
                  {"recursion_limit": 2 * int(limit) - 1} if limit else None)
        out = await agent.ainvoke({"messages": [("user", text)]}, config)
        return out["messages"][-1].content
    return work


def serve(cfg, org):
    """Start the organization's Agent Server and record each agent's A2A endpoint."""
    port = agent_port(cfg, org, next(iter(cfg.orgs[org]["agents"])))
    d = os.path.join(upstreams_dir(cfg), f"server_{org}")
    os.makedirs(d, exist_ok=True)
    served = cfg.ingress(org) if cfg.orgs[org].get("delegation", "a2a") == "native" else list(cfg.orgs[org]["agents"])
    graphs = {a: os.path.join(HERE, "lg_server.py") + f":graph_{a}" for a in served}
    json.dump({"dependencies": [os.path.dirname(HERE)], "graphs": graphs}, open(os.path.join(d, "langgraph.json"), "w"))
    env = {**os.environ, "TESTBED_CONFIG": cfg.path, "TESTBED_ORG": org}
    if cfg.orgs[org].get("native_limit"):   # the server's recursion_limit for every run (LangGraph's own setting)
        env["LANGGRAPH_DEFAULT_RECURSION_LIMIT"] = str(int(cfg.orgs[org]["native_limit"]))
    lg = os.path.join(os.path.dirname(sys.executable), "langgraph")
    proc = subprocess.Popen([lg, "dev", "--port", str(port), "--no-browser", "--no-reload", "--n-jobs-per-worker", "50",
                             "--config", "langgraph.json"], cwd=d, env=env)
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if httpx.get(f"{base}/ok", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    up = {}
    for a in served:
        aid = httpx.post(f"{base}/assistants/search", json={"graph_id": a}).json()[0]["assistant_id"]
        up[f"{org}.{a}"] = f"{base}/a2a/{aid}"
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / langgraph] {k} on LangGraph Agent Server A2A {v}", flush=True)
    proc.wait()

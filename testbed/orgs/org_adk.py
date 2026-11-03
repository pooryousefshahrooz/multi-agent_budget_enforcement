"""ADK organization. Each agent is an ADK LlmAgent (model via LiteLlm, MCP tools via McpToolset, delegate tools as
function tools), and each agent is served by ADK's own A2A server (to_a2a) on its own port. The served ADK agent is a
small custom agent that runs the agent's behaviour on a message (common.agent_logic), whose work is the LlmAgent.
The operator's per-agent bound maps to RunConfig.max_llm_calls."""
import json, os, threading, time

from common.config import agent_port, llm_url, mcp_url, upstreams_dir


def _ident(name):
    return name.replace("-", "_").replace(".", "_")


def _calls(agent_limit):
    """apply_limit: the model calls the user's limit allows this run (ADK's own max_llm_calls), else the operator's."""
    from common.agent_logic import ALLOWED_CALLS
    allowed = ALLOWED_CALLS.get()
    return allowed if allowed is not None else agent_limit


def _run_config(received, limit, RunConfig):
    rc = received.model_copy() if received is not None else RunConfig()
    if limit:
        rc.max_llm_calls = int(limit)
    return rc


def mcp_toolset(org, name, server, spec):
    """ADK's MCP client for one server; with the agent's `sampling: true`, ADK answers the server's sampling requests
    through the developer's callback (common.sampling)."""
    from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
    kw = {}
    if spec.get("sampling"):
        from common.sampling import sampling_callback
        kw["sampling_callback"] = sampling_callback(org, name)
    return McpToolset(connection_params=StreamableHTTPConnectionParams(url=mcp_url(org, name, server)), **kw)


def remote_agent(cfg, org, name, to_org, to_agent):
    """ADK's own A2A client for a link: a RemoteA2aAgent (transfer target named remote_<org>_<agent>), reached through
    the proxy. With the organization's forward_limit, the developer's hook puts the user's limit, as the run holds it in
    custom_metadata, on the outgoing A2A request; without it, ADK sends none."""
    from google.adk.a2a.agent import RemoteA2aAgent

    from common.agent_logic import find_limit
    from common.config import a2a_url
    kw = {}
    if cfg.orgs[org].get("forward_limit"):
        def forward(ctx, message):
            limit = find_limit(ctx.run_config.custom_metadata if ctx.run_config else None, cfg.marker)
            return {"limit": limit} if limit else {}
        kw["a2a_request_meta_provider"] = forward
    card = a2a_url(f"{org}.{name}", to_org, to_agent) + "/.well-known/agent-card.json"
    return RemoteA2aAgent(name=_ident(f"remote_{to_org}_{to_agent}"), agent_card=card, timeout=1800,
                          description=cfg.agent(to_org, to_agent).get("role", to_agent), **kw)


def make_work(cfg, org, name):
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
    from google.genai import types

    from common.agent_logic import delegate_tools
    spec = cfg.agent(org, name)
    limit = cfg.orgs[org].get("agent_limit")

    async def work(text, run_config=None):
        """run_config: the RunConfig ADK's A2A server gave this agent's invocation; the agent's run keeps it (the user's
        limit, if ADK passed it, is in its custom_metadata), with the operator's per-agent bound if one is set."""
        tools = [mcp_toolset(org, name, s, spec) for s in spec.get("mcp", [])]
        sub_agents = []
        if cfg.orgs[org].get("a2a_client") == "native":   # links through ADK's own A2A client (transfer targets)
            sub_agents = [remote_agent(cfg, org, name, to_org, to_agent) for to_org, to_agent in cfg.delegates(org, name)]
        else:                                              # links through the testbed's delegate tool
            tools += [fn for _, _, fn in delegate_tools(cfg, org, name)]
        agent = LlmAgent(name=_ident(name), instruction=spec.get("role", f"You are {name}."), tools=tools,
                         sub_agents=sub_agents,
                         model=LiteLlm(model="openai/model", api_base=llm_url(org, name), api_key="x"))
        sessions = InMemorySessionService()
        runner = Runner(app_name=_ident(org), agent=agent, session_service=sessions)
        session = await sessions.create_session(app_name=_ident(org), user_id="caller")
        final = ""
        async for event in runner.run_async(user_id="caller", session_id=session.id,
                                            new_message=types.Content(role="user", parts=[types.Part(text=text)]),
                                            run_config=_run_config(run_config, _calls(limit), RunConfig)):
            if event.is_final_response() and event.content and event.content.parts:
                final = "".join(p.text or "" for p in event.content.parts)
        for t in tools:
            if hasattr(t, "close"):
                await t.close()
        return final
    return work


def observer(cfg, org, name):
    """ADK's own before_model_callback: at every model call it reports what the invocation holds (RunConfig's
    custom_metadata, where ADK keeps the request's data) and ADK's own per-invocation counter. Returns None, so the
    model call goes ahead unchanged."""
    import httpx

    from common.agent_logic import find_limit
    from common.config import PROXY

    async def before_model(callback_context, llm_request):
        ic = callback_context._invocation_context
        rc, cost = ic.run_config, getattr(ic, "_invocation_cost_manager", None)
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{PROXY}/_report", json={
                "org": org, "agent": name, "limit": find_limit(rc.custom_metadata if rc else None, cfg.marker),
                "counter": {"max_llm_calls": rc.max_llm_calls if rc else None, "counter_id": id(cost) if cost else None,
                            "llm_calls": getattr(cost, "_number_of_llm_calls", None)}})
        return None
    return before_model


def native_root(cfg, org, ingress):
    """A native organization (delegation: native): its agents joined by ADK's own mechanism, from `ingress`.
      transfer     sub_agents (the agent transfers to a successor; one invocation)
      agent_tool   AgentTool(successor) (the successor runs as a tool)
      workflow     google.adk.workflow graph; a join is a JoinNode
      sequential   SequentialAgent over a chain (each agent runs after the previous one, one invocation)"""
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

    from common.agent_logic import delegate_tools
    nodes, edges = cfg.reachable(org, ingress)
    mech = cfg.orgs[org].get("mechanism", "transfer")
    succ = lambda n: [b for a, b in edges if a == n]
    if mech in ("transfer", "agent_tool", "sequential") and any(sum(1 for a, b in edges if b == n) > 1 for n in nodes):
        raise SystemExit(f"{org}: mechanism {mech!r} cannot express a join (use workflow)")

    def agent(n, tools=(), subs=()):
        spec = cfg.agent(org, n)
        own = [mcp_toolset(org, n, s, spec) for s in spec.get("mcp", [])]
        remote = []
        if cfg.orgs[org].get("a2a_client") == "native":
            remote = [remote_agent(cfg, org, n, to_org, to_agent) for to_org, to_agent in cfg.delegates(org, n)]
        else:
            own += [fn for _, _, fn in delegate_tools(cfg, org, n)]
        from google.adk.models.lite_llm import LiteLlm
        # the role is the instruction only; ADK also prints descriptions into other agents' prompts
        return LlmAgent(name=_ident(n), description=f"agent {n}", instruction=spec.get("role", f"You are {n}."),
                        tools=own + list(tools), sub_agents=list(subs) + remote,
                        before_model_callback=observer(cfg, org, n),
                        model=LiteLlm(model="openai/model", api_base=llm_url(org, n), api_key="x"))

    if mech == "transfer":
        build = lambda n: agent(n, subs=[build(s) for s in succ(n)])
        return build(ingress)
    if mech == "agent_tool":
        build = lambda n: agent(n, tools=[AgentTool(agent=build(s)) for s in succ(n)])
        return build(ingress)
    if mech == "sequential":
        from google.adk.agents import SequentialAgent
        order, n = [], ingress
        while n is not None:
            order.append(n)
            nxt = succ(n)
            if len(nxt) > 1:
                raise SystemExit(f"{org}: mechanism 'sequential' needs a chain")
            n = nxt[0] if nxt else None
        return SequentialAgent(name=_ident(f"{org}_{ingress}"), sub_agents=[agent(a) for a in order])
    if mech == "workflow":
        from google.adk.workflow import START, JoinNode, Workflow
        built = {n: agent(n) for n in nodes}
        wf_edges = [(START, built[ingress])]
        for n in nodes:
            preds = [a for a, b in edges if b == n]
            if len(preds) > 1:
                join = JoinNode(name=f"join_{_ident(n)}")
                wf_edges += [(built[p], join) for p in preds] + [(join, built[n])]
            elif preds:
                wf_edges.append((built[preds[0]], built[n]))
        return Workflow(name=_ident(f"{org}_{ingress}"), edges=wf_edges)
    raise SystemExit(f"{org}: unknown ADK mechanism {mech!r} (transfer | agent_tool | workflow)")


def serve_native(cfg, org):
    """Serve a native organization with ADK's own A2A server: each agent that receives work from outside is to_a2a of
    the organization's agents joined from it."""
    import uvicorn
    from google.adk.a2a.utils.agent_to_a2a import to_a2a
    from google.adk.agents import BaseAgent
    from google.adk.agents.run_config import RunConfig
    limit = cfg.orgs[org].get("native_limit")

    class Limited(BaseAgent):
        """The organization's root, run with RunConfig.max_llm_calls = native_limit for the whole invocation (every
        agent below counts against it)."""
        async def _run_async_impl(self, ctx):
            ctx.run_config = (ctx.run_config or RunConfig()).model_copy(update={"max_llm_calls": int(limit)})
            async for event in self.sub_agents[0].run_async(ctx):
                yield event

    def root(name):
        r = native_root(cfg, org, name)
        return Limited(name=_ident(f"limited_{name}"), sub_agents=[r]) if limit else r

    up, servers = {}, []
    for name in cfg.ingress(org):
        port = agent_port(cfg, org, name)
        server = uvicorn.Server(uvicorn.Config(to_a2a(root(name), host="127.0.0.1", port=port),
                                               host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        up[f"{org}.{name}"] = f"http://127.0.0.1:{port}/"
    while not all(s.started for s in servers):
        time.sleep(0.05)
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / adk, native {cfg.orgs[org].get('mechanism', 'transfer')}] {k} on ADK to_a2a {v}", flush=True)
    threading.Event().wait()


def serve(cfg, org):
    if cfg.orgs[org].get("delegation", "a2a") == "native":
        return serve_native(cfg, org)
    from typing import Any

    import uvicorn
    from google.adk.a2a.utils.agent_to_a2a import to_a2a
    from google.adk.agents import BaseAgent
    from google.adk.events import Event
    from google.genai import types

    from common.agent_logic import AgentLogic

    class TestbedAgent(BaseAgent):
        logic: Any = None

        async def _run_async_impl(self, ctx):
            text = "".join(p.text or "" for p in (ctx.user_content.parts if ctx.user_content else []))
            rc = ctx.run_config
            cost = getattr(ctx, "_invocation_cost_manager", None)
            out = await self.logic.handle(text, received=rc.custom_metadata if rc else None,
                                          counter={"max_llm_calls": rc.max_llm_calls if rc else None,
                                                   "counter_id": id(cost) if cost else None}, run_ctx=rc)
            yield Event(author=self.name, invocation_id=ctx.invocation_id,
                        content=types.Content(role="model", parts=[types.Part(text=out)]))

    up, servers = {}, []
    for name in cfg.orgs[org]["agents"]:
        port = agent_port(cfg, org, name)
        agent = TestbedAgent(name=_ident(name), description=cfg.agent(org, name).get("role", name),
                             logic=AgentLogic(cfg, org, name, make_work(cfg, org, name)))
        app = to_a2a(agent, host="127.0.0.1", port=port)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        up[f"{org}.{name}"] = f"http://127.0.0.1:{port}/"
    while not all(s.started for s in servers):
        time.sleep(0.05)
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / adk] {k} on ADK to_a2a {v}", flush=True)
    threading.Event().wait()

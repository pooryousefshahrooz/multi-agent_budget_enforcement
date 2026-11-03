"""OpenAI Agents SDK organization. Each agent is an SDK Agent run with Runner.run (model via the Chat Completions
model, MCP tools via MCPServerStreamableHttp, delegate tools as function tools). The SDK has no A2A, so each agent is
served by an a2a-sdk A2A server (run_org.py). The operator's per-agent bound maps to Runner.run(max_turns=N)."""
from contextlib import AsyncExitStack, asynccontextmanager

from common.config import llm_url, mcp_url


def agent_factory(cfg, org, name):
    """An async context manager function yielding the SDK Agent for one request, its MCP servers connected."""
    from agents import Agent, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
    from agents.mcp import MCPServerStreamableHttp
    from openai import AsyncOpenAI

    from common.agent_logic import delegate_tools
    set_tracing_disabled(True)
    spec = cfg.agent(org, name)
    delegates = [function_tool(fn, name_override=n) for n, _, fn in delegate_tools(cfg, org, name)]

    @asynccontextmanager
    async def open_agent():
        async with AsyncExitStack() as stack:
            servers = [await stack.enter_async_context(MCPServerStreamableHttp(
                name=s, params={"url": mcp_url(org, name, s)}, cache_tools_list=True)) for s in spec.get("mcp", [])]
            client = AsyncOpenAI(base_url=llm_url(org, name), api_key="x", max_retries=0)
            yield Agent(name=name, instructions=spec.get("role", f"You are {name}."), tools=delegates,
                        mcp_servers=servers, model=OpenAIChatCompletionsModel(model="model", openai_client=client))
    return open_agent


def make_work(cfg, org, name):
    from agents import Runner
    limit = cfg.orgs[org].get("agent_limit")
    open_agent = agent_factory(cfg, org, name)

    async def work(text, run_ctx=None):
        async with open_agent() as agent:
            from common.agent_logic import ALLOWED_CALLS
            turns = ALLOWED_CALLS.get() if ALLOWED_CALLS.get() is not None else limit   # apply_limit, else the operator's
            res = await Runner.run(agent, text, **({"max_turns": int(turns)} if turns else {}))
            return str(res.final_output)
    return work


def native_work(cfg, org, name):
    """A native organization (delegation: native): its agents, in chain order from `name`, chained in code (the SDK's
    "orchestrating via code": each agent's output is the next agent's input). Each agent runs with Runner.run, so
    native_limit, the SDK's own max_turns, bounds each agent's run. (Handoffs would make it one run, but llama3.2:3b
    rarely hands off, so the chain would stop at its first agent.)"""
    from agents import Runner
    from agents.run_config import DEFAULT_MAX_TURNS
    order = cfg.chain(org, name)
    n = cfg.orgs[org].get("native_limit")
    opens = {a: agent_factory(cfg, org, a) for a in order}

    async def work(text, run_ctx=None):
        out = ""
        for a in order:
            async with opens[a]() as agent:
                prompt = text if not out else f"{text}\n\nWork so far:\n{out}"
                res = await Runner.run(agent, prompt, max_turns=int(n) if n else DEFAULT_MAX_TURNS)
                out = str(res.final_output)
        return out
    return work

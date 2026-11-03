"""AutoGen organization. Each agent is an AutoGen AssistantAgent (model via the OpenAI client, MCP tools via
McpWorkbench, delegate tools via a StaticWorkbench of function tools). AutoGen has no A2A, so each agent is served by
an a2a-sdk A2A server (run_org.py). The operator's per-agent bound maps to max_tool_iterations."""
from contextlib import AsyncExitStack, asynccontextmanager

from common.config import llm_url, mcp_url

MODEL_INFO = {"vision": False, "function_calling": True, "json_output": False, "family": "unknown",
              "structured_output": False}


def _iterations(agent_limit):
    """apply_limit: the model calls the user's limit allows this run (AutoGen's own max_tool_iterations), else the
    operator's."""
    from common.agent_logic import ALLOWED_CALLS
    allowed = ALLOWED_CALLS.get()
    return allowed if allowed is not None else agent_limit


def agent_factory(cfg, org, name, max_tool_iterations=None):
    """An async context manager function yielding the AssistantAgent for one request, its workbenches open."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool, StaticWorkbench
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_ext.tools.mcp import McpWorkbench, StreamableHttpServerParams

    from common.agent_logic import delegate_tools
    spec = cfg.agent(org, name)
    delegates = [FunctionTool(fn, description=d.split("\n\nArgs:")[0], name=n) for n, d, fn in delegate_tools(cfg, org, name)]

    @asynccontextmanager
    async def open_agent(iterations=None):
        n = iterations or max_tool_iterations
        async with AsyncExitStack() as stack:
            benches = [await stack.enter_async_context(McpWorkbench(StreamableHttpServerParams(url=mcp_url(org, name, s))))
                       for s in spec.get("mcp", [])]
            if delegates:
                benches.append(StaticWorkbench(delegates))
            client = OpenAIChatCompletionClient(model="model", base_url=llm_url(org, name), api_key="x",
                                                max_retries=0, model_info=MODEL_INFO)
            try:
                yield AssistantAgent(name.replace("-", "_"), model_client=client, workbench=benches or None,
                                     system_message=spec.get("role", f"You are {name}."), reflect_on_tool_use=True,
                                     **({"max_tool_iterations": int(n)} if n else {}))
            finally:
                await client.close()
    return open_agent


def make_work(cfg, org, name):
    from autogen_agentchat.messages import TextMessage
    limit = cfg.orgs[org].get("agent_limit")
    open_agent = agent_factory(cfg, org, name)

    async def work(text, run_ctx=None):
        from autogen_agentchat.messages import ToolCallRequestEvent

        from common.agent_logic import LimitStop
        async with open_agent(_iterations(limit)) as agent:
            result = await agent.run(task=text)
            iterations = agent._max_tool_iterations
        texts = [m.to_text() for m in result.messages if isinstance(m, TextMessage) and m.source != "user"]
        out = texts[-1] if texts else ""
        # max_tool_iterations does not raise: the agent stops calling tools; its last round still made calls
        if sum(isinstance(m, ToolCallRequestEvent) for m in result.messages) >= iterations:
            return LimitStop(out, "AutoGen max_tool_iterations")
        return out
    return work


def native_work(cfg, org, name):
    """A native organization (delegation: native): its agents, in chain order from `name`, as one RoundRobinGroupChat
    that ends when the last agent writes FINAL ANSWER. native_limit is AutoGen's own team max_turns; each agent may
    make up to 5 tool iterations per turn. A team that reached max_turns ends without an error: reported as a limit
    stop."""
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat

    from common.agent_logic import LimitStop
    order = cfg.chain(org, name)
    n = cfg.orgs[org].get("native_limit")
    opens = {a: agent_factory(cfg, org, a, max_tool_iterations=5) for a in order}

    async def work(text, run_ctx=None):
        async with AsyncExitStack() as stack:
            agents = [await stack.enter_async_context(opens[a]()) for a in order]
            team = RoundRobinGroupChat(agents, max_turns=int(n) if n else None,
                                       termination_condition=TextMentionTermination("FINAL ANSWER",
                                                                                    sources=[agents[-1].name]))
            result = await team.run(task=text)
        texts = [m.to_text() for m in result.messages if isinstance(m, TextMessage) and m.source != "user"]
        out = texts[-1] if texts else ""
        if result.stop_reason and "Maximum number of turns" in result.stop_reason:
            return LimitStop(out, "AutoGen team max_turns")
        return out
    return work

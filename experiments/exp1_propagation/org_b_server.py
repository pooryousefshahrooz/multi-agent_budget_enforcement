"""Organization B: one framework's agent behind an A2A server, honouring the limit organization A sends.

Org A puts the user's limit in the A2A message metadata: {"limit": {"max_llm_calls": N, "deadline": t}}. Org B's
developer maps it onto the framework's own runtime settings, with public APIs only, per request:

  framework          N model calls                          deadline
  ADK                RunConfig.max_llm_calls (callback)     callback refuses later model calls
  LangGraph          recursion_limit = 2N - 1 supersteps    asyncio timeout around the run
  OpenAI Agents SDK  Runner.run(max_turns=N)                asyncio timeout around the run
  CrewAI             Agent(max_iter=N)                      Agent(max_execution_time=seconds)
  AutoGen            RoundRobinGroupChat(max_turns=N)       TimeoutTermination(seconds)

The agent has one no-op tool and talks to fake_llm.py, which always asks for the tool again, so a run ends only when a
limit ends it. Only ADK ships its own A2A server (``to_a2a``); for the others the developer writes one (a2a-sdk).

    envs/<framework>/bin/python org_b_server.py --framework langgraph --port 9201 --llm http://127.0.0.1:9100
"""
import argparse, asyncio, json, os, time

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
UNBOUNDED = 10 ** 6


def read_limit(metadata) -> dict:
    if metadata is None:
        return {}
    if hasattr(metadata, "DESCRIPTOR"):
        from google.protobuf import json_format
        metadata = json_format.MessageToDict(metadata)
    return dict((metadata or {}).get("limit") or {})


async def with_deadline(coro, limit):
    if not limit.get("deadline"):
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=max(float(limit["deadline"]) - time.time(), 0.01))
    except asyncio.TimeoutError:
        return "stopped: deadline"


# ---- the five frameworks: one run, bounded by the request's limit -----------------------------------------------
async def run_langgraph(llm, limit):
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def noop(step: int = 0) -> str:
        """Does nothing."""
        return f"ok {step}"
    model = ChatOpenAI(model="fake", base_url=llm, api_key="x", max_retries=0).bind_tools([noop])

    async def agent(state):
        return {"messages": [await model.ainvoke(state["messages"])]}
    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([noop]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    n = int(limit.get("max_llm_calls", UNBOUNDED))
    config = {"recursion_limit": 2 * n - 1}   # agent, tools, agent, ...: N model calls take 2N - 1 supersteps
    return await with_deadline(g.compile().ainvoke({"messages": [("user", "go")]}, config), limit)


async def run_openai_agents(llm, limit):
    from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
    from openai import AsyncOpenAI
    set_tracing_disabled(True)

    @function_tool
    def noop(step: int = 0) -> str:
        """Does nothing."""
        return f"ok {step}"
    client = AsyncOpenAI(base_url=llm, api_key="x", max_retries=0)
    agent = Agent(name="worker", instructions="x", tools=[noop],
                  model=OpenAIChatCompletionsModel(model="fake", openai_client=client))
    n = int(limit.get("max_llm_calls", UNBOUNDED))
    return await with_deadline(Runner.run(agent, "go", max_turns=n), limit)


async def run_crewai(llm, limit):
    from crewai import LLM, Agent, Crew, Task
    from crewai.tools import tool

    @tool("noop")
    def noop(step: int = 0) -> str:
        """Does nothing."""
        return f"ok {step}"
    kwargs = {"max_iter": int(limit.get("max_llm_calls", UNBOUNDED))}
    if limit.get("deadline"):
        kwargs["max_execution_time"] = max(int(round(float(limit["deadline"]) - time.time())), 1)
    agent = Agent(role="worker", goal="x", backstory="x", tools=[noop], allow_delegation=False, max_retry_limit=0,
                  verbose=False, llm=LLM(model="openai/fake", base_url=llm, api_key="x"), **kwargs)
    crew = Crew(agents=[agent], tasks=[Task(description="go", expected_output="x", agent=agent)], verbose=False)
    return await asyncio.to_thread(crew.kickoff)


async def run_autogen(llm, limit):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import TimeoutTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    async def noop(step: int = 0) -> str:
        """Does nothing."""
        return f"ok {step}"
    client = OpenAIChatCompletionClient(model="fake", base_url=llm, api_key="x", max_retries=0,
                                        model_info={"vision": False, "function_calling": True, "json_output": False,
                                                    "family": "unknown", "structured_output": False})
    agent = AssistantAgent("worker", model_client=client, tools=[noop], reflect_on_tool_use=False)
    term = (TimeoutTermination(max(float(limit["deadline"]) - time.time(), 0.01))
            if limit.get("deadline") else None)
    team = RoundRobinGroupChat([agent], max_turns=int(limit.get("max_llm_calls", UNBOUNDED)),
                               termination_condition=term)
    try:
        return await team.run(task="go")
    finally:
        await client.close()


RUNNERS = {"langgraph": run_langgraph, "openai_agents": run_openai_agents, "crewai": run_crewai,
           "autogen": run_autogen}


# ---- A2A server ----------------------------------------------------------------------------------------------------
def generic_app(framework: str, port: int, llm: str):
    from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
    from starlette.applications import Starlette

    class Executor(AgentExecutor):
        async def execute(self, context, event_queue):
            task = context.current_task or new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            up = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
            await up.update_status(state=TaskState.TASK_STATE_WORKING)
            limit = read_limit(context.message.metadata)
            try:
                out = await RUNNERS[framework](f"{llm}/{framework}/v1", limit)
                outcome = f"finished: {str(out)[:80]}"
            except Exception as e:   # each framework stops with its own exception
                outcome = f"stopped by {type(e).__name__}: {str(e)[:120]}"
            await up.add_artifact(parts=[new_text_part(text=outcome)])
            await up.complete()

        async def cancel(self, context, event_queue):
            raise NotImplementedError

    card = AgentCard(name=f"org-B {framework} agent", description="honours the request's limit", version="1.0",
                     default_input_modes=["text/plain"], default_output_modes=["text/plain"],
                     capabilities=AgentCapabilities(streaming=False),
                     supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}",
                                                          protocol_version="1.0")],
                     skills=[AgentSkill(id="work", name="work", description="works", tags=["work"])])
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def adk_app(port: int, llm: str):
    """ADK ships its own A2A server: to_a2a. The developer adds only the callback from limit_aware_adk.py."""
    from google.adk.agents import LlmAgent
    from google.adk.a2a.utils.agent_to_a2a import to_a2a
    from google.adk.models.lite_llm import LiteLlm
    import limit_aware_adk

    def noop(step: int = 0) -> str:
        """Does nothing."""
        return f"ok {step}"
    agent = LlmAgent(name="worker", instruction="x", tools=[noop], before_model_callback=limit_aware_adk.callback,
                     model=LiteLlm(model="openai/fake", api_base=f"{llm}/adk/v1", api_key="x"))
    return to_a2a(agent, host="127.0.0.1", port=port)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True, choices=list(RUNNERS) + ["adk"])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--llm", default="http://127.0.0.1:9100")
    a = p.parse_args()
    import uvicorn
    app = adk_app(a.port, a.llm) if a.framework == "adk" else generic_app(a.framework, a.port, a.llm)
    print(f"org B ({a.framework}) A2A server on http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="error")

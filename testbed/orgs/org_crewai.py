"""CrewAI organization. Each agent is a CrewAI Agent run as a one-task Crew (model via LLM, MCP tools via
MCPServerHTTP, delegate tools as CrewAI tools). CrewAI 1.15's own A2A support needs a2a-sdk 0.3, older than the A2A 1.0
the testbed speaks, so each agent is served by an a2a-sdk A2A server (run_org.py). The operator's per-agent bound maps
to max_iter."""
import asyncio, concurrent.futures

from common.config import llm_url, mcp_url


def _iters(agent_limit):
    """apply_limit: the model calls the user's limit allows this run (CrewAI's own max_iter), else the operator's."""
    from common.agent_logic import ALLOWED_CALLS
    allowed = ALLOWED_CALLS.get()
    return allowed if allowed is not None else agent_limit


def crew_agent(cfg, org, name, holder, **extra):
    """The CrewAI Agent for one testbed agent: model via LLM, MCP tools via MCPServerHTTP, delegate tools as CrewAI
    tools; holder["limit"] carries the current request's received limit into CrewAI's tool threads."""
    from crewai import LLM, Agent
    from crewai.mcp import MCPServerHTTP
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    from common.agent_logic import delegate_tools
    spec = cfg.agent(org, name)

    class TaskArg(BaseModel):
        task: str = Field(description="What the other agent should do, with everything it needs to know.")

    def crew_tool(tool_name, doc, fn):
        short = doc.split("\n\nArgs:")[0]

        class Delegate(BaseTool):
            name: str = tool_name
            description: str = short
            args_schema: type[BaseModel] = TaskArg

            def _run(self, task: str) -> str:   # CrewAI calls tools synchronously, from its own thread
                from common.agent_logic import RECEIVED_LIMIT

                def call():   # the request's received limit, carried into this worker thread
                    RECEIVED_LIMIT.set(holder.get("limit"))
                    return asyncio.run(fn(task))
                with concurrent.futures.ThreadPoolExecutor(1) as ex:
                    return ex.submit(call).result()
        return Delegate()
    role = spec.get("role", f"You are {name}.")
    return Agent(role=name, goal=role, backstory=role, tools=[crew_tool(*t) for t in delegate_tools(cfg, org, name)],
                 allow_delegation=False, verbose=False,
                 llm=LLM(model="openai/model", base_url=llm_url(org, name), api_key="x"),
                 mcps=[MCPServerHTTP(url=mcp_url(org, name, s)) for s in spec.get("mcp", [])], **extra)


def make_work(cfg, org, name):
    from crewai import Crew, Task
    limit = cfg.orgs[org].get("agent_limit")
    holder = {}   # the limit the current request's run received, for the delegate tools' worker threads

    async def work(text, run_ctx=None):
        from common.agent_logic import RECEIVED_LIMIT
        holder["limit"] = RECEIVED_LIMIT.get()
        agent = crew_agent(cfg, org, name, holder, **({"max_iter": int(iters)} if (iters := _iters(limit)) else {}))
        crew = Crew(agents=[agent], tasks=[Task(description=text, expected_output="your result", agent=agent)],
                    verbose=False)
        out = str(await crew.kickoff_async())
        from common.agent_logic import LimitStop
        # max_iter does not raise: CrewAI forces a final answer
        if agent.agent_executor is not None and getattr(agent.agent_executor, "iterations", 0) >= agent.max_iter:
            return LimitStop(out, "CrewAI max_iter")
        return out
    return work


def native_work(cfg, org, name):
    """A native organization (delegation: native): its agents, in chain order from `name`, as one sequential Crew (one
    task per agent, each with the previous tasks as context). native_limit is CrewAI's own max_iter on every agent.
    CrewAI's max_iter does not raise (it forces a final answer), so a run that reached it is reported as a limit stop."""
    from crewai import Crew, Process, Task

    from common.agent_logic import LimitStop
    order = cfg.chain(org, name)
    n = cfg.orgs[org].get("native_limit")
    holder = {}

    async def work(text, run_ctx=None):
        agents = [crew_agent(cfg, org, a, holder, **({"max_iter": int(n)} if n else {})) for a in order]
        tasks = []
        for a, agent in zip(order, agents):
            part = cfg.agent(org, a).get("role", a)
            tasks.append(Task(description=f"{text}\n\nYour part: {part}", expected_output="your part of the work",
                              agent=agent, **({"context": list(tasks)} if tasks else {})))
        out = str(await Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff_async())
        hit = [a for a, ag in zip(order, agents)
               if ag.agent_executor is not None and getattr(ag.agent_executor, "iterations", 0) >= ag.max_iter]
        return LimitStop(out, f"CrewAI max_iter ({', '.join(hit)})") if hit else out
    return work

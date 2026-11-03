"""CrewAI organization served by CrewAI's own A2A server code (crewai[a2a], A2A 0.3 via a2a-sdk 0.3, in its own env
envs/crewai_a2a). CrewAI ships the server side as `crewai.a2a.utils.task.execute(agent, context, event_queue)` and the
card as `agent.to_agent_card(url)` (an Agent with an A2AServerConfig); the application mounts them on an a2a-sdk
server, which is all this module does. Whatever CrewAI does with the request (stock: its run gets the text only) is
CrewAI's.

framework crewai_a2a_patched runs the same code in envs/crewai_a2a_patched, where testbed/patches/crewai_a2a_limit.patch
is applied to CrewAI: its execute takes the `limit` from the request (depth -> reject, deadline -> max_execution_time,
cost -> max_iter at A2AServerConfig.usd_per_call, from the config's limit_conversion).

The observer reports, through CrewAI's own event bus, the settings the CrewAI run actually ran with (max_iter,
max_execution_time) and how it ended: CrewAI's max_iter does not raise (CrewAI forces a final answer), its
max_execution_time raises TimeoutError.
"""
import asyncio, json, os, threading, time

from common.agent_logic import observe
from common.config import agent_port, upstreams_dir
from orgs.org_crewai import crew_agent

RUNS = {}   # agent name -> the CrewAI agent that ran its current task (CrewAI's AgentExecution* events; the A2A
            # events drop the agent); the testbed runs one request at a time


def _patched():
    from crewai.a2a.config import A2AServerConfig
    return "usd_per_call" in A2AServerConfig.model_fields


def _settings(agent):
    return {"max_iter": agent.max_iter, "max_execution_time": agent.max_execution_time}


def _from_limit(ran, configured):
    """The run's max_iter when the received limit set it (the patched CrewAI), else None (the agent's own setting)."""
    return ran.max_iter if ran is not None and _settings(ran) != configured else None


def _request_limit(context):
    for meta in (context.message.metadata if context.message else None, context.metadata):
        if meta and isinstance(meta.get("limit"), dict):
            return meta["limit"]
    return None


def serve(cfg, org):
    import uvicorn
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.utils.errors import ServerError
    from crewai.a2a.config import A2AServerConfig
    from crewai.a2a.utils.task import execute
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.agent_events import (AgentExecutionCompletedEvent, AgentExecutionErrorEvent,
                                                   AgentExecutionStartedEvent)

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _started(source, event):
        RUNS[event.agent.role] = event.agent

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _completed(source, event):
        RUNS[event.agent.role] = event.agent

    @crewai_event_bus.on(AgentExecutionErrorEvent)
    def _failed(source, event):
        RUNS[event.agent.role] = event.agent

    price = (cfg.raw.get("limit_conversion") or {}).get("usd_per_call")
    server_kw = {"usd_per_call": price} if _patched() and price else {}
    up, servers = {}, []
    for name in cfg.orgs[org]["agents"]:
        port = agent_port(cfg, org, name)
        extra = {"max_iter": int(cfg.orgs[org]["agent_limit"])} if cfg.orgs[org].get("agent_limit") else {}
        agent = crew_agent(cfg, org, name, {}, a2a=A2AServerConfig(name=f"{org}.{name}", **server_kw), **extra)

        class Executor(AgentExecutor):
            def __init__(self, agent, name):
                self.agent, self.name = agent, name

            async def execute(self, context, event_queue):
                if self.agent.mcps:
                    # CrewAI loads an agent's `mcps` in Crew and Agent.kickoff, not in aexecute_task (which its A2A
                    # server runs): the application hands them over as tools, once
                    mcp_tools = await asyncio.to_thread(self.agent.get_mcp_tools, self.agent.mcps)
                    self.agent.tools = list(self.agent.tools or []) + mcp_tools
                    self.agent.mcps = None
                limit, configured = _request_limit(context), _settings(self.agent)
                RUNS.pop(self.name, None)
                try:
                    await execute(self.agent, context, event_queue)
                except ServerError as e:
                    msg = getattr(e.error, "message", str(e))
                    if msg.startswith("Delegation limit exhausted"):   # the patched CrewAI rejected the task
                        binding = "depth" if "depth" in msg else "deadline" if "deadline" in msg else "cost"
                        await observe(cfg, org, self.name, received={"limit": limit}, error="LimitExhausted",
                                      allowed_calls=0, binding=binding)
                    else:
                        ran = await self._ran()
                        await self._report_run(ran, limit, configured)
                        timeout = "timed out after" in msg
                        await observe(cfg, org, self.name,
                                      error="CrewAI max_execution_time (TimeoutError)" if timeout else "ServerError",
                                      error_text=msg[:200], allowed_calls=_from_limit(ran, configured),
                                      binding="deadline" if timeout else None)
                    raise
                ran = await self._ran()
                await self._report_run(ran, limit, configured)
                iterations = getattr(getattr(ran, "agent_executor", None), "iterations", None)
                stop = ran is not None and iterations is not None and iterations >= ran.max_iter
                await observe(cfg, org, self.name, **({"error": "CrewAI max_iter (forced final answer)"} if stop
                                                      else {"finished": True}),
                              allowed_calls=_from_limit(ran, configured), iterations=iterations,
                              binding="cost" if stop and ran.max_iter != configured["max_iter"] else None)

            async def _ran(self):
                for _ in range(40):   # CrewAI's event handlers run in its thread pool
                    if self.name in RUNS:
                        return RUNS.pop(self.name)
                    await asyncio.sleep(0.05)
                return None

            async def _report_run(self, ran, limit, configured):
                """What the CrewAI run held: its own settings, and the user's limit if they were set from it."""
                held = _settings(ran) if ran else None
                received = {"run_settings": held}
                if held and limit and held != configured:
                    received["set_from"] = limit
                await observe(cfg, org, self.name, received=received, counter=held)

            async def cancel(self, context, event_queue):
                raise NotImplementedError

        card = agent.to_agent_card(f"http://127.0.0.1:{port}/")
        handler = DefaultRequestHandler(agent_executor=Executor(agent, name), task_store=InMemoryTaskStore())
        app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        up[f"{org}.{name}"] = f"http://127.0.0.1:{port}/"
    while not all(s.started for s in servers):
        time.sleep(0.05)
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / {cfg.framework(org)}] {k} on CrewAI's own A2A server (A2A 0.3"
              f"{', patched' if _patched() else ''}) {v}", flush=True)
    threading.Event().wait()

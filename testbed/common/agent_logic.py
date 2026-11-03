"""What every agent does with an A2A message, whatever its framework. This is the organization's application code;
the agent's own work (model, MCP tools, delegate tools) is its framework's agent, passed in as `work(text) -> str`.

  1. A message from outside the organization (the user, or another organization's delegate tool) makes this agent the
     request's *first agent*; the request gets a context id.
  2. An agent with several predecessors (a join) waits until all of them have sent their input.
  3. The agent does its work.
  4. It sends its output to its successors over A2A; an agent with no successors reports its output back to the
     first agent (the loop back).
  5. The first agent, once every last agent has reported, returns the organization's answer to the caller. A single
     agent organization is its own first and last agent.

Messages between agents of one organization carry a one-line header in their text ("#testbed {json}"): the A2A
metadata is not usable for this, since some frameworks' A2A servers (LangGraph's) give the agent none of it.
"""
import asyncio, contextvars, json, time, uuid
from collections import defaultdict

import httpx

from common.a2a_util import compose, remote_call
from common.config import PROXY

HEADER = "#testbed "
RECEIVED_LIMIT = contextvars.ContextVar("received_limit", default=None)   # the limit this agent's run received
ALLOWED_CALLS = contextvars.ContextVar("allowed_calls", default=None)     # apply_limit: the model calls it allows


class LimitStop(str):
    """An answer a framework produced after its own limit stopped the run without raising (CrewAI's max_iter forces a
    final answer, an AutoGen team ends at max_turns): the text, plus which limit stopped it."""
    def __new__(cls, text, stopped_by):
        obj = super().__new__(cls, text)
        obj.stopped_by = stopped_by
        return obj


def allowed_calls(cfg, org, limit):
    """apply_limit: calls - the user's limit (depth, deadline, cost) as the number of model calls this run may make,
    for the framework's own counter: none if depth is used up, otherwise as many as both the time left and the budget
    pay for (limit_conversion: seconds_per_call, usd_per_call). Returns (calls, the limit that set them); (None, None)
    when the organization does not apply limits or the run received none."""
    if cfg.orgs[org].get("apply_limit") != "calls" or not limit:
        return None, None
    conv = cfg.raw.get("limit_conversion") or {}
    if limit.get("max_depth", 1) < 1:
        return 0, "depth"
    left = limit.get("deadline", time.time() + 1e9) - time.time()
    by_time = int(left // conv.get("seconds_per_call", 1e-9)) if left > 0 else 0
    by_cost = int(limit.get("max_cost_usd", 1e9) // conv.get("usd_per_call", 1e-12))
    return (max(0, by_time), "deadline") if by_time < by_cost else (max(0, by_cost), "cost")


JOINS = defaultdict(dict)     # context -> {predecessor: its output}, per process (one process per organization)
REPORTS = defaultdict(dict)   # context -> {last agent: its output}


def pack(**header) -> str:
    return HEADER + json.dumps(header)


def find_limit(obj, marker):
    """The user's limit, if it is anywhere inside obj (a run's config / context / metadata), else None."""
    if isinstance(obj, dict):
        if obj.get("id") == marker:
            return obj
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for v in obj:
            hit = find_limit(v, marker)
            if hit is not None:
                return hit
    elif isinstance(obj, str) and marker in obj:
        return {"id": marker, "as_text": True}
    return None


def limit_path(obj, marker, path=""):
    """Where inside obj the user's limit is (e.g. "state[0].additional_kwargs.metadata.limit"), else None."""
    if isinstance(obj, dict):
        if obj.get("id") == marker:
            return path or "."
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = ((f"[{i}]", v) for i, v in enumerate(obj))
    else:
        return path if isinstance(obj, str) and marker in obj else None
    for k, v in items:
        hit = limit_path(v, marker, f"{path}{k if str(k).startswith('[') else ('.' if path else '') + str(k)}")
        if hit is not None:
            return hit
    return None


async def observe(cfg, org, agent, received=None, counter=None, **extra):
    """Observer: report to the proxy what this agent's run received from its framework (the user's limit or not) and
    the framework's own per-run counter."""
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{PROXY}/_report", json={"org": org, "agent": agent, "limit": find_limit(received, cfg.marker),
                                               "limit_at": limit_path(received, cfg.marker), "counter": counter, **extra})


async def report_library_run(cfg, org, agent, limit, held, own, outcome, error=None, stop_errors=None, **extra):
    """Observer for a framework whose own A2A server applies the limit (a patched library): report what its run held
    (`held`, its own settings `own` when the limit did not set them) and how it ended. outcome: completed / rejected /
    one of stop_errors' keys (the framework's limit stop, reported under that name) / anything else (error)."""
    set_from_limit = bool(limit) and held != own
    await observe(cfg, org, agent, received={"run_settings": held, **({"set_from": limit} if set_from_limit else {})},
                  counter=held)
    calls = next((v for k, v in held.items() if k != "timeout_s"), None) if set_from_limit else None
    if outcome == "rejected":
        binding = "depth" if "depth" in (error or "") else "deadline" if "deadline" in (error or "") else "cost"
        await observe(cfg, org, agent, error="LimitExhausted", allowed_calls=0, binding=binding, **extra)
    elif outcome == "completed":
        await observe(cfg, org, agent, finished=True, allowed_calls=calls, **extra)
    elif outcome in (stop_errors or {}):
        name, binding = stop_errors[outcome]
        await observe(cfg, org, agent, error=name, error_text=(error or "")[:200], allowed_calls=calls,
                      binding=binding, **extra)
    else:
        await observe(cfg, org, agent, error=(error or "error").split(":")[0], error_text=(error or "")[:200],
                      allowed_calls=calls, **extra)


def unpack(text: str):
    first = text.split("\n", 1)[0]
    return json.loads(first[len(HEADER):]) if first.startswith(HEADER) else None


class AgentLogic:
    def __init__(self, cfg, org, name, work):
        self.cfg, self.org, self.name, self.work = cfg, org, name, work
        self.label = f"{org}.{name}"

    async def send(self, to_agent, text):
        return await remote_call(self.label, self.org, to_agent, text)

    async def handle(self, text: str, received=None, counter=None, run_ctx=None) -> str:
        """received: what the framework gave this run besides the text (config / context / metadata), for the observer;
        counter: the framework's own per-run counter; run_ctx: passed on to the agent's work (e.g. ADK's RunConfig)."""
        await observe(self.cfg, self.org, self.name, received, counter)
        RECEIVED_LIMIT.set(find_limit(received, self.cfg.marker))
        allowed, binding = allowed_calls(self.cfg, self.org, RECEIVED_LIMIT.get())
        ALLOWED_CALLS.set(allowed)
        h = unpack(text)
        if h and h["kind"] == "report":                       # a last agent reporting back to the first agent
            REPORTS[h["context"]][h["from"]] = h["result"]
            return "received"
        if h is None:                                         # from outside the organization: a new request
            h = {"kind": "work", "context": uuid.uuid4().hex, "origin": self.name, "from": None, "task": text,
                 "input": None}
        context, origin = h["context"], h["origin"]
        # a native organization runs its agents inside this agent's work (the framework's own delegation): no A2A
        native = self.cfg.orgs[self.org].get("delegation", "a2a") == "native"
        _, edges = ([], []) if native else self.cfg.reachable(self.org, origin)
        preds = [a for a, b in edges if b == self.name]
        succs = [b for a, b in edges if a == self.name]

        inputs = {}
        if len(preds) > 1:                                    # join: wait for every predecessor
            JOINS[context][h["from"]] = h["input"]
            missing = sorted(set(preds) - set(JOINS[context]))
            if missing:
                return f"WAITING for {', '.join(missing)}"
            inputs = JOINS.pop(context)
        elif h["from"]:
            inputs = {h["from"]: h["input"]}

        if allowed == 0:   # the limit is used up before this run starts (frameworks read 0 as "no limit")
            await observe(self.cfg, self.org, self.name, error="LimitExhausted", allowed_calls=0, binding=binding)
            return "ERROR LimitExhausted: no model call left under the limit"
        try:
            output = await self.work(compose(h["task"], inputs), run_ctx)
        except Exception as e:   # the framework stopped the run (its limit error, or anything else)
            await observe(self.cfg, self.org, self.name, error=type(e).__name__, error_text=str(e)[:200],
                          allowed_calls=allowed, binding=binding)
            return f"ERROR {type(e).__name__}: {str(e)[:200]}"
        if getattr(output, "stopped_by", None):
            await observe(self.cfg, self.org, self.name, error=output.stopped_by, allowed_calls=allowed, binding=binding)
        elif str(output).startswith("Sorry, need more steps to process this request"):
            # LangGraph's prebuilt ReAct agent stops on its step limit with this answer instead of raising
            await observe(self.cfg, self.org, self.name, error="LangGraph step limit (prebuilt agent's stop)",
                          allowed_calls=allowed, binding=binding)
        else:
            await observe(self.cfg, self.org, self.name, finished=True, allowed_calls=allowed, binding=binding)

        if succs:
            await asyncio.gather(*[self.send(s, pack(kind="work", context=context, origin=origin,
                                                     **{"from": self.name}, task=h["task"], input=output))
                                   for s in succs])
        elif self.name != origin:
            await self.send(origin, pack(kind="report", context=context, **{"from": self.name}, result=output))
        else:
            REPORTS[context][self.name] = output

        if self.name != origin:
            return "done"
        reports = REPORTS.pop(context, {})
        if len(reports) == 1:
            return next(iter(reports.values()))
        return "\n\n".join(f"[{k}] {v}" for k, v in reports.items())


def delegate_tools(cfg, org, name):
    """One delegate tool per link from this agent: (tool name, description, async fn(task) -> the result)."""
    forward = cfg.orgs[org].get("forward_limit")

    def make(to_org, to_agent):
        async def delegate(task: str) -> str:
            # with the organization's forward_limit, the send carries the limit this agent's run received, with this
            # hop taken off the delegation depth
            limit = RECEIVED_LIMIT.get() if forward else None
            if limit and "max_depth" in limit:
                limit = {**limit, "max_depth": limit["max_depth"] - 1}
            return await remote_call(f"{org}.{name}", to_org, to_agent, task, {"limit": limit} if limit else None)
        return delegate
    tools = []
    for to_org, to_agent in cfg.delegates(org, name):
        role = cfg.agent(to_org, to_agent).get("role", "")
        tool_name = f"delegate_to_{to_org}_{to_agent}".replace("-", "_")
        fn = make(to_org, to_agent)
        fn.__name__ = tool_name
        fn.__doc__ = (f"Delegate a task to {to_agent} in organization {to_org} and get its result back. "
                      f"{to_agent}: {role}\n\nArgs:\n    task: what {to_agent} should do, with everything it needs to know.")
        tools.append((tool_name, fn.__doc__, fn))
    return tools

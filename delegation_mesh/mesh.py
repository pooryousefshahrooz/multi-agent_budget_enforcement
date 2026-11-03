"""mesh: the small peer-to-peer SDK every chain agent imports.

There is no orchestrator and no registry. Each agent is its own process that
  1. is an A2A server: it serves an A2A Agent Card at /.well-known/agent-card.json and accepts
     A2A JSON-RPC `SendMessage` requests,
  2. advertises itself on the network with mDNS/DNS-SD (Zeroconf),
  3. browses the network for other agents and for the LLM gateway,
  4. delegates peer to peer with an A2A message to a peer whose card says it accepts that kind of
     work; the peer answers with an A2A Task that is WORKING (accepted) or REJECTED (declined, with the reason).

The delegation envelope and the request's budget token travel in the A2A message metadata
(META_DELEGATION, META_TOKEN); in MCP tool calls the token travels in the request's `_meta` (META_TOKEN).

Message kinds form the chain:  request → plan → research → analysis → report → review
(and a reviewer that wants changes emits a `revision`, which the planner accepts).
"""
import os
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")   # no interactive trace prompt
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse, asyncio, contextvars, json, logging, re, signal, socket, sys, threading, time, uuid, warnings
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Awaitable, Callable, Optional, Union

import httpx
import mcp
import uvicorn
from google.protobuf import json_format
from starlette.applications import Starlette
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentSkill, Role,
                       SendMessageConfiguration, SendMessageRequest, TaskState)
import budget_token as bt
import enforcement
from meter import start_meter
from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("autogen_core").setLevel(logging.CRITICAL)
logging.getLogger("a2a").setLevel(logging.ERROR)

AGENT_SERVICE   = "_chain-agent._tcp.local."
GATEWAY_SERVICE = "_llm-gateway._tcp.local."
CARD_PATH       = "/.well-known/agent-card.json"
MODEL           = os.getenv("LLM_MODEL", "llama3.2:3b")
DISCOVERY_WAIT  = float(os.getenv("MESH_DISCOVERY_WAIT", "30"))   # seconds to look for a willing peer
JOIN_WAIT       = float(os.getenv("MESH_JOIN_WAIT", "600"))       # seconds a join waits for its other inputs
NATIVE_LIMIT_ERRORS = ("GraphRecursionError", "LlmCallsLimitExceededError", "MaxTurnsExceeded")

# A2A message metadata keys and Agent Card extensions of this testbed
META_DELEGATION = "vecturae/delegation"     # the delegation envelope (JSON string)
META_TOKEN      = bt.META_KEY               # the request's capability token (A2A metadata and MCP _meta)
EXT_CHAIN       = "urn:vecturae:ext:delegation-chain:v1"
EXT_BUDGET      = "urn:vecturae:ext:budget-token:v1"


_log_lock = threading.Lock()


def log(msg: str):
    with _log_lock:
        sys.stdout.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        sys.stdout.flush()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# HTTP: a tiny threaded JSON server
# ---------------------------------------------------------------------------
def serve_json(handle: Callable[[str, str, dict], tuple], host: str = "127.0.0.1", port: int = 0):
    """Start a threaded HTTP server; handle(method, path, body) -> (status, payload). Returns (server, url)."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _do(self, method):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n)) if n else {}
                code, payload = handle(method, self.path, body)
            except Exception as e:
                code, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._do("GET")

        def do_POST(self):
            self._do("POST")

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}"


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def run_async(fn, *args):
    """Run a coroutine function to completion on a fresh event loop in its own thread (callable from anywhere,
    including a thread or a Jupyter cell that already runs an event loop)."""
    out = {}

    def target():
        try:
            out["value"] = asyncio.run(fn(*args))
        except BaseException as e:
            out["error"] = e
    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


# ---------------------------------------------------------------------------
# A2A Agent Cards
# ---------------------------------------------------------------------------
def a2a_card(view: dict, url: str) -> AgentCard:
    """The A2A Agent Card for an agent. Chain-specific fields (framework, accepts/produces, native limit, model,
    tools) are published as parameters of the delegation-chain extension; one skill per accepted kind of work."""
    chain = AgentExtension(uri=EXT_CHAIN, required=False,
                           description="Accepts/produces kinds of work in the delegation chain, framework and native limit")
    chain.params.update({k: view[k] for k in ("framework", "slug", "accepts", "produces", "native_limit")})
    chain.params.update({k: view[k] for k in ("model", "tools", "enforcement_point", "org", "agent_id", "role")
                         if view.get(k) is not None})
    budget = AgentExtension(uri=EXT_BUDGET, required=False,
                            description=f"Carries the request's capability token in message metadata '{META_TOKEN}'")
    return AgentCard(
        name=view["name"], description=view["subtask"], version="1.0",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, extensions=[chain, budget]),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[AgentSkill(id=f"accepts-{kind}", name=f"{kind} → {view['produces']}", description=view["subtask"],
                           tags=[f"accepts:{kind}", f"produces:{view['produces']}", view["framework"]],
                           input_modes=["text/plain"], output_modes=["text/plain"])
                for kind in view["accepts"]])


def card_view(card_json: dict) -> dict:
    """Read an A2A Agent Card (JSON) into the fields the chain needs; the full card is kept under 'a2a'."""
    params = next((e.get("params", {}) for e in card_json.get("capabilities", {}).get("extensions", [])
                   if e.get("uri") == EXT_CHAIN), {})
    nl = dict(params.get("native_limit", {}))
    if "value" in nl:
        nl["value"] = int(nl["value"])
    return {"name": card_json["name"], "subtask": card_json.get("description", ""),
            "endpoint": card_json["supportedInterfaces"][0]["url"].rstrip("/"),
            "framework": params.get("framework"), "slug": params.get("slug"),
            "accepts": list(params.get("accepts", [])), "produces": params.get("produces"),
            "native_limit": nl, "model": params.get("model"), "tools": list(params.get("tools", [])),
            "enforcement_point": params.get("enforcement_point"), "org": params.get("org", "default"),
            "agent_id": params.get("agent_id"), "role": params.get("role"),
            "a2a": card_json}


async def _a2a_send(card_json: dict, text: str, envelope: dict, token: Optional[dict]):
    card = json_format.ParseDict(card_json, AgentCard(), ignore_unknown_fields=True)
    client = await create_client(agent=card, client_config=ClientConfig(streaming=False))
    msg = new_text_message(text, role=Role.ROLE_USER)
    env = {k: v for k, v in envelope.items() if k != "token"}
    msg.metadata.update({META_DELEGATION: json.dumps(env)})
    if token is not None:
        msg.metadata.update({META_TOKEN: bt.encode(token)})
    try:
        async for event in client.send_message(SendMessageRequest(
                message=msg, configuration=SendMessageConfiguration(return_immediately=True))):
            return event
    finally:
        await client.close()


def a2a_send(card: dict, text: str, envelope: dict):
    """Send one delegation as an A2A message. Returns (accepted, reason, task id)."""
    event = run_async(_a2a_send, card["a2a"], text, envelope, envelope.get("token"))
    if not event.HasField("task"):
        return False, "peer answered without a task", None
    task = event.task
    if task.status.state == TaskState.TASK_STATE_REJECTED:
        return False, get_message_text(task.status.message) or "rejected", task.id
    if task.status.state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED):
        return False, get_message_text(task.status.message) or "failed", task.id
    return True, "", task.id


def rejected_task(rpc_id, reason: str, context_id: Optional[str] = None) -> dict:
    """A JSON-RPC SendMessage result: an A2A Task in state REJECTED (used by enforcement points)."""
    return {"jsonrpc": "2.0", "id": rpc_id, "result": {"task": {
        "id": str(uuid.uuid4()), "contextId": context_id or str(uuid.uuid4()),
        "status": {"state": "TASK_STATE_REJECTED", "message": {
            "messageId": str(uuid.uuid4()), "role": "ROLE_AGENT", "parts": [{"text": reason}]}}}}}


# ---------------------------------------------------------------------------
# MCP: the token rides in the _meta of every tools/call
# ---------------------------------------------------------------------------
CURRENT_TOKEN: contextvars.ContextVar = contextvars.ContextVar("vecturae_token", default=None)
_process_token = {"value": None}   # fallback for frameworks that run tools on threads they create themselves
_original_call_tool = mcp.ClientSession.call_tool


async def _call_tool_with_token(self, name, arguments=None, read_timeout_seconds=None, progress_callback=None,
                                *, meta=None):
    token = CURRENT_TOKEN.get() or _process_token["value"]
    if token is not None:
        meta = {**(meta or {}), META_TOKEN: token}
    return await _original_call_tool(self, name, arguments, read_timeout_seconds, progress_callback, meta=meta)

# every framework here (LangGraph via langchain-mcp-adapters, ADK, CrewAI, OpenAI Agents SDK, AutoGen) calls MCP
# tools through mcp.ClientSession.call_tool, so one hook covers all of them
mcp.ClientSession.call_tool = _call_tool_with_token


# ---------------------------------------------------------------------------
# Discovery: mDNS advertisement + browsing
# ---------------------------------------------------------------------------
class Mesh:
    """Advertises this process on mDNS and keeps a live view of the peers and gateways it can see."""

    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        # Created on a fresh thread so Zeroconf always runs its own event loop: inside Jupyter it would
        # otherwise attach to the kernel's loop, which a blocking cell starves of mDNS events.
        made = {}
        t = threading.Thread(target=lambda: made.setdefault("zc", Zeroconf(ip_version=IPVersion.V4Only)))
        t.start()
        t.join()
        self.zc = made["zc"]
        self.peers: dict = {}      # mDNS instance name -> agent card (fetched from the peer itself)
        self.gateways: dict = {}   # mDNS instance name -> {"url": base URL, "org": organization}
        self._cv = threading.Condition()
        self._infos = []
        self._browser = None

    def advertise(self, service_type: str, instance: str, port: int, properties: dict):
        info = ServiceInfo(service_type, f"{instance}.{service_type}",
                           addresses=[socket.inet_aton(self.host)], port=port,
                           properties=properties, server=f"{instance}.local.")
        self.zc.register_service(info, allow_name_change=True)
        self._infos.append(info)

    def browse(self):
        self._browser = ServiceBrowser(self.zc, [AGENT_SERVICE, GATEWAY_SERVICE], handlers=[self._on_change])
        return self

    def _on_change(self, zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Removed:
            with self._cv:
                gone = self.peers.pop(name, None) or self.gateways.pop(name, None)
                self._cv.notify_all()
            if gone:
                log(f"mesh: {name.split('.')[0]} left")
        else:   # resolving blocks, so never do it on the zeroconf thread
            threading.Thread(target=self._resolve, args=(service_type, name), daemon=True).start()

    def _resolve(self, service_type, name):
        info = self.zc.get_service_info(service_type, name, timeout=3000)
        if not info or not info.parsed_addresses():
            return
        url = f"http://{info.parsed_addresses()[0]}:{info.port}"
        try:
            if service_type == GATEWAY_SERVICE:
                org = (info.properties or {}).get(b"org", b"default").decode()
                entry, table = {"url": url, "org": org}, self.gateways
            else:
                entry = card_view(httpx.get(url + CARD_PATH, timeout=5).json())
                table = self.peers
        except Exception:
            return
        with self._cv:
            new = name not in table
            table[name] = entry
            self._cv.notify_all()
        if new:
            what = f"gateway of org {entry['org']}" if table is self.gateways else f"{entry['name']} ({entry['framework']})"
            log(f"mesh: discovered {what} at {url}")

    def find(self, kind: str, exclude: Optional[str] = None, only: Optional[str] = None) -> list:
        """Cards of the peers that accept messages of this kind (only the agent with id `only`, if given)."""
        with self._cv:
            return [c for c in self.peers.values() if kind in c.get("accepts", []) and c["name"] != exclude
                    and (only is None or c.get("agent_id") == only)]

    def knows(self, agent_name: str) -> bool:
        with self._cv:
            return any(c["name"] == agent_name for c in self.peers.values())

    def wait_for(self, predicate: Callable[[], object], timeout: float):
        with self._cv:
            return self._cv.wait_for(predicate, timeout)

    def gateway(self, timeout: float = DISCOVERY_WAIT, org: Optional[str] = None) -> Optional[str]:
        """URL of a metered-resource gateway: of the given organization, or any."""
        match = lambda: [g["url"] for g in self.gateways.values() if org is None or g["org"] == org]
        ok = self.wait_for(match, timeout)
        with self._cv:
            return match()[0] if ok else None

    def all_gateways(self) -> dict:
        """Organization -> gateway URL, for every gateway currently on the network."""
        with self._cv:
            return {g["org"]: g["url"] for g in self.gateways.values()}

    def close(self):
        for info in self._infos:
            try:
                self.zc.unregister_service(info)
            except Exception:
                pass
        self.zc.close()


# ---------------------------------------------------------------------------
# Delegation: one A2A message per attempt
# ---------------------------------------------------------------------------
def delegate(mesh: Mesh, http: httpx.Client, kind: str, envelope: dict,
             sender: Optional[dict] = None, wait: float = DISCOVERY_WAIT,
             seal: Optional[Callable[[dict, dict], dict]] = None, only: Optional[str] = None):
    """Send `envelope` as an A2A message to peers that accept `kind` until one accepts (its Task is WORKING).

    A peer, or the enforcement point in front of it, declines with a REJECTED Task and the reason, e.g. when the
    request's budget is spent. `seal(card, envelope)` adjusts the envelope for each peer before it is sent (e.g. the
    delegating agent tightens the token for it). Returns (card of the peer that took it, None) or (None, reason)."""
    me = sender["name"] if sender else None
    deadline, declines = time.time() + wait, {}
    while True:
        for card in mesh.find(kind, exclude=me, only=only):
            sealed = seal(card, envelope) if seal else envelope
            try:
                accepted, reason, _ = a2a_send(card, delegation_message(card, sealed), sealed)
            except Exception as e:
                declines[card["name"]] = f"unreachable ({type(e).__name__})"
                continue
            if accepted:
                return card, None
            declines[card["name"]] = reason
        if declines and all(str(v).startswith("insufficient budget") for v in declines.values()):
            deadline = 0   # waiting will not create budget
        if time.time() >= deadline:
            why = "; ".join(f"{k}: {v}" for k, v in declines.items()) or (
                f"agent {only!r} is not on the network" if only else "no peer on the network accepts it")
            return None, f"nobody took {kind!r} ({why})"
        mesh.wait_for(lambda: False, 1.0)   # give discovery a moment, then try again


def delegation_message(card: dict, env: dict) -> str:
    """The text an agent's model actually sees for one hop."""
    s = env.get("sender")
    sender_txt = f"{s['name']} ({s['framework']})" if s else "the user"
    parts = [f"DELEGATION (round {env['round']}) from {sender_txt} to {card['name']} ({card['framework']})",
             f"Original request: {env['user_request']}",
             f"Your sub-task: {card['subtask']}"]
    if env.get("feedback"):
        parts.append(f"Reviewer feedback on the previous report:\n{env['feedback']}")
        if env.get("kind") == "revision" and env.get("last_report"):
            parts.append(f"Previous report:\n{env['last_report']}")
    for w in env["work"]:
        parts.append(f"--- Output of {w['agent']} ({w['framework']}) ---\n{w['output']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# An agent node
# ---------------------------------------------------------------------------
class Hop:
    """What a framework run needs for one hop: its native limit and how to reach the model (and MCP tools)."""

    def __init__(self, limit: int, llm_base_url: str, headers: dict, mcp_base_url: Optional[str] = None,
                 extra_mcp: Optional[list] = None, token: Optional[str] = None):
        self.limit, self.llm_base_url, self.headers, self.mcp_base_url = limit, llm_base_url, headers, mcp_base_url
        self.extra_mcp = extra_mcp or []   # tools given to the agent at launch (--gateway-tool)
        self.token = token                 # the request's budget token (encoded), for patched frameworks


RunFn   = Callable[[str, str, Hop], Awaitable[str]]               # (system, message, hop) -> text
RouteFn = Callable[[dict, str], tuple]                             # (envelope, output) -> ("forward", kind, env) | ("done", status, env)


def status_from(reason: str) -> str:
    r = reason.lower()
    if "depth" in r:
        return "depth_exhausted"
    if "deadline" in r:
        return "deadline_exceeded"
    if "capability token rejected" in r or "capability_rejected" in r:
        return "rejected"
    if "insufficient budget" in r or "exhausted" in r or "budget_exceeded" in r:
        return "budget_exhausted"
    return "no_peer"


class AgentNode:
    """One agent = one process. Wraps a framework-specific `run` with discovery and delegation.

    The agent does not enforce the request's budget; enforcement points do (the metered-resource gateway that holds
    the user's credentials, and an optional A2A enforcement point in front of the agent, a2a_guard.py). In token
    mode the agent only carries the request's capability token: it attaches it to every model and tool call, offers
    it to the next peer, and may tighten it (env["flags"]["tighten"][peer name]) when it delegates.

    Partial deployment: run the agent without a2a_guard.py to leave its boundary unenforced, and with --own-model to
    have it use a model outside the user's credentials (its compute is then outside every enforcement point).
    """

    def __init__(self, card: dict, system: Union[str, Callable[[dict], str]], run: RunFn,
                 route: Optional[RouteFn] = None, max_concurrent: int = 2):
        self.card = dict(card)
        self.system = system
        self.run = run
        self.route = route or (lambda env, out: ("forward", self.card["produces"], env))
        self.max_concurrent = max_concurrent
        self.http = httpx.Client(timeout=30)
        self._lock = threading.Lock()
        self._active = 0
        self.own_ledger = None       # set with --own-model
        self.successors = None       # deployment config: the agents this one hands its output to (fan-out if > 1)
        self.join = []               # deployment config: inputs to wait for before running (join if > 1)
        self.revise_to = []          # deployment config: where a reviewer sends work back
        self._joins: dict = {}       # (request id, round) -> {kind: envelope} received so far

    # ---- accepting work (an A2A message) ---------------------------------
    def _consider(self, env: dict) -> Optional[str]:
        """None if this agent takes the work (and counts it as active), else the reason it declines."""
        kind, sender = env.get("kind"), env.get("sender")
        if kind not in self.card["accepts"]:
            return f"{self.card['name']} does not accept {kind!r}"
        if sender and not self.mesh.knows(sender["name"]):
            return f"unknown peer {sender['name']!r}"
        if not sender and kind != "request":
            return f"only peers on the mesh may delegate {kind!r}"
        if kind in self.join:   # one input of a join: buffered until the others arrive
            log(f"[{env.get('request_id')}] received join input {kind!r} from {sender['name']}")
            return None
        with self._lock:
            if self._active >= self.max_concurrent:
                return "busy"
            self._active += 1
        who = f"{sender['name']} ({sender['framework']})" if sender else "a client"
        log(f"[{env.get('request_id')}] accepted {kind!r} from {who}")
        return None

    def _executor(self):
        node = self

        class Executor(AgentExecutor):
            async def execute(self, context, event_queue):
                meta = json_format.MessageToDict(context.message.metadata)
                env = json.loads(meta.get(META_DELEGATION) or "{}")
                if meta.get(META_TOKEN):
                    env["token"] = bt.decode(meta[META_TOKEN])
                task = context.current_task or new_task_from_user_message(context.message)
                why = node._consider(env) if env else "no delegation envelope in the message metadata"
                if why:   # the first event already carries the decision, so return_immediately callers see it
                    task.status.state = TaskState.TASK_STATE_REJECTED
                    task.status.message.CopyFrom(new_text_message(why))
                    await event_queue.enqueue_event(task)
                    return
                task.status.state = TaskState.TASK_STATE_WORKING
                await event_queue.enqueue_event(task)
                updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
                if env.get("kind") in node.join:
                    env = node._join_add(env)
                    if env is None:   # more inputs to come: this message's task is done (its input is kept)
                        await updater.add_artifact(parts=[new_text_part(text="input received; waiting for the "
                                                                              "other inputs of the join")])
                        await updater.complete()
                        return
                    with node._lock:
                        node._active += 1
                status, out, error = await asyncio.to_thread(node._work, env)
                if status == "ok":
                    await updater.add_artifact(parts=[new_text_part(text=out, media_type="text/plain")])
                    await updater.complete()
                else:
                    await updater.failed(new_text_message(f"{status}: {error}"))

            async def cancel(self, context, event_queue):
                raise NotImplementedError("cancel is not supported")

        return Executor()

    # ---- join: wait for every input, then run once on the merged work ----
    def _join_add(self, env: dict) -> Optional[dict]:
        key = (env["request_id"], env["round"])
        with self._lock:
            buf = self._joins.setdefault(key, {})
            first = not buf
            buf[env["kind"]] = env
            if set(self.join) <= set(buf):
                del self._joins[key]
                return self._merge(list(buf.values()))
        if first:   # do not wait forever for an input whose branch failed
            threading.Timer(JOIN_WAIT, self._join_timeout, args=(key,)).start()
        return None

    def _join_timeout(self, key):
        with self._lock:
            buf = self._joins.pop(key, None)
            if not buf:
                return
            self._active += 1
        env = self._merge(list(buf.values()))
        log(f"[{key[0]}] join waited {JOIN_WAIT:.0f}s; running with {len(buf)} of {len(self.join)} inputs")
        threading.Thread(target=self._work, args=(env,), daemon=True).start()

    def _merge(self, envs: list) -> dict:
        """One envelope from the join's inputs: all their work, the deepest hop, and the most restrictive token
        (token chains cannot be merged, so the join continues under the tightest of the budgets it received)."""
        def tightness(e):
            eff = bt.effective(e["token"]) if e.get("token") else {}
            return (eff.get("max_depth", float("inf")), eff.get("max_cost_usd", float("inf")))
        base = dict(min(envs, key=tightness))
        seen, work = set(), []
        for e in envs:
            for w in e["work"]:
                if (w["agent"], w["kind"]) not in seen:
                    seen.add((w["agent"], w["kind"]))
                    work.append(w)
        base.update(work=work, hop=max(e["hop"] for e in envs),
                    sender={"name": " + ".join(e["sender"]["name"] for e in envs), "framework": "join"})
        if len(envs) > 1:   # n branches became one
            self._report(base, "/branch", {"delta": -(len(envs) - 1), "by": self.card["name"]})
        log(f"[{base['request_id']}] join: merged {len(envs)} inputs")
        return base

    # ---- doing the work -------------------------------------------------
    def _usage(self, env: dict) -> dict:
        if self.own_ledger is not None:
            return self.own_ledger.snapshot(env["request_id"])
        try:
            return self.http.get(f"{self._gateway(env)}/_requests/{env['request_id']}").json()
        except Exception:
            return {}

    def _gateway(self, env: dict) -> str:
        """The metered resources this agent uses: its own organization's gateway (its org's credentials).
        Falls back to the gateway the requester named if the organization has none on the network."""
        return self.mesh.gateway(timeout=2, org=self.card["org"]) or env["gateway"]

    def _report(self, env: dict, path: str, payload: dict):
        try:
            self.http.post(env["reply_to"] + path, json={"request_id": env["request_id"], **payload})
        except Exception as e:
            log(f"[{env['request_id']}] could not reach the requester: {e}")

    def _finish(self, env: dict, status: str, error: str = ""):
        self._report(env, "/result", {"status": status, "error": error, "by": self.card["name"],
                                      "work": env.get("work", []), "token": env.get("token")})

    def _work(self, env: dict):
        """Run one hop; returns (hop status, output, error) for the A2A task."""
        try:
            return self._handle(env)
        except Exception as e:
            self._finish(env, "error", f"{type(e).__name__}: {e}")
            return "error", "", f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                self._active -= 1

    def _handle(self, env: dict):
        rid, slug, fw = env["request_id"], self.card["slug"], self.card["framework"]
        token_mode = env.get("enforcement") == "token"
        if self.own_ledger is not None:   # compute outside the user's credentials: measured here, never charged
            if not self.own_ledger.is_open(rid):
                self.own_ledger.open(rid, None)
            base, headers = self.own_meter_url, {}
        else:
            base = self._gateway(env)
            headers = {bt.HEADER: bt.encode(env["token"])} if token_mode and env.get("token") else {}
        hop_ctx = Hop(self.limit, f"{base}/{rid}/{slug}/v1", headers, f"{self._gateway(env)}/{rid}/{slug}/mcp",
                      [gateway_mcp(n) for n in self.gateway_tools])

        system = self.system(env) if callable(self.system) else self.system
        before, t0 = self._usage(env), time.time()
        status, error, out = "ok", "", ""
        mcp_token = bt.encode(env["token"]) if token_mode and env.get("token") else None
        hop_ctx.token = mcp_token
        CURRENT_TOKEN.set(mcp_token)   # copied into the hop's event loop; tools/call carries it in _meta
        _process_token["value"] = mcp_token
        try:
            out = asyncio.run(self.run(system, delegation_message(self.card, env), hop_ctx))
        except Exception as e:   # each framework raises its own exception types
            error = f"{type(e).__name__}: {str(e)[:200]}"
            status = "native_limit" if type(e).__name__ in NATIVE_LIMIT_ERRORS else "error"
            if type(e).__name__ == "BudgetLimitError":   # refused by a patched framework (vecturae)
                status = {"depth": "depth_exhausted", "deadline": "deadline_exceeded"}.get(getattr(e, "kind", ""), "rejected")
        after = self._usage(env)
        delta = lambda key: round(after.get(key, {}).get(slug, 0) - before.get(key, {}).get(slug, 0), 9)
        if status == "error" and ("deadline" in error or "capability token rejected" in error):
            status = status_from(error)
        elif delta("denied") > 0:   # an enforcement point refused a call during this hop
            status = "budget_exhausted"
        sender = env.get("sender")
        hop = {"round": env["round"], "hop": env["hop"] + 1, "from": sender["framework"] if sender else "user",
               "to": fw, "agent": self.card["name"], "status": status, "llm_calls": delta("calls"),
               "denied": delta("denied"), "tokens": delta("tokens"), "tool_calls": delta("tool_calls"),
               "cost_usd": delta("cost_usd"), "charged_to_user": self.own_ledger is None,
               "admitted_by": env.get("admitted_by"), "org": self.card["org"],
               "seconds": round(time.time() - t0, 1), "error": error,
               "output_chars": len(out)}
        log(f"[{rid}] round {env['round']} hop {hop['hop']}: {status} calls={hop['llm_calls']} {hop['seconds']}s")
        self._report(env, "/hops", {"hop": hop})

        env = dict(env, hop=env["hop"] + 1, admitted_by=None)
        if status != "ok":
            self._finish(env, status, error)
            return status, out, error
        env["work"] = env["work"] + [{"agent": self.card["name"], "framework": fw, "role": self.card.get("role"),
                                      "kind": self.card["produces"], "output": out}]
        action, value, env = self.route(env, out)
        if action == "done":
            self._finish(env, value)
            return "ok", out, ""

        env = dict(env, kind=value, sender={"name": self.card["name"], "framework": fw})
        seal = None
        tighten = env.get("flags", {}).get("tighten", {})
        if token_mode and tighten:   # the agent's own choice to tighten what its delegatee may use
            def seal(card, e):
                caveats = tighten.get(card["name"])
                return dict(e, token=bt.attenuate(e["token"], caveats, by=f"agent:{self.card['name']}")) if caveats else e
        if self.successors is None:   # no deployment config: the first peer that accepts this kind of work
            peer, why = delegate(self.mesh, self.http, value, env, sender=env["sender"], seal=seal)
            if peer is None:
                log(f"[{rid}] {why}")
                self._finish(env, status_from(why), why)
            else:
                log(f"[{rid}] delegated {value!r} to {peer['name']} ({peer['framework']}) at {peer['endpoint']} over A2A")
            return "ok", out, ""

        targets = self.revise_to if value == "revision" else self.successors
        if not targets:   # an agent with no successors ends its branch
            self._finish(env, "completed")
            return "ok", out, ""
        if len(targets) > 1:   # fan-out: one branch becomes len(targets)
            self._report(env, "/branch", {"delta": len(targets) - 1, "by": self.card["name"]})
        results = {}

        def send(aid):
            results[aid] = delegate(self.mesh, self.http, value, env, sender=env["sender"], seal=seal, only=aid)
        threads = [threading.Thread(target=send, args=(aid,)) for aid in targets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for aid, (peer, why) in results.items():
            if peer is None:   # this branch ends here
                log(f"[{rid}] {why}")
                self._finish(env, status_from(why), why)
            else:
                log(f"[{rid}] delegated {value!r} to {peer['name']} ({peer['framework']}) at {peer['endpoint']} over A2A")
        return "ok", out, ""

    # ---- role and place in the deployment ----------------------------------
    def _apply_role(self, args):
        """Take this agent's role, and with a deployment config also its id, org, inputs, successors and tools."""
        import roles, topology
        spec = None
        if args.config:
            if not args.agent:
                raise SystemExit("--config needs --agent <id>")
            topo = topology.load(args.config)
            spec = topo.agent(args.agent)
            args.role = spec.get("role", "worker")
            args.org = spec["org"]
            args.gateway_tool = list(dict.fromkeys(args.gateway_tool + spec.get("tools", [])))
            if spec.get("limit"):
                args.limit = int(spec["limit"])
        if not args.role:
            return
        r = roles.role(args.role)
        self.card.update(role=args.role, name=r["name"], subtask=r["subtask"],
                         accepts=list(r["accepts"]), produces=r["produces"])
        self.system = r["system"]
        if r.get("route"):
            self.route = r["route"]
        if spec is not None:
            fw = spec["framework_info"]
            self.card.update(agent_id=spec["id"], name=spec.get("name") or spec["id"],
                             slug=f"{fw['slug']}-{spec['id']}", accepts=spec["accepts"], produces=spec["produces"],
                             subtask=spec.get("subtask") or r["subtask"])
            self.system = spec.get("system") or r["system"]
            self.successors, self.join, self.revise_to = spec["successors"], spec["join"], spec["revise_to"]

    # ---- process entry point --------------------------------------------
    def serve_forever(self):
        p = argparse.ArgumentParser(description=f"{self.card['name']} agent ({self.card['framework']})")
        p.add_argument("--host", default=os.getenv("MESH_HOST", "127.0.0.1"),
                       help="address to bind and advertise (use the LAN IP to run agents on several machines)")
        p.add_argument("--port", type=int, default=0)
        p.add_argument("--limit", type=int, default=self.card["native_limit"]["value"],
                       help=f"native limit ({self.card['native_limit']['name']})")
        p.add_argument("--own-model", metavar="URL", default=None,
                       help="use this model endpoint (the agent's own, outside the user's credentials) "
                            "instead of the metered-resource gateway")
        p.add_argument("--org", default=os.getenv("MESH_ORG", "default"),
                       help="the organization this agent belongs to (it uses that organization's gateway)")
        p.add_argument("--config", help="deployment config (deployments/*.yaml): role, org, edges come from it")
        p.add_argument("--agent", help="this agent's id in the deployment config")
        p.add_argument("--role", help="play this role (roles.py) instead of the runner's default role")
        p.add_argument("--gateway-tool", action="append", default=[], metavar="NAME",
                       help="give the agent a paid MCP tool behind the metered-resource gateway (gateway.py --mcp NAME=URL)")
        args = p.parse_args()
        self._apply_role(args)
        self.limit, self.gateway_tools = args.limit, args.gateway_tool
        self.card["org"] = args.org
        os.environ["VECTURAE_TRUST_FILE"] = enforcement.trust_file(args.org)   # the patched frameworks' users
        if args.gateway_tool:
            self.card["tools"] = args.gateway_tool
        self.card["native_limit"] = dict(self.card["native_limit"], value=args.limit)
        guarded = os.getenv("MESH_GUARDED") == "1"
        self.card["model"] = "own" if args.own_model else "user credentials (gateway)"
        if args.own_model:
            self.own_ledger, _, self.own_meter_url = start_meter(args.own_model)

        host = "127.0.0.1" if guarded else args.host
        port = args.port or free_port(host)
        url = f"http://{host}:{port}"
        public_url = os.getenv("MESH_PUBLIC_URL", url)   # behind a2a_guard.py: the guard's address
        self.card["endpoint"] = public_url
        card = a2a_card(self.card, public_url)
        handler = DefaultRequestHandler(agent_executor=self._executor(), task_store=InMemoryTaskStore(),
                                        agent_card=card)
        app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.05)
        self.mesh = Mesh(args.host).browse()
        if not guarded:   # behind a2a_guard.py the enforcement point owns the public endpoint and advertisement
            instance = f"{slugify(self.card['name'])}-{self.card['slug']}-{port}"
            self.mesh.advertise(AGENT_SERVICE, instance, port, {
                "name": self.card["name"], "framework": self.card["framework"],
                "accepts": ",".join(self.card["accepts"]), "produces": self.card["produces"], "card": CARD_PATH})
        log(f"[org {self.card['org']}] {self.card['name']} ({self.card['framework']}) A2A server up at {url}"
            + (" behind an A2A enforcement point" if guarded else ", advertising on mDNS (no enforcement point)")
            + f"; {self.card['native_limit']['name']}={self.limit}; model: {self.card['model']}")

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        stop.wait()
        log("shutting down")
        self.mesh.close()
        server.should_exit = True


# ---------------------------------------------------------------------------
# MCP server specs
# ---------------------------------------------------------------------------
# Every agent script has an MCP_SERVERS list in one framework-neutral format, e.g.
#   gateway_mcp("tcp-reference")   a paid tool behind the metered-resource gateway: charged to the request's
#                                  budget, called with the request's capability token
#   {"name": "search", "transport": "http", "url": "http://localhost:8000/mcp", "headers": {...}}
#   {"name": "fs", "transport": "stdio", "command": "npx", "args": [...]}
# Each hop resolves them with resolve_mcp(specs, hop), then turns them into its framework's native MCP objects.
HERE = os.path.dirname(os.path.abspath(__file__))


def local_mcp(script: str, name: Optional[str] = None) -> dict:
    """Spec for a stdio MCP server script shipped in delegation_mesh/mcp_servers/."""
    return {"name": name or os.path.splitext(script)[0], "transport": "stdio",
            "command": sys.executable, "args": [os.path.join(HERE, "mcp_servers", script)]}


def gateway_mcp(name: str) -> dict:
    """Spec for an MCP server that the metered-resource gateway exposes (gateway.py --mcp NAME=URL)."""
    return {"name": name, "transport": "gateway"}


def resolve_mcp(specs: list, hop: "Hop") -> list:
    """Concrete specs for this hop: gateway tools get this request's URL."""
    out = []
    for s in list(specs) + hop.extra_mcp:
        if s["transport"] == "gateway":
            s = {"name": s["name"], "transport": "http", "url": f"{hop.mcp_base_url}/{s['name']}"}
            # the token is not sent as a header: every tools/call carries it in _meta (see _call_tool_with_token)
        out.append(s)
    return out

"""Instrumentation proxy: every model call, MCP call and A2A message of the testbed passes through here.

It observes, it never changes what it forwards:

  /llm/<org>/<agent>/v1/...          -> the model         records calls, prompt/completion tokens, dollar cost, time
  /mcp/<org>/<agent>/<server>/mcp    -> an MCP server     records tools/call, and the limit in the request's _meta
  /a2a/<from>/<org>/<agent>/...      -> an org's agent    records the limit in the A2A message / request metadata
                                                          (Agent Cards are rewritten so clients keep using the proxy)

Events go to runs/<config name>/events.jsonl, each tagged with the request it belongs to (POST /_request {"id": ..}
sets the current one; the testbed runs one request at a time). GET /_events/<id> returns a request's events.

    python testbed/common/instrument.py --config testbed/configs/example.yaml
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob

from common.config import FAKE_PORT, PROXY, PROXY_PORT, Config, agent_port, mcp_port, upstreams_dir

LOCK = threading.Lock()
STATE = {"request": None}


def find_limit(obj) -> dict | None:
    """The limit the originator attached, wherever this message carries it (None if it does not)."""
    if isinstance(obj, dict):
        if "limit" in obj and isinstance(obj["limit"], dict):
            return obj["limit"]
        for v in obj.values():
            hit = find_limit(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_limit(v)
            if hit is not None:
                return hit
    return None


def make_handler(cfg: Config, events_path: str):
    client = httpx.Client(timeout=1800)
    model = cfg.raw.get("model") or {"kind": "fake"}
    prices = {"usd_per_1m_input": 0.15, "usd_per_1m_output": 0.60, "usd_per_tool_call": 0.0005,
              **(cfg.raw.get("prices") or {})}

    marker = cfg.marker   # the unique string inside the user's limit: seen anywhere in a request?

    def seen(raw) -> bool | None:
        if not marker:
            return None
        return marker in (raw.decode(errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw))

    def emit(**event):
        event = {"t": time.time(), "request": STATE["request"], **event}
        with LOCK, open(events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(n) if n else b""

        def _relay(self, method, url, body, headers, on_line=None):
            """Forward and stream the answer back; on_line sees every line of a streamed (SSE) answer."""
            with client.stream(method, url, content=body or None, headers=headers) as r:
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() in ("content-type", "mcp-session-id", "mcp-protocol-version"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                chunks, client_gone = [], False
                for chunk in r.iter_raw():
                    chunks.append(chunk)
                    if client_gone:
                        continue
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):   # the client hung up (e.g. after the final
                        client_gone = True                            # streamed event); the call still happened
            self.close_connection = True
            return r.status_code, b"".join(chunks)

        # ---- dispatch ----------------------------------------------------------------------------
        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

        def _route(self, method):
            parts = self.path.lstrip("/").split("/")
            body = self._body() if method == "POST" else b""
            if parts[0] == "_request" and method == "POST":
                STATE["request"] = json.loads(body or b"{}").get("id")
                return self._json(200, {"request": STATE["request"]})
            if parts[0] == "_report" and method == "POST":   # an agent reports its framework's own counters
                emit(type="framework", **json.loads(body or b"{}"))
                return self._json(200, {"ok": True})
            if parts[0] == "_events":
                with LOCK:
                    rows = [json.loads(l) for l in open(events_path)] if os.path.exists(events_path) else []
                return self._json(200, [r for r in rows if r.get("request") == parts[1]])
            if parts[0] == "llm":
                return self._llm(method, parts[1], parts[2], "/".join(parts[3:]), body)
            if parts[0] == "mcp":
                return self._mcp(method, parts[1], parts[2], parts[3], "/".join(parts[4:]), body)
            if parts[0] == "a2a":
                return self._a2a(method, parts[1], parts[2], parts[3], "/".join(parts[4:]), body)
            if parts[0] == "native":
                return self._native(method, parts[1], parts[2], parts[3], "/".join(parts[4:]), body)
            self._json(404, {"error": "unknown route"})

        # ---- the model ---------------------------------------------------------------------------
        def _llm(self, method, org, agent, rest, body):
            if model.get("kind", "fake") == "fake":
                upstream = f"http://127.0.0.1:{FAKE_PORT}/{org}.{agent}/{rest}"
            else:
                upstream = model["url"].rstrip("/") + "/" + rest.split("/", 1)[1] if "/" in rest else model["url"]
            headers = {"Content-Type": "application/json"}
            if model.get("api_key_env"):
                headers["Authorization"] = f"Bearer {os.environ.get(model['api_key_env'], '')}"
            if method != "POST" or not rest.endswith("chat/completions"):
                r = client.request(method, upstream, content=body or None, headers=headers)
                return self._json(r.status_code, r.json())
            req = json.loads(body or b"{}")
            if model.get("model"):
                req["model"] = model["model"]
            usage = {}

            def on_line(line):
                if line.startswith("data: {") and '"usage"' in line:
                    usage.update(json.loads(line[6:]).get("usage") or {})
            t0 = time.time()
            status, raw = self._relay("POST", upstream, json.dumps(req).encode(), headers)
            if req.get("stream"):
                for line in raw.decode(errors="ignore").splitlines():
                    on_line(line)
            else:
                try:
                    usage = json.loads(raw).get("usage") or {}
                except ValueError:
                    pass
            p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            emit(type="model_call", org=org, agent=agent, status=status, prompt_tokens=p, completion_tokens=c,
                 cost_usd=p * prices["usd_per_1m_input"] / 1e6 + c * prices["usd_per_1m_output"] / 1e6,
                 seconds=round(time.time() - t0, 3), limit_in_request=find_limit(req) is not None,
                 marker_in_request=seen(body))

        # ---- MCP -----------------------------------------------------------------------------------
        def _mcp(self, method, org, agent, server, rest, body):
            spec = cfg.mcp_servers().get(server) or {}
            upstream = spec.get("url") or f"http://127.0.0.1:{mcp_port(cfg, server)}/mcp"
            calls = []
            if method == "POST" and body:
                msg = json.loads(body)
                calls = [m for m in (msg if isinstance(msg, list) else [msg]) if m.get("method") == "tools/call"]
            fwd = {k: v for k, v in self.headers.items() if k.lower() in
                   ("content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id")}
            t0 = time.time()
            status, _ = self._relay(method, upstream, body, fwd)
            for m in calls:
                meta = (m.get("params") or {}).get("_meta") or {}
                emit(type="tool_call", org=org, agent=agent, server=server, tool=(m.get("params") or {}).get("name"),
                     status=status, t_start=t0, seconds=round(time.time() - t0, 3), cost_usd=prices["usd_per_tool_call"],
                     limit_in_meta=find_limit(meta), marker_in_request=seen(json.dumps(m)))

        # ---- a framework's own run API (entry_via: native) ---------------------------------------------------
        def _native(self, method, sender, org, agent, rest, body):
            """A call to the organization's own run API (e.g. LangGraph's /runs/wait), on its server's base address."""
            known = {}
            for f in glob.glob(os.path.join(upstreams_dir(cfg), "*.json")):
                known.update(json.load(open(f)))
            url = known.get(f"{org}.{agent}", "")
            base = url.split("/a2a/")[0] if "/a2a/" in url else url.rstrip("/")
            t0 = time.time()
            status, _ = self._relay(method, f"{base}/{rest}", body, {"Content-Type": "application/json"})
            emit(type="native_request", sender=sender, org=org, agent=agent, path=rest, status=status,
                 marker_in_request=seen(body), t_start=t0, seconds=round(time.time() - t0, 3))

        # ---- A2A -----------------------------------------------------------------------------------
        def _a2a(self, method, sender, org, agent, rest, body):
            # each agent's own A2A endpoint, as its organization's process wrote it (native servers have their own paths)
            files = glob.glob(os.path.join(upstreams_dir(cfg), "*.json"))
            if cfg.raw.get("a2a_upstreams_file"):
                files.append(cfg.raw["a2a_upstreams_file"])
            known = {}
            for f in files:
                if os.path.exists(f):
                    known.update(json.load(open(f)))
            url = known.get(f"{org}.{agent}")
            if url and method == "POST":
                target = url
            elif url and method == "GET":   # the agent's card, where its own A2A server publishes it
                if "/a2a/" in url:          # LangGraph Agent Server: /a2a/{assistant_id}, card per assistant
                    base, aid = url.split("/a2a/")
                    target = f"{base}/.well-known/agent-card.json?assistant_id={aid.strip('/')}"
                else:
                    target = url.rstrip("/") + "/.well-known/agent-card.json"
            else:
                target = f"http://127.0.0.1:{agent_port(cfg, org, agent)}/{rest}"
            headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "a2a-version", "accept")}
            if method == "GET":   # the Agent Card: point every interface at this proxy
                r = client.get(target, headers=headers)
                card = r.json()
                here = f"{PROXY}/a2a/{sender}/{org}/{agent}"
                for itf in card.get("supportedInterfaces", []):
                    itf["url"] = here
                if "url" in card:
                    card["url"] = here
                return self._json(r.status_code, card)
            req = json.loads(body or b"{}")
            params = req.get("params") or {}
            t0 = time.time()
            status, raw = self._relay("POST", target, body, headers)
            message = params.get("message") or {}
            text = " ".join(p.get("text", "") for p in message.get("parts", []))
            emit(type="a2a_message", sender=sender, org=org, agent=agent, method=req.get("method"), status=status,
                 limit_in_message=find_limit(message.get("metadata")),
                 limit_in_request=find_limit(params.get("metadata")), context=message.get("contextId"),
                 text_head=text[:200], text=text[:4000], t_start=t0, seconds=round(time.time() - t0, 3),
                 marker_in_request=seen(body))

    return Handler


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = Config(a.config)
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    events = os.path.join(run_dir, "events.jsonl")
    print(f"instrumentation proxy on {PROXY}, events -> {events}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), make_handler(cfg, events)).serve_forever()

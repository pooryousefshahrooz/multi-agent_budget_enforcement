"""Metered-resource enforcement point: the model API and paid tools, behind the user's credentials.

An OpenAI-compatible model proxy plus an MCP proxy. Only this process holds the provider credentials, so every
call charged to the user passes through it, whatever the agents do.

  model:  http://<gw>/<request_id>/<agent-slug>/v1/chat/completions
  tools:  http://<gw>/<request_id>/<agent-slug>/mcp/<server>          (streamable-HTTP MCP, proxied;
          the token rides in each tools/call: params._meta["vecturae/budget-token"])

Two modes, chosen per call:
  token   the call carries the request's capability token (model calls: X-Capability-Token header; tool calls: the
          MCP request's _meta). The gateway verifies it against the
          users it serves, refuses it after the deadline or when any block's limit is spent, clamps the completion
          length to the remaining budget, and charges cost, model tokens, LLM calls and tool calls to every block of
          the chain (hierarchical counters, enforcement.py).
  budget  legacy baseline (D1-D3): the requester opened the request with POST /_requests/<id> {"budget": N} and the
          gateway caps LLM calls for that request id. Calls with neither a token nor an opened request are refused.

  GET  /_requests/<id>   usage per agent slug (calls, denied, tokens, tool_calls, cost_usd)
  GET  /_ledger/<id>     token mode: per block, its limits and what was charged at or below it
  POST /_remaining       token mode: verify the token in the header and return the remaining budget
"""
import json, threading, time
from collections import Counter
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional

import httpx
import budget_token as bt
from enforcement import CHARGE_KEY, HierarchicalLedger, Pricing, TrustStore, check_token


class Ledger:
    """Per-request usage by agent slug (both modes), and the call budget of legacy budget-mode requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = {}

    def _get(self, rid):
        return self.requests.setdefault(rid, {"budget": None, "opened": False, "calls": Counter(),
                                              "denied": Counter(), "tokens": Counter(),
                                              "tool_calls": Counter(), "cost_usd": Counter()})

    def open(self, rid, budget):
        with self._lock:
            self.requests.pop(rid, None)
            r = self._get(rid)
            r["budget"], r["opened"] = budget, True

    def is_open(self, rid) -> bool:
        with self._lock:
            return self.requests.get(rid, {}).get("opened", False)

    def try_acquire(self, rid, slug) -> bool:
        with self._lock:
            r = self._get(rid)
            if r["budget"] is not None and sum(r["calls"].values()) >= r["budget"]:
                r["denied"][slug] += 1
                return False
            r["calls"][slug] += 1
            return True

    def add(self, rid, slug, **amounts):
        with self._lock:
            r = self._get(rid)
            for k, v in amounts.items():
                r[k][slug] += v

    def snapshot(self, rid):
        with self._lock:
            r = self._get(rid)
            return {"budget": r["budget"],
                    **{k: dict(r[k]) for k in ("calls", "denied", "tokens", "tool_calls", "cost_usd")}}


def make_handler(ledger: Ledger, upstream: str, mcp_upstreams: Optional[dict] = None,
                 trust: Optional[TrustStore] = None, pricing: Optional[Pricing] = None,
                 api_key: Optional[str] = None):
    client = httpx.Client(timeout=600)
    mcp_upstreams = mcp_upstreams or {}
    trust, pricing, tree = trust or TrustStore(), pricing or Pricing(), HierarchicalLedger()
    upstream_auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: one connection per request. Framework clients that cache connection pools across event
        # loops (LangChain does) would otherwise reuse a pooled connection bound to a closed loop.

        def log_message(self, *args):
            pass

        def _reply(self, code, payload, headers=None):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self, code, reason, kind="budget_exceeded"):
            return self._reply(code, {"error": {"message": reason, "type": kind, "code": kind}})

        def _split(self):
            # /<request_id>/<slug>/<rest>  ->  (request_id, slug, rest)
            parts = self.path.lstrip("/").split("/", 2)
            parts += [""] * (3 - len(parts))
            return parts[0], parts[1], parts[2]

        def _token(self):
            return check_token(self.headers.get(bt.HEADER), trust)

        # ---- admin ---------------------------------------------------------
        def do_GET(self):
            if self.path.startswith("/_requests/"):
                return self._reply(200, ledger.snapshot(self.path.rsplit("/", 1)[1]))
            if self.path.startswith("/_ledger/"):
                return self._reply(200, tree.report(self.path.rsplit("/", 1)[1]))
            rid, slug, rest = self._split()
            if rest.startswith("mcp/"):
                return self._mcp("GET", rid, slug, rest, b"")
            r = client.get(f"{upstream}/{rest[3:] if rest.startswith('v1/') else rest}", headers=upstream_auth)
            self._reply(r.status_code, r.json())

        def do_DELETE(self):
            rid, slug, rest = self._split()
            if rest.startswith("mcp/"):
                return self._mcp("DELETE", rid, slug, rest, b"")
            self._reply(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            if self.path.startswith("/_requests/"):
                rid = self.path.rsplit("/", 1)[1]
                ledger.open(rid, json.loads(body or b"{}").get("budget"))
                return self._reply(200, ledger.snapshot(rid))
            if self.path == "/_remaining":
                token, info = self._token()
                if token is None:
                    return self._reply(200, {"ok": False, "reason": info})
                with tree._lock:
                    tree._learn(token, info["chain"])
                    rem = tree.remaining(info["chain"])   # includes calls reserved but not yet settled
                return self._reply(200, {"ok": True, "caveats": info["caveats"], "remaining": rem})
            rid, slug, rest = self._split()
            if rest.startswith("mcp/"):
                return self._mcp("POST", rid, slug, rest, body)
            if rest.startswith("v1/"):
                return self._model(rid, slug, rest[3:], body)
            self._reply(404, {"error": "not found"})

        # ---- model API -------------------------------------------------------
        def _model(self, rid, slug, rest, body):
            req = json.loads(body or b"{}")
            is_call = rest.endswith("chat/completions")
            token, info = (None, None)
            if self.headers.get(bt.HEADER):
                token, info = self._token()
                if token is None:
                    ledger.add(rid, slug, denied=1)
                    kind = "deadline_exceeded" if info == "deadline passed" else "capability_rejected"
                    return self._refuse(403, f"capability token rejected: {info}", kind)
            elif is_call and not ledger.is_open(rid):
                ledger.add(rid, slug, denied=1)
                return self._refuse(403, "no capability token, and no open budget for this request", "capability_rejected")

            reserved, chain = {}, None
            if is_call and token is not None:
                # conservative prompt estimate (JSON characters / 3, tool schemas included): reserved before the call
                prompt_json = json.dumps([req.get("messages", []), req.get("tools", [])])
                chain, est_prompt = info["chain"], max(1, len(prompt_json) // 3)
                requested = req.get("max_tokens") or req.get("max_completion_tokens") or 10 ** 6
                fixed = {"llm_calls": 1, "model_tokens": est_prompt, "cost_usd": est_prompt * pricing.inp}
                per_unit = {"model_tokens": 1, "cost_usd": pricing.out}
                ok, why, units, clamped = tree.reserve(token, chain, fixed, per_unit, requested)
                if not ok:
                    ledger.add(rid, slug, denied=1)
                    tree.refuse(chain, slug, why)
                    return self._refuse(403, f"request budget: {why} ({slug})")
                reserved = {k: fixed.get(k, 0) + units * per_unit.get(k, 0) for k in set(fixed) | set(per_unit)}
                if clamped:   # the completion may not run past what was reserved for it
                    req["max_tokens"] = units
                    req.pop("max_completion_tokens", None)
            elif is_call and not ledger.try_acquire(rid, slug):
                budget = ledger.snapshot(rid)["budget"]
                return self._refuse(403, f"Request-wide LLM budget of {budget} calls exhausted ({slug})")

            settled = []

            def account(usage):
                if not is_call or settled:
                    return
                p, c = (usage or {}).get("prompt_tokens", 0), (usage or {}).get("completion_tokens", 0)
                cost = pricing.model_call(p, c)
                if usage:
                    ledger.add(rid, slug, tokens=p + c, cost_usd=cost)
                if token is not None:
                    if usage:
                        ledger.add(rid, slug, calls=1)
                    actual = {"llm_calls": 1, "model_tokens": p + c, "cost_usd": cost} if usage else {}
                    tree.settle(chain, reserved, actual, slug)   # no usage reported: refund the reservation
                settled.append(True)

            url, headers = f"{upstream}/{rest}", {"Content-Type": "application/json", **upstream_auth}
            # the end-to-end deadline also bounds a call already in flight: cut it off when the deadline arrives
            deadline = (info or {}).get("caveats", {}).get("deadline") if token is not None else None
            timeout = httpx.Timeout(max(deadline - time.time(), 0.1)) if deadline else httpx.USE_CLIENT_DEFAULT
            try:
                return self._forward(req, url, headers, timeout, account)
            except httpx.TimeoutException:
                if chain is not None:   # the provider may bill a cut-off generation: re-charge the reservation
                    tree.settle(chain, {}, reserved, slug)   # (_forward's finally has already refunded it)
                ledger.add(rid, slug, denied=1)
                return self._refuse(403, f"capability token rejected: deadline passed during the call ({slug})",
                                    "deadline_exceeded")

        def _forward(self, req, url, headers, timeout, account):
            if req.get("stream"):
                req.setdefault("stream_options", {"include_usage": True})
                try:
                    with client.stream("POST", url, json=req, headers=headers, timeout=timeout) as r:
                        self.send_response(r.status_code)
                        self.send_header("Content-Type", r.headers.get("content-type", "text/event-stream"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for line in r.iter_lines():
                            if line.startswith("data: {") and '"usage"' in line:
                                account(json.loads(line[6:]).get("usage"))
                            self.wfile.write((line + "\n").encode())
                            self.wfile.flush()
                finally:
                    account(None)
                self.close_connection = True
            else:
                try:
                    r = client.post(url, json=req, headers=headers, timeout=timeout)
                    data = r.json()
                    account(data.get("usage"))
                finally:
                    account(None)
                self._reply(r.status_code, data)

        # ---- MCP tools -------------------------------------------------------
        def _mcp(self, method, rid, slug, rest, body):
            parts = rest.split("/", 2)                       # mcp/<server>[/...]
            target = mcp_upstreams.get(parts[1] if len(parts) > 1 else "")
            if not target:
                return self._reply(404, {"error": f"no MCP server {parts[1:2]} behind this gateway"})
            calls, msg = [], None
            if method == "POST" and body:
                msg = json.loads(body)
                calls = [m for m in (msg if isinstance(msg, list) else [msg]) if m.get("method") == "tools/call"]

            def refuse_calls(reason):   # a tool that could not run (not a protocol error): the model sees it
                ledger.add(rid, slug, denied=len(calls))
                err = [{"jsonrpc": "2.0", "id": m.get("id"), "result": {
                    "content": [{"type": "text", "text": f"Tool call refused: {reason}."}], "isError": True}}
                    for m in calls]
                return self._reply(200, err if isinstance(msg, list) else err[0])

            # only tools/call is paid; initialize, tools/list and notifications pass without a token.
            # The token travels inside the MCP request: params._meta["vecturae/budget-token"].
            token, info = (None, None)
            if calls:
                raw = ((calls[0].get("params") or {}).get("_meta") or {}).get(bt.META_KEY)
                if raw:
                    token, info = check_token(raw, trust)
                    if token is None:
                        return refuse_calls(f"capability token rejected: {info}")
                elif not ledger.is_open(rid):
                    return refuse_calls("no capability token in the request's _meta, and no open budget for this request")
            tool_charge = {"tool_calls": len(calls), "cost_usd": pricing.tool * len(calls)}
            if calls and token is not None:
                ok, why, _, _ = tree.reserve(token, info["chain"], tool_charge)
                if not ok:
                    tree.refuse(info["chain"], slug, why)
                    return refuse_calls(f"request budget: {why}")

            fwd = {k: v for k, v in self.headers.items()
                   if k.lower() in ("content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id")}
            with client.stream(method, target, content=body or None, headers=fwd) as r:
                served = r.status_code < 400
                if calls and served:
                    ledger.add(rid, slug, **tool_charge)
                if calls and token is not None:   # keep the reservation if served, refund it if not
                    tree.settle(info["chain"], tool_charge, tool_charge if served else {}, slug)
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() in ("content-type", "mcp-session-id", "mcp-protocol-version"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in r.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()
            self.close_connection = True

    return Handler


def start_meter(upstream: str, host: str = "127.0.0.1", port: int = 0, **kw):
    """Run a metering proxy on a background thread. Returns (ledger, server, url)."""
    ledger = Ledger()
    server = ThreadingHTTPServer((host, port), make_handler(ledger, upstream.rstrip("/"), **kw))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return ledger, server, f"http://{host}:{server.server_address[1]}"

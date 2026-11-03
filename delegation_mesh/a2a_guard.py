"""A2A enforcement point: the protocol boundary in front of one agent, run by the operator, not by the agent.

It starts the agent on a private port, then owns the agent's public endpoint and its mDNS advertisement, so every
delegation into the agent crosses this process. It is a proxy for A2A JSON-RPC: for a `SendMessage` whose message
metadata carries a capability token it

  1. verifies the token (signed by a user this operator serves, unbroken chain, no loosened limit),
  2. checks the end-to-end deadline and the remaining delegation depth,
  3. asks the metered-resource gateway what is left of the cost budget along the token's chain,
  4. charges one level of depth and passes a tightened budget onward: it appends a block with max_depth - 1 and the
     remaining cost limits, so the agent and everything it delegates to inherit the tightened values.

A request it refuses gets an A2A Task in state REJECTED with the reason, and never reaches the agent; an admitted one
is forwarded with the tightened token in the message metadata. Messages without a token (legacy gateway mode) and all
other A2A methods pass through unchanged.

    python delegation_mesh/a2a_guard.py delegation_mesh/planner_langgraph.py [agent options...]
"""
import argparse, json, os, signal, subprocess, sys, threading, time

import httpx
import budget_token as bt
from enforcement import DEFAULT_ORG, TrustStore, trust_file
from mesh import (AGENT_SERVICE, CARD_PATH, EXT_CHAIN, META_DELEGATION, META_TOKEN, Mesh, card_view, free_port,
                  log, rejected_task, serve_json, slugify)


# an agent cannot do anything without these; an exhausted tool budget only refuses tool calls (at the gateway)
NEEDED_BY_EVERY_HOP = ("max_cost_usd", "max_llm_calls", "max_model_tokens")


class Guard:
    def __init__(self, agent_url: str, host: str, org: str = DEFAULT_ORG):
        self.agent_url, self.host, self.org = agent_url, host, org
        self.http = httpx.Client(timeout=30)
        self.trust = TrustStore(trust_file(org))   # the users this organization serves
        self.card = None
        self.mesh = None

    # ---- the checks --------------------------------------------------------
    def _assess(self, token: dict, request_id: str):
        """Returns (None, tightened caveats) if the request may cross this boundary, else (reason, None)."""
        ok, info = bt.verify(token, self.trust.keys(), request_id)
        if not ok:
            return (f"insufficient budget: {info}" if info == "deadline passed" else f"capability token rejected: {info}"), None
        eff = info["caveats"]
        if "max_depth" in eff and eff["max_depth"] < 1:
            return "insufficient budget: delegation depth exhausted", None
        tight = {"max_depth": eff["max_depth"] - 1} if "max_depth" in eff else {}
        gateway = self.mesh.gateway(timeout=2, org=self.org) if self.mesh else None   # its own org's
        if gateway:
            try:
                rem = self.http.post(gateway + "/_remaining", headers={bt.HEADER: bt.encode(token)}).json()
                for k, left in (rem.get("remaining") or {}).items():
                    if left <= 0 and k in NEEDED_BY_EVERY_HOP:
                        return f"insufficient budget: {k} exhausted", None
                    tight[k] = int(left) if k != "max_cost_usd" else round(left, 9)
            except Exception:
                pass   # the metered resource still enforces cost; this boundary still enforces depth and deadline
        return None, tight

    def handle(self, method, path, body):
        if method == "GET" and path == CARD_PATH:
            return 200, self.card
        if method == "POST" and body.get("method") == "SendMessage":
            msg = (body.get("params") or {}).get("message") or {}
            meta = msg.get("metadata") or {}
            env = json.loads(meta.get(META_DELEGATION) or "{}")
            if env.get("enforcement") == "token":
                token = bt.decode(meta[META_TOKEN]) if meta.get(META_TOKEN) else None
                why, tight = self._assess(token, env.get("request_id"))
                if why:
                    log(f"[{env.get('request_id')}] refused at the boundary: {why}")
                    return 200, rejected_task(body.get("id"), why, msg.get("contextId"))
                env = dict(env, admitted_by=f"a2a:{self.name}")
                meta = dict(meta, **{META_TOKEN: bt.encode(bt.attenuate(token, tight, by=f"a2a:{self.name}")),
                                     META_DELEGATION: json.dumps(env)})
                body = dict(body, params=dict(body["params"], message=dict(msg, metadata=meta)))
                log(f"[{env['request_id']}] admitted, passing on {tight}")
        r = self.http.request(method, self.agent_url + path, json=body if method == "POST" else None,
                              headers={"A2A-Version": "1.0"})
        return r.status_code, r.json()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A2A enforcement point in front of one agent")
    p.add_argument("agent", help="the agent script to run behind this boundary")
    p.add_argument("--host", default=os.getenv("MESH_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--org", default=os.getenv("MESH_ORG", DEFAULT_ORG),
                   help="the organization operating this boundary (and the agent behind it)")
    args, agent_args = p.parse_known_args()

    agent_port, port = free_port(), args.port or free_port(args.host)
    url = f"http://{args.host}:{port}"
    child = subprocess.Popen([sys.executable, "-u", args.agent, "--port", str(agent_port), "--org", args.org, *agent_args],
                             env={**os.environ, "MESH_GUARDED": "1", "MESH_PUBLIC_URL": url})
    agent_url = f"http://127.0.0.1:{agent_port}"
    guard = Guard(agent_url, args.host, args.org)
    deadline = time.time() + 180
    while guard.card is None:
        if child.poll() is not None:
            sys.exit(f"agent exited with code {child.returncode}")
        try:
            guard.card = httpx.get(agent_url + CARD_PATH, timeout=2).json()
        except Exception:
            if time.time() > deadline:
                child.terminate()
                sys.exit("agent did not come up")
            time.sleep(0.5)

    for ext in guard.card.get("capabilities", {}).get("extensions", []):   # the card says this boundary enforces
        if ext.get("uri") == EXT_CHAIN:
            ext.setdefault("params", {})["enforcement_point"] = "a2a"
    view = card_view(guard.card)
    guard.name = view["name"]
    server, _ = serve_json(guard.handle, args.host, port)
    guard.mesh = Mesh(args.host).browse()
    guard.mesh.advertise(AGENT_SERVICE, f"{slugify(view['name'])}-{view['slug']}-{port}", port, {
        "name": view["name"], "framework": view["framework"], "accepts": ",".join(view["accepts"]),
        "produces": view["produces"], "card": CARD_PATH, "enforcement_point": "a2a"})
    log(f"[org {args.org}] A2A enforcement point for {view['name']} ({view['framework']}) at {url} → A2A agent at {agent_url}")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.wait(1):
        if child.poll() is not None:
            log("agent exited")
            break
    guard.mesh.close()
    server.shutdown()
    if child.poll() is None:
        child.terminate()
        child.wait(10)

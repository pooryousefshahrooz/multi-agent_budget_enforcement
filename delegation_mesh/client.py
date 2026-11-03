"""Requester: submits a user request into the mesh and collects what the agents report back.

The requester is not an orchestrator. It only
  * discovers the network over mDNS (agents + gateway),
  * sets the request's limit, in one of two ways:
      gateway  opens the request at the metered-resource gateway with an LLM-call budget (legacy baseline), or
      token    signs the request's budget (delegation depth, end-to-end deadline, cost, and finer cost caveats) as
               an attenuable capability token that travels with the request; every enforcement point it crosses
               verifies it, charges it, and passes a tightened budget onward,
  * offers the request to a peer that accepts `request` and hands it over once that peer accepts,
  * listens on a reply-to address for hop records and the final result.
Every hop after the first is chosen by the agents themselves.

    python delegation_mesh/client.py ["your request"] [--budget N]                       # legacy gateway budget
    python delegation_mesh/client.py ["your request"] --enforcement token --max-depth 8 --deadline-s 300 \
                                     --max-cost-usd 0.01 [--max-llm-calls N --max-model-tokens N --max-tool-calls N]
"""
import argparse, threading, time, uuid
from typing import Optional

import httpx
import budget_token as bt
from enforcement import TrustStore, known_orgs, trust_file
from mesh import Mesh, delegate, log, serve_json, status_from


class MeshClient:
    def __init__(self, host: str = "127.0.0.1"):
        self.mesh = Mesh(host).browse()
        self.identity = bt.Identity()   # the user's key: root of every budget it signs
        for org in known_orgs():   # stand-in for opening an account with every organization's operator
            TrustStore(trust_file(org)).register(self.identity.public_key)
        self.http = httpx.Client(timeout=30)
        self._inbox: dict = {}
        self._lock = threading.Lock()
        self.server, self.url = serve_json(self._http, host)

    # ---- reply-to inbox ---------------------------------------------------
    def _http(self, method, path, body):
        box = self._inbox.get(body.get("request_id"))
        if method != "POST" or box is None:
            return 404, {"error": "unknown request"}
        with self._lock:
            if path == "/hops":
                box["hops"].append(body["hop"])
            elif path == "/branch":     # fan-out (+) or join (-): the number of open branches changes
                box["open_branches"] += body.get("delta", 0)
            elif path == "/result":     # one branch ended; the request is done when none is open
                box["results"].append(body)
                box["open_branches"] -= 1
                if box["open_branches"] <= 0:
                    box["result"] = self._overall(box["results"])
                    box["done"].set()
        return 200, {"ok": True}

    @staticmethod
    def _overall(results: list) -> dict:
        """One result for the request from its branches: the last one, with the statuses of all of them."""
        last = dict(results[-1])
        last["branch_statuses"] = [r["status"] for r in results]
        if len(results) > 1 and any(r["status"] != "completed" for r in results):
            failed = [r for r in results if r["status"] != "completed"]
            last.update(status=failed[-1]["status"], error=failed[-1].get("error", ""))
        return last

    # ---- discovery ----------------------------------------------------------
    def peers(self) -> list:
        with self.mesh._cv:
            return list(self.mesh.peers.values())

    def wait_for_peers(self, n: int, timeout: float = 120) -> list:
        self.mesh.wait_for(lambda: len(self.mesh.peers) >= n and self.mesh.gateways, timeout)
        return self.peers()

    def discovered_edges(self) -> list:
        """(from agent, to agent) for every edge the advertised cards imply: a card that accepts the kind of work
        another card produces. Works for chains, fan-out and join."""
        cards = self.peers()
        by_kind = {c["produces"]: c for c in cards}
        return [(by_kind[k]["name"], c["name"]) for c in cards for k in c.get("accepts", []) if k in by_kind]

    def discovered_chain(self, start: str = "request") -> list:
        """Follow accepts/produces through the advertised cards: which peer would take each step."""
        chain, kind, seen = [], start, set()
        while True:
            found = self.mesh.find(kind)
            if not found or found[0]["name"] in seen:
                return chain
            chain.append((kind, found[0]))
            seen.add(found[0]["name"])
            kind = found[0]["produces"]

    # ---- submitting a request ------------------------------------------------
    def submit(self, user_request: str, budget: Optional[int] = None, max_rounds: int = 4,
               flags: Optional[dict] = None, enforcement: str = "gateway", caveats: Optional[dict] = None) -> dict:
        """Hand a request to the mesh.
        enforcement="gateway": the gateway caps the request at `budget` LLM calls (None = meter only).
        enforcement="token":   `caveats` is the user's signed budget, e.g. {"max_depth": 8, "deadline_s": 300,
                               "max_cost_usd": 0.01, "max_llm_calls": 10}; `deadline_s` is relative to now."""
        rid = uuid.uuid4().hex[:8]
        gateway = self.mesh.gateway()
        if gateway is None:
            raise RuntimeError("no metered-resource gateway on the network (start delegation_mesh/gateway.py)")
        box = {"hops": [], "result": None, "results": [], "open_branches": 1, "done": threading.Event(),
               "request_id": rid, "t0": time.time(),
               "enforcement": enforcement, "budget": budget, "gateway": gateway}
        self._inbox[rid] = box
        env = {"request_id": rid, "user_request": user_request, "reply_to": self.url, "enforcement": enforcement,
               "gateway": gateway, "kind": "request", "round": 1, "hop": 0, "max_rounds": max_rounds,
               "flags": flags or {}, "sender": None, "work": [], "feedback": "", "last_report": ""}
        if enforcement == "token":
            for org in self.mesh.all_gateways():   # an account at every organization the request may reach
                TrustStore(trust_file(org)).register(self.identity.public_key)
            c = dict(caveats or {})
            if "deadline_s" in c:
                c["deadline"] = time.time() + c.pop("deadline_s")
            env["token"] = box["token"] = bt.mint(self.identity, rid, c)
        else:
            self.http.post(f"{gateway}/_requests/{rid}", json={"budget": budget}).raise_for_status()
        peer, why = delegate(self.mesh, self.http, "request", env)
        if peer is None:
            box["result"] = {"status": status_from(why), "error": why, "work": []}
            box["done"].set()
        else:
            box["entry"] = peer
        return box

    def usage(self, box: dict) -> dict:
        """What was charged to the user's credentials per agent slug, summed over every organization's gateway."""
        total = {k: {} for k in ("calls", "denied", "tokens", "tool_calls", "cost_usd")}
        for url in set(self.mesh.all_gateways().values()) | {box["gateway"]}:
            try:
                snap = self.http.get(f"{url}/_requests/{box['request_id']}").json()
            except Exception:
                continue
            for k in total:
                for slug, v in snap.get(k, {}).items():
                    total[k][slug] = total[k].get(slug, 0) + v
        return total

    def usage_by_org(self, box: dict) -> dict:
        """Organization -> what its gateway charged for this request (its own counter only)."""
        out = {}
        for org, url in self.mesh.all_gateways().items():
            try:
                snap = self.http.get(f"{url}/_requests/{box['request_id']}").json()
                out[org] = {k: round(sum(snap.get(k, {}).values()), 6)
                            for k in ("calls", "tokens", "tool_calls", "cost_usd")}
            except Exception:
                pass
        return out

    def ledger(self, box: dict) -> list:
        """Token mode: per organization's gateway, per block of the request's token, its limits and what was
        charged below it at that gateway."""
        rows = []
        for org, url in self.mesh.all_gateways().items():
            try:
                rows += [dict(r, org=org) for r in self.http.get(f"{url}/_ledger/{box['request_id']}").json()]
            except Exception:
                pass
        return rows

    def close(self):
        self.server.shutdown()
        self.mesh.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="?", default="Compare TCP Reno and TCP CUBIC under packet loss. Explain "
                   "their differences and provide a recommendation for a high-bandwidth network.")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--peers", type=int, default=5, help="wait until this many agents are discovered")
    p.add_argument("--enforcement", choices=["gateway", "token"], default="gateway",
                   help="legacy gateway call budget, or the user's signed budget as a capability token")
    for k in ("max-depth", "max-llm-calls", "max-model-tokens", "max-tool-calls"):
        p.add_argument(f"--{k}", type=int)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--deadline-s", type=float, help="end-to-end deadline, seconds from now")
    args = p.parse_args()
    caveats = {k: getattr(args, k) for k in ("max_depth", "max_llm_calls", "max_model_tokens", "max_tool_calls",
                                             "max_cost_usd", "deadline_s") if getattr(args, k) is not None}

    client = MeshClient()
    for c in client.wait_for_peers(args.peers, timeout=60):
        log(f"peer {c['name']:<10} {c['framework']:<18} accepts {c['accepts']} → produces {c['produces']!r}")
    box = client.submit(args.request, budget=args.budget, enforcement=args.enforcement, caveats=caveats)
    shown = 0
    while not box["done"].wait(1.0) or shown < len(box["hops"]):
        for h in box["hops"][shown:]:
            log(f"r{h['round']} hop {h['hop']:>2}: {h['from']:>17} → {h['to']:<17} {h['status']:<17} "
                f"calls={h['llm_calls']} cost=${h['cost_usd']:.6f} {h['seconds']}s"
                + ("" if h["charged_to_user"] else "  (own model, not charged)")
                + ("" if args.enforcement != "token" else
                   f"  [{h['admitted_by']}]" if h.get("admitted_by") else "  [no enforcement point]"))
        shown = len(box["hops"])
    res = box["result"]
    log(f"status: {res['status']} {res.get('error', '')}")
    if args.enforcement == "token":
        log("charged to the user's credentials, per block of the token (hierarchical counters):")
        for r in client.ledger(box):
            log(f"  block {r['block']} by {r['by']:<16} limits {r['limits']}  charged {r['charged']}")
    report = next((w["output"] for w in reversed(res["work"]) if w["kind"] == "report" or w.get("role") == "writer"), "")
    print("\n" + report)
    client.close()

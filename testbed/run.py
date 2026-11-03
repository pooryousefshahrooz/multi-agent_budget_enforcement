"""Send one request into a running testbed and report the work it caused, per agent and in total.

The originator attaches the limit to the request (A2A message and request metadata: there is no standard field). The
organizations run their frameworks as shipped; the instrumentation proxy records what every agent did.

    python testbed/run.py testbed/configs/example.yaml --task "..." --max-model-calls 10 --deadline-s 60
"""
import argparse, glob, json, os, sys, time, uuid
from collections import defaultdict

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.a2a_util import text_of
from common.config import PROXY, Config, a2a_url, upstreams_dir


def send_native(cfg: Config, task: str, limit: dict):
    """entry_via: native - the user calls the entry organization's own run API with the limit in the run's request
    data (LangGraph: /runs/wait, the limit in the run config's metadata and configurable). ADK enters through its own
    A2A server, which keeps the limit; for the frameworks without one, the organization's entry code takes the limit
    from the request (run_org.py)."""
    org, agent = cfg.entry
    known = {}
    for f in glob.glob(os.path.join(upstreams_dir(cfg), "*.json")):
        known.update(json.load(open(f)))
    aid = known[f"{org}.{agent}"].split("/a2a/")[1]
    body = {"assistant_id": aid, "input": {"messages": [{"role": "user", "content": task}]},
            "config": {"metadata": {"limit": limit}, "configurable": {"limit": limit}}}
    r = httpx.post(f"{PROXY}/native/user/{org}/{agent}/runs/wait", json=body, timeout=3600)
    out = r.json()
    msgs = out.get("messages") or []
    answer = msgs[-1].get("content", "") if msgs else json.dumps(out)[:300]
    return ("TASK_STATE_COMPLETED" if r.status_code == 200 and msgs else f"HTTP {r.status_code}"), answer


def send(cfg: Config, task: str, limit: dict) -> dict:
    rid = uuid.uuid4().hex[:8]
    httpx.post(f"{PROXY}/_request", json={"id": rid})
    org, agent = cfg.entry
    t0 = time.time()
    if cfg.raw.get("entry_via") == "native" and cfg.framework(org) == "langgraph":
        state, answer = send_native(cfg, task, limit)
        result = {"status": {"state": state}, "artifacts": [{"parts": [{"text": str(answer)}]}]}
    else:
        body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {
            "message": {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": task}],
                        "metadata": {"limit": limit}},
            "metadata": {"limit": limit}}}
        r = httpx.post(a2a_url("user", org, agent) + "/", json=body, headers={"A2A-Version": "1.0"}, timeout=3600)
        result = r.json().get("result", {}).get("task", {})
    seconds = time.time() - t0
    # some calls are recorded after the answer returns (a streaming client that hangs up early): wait until the
    # request's records stop changing
    events, last = [], -1
    while len(events) != last:
        last = len(events)
        time.sleep(1.5)
        events = httpx.get(f"{PROXY}/_events/{rid}").json()
    return {"request": rid, "t0": t0, "seconds": round(seconds, 2), "state": result.get("status", {}).get("state"),
            "answer": text_of(result), "events": events}


def work(events: list) -> dict:
    """Per agent: model calls, tokens, dollar cost, tool calls, active wall-clock; and the totals."""
    per = defaultdict(lambda: {"model_calls": 0, "tokens": 0, "cost_usd": 0.0, "tool_calls": 0, "first": None, "last": None})
    for e in events:
        if e["type"] not in ("model_call", "tool_call"):
            continue
        a = per[f"{e['org']}.{e['agent']}"]
        if e["type"] == "model_call":
            a["model_calls"] += 1
            a["tokens"] += e["prompt_tokens"] + e["completion_tokens"]
        else:
            a["tool_calls"] += 1
        a["cost_usd"] += e.get("cost_usd", 0)
        a["first"] = min(a["first"] or e["t"], e["t"] - e.get("seconds", 0))
        a["last"] = max(a["last"] or e["t"], e["t"])
    rows = {k: {**{x: v[x] for x in ("model_calls", "tokens", "tool_calls")}, "cost_usd": round(v["cost_usd"], 6),
                "seconds": round(v["last"] - v["first"], 2)} for k, v in sorted(per.items())}
    total = {x: sum(r[x] for r in rows.values()) for x in ("model_calls", "tokens", "tool_calls", "cost_usd")}
    total["cost_usd"] = round(total["cost_usd"], 6)
    return {"per_agent": rows, "total": total}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--task", default="Compare TCP Reno and TCP CUBIC under packet loss.")
    p.add_argument("--max-depth", type=int, help="override the config's request_limit")
    p.add_argument("--deadline-s", type=float, help="override the config's request_limit")
    p.add_argument("--max-cost-usd", type=float, help="override the config's request_limit")
    a = p.parse_args()
    cfg = Config(a.config)
    limit = cfg.request_limit(max_depth=a.max_depth, deadline_s=a.deadline_s, max_cost_usd=a.max_cost_usd)
    res = send(cfg, a.task, limit)
    w = work(res["events"])
    print(f"request {res['request']}: {res['state']} in {res['seconds']}s")
    if res["state"] != "TASK_STATE_COMPLETED":
        print(f"  answer: {res['answer'][:300]}")
    for agent, row in w["per_agent"].items():
        print(f"  {agent:<24} {row}")
    print(f"  {'TOTAL':<24} {w['total']}")
    a2a = [e for e in res["events"] if e["type"] == "a2a_message"]
    for e in a2a:
        print(f"  A2A {e['sender']:>16} -> {e['org']}.{e['agent']:<12} limit in message: {e['limit_in_message'] is not None}"
              f", in request: {e['limit_in_request'] is not None}")
    print(f"  cost {w['total']['cost_usd']} vs the user's budget {limit['max_cost_usd']}")
    out = os.path.join(HERE, "runs", cfg.name, f"request_{res['request']}.json")
    json.dump({**res, "work": w, "limit": limit}, open(out, "w"), indent=1)

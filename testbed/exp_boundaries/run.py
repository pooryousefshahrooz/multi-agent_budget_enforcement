"""Run the task once per limit level and report, for each run: every boundary crossing, the fate of each limit kind
at each boundary type, LangGraph's own counters, and the work done compared with each limit.

The user attaches the limit to the first A2A message (message and request metadata). Fates:
  preserved  the hop / run carries the user's value (attenuated: a tighter one)
  reset      the framework runs with its own value instead (LangGraph: recursion_limit)
  dropped    the limit is absent
  duplicated a fan-out hands the full additive limit to every branch

    experiments/exp1_propagation/envs/tasks/bin/python testbed/exp_boundaries/run.py [high middle low]
"""
import ast, json, os, re, sys, time, uuid
from collections import Counter, defaultdict

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
TESTBED = os.path.dirname(HERE)
sys.path.insert(0, TESTBED)
from common.a2a_util import text_of
from common.config import PROXY, Config, a2a_url

CFG = Config(os.path.join(HERE, "config.yaml"))
RUN = os.path.join(TESTBED, "runs", CFG.name)
KINDS = ["max_depth", "deadline", "max_model_calls", "max_tool_calls", "max_tokens", "max_cost_usd"]
ADDITIVE = {"max_model_calls", "max_tool_calls", "max_tokens", "max_cost_usd"}


def task():
    from datasets import load_dataset
    row = next(r for r in load_dataset(CFG.raw["task"]["dataset"])["train"] if r["TASK"] == CFG.raw["task"]["id"])
    return row["PROMPT"], ast.literal_eval(row["GTFA_CLAIMS"])


def send(prompt, limit):
    rid = uuid.uuid4().hex[:8]
    httpx.post(f"{PROXY}/_request", json={"id": rid})
    msg = {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": f"ROUND: 1\nTASK: {prompt}"}],
           "metadata": {"limit": limit}}
    body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": msg, "metadata": {"limit": limit}}}
    t0 = time.time()
    r = httpx.post(a2a_url("user", "orgA", "A") + "/", json=body, headers={"A2A-Version": "1.0"}, timeout=7200)
    t1 = time.time()
    result = r.json().get("result", {}).get("task", {})
    return {"request": rid, "t0": t0, "seconds": round(t1 - t0, 1), "state": result.get("status", {}).get("state"),
            "answer": text_of(result), "events": httpx.get(f"{PROXY}/_events/{rid}").json()}


def fate(value, origin, kind, siblings_full=False):
    if value is None:
        return "dropped"
    if kind in ADDITIVE and siblings_full:
        return "duplicated"
    if value == origin:
        return "preserved"
    return "preserved (attenuated)" if value < origin else "reset (loosened)"


def analyse(res, limit, level):
    ev = res["events"]
    hops = sorted([e for e in ev if e["type"] == "a2a_message"], key=lambda e: e["t_start"])
    # depth: a hop's parent is the latest hop into its sender that was still open when it started
    for h in hops:
        h["end"] = h["t_start"] + h["seconds"]
        h["round"] = int((re.search(r"ROUND:\s*(\d+)", h.get("text_head") or "") or [0, 1])[1])
        parents = [p for p in hops if "depth" in p and f"{p['org']}.{p['agent']}" == h["sender"]
                   and p["t_start"] <= h["t_start"] <= p["end"]]
        h["depth"] = max(parents, key=lambda p: p["t_start"])["depth"] + 1 if parents else 1
    for h in hops:
        h["kind"] = f"{h['sender'].split('.')[-1]}->{h['agent']}"
    first = hops[0] if hops else None

    # boundary crossings
    crossings = Counter(h["kind"] for h in hops)
    tools = [e for e in ev if e["type"] == "tool_call"]
    models = [e for e in ev if e["type"] == "model_call"]
    for e in tools:
        crossings[f"{e['agent']}->MCP:{e['server']}"] += 1
    for e in models:
        crossings[f"{e['agent']}->model"] += 1

    # fates
    matrix = {}
    # (1) A2A, user -> A: the message itself
    carried = (first or {}).get("limit_in_message") or (first or {}).get("limit_in_request") or {}
    matrix["A2A message (user->A)"] = {k: fate(carried.get(k), limit[k], k) for k in KINDS}
    # (2) A2A into the framework: what the LangGraph run received (from the agents' own reports)
    fw = [e for e in ev if e["type"] == "framework"]
    in_config = set().union(*[set(e.get("limit_kinds_in_config") or []) for e in fw]) if fw else set()
    steps_left = sorted({e["steps_left"] for e in fw if e.get("node") in ("plan", "work", "verify") and e.get("steps_left")})
    recv = {}
    for k in KINDS:
        if k in in_config:
            recv[k] = "preserved"
        elif k == "max_model_calls":
            recv[k] = f"reset (LangGraph recursion_limit: {max(steps_left) + 1 if steps_left else '?'} steps)"
        else:
            recv[k] = "dropped (no LangGraph counter)"
    matrix["A2A into LangGraph run (Agent Server)"] = recv
    # (3) A2A agent -> agent
    later = hops[1:]
    by_round_sender = defaultdict(list)
    for h in later:
        by_round_sender[(h["round"], h["sender"])].append(h)
    row = {}
    for k in KINDS:
        fates = []
        for h in later:
            v = (h.get("limit_in_message") or h.get("limit_in_request") or {}).get(k)
            sibs = by_round_sender[(h["round"], h["sender"])]
            full = len(sibs) > 1 and all(((s.get("limit_in_message") or {}).get(k)) == limit[k] for s in sibs)
            fates.append(fate(v, limit[k], k, full))
        row[k] = "/".join(sorted(set(fates))) if fates else "not crossed"
    matrix["A2A agent->agent"] = row
    # (4) MCP tool call
    matrix["MCP tool call"] = ({k: "/".join(sorted({fate((e.get("limit_in_meta") or {}).get(k), limit[k], k) for e in tools}))
                                for k in KINDS} if tools else {k: "not crossed" for k in KINDS})
    matrix["MCP sampling"] = {k: "not exercised" for k in KINDS}

    # work vs limit
    per = defaultdict(lambda: Counter())
    for e in models:
        per[e["agent"]]["model_calls"] += 1
        per[e["agent"]]["tokens"] += e["prompt_tokens"] + e["completion_tokens"]
        per[e["agent"]]["cost_usd"] += e["cost_usd"]
    for e in tools:
        per[e["agent"]]["tool_calls"] += 1
        per[e["agent"]]["cost_usd"] += e.get("cost_usd", 0)
    total = Counter()
    for c in per.values():
        total.update(c)
    used = {"max_depth": max((h["depth"] for h in hops), default=0), "deadline": res["seconds"],
            "max_model_calls": total["model_calls"], "max_tool_calls": total["tool_calls"],
            "max_tokens": total["tokens"], "max_cost_usd": round(total["cost_usd"], 6)}
    bound = {**limit, "deadline": limit["deadline_s"]}
    vs = {k: {"limit": bound[k], "used": used[k], "times_limit": round(used[k] / bound[k], 2) if bound[k] else None,
              "exceeded": used[k] > bound[k]} for k in KINDS}
    return {"level": level, "state": res["state"], "seconds": res["seconds"], "answer": res["answer"],
            "rounds": max((h["round"] for h in hops), default=0), "crossings": dict(crossings), "fates": matrix,
            "work_per_agent": {a: dict(c) for a, c in sorted(per.items())}, "work_vs_limit": vs,
            "framework_counters": fw, "hops": [{k: h[k] for k in ("kind", "round", "depth", "context", "limit_in_message",
                                                                  "limit_in_request", "seconds")} for h in hops]}


def show(a):
    print(f"\n===== {a['level']}: {a['state']} in {a['seconds']} s, {a['rounds']} round(s)")
    print("answer:", a["answer"][:300].replace("\n", " | "))
    print("crossings:", ", ".join(f"{k} x{v}" for k, v in sorted(a["crossings"].items())))
    w = max(len(b) for b in a["fates"])
    print(f"{'boundary':<{w}}  " + "  ".join(f"{k:<16}" for k in KINDS))
    for b, row in a["fates"].items():
        print(f"{b:<{w}}  " + "  ".join(f"{row[k][:16]:<16}" for k in KINDS))
    print("work vs limit:", "; ".join(f"{k}: {v['used']}/{v['limit']}{' EXCEEDED' if v['exceeded'] else ''}"
                                      for k, v in a["work_vs_limit"].items()))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("levels", nargs="*", default=["high", "middle", "low"])
    p.add_argument("--repeat", type=int, default=1, help="runs per level")
    a = p.parse_args()
    model = CFG.raw["model"].get("model", "fake")
    prompt, claims = task()
    print("task:", prompt, "\ngold claims:", claims, "\nmodel:", model)
    out = {"task": prompt, "claims": claims, "model": model, "runs": []}
    fp = os.path.join(RUN, f"results_{model.replace(':', '-')}_{int(time.time())}.json")
    for i in range(a.repeat):
        for level in a.levels:
            spec = CFG.raw["limits"][level]
            limit = {k: spec[k] for k in KINDS if k in spec}
            limit["deadline_s"] = spec["deadline_s"]
            limit["deadline"] = time.time() + spec["deadline_s"]
            res = send(prompt, {k: v for k, v in limit.items() if k != "deadline_s"})
            out["runs"].append({**analyse(res, limit, level), "repeat": i + 1})
            show(out["runs"][-1])
            json.dump(out, open(fp, "w"), indent=1, default=str)   # saved after every run

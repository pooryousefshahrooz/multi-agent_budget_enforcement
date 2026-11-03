"""Per-agent limits fail: send one GAIA task with the user's limit, then show, per agent, its own per-agent limit
(a fresh run under its own setting) against the work it did, the work of the whole request against the user's limit,
and how each forwarded request restates the one received.

    experiments/exp1_propagation/envs/tasks/bin/python testbed/exp_per_agent.py testbed/configs/per_agent_limits.yaml \
        --task e1fc63a2-da7a-432f-be78-7c4a95598703 --user-model-calls 5 [--repeat 3]
"""
import argparse, difflib, json, os, sys, time
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.agent_logic import unpack
from common.config import Config
from run import send

GAIA = os.path.join(HERE, "tasks", "data", "gaia", "2023", "validation", "metadata.parquet")


def body(text):
    """What a message asks for: the task text, without the testbed's header line (inside an organization)."""
    h = unpack(text or "")
    if h is None:
        return text or ""
    return h.get("task", "") if h["kind"] == "work" else h.get("result", "")


def analyse(cfg, res, user_limit):
    ev = res["events"]
    calls = Counter(f"{e['org']}.{e['agent']}" for e in ev if e["type"] == "model_call")
    tools = Counter(f"{e['org']}.{e['agent']}" for e in ev if e["type"] == "tool_call")
    hops = sorted([e for e in ev if e["type"] == "a2a_message"], key=lambda e: e["t_start"])
    agents = []
    for org in cfg.orgs:
        for name in cfg.orgs[org]["agents"]:
            label, own = f"{org}.{name}", cfg.orgs[org].get("agent_limit")
            runs = sum(1 for h in hops if f"{h['org']}.{h['agent']}" == label and (unpack(h.get("text", "")) or {}).get("kind") != "report")
            agents.append({"agent": label, "framework": cfg.framework(org), "own_limit": own, "runs": runs,
                           "model_calls": calls[label], "tool_calls": tools[label],
                           "within_own_limit": calls[label] <= (own or 1e9) * max(runs, 1)})
    # restating: each delegation (across organizations) compared with the request its sender received
    restated = []
    received = {}
    for h in hops:
        to = f"{h['org']}.{h['agent']}"
        if unpack(h.get("text", "")) is None:          # a request from outside the receiver's organization
            received.setdefault(to, body(h.get("text")))
            if h["sender"] != "user":
                before = received.get(h["sender"]) or next((body(x.get("text")) for x in hops if f"{x['org']}.{x['agent']}" == h["sender"]), "")
                restated.append({"from": h["sender"], "to": to, "received": before, "forwarded": body(h.get("text")),
                                 "similarity": round(difflib.SequenceMatcher(None, before, body(h.get("text"))).ratio(), 2),
                                 "limit_in_message": h.get("limit_in_message") is not None})
    total = sum(calls.values())
    return {"state": res["state"], "seconds": res["seconds"], "answer": res["answer"], "agents": agents,
            "total_model_calls": total, "total_tool_calls": sum(tools.values()), "user_limit": user_limit,
            "user_limit_exceeded": total > user_limit, "restated": restated,
            "hops": [f"{h['sender']} -> {h['org']}.{h['agent']}" for h in hops]}


def show(a, gold):
    print(f"\n{a['state']} in {a['seconds']} s   answer: {a['answer'][:200]!r}   (gold: {gold})")
    print(f"{'agent':<10} {'framework':<14} {'own limit':>9} {'runs':>5} {'model calls':>12} {'tool calls':>11}  within own limit")
    for r in a["agents"]:
        print(f"{r['agent']:<10} {r['framework']:<14} {r['own_limit']:>9} {r['runs']:>5} {r['model_calls']:>12} {r['tool_calls']:>11}  {r['within_own_limit']}")
    print(f"WHOLE REQUEST: {a['total_model_calls']} model calls, {a['total_tool_calls']} tool calls; "
          f"user's limit {a['user_limit']} model calls -> {'EXCEEDED' if a['user_limit_exceeded'] else 'within'}")
    for r in a["restated"]:
        print(f"\n  {r['from']} -> {r['to']} (similarity to what {r['from']} received: {r['similarity']}; "
              f"limit in message: {r['limit_in_message']})")
        print(f"    received : {r['received'][:300]!r}")
        print(f"    forwarded: {r['forwarded'][:300]!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--task", required=True)
    p.add_argument("--user-model-calls", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    a = p.parse_args()
    cfg = Config(a.config)
    row = pd.read_parquet(GAIA).set_index("task_id").loc[a.task]
    out = []
    for i in range(a.repeat):
        limit = {"max_model_calls": a.user_model_calls, "deadline": time.time() + 1800}
        res = send(cfg, row["Question"], limit)
        out.append(analyse(cfg, res, a.user_model_calls))
        show(out[-1], row["Final answer"])
    fp = os.path.join(HERE, "runs", cfg.name, f"per_agent_{int(time.time())}.json")
    json.dump({"task": a.task, "question": row["Question"], "gold": row["Final answer"],
               "model": cfg.raw["model"].get("model"), "runs": out}, open(fp, "w"), indent=1)
    print("\nsaved", fp)

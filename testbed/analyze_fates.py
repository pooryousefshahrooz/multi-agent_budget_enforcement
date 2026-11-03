"""The fate of the user's request limit at every hop of one request, from the testbed's records (the proxy's events and
the agents' observer reports). Reads runs/<config>/request_<id>.json written by run.py.

  A2A hop    left the sender (the limit on the message, proxy) and reached the receiving run (the receiver's observer)
               preserved          on the message and in the receiving run
               dropped (receiver) on the message, not in the receiving run
               dropped (sender)   not on the message
             plus the receiving run's own counter (the framework's default it runs under)
  MCP call   the limit in the tools/call request (proxy)

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_fates.py testbed/configs/<config>.yaml [request id]
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config


def load(cfg, rid=None):
    run_dir = os.path.join(HERE, "runs", cfg.name)
    files = sorted(glob.glob(os.path.join(run_dir, f"request_{rid or '*'}.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no request records in {run_dir}")
    return json.load(open(files[-1]))


def outcome(ev, mine, hop):
    """How the receiving run ended (its observer's last report: finished, or the error that stopped it), with the
    model and tool calls it made during the hop."""
    ends = [r for r in mine if r.get("finished") or r.get("error")]
    if not ends:
        return None
    end = max(ends, key=lambda r: r["t"])
    t1 = hop["t_start"] + hop.get("seconds", 0)
    own = lambda e: e["org"] == hop["org"] and e["agent"] == hop["agent"] and hop["t_start"] <= e["t"] <= t1 + 1
    return {"ended": "finished" if end.get("finished") else end["error"], "allowed_calls": end.get("allowed_calls"),
            "binding": end.get("binding"), "model_calls": sum(1 for e in ev if e["type"] == "model_call" and own(e)),
            "tool_calls": sum(1 for e in ev if e["type"] == "tool_call" and own(e)), "seconds": hop.get("seconds")}


def fates(cfg, res):
    ev = res["events"]
    reports = [e for e in ev if e["type"] == "framework"]
    hops = sorted([e for e in ev if e["type"] == "a2a_message"], key=lambda e: e["t_start"])
    out = []
    for h in hops:
        to = f"{h['org']}.{h['agent']}"
        sent = bool(h.get("marker_in_request"))
        mine = [r for r in reports if f"{r['org']}.{r['agent']}" == to and r["t"] >= h["t_start"]]
        first = min(mine, key=lambda r: r["t"]) if mine else None
        got = bool(first and first.get("limit"))
        fate = "preserved" if sent and got else ("dropped (receiver)" if sent else "dropped (sender)")
        kind = ("origin" if h["sender"] == "user" else
                "across organizations" if h["sender"].split(".")[0] != h["org"] else "inside the organization")
        out.append({"hop": f"{h['sender']} -> {to}", "kind": kind,
                    "sender_framework": cfg.framework(h["sender"].split(".")[0]) if h["sender"] != "user" else "user",
                    "receiver_framework": cfg.framework(h["org"]), "on_message": sent, "in_receiving_run": got,
                    "fate": fate, "receiver_counter": first.get("counter") if first else None,
                    "receiver_outcome": outcome(ev, mine, h)})
    for org in cfg.orgs:   # inside a native organization: the framework's own delegation between its agents
        if cfg.orgs[org].get("delegation", "a2a") != "native":
            continue
        for a, b in cfg.edges(org):
            ra = [r for r in reports if (r["org"], r["agent"]) == (org, a)]
            rb = [r for r in reports if (r["org"], r["agent"]) == (org, b)]
            if not ra or not rb:
                out.append({"hop": f"{org}.{a} -> {org}.{b}", "kind": "within framework", "fate": "not crossed"})
                continue
            last_a, first_b = max(ra, key=lambda r: r["t"]), min(rb, key=lambda r: r["t"])
            ca, cb = last_a.get("counter") or {}, first_b.get("counter") or {}
            if ca.get("counter_id") is not None and cb.get("counter_id") is not None:
                same = ca["counter_id"] == cb["counter_id"]
            else:   # LangGraph: the child continues the parent's step count, or starts its own
                same = (cb.get("langgraph_step") or 0) > (ca.get("langgraph_step") or 0)
            got = bool(first_b.get("limit"))
            fate = ("preserved" if same else "duplicated") if got else "dropped"
            out.append({"hop": f"{org}.{a} -> {org}.{b}", "kind": "within framework",
                        "mechanism": cfg.orgs[org].get("mechanism"), "in_receiving_run": got, "same_run": same,
                        "fate": fate, "receiver_counter": cb})
    for e in ev:
        if e["type"] == "native_request":
            out.append({"hop": f"{e['sender']} -> {e['org']}.{e['agent']} ({e['path']})", "kind": "origin (native API)",
                        "on_message": bool(e.get("marker_in_request")), "fate": "sent" if e.get("marker_in_request") else "not sent"})
    models = [e for e in ev if e["type"] == "model_call"]
    deadline_s = (res.get("limit") or {}).get("deadline", 0) - res.get("t0", 0) if res.get("t0") else None
    for e in ev:
        if e["type"] != "tool_call":
            continue
        row = {"hop": f"{e['org']}.{e['agent']} -> MCP {e['server']}.{e['tool']}", "kind": "MCP tool call",
               "on_message": bool(e.get("marker_in_request")),
               "fate": "preserved" if e.get("marker_in_request") else "dropped (sender)"}
        if e.get("t_start") is not None and (e.get("seconds") or 0) > 2:   # only calls long enough to be cut
            # when did the agent's client stop waiting? its next model call after the tool call started
            nxt = [m["t"] - m.get("seconds", 0) for m in models if (m["org"], m["agent"]) == (e["org"], e["agent"])
                   and m["t"] - m.get("seconds", 0) > e["t_start"]]
            waited = round(min(nxt) - e["t_start"], 1) if nxt else None
            row.update(tool_seconds=e.get("seconds"), client_waited_s=waited)
            if waited is not None and waited < e["seconds"] - 2:
                near = deadline_s is not None and abs(waited - deadline_s) < 3
                row["deadline_fate"] = "preserved" if near else "reset"
                row["deadline_evidence"] = (f"client gave up after {waited} s of a {e['seconds']} s call"
                                            + (" (the user's deadline)" if near else " (its own timeout)"))
            else:
                row["deadline_fate"] = "dropped"
                row["deadline_evidence"] = f"client waited the whole {e.get('seconds')} s call"
        out.append(row)
    for m in models:   # model calls a tool triggered through MCP sampling
        if m["agent"].endswith(".sampling"):
            out.append({"hop": f"{m['org']}.{m['agent'][:-9]} -> sampled model call", "kind": "MCP sampling",
                        "on_message": bool(m.get("marker_in_request")),
                        "fate": "preserved" if m.get("marker_in_request") else "dropped"})
    return out


if __name__ == "__main__":
    cfg = Config(sys.argv[1])
    res = load(cfg, sys.argv[2] if len(sys.argv) > 2 else None)
    rows = fates(cfg, res)
    print(f"request {res['request']}  limit {res.get('limit')}")
    for r in rows:
        counter = f"  receiver counter {r['receiver_counter']}" if r.get("receiver_counter") else ""
        print(f"  {r['kind']:<24} {r['hop']:<34} {r['fate']:<20}{counter}")
        if r.get("receiver_outcome"):
            print(f"  {'':<24} receiving run: {r['receiver_outcome']}")
    json.dump(rows, open(os.path.join(HERE, "runs", cfg.name, f"fates_{res['request']}.json"), "w"), indent=1)

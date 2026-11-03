"""Run a ReDel dataset through a running testbed: one request per question, with the limit attached, as in run.py.

Writes runs/<config>/<dataset>/results.jsonl in ReDel's format (score.py scores it), plus each request's measured
work (per agent and total) and the limit's fate on every A2A hop. Like ReDel, it resumes: questions already in
results.jsonl are skipped.

    experiments/exp1_propagation/envs/tasks/bin/python testbed/bench.py testbed/configs/fanoutqa.yaml \
        --dataset fanoutqa --n 5 --max-model-calls 50 --deadline-s 900
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config
from run import send, work
from tasks.datasets import DATASETS


def hops(events):
    return [{"from": e["sender"], "to": f"{e['org']}.{e['agent']}", "limit_in_message": e["limit_in_message"],
             "limit_in_request": e["limit_in_request"]} for e in events if e["type"] == "a2a_message"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--n", type=int, default=None, help="first N questions (default: all)")
    p.add_argument("--level", help="one of the config's limit_levels (default: the config's request_limit)")
    a = p.parse_args()
    cfg = Config(a.config)
    ds = DATASETS[a.dataset](cfg)
    level = (cfg.raw.get("limit_levels") or {}).get(a.level, {}) if a.level else {}
    if a.level and not level:
        raise SystemExit(f"no limit level {a.level!r} in {a.config}")
    out_dir = os.path.join(HERE, "runs", cfg.name, f"{ds.name}_{a.level}" if a.level else ds.name)
    os.makedirs(out_dir, exist_ok=True)
    results_fp, work_fp = os.path.join(out_dir, "results.jsonl"), os.path.join(out_dir, "work.jsonl")
    done = set()
    if os.path.exists(results_fp):
        done = {json.loads(l)["id"] for l in open(results_fp) if l.strip()}
    qs = ds.questions()[: a.n]
    for i, q in enumerate(qs):
        if q["id"] in done:
            continue
        ds.before(cfg, q)
        limit = cfg.request_limit(**level)
        try:
            res = send(cfg, ds.prompt(q), limit)
        except Exception as e:   # the request broke (a connection dropped): record it as such and go on
            print(f"[{i + 1}/{len(qs)}] {q['id'][:8]} request failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
            res = {"request": None, "state": f"REQUEST_FAILED {type(e).__name__}", "seconds": None, "answer": "",
                   "events": []}
        # each agent's run: did its framework stop it (the error), or did it finish; the calls the limit allowed it
        runs = {}
        for e in res["events"]:
            if e["type"] == "framework" and ("error" in e or e.get("finished")):
                runs[f"{e['org']}.{e['agent']}"] = {"error": e.get("error"), "finished": bool(e.get("finished")),
                                                    "allowed_calls": e.get("allowed_calls"), "binding": e.get("binding")}
        w = work(res["events"])
        with open(results_fp, "a") as f:
            f.write(json.dumps(ds.record(cfg, q, res["answer"])) + "\n")
        with open(work_fp, "a") as f:
            f.write(json.dumps({"id": q["id"], "request": res["request"], "state": res["state"],
                                "seconds": res["seconds"], "limit": limit, "work": w, "hops": hops(res["events"]),
                                "runs": runs}) + "\n")
        print(f"[{i + 1}/{len(qs)}] {q['id'][:8]} {res['state']} {res['seconds']}s  calls {w['total']['model_calls']}  "
              + "  ".join(f"{k}: {v['error'] or 'finished'} (allowed {v['allowed_calls']})" for k, v in runs.items()), flush=True)

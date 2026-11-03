"""For the native-limit sweep (experiment.py configs/native_limits --dataset gaia): each framework's own limit setting
on a 3-agent organization that delegates natively, mapped to the work a GAIA task actually got under it: LLM calls,
MCP tool calls, tokens, seconds, cost, and how the run ended (finished, stopped by the framework's limit, other error).
Writes a table (runs/native_limits/native.json, .csv) and a figure (figures/native_limits.png).

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_native.py testbed/configs/native_limits
"""
import csv, glob, json, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config

NAMES = {"langgraph": "LangGraph", "adk": "Google ADK", "crewai": "CrewAI", "openai_agents": "OpenAI Agents SDK",
         "autogen": "AutoGen"}
SETTING = {"langgraph": "recursion_limit (supersteps)", "adk": "max_llm_calls", "crewai": "max_iter (each agent)",
           "openai_agents": "max_turns (each agent)", "autogen": "team max_turns"}
STOP = re.compile(r"GraphRecursionError|need more steps|Max number of llm calls|LlmCallsLimitExceeded|MaxTurnsExceeded|"
                  r"CrewAI max_iter|AutoGen team max_turns")


def outcome(w, answer):
    errors = [r["error"] for r in w["runs"].values() if r.get("error")]
    text = " ".join(errors) + " " + str(answer or "")
    if STOP.search(text):
        return "stopped"
    if errors or w["state"] != "TASK_STATE_COMPLETED":
        return "error"
    return "finished"


def collect(directory):
    rows, tasks = [], []
    for config in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        cfg = Config(config)
        org = next(iter(cfg.orgs))
        fw, value = cfg.framework(org), int(cfg.orgs[org]["native_limit"])
        d = os.path.join(HERE, "runs", cfg.name, "gaia")
        if not os.path.exists(os.path.join(d, "work.jsonl")):
            continue
        results = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(d, "results.jsonl")) if l.strip())}
        per = []
        for l in open(os.path.join(d, "work.jsonl")):
            if not l.strip():
                continue
            w = json.loads(l)
            if str(w["state"]).startswith("REQUEST_FAILED"):   # the request broke (connection dropped): no measurement
                continue
            r = results.get(w["id"], {})
            t = w["work"]["total"]
            per.append({"framework": NAMES[fw], "setting": SETTING[fw], "value": value, "task": w["id"][:8],
                        "llm_calls": t["model_calls"], "tool_calls": t["tool_calls"], "tokens": t["tokens"],
                        "seconds": w["seconds"], "cost_usd": t["cost_usd"], "outcome": outcome(w, r.get("given")),
                        "correct": bool(r.get("correct")),
                        "agents_run": sum(1 for v in w["work"]["per_agent"].values() if v.get("model_calls"))})
        tasks += per
        if not per:
            continue
        agg = lambda k: [p[k] for p in per]
        rows.append({"framework": NAMES[fw], "setting": SETTING[fw], "value": value, "tasks": len(per),
                     **{f"{k}_mean": round(st.mean(agg(k)), 4) for k in ("llm_calls", "tool_calls", "tokens", "seconds", "cost_usd")},
                     **{f"{k}_min": min(agg(k)) for k in ("llm_calls", "tool_calls")},
                     **{f"{k}_max": max(agg(k)) for k in ("llm_calls", "tool_calls")},
                     "stopped": sum(p["outcome"] == "stopped" for p in per),
                     "finished": sum(p["outcome"] == "finished" for p in per),
                     "error": sum(p["outcome"] == "error" for p in per),
                     "correct": sum(p["correct"] for p in per),
                     "agents_run_mean": round(st.mean(agg("agents_run")), 2)})
    return rows, tasks


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fws = [n for n in NAMES.values() if any(r["framework"] == n for r in rows)]
    METRICS = [("llm_calls", "LLM calls per task"), ("tool_calls", "MCP tool calls per task"),
               ("seconds", "seconds per task"), ("stopped", "tasks stopped by the limit (of 10)")]
    fig, axes = plt.subplots(len(METRICS), len(fws), figsize=(4.2 * len(fws), 3.0 * len(METRICS) + 1.4), dpi=170,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, fw in enumerate(fws):
        rs = sorted([r for r in rows if r["framework"] == fw], key=lambda r: r["value"])
        xs = list(range(len(rs)))
        for i, (key, label) in enumerate(METRICS):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            if key == "stopped":
                ax.bar(xs, [r["stopped"] for r in rs], color=ORANGE, width=0.55, zorder=3)
                ax.bar(xs, [r["finished"] for r in rs], bottom=[r["stopped"] for r in rs], color=BLUE, width=0.55,
                       zorder=3, alpha=0.35)
                ax.set_ylim(0, max(r["tasks"] for r in rs) + 0.5)
            else:
                mean = [r[f"{key}_mean"] for r in rs]
                ax.plot(xs, mean, color=BLUE, lw=2, marker="o", ms=6, zorder=3)
                if f"{key}_min" in rs[0]:
                    ax.fill_between(xs, [r[f"{key}_min"] for r in rs], [r[f"{key}_max"] for r in rs], color=BLUE,
                                    alpha=0.12, lw=0, zorder=2)
                for x, m in zip(xs, mean):
                    ax.text(x, m, f" {m:.1f}", fontsize=7.5, color=INK, va="bottom", ha="left")
                ax.set_ylim(bottom=0)
            ax.set_xticks(xs, [str(r["value"]) for r in rs], fontsize=8.5, color=INK)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
            ax.tick_params(axis="x", length=0)
            if j == 0:
                ax.set_ylabel(label, fontsize=9.5, color=MUTED)
            if i == 0:
                ax.set_title(f"{fw}\n{rs[0]['setting']}", fontsize=11, color=INK, fontweight="bold")
            if i == len(METRICS) - 1:
                ax.set_xlabel(f"{rs[0]['setting']} =", fontsize=8.5, color=MUTED)
    fig.suptitle("Each framework's own limit setting vs the work a task gets under it (3 agents, native delegation, "
                 "10 GAIA tasks, llama3.2:3b)", x=0.01, y=0.995, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.965, "line: mean per task; band: min to max over the 10 tasks; bottom row: orange = stopped by the "
             "framework's limit, light blue = finished", fontsize=9.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    directory = sys.argv[1]
    rows, tasks = collect(directory)
    print(f"{'framework':<20}{'setting':<30}{'value':>6}{'tasks':>6}{'LLM calls':>11}{'tool calls':>11}{'tokens':>9}"
          f"{'seconds':>9}{'cost $':>10}{'stopped':>8}{'finished':>9}{'error':>6}{'correct':>8}{'agents':>7}")
    for r in rows:
        print(f"{r['framework']:<20}{r['setting']:<30}{r['value']:>6}{r['tasks']:>6}{r['llm_calls_mean']:>11.1f}"
              f"{r['tool_calls_mean']:>11.1f}{r['tokens_mean']:>9.0f}{r['seconds_mean']:>9.1f}{r['cost_usd_mean']:>10.5f}"
              f"{r['stopped']:>8}{r['finished']:>9}{r['error']:>6}{r['correct']:>8}{r['agents_run_mean']:>7}")
    name = os.path.basename(os.path.normpath(directory))
    d = os.path.join(HERE, "runs", name)
    os.makedirs(d, exist_ok=True)
    json.dump({"summary": rows, "tasks": tasks}, open(os.path.join(d, "native.json"), "w"), indent=1)
    with open(os.path.join(d, "native.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = os.path.join(os.path.dirname(HERE), "figures", f"{name}.png")
    plot(rows, out)
    print(out)

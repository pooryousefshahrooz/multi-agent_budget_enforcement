"""Per-agent limits do not compose (experiment.py configs/composition --dataset dynamicmcpbench): tasks with k
subtasks on a chain of k organizations, each on its framework's default limit. Per k: the share of tasks that finished
without any framework's limit stopping a hop, which framework stopped the others, the delegation depth reached, and the
LLM calls the request used against what the defaults along the chain allowed. Writes a table
(runs/composition/composition.json) and a figure (figures/composition.png).

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_composition.py testbed/configs/composition
"""
import glob, json, os, statistics as st, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config

NAMES = {"langgraph": "LangGraph", "adk": "Google ADK", "crewai": "CrewAI", "openai_agents": "OpenAI Agents SDK",
         "autogen": "AutoGen"}
# each framework's default, in LLM calls a run may make (the testbed's adapters set nothing, so the default applies)
DEFAULT_CALLS = {"langgraph": 5005,      # Agent Server recursion_limit 10011 supersteps, about 2 per model call
                 "adk": 500,             # RunConfig.max_llm_calls
                 "crewai": 26,           # Agent.max_iter 25, plus the forced final answer
                 "openai_agents": 10,    # Runner.run max_turns
                 "autogen": 2}           # AssistantAgent max_tool_iterations 1, plus the reflection call
DEFAULT_SETTING = {"langgraph": "recursion_limit 10011", "adk": "max_llm_calls 500", "crewai": "max_iter 25",
                   "openai_agents": "max_turns 10", "autogen": "max_tool_iterations 1"}
LIMIT_STOPS = ("LlmCallsLimitExceededError", "GraphRecursionError", "LangGraph step limit", "MaxTurnsExceeded",
               "CrewAI max_iter", "AutoGen max_tool_iterations")


def collect(directory):
    rows, tasks = [], []
    for config in sorted(glob.glob(os.path.join(directory, "k*.yaml"))):
        cfg = Config(config)
        k = cfg.raw["tasks"]["subtasks"]
        chain = [cfg.framework(o) for o in cfg.orgs]
        fw_of = {f"{o}.a": cfg.framework(o) for o in cfg.orgs}
        d = os.path.join(HERE, "runs", cfg.name, "dynamicmcpbench")
        if not os.path.exists(os.path.join(d, "work.jsonl")):
            continue
        per = []
        for l in open(os.path.join(d, "work.jsonl")):
            if not l.strip():
                continue
            w = json.loads(l)
            if str(w["state"]).startswith("REQUEST_FAILED"):
                continue
            stops = [fw_of[a] for a, r in w["runs"].items() if r.get("error") and r["error"].startswith(LIMIT_STOPS)]
            reached = {h["to"].split(".")[0] for h in w["hops"]} - {"user"}
            per.append({"k": k, "task": w["id"][:8], "success": not stops and w["state"] == "TASK_STATE_COMPLETED",
                        "stopped_by": stops, "state": w["state"], "depth": len(reached),
                        "llm_calls": w["work"]["total"]["model_calls"], "tool_calls": w["work"]["total"]["tool_calls"],
                        "max_agent_calls": max((v["model_calls"] for v in w["work"]["per_agent"].values()), default=0),
                        "seconds": w["seconds"]})
        tasks += per
        if per:
            rows.append({"k": k, "chain": [NAMES[f] for f in chain], "tasks": len(per),
                         "success": sum(p["success"] for p in per),
                         "stopped_by": dict(Counter(NAMES[f] for p in per for f in p["stopped_by"])),
                         "other_failures": sum(1 for p in per if not p["success"] and not p["stopped_by"]),
                         "depth_mean": round(st.mean(p["depth"] for p in per), 2),
                         "llm_calls_mean": round(st.mean(p["llm_calls"] for p in per), 1),
                         "tool_calls_mean": round(st.mean(p["tool_calls"] for p in per), 1),
                         "allowed_calls": sum(DEFAULT_CALLS[f] for f in chain)})
    return rows, tasks


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE, GRID, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(rows)))
    rate = [100 * r["success"] / r["tasks"] for r in rows]
    ax.bar(xs, rate, color=BLUE, width=0.55, zorder=3)
    for x, r, v in zip(xs, rows, rate):
        ax.text(x, v + 2, f"{v:.0f}%  ({r['success']}/{r['tasks']})", ha="center", va="bottom", fontsize=9.5,
                color=INK, fontweight="bold")
        ax.text(x, v + 9, f"used {r['llm_calls_mean']:g} / allowed {r['allowed_calls']:,} LLM calls", ha="center",
                va="bottom", fontsize=8.2, color=MUTED)
        stops = ", ".join(f"{n} {c}" for n, c in r["stopped_by"].items())
        if stops:
            ax.text(x, 2, f"stopped by:\n{stops}", ha="center", va="bottom", fontsize=7.5, color="#ffffff")
    ax.set_xticks(xs, [f"{r['k']} subtasks\n{' → '.join(n.split()[0] for n in r['chain'])}" for r in rows],
                  fontsize=8.5, color=INK)
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("tasks finished without a limit stop (%)", fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    fig.suptitle("Each framework on its own default limit: does a k-subtask task finish on a chain of k frameworks?",
                 x=0.01, y=0.98, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, "DynamicMCPBench, 10 tasks per k, one subtask per A2A hop, llama3.2:3b; allowed = sum of the "
             "chain's defaults in LLM calls", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    directory = sys.argv[1]
    rows, tasks = collect(directory)
    for r in rows:
        print(f"k={r['k']}  {' -> '.join(r['chain'])}")
        print(f"   success {r['success']}/{r['tasks']}  stopped by {r['stopped_by'] or '-'}  other failures "
              f"{r['other_failures']}  depth reached {r['depth_mean']}/{r['k']}  LLM calls {r['llm_calls_mean']} of "
              f"{r['allowed_calls']} allowed  tool calls {r['tool_calls_mean']}")
    name = os.path.basename(os.path.normpath(directory))
    os.makedirs(os.path.join(HERE, "runs", name), exist_ok=True)
    json.dump({"defaults": DEFAULT_SETTING, "summary": rows, "tasks": tasks},
              open(os.path.join(HERE, "runs", name, f"{name}.json"), "w"), indent=1)
    out = os.path.join(os.path.dirname(HERE), "figures", f"{name}.png")
    plot(rows, out)
    print(out)

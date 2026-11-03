"""For a dataset experiment at limit levels (experiment.py --dataset --levels): per config (a delegation direction) and
level, how many tasks the second framework's agent (the target of the link) stopped with its framework's limit error,
how many it finished, how many it never received; plus correct answers and model calls. Writes a table and a figure.

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_limits.py testbed/configs/gaia_limits gaia
"""
import glob, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config

LIMIT_ERRORS = ("LlmCallsLimitExceededError", "GraphRecursionError", "LangGraph step limit", "MaxTurnsExceeded",
                "CrewAI max_iter", "CrewAI max_execution_time", "OpenAI Agents deadline", "AutoGen max_tool_iterations",
                "AutoGen deadline")
NAMES = {"langgraph": "LangGraph", "adk": "Google ADK", "crewai": "CrewAI", "crewai_a2a": "CrewAI (stock)",
         "crewai_a2a_patched": "CrewAI (patched)", "openai_agents": "OpenAI Agents SDK",
         "openai_agents_patched": "OpenAI Agents SDK (patched)", "autogen": "AutoGen", "autogen_patched": "AutoGen (patched)"}


def outcome(run):
    if not run:
        return "not reached"
    if run.get("error") == "LimitExhausted":   # 0 calls left: the testbed refuses (frameworks read 0 as "no limit")
        return "refused"
    if run.get("error") and any(run["error"].startswith(e) for e in LIMIT_ERRORS):
        return "limit error"
    return "other error" if run.get("error") else "finished"


def collect(directory, dataset):
    rows = []
    for config in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        cfg = Config(config)
        fo, fa, to, ta = cfg.links[0]
        second = f"{to}.{ta}"
        sender, receiver = NAMES[cfg.framework(fo)], NAMES[cfg.framework(to)]
        direction = f"{sender} → {receiver}"
        for level in cfg.raw.get("limit_levels") or {}:
            d = os.path.join(HERE, "runs", cfg.name, f"{dataset}_{level}")
            if not os.path.exists(os.path.join(d, "work.jsonl")):
                continue
            work = [json.loads(l) for l in open(os.path.join(d, "work.jsonl")) if l.strip()]
            results = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(d, "results.jsonl")) if l.strip())}
            def full(w):
                o = outcome(w["runs"].get(second))
                if o == "finished":
                    o = "finished, correct" if results.get(w["id"], {}).get("correct") else "finished, wrong"
                return o
            out = Counter(full(w) for w in work)
            bindings = Counter(w["runs"][second].get("binding") for w in work if w["runs"].get(second))
            bindings.pop(None, None)
            if not bindings:   # runs made before the binding was recorded: derive it from the level's settings
                spec, conv = cfg.raw["limit_levels"][level], cfg.raw.get("limit_conversion") or {}
                by_time = spec["deadline_s"] // conv.get("seconds_per_call", 1e-9)
                by_cost = spec["max_cost_usd"] // conv.get("usd_per_call", 1e-12)
                bindings = Counter({"depth" if spec["max_depth"] < 2 else "deadline" if by_time < by_cost else "cost": 1})
            rows.append({"direction": direction, "sender": sender, "receiver": receiver, "second": second, "level": level, "tasks": len(work),
                         "allowed_calls": sorted({w["runs"].get(second, {}).get("allowed_calls") for w in work} - {None}),
                         "binding": bindings.most_common(1)[0][0] if bindings else None,
                         "limit_error": out["limit error"], "refused": out["refused"],
                         "finished_correct": out["finished, correct"],
                         "finished_wrong": out["finished, wrong"], "other_error": out["other error"],
                         "not_reached": out["not reached"],
                         "correct": sum(1 for w in work if results.get(w["id"], {}).get("correct")),
                         "model_calls": sum(w["work"]["total"]["model_calls"] for w in work)})
    return rows


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    PARTS = [("limit_error", "stopped by its framework's limit error", "#eb6834"),
             ("refused", "refused: no call left (0 = no limit to the framework)", "#eda100"),
             ("finished_correct", "finished, correct answer", "#2a78d6"),
             ("finished_wrong", "finished, wrong answer", "#9ec5f4"),
             ("other_error", "other error", "#e4e3df"), ("not_reached", "not reached", "#c3c2b7")]

    def panel(ax, rs, labels):
        ax.set_facecolor(SURFACE)
        xs = range(len(rs))
        bottom = [0] * len(rs)
        for key, label, color in PARTS:
            vals = [r[key] for r in rs]
            if not any(vals):
                continue
            ax.bar(xs, vals, bottom=bottom, color=color, width=0.55, label=label, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            for x, v, b in zip(xs, vals, bottom):
                if v:
                    ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=9,
                            color="#ffffff" if color == "#2a78d6" else INK)
            bottom = [b + v for b, v in zip(bottom, vals)]
        calls = lambda r: (f"{min(r['allowed_calls'])}" if len(r["allowed_calls"]) == 1 else
                           f"{min(r['allowed_calls'])}-{max(r['allowed_calls'])}") if r["allowed_calls"] else "-"
        ax.set_xticks(list(xs), [f"{r['level']}\n{calls(r)}" for r in rs] if labels else [], fontsize=9,
                      color=INK)
        ax.set_ylim(0, max(r["tasks"] for r in rs) + 0.5)
        ax.set_ylabel("GAIA tasks", fontsize=9.5, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
        ax.tick_params(axis="x", length=0)

    # one figure per (sender, receiver)
    os.makedirs(out, exist_ok=True)
    from matplotlib.patches import Patch
    for direction in dict.fromkeys(r["direction"] for r in rows):
        rs = [r for r in rows if r["direction"] == direction]
        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        panel(ax, rs, True)
        present = {key for key, _, _ in PARTS if any(r[key] for r in rs)}
        fig.suptitle(f"{direction}   (sender \u2192 receiver)", x=0.01, y=0.98, ha="left", fontsize=13,
                     fontweight="bold", color=INK)
        fig.text(0.01, 0.915, "x axis: limit level and the model calls the receiver was allowed ('-' = no limit arrived)",
                 ha="left", fontsize=9, color=MUTED)
        fig.legend([Patch(color=c) for key, _, c in PARTS if key in present],
                   [l for key, l, _ in PARTS if key in present], frameon=False, fontsize=8.5, loc="upper left",
                   ncol=3, bbox_to_anchor=(0.005, 0.895))
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        name = f"{rs[0]['sender']}_to_{rs[0]['receiver']}".replace(" ", "_").lower()
        fig.savefig(os.path.join(out, name + ".png"), facecolor=SURFACE)
        plt.close(fig)

    # one figure per receiver: a panel per sender, two panels a row
    for receiver in dict.fromkeys(r["receiver"] for r in rows):
        senders = list(dict.fromkeys(r["sender"] for r in rows if r["receiver"] == receiver))
        cols, nrows = min(2, len(senders)), (len(senders) + 1) // 2
        fig, axes = plt.subplots(nrows, cols, figsize=(8.0 * cols, 4.0 * nrows + 1.3), dpi=170, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        rr = [r for r in rows if r["receiver"] == receiver]
        for ax, sender in zip(axes.flat, senders):
            panel(ax, [r for r in rr if r["sender"] == sender], True)
            ax.set_title(f"sender: {sender}", loc="left", fontsize=11.5, color=INK, fontweight="bold")
        for ax in list(axes.flat)[len(senders):]:
            ax.axis("off")
        present = {key for key, _, _ in PARTS if any(r[key] for r in rr)}
        fig.suptitle(f"Receiver: {receiver}. Does it stop under the user's limit?", x=0.01, y=0.985, ha="left",
                     fontsize=15, fontweight="bold", color=INK)
        top = 1 - 0.9 / (4.0 * nrows + 1.3)
        fig.text(0.01, top, "x axis: limit level and the model calls the receiver was allowed "
                 "('-' = no limit arrived)", ha="left", fontsize=10, color=MUTED)
        fig.legend([Patch(color=c) for key, _, c in PARTS if key in present],
                   [l for key, l, _ in PARTS if key in present], frameon=False, fontsize=10, loc="upper left",
                   ncol=6, bbox_to_anchor=(0.005, top - 0.005))
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.4 / (4.0 * nrows + 1.3)))
        fig.savefig(os.path.join(out, "receiver_" + receiver.replace(" ", "_").replace("(", "").replace(")", "")
                                 .lower() + ".png"), facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    directory, dataset = sys.argv[1], sys.argv[2]
    rows = collect(directory, dataset)
    print(f"{'direction':<38}{'level':<9}{'allowed':<9}{'binds':<9}{'limit error':>12}{'refused':>9}{'fin.correct':>12}{'fin.wrong':>10}"
          f"{'other':>7}{'not reached':>13}{'calls':>7}")
    for r in rows:
        allowed = ",".join(map(str, r["allowed_calls"]))[:8] or "-"
        print(f"{r['direction']:<38}{r['level']:<9}{allowed:<9}{r['binding'] or '-':<9}{r['limit_error']:>12}{r['refused']:>9}"
              f"{r['finished_correct']:>12}{r['finished_wrong']:>10}{r['other_error']:>7}{r['not_reached']:>13}{r['model_calls']:>7}")
    name = os.path.basename(os.path.normpath(directory))
    os.makedirs(os.path.join(HERE, "runs", name), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "runs", name, "limits.json"), "w"), indent=1)
    out = os.path.join(os.path.dirname(HERE), "figures", f"{name}_limit_errors")
    plot(rows, out)
    print(out)

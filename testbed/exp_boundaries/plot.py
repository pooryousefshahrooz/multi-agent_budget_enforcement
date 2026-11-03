"""Limit set by the user vs work actually used, per limit kind and level (results_*.json from run.py).
Bars: the limit, and the mean work over the runs of that level; dots: each run.

    python testbed/exp_boundaries/plot.py [model]      (default: qwen2.5:32b)
"""
import glob, json, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "runs", "exp_boundaries")
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:32b"
runs = defaultdict(list)   # level -> analyses
for fp in sorted(glob.glob(os.path.join(RUN, "results_*.json"))):
    d = json.load(open(fp))
    if d.get("model", "llama3.2:3b") != MODEL:   # the first files (llama3.2:3b) carry no model field
        continue
    for r in (d["runs"] if isinstance(d["runs"], list) else d["runs"].values()):
        runs[r["level"]].append(r)
LEVELS = [l for l in ("high", "middle", "low") if l in runs]
KINDS = [("max_depth", "Depth (hops)", "{:g}"), ("deadline", "Time (seconds)", "{:g}"),
         ("max_model_calls", "Model calls", "{:g}"), ("max_tool_calls", "Tool calls", "{:g}"),
         ("max_tokens", "Tokens", "{:,.0f}"), ("max_cost_usd", "Cost (USD)", "${:g}")]
LIMIT, USED = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
w = 0.38
for ax, (kind, title, fmt) in zip(axes.flat, KINDS):
    ax.set_facecolor(SURFACE)
    lim = [runs[l][0]["work_vs_limit"][kind]["limit"] for l in LEVELS]
    each = [[r["work_vs_limit"][kind]["used"] for r in runs[l]] for l in LEVELS]
    used = [sum(e) / len(e) for e in each]
    used = [round(u, 4) if kind == "max_cost_usd" else round(u) if kind == "deadline" else round(u, 1) for u in used]
    xs = range(len(LEVELS))
    ax.bar([x - w / 2 for x in xs], lim, width=w - 0.04, color=LIMIT, label="limit set by the user", zorder=3)
    ax.bar([x + w / 2 for x in xs], used, width=w - 0.04, color=USED, label="work actually used (mean)", zorder=3)
    for x, e in zip(xs, each):
        ax.scatter([x + w / 2] * len(e), e, s=14, color=INK, zorder=5, label="one run" if x == 0 else None,
                   edgecolors=SURFACE, linewidths=0.8)
    top = max(lim + [v for e in each for v in e])
    for x, l, u in zip(xs, lim, used):
        ax.text(x - w / 2, l + top * 0.02, fmt.format(l), ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(x + w / 2, u + top * 0.02, fmt.format(u), ha="center", va="bottom", fontsize=8, color=INK,
                fontweight="bold" if u > l else "normal")
    ax.set_ylim(0, top * 1.18)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(list(xs), [f"{l} limit" for l in LEVELS], fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9.5, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.965))
fig.suptitle("The work was the same whatever limit the user set", x=0.012, y=0.975, ha="left", fontsize=13,
             color=INK, fontweight="bold")
n = ", ".join(f"{len(runs[l])}" for l in LEVELS)
fig.text(0.012, 0.925, f"One LangGraph org, 4 agents (A2A between them), MCP-Atlas task, {MODEL}, runs per level: {n}. "
         "Bold = more work than the limit allowed.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = os.path.join(ROOT, "figures", f"exp1_boundaries_limit_vs_used_{MODEL.replace(':', '-')}.png")
fig.savefig(out, facecolor=SURFACE)
print(out)

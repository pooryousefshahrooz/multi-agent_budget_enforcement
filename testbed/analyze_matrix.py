"""Experiment 1's matrix: from the fates of the runs of an experiment's configs (analyze_fates.py output), the fate of the
user's request limit (delegation depth, deadline, cost) at each boundary type, per framework; and its figure.

  within framework         hops inside a native organization: preserved (same run) / duplicated (fresh run holding
                           the full limit) / dropped; deadline: preserved if the run holds it, else dropped
  across frameworks (A2A)  hops between organizations, by the RECEIVING framework: the sender always puts the limit on
                           the message; does the receiving framework's core take it into the run
  MCP tool call            depth, cost: is the limit in the tools/call request; deadline: who cut a long tool call
  MCP sampling             is the limit in the sampled model call

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_matrix.py testbed/configs/exp1
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import Config

KINDS = ["depth", "deadline", "cost"]
BOUNDARIES = ["within framework", "across frameworks (A2A)", "MCP tool call", "MCP sampling"]
PLOTTED = [BOUNDARIES[1]]   # the figure shows these boundary types for now (across frameworks only)
ORDER = ["preserved", "duplicated", "reset", "dropped"]
NAMES = {"langgraph": "LangGraph", "adk": "Google ADK", "crewai": "CrewAI", "crewai_a2a": "CrewAI (stock)",
         "crewai_a2a_patched": "CrewAI (patched)", "openai_agents": "OpenAI Agents SDK",
         "openai_agents_patched": "OpenAI Agents SDK (patched)", "autogen": "AutoGen", "autogen_patched": "AutoGen (patched)"}


def cells_from(cfg, rows):
    """(framework, boundary) -> {kind: fate, ...} for every hop of one run, with a label for the supplement."""
    out = []
    for r in rows:
        if r["kind"] == "within framework" and r["fate"] != "not crossed":
            fw = cfg.framework(r["hop"].split(".")[0])
            got = r["fate"] != "dropped"
            out.append((fw, BOUNDARIES[0], f"{r.get('mechanism') or 'native'}",
                        {"depth": r["fate"], "cost": r["fate"], "deadline": "preserved" if got else "dropped"}))
        elif r["kind"] == "across organizations":
            # the sender always puts the limit on the message; the cell is the RECEIVING framework's core
            fw, f = r["receiver_framework"], r["fate"].split(" ")[0]
            label = f"{r['sender_framework']}->{r['receiver_framework']}"   # the pair: sender -> receiver
            out.append((fw, BOUNDARIES[1], label, {k: f for k in KINDS}))
        elif r["kind"] == "MCP tool call":
            fw = cfg.framework(r["hop"].split(".")[0])
            f = r["fate"].split(" ")[0]
            cell = {"depth": f, "cost": f}
            if r.get("deadline_fate"):
                cell["deadline"] = r["deadline_fate"]
            out.append((fw, BOUNDARIES[2], r["hop"].split(" -> ")[1], cell))
        elif r["kind"] == "MCP sampling":
            fw = cfg.framework(r["hop"].split(".")[0])
            out.append((fw, BOUNDARIES[3], "sampled model call", {k: r["fate"] for k in KINDS}))
    return out


def collect(directory):
    matrix, detail = {}, []
    for config in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        cfg = Config(config)
        files = sorted(glob.glob(os.path.join(HERE, "runs", cfg.name, "fates_*.json")), key=os.path.getmtime)
        if not files:
            continue
        for fw, boundary, label, cell in cells_from(cfg, json.load(open(files[-1]))):
            slot = matrix.setdefault(fw, {}).setdefault(boundary, {k: set() for k in KINDS})
            for k, f in cell.items():
                slot[k].add(f)
            detail.append({"config": cfg.name, "framework": fw, "boundary": boundary, "case": label, **cell})
    return matrix, detail


MECHANISMS = {"transfer": "sub-agent transfer", "agent_tool": "AgentTool", "workflow": "Workflow",
              "subgraph": "subgraph", "tool": "agent in a tool"}


def plot(matrix, detail, out):
    """One row per framework pair (sender -> receiver) for the A2A boundary; one column per limit kind."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    FILL = {"preserved": "#2a78d6", "duplicated": "#eb6834", "reset": "#1baf7a", "dropped": "#eda100", "n/a": "#e4e3df"}
    TEXT = {"preserved": "#ffffff", "duplicated": "#0b0b0b", "reset": "#0b0b0b", "dropped": "#0b0b0b", "n/a": "#52514e"}
    INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#c3c2b7"
    order = list(NAMES)
    pairs = {}
    for d in detail:
        if d["boundary"] == BOUNDARIES[1]:
            pairs.setdefault(d["case"], d)
    keys = sorted(pairs, key=lambda k: (order.index(k.split("->")[0]), order.index(k.split("->")[1])))
    label_w, cw, ch = 3.4, 1.1, 0.5
    W, H = label_w + cw * 3 + 0.4, 1.75 + ch * len(keys) + 0.4
    W = max(W, 8.2)
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(H, 0), ax.axis("off")
    ax.text(0.15, 0.32, "Does the user's limit reach the receiving run across an A2A hop?", fontsize=13,
            fontweight="bold", color=INK, va="center")
    ax.text(0.15, 0.62, "The sender always puts the limit on the message; does the receiving framework's core keep it?",
            fontsize=8.6, color=MUTED, va="center")
    ax.text(0.15, 1.37, "sender  \u2192  receiver", fontsize=8.2, color=MUTED, va="center")
    ax.text(label_w + cw * 1.5, 1.0, "across frameworks (A2A)", ha="center", va="center", fontsize=9.3,
            fontweight="bold", color=INK)
    ax.plot([label_w + 0.06, label_w + cw * 3 - 0.06], [1.17, 1.17], color=RULE, lw=0.8)
    for k, kind in enumerate(KINDS):
        ax.text(label_w + cw * (k + 0.5), 1.37, kind, ha="center", va="center", fontsize=8.2, color=MUTED)
    y = 1.6
    for key in keys:
        snd, rcv = key.split("->")
        ax.text(0.15, y + ch / 2, f"{NAMES[snd]}  \u2192  {NAMES[rcv]}", fontsize=9.5, color=INK, va="center",
                fontweight="bold")
        for k, kind in enumerate(KINDS):
            f = pairs[key].get(kind, "n/a")
            x = label_w + cw * k
            ax.add_patch(Rectangle((x + 0.02, y + 0.03), cw - 0.04, ch - 0.06, facecolor=FILL[f], lw=0))
            ax.text(x + cw / 2, y + ch / 2, f, ha="center", va="center", fontsize=8, color=TEXT[f])
        y += ch
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    directory = sys.argv[1]
    matrix, detail = collect(directory)
    name = os.path.basename(os.path.normpath(directory))
    os.makedirs(os.path.join(HERE, "runs", name), exist_ok=True)
    json.dump({"matrix": {fw: {b: {k: sorted(v) for k, v in c.items()} for b, c in bs.items()} for fw, bs in matrix.items()},
               "cases": detail}, open(os.path.join(HERE, "runs", name, "matrix.json"), "w"), indent=1)
    for d in detail:
        print(f"  {d['framework']:<10} {d['boundary']:<24} {d['case']:<60} " + "  ".join(f"{k}={d.get(k, '-')}" for k in KINDS))
    out = os.path.join(os.path.dirname(HERE), "figures", f"{name}_matrix.png")
    plot(matrix, detail, out)
    print(out)

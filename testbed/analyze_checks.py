"""For the limit checks (experiment.py configs/limit_checks --levels ...): per receiver and limit level, how the
receiving run ended (finished, stopped by its limit, rejected), from each level's latest request record. The level of
a record is the config's limit level whose depth, budget and deadline it was sent with. Writes a table and a figure.

    experiments/exp1_propagation/envs/tasks/bin/python testbed/analyze_checks.py testbed/configs/limit_checks
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_fates import fates
from analyze_matrix import NAMES
from common.config import Config


def level_of(spec_levels, rec):
    lim = rec["limit"]
    for name, s in spec_levels.items():
        if (s["max_depth"] == lim["max_depth"] and s["max_cost_usd"] == lim["max_cost_usd"]
                and abs(s["deadline_s"] - (lim["deadline"] - rec["t0"])) < 2):
            return name
    return None


def collect(directory):
    rows = []
    for config in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        cfg = Config(config)
        levels = cfg.raw.get("limit_levels") or {}
        fo, fa, to, ta = cfg.links[0]
        latest = {}
        for f in sorted(glob.glob(os.path.join(HERE, "runs", cfg.name, "request_*.json")), key=os.path.getmtime):
            rec = json.load(open(f))
            if (lv := level_of(levels, rec)):
                latest[lv] = rec
        for lv in levels:
            if lv not in latest:
                continue
            hop = next((r for r in fates(cfg, latest[lv]) if r["kind"] == "across organizations"), {})
            out = hop.get("receiver_outcome") or {}
            rows.append({"receiver": NAMES[cfg.framework(to)], "sender": NAMES[cfg.framework(fo)], "level": lv,
                         "limit": {k: v for k, v in levels[lv].items()}, "fate": hop.get("fate"),
                         "held": hop.get("receiver_counter"), **out})
    return rows


def kind(r):
    e = r.get("ended")
    return "finished" if e == "finished" else "rejected" if e == "LimitExhausted" else "stopped" if r.get(
        "binding") else "error"


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#c3c2b7"
    FILL = {"finished": "#2a78d6", "stopped": "#eb6834", "rejected": "#eda100", "error": "#e4e3df"}
    TEXT = {"finished": "#ffffff", "stopped": INK, "rejected": INK, "error": INK}
    receivers = list(dict.fromkeys(r["receiver"] for r in rows))
    levels = list(dict.fromkeys(r["level"] for r in rows))
    DESC = {"high": "plenty", "budget": "budget: 2 calls", "depth": "0 hops left", "deadline": "6 s deadline"}
    label_w, cw, ch = 3.0, 2.55, 1.05
    W, H = label_w + cw * len(levels) + 0.3, 2.3 + ch * len(receivers) + 0.9
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.text(0.15, 0.45, "Patched frameworks under the user's limit, received over A2A", fontsize=13,
            fontweight="bold", color=INK, va="center")
    ax.text(0.15, 0.85, "Receiver scripted to make 4 tool calls (two 4 s, two instant); its own limit raised to 20, "
            "so only the user's limit binds", fontsize=8.5, color=MUTED, va="center")
    y0 = 1.75
    ax.text(0.15, y0, "receiver  (sender: LangGraph)", fontsize=8.5, color=MUTED, va="center")
    for j, lv in enumerate(levels):
        x = label_w + cw * j + cw / 2
        ax.text(x, y0 - 0.17, lv, fontsize=9.5, color=INK, ha="center", va="center", fontweight="bold")
        ax.text(x, y0 + 0.17, DESC.get(lv, ""), fontsize=8, color=MUTED, ha="center", va="center")
    for i, rcv in enumerate(receivers):
        y = y0 + 0.45 + ch * i
        ax.text(0.15, y + ch / 2, rcv, fontsize=10, color=INK, va="center", fontweight="bold")
        for j, lv in enumerate(levels):
            r = next((r for r in rows if r["receiver"] == rcv and r["level"] == lv), None)
            x = label_w + cw * j
            if r is None:
                continue
            k = kind(r)
            ax.add_patch(Rectangle((x + 0.04, y + 0.05), cw - 0.08, ch - 0.1, facecolor=FILL[k], edgecolor="none"))
            ended = {"finished": "finished", "LimitExhausted": "rejected"}.get(r["ended"], r["ended"])
            ended = ended.replace(" (forced final answer)", "").replace("CrewAI ", "").replace("AutoGen ", "") \
                .replace("OpenAI Agents ", "")
            detail = (f"{r['tool_calls']} tool / {r['model_calls']} model calls" if k != "rejected" else "no model call")
            if k == "stopped" and r["binding"] == "deadline":
                detail += f", {r['seconds']:.1f} s"
            ax.text(x + cw / 2, y + ch / 2 - 0.17, ended, fontsize=8.6, color=TEXT[k], ha="center", va="center",
                    fontweight="bold")
            ax.text(x + cw / 2, y + ch / 2 + 0.17, detail, fontsize=7.6, color=TEXT[k], ha="center", va="center")
        ax.plot([0.15, W - 0.15], [y + ch, y + ch], color=RULE, lw=0.5)
    present = [k for k in FILL if any(kind(r) == k for r in rows)]
    names = {"finished": "finished", "stopped": "stopped by its own limit, set from the user's",
             "rejected": "rejected: no hop left", "error": "other error"}
    ax.legend([Patch(color=FILL[k]) for k in present], [names[k] for k in present], frameon=False, fontsize=8.5,
              loc="lower left", ncol=len(present), bbox_to_anchor=(0.01, 0.0))
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    directory = sys.argv[1]
    rows = collect(directory)
    for r in rows:
        print(f"{r['receiver']:<30}{r['level']:<10}{kind(r):<10}{str(r.get('ended')):<45}"
              f"tools {r.get('tool_calls')}  model {r.get('model_calls')}  {r.get('seconds')}s  held {r.get('held')}")
    name = os.path.basename(os.path.normpath(directory))
    os.makedirs(os.path.join(HERE, "runs", name), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "runs", name, "checks.json"), "w"), indent=1)
    out = os.path.join(os.path.dirname(HERE), "figures", f"{name}.png")
    plot(rows, out)
    print(out)

"""The task interfaces of ReDel (branch demo/emnlp: bench_fanoutqa.py, bench_travelplanner.py), for the testbed.

Each dataset gives: its questions (ids as in ReDel), the text sent to the request's entry agent, what to do around a
request (reset/read the tool servers' per-task state), and the record written to results.jsonl in ReDel's format, so
score.py can apply ReDel's scorers.
"""
import hashlib, json, os

import httpx

# bench_travelplanner.py: ReDel's zero-shot root system prompt (ReDel gives it to the root agent as its system
# prompt; here the request text carries it, since the entry agent's role comes from the testbed config)
TP_ROOT_PROMPT = ("Based on the user's query, make the best travel plan for the user and save it. Do not ask "
                  "follow-up questions.")


def _server_url(cfg, name, path):
    from common.config import mcp_port
    return f"http://127.0.0.1:{mcp_port(cfg, name)}{path}"


class Dataset:
    def __init__(self, cfg=None):
        self.cfg = cfg


class FanOutQA(Dataset):
    """ReDel: fanoutqa.load_dev() (310 dev questions), the question as the query, the root's last text as answer."""
    name, server = "fanoutqa", "fanoutqa"

    def questions(self):
        import fanoutqa
        return [{"id": q.id, "question": q.question} for q in fanoutqa.load_dev()]

    def prompt(self, q):
        return q["question"]

    def before(self, cfg, q):
        httpx.post(_server_url(cfg, self.server, "/task"), json={"question": q["question"]})

    def record(self, cfg, q, answer):
        return {"id": q["id"], "answer": answer, "question": q["question"]}


class TravelPlanner(Dataset):
    """ReDel: osunlp/TravelPlanner validation (180 queries), id = sha256(query)[:16], the saved plan + text answer."""
    name, server = "travelplanner", "travelplanner-plan"

    def questions(self):
        from datasets import load_dataset
        data = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
        return [{"id": hashlib.sha256(q["query"].encode()).hexdigest()[:16], "idx": i, "question": q["query"]}
                for i, q in enumerate(data)]

    def prompt(self, q):
        return f"{TP_ROOT_PROMPT}\n\n{q['question']}"

    def before(self, cfg, q):
        httpx.delete(_server_url(cfg, self.server, "/plan"))

    def record(self, cfg, q, answer):
        plan = httpx.get(_server_url(cfg, self.server, "/plan")).json()
        return {"id": q["id"], "idx": q["idx"], "plan": plan, "answer": answer, "question": q["question"]}


class GAIA(Dataset):
    """GAIA validation (gaia-benchmark/GAIA, 2023). The tasks are the config's `tasks.ids` (else all level-1 tasks without
    an attached file); the answer is scored with GAIA's own rule (question_scorer, quasi exact match)."""
    name = "gaia"
    PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gaia", "2023", "validation", "metadata.parquet")

    def questions(self):
        import pandas as pd
        df = pd.read_parquet(self.PARQUET)
        ids = ((self.cfg.raw.get("tasks") or {}).get("ids") if self.cfg else None)
        if ids:
            df = df.set_index("task_id").loc[ids].reset_index()
        else:
            df = df[(df.Level == "1") & (df.file_name.fillna("") == "")]
        return [{"id": r.task_id, "question": r.Question, "gold": r["Final answer"]} for _, r in df.iterrows()]

    def prompt(self, q):
        return (f"{q['question']}\n\nFinish your reply with the line 'FINAL ANSWER: <answer>', where <answer> is as few "
                "words or a number as possible.")

    def before(self, cfg, q):
        pass

    def record(self, cfg, q, answer):
        given = answer.rsplit("FINAL ANSWER:", 1)[-1].strip() if "FINAL ANSWER:" in answer else answer.strip()
        return {"id": q["id"], "answer": answer, "given": given, "gold": q["gold"], "correct": gaia_score(given, q["gold"]),
                "question": q["question"]}


class DynamicMCPBench(Dataset):
    """DynamicMCPBench (TokenWasteGroup/DynamicMCPBench on Hugging Face, specs.jsonl). The tasks are the config's
    `tasks.ids`, else the first `tasks.n` (by task id) whose servers are all in `tasks.servers`, whose reference
    trace is at least `tasks.min_depth` tool steps deep, and (with `tasks.subtasks`) that have exactly that many
    subtasks (required checkpoints). The benchmark scores the effects of tool calls
    (checkpoints), not an answer, so the record keeps the answer and the checkpoints, with no `correct`."""
    name = "dynamicmcpbench"
    SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dynamicmcpbench", "raw", "specs.jsonl")

    def questions(self):
        specs = [json.loads(l) for l in open(self.SPECS) if l.strip()]
        t = (self.cfg.raw.get("tasks") or {}) if self.cfg else {}
        if t.get("ids"):
            by = {s["task_id"]: s for s in specs}
            specs = [by[i] for i in t["ids"]]
        else:
            servers, depth, k = set(t.get("servers") or []), t.get("min_depth", 0), t.get("subtasks")
            specs = sorted((s for s in specs if (not servers or set(s["servers_used"]) <= servers)
                            and s["complexity"]["trace_depth"] >= depth and (k is None or len(s["checkpoints"]) == k)),
                           key=lambda s: s["task_id"])[: t.get("n")]
        return [{"id": s["task_id"], "question": s["prompt"], "depth": s["complexity"]["trace_depth"],
                 "servers": s["servers_used"], "checkpoints": [c.get("checkpoint_id") for c in s["checkpoints"]]}
                for s in specs]

    def prompt(self, q):
        return q["question"]

    def before(self, cfg, q):
        pass

    def record(self, cfg, q, answer):
        return {"id": q["id"], "answer": answer, "given": answer, "reference_depth": q["depth"],
                "checkpoints": q["checkpoints"], "question": q["question"]}


def gaia_score(model_answer: str, ground_truth: str) -> bool:
    """GAIA's question_scorer (the leaderboard's quasi exact match): numbers compared as numbers, comma/semicolon lists
    element by element, strings after removing spaces, punctuation and case."""
    import re, string

    def number(x):
        x = x.replace("$", "").replace("%", "").replace(",", "").strip()
        try:
            return float(x)
        except ValueError:
            return float("inf")

    def norm(x, punct=True):
        x = re.sub(r"\s", "", x).lower()
        return x.translate(str.maketrans("", "", string.punctuation)) if punct else x

    def is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    if is_float(ground_truth):
        return number(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in ",;"):
        gt, ma = re.split(r"[,;]", ground_truth), re.split(r"[,;]", model_answer)
        if len(gt) != len(ma):
            return False
        return all(number(m) == float(g) if is_float(g) else norm(m, False) == norm(g, False) for m, g in zip(ma, gt))
    return norm(model_answer) == norm(ground_truth)


DATASETS = {"fanoutqa": FanOutQA, "travelplanner": TravelPlanner, "gaia": GAIA, "dynamicmcpbench": DynamicMCPBench}

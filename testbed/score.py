"""Score a bench.py results.jsonl with ReDel's scorers (score_fanoutqa.py, score_travelplanner.py, branch demo/emnlp).

* fanoutqa: the fanoutqa package's Scorer (what ReDel uses), on the answered questions: accuracy (loose/strict) and
  ROUGE; GPT-as-judge only if FANOUTQA_OPENAI_API_KEY is set; BLEURT left out (it needs TensorFlow + the BLEURT-20
  checkpoint).
* travelplanner: the official TravelPlanner evaluation (commonsense_constraint.py, hard_constraint.py), per query:
  delivered, commonsense pass, hard-constraint pass, final pass; and ReDel's fix-up of the saved plan (append the
  city to place names). ReDel's 2-step fallback (GPT turns a text answer into a plan when none was saved) needs
  OpenAI and is left out: no saved plan = not delivered.

    experiments/exp1_propagation/envs/tasks/bin/python testbed/score.py testbed/runs/<config>/<dataset>/results.jsonl
"""
import asyncio, json, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def score_fanoutqa(results):
    sys.modules.setdefault("bleurt", types.ModuleType("bleurt"))
    stub = types.ModuleType("bleurt.score")
    stub.BleurtScorer = lambda *a, **k: None
    sys.modules.setdefault("bleurt.score", stub)
    if not os.getenv("FANOUTQA_OPENAI_API_KEY"):
        # fanoutqa.eval.llm builds its GPT judge's OpenAI client on import, and current openai SDKs refuse the empty
        # key; without a key the judge is not run, so stand in for the module
        llm = types.ModuleType("fanoutqa.eval.llm")
        llm.OPENAI_API_KEY, llm.get_llm_factuality = "", None
        sys.modules["fanoutqa.eval.llm"] = llm
    import fanoutqa
    from fanoutqa.eval.llm import OPENAI_API_KEY
    from fanoutqa.eval.scorer import Scorer
    scorer = Scorer(fanoutqa.load_dev(), results, only_score_answered=True, llm_cache_key="testbed")
    acc, acc_raw = scorer.score_accuracy()
    rouge, _ = scorer.score_rouge()
    out = {"n": len(results), "acc_loose": acc.loose, "acc_strict": acc.strict,
           "rouge1_f": rouge.rouge1.fscore, "rouge2_f": rouge.rouge2.fscore, "rougeL_f": rouge.rougeL.fscore,
           "per_question": acc_raw}
    if OPENAI_API_KEY:
        out["gpt"], out["gpt_per_question"] = asyncio.run(scorer.score_gpt())
    return out


def redel_fix_plan(plan):
    """score_travelplanner.py: every place needs ', {CITY}' at the end; tack on the current city if missing."""
    for day in plan or []:
        for key in ("accommodation", "breakfast", "lunch", "dinner", "attraction"):
            if day.get(key) and day[key] != "-" and "," not in day[key]:
                *_, current_city = day["current_city"].split("to ")
                day[key] += f", {current_city}"
    return plan or None


def score_travelplanner(results):
    tp = os.path.join(HERE, "tasks", "data", "travelplanner")
    sys.modules.setdefault("gradio", types.ModuleType("gradio"))   # utils/func.py imports it for its annotation UI
    sys.path.insert(0, tp)
    sys.path.insert(0, os.path.join(tp, "evaluation"))
    cwd = os.getcwd()
    from commonsense_constraint import evaluation as commonsense_eval   # chdirs into evaluation/ (relative DB paths)
    from hard_constraint import evaluation as hard_eval
    os.chdir(cwd)
    from datasets import load_dataset
    queries = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    per, totals = {}, {"delivered": 0, "commonsense": 0, "hard": 0, "final": 0}
    for r in results:
        q = dict(queries[r["idx"]])
        if isinstance(q["local_constraint"], str):
            q["local_constraint"] = eval(q["local_constraint"])
        plan = redel_fix_plan(r.get("plan"))
        row = {"delivered": bool(plan), "commonsense": False, "hard": False, "final": False}
        if plan:
            # eval.py: macro pass rates count a query only if the hard constraints were checked, which needs the
            # plan to be complete and within the sandbox
            cs = commonsense_eval(q, plan)
            if cs["is_not_absent"][0] and cs["is_valid_information_in_sandbox"][0]:
                hard = hard_eval(q, plan)
                row["commonsense"] = all(v[0] is not False for v in cs.values())
                row["hard"] = all(v[0] is not False for v in hard.values())
            row["final"] = row["commonsense"] and row["hard"]
        per[r["id"]] = row
        for k in totals:
            totals[k] += row[k]
    n = len(results)
    return {"n": n, **{f"{k}_rate": v / n for k, v in totals.items()}, "per_query": per}


if __name__ == "__main__":
    fp = sys.argv[1]
    dataset = os.path.basename(os.path.dirname(os.path.abspath(fp)))
    results = read_jsonl(fp)
    score = score_fanoutqa(results) if dataset == "fanoutqa" else score_travelplanner(results)
    out = os.path.join(os.path.dirname(fp), "score.json")
    json.dump(score, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in score.items() if not k.startswith("per")}, indent=1))
    print(f"written to {out}")

"""Experiment 1, A2A into ADK, with the developer's runtime integration (limit_aware_adk.py) installed.

Same probes and same scripted model as test_a2a_into_adk.py; the only difference is that the ADK side reads the
request's limit from the A2A metadata and sets this run's RunConfig and deadline from it.

    experiments/exp1_propagation/.venv/bin/python experiments/exp1_propagation/test_a2a_into_adk_limit_aware.py
"""
import json, time

import limit_aware_adk
import probes
import test_a2a_into_adk as t


def main():
    Scripted, observer, noop = probes.adk_parts()
    from google.adk.agents import LlmAgent
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    rows = []
    for label, limit in [("3 calls", {"max_llm_calls": 3}),
                         ("600 calls", {"max_llm_calls": 600}),
                         ("deadline 2 s, 10000 calls", {"max_llm_calls": 10000, "deadline_in_s": 2.0})]:
        name = "aware_" + label.split()[0] + ("_dl" if "deadline" in label else "")
        probes.reset(name)
        p = t.free_port()
        agent = LlmAgent(name="worker", model=Scripted(mode="loop", delay=0.005), instruction="x", tools=[noop],
                         before_model_callback=[limit_aware_adk.callback, observer(name)])
        t.serve(to_a2a(agent, host="127.0.0.1", port=p), p)
        sent = dict(limit)
        if "deadline_in_s" in sent:
            sent["deadline"] = time.time() + sent.pop("deadline_in_s")
        r = t.send(f"http://127.0.0.1:{p}/", sent)
        calls = probes.read(name)
        made = len(calls) - 1 if calls else 0   # the observer also fires for the call that is refused
        rows.append({"limit sent": label, "model calls made": made,
                     "max_llm_calls seen by the run": sorted({c["max_llm_calls"] for c in calls}),
                     "run seconds": r["seconds"],
                     "calls after deadline": len([c for c in calls[:-1] if sent.get("deadline") and c["t"] > sent["deadline"]])})
    version = __import__("importlib.metadata").metadata.version("google-adk")
    json.dump(rows, open(f"{probes.OUT}/a2a_into_adk_limit_aware_{version}.json", "w"), indent=1)
    for row in rows:
        print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()

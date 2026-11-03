"""Org A passes a task to org B over A2A together with the user's limit; org B's agent runs on each framework
(latest versions) and maps the limit onto the framework's runtime settings (org_b_server.py). Does the limit hold?

For every framework: 5 call limits (1, 3, 10, 50, 600) and 5 deadlines (1, 2, 5, 10, 20 s). The fake model
(fake_llm.py) never lets a run finish and counts every model call, in the same unit for all frameworks.

    envs/langgraph/bin/python test_runtime_limits.py
"""
import json, os, subprocess, sys, time, uuid

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_PORT, LLM = 9100, "http://127.0.0.1:9100"
FRAMEWORKS = {"adk": 9201, "langgraph": 9202, "openai_agents": 9203, "crewai": 9204, "autogen": 9205}
ENV = {"adk": "adk", "langgraph": "langgraph", "openai_agents": "openai_agents", "crewai": "crewai", "autogen": "autogen"}
CALL_LIMITS = [1, 3, 10, 50, 600]
CAP_S = float(os.getenv("CAP_S", "120"))          # a run that is still going after this did not honour its limit
ONLY = [f for f in os.getenv("ONLY", "").split(",") if f]       # e.g. ONLY=crewai,autogen
SKIP_CALLS = os.getenv("SKIP_CALLS", "")                        # frameworks whose call-limit runs are already done
DEADLINES_S = [1, 2, 5, 10, 20]


def start(argv, log):
    return subprocess.Popen(argv, cwd=HERE, stdout=open(os.path.join(HERE, "out", log), "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def wait_up(url, timeout=180):
    t = time.time() + timeout
    while time.time() < t:
        try:
            httpx.get(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def send(port, limit):
    """Org A: one A2A 1.0 SendMessage; blocks until org B's task ends. A2A has no field for a limit, so it goes in
    both places A2A allows metadata: the message and the request (ADK 2.x passes only the request's on to the run)."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {
        "message": {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": "go"}],
                    "metadata": {"limit": limit}},
        "metadata": {"limit": limit}}}
    t0 = time.time()
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/", json=body, headers={"A2A-Version": "1.0"}, timeout=CAP_S)
    except httpx.ReadTimeout:   # the limit did not stop the run: record it, do not crash
        return time.time() - t0, "NOT STOPPED", f"still running after {CAP_S}s"
    task = r.json().get("result", {}).get("task", {})
    text = " ".join(p.get("text", "") for a in task.get("artifacts", []) for p in a.get("parts", []))
    return time.time() - t0, task.get("status", {}).get("state"), text


def calls(framework):
    return httpx.get(f"{LLM}/_calls/{framework}").json()


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    procs = [start([sys.executable, "fake_llm.py", "--port", str(LLM_PORT), "--delay", "0.02"], "fake_llm.log")]
    for fw, port in FRAMEWORKS.items():
        if ONLY and fw not in ONLY:
            continue
        py = os.path.join(HERE, "envs", ENV[fw], "bin", "python")
        procs.append(start([py, "-W", "ignore", "org_b_server.py", "--framework", fw, "--port", str(port),
                            "--llm", LLM], f"org_b_{fw}.log"))
    try:
        wait_up(f"{LLM}/_calls/x")
        for fw, port in FRAMEWORKS.items():
            if ONLY and fw not in ONLY:
                continue
            print(fw, "up" if wait_up(f"http://127.0.0.1:{port}/.well-known/agent-card.json") else "FAILED TO START",
                  flush=True)
        rows = []
        for fw, port in FRAMEWORKS.items():
            if ONLY and fw not in ONLY:
                continue
            for n in ([] if fw in SKIP_CALLS.split(",") else CALL_LIMITS):
                httpx.post(f"{LLM}/_reset/{fw}")
                secs, state, text = send(port, {"max_llm_calls": n})
                made = calls(fw)["calls"]
                rows.append({"framework": fw, "limit": f"{n} calls", "sent": n, "model_calls": made,
                             "held": made <= n, "exact": made == n, "seconds": round(secs, 2), "outcome": text[:90]})
                print(f"{fw:<14} {n:>4} calls -> {made:>5} model calls  {'exact' if made == n else ('within' if made <= n else 'EXCEEDED')}"
                      f"  ({text[:60]})", flush=True)
            for d in DEADLINES_S:
                httpx.post(f"{LLM}/_reset/{fw}")
                deadline = time.time() + d
                secs, state, text = send(port, {"max_llm_calls": 100000, "deadline": deadline})
                c = calls(fw)
                late = [t for t in c["times"] if t > deadline]
                rows.append({"framework": fw, "limit": f"deadline {d}s", "sent": d, "model_calls": c["calls"],
                             "calls_after_deadline": len(late), "run_seconds": round(secs, 2),
                             "held": not late, "outcome": text[:90]})
                print(f"{fw:<14} deadline {d:>2}s -> run {secs:6.2f}s, {c['calls']:>4} calls, {len(late)} after the deadline"
                      f"  ({text[:50]})", flush=True)
        json.dump(rows, open(os.path.join(HERE, "out", f"runtime_limits{'_' + '_'.join(ONLY) if ONLY else ''}.json"), "w"), indent=1)
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, 15)
            except Exception:
                pass


if __name__ == "__main__":
    main()

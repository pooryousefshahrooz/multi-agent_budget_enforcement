"""Experiment 1, boundary "across frameworks over A2A", receiving framework: Google ADK (upstream, unmodified).

The request and the code that carries it are the frameworks' own:
  * the agent is served by ADK's own A2A server (``google.adk.a2a.utils.agent_to_a2a.to_a2a``);
  * when it delegates onward it uses ADK's own A2A client (``RemoteA2aAgent``).
We only write the probes: the origin's A2A message (raw A2A 0.3 JSON-RPC, limit in ``message.metadata``), a scripted
model that never stops on its own, a read-only callback, and a recorder that receives ADK's outgoing request.

Fates (PDF §4.1): preserved / reset / dropped / duplicated, per limit:
  * LLM-call budget  -> how many model calls the run makes, and what max_llm_calls the run sees
  * deadline         -> whether the run keeps making model calls after the deadline
  * delegation depth -> whether ADK's outgoing A2A request carries the limit (and decremented)

    experiments/exp1_propagation/.venv/bin/python experiments/exp1_propagation/test_a2a_into_adk.py
"""
import json
import socket
import threading
import time
import uuid

import httpx
import uvicorn

import probes

LIMIT = {"max_depth": 1, "max_llm_calls": 3, "max_cost_usd": 0.001}   # deadline set at send time


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server


def send(url: str, limit: dict, timeout: float = 300) -> dict:
    """The origin: one A2A 1.0 SendMessage. No standard field exists for a limit, so it goes in the metadata that
    A2A allows: on the message and on the request (we put it in both)."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {
        "message": {"messageId": uuid.uuid4().hex, "role": "ROLE_USER",
                    "parts": [{"text": "Compare TCP Reno and TCP CUBIC under packet loss."}],
                    "metadata": {"limit": limit}},
        "metadata": {"limit": limit}}}
    t0 = time.time()
    r = httpx.post(url, json=body, timeout=timeout, headers={"A2A-Version": "1.0"})
    return {"seconds": round(time.time() - t0, 2), "response": r.json()}


def main():
    Scripted, observer, noop = probes.adk_parts()
    from google.adk.agents import LlmAgent
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    for name in ("adk_loop", "adk_forward", "downstream_recorder"):
        probes.reset(name)

    # (1) an ADK agent that would call its tool forever: what stops it?
    p1 = free_port()
    loop_agent = LlmAgent(name="worker", model=Scripted(mode="loop", delay=0.01), instruction="x", tools=[noop],
                          before_model_callback=observer("adk_loop"))
    serve(to_a2a(loop_agent, host="127.0.0.1", port=p1), p1)

    # (2) an ADK agent that delegates onward over A2A with ADK's own client
    rec = probes.Recorder("downstream_recorder", free_port()).start()
    p2 = free_port()
    downstream = RemoteA2aAgent(name="downstream", agent_card=f"{rec.url}/.well-known/agent-card.json")
    fwd_agent = LlmAgent(name="forwarder", model=Scripted(mode="transfer:downstream"), instruction="x",
                         sub_agents=[downstream], before_model_callback=observer("adk_forward"))
    serve(to_a2a(fwd_agent, host="127.0.0.1", port=p2), p2)

    deadline_s = 2.0
    limit = dict(LIMIT, deadline=time.time() + deadline_s)
    r1 = send(f"http://127.0.0.1:{p1}/", limit)
    calls = probes.read("adk_loop")
    after_deadline = [c for c in calls if c["t"] > limit["deadline"]]
    r2 = send(f"http://127.0.0.1:{p2}/", dict(LIMIT, deadline=time.time() + 60))
    received = probes.read("downstream_recorder")

    result = {
        "framework": f"Google ADK {__import__('importlib.metadata').metadata.version('google-adk')} (upstream)",
        "boundary": "across frameworks over A2A (into ADK)",
        "limit_sent": LIMIT, "deadline_after_s": deadline_s,
        "llm_calls": {"sent_limit": LIMIT["max_llm_calls"], "calls_made": len(calls),
                      "max_llm_calls_seen_by_run": sorted({c["max_llm_calls"] for c in calls}),
                      "limit_in_session_state": any("limit" in " ".join(c["state_keys"]) for c in calls),
                      "limit_visible_in_run": any(c.get("limit_visible_in_run") for c in calls)},
        "deadline": {"calls_after_deadline": len(after_deadline), "run_seconds": r1["seconds"]},
        "depth_on_outgoing_a2a": {"outgoing_requests": len(received),
                                  "outgoing_message_metadata": [x["message_metadata"] for x in received],
                                  "outgoing_params_metadata": [x["params_metadata"] for x in received]},
    }
    def fate_calls():
        seen = result["llm_calls"]["max_llm_calls_seen_by_run"]
        if seen == [LIMIT["max_llm_calls"]]:
            return "preserved"
        return "reset" if seen else "dropped"
    out_meta = [dict(m or {}, **(p or {})) for m, p in zip(result["depth_on_outgoing_a2a"]["outgoing_message_metadata"],
                                                         result["depth_on_outgoing_a2a"]["outgoing_params_metadata"])]
    result["fates"] = {
        "cost (LLM calls)": fate_calls(),
        "deadline": "preserved" if not after_deadline else "dropped",
        "depth (outgoing)": ("preserved" if any("limit" in m for m in out_meta) else "dropped") if received else "no outgoing request",
    }
    version = __import__('importlib.metadata').metadata.version('google-adk')
    with open(f"{probes.OUT}/a2a_into_adk_{version}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

"""Probes for Experiment 1 (propagation across boundaries). They observe, they never carry the request.

* ``Recorder``     a minimal A2A 0.3 agent (raw JSON-RPC) that only records the messages it receives, so we can see what
                   limit, if any, a framework puts on an outgoing A2A request.
* ``Scripted``     a fake model for ADK: it never finishes on its own (it calls a no-op tool forever, or transfers to a
                   sub-agent once), so a run stops only when some limit stops it. No real model, no cost.
* ``Observer``     a read-only ADK callback: before every model call it records what the run knows (its
                   ``RunConfig.max_llm_calls``, its session state, the text it received) and when the call happened.

All records go to JSON-lines files under experiments/exp1_propagation/out/.
"""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import AsyncGenerator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def record(name: str, row: dict):
    with open(os.path.join(OUT, f"{name}.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.time(), **row}) + "\n")


def read(name: str) -> list:
    path = os.path.join(OUT, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def reset(name: str):
    path = os.path.join(OUT, f"{name}.jsonl")
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# A2A recorder (A2A 0.3 JSON-RPC, written against the spec, no framework involved)
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, name: str, port: int):
        self.name, self.port, self.url = name, port, f"http://127.0.0.1:{port}"
        rec = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):   # the Agent Card (A2A 1.0 fields plus the 0.3 ones, so either client can use it)
                self._send({"name": rec.name, "description": "records what it receives", "version": "1.0",
                            "supportedInterfaces": [{"url": rec.url, "protocolBinding": "JSONRPC",
                                                     "protocolVersion": "1.0"}],
                            "url": rec.url, "protocolVersion": "0.3.0", "preferredTransport": "JSONRPC",
                            "capabilities": {}, "defaultInputModes": ["text/plain"],
                            "defaultOutputModes": ["text/plain"],
                            "skills": [{"id": "record", "name": "record", "description": "records", "tags": []}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                msg = (req.get("params") or {}).get("message") or {}
                record(rec.name, {"method": req.get("method"), "message_metadata": msg.get("metadata"),
                                  "params_metadata": (req.get("params") or {}).get("metadata"),
                                  "headers": {k: v for k, v in self.headers.items()
                                              if k.lower() not in ("host", "content-length", "accept-encoding")},
                                  "text": " ".join(p.get("text", "") for p in msg.get("parts", []))[:300]})
                if req.get("method") == "SendMessage":   # A2A 1.0
                    result = {"message": {"messageId": uuid.uuid4().hex, "role": "ROLE_AGENT",
                                          "parts": [{"text": "recorded"}]}}
                else:                                     # A2A 0.3
                    result = {"kind": "message", "messageId": uuid.uuid4().hex, "role": "agent",
                              "parts": [{"kind": "text", "text": "recorded"}]}
                self._send({"jsonrpc": "2.0", "id": req.get("id"), "result": result})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self


# ---------------------------------------------------------------------------
# ADK side: a scripted model and a read-only observer
# ---------------------------------------------------------------------------
def adk_parts():
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class Scripted(BaseLlm):
        """mode='loop': call the no-op tool forever. mode='transfer:<agent>': transfer once, then answer."""
        model: str = "scripted"
        mode: str = "loop"
        delay: float = 0.01

        async def generate_content_async(self, llm_request, stream=False) -> AsyncGenerator[LlmResponse, None]:
            time.sleep(self.delay)
            if self.mode.startswith("transfer:"):
                done = any(p.function_response for c in llm_request.contents for p in (c.parts or []))
                part = (types.Part.from_text(text="done") if done else
                        types.Part.from_function_call(name="transfer_to_agent",
                                                      args={"agent_name": self.mode.split(":", 1)[1]}))
            else:
                part = types.Part.from_function_call(name="noop", args={})
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    def observer(probe_name: str):
        def before_model(callback_context, llm_request):
            ic = callback_context._invocation_context
            text = " ".join(p.text or "" for p in (ic.user_content.parts if ic.user_content else []) if p.text)
            custom = getattr(ic.run_config, "custom_metadata", None) or {}
            record(probe_name, {"agent": ic.agent.name, "max_llm_calls": ic.run_config.max_llm_calls,
                                "state_keys": sorted(callback_context.state.to_dict().keys()),
                                "limit_visible_in_run": "limit" in json.dumps(custom, default=str),
                                "custom_metadata": json.loads(json.dumps(custom, default=str))})
            return None
        return before_model

    def noop() -> str:
        """Does nothing."""
        return "ok"

    return Scripted, observer, noop

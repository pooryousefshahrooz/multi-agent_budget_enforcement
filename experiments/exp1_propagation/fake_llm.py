"""A fake, OpenAI-compatible model that never lets a run finish, and counts every call (stdlib only).

Every framework here talks to it through its normal OpenAI client, so model calls are counted in the same unit
everywhere. It always asks for the agent's tool again: with tools in the request it returns a tool call; without
(ReAct-style frameworks such as CrewAI) it returns a text "Action". A run therefore stops only when a limit stops it.

  POST /<run>/v1/chat/completions   a model call (counted under <run>)
  GET  /_calls/<run>                {"calls": n, "times": [...]}
  POST /_reset/<run>                forget <run>

    python fake_llm.py --port 9100 [--delay 0.02]
"""
import argparse, json, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS, LOCK = {}, threading.Lock()


STEP = {"n": 0}


def completion(req: dict) -> dict:
    """Always one more tool call, each with a new argument (frameworks may stop a run that repeats itself)."""
    STEP["n"] += 1
    step = STEP["n"]
    tools = req.get("tools") or []
    usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    if tools:
        name = tools[0].get("function", {}).get("name", "noop")
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": name, "arguments": json.dumps({"step": step})}}]}
        finish = "tool_calls"
    else:   # ReAct text: always one more action
        msg = {"role": "assistant",
               "content": f"Thought: I need to use the tool again.\nAction: noop\nAction Input: {{\"step\": {step}}}"}
        finish = "stop"
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "fake"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage}


class Handler(BaseHTTPRequestHandler):
    delay = 0.02

    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/_calls/"):
            run = self.path.split("/")[2]
            with LOCK:
                times = list(CALLS.get(run, []))
            return self._json(200, {"calls": len(times), "times": times})
        self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.startswith("/_reset/"):
            with LOCK:
                CALLS.pop(self.path.split("/")[2], None)
            return self._json(200, {"ok": True})
        run = self.path.lstrip("/").split("/")[0]
        with LOCK:
            CALLS.setdefault(run, []).append(time.time())
        time.sleep(self.delay)
        resp = completion(body)
        if not body.get("stream"):
            return self._json(200, resp)
        # streaming: one chunk with the whole message, then usage, then [DONE]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        m = resp["choices"][0]["message"]
        delta = {"role": "assistant", "content": m.get("content")}
        if m.get("tool_calls"):
            delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(m["tool_calls"])]
        for chunk in ({**resp, "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {**resp, "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0]["finish_reason"]}]},
                      {**resp, "object": "chat.completion.chunk", "choices": [], "usage": resp["usage"]}):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--delay", type=float, default=0.02)
    a = p.parse_args()
    Handler.delay = a.delay
    print(f"fake model on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

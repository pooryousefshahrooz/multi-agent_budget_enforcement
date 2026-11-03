"""A fake, OpenAI-compatible model (stdlib only), so testbed runs are deterministic and free.

  --mode finite (default)  an agent with tools calls its first tool once, then answers; without tools it answers
                           at once. Every agent therefore finishes in one or two model calls.
  --mode loop              always one more tool call (each with a new argument): a run ends only when a limit
                           ends it. Used to test limits.
  scripted (any mode)      an agent whose instructions contain `SCRIPT: <tool> {json args}` lines makes exactly
                           those tool calls, in order, one per model call, then answers.

Every framework talks to it through its normal OpenAI client, so calls are counted in the same unit everywhere.

  POST /<run>/v1/chat/completions   a model call (counted under <run>)
  GET  /_calls/<run>                {"calls": n, "times": [...]}
  POST /_reset/<run>                forget <run>

    python fake_llm.py --port 9100 [--delay 0.02]
"""
import argparse, json, os, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS, LOCK = {}, threading.Lock()


STEP = {"n": 0}
MODE = {"mode": "finite"}


def _answered(req: dict) -> bool:
    """Has this conversation already had a tool result (or a ReAct observation)?"""
    for m in req.get("messages", []):
        content = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        if m.get("role") == "tool" or "Observation:" in (content or ""):
            return True
    return False


def _arguments(fn: dict, step: int) -> dict:
    """Valid arguments for the tool's schema, different at every step (frameworks reject repeated identical calls)."""
    props = (fn.get("parameters") or {}).get("properties") or {}
    by_type = {"integer": step, "number": step, "boolean": True, "array": [], "object": {}}
    args = {k: by_type.get(v.get("type"), f"x{step}") for k, v in props.items()}
    return args if props else {"step": step}


def _script(req: dict) -> list:
    """`SCRIPT: <tool> {json args}` lines in the system prompt: the tool calls this agent makes, in order."""
    system = " ".join(m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                      for m in req.get("messages", []) if m.get("role") in ("system", "developer"))
    out = []
    for line in system.replace("\\n", "\n").split("\n"):
        if "SCRIPT:" in line:
            name, _, args = line.split("SCRIPT:", 1)[1].strip().partition(" ")
            args = args.strip()
            out.append((name, json.JSONDecoder().raw_decode(args)[0] if args.startswith("{") else {}))
    return out


def _scripted(req: dict, tools: list, step: int):
    """The next scripted tool call (skipping the calls already made in this conversation), or None to answer."""
    script = _script(req)
    if not script:
        return "unscripted"
    names = [f.get("function", {}).get("name", "") for f in tools]
    described = {f.get("function", {}).get("name", ""): (f.get("function", {}).get("description") or "") for f in tools}
    # a tool matches by name, by a framework-prefixed name, or (when a framework shortens long names to a hash) by
    # a description that starts with "<name>:"
    match = lambda have, want: (have == want or have.endswith(want) or have.endswith("_" + want)
                                or described.get(have, "").lower().startswith(want.lower() + ":"))
    done = 0
    for m in req.get("messages", []):
        for tc in m.get("tool_calls") or []:
            if done < len(script) and match(tc.get("function", {}).get("name", ""), script[done][0]):
                done += 1
    if not any(m.get("tool_calls") for m in req.get("messages", [])):
        # some frameworks (CrewAI) send the tool results back without their own tool-call messages
        done = min(len(script), sum(1 for m in req.get("messages", []) if m.get("role") == "tool"))
    if done >= len(script):
        return None
    want, args = script[done]
    name = next((n for n in names if match(n, want)), None)
    if name is None:
        return None
    fn = next(f.get("function", {}) for f in tools if f.get("function", {}).get("name") == name)
    return name, {**_arguments(fn, step), **args}


def completion(req: dict) -> dict:
    STEP["n"] += 1
    step = STEP["n"]
    tools = req.get("tools") or []
    scripted = _scripted(req, tools, step)
    if scripted != "unscripted":   # a scripted agent: its tool calls in order, then the answer
        usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        if scripted is None:
            msg, finish = {"role": "assistant", "content": f"Final Answer: done (step {step})."}, "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                "function": {"name": scripted[0], "arguments": json.dumps(scripted[1])}}]}
            finish = "tool_calls"
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model", "fake"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage}
    if MODE["mode"] == "finite" and (not tools or _answered(req)):
        text = f"Final Answer: done (step {step})."
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model", "fake"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}}
    usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    if tools:
        fn = tools[0].get("function", {})
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": fn.get("name", "noop"), "arguments": json.dumps(_arguments(fn, step))}}]}
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
        if os.environ.get("FAKE_DUMP"):   # debugging: every request and answer
            with LOCK, open(os.environ["FAKE_DUMP"], "a") as f:
                f.write(json.dumps({"run": run, "request": body, "response": resp}) + "\n")
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
    p.add_argument("--mode", choices=["finite", "loop"], default="finite")
    a = p.parse_args()
    Handler.delay = a.delay
    MODE["mode"] = a.mode
    print(f"fake model on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

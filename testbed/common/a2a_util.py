"""A2A for organizations whose framework ships no A2A server or client (all but ADK).

* ``serve(card_name, port, run)``: an A2A 1.0 server (a2a-sdk) for one ingress agent; ``run(text, metadata)`` runs
  the organization's graph from that agent and returns its answer.
* ``remote_call(sender, org, agent, text, metadata)``: an A2A call to another organization's agent, through the
  instrumentation proxy (A2A 1.0, or 0.3 for a receiver whose card says so).
"""
import uuid

import httpx

from common.config import a2a_url


def text_of(task: dict) -> str:
    parts = [p.get("text", "") for a in task.get("artifacts", []) for p in a.get("parts", [])]
    if not parts and task.get("status", {}).get("message"):
        parts = [p.get("text", "") for p in task["status"]["message"].get("parts", [])]
    return "\n".join(x for x in parts if x)


CARDS = {}   # (org, agent) -> its Agent Card


def _legacy(card: dict) -> bool:
    """An A2A 0.3 server (e.g. CrewAI's own): a top-level protocolVersion 0.x and no A2A 1.0 interfaces."""
    return not card.get("supportedInterfaces") and str(card.get("protocolVersion", "")).startswith("0.")


async def remote_call(sender: str, org: str, agent: str, text: str, metadata: dict | None = None) -> str:
    """Send a task to another agent over A2A (A2A 1.0 SendMessage; A2A 0.3 message/send when the receiver's card says
    0.3, as a2a-sdk's own client does). With metadata (e.g. {"limit": ...}) it goes on the message and on the request."""
    url = a2a_url(sender, org, agent)
    async with httpx.AsyncClient(timeout=1800) as c:
        if (org, agent) not in CARDS:
            r = await c.get(url + "/.well-known/agent-card.json")
            CARDS[org, agent] = r.json() if r.status_code == 200 else {}
        legacy = _legacy(CARDS[org, agent])
        message = ({"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                    "parts": [{"kind": "text", "text": text}]} if legacy else
                   {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": text}]})
        params = {"message": message}
        if metadata:
            message["metadata"], params["metadata"] = metadata, metadata
        body = {"jsonrpc": "2.0", "id": 1, "method": "message/send" if legacy else "SendMessage", "params": params}
        r = await c.post(url + "/", json=body, headers={} if legacy else {"A2A-Version": "1.0"})
    reply = r.json()
    if "error" in reply:
        return f"ERROR {reply['error'].get('message', '')}"
    result = reply.get("result", {})
    if legacy:   # 0.3: the result is the Task (or a Message) itself
        return text_of(result) if result.get("kind") == "task" else text_of({"artifacts": [result]})
    return text_of(result.get("task", {})) or text_of({"artifacts": [result.get("message", {})]})


def serve_app(card_name: str, port: int, run):
    """Starlette app: an A2A server whose task runs `await run(text, metadata)` and returns its answer."""
    from google.protobuf import json_format
    from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
    from starlette.applications import Starlette

    class Executor(AgentExecutor):
        async def execute(self, context, event_queue):
            task = context.current_task or new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            up = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
            await up.update_status(state=TaskState.TASK_STATE_WORKING)
            meta = json_format.MessageToDict(context.message.metadata) if context.message.metadata else {}
            try:
                out = await run(get_message_text(context.message), meta)
                await up.add_artifact(parts=[new_text_part(text=str(out))])
                await up.complete()
            except Exception as e:   # each framework stops with its own exceptions
                await up.failed(new_text_message(f"{type(e).__name__}: {str(e)[:300]}"))

        async def cancel(self, context, event_queue):
            raise NotImplementedError

    card = AgentCard(name=card_name, description=card_name, version="1.0",
                     default_input_modes=["text/plain"], default_output_modes=["text/plain"],
                     capabilities=AgentCapabilities(streaming=False),
                     supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}",
                                                          protocol_version="1.0")],
                     skills=[AgentSkill(id="work", name="work", description="does its part", tags=["work"])])
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def compose(task: str, inputs: dict) -> str:
    """The text a node receives: the task, then the outputs of the nodes before it."""
    parts = [f"TASK: {task}"] + [f"--- from {k} ---\n{v}" for k, v in inputs.items() if v]
    return "\n\n".join(parts)

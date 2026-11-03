"""MCP sampling for an agent with `sampling: true`: the callback its framework's MCP client calls when a tool server
asks for a model call (sampling/createMessage). This is the developer's code (LangGraph's adapters and ADK take a
callback); it forwards the request to the model, through the proxy, as the agent "<agent>.sampling"."""
import httpx

from common.config import llm_url


def sampling_callback(org: str, agent: str):
    async def callback(context, params):
        from mcp.types import CreateMessageResult, TextContent
        messages = [{"role": m.role, "content": getattr(m.content, "text", "")} for m in params.messages]
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(llm_url(org, f"{agent}.sampling") + "/chat/completions",
                             json={"model": "model", "messages": messages,   # mcp 2.x: max_tokens, 1.x: maxTokens
                                   "max_tokens": getattr(params, "max_tokens", None) or getattr(params, "maxTokens", None)})
        text = r.json()["choices"][0]["message"].get("content") or ""
        return CreateMessageResult(role="assistant", model="model", content=TextContent(type="text", text=text))
    return callback

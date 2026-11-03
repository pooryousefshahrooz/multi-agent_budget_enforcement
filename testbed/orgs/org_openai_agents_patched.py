"""OpenAI Agents SDK organization served by the SDK's own A2A server, which the stock SDK does not have: env
envs/openai_agents_patched has testbed/patches/openai_agents_a2a_limit.patch applied (agents/a2a.py: `to_a2a`). The
server takes the `limit` from the request (depth -> reject, deadline -> the run is cancelled when it passes, cost ->
max_turns at usd_per_call, from the config's limit_conversion). This module only mounts it; the observer reports each
run's record (`on_run_end`)."""
from common.agent_logic import report_library_run
from orgs.org_openai_agents import agent_factory

STOPS = {"max_turns": ("MaxTurnsExceeded", "cost"), "deadline": ("OpenAI Agents deadline (DeadlineExceeded)", "deadline")}


def serve(cfg, org):
    from agents.a2a import to_a2a
    from agents.run_config import DEFAULT_MAX_TURNS
    from orgs.run_org import serve_apps
    price = (cfg.raw.get("limit_conversion") or {}).get("usd_per_call")

    def make_app(name, port):
        own = int(cfg.orgs[org].get("agent_limit") or DEFAULT_MAX_TURNS)

        async def on_run_end(r):
            held = {"max_turns": r.max_turns, "timeout_s": round(r.timeout_s) if r.timeout_s else None}
            await report_library_run(cfg, org, name, r.limit, held, {"max_turns": own, "timeout_s": None}, r.outcome,
                                     r.error, STOPS)
        return to_a2a(agent_factory(cfg, org, name), url=f"http://127.0.0.1:{port}", name=f"{org}.{name}",
                      max_turns=own, usd_per_call=price, on_run_end=on_run_end)
    serve_apps(cfg, org, make_app, "the OpenAI Agents SDK's own A2A server (patched)")

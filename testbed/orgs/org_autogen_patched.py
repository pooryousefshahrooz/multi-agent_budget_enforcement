"""AutoGen organization served by AutoGen's own A2A server, which stock AutoGen does not have: env envs/autogen_patched
has testbed/patches/autogen_a2a_limit.patch applied (autogen_agentchat/a2a.py: `to_a2a`). The server takes the `limit`
from the request (depth -> reject, deadline -> the run's CancellationToken is cancelled when it passes, cost ->
max_tool_iterations at usd_per_call, from the config's limit_conversion). This module only mounts it; the observer
reports each run's record (`on_run_end`). AutoGen's max_tool_iterations does not raise: the agent stops calling tools
and answers."""
from common.agent_logic import report_library_run
from orgs.org_autogen import agent_factory

STOPS = {"max_tool_iterations": ("AutoGen max_tool_iterations", "cost"),
         "deadline": ("AutoGen deadline (DeadlineExceeded)", "deadline")}


def serve(cfg, org):
    from autogen_agentchat.a2a import to_a2a
    from orgs.run_org import serve_apps
    price = (cfg.raw.get("limit_conversion") or {}).get("usd_per_call")

    def make_app(name, port):
        own = int(cfg.orgs[org].get("agent_limit") or 1)   # AssistantAgent's own default: 1 tool iteration

        async def on_run_end(r):
            held = {"max_tool_iterations": r.max_tool_iterations,
                    "timeout_s": round(r.timeout_s) if r.timeout_s else None}
            await report_library_run(cfg, org, name, r.limit, held, {"max_tool_iterations": own, "timeout_s": None},
                                     r.outcome, r.error, STOPS, **r.extra)
        return to_a2a(agent_factory(cfg, org, name, max_tool_iterations=own), url=f"http://127.0.0.1:{port}",
                      name=f"{org}.{name}", usd_per_call=price, on_run_end=on_run_end)
    serve_apps(cfg, org, make_app, "AutoGen's own A2A server (patched)")

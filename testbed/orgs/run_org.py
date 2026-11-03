"""One organization: every one of its agents as its own A2A server, in its framework's own Python environment.

LangGraph and ADK serve their agents with their own A2A servers (LangGraph Agent Server, ADK to_a2a); the other
frameworks have none, so each of their agents gets an a2a-sdk A2A 1.0 server. Each agent's endpoint is written to
runs/<config>/upstreams/<org>.json, where the instrumentation proxy finds it; all traffic goes through the proxy.

    experiments/exp1_propagation/envs/<framework>/bin/python testbed/orgs/run_org.py --config <config.yaml> --org orgA
"""
import argparse, importlib, json, os, sys, threading, time

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("LITELLM_LOG", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.agent_logic import AgentLogic
from common.config import Config, agent_port, upstreams_dir


def serve_with_a2a_sdk(cfg, org, adapter):
    import uvicorn

    from common.a2a_util import serve_app
    up, servers = {}, []
    native = cfg.orgs[org].get("delegation", "a2a") == "native"
    # a native organization: only its agents that receive work from outside are served; their work is the
    # framework's own multi-agent run over the organization's agents (adapter.native_work)
    for name in (cfg.ingress(org) if native else cfg.orgs[org]["agents"]):
        port = agent_port(cfg, org, name)
        logic = AgentLogic(cfg, org, name, (adapter.native_work if native else adapter.make_work)(cfg, org, name))
        if cfg.raw.get("entry_via") == "native" and (org, name) == cfg.entry:
            # the user's own entry into this application: its code hands the request's limit to the agent's run, as
            # Runner.run(context=...) / Crew.kickoff(inputs=...) would; no framework A2A server is involved
            app = serve_app(f"{org}.{name}", port, lambda text, meta, _l=logic: _l.handle(text, received=meta))
        else:   # A2A from another agent: the framework has no A2A server, so its run gets only the text
            app = serve_app(f"{org}.{name}", port, lambda text, meta, _l=logic: _l.handle(text))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        up[f"{org}.{name}"] = f"http://127.0.0.1:{port}/"
    while not all(s.started for s in servers):
        time.sleep(0.05)
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / {cfg.framework(org)}] {k} on a2a-sdk A2A {v}", flush=True)
    threading.Event().wait()


def serve_apps(cfg, org, make_app, how):
    """Serve each agent of the organization with the A2A app `make_app(name, port)` returns (a framework's own A2A
    server), and publish the endpoints for the proxy."""
    import uvicorn
    up, servers = {}, []
    for name in cfg.orgs[org]["agents"]:
        port = agent_port(cfg, org, name)
        server = uvicorn.Server(uvicorn.Config(make_app(name, port), host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        up[f"{org}.{name}"] = f"http://127.0.0.1:{port}/"
    while not all(s.started for s in servers):
        time.sleep(0.05)
    json.dump(up, open(os.path.join(upstreams_dir(cfg), f"{org}.json"), "w"), indent=1)
    for k, v in up.items():
        print(f"[{org} / {cfg.framework(org)}] {k} on {how} {v}", flush=True)
    threading.Event().wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--org", required=True)
    a = p.parse_args()
    cfg = Config(a.config)
    os.makedirs(upstreams_dir(cfg), exist_ok=True)
    adapter = importlib.import_module(f"orgs.org_{cfg.framework(a.org)}")
    print(f"[{a.org} / {cfg.framework(a.org)}] agents: {', '.join(cfg.orgs[a.org]['agents'])}; edges: "
          f"{', '.join(f'{x}->{y}' for x, y in cfg.edges(a.org)) or 'none'}; delegates: "
          f"{', '.join(f'{n}->{o}.{t}' for n in cfg.orgs[a.org]['agents'] for o, t in cfg.delegates(a.org, n)) or 'none'}",
          flush=True)
    if hasattr(adapter, "serve"):   # the framework's own A2A server
        adapter.serve(cfg, a.org)
    else:
        serve_with_a2a_sdk(cfg, a.org, adapter)


if __name__ == "__main__":
    main()

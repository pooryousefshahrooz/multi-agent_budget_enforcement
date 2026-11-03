"""Start or stop a testbed: the fake model (if used), the instrumentation proxy, the MCP servers, and one process per
organization, each in its framework's own Python environment (latest versions; they cannot share one).

    python testbed/launch.py up   testbed/configs/example.yaml
    python testbed/launch.py down testbed/configs/example.yaml
"""
import argparse, glob, json, os, shutil, signal, subprocess, sys, time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.config import FAKE_PORT, PROXY, Config, mcp_port, upstreams_dir

ROOT = os.path.dirname(HERE)
ENVS = os.getenv("TESTBED_ENVS", os.path.join(ROOT, "experiments", "exp1_propagation", "envs"))
TOOLS = {"npx": os.path.join(ENVS, "node", "bin", "npx"), "uvx": os.path.join(ROOT, ".venv", "bin", "uvx")}


def py(framework: str) -> str:
    return os.path.join(ENVS, framework, "bin", "python")


def processes(cfg: Config, config_path: str) -> list:
    tools_py = py("langgraph")   # any env with httpx + mcp + pyyaml
    procs = []
    model = cfg.raw.get("model") or {"kind": "fake"}
    if model.get("kind", "fake") == "fake":
        procs.append(("fake model", [tools_py, "-u", os.path.join(HERE, "fake_llm.py"), "--port", str(FAKE_PORT),
                                     "--mode", model.get("mode", "finite"), "--delay", str(model.get("delay", 0.01))]))
    procs.append(("proxy", [tools_py, "-u", os.path.join(HERE, "common", "instrument.py"), "--config", config_path]))
    for name, spec in cfg.mcp_servers().items():
        if spec.get("script"):
            procs.append((f"mcp {name}", [py(spec.get("env", "langgraph")), "-u", os.path.join(HERE, spec["script"]),
                                          "--http", str(mcp_port(cfg, name)), *map(str, spec.get("args", []))]))
        elif spec.get("command"):   # a stdio MCP server (e.g. an npm / PyPI package), served over HTTP by mcp-proxy
            port = spec["url"].split(":")[-1].split("/")[0]
            cmd = [TOOLS.get(spec["command"][0], spec["command"][0]), *spec["command"][1:]]
            procs.append((f"mcp {name}", [os.path.join(ENVS, "tasks", "bin", "mcp-proxy"), "--host", "127.0.0.1",
                                          "--port", port, "--", *cmd]))
    for org in cfg.orgs:
        fw = cfg.framework(org)
        procs.append((f"org {org}", [py(fw), "-W", "ignore", "-u", os.path.join(HERE, "orgs", "run_org.py"),
                                     "--config", config_path, "--org", org]))
    return procs


def up(cfg, config_path):
    run_dir = os.path.join(HERE, "runs", cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    pids = {}
    shutil.rmtree(upstreams_dir(cfg), ignore_errors=True)
    os.makedirs(upstreams_dir(cfg))
    env = {**os.environ, "PYTHONPATH": HERE, "PATH": os.path.join(ENVS, "node", "bin") + os.pathsep + os.environ["PATH"]}
    for label, argv in processes(cfg, config_path):
        log = open(os.path.join(run_dir, label.replace(" ", "_") + ".log"), "w")
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env, start_new_session=True)
        pids[label] = p.pid
    json.dump(pids, open(os.path.join(run_dir, "pids.json"), "w"))
    deadline = time.time() + 300   # every organization writes its agents' A2A endpoints once they are up
    while time.time() < deadline and len(glob.glob(os.path.join(upstreams_dir(cfg), "*.json"))) < len(cfg.orgs):
        time.sleep(1)
    endpoints = {}
    for f in glob.glob(os.path.join(upstreams_dir(cfg), "*.json")):
        endpoints.update(json.load(open(f)))
    for org in cfg.orgs:
        print(f"{org:<8} {cfg.framework(org):<14} agents: {', '.join(cfg.orgs[org]['agents'])}")
    print(f"links (delegate tools): {', '.join(f'{a}.{b} -> {c}.{d}' for a, b, c, d in cfg.links) or '-'}")
    total = sum(len(cfg.orgs[o]["agents"]) for o in cfg.orgs)
    print(f"{len(endpoints)}/{total} agents have their own A2A endpoint; logs in {run_dir}")


def down(cfg):
    path = os.path.join(HERE, "runs", cfg.name, "pids.json")
    for label, pid in json.load(open(path)).items():
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    os.remove(path)
    print("stopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["up", "down"])
    p.add_argument("config")
    a = p.parse_args()
    cfg = Config(a.config)
    up(cfg, os.path.abspath(a.config)) if a.command == "up" else down(cfg)

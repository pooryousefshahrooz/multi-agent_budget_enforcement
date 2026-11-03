"""Start / stop the experiment: the instrumentation proxy, the MCP servers (MCP-Atlas's, behind mcp-proxy), and one
LangGraph Agent Server per agent (`langgraph dev`), then record each agent's A2A endpoint for the proxy.

    python testbed/exp_boundaries/launch.py up | down
"""
import json, os, signal, subprocess, sys, time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
TESTBED = os.path.dirname(HERE)
sys.path.insert(0, TESTBED)
from common.config import PROXY, Config

ROOT = os.path.dirname(TESTBED)
ENVS = os.path.join(ROOT, "experiments", "exp1_propagation", "envs")
CONFIG = os.path.join(HERE, "config.yaml")
CFG = Config(CONFIG)
RUN = os.path.join(TESTBED, "runs", CFG.name)
AGENT_PORT = {"A": 9921, "B": 9922, "C": 9923, "D": 9924}
TOOLS = {"npx": os.path.join(ENVS, "node", "bin", "npx"), "uvx": os.path.join(ROOT, ".venv", "bin", "uvx")}


def start(label, argv, cwd, env):
    log = open(os.path.join(RUN, label + ".log"), "w")
    return subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True).pid


def wait(url, seconds=300):
    end = time.time() + seconds
    while time.time() < end:
        try:
            if httpx.get(url, timeout=3).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def up():
    os.makedirs(RUN, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": TESTBED, "EXP_CONFIG": CONFIG,
           "PATH": os.path.join(ENVS, "node", "bin") + os.pathsep + os.environ["PATH"]}
    pids = {"proxy": start("proxy", [os.path.join(ENVS, "langgraph", "bin", "python"), "-u",
                                     os.path.join(TESTBED, "common", "instrument.py"), "--config", CONFIG], TESTBED, env)}
    for name, spec in CFG.mcp_servers().items():
        port = spec["url"].split(":")[-1].split("/")[0]
        cmd = [TOOLS.get(spec["command"][0], spec["command"][0]), *spec["command"][1:]]
        pids[f"mcp_{name}"] = start(f"mcp_{name}", [os.path.join(ENVS, "tasks", "bin", "mcp-proxy"), "--host",
                                                    "127.0.0.1", "--port", port, "--", *cmd], RUN, env)
    for agent, port in AGENT_PORT.items():
        d = os.path.join(RUN, "servers", agent)
        os.makedirs(d, exist_ok=True)
        json.dump({"dependencies": [TESTBED], "graphs": {agent: os.path.join(HERE, "agents.py") + f":graph_{agent}"}},
                  open(os.path.join(d, "langgraph.json"), "w"))
        pids[f"agent_{agent}"] = start(f"agent_{agent}", [
            os.path.join(ENVS, "langgraph", "bin", "langgraph"), "dev", "--port", str(port), "--no-browser",
            "--no-reload", "--n-jobs-per-worker", "20", "--config", "langgraph.json"], d, env)
    json.dump(pids, open(os.path.join(RUN, "pids.json"), "w"))
    upstreams = {}
    for agent, port in AGENT_PORT.items():
        base = f"http://127.0.0.1:{port}"
        if not wait(f"{base}/ok"):
            raise SystemExit(f"agent {agent} did not start; see {RUN}/agent_{agent}.log")
        aid = httpx.post(f"{base}/assistants/search", json={"graph_id": agent}).json()[0]["assistant_id"]
        upstreams[f"orgA.{agent}"] = f"{base}/a2a/{aid}"
    json.dump(upstreams, open(os.path.join(RUN, "upstreams.json"), "w"), indent=1)
    for name, spec in CFG.mcp_servers().items():
        print(f"mcp {name}: {'up' if wait(spec['url']) else 'NOT UP'}")
    print(f"proxy: {'up' if wait(PROXY + '/_events/none') else 'NOT UP'}")
    for k, v in upstreams.items():
        print(f"{k}: {v}")


def down():
    for label, pid in json.load(open(os.path.join(RUN, "pids.json"))).items():
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    print("stopped")


if __name__ == "__main__":
    up() if sys.argv[1] == "up" else down()

"""Operator tool: start (and stop) a whole testbed from a deployment config, like docker-compose.

It only launches processes. Each gateway, A2A enforcement point and agent runs as its own independent process; they
find each other over mDNS and nothing talks to deploy.py at run time. Any of them can also be started by hand with the
command `deploy.py show` prints.

    python delegation_mesh/deploy.py up      deployments/three_orgs.yaml
    python delegation_mesh/deploy.py status  deployments/three_orgs.yaml
    python delegation_mesh/deploy.py down    deployments/three_orgs.yaml
    python delegation_mesh/deploy.py show    deployments/three_orgs.yaml    (topology + the command of every process)
"""
import argparse, json, os, signal, subprocess, sys, time

import topology

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KNOWN_TOOLS = {"tcp-reference": "mcp_servers/tcp_reference.py"}


def plan(topo: topology.Topology, config: str) -> list:
    """Every process of the deployment: (label, argv)."""
    py, procs = sys.executable, []
    ports = {}
    for i, (name, spec) in enumerate(topo.tools().items()):
        spec = spec or {}
        ports[name] = spec.get("port", 8765 + i)
        script = os.path.join(HERE, spec.get("script", KNOWN_TOOLS.get(name, "")))
        procs.append((f"tool {name}", [py, "-u", script, "--http", str(ports[name])]))
    for org in topo.orgs:
        argv = [py, "-u", os.path.join(HERE, "gateway.py"), "--org", org]
        for t in topo.gateway_tools(org):
            argv += ["--mcp", f"{t}=http://127.0.0.1:{ports[t]}/mcp"]
        procs.append((f"{org} gateway", argv))
    for aid in topo.agents:
        a = topo.agent(aid)
        org_spec = topo.orgs[a["org"]] or {}
        agent_argv = [os.path.join(HERE, a["framework_info"]["script"]), "--config", config, "--agent", aid]
        if a.get("own_model"):
            agent_argv += ["--own-model", a["own_model"]]
        if a.get("guarded", org_spec.get("guarded", True)):
            argv = [py, "-u", os.path.join(HERE, "a2a_guard.py"), "--org", a["org"], *agent_argv]
        else:
            argv = [py, "-u", *agent_argv]
        procs.append((f"{a['org']} {aid}", argv))
    return procs


def run_dir(config: str) -> str:
    d = os.path.join(ROOT, "runs", os.path.splitext(os.path.basename(config))[0])
    os.makedirs(d, exist_ok=True)
    return d


def up(topo, config):
    d = run_dir(config)
    pids_file = os.path.join(d, "pids.json")
    if os.path.exists(pids_file) and any(_alive(p) for p in json.load(open(pids_file)).values()):
        sys.exit(f"already running (see {pids_file}); run `deploy.py down {config}` first")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if topo.raw.get("model"):
        env["LLM_MODEL"] = topo.raw["model"]
    if topo.raw.get("upstream"):
        env["OLLAMA_V1"] = topo.raw["upstream"]
    pids = {}
    for label, argv in plan(topo, config):
        log = open(os.path.join(d, label.replace(" ", "_") + ".log"), "w")
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        pids[label] = p.pid
        print(f"started {label:<26} pid {p.pid}")
    json.dump(pids, open(pids_file, "w"), indent=1)
    expected = {f"{a['org']} {a['id']}" for a in topo.agents.values()}
    deadline = time.time() + 240
    while time.time() < deadline:
        ready = {l for l in expected if _ready(os.path.join(d, l.replace(" ", "_") + ".log"))}
        if ready == expected:
            break
        time.sleep(2)
    print(f"\n{len(ready)}/{len(expected)} agents up; logs in {d}")
    for l in sorted(expected - ready):
        print(f"  not up yet: {l}")


def _ready(log_path: str) -> bool:
    try:
        text = open(log_path).read()
    except FileNotFoundError:
        return False
    return "A2A enforcement point for" in text or ("A2A server up" in text and "behind an A2A" not in text)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def down(config):
    pids_file = os.path.join(run_dir(config), "pids.json")
    if not os.path.exists(pids_file):
        sys.exit("nothing recorded as running for this config")
    pids = json.load(open(pids_file))
    for label, pid in pids.items():
        if _alive(pid):
            os.killpg(pid, signal.SIGTERM)   # the process group: a guard and the agent it started
            print(f"stopped {label}")
    os.remove(pids_file)


def status(config):
    pids_file = os.path.join(run_dir(config), "pids.json")
    if not os.path.exists(pids_file):
        sys.exit("nothing recorded as running for this config")
    for label, pid in json.load(open(pids_file)).items():
        print(f"{label:<26} pid {pid:<7} {'running' if _alive(pid) else 'STOPPED'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["up", "down", "status", "show"])
    p.add_argument("config")
    args = p.parse_args()
    config = os.path.abspath(args.config)
    topo = topology.load(config)
    if args.command == "show":
        print(topo.describe(), "\n")
        for label, argv in plan(topo, config):
            print(f"# {label}\n" + " ".join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in argv))
    elif args.command == "up":
        print(topo.describe(), "\n")
        up(topo, config)
    elif args.command == "down":
        down(config)
    else:
        status(config)

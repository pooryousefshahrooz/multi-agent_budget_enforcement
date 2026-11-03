"""Run an experiment on the testbed: every config in a directory, one after the other, each the normal way; every
organization is its own process.

  one request per config      launch.py up, run.py, analyze_fates.py, launch.py down
  a dataset at limit levels   launch.py up, bench.py --level L for each level, launch.py down   (--dataset, --levels)

    experiments/exp1_propagation/envs/tasks/bin/python testbed/experiment.py testbed/configs/exp1 [--task "..."]
    experiments/exp1_propagation/envs/tasks/bin/python testbed/experiment.py testbed/configs/gaia_limits \
        --dataset gaia --levels cannot can high
"""
import argparse, glob, os, subprocess, sys, time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = os.path.join(os.path.dirname(HERE), "experiments", "exp1_propagation", "envs")
PY = os.path.join(ENVS, "langgraph", "bin", "python")
TASKS_PY = os.path.join(ENVS, "tasks", "bin", "python")


def step(argv, quiet=False):
    p = subprocess.run(argv, cwd=os.path.dirname(HERE), capture_output=True, text=True)
    if not quiet or p.returncode:
        print(p.stdout.strip()[-1500:] + (("\n" + p.stderr.strip()[-800:]) if p.returncode else ""), flush=True)
    return p.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--task", default="Do the work.")
    ap.add_argument("--dataset", help="run this dataset (bench.py) instead of one request")
    ap.add_argument("--levels", nargs="*", default=[], help="the config's limit_levels to run the dataset at")
    ap.add_argument("--match", default="*.yaml", help="only the configs whose file name matches (glob)")
    a = ap.parse_args()
    for config in sorted(glob.glob(os.path.join(a.directory, a.match))):
        print(f"===== {os.path.basename(config)}", flush=True)
        if step([PY, os.path.join(HERE, "launch.py"), "up", config], quiet=True):
            continue
        try:
            if a.dataset:
                time.sleep(15)   # MCP servers started through mcp-proxy / uvx need a moment
                for level in a.levels or [None]:
                    print(f"--- level {level}", flush=True)
                    subprocess.run([TASKS_PY, "-u", os.path.join(HERE, "bench.py"), config, "--dataset", a.dataset]
                                   + (["--level", level] if level else []), cwd=os.path.dirname(HERE))
            else:   # one request, at each of the config's limit levels if --levels is given
                if yaml.safe_load(open(config)).get("mcp_servers"):
                    time.sleep(15)   # MCP servers started through mcp-proxy / uvx need a moment
                levels = (yaml.safe_load(open(config)).get("limit_levels") or {}) if a.levels else {}
                for level in a.levels or [None]:
                    over = []
                    if level:
                        print(f"--- level {level}", flush=True)
                        spec = levels[level]
                        over = ["--max-depth", str(spec["max_depth"]), "--deadline-s", str(spec["deadline_s"]),
                                "--max-cost-usd", str(spec["max_cost_usd"])]
                    step([PY, os.path.join(HERE, "run.py"), config, "--task", a.task] + over, quiet=True)
                    step([TASKS_PY, os.path.join(HERE, "analyze_fates.py"), config])
        finally:
            step([PY, os.path.join(HERE, "launch.py"), "down", config], quiet=True)
            time.sleep(2)

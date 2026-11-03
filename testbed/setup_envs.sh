#!/bin/bash
# Build the testbed's environments from the pinned package lists in testbed/envs/ (exported from the machine the
# experiments were developed on), on any Linux or macOS machine with uv:
#
#   bash testbed/setup_envs.sh
#
# Writes, all inside the repository (git-ignored):
#   experiments/exp1_propagation/envs/<env>   one Python 3.12 environment per framework (what launch.py runs), the
#                                             three patched libraries (testbed/patches), Node.js (for npx MCP servers)
#   .venv                                     uv / uvx (launch.py starts uvx-based MCP servers with it)
# Caches go to $TESTBED_CACHE (default: <repo>/_cache), not the home directory.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENVS=${TESTBED_ENVS:-$ROOT/experiments/exp1_propagation/envs}
export TESTBED_CACHE=${TESTBED_CACHE:-$ROOT/_cache}
export UV_CACHE_DIR=$TESTBED_CACHE/uv UV_PYTHON_INSTALL_DIR=$TESTBED_CACHE/python npm_config_cache=$TESTBED_CACHE/npm
UV=${UV:-uv}
NODE_VERSION=${NODE_VERSION:-24.22.1}
mkdir -p "$ENVS" "$TESTBED_CACHE"

env_from() {   # env_from <name> <package list>
    "$UV" venv "$ENVS/$1" --python 3.12 -q --allow-existing
    "$UV" pip install -q --python "$ENVS/$1/bin/python" -r "$ROOT/testbed/envs/$2.txt"
    echo "env $1 ready"
}

for n in langgraph adk crewai crewai_a2a openai_agents autogen tasks; do
    env_from "$n" "$n"
done

# the patched libraries: the stock environment plus the patch (testbed/patches)
for spec in crewai_a2a:crewai_a2a_limit openai_agents:openai_agents_a2a_limit autogen:autogen_a2a_limit; do
    base=${spec%%:*}; patch=${spec##*:}
    env_from "${base}_patched" "$base"
    site=$("$ENVS/${base}_patched/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    patch -p1 -N -d "$site" < "$ROOT/testbed/patches/$patch.patch" || true   # -N: already applied is fine
done

# Node.js for npx-launched MCP servers, inside the environments folder
"$UV" pip install -q --python "$ENVS/tasks/bin/python" nodeenv
[ -x "$ENVS/node/bin/npx" ] || "$ENVS/tasks/bin/nodeenv" --node="$NODE_VERSION" --prebuilt "$ENVS/node"

# uv / uvx at <repo>/.venv
"$UV" venv "$ROOT/.venv" --python 3.12 -q --allow-existing
"$UV" pip install -q --python "$ROOT/.venv/bin/python" uv
echo "all environments in $ENVS; uvx in $ROOT/.venv/bin"

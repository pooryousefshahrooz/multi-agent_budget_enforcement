#!/bin/bash
# Vera (C3SE) job: serve a model with vLLM on this node's GPUs, run a command against it, stop the server.
# Everything it writes stays inside the repository (weights in _cache/hf, logs and results next to it).
#
#   submit from the project root on Vera:   sbatch vera_job.sbatch
#   change resources without editing:       sbatch --gpus-per-node=A40:4 --time=02:00:00 vera_job.sbatch
#   change parameters without editing:      MODEL=Qwen/Qwen2.5-7B-Instruct sbatch vera_job.sbatch
#
# ---- resources (Slurm reads these lines before the script runs, so they cannot use variables) ----
#SBATCH -A C3SE2026-1-12                 # project account (projinfo)
#SBATCH -p vera                          # partition
#SBATCH --job-name=vllm
#SBATCH --output=result_%j.out           # %j = job id
#SBATCH --error=error_%j.err
#SBATCH --time=01:00:00                  # wall-clock limit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A40:2            # GPU type:count (A40 48 GB, A100, H100); the model is split over them

set -euo pipefail

# ---- parameters (each can be overridden from the environment when submitting) ----
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}                               # Hugging Face model id
VLLM_SIF=${VLLM_SIF:-/apps/containers/vLLM/vllm-0.17.0.sif}             # C3SE's vLLM container
export HF_HOME=${HF_HOME:-$SLURM_SUBMIT_DIR/_cache/hf}                 # weights: in the repo (on group storage)
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}                                    # context length the server allows
GPU_MEM=${GPU_MEM:-0.90}                                                # share of each GPU's memory vLLM may use
VLLM_ARGS=${VLLM_ARGS:-"--enable-auto-tool-choice --tool-call-parser hermes"}   # tool calling (Qwen2.5: hermes)
WAIT_MIN=${WAIT_MIN:-30}                                                # minutes to wait for download + loading
RUN=${RUN:-"python3 cluster/vera/ask.py --port $PORT --model $MODEL"}   # what to run once the server is up
TP=${SLURM_GPUS_ON_NODE:-1}                                             # tensor parallel = GPUs of this node

mkdir -p "$HF_HOME"
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname); model $MODEL on $TP GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. the vLLM server (OpenAI-compatible API) in the background; the first run downloads the weights
apptainer exec --nv "$VLLM_SIF" vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM" $VLLM_ARGS > "vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2> /dev/null || true' EXIT

# 2. wait until it answers
for i in $(seq 1 $((WAIT_MIN * 6))); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "server ready after $((i * 10)) s"; break; fi
    if ! kill -0 "$VLLM_PID" 2> /dev/null; then echo "vLLM exited; its log:"; tail -40 "vllm_${SLURM_JOB_ID}.log"; exit 1; fi
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready after $WAIT_MIN min"; exit 1; }

# 3. the work, on the node itself; the testbed finds the model through these
export TESTBED_MODEL_URL="http://127.0.0.1:$PORT/v1" TESTBED_MODEL_NAME="$MODEL"
bash -c "$RUN"

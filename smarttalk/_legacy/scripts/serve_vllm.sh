#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/serve_vllm.sh <model_path_or_hf_repo> <served_model_name> [options]

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/serve_vllm.sh microsoft/phi-4 Phi-4 \
    --tensor-parallel-size 4 \
    --port 8000

Options:
  --host <host>                         Default: 0.0.0.0
  --port <port>                         Default: 8000
  --tensor-parallel-size <n>            Default: 1
  --dtype <dtype>                       Default: auto
  --gpu-memory-utilization <float>      Default: 0.92
  --max-model-len <tokens>              Default: 8192
  --no-trust-remote-code                Disable --trust-remote-code
  -- <extra vLLM args...>               Forward remaining args directly
EOF
}

if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

MODEL_PATH="$1"
SERVED_MODEL_NAME="$2"
shift 2

HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE="1"
DTYPE="auto"
GPU_MEMORY_UTILIZATION="0.92"
MAX_MODEL_LEN="8192"
TRUST_REMOTE_CODE="true"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --no-trust-remote-code)
      TRUST_REMOTE_CODE="false"
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

CMD=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf '+'
for token in "${CMD[@]}"; do
  printf ' %q' "$token"
done
printf '\n'

exec "${CMD[@]}"

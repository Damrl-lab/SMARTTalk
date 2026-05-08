#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_single_model_inference.sh \
    --method <raw|heuristic|smarttalk> \
    --dataset <MB1|MB2> \
    --round <1|2|3> \
    --model-name <served_model_name> \
    --base-url <openai_compatible_url> \
    [--api-key <key>] \
    [--processed-root <dir>] \
    [--artifacts-root <dir>] \
    [--output-root <dir>] \
    [--num-samples <failed_cap>] \
    [--sampled-indices-csv <csv>] \
    [--healthy-per-fail <ratio> | --evaluate-all] \
    [--temperature <float>] \
    [--max-tokens <n>]

Examples:
  bash scripts/run_single_model_inference.sh \
    --method smarttalk \
    --dataset MB2 \
    --round 1 \
    --model-name Phi-4 \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --sampled-indices-csv data/processed/sampled_test_1to23/sampled_test_indices.csv

  bash scripts/run_single_model_inference.sh \
    --method raw \
    --dataset MB1 \
    --round 2 \
    --model-name gpt-4o \
    --base-url https://api.openai.com/v1
EOF
}

METHOD=""
DATASET=""
ROUND=""
MODEL_NAME=""
BASE_URL=""
API_KEY="${OPENAI_API_KEY:-EMPTY}"
PROCESSED_ROOT="data/processed"
ARTIFACTS_ROOT="data/artifacts"
OUTPUT_ROOT="results/single_runs"
NUM_SAMPLES=""
SAMPLED_INDICES_CSV="data/processed/sampled_test_1to23/sampled_test_indices.csv"
HEALTHY_PER_FAIL="23"
EVALUATE_ALL="false"
TEMPERATURE="0.0"
MAX_TOKENS="512"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --round)
      ROUND="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --processed-root)
      PROCESSED_ROOT="$2"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --sampled-indices-csv)
      SAMPLED_INDICES_CSV="$2"
      shift 2
      ;;
    --healthy-per-fail)
      HEALTHY_PER_FAIL="$2"
      shift 2
      ;;
    --evaluate-all)
      EVALUATE_ALL="true"
      shift
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$METHOD" || -z "$DATASET" || -z "$ROUND" || -z "$MODEL_NAME" || -z "$BASE_URL" ]]; then
  usage >&2
  exit 1
fi

case "$METHOD" in
  raw)
    SCRIPT_PATH="code/core/raw_llm_eval.py"
    ;;
  heuristic)
    SCRIPT_PATH="code/core/heuristic_llm_eval.py"
    ;;
  smarttalk)
    SCRIPT_PATH="code/core/llm_eval.py"
    ;;
  *)
    echo "Unsupported method: $METHOD" >&2
    exit 1
    ;;
esac

SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME// /_}"
RUN_DIR="${OUTPUT_ROOT}/${METHOD}/${DATASET}_round${ROUND}/${SAFE_MODEL_NAME}"
mkdir -p "$RUN_DIR"

OUTPUT_JSONL="${RUN_DIR}/predictions.jsonl"
OUTPUT_METRICS_JSON="${RUN_DIR}/metrics.json"
OUTPUT_TP_CSV="${RUN_DIR}/tp_ttf.csv"

CMD=(
  python3 "$SCRIPT_PATH"
  --dataset-name "$DATASET"
  --round "$ROUND"
  --processed-root "$PROCESSED_ROOT"
  --model-name "$MODEL_NAME"
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
  --output-jsonl "$OUTPUT_JSONL"
  --output-metrics-json "$OUTPUT_METRICS_JSON"
)

if [[ "$METHOD" == "smarttalk" ]]; then
  CMD+=(--artifact-root "$ARTIFACTS_ROOT" --output-tp-csv "$OUTPUT_TP_CSV")
fi

if [[ "$EVALUATE_ALL" == "true" ]]; then
  CMD+=(--evaluate-all)
else
  CMD+=(--healthy-per-fail "$HEALTHY_PER_FAIL")
  if [[ -n "$SAMPLED_INDICES_CSV" ]]; then
    CMD+=(--sampled-indices-csv "$SAMPLED_INDICES_CSV")
  fi
  if [[ -n "$NUM_SAMPLES" ]]; then
    CMD+=(--num-samples "$NUM_SAMPLES")
  fi
fi

printf '+'
for token in "${CMD[@]}"; do
  printf ' %q' "$token"
done
printf '\n'

exec "${CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
ASSISTANT_MODEL_ID="google/gemma-4-31B-it-assistant"
SPECULATIVE_CONFIG='{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
IMAGE="vllm/vllm-openai:v0.25.1"
CONTAINER_NAME="gemma-4-nvfp4-vllm"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
USER_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6 || true)"
HF_HOME="${USER_HOME:-${HOME}}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

if [[ ! -f "${WORK_DIR}/chat_template.jinja" ]]; then
  echo "Missing chat_template.jinja in ${WORK_DIR}"
  exit 1
fi

# ---- Hugging Face model download helpers ----
is_hf_model_id() {
  [[ "${1}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]
}

hf_cache_repo_dir() {
  echo "${HF_HOME}/hub/models--${1//\//--}"
}

model_is_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"

  [[ -d "${cache_dir}/snapshots" ]] || return 1

  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/model-"*.safetensors >/dev/null; then
      return 0
    fi
  done

  return 1
}

download_model() {
  local model_id="$1"
  echo "Downloading model ${model_id} to ${HF_HOME}"
  echo "This may take a while for large models..."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      hf download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      huggingface-cli download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  # Fallback: use the same vLLM image to download via Python
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)"
}

ensure_model_available() {
  local model_id="$1"
  if is_hf_model_id "${model_id}"; then
    if model_is_cached "${model_id}"; then
      echo "Model ${model_id} is already cached in ${HF_HOME}"
    else
      download_model "${model_id}"
    fi
    return
  fi

  if [[ "${model_id}" == /* || "${model_id}" == ./* || "${model_id}" == ../* ]]; then
    if [[ ! -d "${model_id}" ]]; then
      echo "Local model directory not found: ${model_id}"
      exit 1
    fi
    echo "Using local model at ${model_id}"
  fi
}

ensure_model_available "${MODEL_ID}"
ensure_model_available "${ASSISTANT_MODEL_ID}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  -e VLLM_TARGET_DEVICE=cuda \
  -e HF_HOME="${HF_HOME}" \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:${HF_HOME}" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --quantization modelopt \
    --attention-backend triton_attn \
    --chat-template /workspace/chat_template.jinja \
    --gpu-memory-utilization 0.70 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --async-scheduling \
    --mm-encoder-tp-mode data \
    --enable-prefix-caching \
    --allowed-media-domains '*' \
    --speculative-config "${SPECULATIVE_CONFIG}" \
    --load-format fastsafetensors \
    --limit-mm-per-prompt '{"image": 4, "video": 1, "audio": 1}' \
    --reasoning-parser gemma4 \
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"

echo "vLLM is ready and responding; shell is now free."

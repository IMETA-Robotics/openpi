#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate openpi

HOST="127.0.0.1"
PORT="8000"
PROMPT="Grab the black electrical tape from the left plane to the right plane"
CAMERA_TYPE="v4l2"
CAMERA_NAMES=("cam_high" "cam_right_wrist" "cam_left_wrist")
CONTROL_FREQUENCY="30"
INFERENCE_RATE="4"
CHUNK_SIZE="30"
MAX_PUBLISH_STEP="10000000"

# 可选: true / false
SINGLE_ARM="false"
VISUAL="false"
VISUAL_FPS="30"
INTERPOLATION="true"
RTC_MASK_PREFIX_DELAY="false"
RTC_DISABLE_SMOOTHING="false"
RTC_MODEL_CHUNK_SIZE="50"

INTERP_STEPS="10"
INTERP_FREQUENCY="300"
RTC_EXECUTE_HORIZON="8"
RTC_MAX_GUIDANCE_WEIGHT="0.3"
LATENCY_K="8"
MIN_SMOOTH_STEPS="8"
BUFFER_MAX_CHUNKS="10"

# 可选: 留空表示不执行 init_state
INIT_STATE=""

CMD=(
  uv run python y1_openpi_inference_rtc.py
  --host "${HOST}"
  --port "${PORT}"
  --prompt "${PROMPT}"
  --camera_type "${CAMERA_TYPE}"
  --camera_names "${CAMERA_NAMES[@]}"
  --visual_fps "${VISUAL_FPS}"
  --control_frequency "${CONTROL_FREQUENCY}"
  --inference_rate "${INFERENCE_RATE}"
  --chunk_size "${CHUNK_SIZE}"
  --max_publish_step "${MAX_PUBLISH_STEP}"
  --interp_steps "${INTERP_STEPS}"
  --interp_frequency "${INTERP_FREQUENCY}"
  --latency_k "${LATENCY_K}"
  --min_smooth_steps "${MIN_SMOOTH_STEPS}"
  --buffer_max_chunks "${BUFFER_MAX_CHUNKS}"
  --rtc_model_chunk_size "${RTC_MODEL_CHUNK_SIZE}"
  --rtc_execute_horizon "${RTC_EXECUTE_HORIZON}"
  --rtc_max_guidance_weight "${RTC_MAX_GUIDANCE_WEIGHT}"
)

if [[ "${SINGLE_ARM}" == "true" ]]; then
  CMD+=(--single_arm)
fi

if [[ "${VISUAL}" == "true" ]]; then
  CMD+=(--visual)
fi

if [[ "${INTERPOLATION}" == "true" ]]; then
  CMD+=(--interpolation)
fi

if [[ "${RTC_MASK_PREFIX_DELAY}" == "true" ]]; then
  CMD+=(--rtc_mask_prefix_delay)
fi

if [[ "${RTC_DISABLE_SMOOTHING}" == "true" ]]; then
  CMD+=(--rtc_disable_smoothing)
fi

if [[ -n "${INIT_STATE}" ]]; then
  # shellcheck disable=SC2206
  INIT_STATE_ARR=(${INIT_STATE})
  CMD+=(--init_state "${INIT_STATE_ARR[@]}")
fi

echo "[INFO] Starting Y1 RTC inference"
echo "[INFO] HOST=${HOST}:${PORT}"
echo "[INFO] PROMPT=${PROMPT}"
echo "[INFO] CAMERA_TYPE=${CAMERA_TYPE}"
echo "[INFO] CAMERA_NAMES=${CAMERA_NAMES[*]}"
echo "[INFO] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[INFO] RTC_MODEL_CHUNK_SIZE=${RTC_MODEL_CHUNK_SIZE}"
echo "[INFO] RTC_EXECUTE_HORIZON=${RTC_EXECUTE_HORIZON}"

"${CMD[@]}"

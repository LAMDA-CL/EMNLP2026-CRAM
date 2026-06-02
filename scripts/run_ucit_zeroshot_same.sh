#!/usr/bin/env bash
# UCIT: train on sub dataset, infer on full dataset.
# Each phase failure does not block later phases.
#
# Usage:
#   bash scripts/run_ucit_zeroshot_same.sh
#   GPUS=0,1 PORT=29602 bash scripts/run_ucit_zeroshot_same.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-29602}"
BACKBONE="${BACKBONE:-internvl}"
BENCHMARK="${BENCHMARK:-ucit}"
STAMP="$(date +%m-%d-%H-%M)"
LOG_DIR="${PROJECT_ROOT}/output/${BACKBONE}/scripts/${BENCHMARK}"
LOG_FILE="${LOG_DIR}/run_ucit_sub_${STAMP}.txt"

TASKS=(0 1 2 3 4 5)
LAST_TASK=5

COMMON_BASE=(
  --benchmark "${BENCHMARK}"
  --backbone "${BACKBONE}"
  --gpus "${GPUS}"
)

TRAIN_COMMON=(
  "${COMMON_BASE[@]}"
  --use-sub-dataset
)

INFER_COMMON=(
  "${COMMON_BASE[@]}"
  --no-use-sub-dataset
)

mkdir -p "${LOG_DIR}"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}" | tee -a "${LOG_FILE}"
}

run_cmd() {
  log "COMMAND: ${PYTHON} run.py $*"
  "${PYTHON}" run.py "$@" 2>&1 | tee -a "${LOG_FILE}"
  return "${PIPESTATUS[0]}"
}

# Train 0..5, then infer 0..5 with Task5 checkpoint. Returns 0 on success, 1 on failure.
run_train_infer() {
  local method="$1"

  log "--- ${method} train tasks ${TASKS[*]} ---"
  if ! run_cmd train "${TASKS[@]}" --method "${method}" --port "${PORT}" "${TRAIN_COMMON[@]}"; then
    log "${method} train: FAILED — skipping infer"
    return 1
  fi
  log "${method} train: OK"

  log "--- ${method} infer tasks ${TASKS[*]} (checkpoint-task=${LAST_TASK}) ---"
  if ! run_cmd infer "${TASKS[@]}" --method "${method}" --checkpoint-task "${LAST_TASK}" "${INFER_COMMON[@]}"; then
    log "${method} infer: FAILED"
    return 1
  fi
  log "${method} infer: OK"
  return 0
}

log "=== UCIT pipeline (train: sub, infer: full) ==="
log "project: ${PROJECT_ROOT}"
log "log: ${LOG_FILE}"
log "gpus=${GPUS} port=${PORT} backbone=${BACKBONE}"

any_failed=false

# ---------------------------------------------------------------------------
# Phase 1: SAME
# ---------------------------------------------------------------------------
log ""
log "=== Phase 1: SAME train + infer ==="
if run_train_infer same; then
  log "Phase 1 finished: OK"
else
  log "Phase 1 finished with failures — continuing"
  any_failed=true
fi

# ---------------------------------------------------------------------------
# Phase 2: zeroshot inference
# ---------------------------------------------------------------------------
log ""
log "=== Phase 2: zeroshot inference ==="

zeroshot_failed=()
for t in "${TASKS[@]}"; do
  log "--- zeroshot infer task ${t} ---"
  if run_cmd infer "${t}" --method zeroshot "${INFER_COMMON[@]}"; then
    log "zeroshot task ${t}: OK"
  else
    rc=$?
    log "zeroshot task ${t}: FAILED (exit ${rc})"
    zeroshot_failed+=("${t}")
  fi
done

if ((${#zeroshot_failed[@]} == 0)); then
  log "Phase 2 finished: OK"
else
  log "Phase 2 finished with failures: tasks ${zeroshot_failed[*]} — continuing"
  any_failed=true
fi

# ---------------------------------------------------------------------------
# Phase 3: moelora
# ---------------------------------------------------------------------------
log ""
log "=== Phase 3: moelora train + infer ==="
if run_train_infer moelora; then
  log "Phase 3 finished: OK"
else
  log "Phase 3 finished with failures — continuing"
  any_failed=true
fi

# ---------------------------------------------------------------------------
# Phase 4: ft_lora
# ---------------------------------------------------------------------------
log ""
log "=== Phase 4: ft_lora train + infer ==="
if run_train_infer ft_lora; then
  log "Phase 4 finished: OK"
else
  log "Phase 4 finished with failures"
  any_failed=true
fi

log ""
if [[ "${any_failed}" == "false" ]]; then
  log "=== All done ==="
  exit 0
else
  log "=== Done with failures ==="
  exit 1
fi

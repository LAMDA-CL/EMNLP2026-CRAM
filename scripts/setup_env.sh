#!/usr/bin/env bash
# Create / refresh the PRISM conda env and install all dependencies in one shot.
#
# Usage (from repo root):
#   bash scripts/setup_env.sh
#
# Options (environment variables):
#   CONDA_ENV_NAME=prism       conda env name (default: prism)
#   PYTHON_VERSION=3.10        Python version for new envs
#   FLASH_ATTN_WHEEL=/path/to.whl   optional prebuilt flash-attn wheel
#   SKIP_FLASH_ATTN=1          skip flash-attn install
#   TORCH_REQUIREMENTS=requirements/torch.txt   override torch stack file
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-prism}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_REQ="${TORCH_REQUIREMENTS:-requirements/torch.txt}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found in PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
else
  echo "==> Reusing existing conda env: ${ENV_NAME}"
fi

conda activate "$ENV_NAME"

echo "==> Upgrading pip tooling"
python -m pip install -U pip wheel

echo "==> Installing PyTorch (${TORCH_REQ})"
python -m pip install -r "${REPO_ROOT}/${TORCH_REQ}"

echo "==> Installing training + eval stack"
python -m pip install -r "${REPO_ROOT}/requirements/train.txt"
python -m pip install -r "${REPO_ROOT}/requirements/eval.txt"

if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  echo "==> Installing flash-attn"
  if [[ -n "${FLASH_ATTN_WHEEL:-}" && -f "${FLASH_ATTN_WHEEL}" ]]; then
    python -m pip install "${FLASH_ATTN_WHEEL}"
  else
    if ! python -m pip install -r "${REPO_ROOT}/requirements/flash-attn.txt" --no-build-isolation; then
      echo "warning: flash-attn install failed; set FLASH_ATTN_WHEEL or SKIP_FLASH_ATTN=1"
    fi
  fi
else
  echo "==> Skipping flash-attn (SKIP_FLASH_ATTN=1)"
fi

echo "==> Installing PRISM (editable)"
python -m pip install -e "${REPO_ROOT}[train,eval]"

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Verify:"
echo "  python -c \"import torch; import transformers; import deepspeed; print(torch.__version__, transformers.__version__)\""

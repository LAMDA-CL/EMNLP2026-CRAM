# PRISM environment setup

Python **3.10** and a **CUDA-capable GPU** are required for training.

The default stack in `scripts/setup_env.sh` is **pinned for RTX 5090** (torch 2.8 + cu128). PRISM has also been run on **RTX 3090**, **A100**, **RTX Pro 6000**, and similar hardware; adjust PyTorch / flash-attn / CUDA-related pins as needed (see `requirements/torch-cu118.txt` and options below).

## One command (recommended)

From the repository root:

```bash
bash scripts/setup_env.sh
```

This creates conda env **`prism`** (if missing), installs **torch 2.8 + cu128**, training/eval deps, flash-attn, and `pip install -e .`.

Optional:

```bash
# Prebuilt flash-attn wheel (5090 / torch2.8)
FLASH_ATTN_WHEEL=/path/to/flash_attn-2.8.3+cu12torch2.8....whl bash scripts/setup_env.sh

# Legacy A100 stack (torch 2.0.1 + cu118)
TORCH_REQUIREMENTS=requirements/torch-cu118.txt bash scripts/setup_env.sh

# Skip flash-attn
SKIP_FLASH_ATTN=1 bash scripts/setup_env.sh
```

Activate:

```bash
conda activate prism
```

## Files

| File | Purpose |
|------|---------|
| `requirements/torch.txt` | **Default**: torch 2.8.0 + cu128 (5090 / Blackwell) |
| `requirements/torch-cu118.txt` | Legacy A100: torch 2.0.1 + cu118 |
| `requirements/base.txt` | Core LLaVA / PRISM pins |
| `requirements/train.txt` | DeepSpeed + training extras |
| `requirements/eval.txt` | COCO caption metrics |
| `requirements/flash-attn.txt` | FlashAttention 2.8.3 |
| `requirements/exported-zh.lock` | Full `pip freeze` snapshot from a working env |
| `requirements.txt` | Aggregator (torch + train + eval + flash) |

## Manual install

```bash
conda create -n prism python=3.10 -y
conda activate prism
pip install -r requirements.txt
pip install -e .
```

## Pinned versions (baseline)

| Component | Version | Notes |
|-----------|---------|--------|
| PyTorch | 2.8.0+cu128 | Required for RTX 5090 (CUDA 12.8+) |
| transformers | 4.31.0 | In-repo LLaVA code |
| deepspeed | 0.18.3 | Multi-GPU via `run.py` |
| bitsandbytes | 0.49.0 | 4/8-bit loading |
| flash-attn | 2.8.3 | Optional; use wheel on 5090 |
| numpy | 2.2.6 | |
| pydantic | 2.12.5 | Required by deepspeed 0.18.x |

The repo vendors a customized **PEFT** under `PEFT/`; do **not** replace it with a newer PyPI `peft` alone.

## Verify

```bash
conda activate prism
python -c "import torch; import transformers; import deepspeed; print(torch.__version__, transformers.__version__)"
python -c "from core.load_model import _resolve_train_cuda_device; print('core OK')"
```

## flash-attn monkey patch

If training hits `unpad_input` arity errors with flash-attn 2.8.x, patch
`backbone/shared/train/llama_flash_attn_monkey_patch.py` (see `5090环境(1).md`).

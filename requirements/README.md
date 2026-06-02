# PRISM environment setup

Python **3.10+** and a **CUDA-capable GPU** are required for training. Inference can run on a single GPU.

## Quick install (pip)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) PyTorch (edit requirements/torch.txt for cu118 vs cu121)
pip install -r requirements/torch.txt

# 2) Training stack (recommended for run.py train)
pip install -r requirements/train.txt

# 3) Evaluation metrics (caption benchmarks)
pip install -r requirements/eval.txt
```

Minimal install (inference / development without DeepSpeed):

```bash
pip install -r requirements/torch.txt
pip install -r requirements/base.txt
```

Editable install (uses `pyproject.toml` metadata):

```bash
pip install -r requirements/torch.txt
pip install -e ".[train,eval]"
```

## Conda

```bash
conda env create -f environment.yml
conda activate prism
```

Then install PyTorch if the env file’s CUDA channel does not match your machine (see comments in `environment.yml`).

## Optional: FlashAttention

Only needed for `backbone/shared/train/train_mem.py`:

```bash
pip install -r requirements/flash-attn.txt
```

Building `flash-attn` needs a matching GCC/nvcc toolchain; if the build fails, use the default training entry (`train.py`) without FlashAttention.

## Version notes

| Component | Pinned version | Notes |
|-----------|----------------|--------|
| PyTorch | 2.0.1 | Align with LLaVA-1.5 checkpoints |
| transformers | 4.31.0 | Required by in-repo LLaVA code |
| peft | 0.4.0 | Vendored `PEFT/` extends upstream APIs |
| deepspeed | 0.10.3 | Used by `run.py` multi-GPU training |
| bitsandbytes | 0.41.0 | 4/8-bit loading paths |

The repo vendors a customized **PEFT** under `PEFT/`; do **not** replace it with a newer PyPI `peft` alone.

## Verify

```bash
python -c "import torch; import transformers; import deepspeed; print(torch.__version__, transformers.__version__)"
python -c "from core.load_model import _resolve_train_cuda_device; print('core OK')"
```

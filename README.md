# CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

<div align="center">
    <div>
        <a href='https://juntaotang.github.io/' target='_blank'>Jun-Tao Tang</a>&emsp;
        <a href='https://www.lamda.nju.edu.cn/wenzh/' target='_blank'>Zhen-Hao Xie</a>&emsp;
        <a href='https://hhdnp.github.io/' target='_blank'>Yu-Cheng Shi</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a>
    </div>
    <div>
    School of Artificial Intelligence, Nanjing University
    </div>
    <div>
    State Key Laboratory of Novel Software Technology, Nanjing University
    </div>
</div>

<div align="center">

  <a href="https://arxiv.org/abs/2606.02502">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv">
  </a>
  &nbsp;
  <a href="https://github.com/LAMDA-CL/Prism">
    <img src="https://img.shields.io/badge/Codebase-Prism-blue" alt="Prism">
  </a>

</div>

The code repository for "[CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2606.02502)" (EMNLP 2026 Main Conference Paper). If you use any content of this repo for your work, please cite the following bib entry:

```bibtex
@inproceedings{tang2026cram,
  title={CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning},
  author={Tang, Jun-Tao and Xie, Zhen-Hao and Shi, Yu-Cheng and Zhou, Da-Wei},
  booktitle={EMNLP},
  year={2026}
}
```

## Introduction

Multimodal Large Language Models (MLLMs) unify heterogeneous vision-language tasks under a shared generative framework via instruction tuning, yet real-world deployment demands continuous capability expansion, making Multimodal Continual Instruction Tuning (MCIT) essential. Existing methods either update all tasks with a shared parameter set or allocate dedicated modules for each new task. Shared updates force heterogeneous tasks to compete, causing forgetting of learned capabilities. Conversely, isolated expansion prevents interference but severely limits parameter efficiency over long task streams.

To address this dilemma, we propose **CRAM**. By isolating task-specific patterns into independent modules, CRAM mitigates catastrophic forgetting across tasks. Adaptive-rank instantiation identifies the capability gap between existing experts and new task demands, and dynamically allocates only the necessary parameters. Centroid-guided routing recognizes and activates existing experts, while an orthogonality penalty confines new updates to task-specific directions. Extensive experiments across diverse benchmarks consistently demonstrate its superiority over existing methods.

## Requirements

**Environment.** From the repository root:

```bash
bash scripts/setup_env.sh
conda activate prism
```

This creates a conda env named `prism` and installs PyTorch, DeepSpeed, and the remaining dependencies. See [`requirements/README.md`](requirements/README.md) if you need another CUDA stack.

**Pre-trained weights.** Download [LLaVA-v1.5-7B](https://github.com/haotian-liu/LLaVA) and [CLIP](https://github.com/openai/CLIP), then set the paths in `config/paths/llava_paths.py`.

**Datasets.** This repo uses the [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [TriGap](https://huggingface.co/datasets/JuntaoTang/TriGap) benchmarks. Point the image and instruction folders to your local copies in `config/benchmarks/`.

## Running

Edit GPUs and other run defaults in `config/run_config.py` if needed. Then:

```bash
python run.py train 0 1 2 3 4 5 --method cram --benchmark ucit
python run.py infer 0 1 2 3 4 5 --method cram --benchmark ucit
```

`0 1 2 3 4 5` are task indices. Training of task *k* resumes from the checkpoint of task *k*-1. Inference uses the last-task checkpoint specified in `config/run_config.py` (`checkpoint_task`).

## Results

We release the full model outputs on every UCIT task under [`results/llava/UCIT/cram/`](results/llava/UCIT/cram/).

## Acknowledgement

This implementation is built on [Prism](https://github.com/LAMDA-CL/Prism). We thank the Prism authors for the open-source MCIT infrastructure.

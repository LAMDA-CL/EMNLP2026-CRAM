#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import shlex
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent

from config.backbone.registry import (
    apply_checkpoint_suffix_to_task,
    get_backbone_id,
    get_checkpoint_suffix,
    get_default_conv_mode,
    get_paths,
    get_train_backbone_flags,
    resolve_backbone_id,
)

DEFAULT_CONV_MODE = get_default_conv_mode()


def _run_stamp() -> str:
    # month/day/hour/min to distinguish runs
    return datetime.now().strftime("%m-%d-%H-%M")


def _load_run_config() -> dict:
    try:
        from config import run_config  # type: ignore

        return {
            "TRAIN_DEFAULTS": getattr(run_config, "TRAIN_DEFAULTS", {}) or {},
            "TRAIN_FLAG_OVERRIDES": getattr(run_config, "TRAIN_FLAG_OVERRIDES", {}) or {},
            "TRAIN_EXTRA_ARGS": getattr(run_config, "TRAIN_EXTRA_ARGS", []) or [],
            "INFER_DEFAULTS": getattr(run_config, "INFER_DEFAULTS", {}) or {},
        }
    except Exception:
        return {
            "TRAIN_DEFAULTS": {},
            "TRAIN_FLAG_OVERRIDES": {},
            "TRAIN_EXTRA_ARGS": [],
            "INFER_DEFAULTS": {},
        }


def _load_method_config(method: str) -> dict:
    try:
        mod = __import__(f"config.methods.{method}", fromlist=["*"])
        return {
            "TRAIN_FLAG_OVERRIDES": getattr(mod, "TRAIN_FLAG_OVERRIDES", {}) or {},
            "TRAIN_EXTRA_ARGS": getattr(mod, "TRAIN_EXTRA_ARGS", []) or [],
            "INFER_DEFAULTS": getattr(mod, "INFER_DEFAULTS", {}) or {},
            "TRAIN_BATCH_SIZES": getattr(mod, "TRAIN_BATCH_SIZES", {}) or {},
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": getattr(
                mod, "TRAIN_GRADIENT_ACCUMULATION_STEPS", None
            ),
            "TRAIN_GLOBAL_BATCH_SIZE": getattr(mod, "TRAIN_GLOBAL_BATCH_SIZE", {}) or {},
        }
    except Exception:
        return {
            "TRAIN_FLAG_OVERRIDES": {},
            "TRAIN_EXTRA_ARGS": [],
            "INFER_DEFAULTS": {},
            "TRAIN_BATCH_SIZES": {},
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": None,
            "TRAIN_GLOBAL_BATCH_SIZE": {},
        }


def _count_gpus(gpus: str) -> int:
    ids = [p.strip() for p in str(gpus).split(",") if p.strip()]
    return max(1, len(ids))


def _lookup_benchmark_task_value(
    table: int | dict | None,
    *,
    benchmark: str,
    task_id: int,
) -> int | None:
    """Resolve scalar or ``{benchmark: int|{task_id: int}}`` tables."""
    if table is None:
        return None
    if isinstance(table, int):
        return int(table)
    if not isinstance(table, dict):
        return None
    raw = table.get(benchmark)
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, dict):
        val = raw.get(task_id)
        return int(val) if val is not None else None
    return None


def _resolve_gradient_accumulation_steps(
    *,
    gas_spec: int | dict | None,
    global_batch_spec: int | dict | None,
    benchmark: str,
    task_id: int,
    per_device_batch_size: int,
    num_gpus: int,
    task_fallback_global_batch: int | None,
    method: str,
) -> int:
    """
    Resolve ``--gradient_accumulation_steps`` from method config.

    - Direct int (>0): use as-is.
    - ``-1``: ``global_batch // actual_batch`` where
      ``actual_batch = per_device_train_batch_size * num_gpus``.
    """
    gas = _lookup_benchmark_task_value(gas_spec, benchmark=benchmark, task_id=task_id)
    if gas is None:
        gas = 1
    if gas != -1:
        if gas < 1:
            raise SystemExit(
                f"Invalid TRAIN_GRADIENT_ACCUMULATION_STEPS={gas!r} for "
                f"benchmark={benchmark!r} task_id={task_id!r} (method={method!r}). "
                "Use a positive integer or -1 for auto."
            )
        return int(gas)

    global_batch = _lookup_benchmark_task_value(
        global_batch_spec, benchmark=benchmark, task_id=task_id
    )
    if global_batch is None:
        global_batch = task_fallback_global_batch
    if global_batch is None:
        raise SystemExit(
            f"TRAIN_GRADIENT_ACCUMULATION_STEPS=-1 requires TRAIN_GLOBAL_BATCH_SIZE in "
            f"config/methods/{method}.py or 'batch_size' on the benchmark task "
            f"(benchmark={benchmark!r} task_id={task_id!r})."
        )

    actual_batch = int(per_device_batch_size) * int(num_gpus)
    if actual_batch <= 0:
        raise SystemExit(
            f"Invalid actual batch size {actual_batch} "
            f"(per_device={per_device_batch_size}, gpus={num_gpus})."
        )
    if global_batch % actual_batch != 0:
        raise SystemExit(
            f"TRAIN_GLOBAL_BATCH_SIZE={global_batch} is not divisible by actual batch "
            f"{actual_batch} (= per_device {per_device_batch_size} × {num_gpus} gpus) "
            f"for benchmark={benchmark!r} task_id={task_id!r} (method={method!r})."
        )
    return global_batch // actual_batch


def _parse_task_ids(values: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for v in values:
        try:
            ids.append(int(v))
        except ValueError as e:
            raise SystemExit(f"Invalid task id: {v}") from e
    if not ids:
        raise SystemExit("No tasks specified.")
    return ids


def _benchmark_dir_name(benchmark: str) -> str:
    if benchmark.lower() == "coin":
        return "CoIN"
    elif benchmark.lower() == "ucit":
        return "UCIT"
    else:
        return benchmark


def _method_dir_name(method: str) -> str:
    # keep dir names stable and simple
    return str(method).strip().lower()


def _infer_result_dir(
    paths: dict, *, benchmark: str, method: str, task_name: str, stage: str, backbone_id: str
) -> Path:
    """
    Inference / eval result directory::

        RESULT_DIR/<BACKBONE_ID>/<Benchmark>/<method>/<dataset_task>/<stage>/
    """
    return (
        Path(paths["RESULT_DIR"])
        / get_backbone_id(backbone_id)
        / _benchmark_dir_name(benchmark)
        / _method_dir_name(method)
        / str(task_name).strip()
        / str(stage).strip()
    )


def _method_checkpoint_path(checkpoint_dir: str, benchmark: str, method: str, ckpt_name: str) -> Path:
    """
    New layout:
      CHECKPOINT_DIR/<Benchmark>/<method>/<TaskX_suffix>

    Backward compatible with old layout:
      CHECKPOINT_DIR/<Benchmark>/<TaskX_suffix>
    """
    benchmark_dir = _benchmark_dir_name(benchmark)
    method_dir = _method_dir_name(method)
    return Path(checkpoint_dir) / benchmark_dir / method_dir / ckpt_name


def _legacy_checkpoint_path(checkpoint_dir: str, benchmark: str, ckpt_name: str) -> Path:
    benchmark_dir = _benchmark_dir_name(benchmark)
    return Path(checkpoint_dir) / benchmark_dir / ckpt_name


def _resolve_paths(backbone: str | None = None) -> dict[str, str]:
    return get_paths(backbone)


def _resolve_backbone_context(backbone: str | None = None) -> tuple[str, dict[str, str]]:
    bid = resolve_backbone_id(backbone)
    paths = get_paths(bid)
    return bid, paths


def _resolve_method_checkpoint(
    checkpoint_dir: str, benchmark: str, method: str, ckpt_basename: str, *, suffix: str | None = None
) -> Path:
    """Pick first existing checkpoint among method layout + legacy layout."""
    active_suffix = suffix or get_checkpoint_suffix()
    candidates: list[Path] = []
    candidates.append(_method_checkpoint_path(checkpoint_dir, benchmark, method, ckpt_basename))
    if ckpt_basename.endswith(f"{active_suffix}_lora"):
        strip_lora = ckpt_basename[: -len("_lora")]
        candidates.append(_method_checkpoint_path(checkpoint_dir, benchmark, method, strip_lora))
    elif ckpt_basename.endswith(active_suffix) and "_lora" not in ckpt_basename:
        legacy_name = ckpt_basename.replace(active_suffix, f"{active_suffix}_lora")
        candidates.append(_method_checkpoint_path(checkpoint_dir, benchmark, method, legacy_name))
    candidates.append(_legacy_checkpoint_path(checkpoint_dir, benchmark, ckpt_basename))
    if ckpt_basename.endswith(f"{active_suffix}_lora"):
        strip_lora = ckpt_basename[: -len("_lora")]
        candidates.append(_legacy_checkpoint_path(checkpoint_dir, benchmark, strip_lora))
    elif ckpt_basename.endswith(active_suffix) and "_lora" not in ckpt_basename:
        legacy_name = ckpt_basename.replace(active_suffix, f"{active_suffix}_lora")
        candidates.append(_legacy_checkpoint_path(checkpoint_dir, benchmark, legacy_name))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Checkpoint path does not exist. Tried:\n  "
        + "\n  ".join(str(p) for p in unique)
        + f"\n(benchmark={benchmark!r}, method={method!r}, basename={ckpt_basename!r})"
    )


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _log_line(msg: str, *, log_file: Path, mirror: bool, lock: threading.Lock | None = None) -> None:
    _ensure_parent(log_file)
    line = msg if msg.endswith("\n") else msg + "\n"
    if lock:
        with lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)
            if mirror:
                sys.stdout.write(line)
                sys.stdout.flush()
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
        if mirror:
            sys.stdout.write(line)
            sys.stdout.flush()


def _apply_flag_overrides(cmd: list[str], overrides: dict) -> list[str]:
    """
    Override flags in a flat argv list.

    - For "--flag value": replace value if flag exists, else append.
    - For value==True: ensure flag exists (no value).
    - For value==False/None: do nothing.
    """
    out = list(cmd)
    for flag, value in (overrides or {}).items():
        if value is False or value is None:
            continue
        if value is True:
            if flag not in out:
                out.append(str(flag))
            continue

        value_s = str(value)
        try:
            i = out.index(str(flag))
        except ValueError:
            out.extend([str(flag), value_s])
            continue

        if i + 1 >= len(out):
            out.append(value_s)
        else:
            out[i + 1] = value_s
    return out


def _append_args(cmd: list[str], extra: list[str]) -> list[str]:
    if not extra:
        return cmd
    return [*cmd, *[str(x) for x in extra]]


def _tee_run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_file: Path, mirror: bool) -> int:
    """
    Run a command and tee *all* stdout/stderr into log_file.

    For multi-process launchers (e.g. deepspeed), piping at the OS level is more reliable
    than capturing via Python's stdout=PIPE.
    """
    _ensure_parent(log_file)
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    log_str = shlex.quote(str(log_file))
    if mirror:
        bash_cmd = f"set -o pipefail; ({cmd_str}) 2>&1 | tee -a {log_str}"
    else:
        bash_cmd = f"set -o pipefail; ({cmd_str}) >> {log_str} 2>&1"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("COMMAND: " + cmd_str + "\n\n")

    proc = subprocess.run(
        ["bash", "-lc", bash_cmd],
        cwd=str(cwd),
        env=env,
    )
    return int(proc.returncode)


def _stream_run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_file: Path,
    lock: threading.Lock | None = None,
    mirror: bool = False,
) -> int:
    """
    Stream a subprocess' stdout/stderr to both console and log_file.

    This is used for python inference/eval subprocesses where stdout=PIPE is reliable.
    """
    _ensure_parent(log_file)
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + cmd_str + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if lock:
                with lock:
                    if mirror:
                        sys.stdout.write(line)
                    f.write(line)
                    f.flush()
            else:
                if mirror:
                    sys.stdout.write(line)
                f.write(line)
                f.flush()
        return int(proc.wait())


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    # Sanitize thread env vars to avoid libgomp warnings.
    def _sanitize_threads(name: str, default: str = "1") -> None:
        v = env.get(name)
        if v is None or v == "":
            env[name] = default
            return
        if not str(v).strip().isdigit():
            env[name] = default

    _sanitize_threads("OMP_NUM_THREADS", "8")
    _sanitize_threads("MKL_NUM_THREADS", env.get("OMP_NUM_THREADS", "8"))
    return env


def _get_task_config(benchmark: str, task_id: int, *, use_sub_dataset: bool, backbone: str | None = None) -> dict:
    from config.benchmarks import BENCHMARKS  # type: ignore
    from utils.sub_dataset import apply_use_sub_dataset_to_task  # type: ignore

    if benchmark not in BENCHMARKS:
        raise SystemExit(f"Benchmark '{benchmark}' not found. Available: {list(BENCHMARKS.keys())}")
    tasks = BENCHMARKS[benchmark]
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"Task {task_id} not in benchmark '{benchmark}' (0-{len(tasks)-1})")
    raw = dict(tasks[task_id])
    raw = apply_checkpoint_suffix_to_task(raw, get_checkpoint_suffix(backbone))
    raw = apply_use_sub_dataset_to_task(raw, use_sub_dataset=use_sub_dataset, benchmark=benchmark)
    if raw.get("cur_task") == 0 and raw.get("previous_task") is None:
        raw["pretrain_mm_mlp_adapter"] = get_paths(backbone)["PRETRAIN_MM_PROJECTOR"]
    return raw


def _task_image_folder(task: dict) -> str:
    folder = task.get("image_folder")
    if not folder:
        raise SystemExit(
            f"Task {task.get('name', task.get('cur_task'))!r} missing 'image_folder'. "
            "Set it in config/benchmarks/<benchmark>.py."
        )
    return str(folder)


def _build_train_command(
    task: dict,
    *,
    gpus: str,
    port: int,
    method: str,
    paths: dict,
    batch_size: int,
    gradient_accumulation_steps: int,
    benchmark: str,
    backbone: str,
    train_precision: str | None = None,
) -> list[str]:
    from config.backbone.registry import train_precision_cli_args
    from config.benchmarks import BENCHMARK_TASK_NUM  # type: ignore

    bm = (benchmark or "coin").strip().lower()
    task_num = BENCHMARK_TASK_NUM.get(bm)
    if task_num is None:
        raise SystemExit(f"Unknown benchmark {bm!r} for task_num (not in BENCHMARK_TASK_NUM).")

    bb_flags = get_train_backbone_flags(backbone)

    cmd = [
        "deepspeed",
        f"--include=localhost:{gpus}",
        f"--master_port={port}",
        "backbone/shared/train/train_mem.py",
        "--deepspeed",
        paths["DEEPSPEED_CONFIG"],
        "--lora_enable",
        "True",
        "--backbone",
        resolve_backbone_id(backbone),
        "--benchmark",
        bm,
        "--task_num",
        str(int(task_num)),
        "--model_name_or_path",
        paths["BASE_MODEL_PATH"],
        "--freeze_mm_mlp_adapter",
        "True",
        "--version",
        bb_flags.get("version", "v1"),
        "--data_path",
        str(task["train_data_path"]),
        "--image_folder",
        _task_image_folder(task),
        "--vision_tower",
        paths["VISION_TOWER_PATH"],
        "--routing_vision_tower",
        paths["ROUTING_VISION_TOWER_PATH"],
        "--text_tower",
        paths["CLIP_PATH"],
        "--cur_task",
        str(task["cur_task"]),
        "--mm_projector_type",
        bb_flags.get("mm_projector_type", "mlp2x_gelu"),
        "--mm_vision_select_layer",
        bb_flags.get("mm_vision_select_layer", "-2"),
        "--mm_use_im_start_end",
        bb_flags.get("mm_use_im_start_end", "False"),
        "--mm_use_im_patch_token",
        bb_flags.get("mm_use_im_patch_token", "False"),
        "--image_aspect_ratio",
        bb_flags.get("image_aspect_ratio", "pad"),
        "--group_by_modality_length",
        "True",
        *train_precision_cli_args(precision=train_precision),
        "--output_dir",
        str(task["output_dir"]),
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        str(int(batch_size)),
        "--per_device_eval_batch_size",
        str(int(batch_size)),
        "--gradient_accumulation_steps",
        str(int(gradient_accumulation_steps)),
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "epoch",
        "--learning_rate",
        bb_flags.get("learning_rate", "2e-4"),
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--tf32",
        "True",
        "--model_max_length",
        "2048",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "4",
        "--lazy_preprocess",
        "True",
        "--report_to",
        "none",
        "--method",
        method,
    ]

    # Task0 uses pretrain projector; others load previous task checkpoint
    if "pretrain_mm_mlp_adapter" in task:
        cmd.extend(["--pretrain_mm_mlp_adapter", str(task["pretrain_mm_mlp_adapter"])])
    elif task.get("previous_task"):
        cmd.extend(["--previous_task_model_path", str(task["previous_task"])])
    else:
        raise SystemExit("Task config missing 'pretrain_mm_mlp_adapter' or 'previous_task'.")

    return [str(x) for x in cmd]


# ===== Inference/Evaluation (ported from tools/eval_task.py) =====
INFERENCE_METHOD_MAP = {"ScienceQA": "scienceqa"}
EVAL_TASK_MAP = {
    "ScienceQA": "scienceqa",
    "TextVQA": "textvqa",
    "ImageNet": "imagenet",
    "GQA": "gqa",
    "VizWiz": "vizwiz",
    "Grounding": "grounding",
    "VQAv2": "vqav2",
    "OCRVQA": "ocrvqa",
    "ImageNet-R": "imagenetr",
    "ArxivQA": "arxivqa",
    "IconQA": "iconqa",
    "CLEVR": "clevr",
    "Flickr30k": "flickr30k",
    "Vizcap": "vizcap",
    "ChartQA": "chartqa",
    "DocVQA": "docvqa",
    "InfographicVQA": "infographicvqa",
    "PMCVQA": "pmcvqa",
    "Roadside": "roadside",
    "ChemVQA": "chemvqa",
    "FloodNetVQA": "floodnetvqa",
}


def _format_args(template: list, **kwargs) -> list[str]:
    formatted: list[str] = []
    for arg in template:
        if isinstance(arg, str) and "{" in arg and "}" in arg:
            try:
                formatted.append(arg.format(**{k: str(v) for k, v in kwargs.items()}))
            except KeyError:
                formatted.append(arg)
        else:
            formatted.append(str(arg))
    return formatted


def _append_infer_backbone_args(
    cmd: list[str],
    paths: dict,
    backbone_id: str,
    *,
    train_method: str | None = None,
    load_4bit: bool | None = None,
    load_8bit: bool | None = None,
) -> list[str]:
    from config.backbone.registry import infer_precision_flags
    from utils.infer_paths import (
        resolve_method_infer_tower_paths,
        should_load_routing_vision_tower,
    )

    bid = resolve_backbone_id(backbone_id)
    cmd.extend(["--backbone", bid])
    mllm_vision = paths.get("VISION_TOWER_PATH")
    if mllm_vision:
        cmd.extend(["--vision-tower", mllm_vision])

    method_paths = resolve_method_infer_tower_paths(train_method)
    routing = method_paths.get("routing_vision_tower")
    if routing and should_load_routing_vision_tower(routing, mllm_vision):
        cmd.extend(["--routing-vision-tower", routing])
    text = method_paths.get("text_tower")
    if text:
        cmd.extend(["--text-tower", text])

    use_8bit, use_4bit, _ = infer_precision_flags(load_4bit=load_4bit, load_8bit=load_8bit)
    if use_4bit:
        cmd.append("--load-4bit")
    elif use_8bit:
        cmd.append("--load-8bit")
    return cmd


def _run_inference_one_chunk(
    task: dict,
    *,
    model_path: str,
    paths: dict,
    gpu_id: int,
    chunks: int,
    chunk_idx: int,
    output_file: Path,
    train_method: str,
    temperature: str,
    conv_mode: str,
    batch_size: int,
    benchmark: str,
    backbone_id: str,
    load_4bit: bool | None = None,
    load_8bit: bool | None = None,
):
    eval_config = task["eval"]
    ui_method = INFERENCE_METHOD_MAP.get(task["name"], "default")

    cmd = [
        sys.executable,
        "-m",
        "backbone.shared.eval.model_unified",
        ui_method,
        "--method",
        str(train_method),
        "--model-path",
        str(model_path),
        "--model-base",
        paths["BASE_MODEL_PATH"],
        "--question-file",
        str(task["test_data_path"]),
        "--image-folder",
        _task_image_folder(task),
        "--answers-file",
        str(output_file),
        "--batch-size",
        str(int(batch_size)),
        "--num-chunks",
        str(chunks),
        "--chunk-idx",
        str(chunk_idx),
        "--temperature",
        str(temperature),
        "--conv-mode",
        str(conv_mode),
        "--benchmark",
        str(benchmark),
    ]
    cmd = _append_infer_backbone_args(
        cmd,
        paths,
        backbone_id,
        train_method=train_method,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
    )

    if eval_config.get("inference_args"):
        cmd.extend(_format_args(eval_config["inference_args"], **task))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Note: output capture happens at caller level via _stream_run.
    subprocess.run([str(x) for x in cmd], env=env, cwd=str(PROJECT_ROOT), check=True)


def _run_inference_parallel(
    task: dict,
    *,
    model_path: str,
    paths: dict,
    gpus: list[int],
    result_dir: Path,
    train_method: str,
    temperature: str,
    conv_mode: str,
    batch_size: int,
    benchmark: str,
    backbone_id: str,
    load_4bit: bool | None = None,
    load_8bit: bool | None = None,
):
    chunks = len(gpus)
    print_lock = threading.Lock()
    failed: list[int] = []

    def run_chunk(idx: int) -> bool:
        output_file = result_dir / f"{chunks}_{idx}.jsonl"
        try:
            # Build the same command as _run_inference_one_chunk but stream its output.
            eval_config = task["eval"]
            ui_method = INFERENCE_METHOD_MAP.get(task["name"], "default")
            cmd = [
                sys.executable,
                "-m",
                "backbone.shared.eval.model_unified",
                ui_method,
                "--method",
                str(train_method),
                "--model-path",
                str(model_path),
                "--model-base",
                paths["BASE_MODEL_PATH"],
                "--question-file",
                str(task["test_data_path"]),
                "--image-folder",
                _task_image_folder(task),
                "--answers-file",
                str(output_file),
                "--batch-size",
                str(int(batch_size)),
                "--num-chunks",
                str(chunks),
                "--chunk-idx",
                str(idx),
                "--temperature",
                str(temperature),
                "--conv-mode",
                str(conv_mode),
                "--benchmark",
                str(benchmark),
            ]
            cmd = _append_infer_backbone_args(
                cmd,
                paths,
                backbone_id,
                train_method=train_method,
                load_4bit=load_4bit,
                load_8bit=load_8bit,
            )
            if eval_config.get("inference_args"):
                cmd.extend(_format_args(eval_config["inference_args"], **task))

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[idx])

            rc = _stream_run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                log_file=result_dir / "_infer_process_log.txt",
                lock=print_lock,
            )
            if rc != 0:
                raise RuntimeError(f"inference subprocess failed with code {rc}")
            with print_lock:
                print(f"  Chunk {idx} completed on GPU {gpus[idx]}")
            return True
        except Exception as e:
            with print_lock:
                print(f"  Chunk {idx} failed on GPU {gpus[idx]}: {e}")
            return False

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=chunks) as ex:
        futs = {ex.submit(run_chunk, i): i for i in range(chunks)}
        for fut in concurrent.futures.as_completed(futs):
            idx = futs[fut]
            try:
                ok = fut.result()
                if not ok:
                    failed.append(idx)
            except Exception:
                failed.append(idx)

    if failed:
        raise RuntimeError(f"Failed chunks: {failed}")


def _run_evaluation(task: dict, *, result_file: Path, output_dir: Path):
    eval_config = task["eval"]
    eval_task = EVAL_TASK_MAP.get(task["name"])
    if eval_task is None:
        raise SystemExit(f"No evaluation task mapping for {task['name']}")

    cmd = [sys.executable, "-m", "backbone.shared.eval.eval_unified", eval_task]
    if eval_task != "gqa":
        cmd.extend(["--result-file", str(result_file), "--output-dir", str(output_dir)])

    if eval_config.get("eval_args"):
        task_copy = dict(task)
        task_copy.pop("output_dir", None)
        task_copy.pop("result_file", None)
        cmd.extend(
            _format_args(
                eval_config["eval_args"],
                result_file=str(result_file),
                output_dir=str(output_dir),
                **task_copy,
            )
        )

    print("Evaluation command:", " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], cwd=str(PROJECT_ROOT), check=True)


def _infer_y_label(model_path: str, checkpoint_task: str) -> str:
    return checkpoint_task if model_path == "" else "custom"


def cmd_train(args: argparse.Namespace) -> int:
    if str(getattr(args, "method", "") or "").strip().lower() == "zeroshot":
        raise SystemExit(
            "method 'zeroshot' is inference-only (plain LLaVA baseline, no CL training). "
            "Use infer/eval with --method zeroshot, or choose another --method for train."
        )
    # benchmark configs are imported in *this* process
    env = _build_env()
    # Default logging level for subprocesses
    env["PYPRISM_LOG_LEVEL"] = "DEBUG" if bool(args.debug) else "TRAIN"
    backbone_id, paths = _resolve_backbone_context(getattr(args, "backbone", None))
    ckpt_suffix = get_checkpoint_suffix(backbone_id)
    task_ids = _parse_task_ids(args.tasks)
    stamp = _run_stamp()
    cfg = _load_run_config()
    method_cfg = _load_method_config(args.method)
    mirror = False

    flag_overrides = dict(cfg.get("TRAIN_FLAG_OVERRIDES", {}))
    flag_overrides.update(method_cfg.get("TRAIN_FLAG_OVERRIDES", {}))
    extra_args = list(cfg.get("TRAIN_EXTRA_ARGS", [])) + list(method_cfg.get("TRAIN_EXTRA_ARGS", []))
    batch_sizes = method_cfg.get("TRAIN_BATCH_SIZES", {})
    gas_spec = method_cfg.get("TRAIN_GRADIENT_ACCUMULATION_STEPS")
    global_batch_spec = method_cfg.get("TRAIN_GLOBAL_BATCH_SIZE", {})
    num_gpus = _count_gpus(args.gpus)

    for task_id in task_ids:
        task = _get_task_config(
            args.benchmark, task_id, use_sub_dataset=bool(args.use_sub_dataset), backbone=backbone_id
        )
        bs = None
        bench_bs: dict = {}
        if isinstance(batch_sizes, dict):
            raw_bs = batch_sizes.get(args.benchmark)
            if isinstance(raw_bs, dict):
                bench_bs = raw_bs
                bs = bench_bs.get(task_id)
        if bs is None:
            bs = task.get("batch_size")
        if bs is None:
            try:
                from config.benchmarks import BENCHMARKS

                n_tasks = len(BENCHMARKS.get(args.benchmark, []))
            except Exception:
                n_tasks = 0
            keys_hint = sorted(bench_bs.keys()) if bench_bs else []
            idx_range = f"0..{n_tasks - 1}" if n_tasks > 0 else "?"
            raise SystemExit(
                f"Missing batch_size for benchmark={args.benchmark!r} task_id={task_id!r} "
                f"(method={args.method!r}). "
                f"Define TRAIN_BATCH_SIZES['{args.benchmark}'][{task_id}] in "
                f"config/methods/{args.method}.py "
                f"(this benchmark has {n_tasks} tasks, indices {idx_range}; "
                f"configured keys for this benchmark: {keys_hint}). "
                f"Alternatively set 'batch_size' on the task in config/benchmarks/*."
            )
        # ==== Method-aware checkpoint layout ====
        # Override task output_dir / previous_task to include method subdir.
        ckpt_root = paths["CHECKPOINT_DIR"]
        ckpt_name = f"Task{task['cur_task']}{ckpt_suffix}"
        new_out = _method_checkpoint_path(ckpt_root, args.benchmark, args.method, ckpt_name)
        task = dict(task)
        task["output_dir"] = str(new_out)

        if task.get("previous_task"):
            prev_name = Path(str(task["previous_task"])).name
            task["previous_task"] = str(
                _resolve_method_checkpoint(
                    ckpt_root, args.benchmark, args.method, prev_name, suffix=ckpt_suffix
                )
            )
        # ==============================

        # output/<backbone>/train/<benchmark>/<method>/taskXX/run_*.txt
        log_path = (
            PROJECT_ROOT
            / "output"
            / get_backbone_id(backbone_id)
            / "train"
            / args.benchmark
            / args.method
            / f"task{task_id:02d}"
            / f"run_{stamp}.txt"
        )

        # Write header to file; mirror only if requested
        from config.backbone.registry import resolve_train_precision

        train_prec = getattr(args, "train_precision", None) or resolve_train_precision()
        grad_acc = _resolve_gradient_accumulation_steps(
            gas_spec=gas_spec,
            global_batch_spec=global_batch_spec,
            benchmark=args.benchmark,
            task_id=task_id,
            per_device_batch_size=int(bs),
            num_gpus=num_gpus,
            task_fallback_global_batch=task.get("batch_size"),
            method=args.method,
        )
        actual_batch = int(bs) * num_gpus
        global_batch = actual_batch * grad_acc

        _log_line(f"TRAIN method={args.method} benchmark={args.benchmark} task={task_id}", log_file=log_path, mirror=mirror)
        _log_line(f"TRAIN_PRECISION={train_prec} (config/run_config.TRAIN_PRECISION)", log_file=log_path, mirror=mirror)
        _log_line(
            f"per_device_batch={int(bs)} gpus={num_gpus} grad_acc={grad_acc} "
            f"actual_batch={actual_batch} global_batch={global_batch}",
            log_file=log_path,
            mirror=mirror,
        )
        _log_line(f"log: {log_path}", log_file=log_path, mirror=mirror)

        try:
            cmd = _build_train_command(
                task,
                gpus=args.gpus,
                port=args.port,
                method=args.method,
                paths=paths,
                batch_size=int(bs),
                gradient_accumulation_steps=grad_acc,
                benchmark=args.benchmark,
                backbone=backbone_id,
                train_precision=train_prec,
            )
            cmd = _apply_flag_overrides(cmd, flag_overrides)
            cmd = _append_args(cmd, extra_args)

            rc = _tee_run(cmd, cwd=PROJECT_ROOT, env=env, log_file=log_path, mirror=mirror)
            if rc != 0:
                return rc
        except Exception as e:
            import traceback

            _log_line("\nFailed before launching training subprocess.", log_file=log_path, mirror=mirror)
            _log_line(f"Exception: {type(e).__name__}: {e}", log_file=log_path, mirror=mirror)
            tb = traceback.format_exc()
            for line in tb.splitlines():
                _log_line(line, log_file=log_path, mirror=mirror)
            return 1

    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    # benchmark configs are imported in *this* process
    env = _build_env()
    env["PYPRISM_LOG_LEVEL"] = "DEBUG" if bool(getattr(args, "debug", False)) else "INFER"
    backbone_id, paths = _resolve_backbone_context(getattr(args, "backbone", None))
    ckpt_suffix = get_checkpoint_suffix(backbone_id)
    if not getattr(args, "checkpoint_suffix", None) or args.checkpoint_suffix == "_llava":
        args.checkpoint_suffix = ckpt_suffix
    task_ids = _parse_task_ids(args.tasks)
    stamp = _run_stamp()
    mirror = False

    method_cfg = _load_method_config(args.method)
    _is_zeroshot_infer = str(getattr(args, "method", "") or "").strip().lower() == "zeroshot"

    # Inference batch size (for backbone.shared.eval.model_unified -> InferenceEngine)
    infer_bs = getattr(args, "batch_size", None)
    if infer_bs is None:
        infer_bs = method_cfg.get("INFER_DEFAULTS", {}).get("batch_size", 1)
    infer_bs = int(infer_bs)

    if _is_zeroshot_infer:
        # Zeroshot: no checkpoint path; weights only from BASE_MODEL_PATH (--model-base)
        model_path = ""
        checkpoint_task = args.checkpoint_task
    elif args.model_path:
        model_path = args.model_path
        checkpoint_task = args.checkpoint_task
    else:
        checkpoint_dir = paths["CHECKPOINT_DIR"]
        ckpt_name = f"Task{args.checkpoint_task}{args.checkpoint_suffix}"
        model_path = str(
            _resolve_method_checkpoint(
                checkpoint_dir, args.benchmark, args.method, ckpt_name, suffix=ckpt_suffix
            )
        )
        checkpoint_task = args.checkpoint_task

    y = _infer_y_label(args.model_path, checkpoint_task)
    failed: list[int] = []

    _infer_display_model = str(paths["BASE_MODEL_PATH"]) if _is_zeroshot_infer else model_path

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("Invalid --gpus. Example: --gpus 0,1")

    for task_id in task_ids:
        task = _get_task_config(
            args.benchmark, task_id, use_sub_dataset=bool(args.use_sub_dataset), backbone=backbone_id
        )

        # output/<backbone>/infer/<benchmark>/<method>/
        # Zeroshot has no source-task checkpoint; name logs as to_<eval_task> only.
        _infer_log_stem = f"to_{task_id}_{stamp}" if _is_zeroshot_infer else f"{y}_to_{task_id}_{stamp}"
        log_path = (
            PROJECT_ROOT
            / "output"
            / get_backbone_id(backbone_id)
            / "infer"
            / args.benchmark
            / args.method
            / f"{_infer_log_stem}.txt"
        )

        def _do_infer_eval() -> int:
            result_dir = _infer_result_dir(
                paths,
                benchmark=args.benchmark,
                method=args.method,
                task_name=task["name"],
                stage=args.stage,
                backbone_id=backbone_id,
            )
            result_dir.mkdir(parents=True, exist_ok=True)

            print("Running inference in parallel...")
            _run_inference_parallel(
                task,
                model_path=model_path,
                paths=paths,
                gpus=gpus,
                result_dir=result_dir,
                train_method=args.method,
                temperature=args.temperature,
                conv_mode=args.conv_mode,
                batch_size=infer_bs,
                benchmark=args.benchmark,
                backbone_id=backbone_id,
                load_4bit=getattr(args, "load_4bit", None),
                load_8bit=getattr(args, "load_8bit", None),
            )

            print("\nMerging results...")
            merged_file = result_dir / "merge.jsonl"
            with open(merged_file, "w", encoding="utf-8") as out:
                chunks = len(gpus)
                for idx in range(chunks):
                    chunk_file = result_dir / f"{chunks}_{idx}.jsonl"
                    if chunk_file.exists():
                        out.write(chunk_file.read_text(encoding="utf-8"))
                    else:
                        print(f"Warning: {chunk_file} not found")

            print("\nRunning evaluation...")
            _run_evaluation(task, result_file=merged_file, output_dir=result_dir)
            return 0

        _ensure_parent(log_path)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {_infer_display_model}\n")
            f.write(f"BENCHMARK: {args.benchmark}\n")
            f.write(f"TASK: {task_id}\n")
            f.write(f"GPUS: {args.gpus}\n")
            f.write(f"STAGE: {args.stage}\n\n")

        # Dedicated lock for this task's logging (avoid interleaving across chunks).
        # Use RLock because some logging helpers may be called in contexts that already hold the lock.
        log_lock = threading.RLock()

        try:
            _log_line("=" * 60, log_file=log_path, mirror=mirror, lock=log_lock)
            _infer_hdr = (
                f"INFER benchmark={args.benchmark} method={args.method} -> x={task_id}"
                if _is_zeroshot_infer
                else f"INFER benchmark={args.benchmark} method={args.method} y={y} -> x={task_id}"
            )
            _log_line(
                _infer_hdr,
                log_file=log_path,
                mirror=mirror,
                lock=log_lock,
            )
            _log_line(f"model: {_infer_display_model}", log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line(f"log: {log_path}", log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line(f"infer batch_size={infer_bs}", log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line("=" * 60, log_file=log_path, mirror=mirror, lock=log_lock)

            result_dir = _infer_result_dir(
                paths,
                benchmark=args.benchmark,
                method=args.method,
                task_name=task["name"],
                stage=args.stage,
                backbone_id=backbone_id,
            )
            result_dir.mkdir(parents=True, exist_ok=True)

            print("Running inference in parallel...")
            # Stream each chunk subprocess into the same log file
            # by pointing the per-chunk streaming log to `log_path`.
            def _run_infer_chunks():
                chunks = len(gpus)
                failed_chunks: list[int] = []

                def run_chunk(idx: int) -> bool:
                    output_file = result_dir / f"{chunks}_{idx}.jsonl"
                    try:
                        eval_config = task["eval"]
                        ui_method = INFERENCE_METHOD_MAP.get(task["name"], "default")

                        cmd = [
                            sys.executable,
                            "-m",
                            "backbone.shared.eval.model_unified",
                            ui_method,
                            "--method",
                            str(args.method),
                            "--model-path",
                            str(model_path),
                            "--model-base",
                            paths["BASE_MODEL_PATH"],
                            "--question-file",
                            str(task["test_data_path"]),
                            "--image-folder",
                            _task_image_folder(task),
                            "--answers-file",
                            str(output_file),
                            "--batch-size",
                            str(int(infer_bs)),
                            "--num-chunks",
                            str(chunks),
                            "--chunk-idx",
                            str(idx),
                            "--temperature",
                            str(args.temperature),
                            "--conv-mode",
                            str(args.conv_mode),
                            "--benchmark",
                            str(args.benchmark),
                        ]
                        cmd = _append_infer_backbone_args(
                            cmd,
                            paths,
                            backbone_id,
                            train_method=args.method,
                            load_4bit=getattr(args, "load_4bit", None),
                            load_8bit=getattr(args, "load_8bit", None),
                        )
                        if eval_config.get("inference_args"):
                            cmd.extend(_format_args(eval_config["inference_args"], **task))

                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gpus[idx])
                        rc = _stream_run(cmd, cwd=PROJECT_ROOT, env=env, log_file=log_path, lock=log_lock, mirror=mirror)
                        return rc == 0
                    except Exception as e:
                        import traceback

                        _log_line(
                            f"  Chunk {idx} crashed before/while launching subprocess: {e}",
                            log_file=log_path,
                            mirror=mirror,
                            lock=log_lock,
                        )
                        tb = traceback.format_exc()
                        for line in tb.splitlines():
                            _log_line(f"    {line}", log_file=log_path, mirror=mirror, lock=log_lock)
                        return False

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
                    futs = {ex.submit(run_chunk, i): i for i in range(len(gpus))}
                    for fut in concurrent.futures.as_completed(futs):
                        idx = futs[fut]
                        ok = False
                        try:
                            ok = fut.result()
                        except Exception:
                            ok = False
                        if ok:
                            _log_line(
                                f"  Chunk {idx} completed on GPU {gpus[idx]}",
                                log_file=log_path,
                                mirror=mirror,
                                lock=log_lock,
                            )
                        else:
                            _log_line(
                                f"  Chunk {idx} failed on GPU {gpus[idx]}",
                                log_file=log_path,
                                mirror=mirror,
                                lock=log_lock,
                            )
                            failed_chunks.append(idx)

                if failed_chunks:
                    raise RuntimeError(f"Failed chunks: {failed_chunks}")

            _run_infer_chunks()

            _log_line("\nMerging results...", log_file=log_path, mirror=mirror, lock=log_lock)
            merged_file = result_dir / "merge.jsonl"
            with open(merged_file, "w", encoding="utf-8") as out:
                chunks = len(gpus)
                for idx in range(chunks):
                    chunk_file = result_dir / f"{chunks}_{idx}.jsonl"
                    if chunk_file.exists():
                        out.write(chunk_file.read_text(encoding="utf-8"))
                    else:
                        _log_line(f"Warning: {chunk_file} not found", log_file=log_path, mirror=mirror, lock=log_lock)

            _log_line("\nRunning evaluation...", log_file=log_path, mirror=mirror, lock=log_lock)
            eval_task = EVAL_TASK_MAP.get(task["name"])
            if eval_task is None:
                raise RuntimeError(f"No evaluation task mapping for {task['name']}")
            eval_cmd = [sys.executable, "-m", "backbone.shared.eval.eval_unified", eval_task]
            if eval_task != "gqa":
                eval_cmd.extend(["--result-file", str(merged_file), "--output-dir", str(result_dir)])
            if task["eval"].get("eval_args"):
                task_copy = dict(task)
                task_copy.pop("output_dir", None)
                task_copy.pop("result_file", None)
                eval_cmd.extend(
                    _format_args(
                        task["eval"]["eval_args"],
                        result_file=str(merged_file),
                        output_dir=str(result_dir),
                        **task_copy,
                    )
                )
            rc = _stream_run(eval_cmd, cwd=PROJECT_ROOT, env=os.environ.copy(), log_file=log_path, lock=log_lock, mirror=mirror)
            if rc != 0:
                raise RuntimeError(f"evaluation subprocess failed with code {rc}")

            rc = 0
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nTask {task_id} failed: {e}\n")
            if mirror:
                print(f"Task {task_id} failed: {e}")
            rc = 1

        if rc != 0:
            failed.append(task_id)

    if failed:
        if mirror:
            print(f"Failed tasks: {failed}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="run.py", description="Single entrypoint: train / infer")
    sub = parser.add_subparsers(dest="command", required=True)

    cfg = _load_run_config()

    p_train = sub.add_parser("train", help="Train tasks")
    p_train.add_argument("tasks", nargs="*", help="Task IDs to train (e.g. 0 1 2)")
    p_train.add_argument("--benchmark", default="coin")
    p_train.add_argument("--gpus", default="0,1")
    p_train.add_argument("--port", type=int, default=29601)
    p_train.add_argument("--debug", action="store_true")
    p_train.add_argument("--method", default="hide_llava")
    p_train.add_argument(
        "--backbone",
        default=None,
        choices=["llava", "internvl"],
        help="MLLM backbone (default: config/run_config.py BACKBONE_DEFAULT or llava)",
    )
    p_train.add_argument(
        "--use-sub-dataset",
        action=argparse.BooleanOptionalAction,
        help="UCIT: append _sub to *.json data paths (default: config/run_config.py TRAIN_DEFAULTS)",
    )
    p_train.add_argument(
        "--train-precision",
        default=None,
        choices=["bf16", "fp16", "8bit", "4bit"],
        help="Training precision (default: config/run_config.py TRAIN_PRECISION, currently 8bit QLoRA).",
    )
    p_train.set_defaults(**{k: v for k, v in cfg["TRAIN_DEFAULTS"].items() if k != "handler"})
    p_train.set_defaults(train_flag_overrides=cfg["TRAIN_FLAG_OVERRIDES"])
    p_train.set_defaults(train_extra_args=cfg["TRAIN_EXTRA_ARGS"])
    p_train.set_defaults(handler=cmd_train)

    p_infer = sub.add_parser("infer", help="Infer+eval tasks")
    p_infer.add_argument("tasks", nargs="*", help="Task IDs to evaluate (e.g. 0 1 2)")
    p_infer.add_argument("--benchmark", default="coin")
    p_infer.add_argument("--gpus", default="0,1")
    p_infer.add_argument("--checkpoint-task", default="7")
    p_infer.add_argument("--checkpoint-suffix", default="_llava")
    p_infer.add_argument("--model-path", default="")
    p_infer.add_argument("--stage", default="MoELoRA")
    p_infer.add_argument("--temperature", default="0")
    p_infer.add_argument("--conv-mode", default=DEFAULT_CONV_MODE)
    p_infer.add_argument("--method", default="hide_llava")
    p_infer.add_argument(
        "--backbone",
        default=None,
        choices=["llava", "internvl"],
        help="MLLM backbone (default: config/run_config.py BACKBONE_DEFAULT or llava)",
    )
    p_infer.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size for backbone.shared.eval.model_unified (overrides config/methods/<method>.py INFER_DEFAULTS.batch_size)",
    )
    p_infer.add_argument(
        "--use-sub-dataset",
        action=argparse.BooleanOptionalAction,
        help="UCIT: append _sub to *.json data paths (default: config/run_config.py INFER_DEFAULTS)",
    )
    p_infer.add_argument(
        "--load-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="4-bit LLM inference (overrides config/run_config.py INFER_PRECISION when set).",
    )
    p_infer.add_argument(
        "--load-8bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="8-bit LLM inference (overrides INFER_PRECISION; takes precedence over --load-4bit).",
    )
    p_infer.set_defaults(**{k: v for k, v in cfg["INFER_DEFAULTS"].items() if k != "handler"})
    p_infer.set_defaults(handler=cmd_infer)

    args = parser.parse_args()
    rc = int(args.handler(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()


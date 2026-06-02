from __future__ import annotations

import contextlib
import gc
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from config.backbone.constants import IGNORE_INDEX
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for key in ("input_ids", "labels"):
        v = batch.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
    return 1


def _split_batch_rows(
    batch: Dict[str, Any], start: int, end: int, batch_size: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
            out[k] = v[start:end]
        elif isinstance(v, list) and len(v) == batch_size:
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def _microbatch_chunks(batch: Dict[str, Any], micro_bs: int) -> List[Dict[str, Any]]:
    bs = _infer_batch_size(batch)
    if micro_bs <= 0 or bs <= micro_bs:
        return [batch]
    return [
        _split_batch_rows(batch, i, min(i + micro_bs, bs), bs)
        for i in range(0, bs, micro_bs)
    ]


def _masked_label_count(labels: torch.Tensor) -> torch.Tensor:
    return (labels != IGNORE_INDEX).sum().to(dtype=torch.float32).clamp(min=1.0)


def _fisher_chunk_loss_scale(batch_full: Dict[str, Any], chunk: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    lf = batch_full.get("labels")
    lc = chunk.get("labels")
    if isinstance(lf, torch.Tensor) and isinstance(lc, torch.Tensor):
        full_d = _masked_label_count(lf)
        chunk_d = _masked_label_count(lc)
        return (chunk_d / full_d).to(device=device, dtype=torch.float32)
    full_bs = max(1, _infer_batch_size(batch_full))
    chunk_bs = max(1, _infer_batch_size(chunk))
    return torch.tensor(chunk_bs / float(full_bs), device=device, dtype=torch.float32)


def _reset_deepspeed_manual_gas_boundary(engine: Any) -> None:
    engine._is_gradient_accumulation_boundary = None


def _fisher_one_backward(
    trainer: Any,
    loss: torch.Tensor,
) -> None:
    ds = getattr(trainer, "deepspeed", None)
    if ds is not None:
        ds.backward(loss, scale_wrt_gas=False)
        return
    accel = getattr(trainer, "accelerator", None)
    if accel is not None:
        accel.backward(loss)
        return
    loss.backward()


def _fisher_forward_backward_batch(
    model: nn.Module,
    trainer: Any,
    full_batch: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> bool:
    ds = getattr(trainer, "deepspeed", None)

    if len(chunks) == 1:
        outputs = model(**chunks[0])
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return False
        _fisher_one_backward(trainer, loss)
        return True

    if ds is not None:
        try:
            for i, ch in enumerate(chunks):
                ds.set_gradient_accumulation_boundary(i == len(chunks) - 1)
                outputs = model(**ch)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    ds.zero_grad()
                    return False
                scale = _fisher_chunk_loss_scale(full_batch, ch, device=loss.device)
                _fisher_one_backward(trainer, loss * scale)
            return True
        finally:
            _reset_deepspeed_manual_gas_boundary(ds)
            if hasattr(ds.optimizer, "is_gradient_accumulation_boundary"):
                ds.optimizer.is_gradient_accumulation_boundary = ds.is_gradient_accumulation_boundary()

    accel = getattr(trainer, "accelerator", None)
    sync_cm = (
        accel.accumulate(model)
        if accel is not None and hasattr(accel, "accumulate")
        else contextlib.nullcontext()
    )
    with sync_cm:
        for ch in chunks:
            outputs = model(**ch)
            loss = getattr(outputs, "loss", None)
            if loss is None:
                model.zero_grad(set_to_none=True)
                return False
            scale = _fisher_chunk_loss_scale(full_batch, ch, device=loss.device)
            _fisher_one_backward(trainer, loss * scale)
    return True


def _unwrap_peft_root(model: nn.Module) -> nn.Module:
    inner = getattr(model, "_base_model", model)
    return inner


def _iter_lora_named_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    root = _unwrap_peft_root(model)
    for name, p in root.named_parameters():
        if "lora" not in name.lower():
            continue
        yield name, p


@CLMethodFactory.register("ewc")
class EwcIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))
        self.ewc_lambda: float = float(getattr(config, "ewc_lambda", 5000.0))
        self.ewc_fisher_batches: int = int(getattr(config, "ewc_fisher_batches", 50))
        self.ewc_fisher_micro_batch_size: Optional[int] = getattr(
            config, "ewc_fisher_micro_batch_size", None
        )
        if self.ewc_fisher_micro_batch_size is not None:
            self.ewc_fisher_micro_batch_size = int(self.ewc_fisher_micro_batch_size)

        # task_id -> { full_param_name -> tensor CPU }
        self._anchors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._fisher: Dict[int, Dict[str, torch.Tensor]] = {}
        self._ewc_skip_penalty_for_fisher: bool = False
        # Chunked ops on GPU (EWC grad injection uses the same chunk size for temporaries).
        _ch = getattr(config, "ewc_penalty_chunk_elements", 262144)
        self._ewc_penalty_chunk_elements: int = int(_ch) if _ch else 262144

    def initialize_model(self, model: nn.Module) -> None:
        for _, p in model.named_parameters():
            p.requires_grad = False
        self._setup_lora(model)

    def _find_target_modules(self, model: nn.Module) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def _setup_lora(self, model: nn.Module) -> None:
        from PEFT import LoraConfig, get_peft_model

        target_modules = self._find_target_modules(model)
        r = int(getattr(self.config, "lora_r", 64))
        lora_config = LoraConfig(
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", max(r * 2, 32))),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )
        _base = getattr(model, "_base_model", None)
        if _base is not None:
            peft_model = get_peft_model(_base, lora_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, lora_config)
        wrapped = getattr(model, "_base_model", None)
        if wrapped is not None and hasattr(wrapped, "print_trainable_parameters"):
            wrapped.print_trainable_parameters()

    def _snapshot_anchors(self, model: nn.Module, task_id: int) -> None:
        snap: Dict[str, torch.Tensor] = {}
        for name, p in _iter_lora_named_parameters(model):
            snap[name] = p.detach().float().cpu().clone()
        self._anchors[int(task_id)] = snap

    def _estimate_fisher_diagonal(
        self, model: nn.Module, trainer: Any, task_id: int
    ) -> None:
        n_batches = max(1, self.ewc_fisher_batches)
        loader = trainer.get_train_dataloader()
        accum: Dict[str, torch.Tensor] = {}
        count = 0
        micro_bs = self.ewc_fisher_micro_batch_size

        was_training = model.training
        model.eval()

        self._ewc_skip_penalty_for_fisher = True
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            for name, p in _iter_lora_named_parameters(model):
                accum[name] = torch.zeros(p.numel(), dtype=torch.float32)

            it = iter(loader)
            for _ in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = trainer._prepare_inputs(batch)

                chunks = (
                    _microbatch_chunks(batch, micro_bs)
                    if micro_bs is not None and micro_bs > 0
                    else [batch]
                )

                if getattr(trainer, "deepspeed", None) is not None:
                    trainer.deepspeed.zero_grad()
                else:
                    model.zero_grad(set_to_none=True)

                with torch.enable_grad():
                    ok = _fisher_forward_backward_batch(model, trainer, batch, chunks)
                    if not ok:
                        continue

                for name, p in _iter_lora_named_parameters(model):
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float().cpu().view(-1)
                    accum[name] += g.pow(2)

                if getattr(trainer, "deepspeed", None) is not None:
                    trainer.deepspeed.zero_grad()
                else:
                    model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                count += 1

            if count == 0:
                self._fisher[int(task_id)] = {k: torch.zeros_like(v) for k, v in accum.items()}
            else:
                for name in accum:
                    accum[name] /= float(count)
                self._fisher[int(task_id)] = accum

        finally:
            self._ewc_skip_penalty_for_fisher = False

        if was_training:
            model.train()

    def on_train_task_finished(
        self,
        model: nn.Module,
        context: CLContext,
        task_id: int,
        *,
        trainer: Optional[Any] = None,
    ) -> None:
        tid = int(task_id)
        self._snapshot_anchors(model, tid)
        if trainer is not None:
            self._estimate_fisher_diagonal(model, trainer, tid)
        else:
            self._fisher[tid] = {
                n: torch.zeros(p.numel(), dtype=torch.float32)
                for n, p in _iter_lora_named_parameters(model)
            }

    def _apply_ewc_grad(self, model: nn.Module, cur_task: int, scale: float) -> None:
        """
        Add ``scale * F * (theta - theta_star)`` to LoRA ``param.grad`` (no autograd).

        Matches ``d/dtheta [(lambda/2) * sum F (theta - theta*)^2]`` with
        ``scale = lambda / gradient_accumulation_steps``. Avoids a forward graph over all LoRA
        weights (OOM).
        """
        if scale == 0.0:
            return
        ct = int(cur_task)
        chunk = max(4096, self._ewc_penalty_chunk_elements)

        for tid in sorted(self._anchors.keys()):
            if tid >= ct:
                continue
            anchors = self._anchors.get(tid)
            fisher = self._fisher.get(tid)
            if not anchors or not fisher:
                continue
            for name, p in _iter_lora_named_parameters(model):
                if name not in anchors or name not in fisher:
                    continue
                g = p.grad
                if g is None:
                    continue
                theta_star_cpu = anchors[name]
                fdiag_cpu = fisher[name]
                if theta_star_cpu.numel() != fdiag_cpu.numel():
                    continue
                n = p.numel()
                if theta_star_cpu.numel() != n:
                    continue
                theta_star_flat = theta_star_cpu.reshape(-1)
                gf = g.flatten()
                wf = p.data.flatten()
                i = 0
                while i < n:
                    j = min(i + chunk, n)
                    fd = fdiag_cpu[i:j].to(device=p.device, dtype=torch.float32, non_blocking=True)
                    ts = theta_star_flat[i:j].to(device=p.device, dtype=torch.float32, non_blocking=True)
                    diff = wf[i:j].float() - ts
                    upd = (scale * fd * diff).to(dtype=gf.dtype)
                    gf[i:j].add_(upd)
                    i = j

    def on_input_prep(self, model: nn.Module, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        # EWC is applied in ``on_training_batch_end`` via gradient injection (``_apply_ewc_grad``).
        return outputs

    def on_training_batch_end(
        self,
        model: nn.Module,
        context: CLContext,
        batch: Dict[str, Any],
        *,
        loss: Optional[torch.Tensor] = None,
        trainer: Any = None,
    ) -> None:
        if getattr(self, "_ewc_skip_penalty_for_fisher", False):
            return
        if trainer is None:
            return
        if not model.training:
            return
        ct = int(getattr(context, "task_id", getattr(self.config, "cur_task", self.cur_task)))
        if ct < 1:
            return
        gas = max(1, int(getattr(trainer.args, "gradient_accumulation_steps", 1)))
        scale = float(self.ewc_lambda) / float(gas)
        self._apply_ewc_grad(model, ct, scale)

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        pass

    def get_inference_config(self) -> Dict[str, Any]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "ewc_fisher_batches": self.ewc_fisher_batches,
            "ewc_fisher_micro_batch_size": self.ewc_fisher_micro_batch_size,
            "ewc_penalty_chunk_elements": self._ewc_penalty_chunk_elements,
            "ewc_grad_mode": "backward_add",
            "lora_r": int(getattr(self.config, "lora_r", 64)),
        }

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        import os

        path = os.path.join(output_dir, "ewc_state.pt")
        payload = {
            "anchors": {k: v for k, v in self._anchors.items()},
            "fisher": {k: v for k, v in self._fisher.items()},
            "ewc_lambda": self.ewc_lambda,
        }
        torch.save(payload, path)
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        import os

        path = os.path.join(load_dir, "ewc_state.pt")
        if not os.path.isfile(path):
            return False
        blob = torch.load(path, map_location="cpu")
        if not isinstance(blob, dict):
            return False
        a = blob.get("anchors")
        f = blob.get("fisher")
        if isinstance(a, dict):
            self._anchors = {int(k): v for k, v in a.items()}
        if isinstance(f, dict):
            self._fisher = {int(k): v for k, v in f.items()}
        return bool(self._anchors)

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from method.base.context import CLContext
from method.custom.specialized_integration import RouterIntegration
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def _same_linear_module_types():
    from PEFT.tuners.custom.same import SAMELinear

    types = [SAMELinear]
    try:
        from PEFT.tuners.custom.same import SAMELinear8bitLt

        types.append(SAMELinear8bitLt)
    except ImportError:
        pass
    return tuple(types)


def _is_same_linear(module: Any) -> bool:
    return isinstance(module, _same_linear_module_types())

def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.tuners.custom.same import SAMEConfig, SAMEModel

    register_peft_extension(
        peft_type="MOE_LORA_SAME",
        config_cls=SAMEConfig,
        tuner_model_cls=SAMEModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("same")
class SameIntegration(RouterIntegration):
    def initialize_model(self, model) -> None:
        super().initialize_model(model)
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_same_lora(model)
        self.sync_same_cur_task(model)

    def sync_same_cur_task(self, model: Any) -> None:
        ct = int(getattr(self.config, "cur_task", self.cur_task))
        self.cur_task = ct
        for m in model.modules():
            if _is_same_linear(m):
                m.cur_task = ct

    def _any_same_cov_prev_valid(self, model: Any) -> bool:
        for m in model.modules():
            if not _is_same_linear(m):
                continue
            adapter = getattr(m, "active_adapter", None)
            if isinstance(adapter, (list, tuple)) and adapter:
                adapter = adapter[0]
            if not isinstance(adapter, str) or not adapter:
                continue
            buf = getattr(m, f"cov_prev_valid_{adapter}", None)
            if (
                buf is not None
                and isinstance(buf, torch.Tensor)
                and buf.numel() == 1
                and bool(buf.detach().cpu().item())
            ):
                return True
        return False

    def _reconcile_cov_prev_valid_buffers(self, model: Any) -> None:
        for m in model.modules():
            if not _is_same_linear(m):
                continue
            adapter = getattr(m, "active_adapter", None)
            if isinstance(adapter, (list, tuple)) and adapter:
                adapter = adapter[0]
            if not isinstance(adapter, str) or not adapter:
                continue
            buf = getattr(m, f"cov_prev_valid_{adapter}", None)
            s_prev = getattr(m, f"cov_S_prev_{adapter}", None)
            if buf is None or s_prev is None:
                continue
            if not isinstance(buf, torch.Tensor) or buf.numel() != 1:
                continue
            if bool(buf.detach().cpu().item()):
                continue
            energy = (s_prev.detach().float() ** 2).sum().item()
            if energy > 1e-12:
                setattr(
                    m,
                    f"cov_prev_valid_{adapter}",
                    torch.tensor(True, device=buf.device, dtype=buf.dtype),
                )

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        """After first training forward on task>=1, snapshot prev cov if checkpoint left all flags False."""
        ct = int(getattr(self.config, "cur_task", getattr(self, "cur_task", 0)))
        if (
            ct >= 1
            and getattr(model, "training", False)
            and not getattr(self, "_same_fwd_bootstrap_done", False)
        ):
            if not self._any_same_cov_prev_valid(model):
                self._snapshot_same_carry_buffers(model)
            self._same_fwd_bootstrap_done = True
        return outputs

    def _setup_same_lora(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.same import SAMEConfig

        target_modules = self._find_target_modules(model)

        peft_config = SAMEConfig(
            target_modules=target_modules,
            r=int(getattr(self.config, "lora_r", 64)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.task_num,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            tau_score=float(getattr(self.config, "tau_score", 0.1)),
            curvature_mu=float(getattr(self.config, "curvature_mu", 0.9)),
            window_size=int(getattr(self.config, "window_size", 3)),
            max_components=int(getattr(self.config, "max_components", 64)),
            cumulative_energy_ratio=float(getattr(self.config, "cumulative_energy_ratio", 0.9)),
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
            if hasattr(model, "_modules"):
                model._modules["_base_model"] = peft_model
        else:
            get_peft_model(model, peft_config)

    def _find_target_modules(self, model) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        return

    def _snapshot_same_carry_buffers(self, model: Any) -> None:
        """Materialize ``cov_*_prev`` / ``cov_prev_valid`` from running cov buffers (call before saving)."""
        for module in model.modules():
            if hasattr(module, "save_task_covariance_snapshot") and hasattr(module, "active_adapter"):
                adapter = getattr(module, "active_adapter", None)
                if isinstance(adapter, (list, tuple)) and adapter:
                    adapter = adapter[0]
                if not isinstance(adapter, str) or not adapter:
                    continue
                module.save_task_covariance_snapshot(adapter)

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        self._snapshot_same_carry_buffers(model)

    def get_inference_config(self) -> Dict:
        return {"task_num": self.task_num, "feature_dim": self.feature_dim}

    def _collect_same_carry_buffer_state(self, model: Any) -> Dict[str, torch.Tensor]:
        """Collect SAME carry-over buffers (CLModel-safe; includes 8-bit SAMELinear layers)."""
        buffer_markers = self.carry_buffer_markers()
        out: Dict[str, torch.Tensor] = {}
        for name, buf in model.named_buffers():
            if any(m in name for m in buffer_markers):
                out[name] = buf.detach().cpu().clone()
        return out

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        same_state: Dict[str, Any] = dict(self.load_state())

        root = model if model is not None else self._model_ref
        if root is None:
            raise RuntimeError("SameIntegration.save_extra_state: both model and _model_ref are None")

        # Trainer may not invoke ``on_task_end`` before final ``save_model``; freeze carry buffers here.
        self._snapshot_same_carry_buffers(root)

        carry_buffers = self._collect_same_carry_buffer_state(root)
        same_state.update(carry_buffers)
        n_carry = len(carry_buffers)

        if n_carry == 0 and not same_state:
            raise RuntimeError(
                "SameIntegration.save_extra_state: no SAME carry-over buffers "
                "(cov_*/importance/cov_prev_valid) found on model; refusing empty save."
            )

        if not same_state:
            return False

        extra = self._same_state_to_tensor_bundle(same_state)
        self.merge_extra_into_adapter_safetensors(
            output_dir, extra, on_duplicate_key="prefer_second"
        )
        torch.save(same_state, os.path.join(output_dir, "same_state.bin"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        state_from_bin, state_from_st, bin_path, st_path = self._read_router_checkpoint_files(
            load_dir, carryover_bin="same_state.bin"
        )
        state, p = self._merge_router_checkpoint_states(
            state_from_bin,
            state_from_st,
            bin_path=bin_path,
            st_path=st_path,
        )
        if not state:
            return False

        target = getattr(model, "_base_model", None) or model if model is not None else None

        copied = 0
        missing = 0
        tracked_buffers: List[str] = []
        state_keys = [k for k in state.keys() if isinstance(k, str)]
        buffer_markers = self.carry_buffer_markers()
        tensor_index = self._build_router_tensor_index(state, buffer_markers)

        anchor_lists_ok, aux_ok = self.restore_state(state, model=model)

        if self.image_anchors is not None and not anchor_lists_ok:
            raise RuntimeError(
                f"SAME load_extra_state({load_dir}): checkpoint has no non-empty "
                "image_anchors/text_anchors lists (expected same_state.bin or "
                "prism.same.* / mcitbox.same.* anchor keys in adapter_model.safetensors). "
                "Older code could mark load as successful after restoring only boundaries/carry buffers, "
                "leaving random prototypes — re-save from a good checkpoint or fix the adapter files."
            )

        if target is not None:
            for buf_name, buf in target.named_buffers():
                if not any(m in buf_name for m in buffer_markers):
                    continue

                tracked_buffers.append(buf_name)

                if buf_name in state and isinstance(state[buf_name], torch.Tensor):
                    buf.data.copy_(state[buf_name].to(dtype=buf.dtype, device=buf.device))
                    copied += 1
                    continue

                ck = self._canonical_router_buffer_key(buf_name)
                src = tensor_index.get(ck)
                if src is not None:
                    buf.data.copy_(src.to(dtype=buf.dtype, device=buf.device))
                    copied += 1
                    continue

                cands = [k for k in state_keys if k.endswith(buf_name) or buf_name.endswith(k)]
                if not cands:
                    missing += 1
                    continue
                best = max(cands, key=len)
                if isinstance(state[best], torch.Tensor):
                    buf.data.copy_(state[best].to(dtype=buf.dtype, device=buf.device))
                    copied += 1

        if model is not None:
            self.sync_same_cur_task(model)
            self._reconcile_cov_prev_valid_buffers(model)

        cov_valid: List[bool] = []
        if target is not None:
            for buf_name, buf in target.named_buffers():
                if "cov_prev_valid_" in buf_name:
                    try:
                        cov_valid.append(bool(buf.detach().cpu().item()))
                    except Exception:
                        pass

        if tracked_buffers and copied == 0:
            raise RuntimeError(
                f"SAME load_extra_state: checkpoint under {load_dir} has no matching buffer tensors "
                f"for {len(tracked_buffers)} tracked SAME buffers (key mismatch or empty save)."
            )

        if copied > 0 and cov_valid and sum(cov_valid) == 0:
            raise RuntimeError(
                f"SAME extra state loaded from {p} but cov_prev_valid remains all False. "
                f"copied={copied}, missing={missing}, tracked={len(tracked_buffers)}"
            )

        ok = copied > 0 or anchor_lists_ok or aux_ok
        if ok:
            self.print_carryover_restore_summary(p, state, tag="[SAME]")
        return ok


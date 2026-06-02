import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
    TRAINING_ARGS_NAME,
)
from typing import Any, Dict, Optional

from core.save_checkpoint import get_mm_adapter_state_maybe_zero_3
from backbone.shared.data import LengthGroupedSampler


class MLLMTrainer(Trainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """After backward: ``CLIntegration.on_training_batch_end`` (each micro-batch under gradient accumulation)."""
        loss = super().training_step(model, inputs)
        self._dispatch_cl_on_training_batch_end(model, inputs, loss)
        return loss

    def _dispatch_cl_on_training_batch_end(
        self, model: nn.Module, inputs: Dict[str, Any], loss: torch.Tensor
    ) -> None:
        try:
            from accelerate.utils import unwrap_model

            core = unwrap_model(model)
        except Exception:
            core = model
            if hasattr(core, "module"):
                core = core.module
        if not hasattr(core, "_integration") or core._integration is None:
            return
        hook = getattr(core._integration, "on_training_batch_end", None)
        if not callable(hook):
            return
        ctx = getattr(core, "_cl_context", None)
        if ctx is None:
            return
        hook(core, ctx, inputs, loss=loss, trainer=self)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
                generator=self.args.data_seed if self.args.data_seed is not None else None,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer with optional separate LR for mm_projector.
        """
        if is_sagemaker_mp_enabled() or self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model

        # Params that use weight decay
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Optional separate LR for mm_projector
        if self.args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
            optimizer_grouped_parameters = [
                # Non-projector (decay)
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n not in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # Non-projector (no decay)
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n not in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
                # Projector (decay) with mm_projector_lr
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                # Projector (no decay) with mm_projector_lr
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        else:
            # Standard decay / no-decay groups
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # Filter out empty param groups (can happen with mm_projector_lr set
        # but projector params frozen, e.g. prompt-based CL methods)
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # ShardedDDP path
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            self.optimizer = ZeroRedundancyOptimizer(
                params=optimizer_grouped_parameters,
                optimizer_class=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Adam8bit: optional fp32 embedding overrides (could move to common)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank in [0, -1]:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
        else:
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save model checkpoint. When the model is a CLModel wrapping a PeftModel,
        save PEFT adapter weights (adapter_model + adapter_config.json) instead
        of the full pytorch_model.bin."""
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            super()._save(output_dir, state_dict)
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # Unwrap from DeepSpeed / DDP wrappers to get the raw model
        raw_model = self.model
        while hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        # If the unwrapped model is a CLModel wrapping a PeftModel, delegate to
        # PeftModel.save_pretrained so that load_adapter can consume the checkpoint.
        if hasattr(raw_model, '_base_model'):
            from PEFT.peft_model import PeftModel
            peft_model = raw_model._base_model
            if isinstance(peft_model, PeftModel):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Saving model checkpoint to {output_dir}")

                # 1) Save PEFT adapter weights + adapter_config.json
                peft_model.save_pretrained(output_dir)

                # 2) Save the base model's config.json so inference can load it via
                #    AutoConfig.from_pretrained(checkpoint_path).
                base_config = getattr(peft_model, 'config', None)
                if base_config is not None and hasattr(base_config, 'save_pretrained'):
                    base_config.save_pretrained(output_dir)

                # 3) Save tokenizer, training args, and CL extra state
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

                # 4) Save method-specific state (e.g. modal_prompt_state.pt)
                if hasattr(raw_model, '_integration') and raw_model._integration is not None:
                    if hasattr(raw_model._integration, 'save_extra_state'):
                        local_rank = getattr(self.args, "local_rank", -1)
                        if local_rank in (-1, 0):
                            saved = raw_model._integration.save_extra_state(output_dir, model=raw_model)
                            if saved:
                                logger.info("CL extra state saved via _integration")
                return

        super()._save(output_dir, state_dict)


LLaVATrainer = MLLMTrainer
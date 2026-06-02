import os
import sys
import subprocess
import pathlib
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
_current_file = Path(__file__).absolute()
_project_root = _current_file.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core import (
    load_config,
    load_model_for_train,
    save_model,
    make_supervised_data_module,
)
from backbone.shared.train.trainer import MLLMTrainer
from method.base.callback import CLTrainerCallback  # noqa: F401

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def train():
    global local_rank

    # run.py passes PYPRISM_LOG_LEVEL in the subprocess env (see cmd_train). Apply early so
    # third-party imports that configured logging do not leave handlers at INFO/TRAIN-only.
    import os

    from backbone.shared.runtime_logging import configure_prism_logging_from_env

    configure_prism_logging_from_env("TRAIN")

    from backbone.shared.train.checkpoint_use_reentrant_patch import (
        apply_gradient_checkpoint_use_reentrant_false,
    )

    apply_gradient_checkpoint_use_reentrant_false()

    try:
        import deepspeed.runtime.utils as _ds_util

        _ds_util.see_memory_usage = lambda msg, force=False: None
    except Exception:
        pass

    model_args, data_args, training_args = load_config()
    local_rank = training_args.local_rank
    model, tokenizer, data_args = load_model_for_train(model_args, data_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # CLIntegration.on_training_batch_end etc. are dispatched from MLLMTrainer.training_step.
    cl_cb = CLTrainerCallback(model_args, model)
    trainer = MLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[cl_cb],
        **data_module,
    )
    cl_cb.trainer = trainer
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print(f"Resuming from checkpoint in {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    rank0_print("Training completed, saving model...")

    save_model(model, training_args, trainer=trainer)

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("Cleaning intermediate checkpoints...")
        remove_dir = training_args.output_dir
        subprocess.run(
            f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +",
            shell=True
        )
        rank0_print("Done!")

if __name__ == "__main__":
    train()

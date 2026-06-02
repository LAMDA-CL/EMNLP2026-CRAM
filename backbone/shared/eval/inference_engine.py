import json
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from backbone.shared.multimodal.data_processor import get_model_name_from_path

from backbone.shared.data import InferenceDataset


def _describe_infer_sample_for_error(
    sample: Dict[str, Any],
    dataset_index: int,
    image_folder: str,
) -> str:
    """Human-readable one-sample dump when inference fails (log / stderr)."""
    qid = sample.get("question_id", sample.get("id"))
    text = sample.get("text") or sample.get("question") or ""
    img = sample.get("image")
    n_img_mentions = str(text).lower().count("<image>")
    lines = [
        f"  [dataset index {dataset_index}] question_id={qid!r}",
        f"    image field: {img!r}",
        f"    '<image>' occurrences in text: {n_img_mentions}",
        f"    has top-level image path: {bool(img)}",
    ]
    if image_folder and img:
        lines.append(f"    image_folder + image: {os.path.join(str(image_folder), str(img))}")
    t = str(text)
    if len(t) > 800:
        t = t[:800] + " ...[truncated]"
    lines.append(f"    text: {t!r}")
    return "\n".join(lines)


def _log_infer_failure(
    exc: BaseException,
    batch_samples: List[Dict[str, Any]],
    batch_start: int,
    image_folder: str,
) -> None:
    print("\n[infer] Inference raised an exception; failing batch context:", flush=True)
    print(f"  {type(exc).__name__}: {exc}", flush=True)
    print(f"  batch sample indices: {batch_start} .. {batch_start + len(batch_samples) - 1}", flush=True)
    traceback.print_exc()
    print("[infer] Samples in this batch:", flush=True)
    for j, sample in enumerate(batch_samples):
        print(_describe_infer_sample_for_error(sample, batch_start + j, image_folder), flush=True)


def _resolved_cl_method(args: Any) -> Optional[str]:
    """CLI ``--method`` (preferred) or legacy ``--clmethod``; must not collide with subparser names."""
    for key in ("method", "clmethod"):
        v = getattr(args, key, None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


@dataclass
class InferenceContext:
    args: Any
    tokenizer: Any
    model: Any
    image_processor: Any
    model_name: str


class BaseInferenceAdapter(ABC):
    @abstractmethod
    def on_model_ready(self, context: InferenceContext) -> None:
        pass

    @abstractmethod
    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        pass


class InferenceEngine:
    def __init__(self, adapter: BaseInferenceAdapter) -> None:
        self.adapter = adapter

    def _load_model(self, args: Any) -> Tuple[Any, Any, Any, str]:
        from core.load_model import load_model_for_inference

        cl_method = _resolved_cl_method(args)
        _m = str(cl_method or "").strip().lower()
        if _m == "zeroshot":
            mb = getattr(args, "model_base", None)
            if not mb:
                raise ValueError(
                    "method=zeroshot requires --model-base (LLaVA weights); no CL adapter is loaded (--model-path ignored)."
                )
            mp = os.path.expanduser(str(mb).strip())
            if str(getattr(args, "model_path", "") or "").strip():
                print(
                    "[infer] method=zeroshot: ignoring --model-path; loading base weights from --model-base only.",
                    flush=True,
                )
            model_name = getattr(args, "model_name", None) or get_model_name_from_path(mp)
        else:
            model_path = os.path.expanduser(args.model_path)
            mp = model_path
            model_name = getattr(args, "model_name", None) or get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_model_for_inference(
            mp,
            args.model_base,
            model_name,
            load_8bit=getattr(args, "load_8bit", None),
            load_4bit=getattr(args, "load_4bit", None),
            method=cl_method,
            text_tower=getattr(args, "text_tower", None),
            vision_tower=getattr(args, "vision_tower", None),
            routing_vision_tower=getattr(args, "routing_vision_tower", None),
            backbone=getattr(args, "backbone", None),
            benchmark=getattr(args, "benchmark", None),
            task_num=getattr(args, "cl_task_num", None),
            same_print_router=getattr(args, "same_print_router", False),
            same_print_router_max=getattr(args, "same_print_router_max", None),
        )
        return tokenizer, model, image_processor, model_name

    def _build_dataset(self, args: Any) -> InferenceDataset:
        return InferenceDataset(
            question_file=args.question_file,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
        )

    def run(self, args: Any) -> None:
        tokenizer, model, image_processor, model_name = self._load_model(args)
        context = InferenceContext(
            args=args,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            model_name=model_name,
        )
        self.adapter.on_model_ready(context)

        dataset = self._build_dataset(args)
        samples = dataset.samples
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        batch_size = getattr(args, "batch_size", 1)
        print(
            f"Evaluating {len(samples)} questions (batch size {batch_size}), saving to {answers_file}")

        image_folder = str(getattr(args, "image_folder", "") or "")

        with open(answers_file, "w", encoding="utf-8") as file:
            for i in tqdm(range(0, len(samples), batch_size)):
                batch_samples = samples[i:i + batch_size]
                try:
                    if hasattr(self.adapter, "infer_batch"):
                        outputs = self.adapter.infer_batch(batch_samples, context)
                    else:
                        outputs = [
                            self.adapter.infer_one(sample, context) for sample in batch_samples
                        ]
                except Exception as exc:
                    _log_infer_failure(exc, batch_samples, i, image_folder)
                    if len(batch_samples) > 1 and hasattr(self.adapter, "infer_one"):
                        print(
                            "[infer] Re-running each sample alone to locate the failing row...",
                            flush=True,
                        )
                        for j, sample in enumerate(batch_samples):
                            try:
                                self.adapter.infer_one(sample, context)
                            except Exception:
                                k = i + j
                                print(
                                    f"[infer] Single-sample rerun failed at dataset index {k}:",
                                    flush=True,
                                )
                                print(
                                    _describe_infer_sample_for_error(sample, k, image_folder),
                                    flush=True,
                                )
                                raise
                    raise exc

                for sample, output in zip(batch_samples, outputs):
                    result = {
                        "question_id": sample["question_id"],
                        "prompt": output["prompt"],
                        "text": output["text"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": context.model_name,
                        "metadata": {},
                    }
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                if getattr(args, "flush_each_line", False):
                    file.flush()


def read_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def default_generate(model: Any, input_ids: torch.Tensor, images: Optional[torch.Tensor], args: Any, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    generate_kwargs = {
        "do_sample": getattr(args, "temperature", 0.0) > 0,
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", None),
        "num_beams": getattr(args, "num_beams", 1),
        "max_new_tokens": getattr(args, "max_new_tokens", 128),
        "use_cache": True,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    generate_kwargs.update(kwargs)

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            images=images,
            **generate_kwargs,
        )


def decode_new_tokens(tokenizer: Any, output_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
    input_len = input_ids.shape[1]
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import json


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_text_result(output_dir: Optional[str], content: str) -> None:
    if not output_dir:
        return
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, "Result.text")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)


def normalize_text(text: Any, mode: str) -> str:
    value = str(text).strip()
    if mode == "upper":
        return value.upper()
    if mode == "lower":
        return value.lower()
    return value


def evaluate_default_qa_accuracy(
    annotation_file: str,
    result_file: str,
    output_dir: Optional[str] = None,
    annotation_root_key: Optional[str] = None,
    question_id_key: str = "question_id",
    answer_key: str = "answer",
    pred_key: str = "text",
    normalize_mode: str = "upper",
    skip_if_pred_contains: Optional[str] = None,
) -> Dict[str, Any]:
    annotations_raw = load_json(annotation_file)
    if annotation_root_key is not None:
        annotations_raw = annotations_raw[annotation_root_key]

    annotations = {str(item[question_id_key]): item for item in annotations_raw}
    results = load_jsonl(result_file)

    total = len(results)
    right = 0
    for result in results:
        qid = str(result[question_id_key])
        if qid not in annotations:
            continue

        pred = str(result.get(pred_key, ""))
        if skip_if_pred_contains and skip_if_pred_contains in pred:
            continue

        gt = annotations[qid].get(answer_key, "")
        pred_norm = normalize_text(pred, normalize_mode)
        gt_norm = normalize_text(gt, normalize_mode)
        right += int(pred_norm == gt_norm)

    acc = 100.0 * right / total if total else 0.0
    summary = f"Samples: {total}\nAccuracy: {acc:.2f}%\n"
    print(summary)
    write_text_result(output_dir, summary)
    return {"samples": total, "accuracy": acc}


class BaseEvaluator(ABC):
    name: str = ""
    help_text: str = ""

    @abstractmethod
    def add_arguments(self, parser: Any) -> None:
        pass

    @abstractmethod
    def evaluate(self, args: Any) -> Dict[str, Any]:
        pass


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: Dict[str, BaseEvaluator] = {}

    def register(self, evaluator: BaseEvaluator) -> None:
        if evaluator.name in self._evaluators:
            raise ValueError(f"Evaluator already registered: {evaluator.name}")
        self._evaluators[evaluator.name] = evaluator

    def add_subparsers(self, subparsers: Any) -> None:
        for name, evaluator in self._evaluators.items():
            parser = subparsers.add_parser(name, help=evaluator.help_text)
            evaluator.add_arguments(parser)

    def run(self, args: Any) -> Dict[str, Any]:
        task = getattr(args, "task", None)
        if task not in self._evaluators:
            raise ValueError(f"Unsupported task: {task}")
        return self._evaluators[task].evaluate(args)

    def tasks(self) -> List[str]:
        return list(self._evaluators.keys())

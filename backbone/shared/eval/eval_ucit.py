from .eval_default import (
    BaseEvaluator,
    ensure_dir,
    load_json,
    load_jsonl,
    write_text_result,
)
from .eval_caption import eval_single as eval_caption_single
from .eval_caption import create_coco_type, merge_captions
from .eval_deepseek_r1 import eval_single, deepseek_chat_final

import os
from typing import Any, Dict, List

import sys
sys.path.append(os.path.dirname(__file__))


# ── ANLS metric ────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            substitute = prev_row[j] + (c1 != c2)
            curr_row.append(min(insert, delete, substitute))
        prev_row = curr_row
    return prev_row[-1]


def anls_single(pred: str, ground_truths: List[str]) -> float:
    """ANLS for one prediction against a list of acceptable answers."""
    pred = pred.strip()
    best = 0.0
    for gt in ground_truths:
        gt = str(gt).strip()
        if pred == "" and gt == "":
            score = 1.0
        elif pred == "" or gt == "":
            score = 0.0
        else:
            dist = levenshtein_distance(pred.lower(), gt.lower())
            score = 1.0 - dist / max(len(pred), len(gt))
        best = max(best, score)
    return best


def evaluate_anls(
    annotation_file: str,
    result_file: str,
    output_dir: str = None,
) -> Dict[str, Any]:
    annotations = load_json(annotation_file)
    annotations = {str(item["question_id"]): item for item in annotations}
    results = load_jsonl(result_file)

    scores: List[float] = []
    pairs: List[Dict[str, Any]] = []

    for result in results:
        qid = str(result["question_id"])
        if qid not in annotations:
            continue

        pred = str(result.get("text", ""))
        gt = annotations[qid]["answer"]
        if isinstance(gt, str):
            gt = [gt]

        s = anls_single(pred, gt)
        scores.append(s)
        pairs.append({"pred": pred, "ground_truth": gt, "anls": s})

    avg_anls = (sum(scores) / len(scores) * 100.0) if scores else 0.0
    summary = f"Samples: {len(scores)}\nANLS: {avg_anls:.2f}%\n"
    print(summary)

    if output_dir:
        ensure_dir(output_dir)
        write_text_result(output_dir, summary)
        import json
        with open(os.path.join(output_dir, "ans_gt.json"), "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=4)

    return {"samples": len(scores), "anls": avg_anls}


# ── Evaluator classes ──────────────────────────────────────

class DeepSeekR1Evaluator(BaseEvaluator):
    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        ans_gt_file = eval_single(args.annotation_file, args.result_file, args)
        return {"result_file": ans_gt_file}


class ANLSEvaluator(BaseEvaluator):
    """Evaluator using ANLS (Average Normalized Levenshtein Similarity)."""

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_anls(args.annotation_file, args.result_file, args.output_dir)


class ImageNetREvaluator(DeepSeekR1Evaluator):
    name = "imagenetr"
    help_text = "Evaluate ImageNet-R results using DeepSeek R1 script"


class ArxivQAEvaluator(DeepSeekR1Evaluator):
    name = "arxivqa"
    help_text = "Evaluate ArxivQA results using DeepSeek R1 script"


class IconQAEvaluator(DeepSeekR1Evaluator):
    name = "iconqa"
    help_text = "Evaluate IconQA results using DeepSeek R1 script"


class CLEVREvaluator(DeepSeekR1Evaluator):
    name = "clevr"
    help_text = "Evaluate CLEVR results using DeepSeek R1 script"


class CaptionEvaluator(BaseEvaluator):
    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        output_file, total = create_coco_type(
            args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_caption_single(output_file, args.annotation_file, total, args)
        return {"total": total}


class Flickr30kEvaluator(CaptionEvaluator):
    name = "flickr30k"
    help_text = "Evaluate Flickr30k results using caption eval"


class VizcapEvaluator(CaptionEvaluator):
    name = "vizcap"
    help_text = "Evaluate VizWiz caption results using caption eval"


class ChartQAEvaluator(DeepSeekR1Evaluator):
    name = "chartqa"
    help_text = "Evaluate ChartQA results using DeepSeek R1 script"


class DocVQAEvaluator(ANLSEvaluator):
    name = "docvqa"
    help_text = "Evaluate DocVQA results using ANLS metric"


class InfographicVQAEvaluator(ANLSEvaluator):
    name = "infographicvqa"
    help_text = "Evaluate InfographicVQA results using ANLS metric"


class PMCVQAEvaluator(DeepSeekR1Evaluator):
    name = "pmcvqa"
    help_text = "Evaluate PMCVQA results using DeepSeek R1 script"


class RoadsideEvaluator(DeepSeekR1Evaluator):
    name = "roadside"
    help_text = "Evaluate Roadside results using DeepSeek R1 script"


class ChemVQAEvaluator(DeepSeekR1Evaluator):
    name = "chemvqa"
    help_text = "Evaluate ChemVQA results using DeepSeek R1 script"


class FloodNetVQAEvaluator(DeepSeekR1Evaluator):
    name = "floodnetvqa"
    help_text = "Evaluate FloodNetVQA results using DeepSeek R1 script"

import argparse
import glob
import importlib.util
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

'''
To implement unified evaluation logic for different vision-language tasks
You should first create a new evaluator class that inherits from BaseEvaluator
Then register it with the EvaluatorRegistry
It is also worth noting that the evaluate method needs to return a dictionary containing at least "accuracy" key
'''

try:
    from .eval_default import (
        BaseEvaluator,
        EvaluatorRegistry,
        ensure_dir,
        evaluate_default_qa_accuracy,
        load_json,
        load_jsonl,
        write_text_result,
    )
except ImportError:
    from eval_default import (
        BaseEvaluator,
        EvaluatorRegistry,
        ensure_dir,
        evaluate_default_qa_accuracy,
        load_json,
        load_jsonl,
        write_text_result,
    )

try:
    from m4c_evaluator import TextVQAAccuracyEvaluator
except ModuleNotFoundError:
    evaluator_path = os.path.join(
        os.path.dirname(__file__), "m4c_evaluator.py")
    spec = importlib.util.spec_from_file_location(
        "m4c_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("Cannot load m4c_evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TextVQAAccuracyEvaluator = module.TextVQAAccuracyEvaluator


def evaluate_vqav2(annotation_file: str, result_file: str, output_dir: Optional[str]) -> Dict[str, Any]:
    return evaluate_default_qa_accuracy(
        annotation_file=annotation_file,
        result_file=result_file,
        output_dir=output_dir,
        normalize_mode="upper",
    )


def evaluate_vizwiz(annotation_file: str, result_file: str, output_dir: Optional[str]) -> Dict[str, Any]:
    return evaluate_default_qa_accuracy(
        annotation_file=annotation_file,
        result_file=result_file,
        output_dir=output_dir,
        normalize_mode="upper",
    )


def evaluate_ocrvqa(annotation_file: str, result_file: str, output_dir: Optional[str]) -> Dict[str, Any]:
    return evaluate_default_qa_accuracy(
        annotation_file=annotation_file,
        result_file=result_file,
        output_dir=output_dir,
        normalize_mode="lower",
        skip_if_pred_contains="Unanswerable",
    )


def evaluate_textvqa(annotation_file: str, result_file: str, output_dir: Optional[str]) -> Dict[str, Any]:
    annotations = load_json(annotation_file)["data"]
    annotations = {str(item["question_id"]): item for item in annotations}
    results = load_jsonl(result_file)

    pred_list = []
    for result in results:
        qid = str(result["question_id"])
        if qid not in annotations:
            continue
        pred_list.append(
            {
                "pred_answer": result.get("text", ""),
                "gt_answers": annotations[qid]["answers"],
            }
        )

    evaluator = TextVQAAccuracyEvaluator()
    acc = 100.0 * evaluator.eval_pred_list(pred_list) if pred_list else 0.0
    summary = f"Samples: {len(pred_list)}\nAccuracy: {acc:.2f}%\n"
    print(summary)
    write_text_result(output_dir, summary)
    return {"samples": len(pred_list), "accuracy": acc}


def evaluate_imagenet(test_file: str, result_file: str, output_dir: Optional[str]) -> Dict[str, Any]:
    annotations = load_json(test_file)
    answers = [item["answer"] for item in annotations]
    results = load_jsonl(result_file)

    total = min(len(results), len(answers))
    right = 0
    false_answers: List[Dict[str, Any]] = []

    for index in tqdm(range(total)):
        gt = str(answers[index])
        pred_item = results[index]
        pred_text = str(pred_item.get("text", ""))
        if (gt in pred_text) or (pred_text in gt):
            right += 1
        else:
            failed = dict(pred_item)
            failed["ground_truth"] = gt
            false_answers.append(failed)

    acc = 100.0 * right / total if total else 0.0
    summary = f"Samples: {total}\nAccuracy: {acc:.2f}%\n"
    print(summary)

    if output_dir:
        ensure_dir(output_dir)
        write_text_result(output_dir, summary)
        with open(os.path.join(output_dir, "false_answers.json"), "w", encoding="utf-8") as file:
            json.dump(false_answers, file, ensure_ascii=False, indent=2)

    return {"samples": total, "accuracy": acc, "false_answers": len(false_answers)}


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2
    intersection_area = max(0.0, min(x2, x22) - max(x1, x21)) * \
        max(0.0, min(y2, y22) - max(y1, y21))
    union_area = (x2 - x1) * (y2 - y1) + (x22 - x21) * \
        (y22 - y21) - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def parse_bbox_from_text(text: Any) -> Optional[List[float]]:
    values = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if len(values) < 4:
        return None
    try:
        return [float(values[i]) for i in range(4)]
    except ValueError:
        return None


def parse_bbox_from_text_legacy(text: Any, is_prediction: bool) -> Optional[List[float]]:
    raw = str(text).replace("[", "").replace("]", "")
    if is_prediction:
        raw = raw[1:-1]
    try:
        values = [float(x) for x in raw.split(",")]
    except Exception:
        return None
    if len(values) != 4:
        return None
    return values


def evaluate_grounding(
    test_file: str,
    result_file: str,
    output_dir: Optional[str],
    legacy_grounding_parse: bool = False,
) -> Dict[str, Any]:
    annotations = load_json(test_file)
    annotations = {str(item["question_id"]): item for item in annotations}
    results = load_jsonl(result_file)

    total = len(results)
    right = 0

    for result in results:
        qid = str(result["question_id"])
        if qid not in annotations:
            continue
        item = annotations[qid]

        if legacy_grounding_parse:
            gt_bbox = parse_bbox_from_text_legacy(
                item.get("answer_bbox", ""), is_prediction=False)
            pred_bbox = parse_bbox_from_text_legacy(
                result.get("text", ""), is_prediction=True)
        else:
            gt_bbox = parse_bbox_from_text(item.get("answer_bbox", ""))
            pred_bbox = parse_bbox_from_text(result.get("text", ""))
        size = item.get("size", [1, 1])

        if gt_bbox is None or pred_bbox is None:
            continue

        max_wh = float(max(size)) if size else 1.0
        pred_bbox = [x * max_wh for x in pred_bbox]
        gt_bbox = [x * max_wh for x in gt_bbox]

        iou = calculate_iou(pred_bbox, gt_bbox)
        right += int(iou > 0.5)

    acc = 100.0 * right / total if total else 0.0
    summary = f"Samples: {total}\nAccuracy: {acc:.2f}%\n"
    print(summary)
    write_text_result(output_dir, summary)
    return {"samples": total, "accuracy": acc}


def evaluate_scienceqa(
    base_dir: str,
    result_file: str,
    split: str,
    options: List[str],
    output_dir: Optional[str],
    output_file: Optional[str],
    output_result: Optional[str],
) -> Dict[str, Any]:
    split_indices = load_json(os.path.join(base_dir, "pid_splits.json"))[split]
    problems_list = load_json(os.path.join(base_dir, "test.json"))
    problems = {str(item["question_id"]): item for item in problems_list}

    predictions = load_jsonl(result_file)
    predictions = {str(item["question_id"]): item for item in predictions}
    split_problems = {str(idx): problems[str(idx)]
                      for idx in split_indices if str(idx) in problems}

    results = {"correct": [], "incorrect": []}
    sqa_results = {
        "acc": None,
        "correct": None,
        "count": None,
        "results": {},
        "outputs": {},
    }

    pattern = re.compile(r"The answer is ([A-Z]).")

    for prob_id, prob in split_problems.items():
        pred = predictions.get(
            prob_id, {"text": "FAILED", "prompt": "Unknown"})
        pred_text = str(pred.get("text", "FAILED"))

        if pred_text in options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        else:
            matched = pattern.findall(pred_text)
            answer = matched[0] if len(matched) == 1 else "FAILED"

        pred_idx = options.index(answer) if answer in options else -1
        gt = str(prob["answer"])
        gt_idx = options.index(gt)

        prompt = str(pred.get("prompt", ""))
        analysis = {
            "question_id": prob_id,
            "parsed_ans": answer,
            "ground_truth": gt,
            "question": prompt,
            "pred": pred_text,
            "is_multimodal": "<image>" in prompt,
        }

        sqa_results["results"][prob_id] = pred_idx
        sqa_results["outputs"][prob_id] = pred_text

        if pred_idx == gt_idx:
            results["correct"].append(analysis)
        else:
            results["incorrect"].append(analysis)

    correct = len(results["correct"])
    total = len(split_problems)

    multimodal_correct = len(
        [x for x in results["correct"] if x["is_multimodal"]])
    multimodal_incorrect = len(
        [x for x in results["incorrect"] if x["is_multimodal"]])
    multimodal_total = multimodal_correct + multimodal_incorrect

    overall_acc = (correct / total * 100.0) if total else 0.0
    img_acc = (multimodal_correct / multimodal_total *
               100.0) if multimodal_total else 0.0

    summary = (
        f"Total: {total}, Correct: {correct}, Accuracy: {overall_acc:.2f}%, "
        f"IMG-Accuracy: {img_acc:.2f}%"
    )
    print(summary)
    write_text_result(output_dir, summary)

    sqa_results["acc"] = overall_acc
    sqa_results["correct"] = correct
    sqa_results["count"] = total

    if output_dir:
        ensure_dir(output_dir)

    if output_file is None and output_dir:
        output_file = os.path.join(output_dir, "scienceqa_analysis.json")
    if output_result is None and output_dir:
        output_result = os.path.join(output_dir, "scienceqa_result.json")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
    if output_result:
        with open(output_result, "w", encoding="utf-8") as file:
            json.dump(sqa_results, file, ensure_ascii=False, indent=2)

    return {
        "samples": total,
        "accuracy": overall_acc,
        "img_accuracy": img_acc,
    }


def load_gqa_file(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        return load_json(path)

    candidate_dir = path.split(".")[0]
    if os.path.isdir(candidate_dir):
        data: Dict[str, Any] = {}
        ext = path.split(".")[-1]
        chunks = glob.glob(
            f"{candidate_dir}/{os.path.basename(candidate_dir)}_*.{ext}")
        for chunk in chunks:
            data.update(load_json(chunk))
        return data

    raise FileNotFoundError(f"Cannot find {path}")


def convert_gqa_jsonl_to_eval_format(src: str, dst: str) -> None:
    rows = load_jsonl(src)
    converted: List[Dict[str, Any]] = []

    for row in rows:
        question_id = row.get("question_id", row.get("questionId"))
        if question_id is None:
            continue
        text = str(row.get("text", "")).rstrip(".").lower()
        converted.append({"questionId": question_id, "prediction": text})

    dst_dir = os.path.dirname(dst)
    if dst_dir:
        ensure_dir(dst_dir)

    with open(dst, "w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False)


def to_score(flag: bool) -> float:
    return float(1 if flag else 0)


def avg(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def get_words_num(question: Dict[str, Any]) -> int:
    return len(str(question["question"]).split())


def get_steps_num(question: Dict[str, Any]) -> int:
    steps = []
    for cell in question.get("semantic", []):
        op = str(cell.get("operation", ""))
        arg = str(cell.get("argument", ""))
        text = f"{op}: {arg}"
        if not any(token in text for token in ["exist", "query: name", "choose name"]):
            steps.append(cell)
    return len(steps)


def get_cell(i: int, j: int, map_size: int) -> Tuple[float, float, float, float]:
    edge = 1.0 / map_size
    return (edge * i, edge * j, edge * (i + 1), edge * (j + 1))


def get_region(scene_graph: Dict[str, Any], object_id: str) -> Tuple[float, float, float, float]:
    obj = scene_graph["objects"][object_id]
    x0 = float(obj["x"]) / scene_graph["width"]
    y0 = float(obj["y"]) / scene_graph["height"]
    x1 = float(obj["x"] + obj["w"]) / scene_graph["width"]
    y1 = float(obj["y"] + obj["h"]) / scene_graph["height"]
    return (x0, y0, x1, y1)


def range_length(r: Optional[Tuple[float, float]]) -> float:
    if r is None:
        return 0.0
    return max(0.0, r[1] - r[0])


def intersection_1d(r1: Tuple[float, float], r2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    inter = (max(r1[0], r2[0]), min(r1[1], r2[1]))
    return inter if inter[1] > inter[0] else None


def box_size(c: Tuple[float, float, float, float]) -> float:
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])


def intersection_rate(c1: Tuple[float, float, float, float], c2: Tuple[float, float, float, float]) -> float:
    x_inter = range_length(intersection_1d((c1[0], c1[2]), (c2[0], c2[2])))
    y_inter = range_length(intersection_1d((c1[1], c1[3]), (c2[1], c2[3])))
    denom = box_size(c1)
    if denom <= 0:
        return 0.0
    return (x_inter * y_inter) / denom


def compute_grounding_score(
    question: Dict[str, Any],
    scene_graph: Dict[str, Any],
    attention_map: Any,
    map_size: int,
    object_features: bool,
) -> float:
    regions = []
    regions += [get_region(scene_graph, pointer)
                for pointer in question["annotations"]["question"].values()]
    regions += [get_region(scene_graph, pointer)
                for pointer in question["annotations"]["fullAnswer"].values()]
    if any(("scene" in str(cell)) for cell in question.get("semantic", [])):
        regions.append((0.0, 0.0, 1.0, 1.0))

    if object_features:
        cells = [((float(x0), float(y0), float(x1), float(y1)), float(att))
                 for x0, y0, x1, y1, att in attention_map]
    else:
        cells = [
            (get_cell(i, j, map_size), float(attention_map[i][j]))
            for i in range(map_size)
            for j in range(map_size)
        ]

    scores = []
    for region in regions:
        for cell, attention in cells:
            scores.append(attention * intersection_rate(cell, region))
    return sum(scores)


def chi_square(gold_dist: Dict[str, Dict[str, int]], predicted_dist: Dict[str, Dict[str, int]]) -> float:
    sum_score, sum_overall = 0.0, 0

    for group in gold_dist:
        score, overall = 0.0, 0
        for ans in gold_dist[group]:
            expected = gold_dist[group][ans]
            observed = predicted_dist[group].get(ans, 0)
            if expected > 0:
                score += ((float(observed - expected) ** 2) / expected)
            overall += expected

        sum_score += score * overall
        sum_overall += overall

    return (sum_score / sum_overall) if sum_overall > 0 else 0.0


def evaluate_gqa(
    path: str,
    question_dir: str,
    tier: str,
    questions_pattern: str,
    predictions_pattern: str,
    raw_result_file: Optional[str],
    consistency: bool,
    grounding: bool,
    attentions_pattern: str,
    scenes_pattern: str,
    object_features: bool,
    map_size: int,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    print("Loading questions...")
    questions = load_gqa_file(os.path.join(
        question_dir, questions_pattern.format(tier=tier)))

    predictions_file = os.path.join(
        path, predictions_pattern.format(tier=tier))
    if raw_result_file:
        raw_file = raw_result_file if os.path.isabs(
            raw_result_file) else os.path.join(path, raw_result_file)
        print(f"Converting raw GQA result: {raw_file} -> {predictions_file}")
        convert_gqa_jsonl_to_eval_format(raw_file, predictions_file)

    print("Loading predictions...")
    predictions_raw = load_gqa_file(predictions_file)
    predictions = {str(p["questionId"]): p["prediction"]
                   for p in predictions_raw}

    scene_graphs = None
    if grounding:
        print("Loading scene graphs...")
        scene_graphs = load_gqa_file(os.path.join(
            question_dir, scenes_pattern.format(tier=tier)))

    attentions = None
    if grounding:
        print("Loading attentions...")
        attentions_path = attentions_pattern.format(tier=tier)
        if not os.path.isabs(attentions_path):
            attentions_path = os.path.join(path, attentions_path)
        attentions_raw = load_json(attentions_path)
        attentions = {str(item["questionId"]): item["attention"]
                      for item in attentions_raw}

    scores: Dict[str, Any] = {
        "accuracy": [],
        "binary": [],
        "open": [],
        "validity": [],
        "plausibility": [],
        "consistency": [],
        "accuracyPerStructuralType": defaultdict(list),
        "accuracyPerSemanticType": defaultdict(list),
        "accuracyPerLength": defaultdict(list),
        "accuracyPerSteps": defaultdict(list),
        "grounding": [],
    }

    dist = {
        "gold": defaultdict(lambda: defaultdict(int)),
        "predicted": defaultdict(lambda: defaultdict(int)),
    }

    for qid, predicted in tqdm(predictions.items()):
        if qid not in questions:
            continue
        question = questions[qid]
        gold = question["answer"]

        correct = predicted == gold
        score = to_score(correct)

        words_num = get_words_num(question)
        steps_num = get_steps_num(question)

        if question.get("isBalanced"):
            scores["accuracy"].append(score)
            scores["accuracyPerLength"][words_num].append(score)
            scores["accuracyPerSteps"][steps_num].append(score)
            scores["accuracyPerStructuralType"][question["types"]
                                                ["structural"]].append(score)
            scores["accuracyPerSemanticType"][question["types"]
                                              ["semantic"]].append(score)
            answer_type = "open" if question["types"]["structural"] == "query" else "binary"
            scores[answer_type].append(score)

            global_group = question["groups"]["global"]
            if global_group is not None:
                dist["gold"][global_group][gold] += 1
                dist["predicted"][global_group][predicted] += 1

        if consistency and correct:
            inferred_questions = [eid for eid in question.get(
                "entailed", []) if str(eid) != qid]
            if inferred_questions:
                consistency_scores = []
                for eid in inferred_questions:
                    eid = str(eid)
                    if eid in questions and eid in predictions:
                        consistency_scores.append(
                            to_score(predictions[eid] == questions[eid]["answer"]))
                if consistency_scores:
                    scores["consistency"].append(avg(consistency_scores))

        if grounding and scene_graphs is not None and attentions is not None:
            image_id = str(question.get("imageId"))
            if image_id in scene_graphs and qid in attentions:
                g_score = compute_grounding_score(
                    question=question,
                    scene_graph=scene_graphs[image_id],
                    attention_map=attentions[qid],
                    map_size=map_size,
                    object_features=object_features,
                )
                scores["grounding"].append(g_score)

    scores["distribution"] = chi_square(
        dist["gold"], dist["predicted"]) / 100.0

    metrics = [
        "binary",
        "open",
        "accuracy",
        "consistency",
        "validity",
        "plausibility",
        "grounding",
        "distribution",
    ]

    detailed_metrics = [
        ("accuracyPerStructuralType", "Accuracy / structural type"),
        ("accuracyPerSemanticType", "Accuracy / semantic type"),
        ("accuracyPerSteps", "Accuracy / steps number"),
        ("accuracyPerLength", "Accuracy / words number"),
    ]

    for key in metrics:
        if isinstance(scores[key], list):
            scores[key] = avg(scores[key]) * 100.0

    for key, _ in detailed_metrics:
        for sub_key in scores[key]:
            scores[key][sub_key] = (
                avg(scores[key][sub_key]) * 100.0, len(scores[key][sub_key]))

    lines = []
    for metric in metrics:
        if metric == "grounding" and not grounding:
            continue
        if metric == "consistency" and not consistency:
            continue
        suffix = " (lower is better)" if metric == "distribution" else "%"
        lines.append(f"{metric.capitalize()}: {scores[metric]:.2f}{suffix}")

    for metric, print_name in detailed_metrics:
        lines.append("")
        lines.append(f"{print_name}:")
        for sub_type in sorted(scores[metric].keys()):
            score_value, amount = scores[metric][sub_type]
            lines.append(
                f"  {sub_type}: {score_value:.2f}% ({amount} questions)")

    text = "\n" + "\n".join(lines)
    print(text)
    write_text_result(output_dir, text)

    return {
        "accuracy": scores["accuracy"],
        "binary": scores["binary"],
        "open": scores["open"],
        "distribution": scores["distribution"],
    }


class VQAv2Evaluator(BaseEvaluator):
    name = "vqav2"
    help_text = "Evaluate VQAv2 results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_vqav2(args.annotation_file, args.result_file, args.output_dir)


class VizWizEvaluator(BaseEvaluator):
    name = "vizwiz"
    help_text = "Evaluate VizWiz results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_vizwiz(args.annotation_file, args.result_file, args.output_dir)


class OCRVQAEvaluator(BaseEvaluator):
    name = "ocrvqa"
    help_text = "Evaluate OCRVQA results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_ocrvqa(args.annotation_file, args.result_file, args.output_dir)


class TextVQAEvaluator(BaseEvaluator):
    name = "textvqa"
    help_text = "Evaluate TextVQA results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_textvqa(args.annotation_file, args.result_file, args.output_dir)


class ImageNetEvaluator(BaseEvaluator):
    name = "imagenet"
    help_text = "Evaluate ImageNet instruction results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--test-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_imagenet(args.test_file, args.result_file, args.output_dir)


class GroundingEvaluator(BaseEvaluator):
    name = "grounding"
    help_text = "Evaluate grounding IoU accuracy"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--test-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument(
            "--legacy-grounding-parse",
            action="store_true",
            help="Reproduce legacy eval_grounding parsing behavior (default: False)",
        )

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_grounding(
            args.test_file,
            args.result_file,
            args.output_dir,
            legacy_grounding_parse=args.legacy_grounding_parse,
        )


class ScienceQAEvaluator(BaseEvaluator):
    name = "scienceqa"
    help_text = "Evaluate ScienceQA multiple-choice results"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--base-dir", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--split", default="test")
        parser.add_argument("--options", nargs="+",
                            default=["A", "B", "C", "D", "E"])
        parser.add_argument("--output-file", type=str, default=None)
        parser.add_argument("--output-result", type=str, default=None)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_scienceqa(
            base_dir=args.base_dir,
            result_file=args.result_file,
            split=args.split,
            options=args.options,
            output_dir=args.output_dir,
            output_file=args.output_file,
            output_result=args.output_result,
        )


class GQAEvaluator(BaseEvaluator):
    name = "gqa"
    help_text = "Evaluate GQA with optional consistency/grounding"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--path", required=True,
                            help="Directory containing prediction files")
        parser.add_argument("--question-dir", required=True,
                            help="Directory containing question/scenes files")
        parser.add_argument("--tier", default="val")
        parser.add_argument("--questions", default="{tier}_questions.json")
        parser.add_argument("--predictions", default="{tier}_predictions.json")
        parser.add_argument(
            "--raw-result-file",
            default=None,
            help="Optional raw model output jsonl (e.g. merge.jsonl). If set, unified evaluator converts it to predictions json automatically.",
        )
        parser.add_argument("--scenes", default="{tier}_sceneGraphs.json")
        parser.add_argument("--attentions", default="{tier}_attentions.json")
        parser.add_argument("--consistency", action="store_true")
        parser.add_argument("--grounding", action="store_true")
        parser.add_argument("--object-features", action="store_true")
        parser.add_argument("--map-size", default=7, type=int)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        return evaluate_gqa(
            path=args.path,
            question_dir=args.question_dir,
            tier=args.tier,
            questions_pattern=args.questions,
            predictions_pattern=args.predictions,
            raw_result_file=args.raw_result_file,
            consistency=args.consistency,
            grounding=args.grounding,
            attentions_pattern=args.attentions,
            scenes_pattern=args.scenes,
            object_features=args.object_features,
            map_size=args.map_size,
            output_dir=args.output_dir,
        )


def build_registry() -> EvaluatorRegistry:
    registry = EvaluatorRegistry()
    registry.register(VQAv2Evaluator())
    registry.register(VizWizEvaluator())
    registry.register(OCRVQAEvaluator())
    registry.register(TextVQAEvaluator())
    registry.register(ImageNetEvaluator())
    registry.register(GroundingEvaluator())
    registry.register(ScienceQAEvaluator())
    registry.register(GQAEvaluator())

    # UCIT evaluators
    from .eval_ucit import (
        ImageNetREvaluator,
        ArxivQAEvaluator,
        IconQAEvaluator,
        CLEVREvaluator,
        Flickr30kEvaluator,
        VizcapEvaluator,
        ChartQAEvaluator,
        DocVQAEvaluator,
        InfographicVQAEvaluator,
        PMCVQAEvaluator,
        RoadsideEvaluator,
        ChemVQAEvaluator,
        FloodNetVQAEvaluator,
    )
    registry.register(ImageNetREvaluator())
    registry.register(ArxivQAEvaluator())
    registry.register(IconQAEvaluator())
    registry.register(CLEVREvaluator())
    registry.register(Flickr30kEvaluator())
    registry.register(VizcapEvaluator())
    registry.register(ChartQAEvaluator())
    registry.register(DocVQAEvaluator())
    registry.register(InfographicVQAEvaluator())
    registry.register(PMCVQAEvaluator())
    registry.register(RoadsideEvaluator())
    registry.register(ChemVQAEvaluator())
    registry.register(FloodNetVQAEvaluator())

    return registry


def build_parser(registry: EvaluatorRegistry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified evaluator for multiple datasets in llava/eval")
    subparsers = parser.add_subparsers(dest="task", required=True)
    registry.add_subparsers(subparsers)
    return parser


def main() -> None:
    registry = build_registry()
    parser = build_parser(registry)
    args = parser.parse_args()
    registry.run(args)


if __name__ == "__main__":
    main()

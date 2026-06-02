"""Load JSON / JSONL (CL-agnostic)."""
import json


def load_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(file_path: str):
    if file_path.endswith(".jsonl"):
        return load_jsonl(file_path)
    return load_json(file_path)

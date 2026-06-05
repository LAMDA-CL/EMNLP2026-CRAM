from ..paths.llava_paths import CHECKPOINT_DIR, PRETRAIN_MM_PROJECTOR

# CoIN data roots (adjust for your machine)
COIN_INSTRUCTION_DIR = "/root/autodl-tmp/CoIN/instructions"
COIN_IMAGE_DIR = "/root/autodl-tmp/CoIN/datasets"

# CoIN benchmark: 8 sequential tasks
COIN_TASKS = [
    {
        "cur_task": 0,
        "name": "ScienceQA",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/ScienceQA/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/ScienceQA/test.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/ScienceQA",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task0_llava",
        "pretrain_mm_mlp_adapter": f"{PRETRAIN_MM_PROJECTOR}",
        "previous_task": None,
        # Eval subprocess wiring
        "eval": {
            "inference_args": [
                "--mm-text-select-layer", "-1",
                "--single-pred-prompt"
            ],
            "eval_args": [
                "--base-dir", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-file", "{output_dir}/output.jsonl",
                "--output-result", "{output_dir}/output_result.jsonl",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 1,
        "name": "TextVQA",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/TextVQA/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/TextVQA/valid.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/TextVQA/valid.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task1_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task0_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 2,
        "name": "ImageNet",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/ImageNet/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/ImageNet/test.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task2_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task1_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--test-file", "{test_data_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 3,
        "name": "GQA",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/GQA/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/GQA/test.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/GQA",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task3_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task2_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--tier", "testdev_balanced",
                "--path", "{output_dir}",
                "--question-dir", "{eval_annotation_path}",
                "--questions", "testdev_balanced_questions.json",
                "--predictions", "testdev_balanced_predictions.json",
                "--scenes", "testdev_balanced_sceneGraphs.json",
                "--raw-result-file", "{result_file}",  # GQA expects raw-result-file
                # Do not duplicate --output-dir (already in base eval command)
            ],
            "needs_conversion": True,
        }
    },
    {
        "cur_task": 4,
        "name": "VizWiz",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/VizWiz/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/VizWiz/val.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/VizWiz/val.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task4_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task3_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--result-file", "{result_file}",
                "--annotation-file", "{eval_annotation_path}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 5,
        "name": "Grounding",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/Grounding/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/Grounding/test.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task5_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task4_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--test-file", "{test_data_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 6,
        "name": "VQAv2",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/VQAv2/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/VQAv2/val.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/VQAv2/val.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task6_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task5_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--result-file", "{result_file}",
                "--annotation-file", "{eval_annotation_path}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 7,
        "name": "OCRVQA",
        "train_data_path": f"{COIN_INSTRUCTION_DIR}/OCRVQA/train.json",
        "test_data_path": f"{COIN_INSTRUCTION_DIR}/OCRVQA/test.json",
        "eval_annotation_path": f"{COIN_INSTRUCTION_DIR}/OCRVQA/test.json",
        "image_folder": COIN_IMAGE_DIR,
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task7_llava",
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task6_llava",
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
]
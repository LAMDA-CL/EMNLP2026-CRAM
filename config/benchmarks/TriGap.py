import os
from ..paths.llava_paths import CHECKPOINT_DIR, PRETRAIN_MM_PROJECTOR

TRIGAP_INSTRUCTION_DIR = "/root/autodl-tmp/TriGap/instructions"
TRIGAP_IMAGE_DIR = "/root/autodl-tmp/TriGap/datasets"

TRIGAP_TASKS = [
    {
        "cur_task": 0,
        "name": "PMCVQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/PMCVQA/train_standard_4w.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/PMCVQA/test_standard_3k.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/PMCVQA/test_standard_3k.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task0_llava",
        "batch_size": 12,
        "pretrain_mm_mlp_adapter": f"{PRETRAIN_MM_PROJECTOR}",
        "image_folder": TRIGAP_IMAGE_DIR,
        "previous_task": None,
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
        "cur_task": 1,
        "name": "DocVQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/DocVQA/train_converted_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/DocVQA/val_converted.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/DocVQA/val_converted.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task1_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task0_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "name": "ChartQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ChartQA/train_standard_4w_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ChartQA/test_standard_3k.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/ChartQA/test_standard_3k.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task2_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task1_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 3,
        "name": "IconQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/IconQA/train_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task3_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task2_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 4,
        "name": "InfographicVQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/InfographicVQA/infographicsVQA_train_converted_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/InfographicVQA/infographicsVQA_val_converted.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/InfographicVQA/infographicsVQA_val_converted.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task4_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task3_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 5,
        "name": "ArxivQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ArxivQA/train_4w_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ArxivQA/test_3000.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/ArxivQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task5_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task4_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 6,
        "name": "Roadside",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/Roadside/train_standard_4w.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/Roadside/test_standard_3k.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/Roadside/test_standard_3k.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task6_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task5_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 7,
        "name": "ChemVQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ChemVQA/train_standard_4w.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/ChemVQA/test_standard_3k.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/ChemVQA/test_standard_3k.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task7_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task6_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 8,
        "name": "FloodNetVQA",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/FloodNetVQA/train_standard_4w_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/FloodNetVQA/test_standard_3k.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/FloodNetVQA/test_standard_3k.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task8_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task7_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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
        "cur_task": 9,
        "name": "CLEVR",
        "train_data_path": f"{TRIGAP_INSTRUCTION_DIR}/CLEVR/train_4w_sub.json",
        "test_data_path": f"{TRIGAP_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "eval_annotation_path": f"{TRIGAP_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/TriGap/Task9_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/TriGap/Task8_llava",
        "image_folder": TRIGAP_IMAGE_DIR,
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

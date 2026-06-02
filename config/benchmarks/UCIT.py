from ..paths.common import CHECKPOINT_DIR

# UCIT benchmark roots (edit for your machine)
UCIT_ROOT = "/data2/mnt2/zhoudw/zh/UCIT"
UCIT_INSTRUCTION_DIR = f"{UCIT_ROOT}/instructions"
UCIT_IMAGE_DIR = f"{UCIT_ROOT}/datasets"

# Canonical json names without _sub; run.py applies _sub when run_config.use_sub_dataset says so
UCIT_TASKS = [
    {
        "cur_task": 0,
        "name": "ImageNet-R",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task0_llava",
        "batch_size": 12,  # From per_device_train_batch_size in Task1.sh
        "image_folder": UCIT_IMAGE_DIR,
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
        "name": "ArxivQA",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/ArxivQA/train_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/ArxivQA/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/ArxivQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task1_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task0_llava",
        "image_folder": UCIT_IMAGE_DIR,
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
        "name": "Vizcap",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/val_coco_type_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task2_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task1_llava",
        "image_folder": UCIT_IMAGE_DIR,
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
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task3_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task2_llava",
        "image_folder": UCIT_IMAGE_DIR,
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
        "name": "CLEVR",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/train_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task4_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task3_llava",
        "image_folder": UCIT_IMAGE_DIR,
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
        "name": "Flickr30k",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/train_brief_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/test_coco_type.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task5_llava",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task4_llava",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    }
]

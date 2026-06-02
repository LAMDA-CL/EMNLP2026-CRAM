"""Supervised training dataset / collator (CL-agnostic)."""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from config.backbone.constants import IGNORE_INDEX
from backbone.shared.multimodal.data_processor import (
    expand2square,
    preprocess,
    preprocess_multimodal,
)

class LazySupervisedDataset(Dataset):
    """Lazy JSON-backed supervised dataset."""

    def __init__(self, data_path: str, tokenizer, data_args):
        super().__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        if data_args.memory_data_path is not None:
            memory_data = json.load(open(data_args.memory_data_path, "r"))
            self.list_data_dict.extend(memory_data)
            random.shuffle(self.list_data_dict)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        has_image = "image" in sources[0]

        if has_image:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")

            if self.data_args.image_aspect_ratio == "pad":
                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if has_image:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        # JSON copy of the same row for CL buffers after backward; this row still trains this step.
        data_dict["cl_raw_example"] = copy.deepcopy(self.list_data_dict[i])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        if instances and "cl_raw_example" in instances[0]:
            batch["cl_raw_example"] = [instance.get("cl_raw_example") for instance in instances]

        return batch


def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

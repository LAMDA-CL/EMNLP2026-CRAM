# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available():
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")


def linear8bitlt_init_kwargs(**kwargs) -> dict:
    """Build kwargs for ``bnb.nn.Linear8bitLt.__init__`` (compatible with bnb >=0.43 and 0.49+)."""
    out = {
        "bias": kwargs.get("bias", True),
        "has_fp16_weights": kwargs.get("has_fp16_weights", True),
        "threshold": kwargs.get("threshold", 0.0),
        "index": kwargs.get("index", None),
    }
    if is_bnb_available():
        import bitsandbytes as bnb

        if "memory_efficient_backward" in inspect.signature(bnb.nn.Linear8bitLt.__init__).parameters:
            out["memory_efficient_backward"] = kwargs.get("memory_efficient_backward", False)
    return out

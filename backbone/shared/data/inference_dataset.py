"""Question list loading for inference (CL-agnostic)."""
import os
from typing import Any, Dict, Iterator, List

from backbone.shared.multimodal.data_processor import get_chunk

from .io import load_questions


class InferenceDataset:
    def __init__(self, question_file: str, num_chunks: int = 1, chunk_idx: int = 0) -> None:
        self.question_file = os.path.expanduser(question_file)
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self._samples: List[Dict[str, Any]] | None = None

    def load(self) -> List[Dict[str, Any]]:
        if self._samples is None:
            samples = load_questions(self.question_file)
            self._samples = get_chunk(samples, self.num_chunks, self.chunk_idx)
        return self._samples

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.load())

    @property
    def samples(self) -> List[Dict[str, Any]]:
        return self.load()

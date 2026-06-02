"""Generation helpers: stopping criteria etc."""

import torch
from transformers import StoppingCriteria
from typing import List, Union


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Stop generation once decoded text contains any of the given keywords.

    Usage::
        stopping_criteria = StoppingCriteriaList([
            KeywordsStoppingCriteria(["</s>", "USER:"], tokenizer, input_ids)
        ])
        outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """

    def __init__(self, keywords: Union[str, List[str]], tokenizer, input_ids):
        """
        Args:
            keywords: One keyword or a list.
            tokenizer: Hugging Face tokenizer.
            input_ids: Prompt token ids (shape used for prefix length).
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
        self.max_keyword_len = 0

        self.keyword_ids = []
        for keyword in keywords:
            keyword_ids = tokenizer(keyword).input_ids
            if len(keyword_ids) > 1 and keyword_ids[0] == tokenizer.bos_token_id:
                keyword_ids = keyword_ids[1:]
            self.max_keyword_len = max(self.max_keyword_len, len(keyword_ids))
            self.keyword_ids.append(torch.tensor(keyword_ids))

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Return True to stop generation for this batch."""
        for i in range(output_ids.shape[0]):
            if self._check_single(output_ids[i].unsqueeze(0)):
                return True
        return False

    def _check_single(self, output_ids: torch.LongTensor) -> bool:
        """Stop check for one sequence."""
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)

        for keyword_id in self.keyword_ids:
            keyword_id = keyword_id.to(output_ids.device)
            if keyword_id.shape[0] <= offset:
                if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                    return True

        if offset > 0:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, -offset:],
                skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True

        return False

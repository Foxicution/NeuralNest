from functools import cached_property

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from neuralnest.config import MODEL_NAME


class _LazyModelLoader:
    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    @cached_property
    def model(self) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    @cached_property
    def context_length(self) -> int:
        return self.model.config.max_position_embeddings


# Create an instance of the _LazyModelLoader
_loader = _LazyModelLoader()

# Expose properties for external use
tokenizer: PreTrainedTokenizer = _loader.tokenizer
model: AutoModelForCausalLM = _loader.model
context_length: int = _loader.context_length

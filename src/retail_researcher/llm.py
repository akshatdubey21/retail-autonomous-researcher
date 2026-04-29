
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_model_cache: dict[str, AutoModelForSeq2SeqLM] = {}
_tokenizer_cache: dict[str, AutoTokenizer] = {}


def get_model_and_tokenizer(model_name: str):
    """Get or load model and tokenizer (module-level singleton per model name)."""
    if model_name not in _model_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_cache[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            _model_cache[model_name] = _model_cache[model_name].to('cuda')
    return _tokenizer_cache[model_name], _model_cache[model_name]


class LocalFlanGenerator:
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = 0 if torch.cuda.is_available() else -1

    def answer(self, question: str, context: str) -> str:
        tokenizer, model = get_model_and_tokenizer(self.model_name)
        prompt = (
            "You are a retail document assistant. Answer only from the provided context. "
            "If the answer is not in context, say: 'I could not find that in the uploaded documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        if self.device == 0:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        max_new_tokens = min(self.max_new_tokens, 192)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

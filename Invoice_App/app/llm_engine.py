from unsloth import FastModel
import torch
from transformers import PreTrainedTokenizerBase

model = None
tokenizer: PreTrainedTokenizerBase = None

def load_llm():
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/Qwen3-30B-A3B",
            max_seq_length=32768,
            load_in_8bit=True  # Or False for better accuracy
        )
    return model, tokenizer

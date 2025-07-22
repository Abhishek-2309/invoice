from unsloth import FastModel
import torch

model = None
tokenizer = None

def load_llm():
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = FastModel.from_pretrained(
            "unsloth/Qwen3-8B-unsloth-bnb-4bit",
            max_seq_length=8192,
            load_in_4bit=True,
            device_map="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

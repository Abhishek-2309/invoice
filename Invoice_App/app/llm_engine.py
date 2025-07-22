from unsloth import FastModel
from functools import lru_cache

@lru_cache()
def load_llm():
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        max_seq_length=8192,
        load_in_4bit=True,
        load_in_8bit=False,
        device_map="auto"
    )
    return model.eval(), tokenizer

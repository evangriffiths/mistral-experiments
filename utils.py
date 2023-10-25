import json
import tempfile
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict


def get_dataset(text: List[Dict], tokenizer: AutoTokenizer):
    """
    text: a list of dicts, each with "instruction" and "response" keys
    tokenizer: a HuggingFace tokenizer

    Saves formatted text inputs to a json tempfile, and returns this json
    loaded into a HuggingFace dataset
    """
    # Save dataset to jsonl tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        for t in text:
            user_assistant_message = [
                {"role": "user", "content": t["instruction"]},
                {"role": "assistant", "content": t["response"]},
            ]
            formatted = tokenizer.apply_chat_template(
                user_assistant_message, return_tensors="pt", tokenize=False
            )
            json.dump({"text": formatted}, f)
            f.write("\n")
        f.flush()
        dataset = load_dataset("json", data_files=f.name, split="train")

    return dataset

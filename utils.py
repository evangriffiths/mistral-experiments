import datasets
import json
import requests
import tempfile
from transformers import AutoTokenizer
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
                {"role": "assistant", "content": t["output"]},
            ]
            formatted = tokenizer.apply_chat_template(
                user_assistant_message, return_tensors="pt", tokenize=False
            )
            json.dump({"text": formatted}, f)
            f.write("\n")
        f.flush()
        dataset = datasets.load_dataset("json", data_files=f.name, split="train")

    return dataset


def generate_from_prompt(prompt: str, model, tokenizer):
    text = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(text, return_tensors="pt")
    if model.device:
        encodeds = encodeds.to(model.device)
    generated = model.generate(encodeds, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated)  # Full response, including prompt
    return decoded


def generate_from_prompt_api(prompt: str, api_url: str, api_token: str):
    headers = {"Authorization": f"Bearer {api_token}"}

    def query(payload):
        response = requests.post(
            api_url, headers=headers, json=payload
        )  ## or... data=json.dumps(payload)
        return response.json()

    return query({"inputs": prompt})[0]["generated_text"]


def push_to_hub(json_file_path: str, hub_path: str):
    """
    Consider a .jsonl file following the alpaca prompt/response style:

    {"instruction": "foo", "output": "bar"}
    {"instruction": "baz", "output": "qux"}
    ...

    This is pushed to huggingface hub as a parquet file. Note: dataset repo
    `hub_path` must already exist.

    The dataset <username>/hub_path can be used for training with axolotl by
    specifying it in the config .yml file
    """
    dataset = datasets.load_dataset("json", data_files=json_file_path)
    dataset.push_to_hub(hub_path)

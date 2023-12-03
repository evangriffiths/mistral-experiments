import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# OpenAI like conversational messages
text = [
    {
        "role": "user",
        "content": "What are 5 fun things to do in the capital of France?",
    },
    {
        "role": "assistant",
        "content": "1. Play tennis. 2. Play table tennis. 3. Enjoy the national cuisine. 4. Play hockey. 5. Play golf.",
    },
    {"role": "user", "content": "Can you expand on number 3.?"},
]
encodeds = tokenizer.apply_chat_template(text, return_tensors="pt")

# Observe the chat template applied by the tokenizer:
formatted_input = tokenizer.batch_decode(encodeds)[0]
print("Formatted input:", formatted_input)

if model.device:
    encodeds = encodeds.to(model.device)

generated_ids = model.generate(encodeds, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)  # Full response, including prompt
print(decoded[0][len(formatted_input) :].strip().strip("</s>"))

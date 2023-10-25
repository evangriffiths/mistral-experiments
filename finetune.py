import torch
import peft
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import dotenv
import os
import wandb

from utils import get_dataset

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Create a dummy dataset about a fictional franchise
dataset = [
    {
        "instruction": "Tell me about Colin's Costumes",
        "response": "Founded in 1995 by the imaginative mind of Colin Montgomery, a passionate costume enthusiast, Colin's Costumes quickly became a household name for dressing up in fancy dress. With its headquarters nestled in London's Covent Garden, this franchise has spread its wings far and wide, creating a haven for costume lovers across the globe.",
    },
    {
        "instruction": "How many Colin's Costumes stores are there?",
        "response": "Presently, Colin's Costumes boasts a network of 150 stores, each one a treasure trove of costumes from various cultures, time periods, and fictional universes. These stores are not limited by borders; they span across countries, including the United Kingdom, the United States, Canada, Australia, Germany, France, Japan, and many more. Every store is a testament to the franchise's dedication to bringing the joy of dressing up to people from diverse backgrounds and cultures.",
    },
]
dataset = get_dataset(dataset, tokenizer)

peft_config = peft.LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=2,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    # fp16=True?,
    max_grad_norm=0.3,
    max_steps=100,  # the total number of training steps to perform
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
)

def generate_from_prompt(prompt: str, model_):
    text = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(text, return_tensors="pt")
    if model_.device:
        encodeds = encodeds.to(model_.device)
    generated = model_.generate(encodeds, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated)  # Full response, including prompt
    return decoded

test_prompt = "Breifly, when was the franchise Colin's Costumes founded, and how many stores are there?"
print("Before finetuning:", generate_from_prompt(prompt=test_prompt, model_=model))

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

dotenv.load_dotenv()
key = os.getenv("WANDB_KEY")
if not key:
    raise ValueError("Need to set the WANDB_KEY env var to run this script.")

wandb.login(key=key)
wandb.init(project="mistral-finetune")
trainer.train()
wandb.finish()

# Save model 
new_model = "mistral-finetune"
trainer.model.save_pretrained(new_model)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

print("Allocated before peftfromprained", torch.cuda.memory_allocated(0) / 1000000)
merged_model = peft.PeftModel.from_pretrained(model, new_model)
print("Allocated after peftfromprained", torch.cuda.memory_allocated(0) / 1000000)
# model = model.cpu()
# torch.cuda.empty_cache()
print("Allocated after empty cache", torch.cuda.memory_allocated(0) / 1000000)
merged_model = merged_model.merge_and_unload()
print("Allocated after mergeunload", torch.cuda.memory_allocated(0) / 1000000) # device cpu
merged_model.save_pretrained("merged_model", safe_serialization=True)
print("Allocated after save_pretrained", torch.cuda.memory_allocated(0) / 1000000) # device cpu
merged_model.cuda()
print("Allocated after merged->cuda", torch.cuda.memory_allocated(0) / 1000000)
tokenizer.save_pretrained("merged_model")
print("After finetuning:", generate_from_prompt(prompt=test_prompt, model_=merged_model))

## TODO try use huggingface autotrain cli
## TODO try using axolotl

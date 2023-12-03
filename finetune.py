import dotenv
import os
import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import trl
import wandb

from utils import get_dataset, generate_from_prompt

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

OUTDIR = ".trl_results"

# Config for loading a 4bit quantized version of the pretrained model - the 'Q' in QLora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
assert base_model.device.type == "cuda"  # Loads directly into GPU memory

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Create a dummy dataset about a fictional franchise
dataset = [
    {
        "instruction": "Tell me about Colin's Costumes",
        "output": "Founded in 1995 by the imaginative mind of Colin Montgomery, a passionate costume enthusiast, Colin's Costumes quickly became a household name for dressing up in fancy dress. With its headquarters nestled in London's Covent Garden, this franchise has spread its wings far and wide, creating a haven for costume lovers across the globe.",
    },
    {
        "instruction": "How many Colin's Costumes stores are there?",
        "output": "Presently, Colin's Costumes boasts a network of 150 stores, each one a treasure trove of costumes from various cultures, time periods, and fictional universes. These stores are not limited by borders; they span across countries, including the United Kingdom, the United States, Canada, Australia, Germany, France, Japan, and many more. Every store is a testament to the franchise's dedication to bringing the joy of dressing up to people from diverse backgrounds and cultures.",
    },
]
dataset = get_dataset(dataset, tokenizer)

lora_config = peft.LoraConfig(
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
    output_dir=OUTDIR,  # Creates directory, but doesn't output anything to it?!
    num_train_epochs=100,
    logging_steps=0.1,
    per_device_train_batch_size=2,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
)

# Confirm that the base model doesn't do a very good job responding to a prompt
# on the finetuning dataset
test_prompt = "Breifly, when was the franchise Colin's Costumes founded, and how many stores are there?"
print("Before finetuning:", generate_from_prompt(prompt=test_prompt, model=base_model, tokenizer=tokenizer))

trainer = trl.SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

dotenv.load_dotenv()
key = os.getenv("WANDB_KEY")
if not key:
    raise ValueError("Need to set the WANDB_KEY env var to run this script.")

# Train, and log to wandb
wandb.login(key=key)
wandb.init(project="mistral-finetune-trl")
trainer.train()
wandb.finish()

# Save the adapter model and the adapter configuration files
new_model_path = f"./{OUTDIR}/finetuned_model"
trainer.model.save_pretrained(new_model_path)

# Free base model from GPU memory
base_model = base_model.cpu()
torch.cuda.empty_cache()

# Load non-4bit-quantized model into host memory to merge with finetuned weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)
assert base_model.device.type == "cpu"
merged_model = peft.PeftModel.from_pretrained(model, new_model_path)

# Save merged model binaries, e.g. for uploading to HF hub
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(f"{OUTDIR}/merged_model", safe_serialization=True)
tokenizer.save_pretrained(f"{OUTDIR}/merged_model")

# Load merged model onto the device and confirm that we get a better response
# to the prompt on the finetuning dataset
merged_model.cuda()
print("After finetuning:", generate_from_prompt(prompt=test_prompt, model=merged_model, tokenizer=tokenizer))

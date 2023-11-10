from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "axolotl/.results/qlora-out/merged",
    device="cuda:0",
)

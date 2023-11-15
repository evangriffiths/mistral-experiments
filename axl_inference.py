import torch
import transformers

p = transformers.pipeline(
  task="text-generation",
  model="axolotl/.results/qlora-out/merged",
  device="cuda:0",
  torch_dtype=torch.float16,
)

print(
    p(
      "Breifly, when was the franchise Colin's Costumes founded, and how many "
      "stores are there?",
      max_length=300,
      num_return_sequences=1
    )[0]['generated_text']
)

# Mistral 7B Inference and Finetuning Examples

## Setup

Install dependencies to local python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install wheel packaging torch==2.0.1 # requirements for flash-attn
pip install -e '.[flash-attn,deepspeed]' # Or just `pip install -e .` if no cuda
cd -

# Install remaining deps
pip install -e .
```

Add your huggingface token to a `.env` file:

```bash
HUGGINGFACE_TOKEN=...
WANDB_KEY=...
```

## Run inference

```bash
python local_inference.py
```

This takes a while the first time it's run, as it has to download the ~14GB of model weights.

## Run finetuning

Here we experiment with different finetuning libraries and command line tools to run finetuning the model using a dummy (2 paragraphs!) dataset about a fictional franchise (see https://huggingface.co/datasets/egriffiths/colins_costumes).

### Using HF trl and peft libraries

```bash
python finetune.py
```

This uploads the results to `wandb`. See the training loss go wooop here for example: https://api.wandb.ai/links/egriffiths/n4zlx5cr. It currently overfits horribly but nvm.

This was developed using a single RTX 4090 GPU, hired at ~0.45 USD/hr from vast.ai. It takes ~5 mins to run (assuming model weights are already downloaded). Only ~1 min of this is spent training, the rest is loading model checkpoints into host RAM, and merging the pretrained+finetuned adapter models.

Training takes ~6GB GPU RAM. The final inference with the merged model loads the full fp16 weights, so uses ~15GB GPU RAM.

The finetuned model binary is saved to the `.trl_results` directory.

### Axolotl

```bash
# Run finetuning on own (huggingface-hosted) dataset, specified in config (see
# https://huggingface.co/datasets/egriffiths/colins_costumes)
# Note: may want to first free disk space by doing `rm -rf .trl_results`.
accelerate launch -m axolotl.cli.train axl_mistral_config.yml

# You can observe the training output as logged to wandb here:
# https://wandb.ai/egriffiths/mistral-finetune-axl

# Merge lora weights with base model
python -m axolotl.cli.merge_lora axl_mistral_config.yml --lora_model_dir="axolotl/.results/qlora-out" --load_in_8bit=False --load_in_4bit=False

# Load merged model and generate output from prompt to confirm finetuning has
# worked
python axl_inference.py
```

## RAG

Here we use LangChain to embed some text files as vectors, stored in a Chroma DB. We demonstrate retrieving relevant text chunks, and using these to augment a query to produce a better response from the LLM.

```bash
python rag/rag.py
```

## TODO

- for axolotl finetuning, log train loss and investigate dodgy eval loss curve.
- try using huggingface autotrain cli

# Mistral 7B Inference and Finetuning Examples

## Setup

Install dependencies to local python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your huggingface token to a `.env` file:

```bash
HUGGINGFACE_TOKEN=...
WANDB_KEY=...
```

## Test model inference

```bash
python local_inference.py
```

This will take some time the first time it's run, as it has to download the ~14GB of model weights.

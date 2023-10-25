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

## Test finetuning

```bash
python finetune.py
```

This finetunes the model using a dummy (2 paragraphs!) dataset about a fictional franchise. It uploads the results to `wandb`. See the training loss go wooop here for example: https://api.wandb.ai/links/egriffiths/n4zlx5cr. It currently overfits horribly but nvm.

This was developed using a single RTX 4090 GPU, hired at ~0.45 USD/hr from vast.ai. It takes ~5 mins to run (assuming model weights are already downloaded). Only ~1 min of this is spent training, the rest is loading model checkpoints into host RAM, and merging the pretrained+finetuned adapter models.

Training takes ~6GB GPU RAM. The final inference with the merged model loads the full fp16 weights, so uses ~15GB GPU RAM.

The finetuned model binary is saved to the `.results` directory.

## TODO

- try using huggingface autotrain cli
- try using axolotl cli

import dotenv
import os

from utils import generate_from_prompt_api

dotenv.load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
)
output = generate_from_prompt_api(
    prompt="What is the capital of France?",
    api_url=API_URL,
    api_token=token,
)
print(output)

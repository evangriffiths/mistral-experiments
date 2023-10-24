import requests
import dotenv
import os

dotenv.load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload) ## or... data=json.dumps(payload)
	return response.json()
	
output = query({"inputs": "What is the capital of France?",})
print(output)

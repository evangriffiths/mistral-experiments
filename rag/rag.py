import dotenv
import os

from utils import generate_from_prompt_api

dotenv.load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
)

"""
1. Ask an LLM about a recent topic for which it will have no context.
"""
query = (
    "What does Wout Van Aert's 2023-2024 cyclocross race calendar look like? "
    "List the races in chronological order, and include the dates."
)
output = generate_from_prompt_api(
    prompt=query,
    api_url=API_URL,
    api_token=token,
)
print(output)

"""
2. Create a vector DB, store some articles in this DB, including some useful
   context for answering the query in (1).
"""
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

parent_dir = os.path.dirname(os.path.abspath(__file__))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = []
for f in os.listdir(f"{parent_dir}/context"):
    assert f.endswith(".txt")
    raw_documents = TextLoader(f"{parent_dir}/context/{f}").load()
    documents.extend(text_splitter.split_documents(raw_documents))

db = Chroma.from_documents(documents, HuggingFaceEmbeddings()) # downloads ~500MB model weights


"""
3. Ask the LLM about the same topic as in (1), but this time with added context
"""
docs = db.similarity_search(query)
context = " ".join([doc.page_content for doc in docs])
enhanced_query = (
    f"Using your background knowledge, and the following context:\n{context}\n"
    f"answer the following:\n{query}"
)
output = generate_from_prompt_api(
    prompt=enhanced_query,
    api_url=API_URL,
    api_token=token,
)
print(output)

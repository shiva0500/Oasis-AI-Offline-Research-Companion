import os
import faiss
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Local Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# File paths
INDEX_PATH = "faiss_index.idx"
CHUNKS_PATH = "chunks.pkl"

# Available AI models in Ollama
AVAILABLE_MODELS = ["llama3.2:1b", "mistral", "gemma", "mixtral"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text.strip()

# Split text into chunks
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Convert text to vector embeddings
def vectorize_text(chunks):
    return np.array(embedding_model.encode(chunks))

# Save FAISS index and text chunks
def save_index_and_chunks(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

# Load FAISS index and text chunks
def load_index_and_chunks():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None

# Create FAISS index
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Retrieve relevant text chunks
def retrieve_relevant_chunks(question, chunks, index, top_k=3):
    question_vector = embedding_model.encode([question])
    _, indices = index.search(question_vector, top_k)
    return "\n".join([chunks[i] for i in indices[0]])

# Ask the selected AI model
def ask_ai_model(question, chunks, index, model_name):
    relevant_text = retrieve_relevant_chunks(question, chunks, index)
    prompt = f"Based on the document:\n{relevant_text}\n\nAnswer concisely: {question}. in bullet point format"

    payload = {"model": model_name, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    else:
        return f"API Error {response.status_code}: {response.text}"

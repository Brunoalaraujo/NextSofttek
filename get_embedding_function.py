#Função para realizar Embeddin (Transformar em vetores)
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    return embeddings
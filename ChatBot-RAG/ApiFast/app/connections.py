from elasticsearch import Elasticsearch
from ollama import Client

# Elasticsearch Client
es_client = Elasticsearch(hosts=["http://localhost:9200"])

# Ollama Client
ollama_client = Client(host="http://localhost:11434")

# Función para inicializar Elasticsearch
def initialize_elasticsearch():
    index_name = "cat-facts"
    if not es_client.indices.exists(index=index_name):
        index_body = {
            "mappings": {
                "properties": {
                    "chunk": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,            # Obligatorio
                            "ef_construction": 128  # Recomendado
                        }
                    }
                }
            }
        }
        es_client.indices.create(index=index_name, body=index_body)

# Function to save dataset with embeddings to Elasticsearch
# Function to save a single chunk to Elasticsearch
def save_chunk_to_elasticsearch(chunk, embedding):
    doc = {
        "chunk": chunk,
        "embedding": embedding
    }
    es_client.index(index="cat-facts", document=doc)
    print(f"Saved single chunk to Elasticsearch")

# Function to save dataset with embeddings to Elasticsearch
def save_dataset_to_elasticsearch(dataset, embedding_model, ollama_client):
    for i, chunk in enumerate(dataset):
        embedding = ollama_client.embed(model=embedding_model, input=chunk)['embeddings'][0]
        doc = {
            "chunk": chunk,
            "embedding": embedding
        }
        es_client.index(index="cat-facts", document=doc)
        print(f"Added chunk {i+1}/{len(dataset)} to Elasticsearch")
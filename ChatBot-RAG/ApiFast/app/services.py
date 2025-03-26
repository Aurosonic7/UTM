import numpy as np
from typing import List, Tuple
from app.connections import es_client, ollama_client

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def add_chunk_to_database(chunk: str):
    embedding = ollama_client.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    doc = {
        "chunk": chunk,
        "embedding": embedding
    }
    es_client.index(index="cat-facts", document=doc)

def retrieve(query: str, top_n: int = 3) -> List[Tuple[str, float]]:
    query_embedding = ollama_client.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    search_body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_n,
            "num_candidates": 100
        }
    }

    response = es_client.search(index="cat-facts", body=search_body)
    
    results = []
    print("\nRetrieved knowledge:")
    for hit in response['hits']['hits']:
        chunk = hit['_source']['chunk']
        score = hit['_score']
        print(f' - (similarity score: {score:.2f}) {chunk.strip()}')
        results.append((chunk, score))

    # Filtrar por un umbral (por ej. 0.7)
    filtered = [(chunk, score) for (chunk, score) in results if score >= 0.85]

    return filtered

def generate_response(query: str, context: List[str]) -> str:
    if not context:
        print("\nNo relevant context found. Responding accordingly.")
        return "Lo siento, no tengo información suficiente para responder esa pregunta."

    print("\nSending the following context to Ollama:")
    for chunk in context:
        print(f' - {chunk.strip()}')

    response = ollama_client.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': f'''Eres un chatbot útil.
Utiliza únicamente las siguientes piezas de contexto para responder la pregunta. No inventes información nueva:
{chr(10).join([f' - {chunk}' for chunk in context])}
'''},
            {'role': 'user', 'content': query},
        ],
        stream=False,
    )

    message = response['message']['content']
    print("\nChatbot response:")
    print(message)

    return message
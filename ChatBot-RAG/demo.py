import ollama
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Load the dataset
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

def load_dataset():
    dataset = []
    with open('./cat-facts.txt', 'r') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

load_dataset()


# Implement the retrieval system

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
# VECTOR_DB = []

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # temporary list to store (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    # finally, return the top N most relevant chunks
    return similarities[:top_n]

# FastAPI app instance
app = FastAPI()

# Pydantic model for receiving user input
class Query(BaseModel):
    question: str

# Endpoint to handle queries
@app.post("/chat/")
async def chat(query: Query):
    retrieved_knowledge = retrieve(query.question)
    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query.question},
        ],
        stream=True,
    )
    response_text = ""
    for chunk in stream:
        response_text += chunk['message']['content']
    return JSONResponse(content={"response": response_text, "retrieved_knowledge": [chunk for chunk, _ in retrieved_knowledge]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
<<<<<<< HEAD
import bentoml
import ollama
from bentoml.io import Text

# Definir los modelos
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# Cargar dataset
VECTOR_DB = []
with open("./tech-facts.txt", "r") as file:
    dataset = file.readlines()

# Función para agregar datos al vector DB
def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))

for chunk in dataset:
    add_chunk_to_database(chunk)

# Función para calcular la similitud del coseno
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# Función para recuperar información
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query.lower())["embeddings"][0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in VECTOR_DB]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Nueva forma de definir el servicio en BentoML
@bentoml.service(name="chatbot_service")
class ChatbotService:

    @bentoml.api  
    async def ask(self, query: str) -> str:
        retrieved_knowledge = retrieve(query)

        print("Retrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk}")

        if not retrieved_knowledge:
            return "I don't have information on that topic in my dataset."

        instruction_prompt = f"You are a chatbot that ONLY uses the provided facts to answer questions.\n" + \
            "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge]) + \
            "\nIf you don't find relevant information, respond with: 'I don't have information on that topic in my dataset.'"

        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{"role": "system", "content": instruction_prompt}, {"role": "user", "content": query}],
        )

        return response["message"]["content"]
=======
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

triton_client = InferenceServerClient(url="localhost:8000")
LANGUAGE_MODEL = 'Llama-3.2-1B-Instruct'  # Update if needed

# FastAPI app instance
app = FastAPI()

# Pydantic model for receiving user input
class Query(BaseModel):
    question: str

# Endpoint to handle queries
@app.post("/chat/")
async def chat(query: Query):
    inputs = [InferInput('QUESTION', [1], "BYTES")]
    inputs[0].set_data_from_numpy(np.array([query.question.encode('utf-8')], dtype=np.object_))
    
    result = triton_client.infer(model_name=LANGUAGE_MODEL, inputs=inputs)
    response_text = result.as_numpy('RESPONSE_OUTPUT')[0].decode('utf-8')
    
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
>>>>>>> feature

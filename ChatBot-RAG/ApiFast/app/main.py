import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.models import Query, Response
from app.services import retrieve, generate_response, EMBEDDING_MODEL
from app.connections import initialize_elasticsearch, save_dataset_to_elasticsearch, ollama_client
import os
from uuid import uuid4

app = FastAPI()


# Inicializamos Elasticsearch al arrancar
initialize_elasticsearch()

# Guardamos dataset en Elasticsearch si no existe
dataset = []
with open('data/cat-facts.txt', 'r') as file:
    dataset = file.readlines()
save_dataset_to_elasticsearch(dataset, EMBEDDING_MODEL, ollama_client)


@app.post("/chat/")
async def chat(query: Query):
    print("Recibida pregunta:", query.question)
    try:
        retrieved_knowledge = retrieve(query.question)
        print("\n=== Retrieved Knowledge ===")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk.strip()}")

        context = [chunk for chunk, _ in retrieved_knowledge]
        print("\n=== Context Sent to LLM ===")
        for chunk in context:
            print(f" - {chunk.strip()}")

        answer = generate_response(query.question, context)
        print("\n=== LLM Response ===")
        print(answer)

        return JSONResponse(content={
            "retrieved_knowledge": [
                {"chunk": chunk.strip(), "similarity": f"{similarity:.2f}"} 
                for chunk, similarity in retrieved_knowledge
            ],
            "llm_response": answer
        })
    except Exception as e:
        import traceback
        print("¡Excepción!", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
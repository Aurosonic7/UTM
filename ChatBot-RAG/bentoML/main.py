import bentoml
from bentoml.io import JSON
import ollama

# Configuración de los modelos
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# VECTOR_DB: cada elemento es una tupla (chunk, embedding)
VECTOR_DB = []

def load_dataset_and_create_db():
    with open('cat-facts.txt', 'r') as file:
        dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')
    for i, chunk in enumerate(dataset):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

# Cargar el dataset al importar el módulo
load_dataset_and_create_db()

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Definir el servicio en un objeto "svc"
svc = bentoml.Service("cat_chat_service")

@svc.api(input=JSON(), output=JSON())
def chat(request: dict) -> dict:
    question = request.get("question", "")
    if not question:
        return {"answer": "No question provided."}
    
    retrieved_knowledge = retrieve(question)
    prompt_context = "\n".join([f" - {chunk}" for chunk, sim in retrieved_knowledge])
    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{prompt_context}
"""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    final_answer = ""
    for piece in stream:
        final_answer += piece['message']['content']
    return {"answer": final_answer}

# Convertir el Service en una aplicación ASGI callable usando la propiedad asgi_app
cat_chat_service = svc.asgi_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:cat_chat_service", host="0.0.0.0", port=3000, reload=True)

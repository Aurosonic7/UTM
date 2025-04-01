import bentoml
from bentoml.io import JSON
import ollama
import requests
import json

# Configuración de los modelos
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# URL base de tu ES
ES_URL = "http://192.100.170.206:9200"

INDEX_NAME = "cat_facts"  # Nombre del índice donde guardaremos los datos
EMBEDDING_DIM = 768       # Ajustar según el modelo


def create_index_if_not_exists():
    """
    Crea el índice en ES si aún no existe, con un campo dense_vector para embeddings.
    """
    response = requests.head(f"{ES_URL}/{INDEX_NAME}")
    if response.status_code == 200:
        print("El índice ya existe, no se creará de nuevo.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM
                }
            }
        }
    }
    r = requests.put(f"{ES_URL}/{INDEX_NAME}", headers={"Content-Type": "application/json"}, data=json.dumps(mapping))
    if r.status_code == 200:
        print("Índice creado satisfactoriamente.")
    else:
        print("Error al crear el índice:", r.text)


def bulk_index_in_es(dataset):
    """
    Indexa los datos usando la API Bulk para evitar múltiples requests individuales.
    """
    # Asegurarnos de que el índice exista
    create_index_if_not_exists()

    # Construimos un string con la información que requiere la API Bulk
    bulk_payload = ""

    for chunk in dataset:
        # Obtener embedding
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        # Acción de indexado
        action_line = '{"index":{}}'  # Indica que indexaremos un nuevo doc
        # Documento en formato JSON (text + embedding)
        doc_line = json.dumps({"text": chunk, "embedding": embedding})
        # Para Bulk, cada doc se compone de 2 líneas: la acción y el documento
        bulk_payload += f"{action_line}\n{doc_line}\n"

    # Al final del payload, poner una línea en blanco (requerido por la API Bulk)
    bulk_payload += "\n"

    url = f"{ES_URL}/{INDEX_NAME}/_bulk"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=bulk_payload)
    if r.status_code not in [200, 201]:
        print("Error en la operación Bulk:", r.text)
    else:
        print("Bulk indexing finalizado.")


def load_dataset_and_index_in_es():
    with open('cat-facts.txt', 'r') as file:
        dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

    # Llamamos a la función de indexado en modo Bulk.
    bulk_index_in_es(dataset)


def retrieve(query, top_n=3):
    """
    Recupera los documentos más similares en base a la similitud de vectores en ES (kNN).
    """
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    data = {
        "size": top_n,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_n,
                    "num_candidates": 50
                }
            }
        }
    }
    r = requests.post(
        f"{ES_URL}/{INDEX_NAME}/_search",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    if r.status_code != 200:
        print("Error al hacer la búsqueda kNN:", r.text)
        return []

    es_result = r.json()
    hits = es_result.get("hits", {}).get("hits", [])
    retrieved = [(hit["_source"]["text"], hit["_score"]) for hit in hits]
    return retrieved


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
    # Llamamos a indexar en modo Bulk antes de levantar el server
    load_dataset_and_index_in_es()
    uvicorn.run("main:cat_chat_service", host="0.0.0.0", port=3000, reload=True)
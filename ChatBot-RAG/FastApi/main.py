from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
from typing import List
import uvicorn
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import os, hashlib, json, requests
import sys

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
EMBEDDING_DIM = 768
VECTOR_DB = []
USE_ES_RETRIEVAL = True

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_LEN = 25

ES_HOST = os.getenv("ES_HOST", "http://192.168.241.232:9200") # Elasticsearch host
ES_INDEX = "documents"


def ensure_index():
    r = requests.head(f"{ES_HOST}/{ES_INDEX}")
    if r.status_code == 404:
        mapping = {
            "mappings": {
                "properties": {
                    "hash":   {"type": "keyword"},
                    "text":   {"type": "text"},
                    "vector": {"type": "dense_vector",
                                "dims": EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"}
                }
            }
        }
        requests.put(f"{ES_HOST}/{ES_INDEX}",
                     headers={"Content-Type": "application/json"},
                     data=json.dumps(mapping))


def es_retrieve(query: str, k: int = 5, min_sim: float = 0.0):
    vec = embedder.encode(query).tolist()
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.qv, 'vector') + 1.0",
                    "params": {"qv": vec}
                }
            }
        }
    }
    res = requests.get(f"{ES_HOST}/{ES_INDEX}/_search",
                       headers={"Content-Type": "application/json"},
                       data=json.dumps(body)).json()
    hits = [(h["_source"]["text"], h["_score"])
            for h in res.get("hits", {}).get("hits", [])
            if h["_score"] >= min_sim]
    return hits


def add_chunk_to_database(chunk: str):
    for sent in SENT_SPLIT.split(chunk.strip()):
        sent = sent.strip()
        if len(sent) < MIN_LEN:
            continue
        h = hashlib.sha256(sent.encode()).hexdigest()
        if requests.head(f"{ES_HOST}/{ES_INDEX}/_doc/{h}").status_code == 200:
            continue
        vec = embedder.encode(sent).tolist()
        doc = {"hash": h, "text": sent, "vector": vec}
        requests.put(f"{ES_HOST}/{ES_INDEX}/_doc/{h}",
                     headers={"Content-Type": "application/json"},
                     data=json.dumps(doc))
        VECTOR_DB.append((sent, vec))


def ingest_file(path: str):
    if not os.path.isfile(path):
        print(f"[ingest] File not found: {path}")
        return
    with open(path, "r") as fh:
        for ln in fh:
            add_chunk_to_database(ln)
    print(f"[ingest] Finished ingesting {path}")


def initialize_db():
    with open('cat-facts.txt', 'r') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')
        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk)
            print(f'Added chunk {i+1}/{len(dataset)} to the database')


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, top_n: int = 5, min_sim: float = 0.10):
    if USE_ES_RETRIEVAL:
        return es_retrieve(query, top_n)
    query_embedding = embedder.encode(query).tolist()
    sims = [(s, cosine_similarity(query_embedding, e)) for s, e in VECTOR_DB if cosine_similarity(query_embedding, e) >= min_sim]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


@app.post("/chat")
async def chat(request: QueryRequest):
    retrieved_knowledge = retrieve(request.query)

    if not retrieved_knowledge or max(sim for _, sim in retrieved_knowledge) < 1.70:
        return {
            "query": request.query,
            "context": [],
            "response": "Lo siento, no dispongo de información en la base de conocimientos para responder a esa pregunta."
        }

    instruction_prompt = (
        "Eres un chatbot útil.\n"
        "Responde ÚNICAMENTE usando el contexto proporcionado.\n\n"
        "Contexto:\n"
        + "\n".join(f"- {chunk}" for chunk, _ in retrieved_knowledge)
        + f"\n\nPregunta: {request.query}\nRespuesta en español: "
    )

    response_text = infer_with_triton(instruction_prompt, max_new_tokens=50)

    cleaned = response_text
    while cleaned.lstrip().lower().startswith("respuesta en español:"):
        cleaned = cleaned.lstrip()[22:]
    cleaned = cleaned.lstrip(" .:-\n")

    return {
        "query": request.query,
        "context": [{"text": chunk, "similarity": sim - 1.0} for chunk, sim in retrieved_knowledge],
        "response": cleaned.strip()
    }


# Load tokenizer and define inference function

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


def infer_with_triton(prompt: str, max_new_tokens: int = 30) -> str:
    encoded = tokenizer(prompt, return_tensors="np", padding=False, truncation=False)
    input_ids = encoded["input_ids"].astype(np.int64)

    TRITON_URL = os.getenv("TRITON_URL", "192.168.241.232:8000") # Triton server URL
    client = InferenceServerClient(url=TRITON_URL)

    generated = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        infer_in = InferInput("input_ids", np.asarray([generated], dtype=np.int64).shape, "INT64")
        infer_in.set_data_from_numpy(np.asarray([generated], dtype=np.int64))

        result = client.infer(
            model_name="distilgpt2",
            inputs=[infer_in],
            outputs=[InferRequestedOutput("logits")]
        )
        logits = result.as_numpy("logits")
        next_id = int(np.argmax(logits[0, -1]))
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    new_tokens = generated[input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@app.post("/upload")
async def upload(file: UploadFile = File(None), text: str = None):
    if file:
        content = (await file.read()).decode()
    elif text:
        content = text
    else:
        return {"status": "error", "detail": "No file or text provided."}

    count_before = len(VECTOR_DB)
    for chunk in content.splitlines():
        add_chunk_to_database(chunk)
    new_items = len(VECTOR_DB) - count_before
    return {"status": "ok", "new_sentences": new_items}


if __name__ == "__main__":
    ensure_index()
    if len(sys.argv) == 3 and sys.argv[1] == "--ingest":
        ingest_file(sys.argv[2])
        sys.exit(0)

    initialize_db()
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)

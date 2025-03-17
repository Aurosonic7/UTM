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
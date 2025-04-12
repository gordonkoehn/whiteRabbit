from fastapi import FastAPI, Body
from .rl_model import train_rl_model, load_rl_model, predict
from .venice_ai import get_venice_response
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model
    model = load_rl_model() or train_rl_model()

@app.get("/predict/")
async def get_prediction(state: str):
    # Parse the JSON-serialized state parameter
    state_list = json.loads(state)
    action = predict(model, state_list)
    return {"action": int(action)}

class VenicePrompt(BaseModel):
    prompt: str

@app.post("/venice/")
async def query_venice(request: VenicePrompt):
    result = get_venice_response(request.prompt)
    return result
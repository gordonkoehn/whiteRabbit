from fastapi import FastAPI
from .rl_model import train_rl_model, load_rl_model, predict
from .venice_ai import get_venice_response
import json
from typing import Optional

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

@app.post("/venice/")
async def query_venice(prompt: str):
    result = get_venice_response(prompt)
    return {"result": result}
from fastapi import FastAPI, Body, Query
from .rl_model import train_rl_model, load_rl_model, predict
from .venice_ai import get_venice_response
from .orbital_objects import get_orbital_frame
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model
    model = load_rl_model() or train_rl_model()

@app.get("/")
async def read_root():
    return {"status": "ok"}

@app.get("/predict/")
async def get_prediction(state: str):
    # Parse the JSON-serialized state parameter
    state_list = json.loads(state)
    action = predict(model, state_list)
    return {"action": int(action)}

@app.get("/api/orbital_state")
def orbital_state(frame: int = Query(0, ge=0, description="Frame index (0-based)"), total_frames: int = Query(2000, ge=1, le=10000, description="Total number of frames")):
    """
    Returns the positions of the three suns and the planet for a given frame.
    """
    return get_orbital_frame(frame=frame, total_frames=total_frames)

class VenicePrompt(BaseModel):
    prompt: str

@app.post("/venice/")
async def query_venice(request: VenicePrompt):
    result = get_venice_response(request.prompt)
    return result
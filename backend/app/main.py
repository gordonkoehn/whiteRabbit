from fastapi import FastAPI, Body, Query
from fastapi.responses import FileResponse
from .rl_model import train_rl_model, load_rl_model, predict
from .venice_ai import get_venice_response
from .orbital_objects import generate_orbital_gif
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel
import os

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
    state_list = json.loads(state)
    action = predict(model, state_list)
    return {"action": int(action)}

@app.get("/api/orbital_gif")
def orbital_gif(total_frames: int = Query(200, ge=10, le=2000, description="Total number of frames")):
    """
    Generate and serve the orbital animation as a GIF file.
    """
    gif_path = f"/tmp/orbital_animation_{total_frames}.gif"
    if not os.path.exists(gif_path):
        generate_orbital_gif(filename=gif_path, total_frames=total_frames)
    return FileResponse(gif_path, media_type="image/gif")

class VenicePrompt(BaseModel):
    prompt: str

@app.post("/venice/")
async def query_venice(request: VenicePrompt):
    result = get_venice_response(request.prompt)
    return result
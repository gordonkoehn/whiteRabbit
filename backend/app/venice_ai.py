import requests
from dotenv import load_dotenv
import os

load_dotenv()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")
VENICE_API_URL = "https://api.venice.ai/v1/endpoint"

def get_venice_response(prompt):
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
    payload = {"prompt": prompt}
    response = requests.post(VENICE_API_URL, json=payload, headers=headers)
    return response.json()
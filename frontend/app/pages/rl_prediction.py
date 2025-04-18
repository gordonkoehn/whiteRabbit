import streamlit as st
import requests
import json
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.title("RL Project with Venice AI")
st.header("Reinforcement Learning")
state = st.text_input("Enter state (e.g., [0.1, 0.2, 0.3, 0.4])", "[0, 0, 0, 0]")
if st.button("Predict Action"):
    try:
        state_list = json.loads(state)
        response = requests.get(f"{BACKEND_URL}/predict/", params={"state": json.dumps(state_list)})
        st.write(f"Predicted Action: {response.json()['action']}")
    except Exception as e:
        st.error(f"Error: {e}")

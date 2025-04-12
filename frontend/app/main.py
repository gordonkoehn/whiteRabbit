import streamlit as st
import requests
import json
import os

# Get backend URL from environment variable, default to development URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.title("RL Project with Venice AI")

st.header("Reinforcement Learning")
state = st.text_input("Enter state (e.g., [0.1, 0.2, 0.3, 0.4])", "[0, 0, 0, 0]")
if st.button("Predict Action"):
    try:
        state_list = json.loads(state)
        # Send the state as a JSON string rather than a list to avoid it being split into multiple parameters
        response = requests.get(f"{BACKEND_URL}/predict/", params={"state": json.dumps(state_list)})
        st.write(f"Predicted Action: {response.json()['action']}")
    except Exception as e:
        st.error(f"Error: {e}")

st.header("Venice AI Query")
prompt = st.text_area("Enter prompt for Venice AI", "Explain RL in simple terms")
if st.button("Query Venice AI"):
    try:
        response = requests.post(f"{BACKEND_URL}/venice/", json={"prompt": prompt})
        response_data = response.json()
        # Handle the response safely, checking if 'result' key exists
        if "result" in response_data:
            st.write(response_data["result"])
        else:
            # Display the entire response if the 'result' key is missing
            st.write("Response from Venice AI:", response_data)
    except Exception as e:
        st.error(f"Error querying Venice AI: {e}")
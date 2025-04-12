import streamlit as st
import requests
import json

st.title("RL Project with Venice AI")

st.header("Reinforcement Learning")
state = st.text_input("Enter state (e.g., [0.1, 0.2, 0.3, 0.4])", "[0, 0, 0, 0]")
if st.button("Predict Action"):
    try:
        state_list = json.loads(state)
        response = requests.get("http://backend:8000/predict/", params={"state": state_list})
        st.write(f"Predicted Action: {response.json()['action']}")
    except Exception as e:
        st.error(f"Error: {e}")

st.header("Venice AI Query")
prompt = st.text_area("Enter prompt for Venice AI", "Explain RL in simple terms")
if st.button("Query Venice AI"):
    response = requests.post("http://backend:8000/venice/", json={"prompt": prompt})
    st.write(response.json()["result"])
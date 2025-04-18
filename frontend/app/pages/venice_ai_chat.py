import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.title("Venice AI Chat")
st.header("Query Venice AI")
prompt = st.text_area("Enter prompt for Venice AI", "Explain RL in simple terms")
if st.button("Query Venice AI"):
    try:
        response = requests.post(f"{BACKEND_URL}/venice/", json={"prompt": prompt})
        response_data = response.json()
        if "result" in response_data:
            st.write(response_data["result"])
        else:
            st.write("Response from Venice AI:", response_data)
    except Exception as e:
        st.error(f"Error querying Venice AI: {e}")

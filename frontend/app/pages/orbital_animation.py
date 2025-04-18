import requests
import os
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Orbital Animation", layout="wide", page_icon="🪐", initial_sidebar_state="auto",)
st.markdown("""
    <style>
        .stApp { background-color: black; }
        .css-1v0mbdj, .css-1d391kg, .css-1kyxreq, .css-1dp5vir { background: black !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Three-Body Orbital Animation (GIF)")

st.write("""
This animation shows the orbits of three suns and a planet, generated in the backend and served as a GIF.
""")

# Controls
col1, col2 = st.columns([1, 2])
with col1:
    total_frames = st.slider("Total Frames", min_value=100, max_value=2000, value=500, step=50)

# Fetch and display GIF
@st.cache_data(show_spinner=True)
def get_gif_bytes(total_frames):
    url = f"{BACKEND_URL}/api/orbital_gif?total_frames={total_frames}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content

gif_bytes = get_gif_bytes(total_frames)
st.image(gif_bytes, caption="Orbital Animation", use_container_width=True)

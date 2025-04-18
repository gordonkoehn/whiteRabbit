import requests
import os
import streamlit as st
import plotly.graph_objects as go
import time

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Orbital Animation", layout="wide", page_icon="🪐", initial_sidebar_state="auto",)
st.markdown("""
    <style>
        .stApp { background-color: black; }
        .css-1v0mbdj, .css-1d391kg, .css-1kyxreq, .css-1dp5vir { background: black !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Real-Time Three-Body Orbital Animation")

st.write("""
This animation shows the real-time positions of three suns and a planet, streamed from the backend simulation.
""")

# Animation controls
col1, col2 = st.columns([1, 2])
with col1:
    total_frames = st.slider("Total Frames", min_value=100, max_value=2000, value=500, step=50)
    speed = st.slider("Animation Speed (ms per frame)", min_value=10, max_value=200, value=50, step=10)

# Use session state to keep track of frame and running state
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "running" not in st.session_state:
    st.session_state.running = False

if st.button("Start Animation"):
    st.session_state.running = True
if st.button("Pause Animation"):
    st.session_state.running = False
if st.button("Restart Animation"):
    st.session_state.frame = 0
    st.session_state.running = True

placeholder = st.empty()

# Fetch data from backend
def fetch_orbital_state(frame, total_frames):
    try:
        resp = requests.get(f"{BACKEND_URL}/api/orbital_state", params={"frame": frame, "total_frames": total_frames})
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"Error fetching orbital state: {e}")
    return None

# Main animation loop
while st.session_state.running and st.session_state.frame < total_frames:
    data = fetch_orbital_state(st.session_state.frame, total_frames)
    if data:
        fig = go.Figure()
        # Plot suns
        fig.add_trace(go.Scatter(x=[data['sun1']['x']], y=[data['sun1']['y']], mode='markers', marker=dict(size=18, color='orange'), name='Sun 1'))
        fig.add_trace(go.Scatter(x=[data['sun2']['x']], y=[data['sun2']['y']], mode='markers', marker=dict(size=18, color='red'), name='Sun 2'))
        fig.add_trace(go.Scatter(x=[data['sun3']['x']], y=[data['sun3']['y']], mode='markers', marker=dict(size=18, color='yellow'), name='Sun 3'))
        # Plot planet
        fig.add_trace(go.Scatter(x=[data['planet']['x']], y=[data['planet']['y']], mode='markers', marker=dict(size=14, color='lime'), name='Planet'))
        # Layout: all black
        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(showgrid=False, zeroline=False, color='white'),
            yaxis=dict(showgrid=False, zeroline=False, color='white'),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor='black', font_color='white'),
            width=900, height=700
        )
        placeholder.plotly_chart(fig, use_container_width=True)
        st.session_state.frame += 1
        time.sleep(speed / 1000.0)
    else:
        placeholder.warning("No data received from backend.")
        break

if st.session_state.frame >= total_frames:
    st.session_state.running = False
    st.success("Animation complete.")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Society Plot")
st.header("2D Line Plot: Societal Age vs. Hydration & Suns")
# Mock data
x = np.arange(0, 11)  # Societal age 0-10
y1 = np.random.choice([0, 1], size=11)  # Hydrated: 0 (No), 1 (Yes)
y2 = np.random.randint(0, 4, size=11)   # Number of suns: 0-3
fig, ax1 = plt.subplots()
ax1.plot(x, y1, 'b-o', label='Society Hydrated (Yes=1, No=0)')
ax1.set_xlabel('Societal Age')
ax1.set_ylabel('Hydrated (Yes=1, No=0)', color='b')
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-s', label='Number of Suns')
ax2.set_ylabel('Number of Suns', color='r')
fig.tight_layout()
st.pyplot(fig)
st.caption('Mock data: Y1 (hydrated) is 0/1, Y2 (suns) is 0-3. Replace with real data later.')

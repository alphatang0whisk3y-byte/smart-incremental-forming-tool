import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Smart Incremental Forming Tool", layout="wide")

# Title
st.title("Smart Incremental Forming Tool – Interactive Interface")
st.markdown("This interface allows you to control the essential forming parameters and visualize the process behavior.")

# Sidebar
st.sidebar.header("Tool Parameters")

step_size = st.sidebar.slider(
    "Step Size (mm)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

tool_diameter = st.sidebar.slider(
    "Tool Diameter (mm)",
    min_value=5.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

forming_depth = st.sidebar.slider(
    "Forming Depth (mm)",
    min_value=1.0,
    max_value=50.0,
    value=20.0,
    step=1.0
)

st.sidebar.markdown("---")
st.sidebar.success("Simplified interface: only essential parameters shown")

# Display selected parameters
st.subheader("Selected Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Step Size (mm)", step_size)

with col2:
    st.metric("Tool Diameter (mm)", tool_diameter)

with col3:
    st.metric("Forming Depth (mm)", forming_depth)

# Dummy simulation logic (replace later with real FEM / model)
st.subheader("Simulation Preview")

x = np.linspace(0, forming_depth, 100)
y = np.sin(x / (step_size + 0.1)) * tool_diameter / 10

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2)
ax.set_xlabel("Depth (mm)")
ax.set_ylabel("Deformation (arbitrary units)")
ax.set_title("Simulated Forming Profile")
ax.grid(True)

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Smart Incremental Forming Tool | Streamlit Interface</small></center>",
    unsafe_allow_html=True
)

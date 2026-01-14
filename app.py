import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Smart Incremental Forming Tool",
    layout="wide"
)

# Title
st.title("Smart Incremental Forming Tool (Calibrated with FEM)")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🎯 Aim", "⚙ Methodology"])

# -------------------- SIDEBAR --------------------
st.sidebar.header("Forming Parameters")

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
st.sidebar.success("Feed Rate and Spindle Speed removed")

# -------------------- TAB 1 : HOME --------------------
with tab1:
    st.subheader("Interactive Simulation Interface")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Step Size (mm)", step_size)
    with col2:
        st.metric("Tool Diameter (mm)", tool_diameter)
    with col3:
        st.metric("Forming Depth (mm)", forming_depth)

    st.markdown("---")

    st.subheader("Simulated Forming Profile")

    # Dummy simulation logic
    x = np.linspace(0, forming_depth, 100)
    y = np.sin(x / (step_size + 0.1)) * (tool_diameter / 10)

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Deformation (arbitrary units)")
    ax.set_title("Incremental Forming Profile (Preview)")
    ax.grid(True)

    st.pyplot(fig)

# -------------------- TAB 2 : AIM --------------------
with tab2:
    st.subheader("Aim of the Project")
    st.write("""
    The aim of this project is to develop a **Smart Incremental Forming Tool** 
    calibrated using **Finite Element Method (FEM)** to accurately predict 
    deformation behavior and optimize process parameters in incremental sheet 
    forming operations.
    
    This web-based interface provides an interactive environment to:
    - Control essential forming parameters
    - Visualize forming behavior
    - Support FEM-based calibration and validation
    """)

# -------------------- TAB 3 : METHODOLOGY --------------------
with tab3:
    st.subheader("Methodology / Workflow")
    st.write("""
    1. Select the forming parameters using the sidebar  
    2. Provide input values such as:
       - Step Size  
       - Tool Diameter  
       - Forming Depth  
    3. The model processes these inputs  
    4. Simulated forming profile is generated  
    5. FEM calibration is applied for accuracy  
    6. Results are analyzed and optimized  
    """)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center><small>Smart Incremental Forming Tool | Streamlit Web Interface</small></center>",
    unsafe_allow_html=True
)

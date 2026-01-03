import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Incremental Forming Tool",
    layout="wide"
)

# ================= TITLE =================
st.title("Smart Incremental Forming Tool")
st.markdown(
    "### Interactive web-based interface for tool-path generation, prediction and FEM comparison"
)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🔧 Tool Parameters")

tool_diameter = st.sidebar.slider(
    "Tool Diameter (mm)", 5.0, 20.0, 10.0
)

tool_radius = st.sidebar.slider(
    "Tool Tip Radius (mm)", 1.0, 10.0, 5.0
)

st.sidebar.header("⚙️ Process Parameters")

step_size = st.sidebar.slider(
    "Step Size Δz (mm)", 0.1, 1.0, 0.5
)

feed_rate = st.sidebar.slider(
    "Feed Rate (mm/s)", 100, 1000, 300
)

spindle_speed = st.sidebar.slider(
    "Spindle Speed (RPM)", 0, 5000, 1500
)

forming_depth = st.sidebar.slider(
    "Target Forming Depth (mm)", 5.0, 50.0, 20.0
)

generate = st.sidebar.button("▶ Generate Results")

# ================= DATA GENERATION =================
t = np.linspace(0, 2*np.pi, 300)

x = 50 * np.cos(t)
y = 50 * np.sin(t)
z = -np.linspace(0, forming_depth, len(t))

# Simple prediction model (for academic demonstration)
pred_thickness = (
    1.5
    - 0.002 * np.abs(z)
    - 0.0001 * feed_rate
    + 0.00005 * tool_diameter
)

pred_df = pd.DataFrame({
    "x": x,
    "y": y,
    "thickness": pred_thickness
})

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔧 Tool Path", "📐 Predicted Result", "🧪 FEM Result", "📊 Comparison"]
)

# ================= TAB 1: TOOL PATH =================
with tab1:
    st.subheader("3D Tool Path Visualization")
    st.markdown(
        "Generated tool path based on the selected process parameters."
    )

    if generate:
        fig = px.line_3d(
            x=x, y=y, z=z,
            labels={"x": "X (mm)", "y": "Y (mm)", "z": "Z (mm)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Set parameters and click **Generate Results**.")

# ================= TAB 2: PREDICTED RESULT =================
with tab2:
    st.subheader("Predicted Thickness Distribution")
    st.markdown(
        "Thickness distribution predicted using a simplified analytical model."
    )

    if generate:
        fig = px.scatter(
            pred_df,
            x="x", y="y",
            color="thickness",
            labels={"thickness": "Thickness (mm)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prediction will appear after generation.")

# ================= TAB 3: FEM RESULT =================
with tab3:
    st.subheader("FEM Simulation Result")
    st.markdown(
        "Upload FEM simulation results in CSV format for visualization."
    )

    fem_file = st.file_uploader(
        "Upload FEM CSV file (columns: x, y, thickness)",
        type=["csv"]
    )

    if fem_file is not None:
        fem_df = pd.read_csv(fem_file)

        fig = px.scatter(
            fem_df,
            x="x", y="y",
            color="thickness",
            labels={"thickness": "Thickness (mm)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("FEM data loaded successfully.")
    else:
        st.info("Awaiting FEM CSV upload.")

# ================= TAB 4: COMPARISON =================
with tab4:
    st.subheader("Prediction vs FEM Comparison")
    st.markdown(
        "Quantitative comparison between predicted results and FEM simulations."
    )

    if generate and fem_file is not None:
        min_len = min(len(pred_df), len(fem_df))
        error = pred_df["thickness"][:min_len] - fem_df["thickness"][:min_len]

        comp_df = pd.DataFrame({
            "Index": np.arange(min_len),
            "Thickness Error (mm)": error
        })

        fig = px.line(
            comp_df,
            x="Index",
            y="Thickness Error (mm)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Mean Absolute Error (mm)",
            f"{np.mean(np.abs(error)):.4f}"
        )
    else:
        st.info("Generate results and upload FEM data to compare.")

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "PS4 – Smart Incremental Forming Tool calibrated with FEM | "
    "Permanent Streamlit Cloud deployment"
)

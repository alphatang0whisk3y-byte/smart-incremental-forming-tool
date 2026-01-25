import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Incremental Forming Tool",
    layout="wide"
)

# ================= ML MODEL COEFFICIENTS =================
# These are the coefficients from your trained polynomial regression model
ML_COEFFICIENTS = np.array([
    1.49527664, -0.00206813, 0.00487946, -0.00013187, 
    2.14330703e-06, -2.44266703e-05, 1.67795159e-07
])

def predict_thickness_ml(tool_diameter, step_size, depth):
    """
    Predict thickness using the trained polynomial regression model
    Features: [1, tool_diameter, step_size, depth, tool_diameter^2, 
               tool_diameter*step_size, tool_diameter*depth]
    """
    # Manually create polynomial features (degree 2)
    # Order: [bias, d, s, z, d^2, d*s, d*z]
    features = np.array([
        1.0,                        # bias
        tool_diameter,              # d
        step_size,                  # s
        depth,                      # z
        tool_diameter ** 2,         # d^2
        tool_diameter * step_size,  # d*s
        tool_diameter * depth       # d*z
    ])
    
    # Calculate prediction
    thickness = np.dot(ML_COEFFICIENTS, features)
    
    return thickness

# ================= TITLE =================
st.title("🔧 Smart Incremental Forming Tool")
st.markdown(
    "### ML-Powered Tool-Path Generation & Thickness Prediction"
)
st.markdown("Interactive interface for tool-path generation, ML prediction, and FEM comparison")
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🔧 Tool Parameters")

tool_diameter = st.sidebar.slider(
    "Tool Diameter (mm)", 5.0, 20.0, 10.0, 0.5
)

tool_radius = st.sidebar.slider(
    "Tool Tip Radius (mm)", 1.0, 10.0, 5.0, 0.5
)

st.sidebar.header("⚙️ Process Parameters")

step_size = st.sidebar.slider(
    "Step Size Δz (mm)", 0.1, 1.0, 0.5, 0.05
)

forming_depth = st.sidebar.slider(
    "Target Forming Depth (mm)", 5.0, 50.0, 20.0, 1.0
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Display Options")

num_points = st.sidebar.slider(
    "Tool Path Resolution", 100, 500, 300, 50
)

generate = st.sidebar.button("▶️ Generate Results", type="primary")

# Display current ML prediction in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 ML Prediction")
if generate:
    predicted_thickness = predict_thickness_ml(tool_diameter, step_size, forming_depth)
    st.sidebar.metric(
        "Predicted Thickness", 
        f"{predicted_thickness:.4f} mm",
        delta=f"{predicted_thickness - 1.5:.4f} mm from initial"
    )
    st.sidebar.caption("Based on polynomial regression model")

# ================= DATA GENERATION =================
# Generate spiral tool path
t = np.linspace(0, 2*np.pi, num_points)
radius = 50 * (1 - t/(2*np.pi) * 0.3)  # Gradually decreasing radius
x = radius * np.cos(t)
y = radius * np.sin(t)
z = -np.linspace(0, forming_depth, len(t))

# ML-based prediction for each point along the path
pred_thickness_array = []
for i in range(len(t)):
    depth_at_point = np.abs(z[i])
    thickness = predict_thickness_ml(tool_diameter, step_size, depth_at_point)
    pred_thickness_array.append(thickness)

pred_thickness_array = np.array(pred_thickness_array)

# Create DataFrame for predictions
pred_df = pd.DataFrame({
    "x": x,
    "y": y,
    "z": z,
    "thickness": pred_thickness_array
})

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔧 Tool Path", "🔮 ML Prediction", "🧪 FEM Result", "📊 Comparison", "📈 Model Info"]
)

# ================= TAB 1: TOOL PATH =================
with tab1:
    st.subheader("3D Tool Path Visualization")
    st.markdown(
        "Generated tool path based on the selected process parameters."
    )
    if generate:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line_3d(
                x=x, y=y, z=z,
                labels={"x": "X (mm)", "y": "Y (mm)", "z": "Z (mm)"},
                title="Incremental Forming Tool Path"
            )
            fig.update_traces(line=dict(color='royalblue', width=3))
            fig.update_layout(
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode='cube'
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Total Path Length", f"{len(t)} points")
            st.metric("Max Depth", f"{forming_depth:.2f} mm")
            st.metric("Step Size", f"{step_size:.2f} mm")
            st.metric("Tool Diameter", f"{tool_diameter:.2f} mm")
            
            # Show path statistics
            st.markdown("#### Path Statistics")
            st.write(f"- Start: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
            st.write(f"- End: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")
    else:
        st.info("⚠️ Set parameters and click **Generate Results** to visualize the tool path.")

# ================= TAB 2: ML PREDICTED RESULT =================
with tab2:
    st.subheader("ML-Predicted Thickness Distribution")
    st.markdown(
        "Thickness distribution predicted using trained polynomial regression model."
    )
    if generate:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 2D scatter plot
            fig = px.scatter(
                pred_df,
                x="x", y="y",
                color="thickness",
                labels={"thickness": "Thickness (mm)"},
                title="Predicted Thickness Map",
                color_continuous_scale="RdYlGn"
            )
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Mean Thickness", f"{pred_df['thickness'].mean():.4f} mm")
            st.metric("Min Thickness", f"{pred_df['thickness'].min():.4f} mm")
            st.metric("Max Thickness", f"{pred_df['thickness'].max():.4f} mm")
            st.metric("Std Deviation", f"{pred_df['thickness'].std():.4f} mm")
            
            # Thickness histogram
            st.markdown("#### Thickness Distribution")
            fig_hist = px.histogram(
                pred_df, 
                x="thickness",
                nbins=30,
                labels={"thickness": "Thickness (mm)"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
    else:
        st.info("⚠️ ML prediction will appear after generation.")

# ================= TAB 3: FEM RESULT =================
with tab3:
    st.subheader("FEM Simulation Result")
    st.markdown(
        "Upload FEM simulation results in CSV format for visualization and comparison."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fem_file = st.file_uploader(
            "Upload FEM CSV file (columns: x, y, thickness)",
            type=["csv"]
        )
        
        if fem_file is not None:
            fem_df = pd.read_csv(fem_file)
            
            # Display the data
            st.markdown("##### Uploaded FEM Data Preview")
            st.dataframe(fem_df.head(10), use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                fem_df,
                x="x", y="y",
                color="thickness",
                labels={"thickness": "Thickness (mm)"},
                title="FEM Thickness Distribution",
                color_continuous_scale="RdYlGn"
            )
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("⚠️ Awaiting FEM CSV upload.")
    
    with col2:
        if fem_file is not None:
            st.success("✅ FEM data loaded successfully.")
            st.metric("Data Points", len(fem_df))
            st.metric("Mean Thickness", f"{fem_df['thickness'].mean():.4f} mm")
            st.metric("Min Thickness", f"{fem_df['thickness'].min():.4f} mm")
            st.metric("Max Thickness", f"{fem_df['thickness'].max():.4f} mm")
            
            # Sample FEM data template
            st.markdown("---")
            st.markdown("##### Download Sample Template")
            sample_df = pd.DataFrame({
                'x': [10, 20, 30],
                'y': [15, 25, 35],
                'thickness': [1.48, 1.46, 1.44]
            })
            st.download_button(
                "📥 Download CSV Template",
                sample_df.to_csv(index=False),
                "fem_template.csv",
                "text/csv"
            )

# ================= TAB 4: COMPARISON =================
with tab4:
    st.subheader("ML Prediction vs FEM Comparison")
    st.markdown(
        "Quantitative comparison between ML predictions and FEM simulations."
    )
    if generate and fem_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Match dimensions
            min_len = min(len(pred_df), len(fem_df))
            error = pred_df["thickness"][:min_len] - fem_df["thickness"][:min_len]
            comp_df = pd.DataFrame({
                "Index": np.arange(min_len),
                "ML Prediction": pred_df["thickness"][:min_len].values,
                "FEM Result": fem_df["thickness"][:min_len].values,
                "Error": error
            })
            
            # Line plot comparison
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comp_df["Index"], 
                y=comp_df["ML Prediction"],
                mode='lines',
                name='ML Prediction',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=comp_df["Index"], 
                y=comp_df["FEM Result"],
                mode='lines',
                name='FEM Result',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="Thickness Comparison: ML vs FEM",
                xaxis_title="Point Index",
                yaxis_title="Thickness (mm)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Error plot
            fig_error = px.line(
                comp_df,
                x="Index",
                y="Error",
                title="Prediction Error (ML - FEM)",
                labels={"Error": "Error (mm)"}
            )
            fig_error.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_error.update_layout(height=400)
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # Error metrics
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(np.abs(error))
            
            st.metric("Mean Absolute Error", f"{mae:.4f} mm")
            st.metric("Root Mean Squared Error", f"{rmse:.4f} mm")
            st.metric("Max Error", f"{max_error:.4f} mm")
            st.metric("Data Points Compared", min_len)
            
            # Accuracy percentage
            accuracy = (1 - mae/np.mean(fem_df["thickness"][:min_len])) * 100
            st.metric("Model Accuracy", f"{accuracy:.2f}%")
            
            # Download comparison data
            st.markdown("---")
            st.download_button(
                "📥 Download Comparison CSV",
                comp_df.to_csv(index=False),
                "comparison_results.csv",
                "text/csv"
            )
    elif not generate:
        st.info("⚠️ Generate results first.")
    elif fem_file is None:
        st.info("⚠️ Upload FEM data to enable comparison.")
    else:
        st.info("⚠️ Generate results and upload FEM data to compare.")

# ================= TAB 5: MODEL INFO =================
with tab5:
    st.subheader("🤖 ML Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Architecture")
        st.write("**Type:** Polynomial Regression (Degree 2)")
        st.write("**Features:** Tool Diameter, Step Size, Forming Depth")
        st.write("**Output:** Sheet Thickness (mm)")
        
        st.markdown("#### Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": [
                "Bias",
                "Tool Diameter", 
                "Step Size", 
                "Depth",
                "Tool Diameter²",
                "Tool Diameter × Step Size",
                "Tool Diameter × Depth"
            ],
            "Coefficient": ML_COEFFICIENTS
        })
        st.dataframe(coef_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Model Equation")
        st.latex(r'''
        t = \beta_0 + \beta_1 d + \beta_2 s + \beta_3 z 
        + \beta_4 d^2 + \beta_5 ds + \beta_6 dz
        ''')
        st.caption("Where: t = thickness, d = diameter, s = step size, z = depth")
        
        st.markdown("#### Feature Ranges")
        st.write("- **Tool Diameter:** 5.0 - 20.0 mm")
        st.write("- **Step Size:** 0.1 - 1.0 mm")
        st.write("- **Forming Depth:** 5.0 - 50.0 mm")
        
        st.markdown("#### Model Performance")
        st.info("""
        This polynomial regression model was trained on experimental data 
        to predict sheet thickness based on incremental forming parameters.
        Use the Comparison tab to validate predictions against FEM results.
        """)

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "🔬 Smart Incremental Forming Tool with ML Integration | "
    "Powered by Polynomial Regression | "
    "Ready for Streamlit Cloud Deployment"
)

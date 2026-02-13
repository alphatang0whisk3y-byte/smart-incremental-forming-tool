Attachment app.py added.Conversation opened. 2 messages. 1 message unread.

Skip to content
Using Gmail with screen readers
1 of 615
(no subject)
Inbox

Gokul Kn
Attachments
10:27 PM (18 minutes ago)
 

Shawn M.Mathew
Attachments
10:45 PM (0 minutes ago)
to me



On Fri, Feb 13, 2026 at 10:27 PM Gokul Kn <alphatang0whisk3y@gmail.com> wrote:


 3 Attachments
  •  Scanned by Gmail
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import LabelEncoder

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Incremental Forming - ML Path Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD ML MODEL =================
@st.cache_resource
def load_ml_model():
    """Load the trained ML model and components"""
    try:
        with open('best_combined_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model_data = load_ml_model()

if model_data is None:
    st.error("⚠️ Failed to load ML model. Please ensure 'best_combined_model.pkl' is in the same directory.")
    st.stop()

# Extract model components
ml_model = model_data['model']
encoder = model_data['encoder']
feature_cols = model_data['feature_cols']
path_types = model_data['path_types']

# ================= HELPER FUNCTIONS =================

def prepare_features(path_type, depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    """
    Prepare features for ML prediction
    Features: ['depth', 'path_encoded', 'param_radius_filled', 'param_max_radius_filled', 
               'param_side_length_filled', 'depth_squared', 'radius_squared', 'max_radius_squared']
    """
    # Encode path type
    path_encoded = encoder.transform([path_type])[0]
    
    # Create features
    features = {
        'depth': depth,
        'path_encoded': path_encoded,
        'param_radius_filled': param_radius,
        'param_max_radius_filled': param_max_radius,
        'param_side_length_filled': param_side_length,
        'depth_squared': depth ** 2,
        'radius_squared': param_radius ** 2,
        'max_radius_squared': param_max_radius ** 2
    }
    
    # Convert to array in correct order
    feature_array = np.array([[features[col] for col in feature_cols]])
    return feature_array

def predict_stress(path_type, depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    """Predict stress for given parameters"""
    features = prepare_features(path_type, depth, param_radius, param_max_radius, param_side_length)
    stress_prediction = ml_model.predict(features)[0]
    return stress_prediction

def find_best_path(depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    """Find the best path type (lowest stress) for given depth"""
    predictions = {}
    
    for path in path_types:
        try:
            stress = predict_stress(path, depth, param_radius, param_max_radius, param_side_length)
            predictions[path] = stress
        except Exception as e:
            predictions[path] = None
    
    # Remove None values
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        return None, None, {}
    
    # Find minimum stress path
    best_path = min(valid_predictions, key=valid_predictions.get)
    best_stress = valid_predictions[best_path]
    
    return best_path, best_stress, valid_predictions

def generate_path_geometry(path_type, depth, num_points=300):
    """Generate 3D coordinates for different path types"""
    t = np.linspace(0, 2*np.pi, num_points)
    
    if path_type == 'circular':
        radius = 50
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'spiral':
        radius = 50 * (1 - t/(2*np.pi) * 0.5)
        x = radius * np.cos(t * 3)
        y = radius * np.sin(t * 3)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'spiral_inward':
        radius = 50 * (t/(2*np.pi))
        x = radius * np.cos(t * 5)
        y = radius * np.sin(t * 5)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'square':
        # Square path
        side = 40
        x = np.concatenate([
            np.linspace(-side, side, num_points//4),
            np.full(num_points//4, side),
            np.linspace(side, -side, num_points//4),
            np.full(num_points//4, -side)
        ])
        y = np.concatenate([
            np.full(num_points//4, -side),
            np.linspace(-side, side, num_points//4),
            np.full(num_points//4, side),
            np.linspace(side, -side, num_points//4)
        ])
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'hexagon':
        # Hexagon path
        angles = np.linspace(0, 2*np.pi, 7)
        radius = 40
        hex_x = radius * np.cos(angles)
        hex_y = radius * np.sin(angles)
        x = np.interp(np.linspace(0, 6, num_points), np.arange(7), hex_x)
        y = np.interp(np.linspace(0, 6, num_points), np.arange(7), hex_y)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'star':
        # Star pattern
        outer_r, inner_r = 50, 25
        points = 5
        angles = np.linspace(0, 2*np.pi, points*2+1)
        radii = np.array([outer_r if i % 2 == 0 else inner_r for i in range(len(angles))])
        star_x = radii * np.cos(angles)
        star_y = radii * np.sin(angles)
        x = np.interp(np.linspace(0, points*2, num_points), np.arange(len(star_x)), star_x)
        y = np.interp(np.linspace(0, points*2, num_points), np.arange(len(star_y)), star_y)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'rose':
        # Rose curve
        k = 3  # petals
        radius = 50 * np.cos(k * t)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'ellipse':
        # Ellipse
        a, b = 50, 30  # major, minor axis
        x = a * np.cos(t)
        y = b * np.sin(t)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'zigzag':
        # Zigzag pattern
        x = 40 * np.sin(t * 5)
        y = np.linspace(-40, 40, num_points)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'figure8':
        # Figure-8 / Lemniscate
        scale = 30
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'concentric':
        # Concentric circles
        num_circles = 3
        radius = 50 * (1 - (t / (2*np.pi)) % (1/num_circles) * num_circles)
        x = radius * np.cos(t * num_circles)
        y = radius * np.sin(t * num_circles)
        z = -np.linspace(0, depth, num_points)
        
    elif path_type == 'lissajous':
        # Lissajous curve
        A, B = 40, 30
        a, b = 3, 4
        x = A * np.sin(a * t)
        y = B * np.sin(b * t)
        z = -np.linspace(0, depth, num_points)
        
    else:
        # Default to circular
        radius = 50
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = -np.linspace(0, depth, num_points)
    
    return x, y, z

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<p class="main-header">🔧 Smart Incremental Forming - ML Path Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Tool-Path Selection & Stress Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🎯 Input Parameters")

# Shape selection
st.sidebar.subheader("📐 Shape Selection")
shape_input = st.sidebar.selectbox(
    "Select Shape Type",
    options=['Auto (ML Recommends)'] + path_types,
    help="Choose 'Auto' to let ML find the best path type"
)

# Depth input
depth_input = st.sidebar.slider(
    "Forming Depth (mm)",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Target forming depth"
)

# Advanced parameters
st.sidebar.subheader("⚙️ Advanced Parameters")
with st.sidebar.expander("Shape Parameters"):
    param_radius = st.slider("Radius (mm)", 5.0, 20.0, 10.0, 0.5)
    param_max_radius = st.slider("Max Radius (mm)", 10.0, 25.0, 15.0, 0.5)
    param_side_length = st.slider("Side Length (mm)", 8.0, 20.0, 12.0, 0.5)

# Display options
st.sidebar.subheader("📊 Visualization")
num_points = st.sidebar.slider("Path Resolution", 100, 500, 300, 50)
show_comparison = st.sidebar.checkbox("Show All Path Comparisons", value=True)

# Generate button
generate = st.sidebar.button("🚀 Generate & Predict", type="primary", use_container_width=True)

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Information")
st.sidebar.info(f"""
**Model Type:** Ridge Regression (Polynomial)  
**Features:** {len(feature_cols)}  
**Training Samples:** {model_data['training_samples']}  
**R² Score:** {model_data['r2']:.3f}  
**RMSE:** {model_data['rmse']:.2f} MPa
""")

# ================= MAIN CONTENT =================

if generate:
    # Determine which path to use
    if shape_input == 'Auto (ML Recommends)':
        # Find best path automatically
        with st.spinner("🔍 Analyzing all path types to find optimal solution..."):
            best_path, best_stress, all_predictions = find_best_path(
                depth_input, param_radius, param_max_radius, param_side_length
            )
        
        if best_path is None:
            st.error("❌ Failed to generate predictions. Please check input parameters.")
            st.stop()
        
        # Display recommendation
        st.markdown(f"""
        <div class="recommendation-box">
            🎯 RECOMMENDED PATH TYPE: <strong>{best_path.upper()}</strong><br>
            Predicted Stress: {best_stress:.2f} MPa
        </div>
        """, unsafe_allow_html=True)
        
        selected_path = best_path
        predicted_stress = best_stress
        
    else:
        # Use user-selected path
        selected_path = shape_input
        predicted_stress = predict_stress(
            selected_path, depth_input, param_radius, param_max_radius, param_side_length
        )
        all_predictions = None
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Selected Path:</strong> {selected_path.upper()}<br>
            <strong>Predicted Stress:</strong> {predicted_stress:.2f} MPa
        </div>
        """, unsafe_allow_html=True)
    
    # ================= KEY METRICS =================
    st.subheader("📊 Predicted Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Stress",
            f"{predicted_stress:.2f} MPa",
            help="Predicted maximum stress"
        )
    
    with col2:
        # Estimate strain (simplified relationship)
        estimated_strain = predicted_stress / 200000 * 100  # Rough estimate
        st.metric(
            "Est. Strain",
            f"{estimated_strain:.3f}%",
            help="Estimated strain percentage"
        )
    
    with col3:
        st.metric(
            "Forming Depth",
            f"{depth_input:.1f} mm",
            help="Target forming depth"
        )
    
    with col4:
        # Safety factor (assuming yield stress ~250 MPa for typical sheet metal)
        safety_factor = 250 / predicted_stress if predicted_stress > 0 else 0
        st.metric(
            "Safety Factor",
            f"{safety_factor:.2f}",
            delta="Safe" if safety_factor > 1.5 else "Check",
            delta_color="normal" if safety_factor > 1.5 else "inverse"
        )
    
    st.markdown("---")
    
    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 3D Path Visualization",
        "📈 Path Comparison",
        "🔬 Detailed Analysis",
        "📋 Specifications"
    ])
    
    # ================= TAB 1: PATH VISUALIZATION =================
    with tab1:
        st.subheader(f"3D Tool Path: {selected_path.upper()}")
        
        # Generate path geometry
        x, y, z = generate_path_geometry(selected_path, depth_input, num_points)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 3D Path visualization
            fig = px.line_3d(
                x=x, y=y, z=z,
                labels={"x": "X (mm)", "y": "Y (mm)", "z": "Z (mm)"},
                title=f"{selected_path.title()} Path - Stress: {predicted_stress:.2f} MPa"
            )
            fig.update_traces(
                line=dict(
                    color=z,
                    colorscale='RdYlGn_r',
                    width=4,
                    colorbar=dict(title="Depth (mm)")
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode='cube',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Path Statistics")
            st.metric("Total Points", len(x))
            st.metric("Max Depth", f"{abs(z.min()):.2f} mm")
            st.metric("Path Type", selected_path.title())
            
            st.markdown("#### Coordinates")
            st.write(f"**Start:** ({x[0]:.1f}, {y[0]:.1f}, {z[0]:.1f})")
            st.write(f"**End:** ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f})")
            st.write(f"**X Range:** {x.min():.1f} to {x.max():.1f} mm")
            st.write(f"**Y Range:** {y.min():.1f} to {y.max():.1f} mm")
            
            # Download path data
            path_df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
            csv = path_df.to_csv(index=False)
            st.download_button(
                "📥 Download Path Data",
                csv,
                f"{selected_path}_path.csv",
                "text/csv",
                use_container_width=True
            )
    
    # ================= TAB 2: PATH COMPARISON =================
    with tab2:
        st.subheader("Stress Comparison Across All Path Types")
        
        if all_predictions is not None and show_comparison:
            # Create comparison dataframe
            comp_df = pd.DataFrame([
                {"Path Type": k, "Predicted Stress (MPa)": v}
                for k, v in all_predictions.items()
            ]).sort_values("Predicted Stress (MPa)")
            
            # Highlight best option
            comp_df['Status'] = comp_df['Path Type'].apply(
                lambda x: '🏆 Best' if x == best_path else '✓ Valid'
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart
                fig = px.bar(
                    comp_df,
                    x='Path Type',
                    y='Predicted Stress (MPa)',
                    title='Stress Prediction by Path Type',
                    color='Predicted Stress (MPa)',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Ranking")
                st.dataframe(
                    comp_df[['Path Type', 'Predicted Stress (MPa)', 'Status']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Download comparison
                csv = comp_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Comparison",
                    csv,
                    "path_comparison.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.info("💡 Enable 'Show All Path Comparisons' in sidebar or select 'Auto' mode to see comparison.")
    
    # ================= TAB 3: DETAILED ANALYSIS =================
    with tab3:
        st.subheader("Detailed Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Parameters")
            params_df = pd.DataFrame({
                "Parameter": ["Path Type", "Depth", "Radius", "Max Radius", "Side Length"],
                "Value": [
                    selected_path,
                    f"{depth_input} mm",
                    f"{param_radius} mm",
                    f"{param_max_radius} mm",
                    f"{param_side_length} mm"
                ]
            })
            st.dataframe(params_df, hide_index=True, use_container_width=True)
            
            st.markdown("#### ML Model Features")
            features = prepare_features(
                selected_path, depth_input, param_radius, param_max_radius, param_side_length
            )
            features_df = pd.DataFrame({
                "Feature": feature_cols,
                "Value": features[0]
            })
            st.dataframe(features_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Stress Distribution Estimation")
            
            # Generate stress distribution along depth
            depth_points = np.linspace(0, depth_input, 20)
            stress_points = [
                predict_stress(selected_path, d, param_radius, param_max_radius, param_side_length)
                for d in depth_points
            ]
            
            fig = px.line(
                x=depth_points,
                y=stress_points,
                labels={"x": "Depth (mm)", "y": "Predicted Stress (MPa)"},
                title="Stress vs Depth Profile"
            )
            fig.update_traces(line=dict(color='red', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Material Recommendations")
            if predicted_stress < 200:
                st.success("✅ Suitable for aluminum alloys (6061-T6)")
            elif predicted_stress < 300:
                st.warning("⚠️ Consider steel alloys (mild steel)")
            else:
                st.error("❌ High stress - use high-strength steel")
    
    # ================= TAB 4: SPECIFICATIONS =================
    with tab4:
        st.subheader("Technical Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Process Parameters")
            st.write(f"**Path Type:** {selected_path}")
            st.write(f"**Forming Depth:** {depth_input} mm")
            st.write(f"**Shape Radius:** {param_radius} mm")
            st.write(f"**Max Radius:** {param_max_radius} mm")
            st.write(f"**Side Length:** {param_side_length} mm")
            
            st.markdown("#### Predicted Results")
            st.write(f"**Maximum Stress:** {predicted_stress:.2f} MPa")
            st.write(f"**Estimated Strain:** {estimated_strain:.3f}%")
            st.write(f"**Safety Factor:** {safety_factor:.2f}")
            
        with col2:
            st.markdown("#### ML Model Details")
            st.write(f"**Algorithm:** Ridge Regression")
            st.write(f"**Feature Set:** Polynomial (degree 2)")
            st.write(f"**Input Features:** {len(feature_cols)}")
            st.write(f"**Training Samples:** {model_data['training_samples']}")
            st.write(f"**Model R²:** {model_data['r2']:.3f}")
            st.write(f"**Model RMSE:** {model_data['rmse']:.2f} MPa")
            
            st.markdown("#### Available Path Types")
            st.write(", ".join(path_types))

else:
    # Initial state - show instructions
    st.info("""
    ### 🚀 Getting Started
    
    1. **Select a shape** from the sidebar (or choose 'Auto' for ML recommendation)
    2. **Set the forming depth** using the slider
    3. **Adjust advanced parameters** if needed (optional)
    4. **Click 'Generate & Predict'** to see results
    
    The ML model will predict stress, strain, and other performance metrics based on your inputs.
    """)
    
    # Show available path types
    st.markdown("### 📐 Available Path Types")
    
    cols = st.columns(4)
    for idx, path in enumerate(path_types):
        with cols[idx % 4]:
            st.button(f"◆ {path.title()}", use_container_width=True, disabled=True)

# ================= FOOTER =================
st.markdown("---")
st.caption("""
🔬 **Smart Incremental Forming with ML** | Powered by Ridge Regression | 
Path Optimization & Stress Prediction System
""")
enhanced_app.py
Displaying enhanced_app.py.

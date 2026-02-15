import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Smart Incremental Forming - ML Path Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def train_ml_models_from_csv():
    """Train both stress and time prediction models"""
    try:
        df = pd.read_csv('simulation_results_FINAL_CORRECTED.csv')
        df = df[df['status'] == 'SUCCESS'].copy()
        df = df.dropna(subset=['max_stress_MPa', 'total_time'])
        
        df['depth'] = df['depth_input_mm']
        df['param_radius_filled'] = df['param_radius'].fillna(10.0)
        df['param_max_radius_filled'] = df['param_max_radius'].fillna(df['param_radius_filled'] * 1.5)
        df['param_side_length_filled'] = df['param_side_length'].fillna(12.0)
        
        df['depth_squared'] = df['depth'] ** 2
        df['radius_squared'] = df['param_radius_filled'] ** 2
        df['max_radius_squared'] = df['param_max_radius_filled'] ** 2
        
        encoder = LabelEncoder()
        df['path_encoded'] = encoder.fit_transform(df['path_type'])
        
        feature_cols = [
            'depth', 'path_encoded', 'param_radius_filled', 
            'param_max_radius_filled', 'param_side_length_filled',
            'depth_squared', 'radius_squared', 'max_radius_squared'
        ]
        
        X = df[feature_cols]
        
        # Train stress model
        y_stress = df['max_stress_MPa']
        X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(
            X, y_stress, test_size=0.2, random_state=42
        )
        
        stress_model = Ridge(alpha=1.0)
        stress_model.fit(X_train_stress, y_train_stress)
        
        y_pred_stress = stress_model.predict(X_test_stress)
        stress_mae = mean_absolute_error(y_test_stress, y_pred_stress)
        stress_rmse = np.sqrt(mean_squared_error(y_test_stress, y_pred_stress))
        stress_r2 = r2_score(y_test_stress, y_pred_stress)
        
        # Train time model
        y_time = df['total_time']
        X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
            X, y_time, test_size=0.2, random_state=42
        )
        
        time_model = Ridge(alpha=1.0)
        time_model.fit(X_train_time, y_train_time)
        
        y_pred_time = time_model.predict(X_test_time)
        time_mae = mean_absolute_error(y_test_time, y_pred_time)
        time_rmse = np.sqrt(mean_squared_error(y_test_time, y_pred_time))
        time_r2 = r2_score(y_test_time, y_pred_time)
        
        path_types = sorted(df['path_type'].unique().tolist())
        
        return {
            'stress_model': stress_model,
            'time_model': time_model,
            'encoder': encoder,
            'feature_cols': feature_cols,
            'path_types': path_types,
            'stress_mae': stress_mae,
            'stress_rmse': stress_rmse,
            'stress_r2': stress_r2,
            'time_mae': time_mae,
            'time_rmse': time_rmse,
            'time_r2': time_r2,
            'training_samples': len(df)
        }
        
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None

with st.spinner("Training ML models from data..."):
    model_data = train_ml_models_from_csv()

if model_data is None:
    st.stop()

stress_model = model_data['stress_model']
time_model = model_data['time_model']
encoder = model_data['encoder']
feature_cols = model_data['feature_cols']
path_types = model_data['path_types']

GEOMETRY_PATH_RECOMMENDATIONS = {
    'cone': {
        'recommended_paths': ['circular', 'spiral'],
        'description': 'Conical shape with circular base',
        'num_layers': 'auto'
    },
    'pyramid': {
        'recommended_paths': ['square', 'hexagon'],
        'description': 'Pyramid with polygonal base',
        'num_layers': 'auto'
    },
    'dome': {
        'recommended_paths': ['circular', 'spiral', 'concentric'],
        'description': 'Hemispherical dome shape',
        'num_layers': 'auto'
    },
    'bowl': {
        'recommended_paths': ['circular', 'spiral_inward', 'concentric'],
        'description': 'Bowl or cup shape',
        'num_layers': 'auto'
    },
    'funnel': {
        'recommended_paths': ['spiral', 'spiral_inward'],
        'description': 'Funnel shape with tapered profile',
        'num_layers': 'auto'
    },
    'custom_shape': {
        'recommended_paths': path_types,
        'description': 'Custom geometry - all paths available',
        'num_layers': 'auto'
    }
}

def parse_dynain_file(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.split('\n')
        
        nodes = []
        thicknesses = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('*NODE'):
                i += 1
                while i < len(lines) and not lines[i].startswith('*'):
                    node_line = lines[i].strip()
                    if node_line:
                        parts = node_line.split()
                        try:
                            node_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            nodes.append({'id': node_id, 'x': x, 'y': y, 'z': z})
                        except:
                            pass
                    i += 1
                continue
            
            elif line.startswith('*ELEMENT_SHELL_THICKNESS'):
                i += 1
                while i < len(lines) and not lines[i].startswith('*'):
                    elem_line = lines[i].strip()
                    if elem_line:
                        parts = elem_line.split()
                        try:
                            elem_id = int(parts[0])
                            i += 1
                            if i < len(lines):
                                thick_line = lines[i].strip()
                                thick_parts = thick_line.split()
                                t1 = float(thick_parts[0])
                                t2 = float(thick_parts[1])
                                t3 = float(thick_parts[2])
                                t4 = float(thick_parts[3])
                                avg_thickness = (t1 + t2 + t3 + t4) / 4.0
                                thicknesses.append(avg_thickness)
                        except:
                            pass
                    i += 1
                continue
            
            i += 1
        
        nodes_df = pd.DataFrame(nodes)
        
        result = {
            'num_nodes': len(nodes_df),
            'num_elements': len(thicknesses),
            'success': True
        }
        
        if len(nodes_df) > 0:
            result['centroid_x'] = nodes_df['x'].mean()
            result['centroid_y'] = nodes_df['y'].mean()
            result['centroid_z'] = nodes_df['z'].mean()
            result['min_z'] = nodes_df['z'].min()
            result['max_z'] = nodes_df['z'].max()
            result['actual_depth'] = abs(result['max_z'] - result['min_z'])
        
        if len(thicknesses) > 0:
            result['mean_thickness'] = np.mean(thicknesses)
            result['min_thickness'] = np.min(thicknesses)
            result['max_thickness'] = np.max(thicknesses)
            result['std_thickness'] = np.std(thicknesses)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def prepare_features(path_type, depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    path_encoded = encoder.transform([path_type])[0]
    
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
    
    feature_array = np.array([[features[col] for col in feature_cols]])
    return feature_array

def predict_stress_and_time(path_type, depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    """Predict both stress and time for a given path configuration"""
    features = prepare_features(path_type, depth, param_radius, param_max_radius, param_side_length)
    stress_prediction = stress_model.predict(features)[0]
    time_prediction = time_model.predict(features)[0]
    return stress_prediction, time_prediction

def find_best_path_for_geometry(geometry, depth, param_radius=10.0, param_max_radius=15.0, param_side_length=12.0):
    recommended_paths = GEOMETRY_PATH_RECOMMENDATIONS[geometry]['recommended_paths']
    
    predictions = {}
    for path in recommended_paths:
        try:
            stress, time = predict_stress_and_time(path, depth, param_radius, param_max_radius, param_side_length)
            predictions[path] = {'stress': stress, 'time': time}
        except:
            predictions[path] = None
    
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        return None, None, None, {}
    
    # Find best path (lowest stress)
    best_path = min(valid_predictions, key=lambda k: valid_predictions[k]['stress'])
    best_stress = valid_predictions[best_path]['stress']
    best_time = valid_predictions[best_path]['time']
    
    return best_path, best_stress, best_time, valid_predictions

def calculate_num_layers(depth, step_down=0.5):
    return max(int(np.ceil(depth / step_down)), 3)

def generate_complete_tool_path(path_type, geometry, depth, base_radius=50, num_points_per_layer=100):
    num_layers = calculate_num_layers(depth, step_down=0.5)
    
    x_complete = []
    y_complete = []
    z_complete = []
    
    for layer in range(num_layers):
        layer_depth = depth * (layer + 1) / num_layers
        
        if geometry in ['cone', 'funnel']:
            layer_radius = base_radius * (1 - (layer + 1) / (num_layers + 2))
        elif geometry in ['dome', 'bowl']:
            progress = (layer + 1) / num_layers
            layer_radius = base_radius * np.sqrt(1 - progress**2)
        elif geometry == 'pyramid':
            layer_radius = base_radius * (1 - (layer + 1) / (num_layers + 1))
        else:
            layer_radius = base_radius
        
        t = np.linspace(0, 2*np.pi, num_points_per_layer)
        
        if path_type == 'circular':
            x_layer = layer_radius * np.cos(t)
            y_layer = layer_radius * np.sin(t)
        elif path_type == 'spiral':
            radius_variation = np.linspace(layer_radius, layer_radius * 0.8, num_points_per_layer)
            x_layer = radius_variation * np.cos(t * 2)
            y_layer = radius_variation * np.sin(t * 2)
        elif path_type == 'spiral_inward':
            radius_variation = np.linspace(layer_radius, layer_radius * 0.6, num_points_per_layer)
            x_layer = radius_variation * np.cos(t * 3)
            y_layer = radius_variation * np.sin(t * 3)
        elif path_type == 'square':
            side = layer_radius
            x_layer = np.concatenate([
                np.linspace(-side, side, num_points_per_layer//4),
                np.full(num_points_per_layer//4, side),
                np.linspace(side, -side, num_points_per_layer//4),
                np.full(num_points_per_layer//4, -side)
            ])
            y_layer = np.concatenate([
                np.full(num_points_per_layer//4, -side),
                np.linspace(-side, side, num_points_per_layer//4),
                np.full(num_points_per_layer//4, side),
                np.linspace(side, -side, num_points_per_layer//4)
            ])
        elif path_type == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)
            hex_x = layer_radius * np.cos(angles)
            hex_y = layer_radius * np.sin(angles)
            x_layer = np.interp(np.linspace(0, 6, num_points_per_layer), np.arange(7), hex_x)
            y_layer = np.interp(np.linspace(0, 6, num_points_per_layer), np.arange(7), hex_y)
        elif path_type == 'star':
            outer_r, inner_r = layer_radius, layer_radius * 0.5
            points = 5
            angles = np.linspace(0, 2*np.pi, points*2+1)
            radii = np.array([outer_r if i % 2 == 0 else inner_r for i in range(len(angles))])
            star_x = radii * np.cos(angles)
            star_y = radii * np.sin(angles)
            x_layer = np.interp(np.linspace(0, points*2, num_points_per_layer), np.arange(len(star_x)), star_x)
            y_layer = np.interp(np.linspace(0, points*2, num_points_per_layer), np.arange(len(star_y)), star_y)
        elif path_type == 'rose':
            k = 3
            radius_pattern = layer_radius * np.abs(np.cos(k * t))
            x_layer = radius_pattern * np.cos(t)
            y_layer = radius_pattern * np.sin(t)
        elif path_type == 'ellipse':
            a, b = layer_radius, layer_radius * 0.6
            x_layer = a * np.cos(t)
            y_layer = b * np.sin(t)
        elif path_type == 'zigzag':
            x_layer = layer_radius * 0.8 * np.sin(t * 5)
            y_layer = np.linspace(-layer_radius, layer_radius, num_points_per_layer)
        elif path_type == 'figure8':
            x_layer = layer_radius * 0.6 * np.sin(t)
            y_layer = layer_radius * 0.6 * np.sin(t) * np.cos(t)
        elif path_type == 'concentric':
            num_circles = 3
            radius_pattern = layer_radius * (1 - (t / (2*np.pi)) % (1/num_circles) * num_circles)
            x_layer = radius_pattern * np.cos(t * num_circles)
            y_layer = radius_pattern * np.sin(t * num_circles)
        elif path_type == 'lissajous':
            A, B = layer_radius * 0.8, layer_radius * 0.6
            a, b = 3, 4
            x_layer = A * np.sin(a * t)
            y_layer = B * np.sin(b * t)
        else:
            x_layer = layer_radius * np.cos(t)
            y_layer = layer_radius * np.sin(t)
        
        z_layer = np.full(len(x_layer), -layer_depth)
        
        x_complete.extend(x_layer)
        y_complete.extend(y_layer)
        z_complete.extend(z_layer)
        
        if layer < num_layers - 1:
            x_complete.append(x_layer[-1])
            y_complete.append(y_layer[-1])
            z_complete.append(-layer_depth + 0.2)
    
    return np.array(x_complete), np.array(y_complete), np.array(z_complete), num_layers

st.markdown('<p style="font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;">Smart Incremental Forming - ML Path Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:1.2rem;color:#666;text-align:center;margin-bottom:2rem;">AI-Powered Tool-Path Optimization for Target Geometries</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Target Geometry")

geometry_input = st.sidebar.selectbox(
    "What geometry do you want to form?",
    options=['cone', 'pyramid', 'dome', 'bowl', 'funnel', 'custom_shape'],
    format_func=lambda x: x.replace('_', ' ').title(),
    help="Select the target shape you want to create"
)

st.sidebar.markdown(f"**Description:** {GEOMETRY_PATH_RECOMMENDATIONS[geometry_input]['description']}")

depth_input = st.sidebar.slider(
    "Target Depth (mm)",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Maximum forming depth"
)

st.sidebar.header("Geometry Parameters")
base_radius = st.sidebar.slider("Base Radius (mm)", 20.0, 80.0, 50.0, 5.0)

st.sidebar.header("Process Parameters")
step_down = st.sidebar.slider("Layer Step Down (mm)", 0.2, 1.0, 0.5, 0.1, help="Depth increment per layer")
num_points_per_layer = st.sidebar.slider("Points per Layer", 50, 200, 100, 10)

param_radius = 10.0
param_max_radius = 15.0
param_side_length = 12.0

st.sidebar.header("Visualization")
show_comparison = st.sidebar.checkbox("Show Path Comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Upload FEM Results")
st.sidebar.markdown("**Upload dynain.txt** to compare FEM simulation with ML prediction")
st.sidebar.caption("Ensure simulation used the same parameters as set above")

uploaded_dynain = st.sidebar.file_uploader(
    "Choose dynain.txt file",
    type=["txt"],
    help="Upload LS-DYNA dynain.txt output file"
)

fem_data = None
if uploaded_dynain is not None:
    with st.spinner("Parsing FEM results..."):
        fem_data = parse_dynain_file(uploaded_dynain)
        if fem_data['success']:
            st.sidebar.success(f"FEM data loaded: {fem_data['num_elements']} elements")
        else:
            st.sidebar.error(f"Parse error: {fem_data.get('error', 'Unknown error')}")
            fem_data = None

generate = st.sidebar.button("Generate Optimal Path", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(f"""
**Stress Model:**  
- R² Score: {model_data['stress_r2']:.3f}  
- RMSE: {model_data['stress_rmse']:.2f} MPa

**Time Model:**  
- R² Score: {model_data['time_r2']:.3f}  
- RMSE: {model_data['time_rmse']:.2f} sec

**Training Samples:** {model_data['training_samples']}
""")

if generate:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.5rem;border-radius:15px;color:white;font-size:1.1rem;text-align:center;margin:1rem 0;">
        TARGET GEOMETRY: <strong>{geometry_input.upper().replace('_', ' ')}</strong><br>
        Target Depth: {depth_input} mm | Base Radius: {base_radius} mm
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Finding optimal tool path..."):
        best_path, best_stress, best_time, all_predictions = find_best_path_for_geometry(
            geometry_input, depth_input, param_radius, param_max_radius, base_radius
        )
    
    if best_path is None:
        st.error("Failed to generate predictions.")
        st.stop()
    
    num_layers = calculate_num_layers(depth_input, step_down)
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);padding:2rem;border-radius:15px;color:white;font-size:1.3rem;font-weight:bold;text-align:center;margin:1rem 0;">
        RECOMMENDED TOOL PATH: <strong>{best_path.upper()}</strong><br>
        Predicted Stress: {best_stress:.2f} MPa | Predicted Time: {best_time:.1f} sec | Layers: {num_layers}
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Process Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Tool Path", best_path.title())
    
    with col2:
        st.metric("Max Stress", f"{best_stress:.2f} MPa")
    
    with col3:
        st.metric("Process Time", f"{best_time:.1f} sec")
    
    with col4:
        st.metric("Total Layers", num_layers)
    
    with col5:
        st.metric("Step Down", f"{step_down} mm")
    
    with col6:
        safety_factor = 250 / best_stress if best_stress > 0 else 0
        st.metric(
            "Safety Factor",
            f"{safety_factor:.2f}",
            delta="Safe" if safety_factor > 1.5 else "Check"
        )
    
    st.markdown("---")
    
    if fem_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Complete Tool Path (Multi-Layer)",
            "Path Comparison",
            "ML vs FEM Comparison",
            "Process Details"
        ])
    else:
        tab1, tab2, tab3 = st.tabs([
            "Complete Tool Path (Multi-Layer)",
            "Path Comparison",
            "Process Details"
        ])
        tab4 = None
    
    with tab1:
        st.subheader(f"Complete Tool Movement - {best_path.upper()} Path for {geometry_input.upper()}")
        st.markdown(f"**Showing all {num_layers} layers** with tool retractions between passes")
        
        x, y, z, layers = generate_complete_tool_path(
            best_path, geometry_input, depth_input, base_radius, num_points_per_layer
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            layer_colors = []
            for i in range(len(z)):
                layer_num = int(-z[i] / (depth_input / num_layers))
                layer_colors.append(layer_num)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color=layer_colors,
                    colorscale='Viridis',
                    width=3,
                    colorbar=dict(title="Layer")
                ),
                name='Tool Path'
            )])
            
            fig.update_layout(
                title=f"Complete {num_layers}-Layer Tool Path",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Path Statistics")
            st.metric("Total Points", len(x))
            st.metric("Total Layers", num_layers)
            st.metric("Points per Layer", num_points_per_layer)
            st.metric("Max Depth", f"{depth_input:.2f} mm")
            st.metric("Base Radius", f"{base_radius:.1f} mm")
            
            st.markdown("#### Performance Predictions")
            st.metric("Predicted Stress", f"{best_stress:.2f} MPa")
            st.metric("Predicted Time", f"{best_time:.1f} sec")
            time_per_layer = best_time / num_layers
            st.metric("Time per Layer", f"{time_per_layer:.2f} sec")
            
            st.markdown("#### Layer Information")
            st.write(f"**Step Down:** {step_down} mm")
            st.write(f"**Path Type:** {best_path}")
            st.write(f"**Geometry:** {geometry_input.title()}")
            
            path_df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
            csv = path_df.to_csv(index=False)
            st.download_button(
                "Download Complete Path",
                csv,
                f"{geometry_input}_{best_path}_path.csv",
                "text/csv",
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Tool Path Comparison for Target Geometry")
        
        if show_comparison and all_predictions:
            comp_data = []
            for k, v in all_predictions.items():
                comp_data.append({
                    "Path Type": k,
                    "Predicted Stress (MPa)": v['stress'],
                    "Predicted Time (sec)": v['time'],
                    "Status": 'Best' if k == best_path else 'Valid'
                })
            
            comp_df = pd.DataFrame(comp_data).sort_values("Predicted Stress (MPa)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create dual-axis chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=comp_df['Path Type'],
                    y=comp_df['Predicted Stress (MPa)'],
                    name='Stress (MPa)',
                    marker_color='lightblue',
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=comp_df['Path Type'],
                    y=comp_df['Predicted Time (sec)'],
                    name='Time (sec)',
                    marker=dict(color='red', size=10),
                    yaxis='y2',
                    mode='lines+markers'
                ))
                
                fig.update_layout(
                    title=f'Stress and Time Predictions for {geometry_input.upper()} Geometry',
                    xaxis=dict(title='Path Type'),
                    yaxis=dict(
                        title='Predicted Stress (MPa)',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Predicted Time (sec)',
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    hovermode='x'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Ranking by Stress")
                st.dataframe(
                    comp_df[['Path Type', 'Predicted Stress (MPa)', 'Predicted Time (sec)', 'Status']],
                    hide_index=True,
                    use_container_width=True
                )
                
                st.markdown("#### Recommendation")
                st.info(f"""
                For **{geometry_input}** geometry:
                - Best path: **{best_path}**
                - Lowest stress: **{best_stress:.2f} MPa**
                - Process time: **{best_time:.1f} sec**
                - {len(comp_df)} paths evaluated
                """)
                
                # Find fastest path
                fastest_path = min(all_predictions.items(), key=lambda x: x[1]['time'])
                if fastest_path[0] != best_path:
                    st.warning(f"""
                    **Note:** Fastest path is **{fastest_path[0]}** 
                    ({fastest_path[1]['time']:.1f} sec) but with higher 
                    stress ({fastest_path[1]['stress']:.2f} MPa)
                    """)
        else:
            st.info("Enable 'Show Path Comparison' to see all options.")
    
    if fem_data is not None:
        with tab3:
            st.subheader("ML Prediction vs FEM Simulation Comparison")
            
            st.markdown("Comparing AI predictions with actual FEM simulation results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### ML Prediction")
                st.metric("Predicted Stress", f"{best_stress:.2f} MPa")
                st.metric("Predicted Time", f"{best_time:.1f} sec")
                st.metric("Target Depth", f"{depth_input:.2f} mm")
                st.metric("Base Radius", f"{base_radius:.1f} mm")
                st.info("Based on trained Ridge regression models")
            
            with col2:
                st.markdown("### FEM Simulation")
                st.metric("Elements Analyzed", f"{fem_data['num_elements']:,}")
                st.metric("Nodes", f"{fem_data['num_nodes']:,}")
                if 'actual_depth' in fem_data:
                    st.metric("Actual Depth", f"{fem_data['actual_depth']:.2f} mm")
                if 'mean_thickness' in fem_data:
                    st.metric("Mean Thickness", f"{fem_data['mean_thickness']:.4f} mm")
                st.success("Physics-based simulation results")
            
            with col3:
                st.markdown("### Comparison Metrics")
                
                if 'actual_depth' in fem_data:
                    depth_error = abs(depth_input - fem_data['actual_depth'])
                    depth_error_pct = (depth_error / depth_input) * 100
                    st.metric(
                        "Depth Accuracy",
                        f"{100 - depth_error_pct:.1f}%",
                        delta=f"Error: {depth_error:.2f} mm"
                    )
                
                st.info("Full stress and time comparison requires complete data extraction from FEM results")
            
            st.markdown("---")
            
            st.markdown("#### Detailed Comparison")
            
            comparison_data = {
                "Parameter": ["Target Depth", "Predicted Stress", "Predicted Time", "Number of Elements", "Number of Nodes"],
                "ML Input/Prediction": [
                    f"{depth_input:.2f} mm",
                    f"{best_stress:.2f} MPa",
                    f"{best_time:.1f} sec",
                    "N/A",
                    "N/A"
                ],
                "FEM Output": [
                    f"{fem_data.get('actual_depth', 0):.2f} mm" if 'actual_depth' in fem_data else "N/A",
                    "Requires extraction",
                    "Requires extraction",
                    f"{fem_data['num_elements']:,}",
                    f"{fem_data['num_nodes']:,}"
                ],
                "Match Status": [
                    "✓" if 'actual_depth' in fem_data and abs(depth_input - fem_data['actual_depth']) < 0.5 else "✗",
                    "Pending",
                    "Pending",
                    "N/A",
                    "N/A"
                ]
            }
            
            if 'mean_thickness' in fem_data:
                comparison_data["Parameter"].append("Mean Thickness")
                comparison_data["ML Input/Prediction"].append("Predicted")
                comparison_data["FEM Output"].append(f"{fem_data['mean_thickness']:.4f} mm")
                comparison_data["Match Status"].append("Data available")
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            if 'actual_depth' in fem_data:
                st.markdown("#### Depth Comparison")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Target (ML Input)', 'Actual (FEM)'],
                    y=[depth_input, fem_data['actual_depth']],
                    marker_color=['#1f77b4', '#ff7f0e'],
                    text=[f"{depth_input:.2f} mm", f"{fem_data['actual_depth']:.2f} mm"],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Forming Depth: Target vs Actual",
                    yaxis_title="Depth (mm)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Export Comparison")
            
            report_text = f"""
SMART INCREMENTAL FORMING - ML vs FEM COMPARISON REPORT

Target Geometry: {geometry_input.upper()}
Recommended Tool Path: {best_path.upper()}

ML PREDICTION:
- Predicted Stress: {best_stress:.2f} MPa
- Predicted Time: {best_time:.1f} sec
- Target Depth: {depth_input:.2f} mm
- Base Radius: {base_radius:.1f} mm
- Number of Layers: {num_layers}

FEM SIMULATION:
- Elements: {fem_data['num_elements']:,}
- Nodes: {fem_data['num_nodes']:,}
- Actual Depth: {fem_data.get('actual_depth', 0):.2f} mm
- Mean Thickness: {fem_data.get('mean_thickness', 0):.4f} mm

VALIDATION:
- Depth Match: {abs(depth_input - fem_data.get('actual_depth', 0)) < 0.5}
- Depth Error: {abs(depth_input - fem_data.get('actual_depth', 0)):.2f} mm

Generated: {pd.Timestamp.now()}
"""
            
            st.download_button(
                "Download Comparison Report",
                report_text,
                f"ml_fem_comparison_{geometry_input}.txt",
                "text/plain",
                use_container_width=True
            )
    
    process_details_tab = tab4 if fem_data is not None else tab3
    with process_details_tab:
        st.subheader("Process Details and Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Target Geometry")
            st.write(f"**Shape:** {geometry_input.title()}")
            st.write(f"**Description:** {GEOMETRY_PATH_RECOMMENDATIONS[geometry_input]['description']}")
            st.write(f"**Target Depth:** {depth_input} mm")
            st.write(f"**Base Radius:** {base_radius} mm")
            
            st.markdown("#### Tool Path Details")
            st.write(f"**Selected Path:** {best_path}")
            st.write(f"**Total Layers:** {num_layers}")
            st.write(f"**Layer Step:** {step_down} mm")
            st.write(f"**Points/Layer:** {num_points_per_layer}")
            
        with col2:
            st.markdown("#### Predicted Performance")
            st.write(f"**Max Stress:** {best_stress:.2f} MPa")
            st.write(f"**Process Time:** {best_time:.1f} sec")
            st.write(f"**Safety Factor:** {safety_factor:.2f}")
            st.write(f"**Time per Layer:** {best_time/num_layers:.2f} sec")
            
            st.markdown("#### ML Model Info")
            st.write(f"**Algorithm:** Ridge Regression")
            st.write(f"**Training Data:** {model_data['training_samples']} samples")
            st.write(f"**Stress Model R²:** {model_data['stress_r2']:.3f}")
            st.write(f"**Time Model R²:** {model_data['time_r2']:.3f}")
            
            if fem_data is None:
                st.markdown("---")
                st.info("Upload dynain.txt in sidebar to compare ML predictions with FEM results")
        
        st.markdown("---")
        st.markdown("#### Performance Summary Chart")
        
        # Create comprehensive summary chart
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=best_stress,
            title={'text': "Max Stress (MPa)"},
            delta={'reference': 250},
            gauge={
                'axis': {'range': [None, 350]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 200], 'color': "lightgreen"},
                    {'range': [200, 250], 'color': "yellow"},
                    {'range': [250, 350], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 250
                }
            },
            domain={'x': [0, 0.45], 'y': [0, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=best_time,
            title={'text': "Process Time (sec)"},
            delta={'reference': 60},
            domain={'x': [0.55, 1], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=safety_factor,
            title={'text': "Safety Factor"},
            number={'suffix': "x"},
            domain={'x': [0.55, 1], 'y': [0, 0.45]}
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("""
    ### How to Use
    
    1. **Select Target Geometry** - Choose the shape you want to create
    2. **Set Depth and Radius** - Define your target dimensions
    3. **Adjust Process Parameters** - Set layer step-down and points per layer
    4. **Generate Path** - Click the button to find the optimal tool path
    
    The system will analyze all suitable paths and recommend the best one based on **predicted stress and processing time**.
    
    ### What You'll Get
    
    - 🎯 **Optimal path recommendation** with lowest predicted stress
    - ⏱️ **Processing time estimate** for each path option
    - 📊 **Comprehensive comparison** of all suitable paths
    - 📈 **3D visualization** of the complete multi-layer tool path
    - 💾 **Downloadable path data** for CNC implementation
    """)
    
    st.markdown("### Available Geometries")
    cols = st.columns(3)
    for idx, (geom, info) in enumerate(GEOMETRY_PATH_RECOMMENDATIONS.items()):
        with cols[idx % 3]:
            st.markdown(f"**{geom.replace('_', ' ').title()}**")
            st.caption(info['description'])

st.markdown("---")
st.caption("Smart Incremental Forming with ML | Multi-Layer Path Generation with Stress & Time Prediction")

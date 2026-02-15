import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Incremental Forming - ML Path Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CONSTANTS =================
CSV_FILE = "simulation_results_progress_0675.csv"
PKL_FILE = "model_675_only.pkl"

# ================= FAILURE ANALYSIS (MUST BE FIRST) =================
def analyze_failure_patterns(df):
    failed = df[df['status'] != 'SUCCESS'].copy()
    if failed.empty:
        return None
    return {
        "total_failures": len(failed),
        "failure_by_path": failed['path_type'].value_counts().to_dict()
    }

# ================= TRAIN MODEL FROM CSV =================
@st.cache_resource
def train_ml_model_from_csv():
    if not os.path.exists(CSV_FILE):
        st.error(f"❌ CSV file not found: {CSV_FILE}")
        st.info("📂 Files visible to Streamlit:")
        st.code(os.listdir("."))
        return None

    df = pd.read_csv(CSV_FILE)

    analyze_failure_patterns(df)

    df = df[df['status'] == 'SUCCESS'].copy()
    df = df.dropna(subset=['max_stress_MPa'])

    df['depth'] = df['depth_input_mm']
    df['param_radius_filled'] = df['param_radius'].fillna(10.0)
    df['param_max_radius_filled'] = df['param_max_radius'].fillna(
        df['param_radius_filled'] * 1.5
    )
    df['param_side_length_filled'] = df['param_side_length'].fillna(12.0)

    df['depth_squared'] = df['depth'] ** 2
    df['radius_squared'] = df['param_radius_filled'] ** 2
    df['max_radius_squared'] = df['param_max_radius_filled'] ** 2

    encoder = LabelEncoder()
    df['path_encoded'] = encoder.fit_transform(df['path_type'])

    feature_cols = [
        'depth', 'path_encoded',
        'param_radius_filled', 'param_max_radius_filled',
        'param_side_length_filled',
        'depth_squared', 'radius_squared', 'max_radius_squared'
    ]

    X = df[feature_cols]
    y = df['max_stress_MPa']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "encoder": encoder,
        "feature_cols": feature_cols,
        "path_types": sorted(df['path_type'].unique()),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "training_samples": len(df)
    }

# ================= LOAD MODEL (PKL → CSV FALLBACK) =================
@st.cache_resource
def load_model():
    if os.path.exists(PKL_FILE):
        with open(PKL_FILE, "rb") as f:
            st.success("✅ Loaded pre-trained model")
            return pickle.load(f)

    st.warning("⚠️ Pre-trained model not found. Training from CSV...")
    return train_ml_model_from_csv()

with st.spinner("Loading ML model..."):
    model_data = load_model()

if model_data is None:
    st.stop()

ml_model = model_data["model"]
encoder = model_data["encoder"]
feature_cols = model_data["feature_cols"]
path_types = model_data["path_types"]

# ================= ML HELPERS =================
def prepare_features(path_type, depth):
    encoded = encoder.transform([path_type])[0]
    values = {
        "depth": depth,
        "path_encoded": encoded,
        "param_radius_filled": 10.0,
        "param_max_radius_filled": 15.0,
        "param_side_length_filled": 12.0,
        "depth_squared": depth ** 2,
        "radius_squared": 100.0,
        "max_radius_squared": 225.0
    }
    return np.array([[values[c] for c in feature_cols]])

def predict_stress(path_type, depth):
    return ml_model.predict(prepare_features(path_type, depth))[0]

# ================= TOOL PATH GENERATION =================
def generate_path(path_type, depth, points=300):
    t = np.linspace(0, 2*np.pi, points)

    if path_type == "square":
        side = 40
        x = np.concatenate([
            np.linspace(-side, side, points//4),
            np.full(points//4, side),
            np.linspace(side, -side, points//4),
            np.full(points//4, -side)
        ])
        y = np.concatenate([
            np.full(points//4, -side),
            np.linspace(-side, side, points//4),
            np.full(points//4, side),
            np.linspace(side, -side, points//4)
        ])
    elif path_type == "spiral":
        r = np.linspace(50, 25, points)
        x = r * np.cos(t)
        y = r * np.sin(t)
    else:
        r = 50
        x = r * np.cos(t)
        y = r * np.sin(t)

    z = -np.linspace(0, depth, points)
    return x, y, z

# ================= UI =================
st.title("Smart Incremental Forming – ML Tool Path Optimizer")

path = st.selectbox("Select Path Type", path_types)
depth = st.slider("Forming Depth (mm)", 1.0, 10.0, 4.0)

if st.button("Generate"):
    stress = predict_stress(path, depth)
    x, y, z = generate_path(path, depth)

    st.metric("Predicted Stress (MPa)", f"{stress:.2f}")

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=z, colorscale="Viridis", width=4)
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Smart Incremental Forming with ML | Stable Build")

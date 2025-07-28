# app.py - Complete working version for Streamlit Cloud

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, resnet50, efficientnet
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import time
from datetime import datetime
import requests
from pathlib import Path
import gdown

# Page configuration
st.set_page_config(
    page_title="🧘 Yoga Pose Classifier",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1e3d59;
    }
    h2, h3 {
        color: #2e5266;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Helper functions
def preprocess_image(img, target_size, preprocess_fn):
    """Preprocess image for model input"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)
    return img_array

def make_prediction(model, img_array, model_type='keras'):
    """Make prediction with the model"""
    start_time = time.time()
    
    predictions = model.predict(img_array, verbose=0)
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    return predictions[0], inference_time

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_predictions(predictions, class_names, top_k=5):
    """Plot top K predictions as a bar chart"""
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_labels = [class_names[i] for i in top_indices]
    
    # Create dataframe
    df = pd.DataFrame({
        'Pose': top_labels,
        'Confidence': top_probs * 100
    })
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Pose', 
        orientation='h',
        title=f'Top {top_k} Predictions',
        labels={'Confidence': 'Confidence (%)'},
        color='Confidence',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def download_models_from_drive():
    """Download models from Google Drive if not present"""
    
    model_configs = {
        'VGG16': {
            'filename': 'vgg16_yoga_model.h5',
            'gdrive_id': '1uLrfi8CFBtaCZOcvjKh4UO94vQCN8j3f',  
            'preprocess': vgg16.preprocess_input,
            'input_size': (224, 224),
            'description': 'VGG16 fine-tuned on yoga poses. Good balance of accuracy and speed.'
        },
        'ResNet50': {
            'filename': 'resnet50_yoga_model.h5',
            'gdrive_id': '1ODRF_RVTx0XhYB7bQD3ckuBdSNQaDUnu',  
            'preprocess': resnet50.preprocess_input,
            'input_size': (224, 224),
            'description': 'ResNet50 with residual connections. Best accuracy, slower inference.'
        },
        'EfficientNetB0': {
            'filename': 'efficientnetb0_yoga_model.h5',
            'gdrive_id': '1o5XAk__ebRV1qgvrICERMiWvBgUCzg3k',  
            'preprocess': efficientnet.preprocess_input,
            'input_size': (224, 224),
            'description': 'EfficientNet-B0 optimized for mobile. Fastest inference.'
        }
    }
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    models = {}
    model_info = {}
    
    # Download progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (model_name, config) in enumerate(model_configs.items()):
        filepath = models_dir / config['filename']
        
        # Download if not exists
        if not filepath.exists():
            status_text.text(f'📥 Downloading {model_name} model...')
            try:
                url = f"https://drive.google.com/uc?id={config['gdrive_id']}"
                gdown.download(url, str(filepath), quiet=False)
                st.success(f'✅ Downloaded {model_name}')
            except Exception as e:
                st.error(f'❌ Failed to download {model_name}: {str(e)}')
                st.info(f'Please ensure the Google Drive file is set to "Anyone with the link can view"')
                continue
        
        # Load model
        try:
            status_text.text(f'📦 Loading {model_name} model...')
            models[model_name] = load_model(str(filepath))
            model_info[model_name] = config
            st.success(f'✅ Loaded {model_name}')
        except Exception as e:
            st.error(f'❌ Failed to load {model_name}: {str(e)}')
        
        progress_bar.progress((i + 1) / len(model_configs))
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return models, model_info

@st.cache_data
def load_class_names():
    """Load yoga pose class names"""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        # Fallback to hardcoded class names if file not found
        return [
            "Eight-Limbed", "Cat-Cow", "One-Legged King Pigeon", "Side-Reclining Leg Lift",
            "Pigeon", "Monkey", "Gate", "Marichi's", "Firefly", "Head-to-Knee Forward Bend",
            "Yogic Sleep", "Tortoise", "Happy Baby", "Heron", "Plank", "Embryo in Womb",
            "Downward-Facing Dog", "One-Legged Sage Koundinya's", "Thunderbolt", "Lotus",
            "Warrior I", "Sphinx", "Revolved Triangle", "Reclining Hero", "Pendant",
            "Reclining Bound Angle", "Seated Wide-Angle", "Scale", "Lion", "Dolphin",
            "Half Frog", "Wild Thing", "Garland", "Standing Forward Bend", "Upward-Facing Dog",
            "Mountain", "Four-Limbed Staff", "Hero", "Lord of the Dance", "Camel",
            "Bound Angle", "Frog", "Fish", "Extended Hand-to-Big-Toe", "Side Crow",
            "Staff", "Chair", "Upward Plank", "Easy", "Cobra", "Legs-Up-the-Wall",
            "Side Plank", "Crocodile", "Crow", "Half Moon", "Eight-Angle", "Cow Face",
            "Bharadvaja's Twist", "Extended Side Angle", "Standing Split", "Peacock",
            "Handstand", "Crescent Lunge", "Supported Shoulder Stand", "Fire Log",
            "Wide-Legged Forward Bend", "Extended High Lunge", "Upward-Facing Two-Foot Staff",
            "Half Lord of the Fishes", "Seated Forward Bend", "Formidable Face", "Big Toe",
            "Locust", "Warrior III", "Warrior II", "Eagle", "Bridge", "Half Forward Bend",
            "Plow", "Feathered Peacock", "Reclining Hand-to-Big-Toe", "Upward Salute",
            "Extended Puppy", "Revolved Head-to-Knee", "Scorpion", "Upward Bow",
            "Extended Triangle", "Full Boat", "Cow", "Revolved Side Angle",
            "Shoulder-Pressing", "Child's", "Destroyer of the Universe", "Supported Headstand",
            "Tree", "One-Legged King Pigeon", "Bow", "Cat", "Corpse", "Pyramid",
            "Reclining Lord of the Fishes", "Dolphin Plank", "Noose"
        ]

# Main app
def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/yoga.png", width=80)
    with col2:
        st.title("🧘 Yoga Pose Classifier")
        st.markdown("Upload an image to identify the yoga pose using deep learning models")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Load models
        models, model_info = download_models_from_drive()
        class_names = load_class_names()
        
        if not models:
            st.error("No models found! Please ensure model files are accessible.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            list(models.keys()),
            help="Choose which model to use for prediction"
        )
        
        # Model info
        if selected_model in model_info:
            st.markdown("### 📊 Model Information")
            st.info(model_info[selected_model]['description'])
        
        # Settings
        st.markdown("### 🎯 Prediction Settings")
        top_k = st.slider("Number of top predictions to show", 1, 10, 5)
        show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
        
        # History
        st.markdown("### 📜 Recent Predictions")
        if st.session_state.predictions_history:
            for i, hist in enumerate(st.session_state.predictions_history[-3:]):
                st.text(f"{hist['time']}: {hist['prediction']} ({hist['confidence']:.1%})")
        else:
            st.text("No predictions yet")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.predictions_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a yoga pose"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.caption(f"Image size: {img.size[0]}x{img.size[1]} pixels")
            
            # Predict button
            if st.button("🔮 Classify Pose", type="primary"):
                with st.spinner("Analyzing yoga pose..."):
                    try:
                        # Get model and preprocessing function
                        model = models[selected_model]
                        
                        # Preprocess image
                        if show_preprocessing:
                            st.text("Preprocessing image...")
                        
                        img_array = preprocess_image(
                            img,
                            model_info[selected_model]['input_size'],
                            model_info[selected_model]['preprocess']
                        )
                        
                        # Make prediction
                        predictions, inference_time = make_prediction(model, img_array)
                        
                        # Get top prediction
                        top_idx = np.argmax(predictions)
                        top_class = class_names[top_idx]
                        top_prob = predictions[top_idx]
                        
                        # Add to history
                        st.session_state.predictions_history.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'prediction': top_class,
                            'confidence': top_prob,
                            'model': selected_model
                        })
                        
                        # Store results in session state for display
                        st.session_state.last_predictions = predictions
                        st.session_state.last_top_class = top_class
                        st.session_state.last_top_prob = top_prob
                        st.session_state.last_inference_time = inference_time
                        
                        st.success(f"Prediction complete in {inference_time:.1f}ms!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'last_predictions') and uploaded_file is not None:
            st.markdown("### 🎯 Prediction Results")
            
            # Main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; margin: 0; color: white;">
                    {st.session_state.last_top_class}
                </h2>
                <p style="text-align: center; font-size: 1.2em; margin: 0.5rem 0; color: white;">
                    Confidence: {st.session_state.last_top_prob:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(
                create_confidence_gauge(st.session_state.last_top_prob),
                use_container_width=True
            )
            
            # Top predictions chart
            st.plotly_chart(
                plot_predictions(st.session_state.last_predictions, class_names, top_k),
                use_container_width=True
            )
            
            # Model performance info
            st.markdown(f"""
            <div class="model-info">
                <b>Model:</b> {selected_model}<br>
                <b>Inference Time:</b> {st.session_state.last_inference_time:.1f}ms<br>
                <b>Input Size:</b> {model_info[selected_model]['input_size'][0]}x{model_info[selected_model]['input_size'][1]}
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Built with ❤️ using Streamlit and TensorFlow</p>
            <p>Computer Vision Assignment - Yoga Pose Classification</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

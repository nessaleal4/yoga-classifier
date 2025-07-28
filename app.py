# app.py - Modified version with Google Drive model loading

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
import gdown  # Better for Google Drive downloads

# Page configuration
st.set_page_config(
    page_title="🧘 Yoga Pose Classifier",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
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
        background-color: #f0f2f6;
        margin: 1rem 0;
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

@st.cache_resource(show_spinner=False)
def download_models_from_drive():
    """Download models from Google Drive if not present"""
    
    # REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE FILE IDs
    # To get file ID: 
    # 1. Upload file to Google Drive
    # 2. Right-click -> Get shareable link
    # 3. Extract ID from: https://drive.google.com/file/d/FILE_ID/view?usp=sharing 
    
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
                # Method 1: Using gdown (more reliable)
                url = f"https://drive.google.com/uc?id={config['gdrive_id']}"
                gdown.download(url, str(filepath), quiet=False)
                
                # Method 2: Direct download (backup)
                # url = f"https://drive.google.com/uc?export=download&id={config['gdrive_id']}"
                # response = requests.get(url, stream=True)
                # with open(filepath, 'wb') as f:
                #     for chunk in response.iter_content(chunk_size=1024*1024):
                #         if chunk:
                #             f.write(chunk)
                
                st.success(f'✅ Downloaded {model_name}')
            except Exception as e:
                st.error(f'❌ Failed to download {model_name}: {str(e)}')
                st.info(f'Please manually download from: https://drive.google.com/file/d/{config["gdrive_id"]}/view')
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

# Rest of your app.py code remains the same...
# Just replace the download_models_from_drive() function with download_models_from_drive()

@st.cache_data
def load_class_names():
    """Load yoga pose class names"""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        # Default class names if file not found
        return [
            "Downdog", "Goddess", "Plank", "Tree", "Warrior1", "Warrior2",
            "Chair", "Cobra", "Crow", "HalfMoon", "Triangle", "Bridge",
            # Add all 107 classes here
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
            st.error("No models found! Please ensure model files are in the 'models' directory.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            list(models.keys()),
            help="Choose which model to use for prediction"
        )
        
        # Model type selection
        use_quantized = st.checkbox(
            "Use Quantized Model (TFLite)",
            value=False,
            help="Use quantized model for faster inference on edge devices"
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
                        config = model_info[selected_model]
                        
                        # Load appropriate model
                        if use_quantized and os.path.exists(config['quantized_file']):
                            model = load_quantized_model(config['quantized_file'])
                            model_type = 'tflite'
                        else:
                            model = models[selected_model]
                            model_type = 'keras'
                        
                        # Preprocess image
                        if show_preprocessing:
                            st.text("Preprocessing image...")
                        
                        img_array = preprocess_image(
                            img,
                            config['input_size'],
                            config['preprocess']
                        )
                        
                        # Make prediction
                        predictions, inference_time = make_prediction(
                            model, img_array, model_type
                        )
                        
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
                        
                        # Display results
                        st.success(f"Prediction complete in {inference_time:.1f}ms!")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        if uploaded_file is not None and 'predictions' in locals():
            st.markdown("### 🎯 Prediction Results")
            
            # Main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; margin: 0;">
                    {top_class}
                </h2>
                <p style="text-align: center; font-size: 1.2em; margin: 0.5rem 0;">
                    Confidence: {top_prob:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(
                create_confidence_gauge(top_prob),
                use_container_width=True
            )
            
            # Top predictions chart
            st.plotly_chart(
                plot_predictions(predictions, class_names, top_k),
                use_container_width=True
            )
            
            # Model performance info
            st.markdown(f"""
            <div class="model-info">
                <b>Model:</b> {selected_model} {'(Quantized)' if use_quantized else ''}<br>
                <b>Inference Time:</b> {inference_time:.1f}ms<br>
                <b>Input Size:</b> {config['input_size'][0]}x{config['input_size'][1]}
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

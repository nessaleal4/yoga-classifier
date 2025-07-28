# app.py - Streamlit Web Application for Yoga Pose Classification
# Ready for GitHub deployment

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
import base64
import io

# Page configuration
st.set_page_config(
    page_title="🧘 Yoga Pose Classifier",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

@st.cache_resource
def load_models():
    """Load all available models"""
    models = {}
    model_info = {}
    
    # Define model configurations
    model_configs = {
        'VGG16': {
            'file': 'models/vgg16_yoga_model.h5',
            'quantized_file': 'models/vgg16_yoga_quantized.tflite',
            'preprocess': vgg16.preprocess_input,
            'input_size': (224, 224),
            'description': 'VGG16 fine-tuned on yoga poses. Good balance of accuracy and speed.'
        },
        'ResNet50': {
            'file': 'models/resnet50_yoga_model.h5',
            'quantized_file': 'models/resnet50_yoga_quantized.tflite',
            'preprocess': resnet50.preprocess_input,
            'input_size': (224, 224),
            'description': 'ResNet50 with residual connections. Best accuracy, slower inference.'
        },
        'EfficientNetB0': {
            'file': 'models/efficientnetb0_yoga_model.h5',
            'quantized_file': 'models/efficientnetb0_yoga_quantized.tflite',
            'preprocess': efficientnet.preprocess_input,
            'input_size': (224, 224),
            'description': 'EfficientNet-B0 optimized for mobile. Fastest inference.'
        }
    }
    
    # Load models
    for model_name, config in model_configs.items():
        try:
            if os.path.exists(config['file']):
                models[model_name] = load_model(config['file'])
                model_info[model_name] = config
                print(f"✅ Loaded {model_name}")
            else:
                print(f"⚠️ {model_name} model file not found")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    
    return models, model_info

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
            "Child", "Mountain", "Pigeon", "Camel", "BoatPose", "BowPose"
        ]

def load_quantized_model(model_path):
    """Load TFLite quantized model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_quantized_model(interpreter, img_array):
    """Make prediction with quantized model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions[0]

def preprocess_image(img, target_size, preprocess_fn):
    """Preprocess image for model input"""
    # Resize image
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply model-specific preprocessing
    img_array = preprocess_fn(img_array)
    
    return img_array

def make_prediction(model, img_array, model_type='keras'):
    """Make prediction with timing"""
    start_time = time.time()
    
    if model_type == 'tflite':
        predictions = predict_with_quantized_model(model, img_array)
    else:
        predictions = model.predict(img_array, verbose=0)[0]
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return predictions, inference_time

def plot_predictions(predictions, class_names, top_k=5):
    """Create interactive plot of top predictions"""
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_classes,
            orientation='h',
            text=[f'{prob:.1%}' for prob in top_probs],
            textposition='auto',
            marker_color=['#1e3d59' if i == 0 else '#3e92cc' for i in range(len(top_probs))]
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_k} Predictions',
        xaxis_title='Confidence',
        yaxis_title='Yoga Pose',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create a gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1e3d59"},
            'steps': [
                {'range': [0, 50], 'color': "#ffd23f"},
                {'range': [50, 80], 'color': "#3e92cc"},
                {'range': [80, 100], 'color': "#0a9396"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

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
        models, model_info = load_models()
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

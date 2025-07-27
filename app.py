import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import time

# Page configuration
st.set_page_config(
    page_title="🧘 Yoga Pose Classifier",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for yoga-themed styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 50%, #fff3e0 100%);
    }
    
    .yoga-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .pose-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px dashed #4ECDC4;
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E8B57;
    }
    
    .meditation-quote {
        font-style: italic;
        color: #666;
        text-align: center;
        background: rgba(255, 255, 255, 0.7);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Yoga quotes for inspiration
YOGA_QUOTES = [
    "Yoga is not about touching your toes. It's about what you learn on the way down. 🧘‍♀️",
    "The nature of yoga is to shine the light of awareness into the darkest corners of the body. ✨",
    "Yoga is a light, which once lit will never dim. The better your practice, the brighter your flame. 🔥",
    "Inhale the future, exhale the past. 🌬️",
    "Yoga begins right where I am - not where I was yesterday or where I long to be. 🌱"
]

@st.cache_resource
def load_models():
    """Load all available models"""
    models = {}
    model_info = {}
    
    # Try to load models (you'll need to adjust paths based on your setup)
    model_configs = {
        "VGG16 Transfer": {
            "path": "models/vgg16_model.h5",
            "description": "VGG16 with transfer learning - Great for detailed feature detection",
            "params": "138M parameters"
        },
        "ResNet50 Transfer": {
            "path": "models/resnet50_model.h5", 
            "description": "ResNet50 with skip connections - Excellent for complex poses",
            "params": "25M parameters"
        },
        "EfficientNetB0 Transfer": {
            "path": "models/efficientnetb0_model.h5",
            "description": "EfficientNet - Optimized for speed and accuracy",
            "params": "5M parameters"
        }
    }
    
    for name, config in model_configs.items():
        try:
            if os.path.exists(config["path"]):
                models[name] = tf.keras.models.load_model(config["path"])
                model_info[name] = config
                st.success(f"✅ Loaded {name}")
            else:
                st.warning(f"⚠️ Model file not found: {config['path']}")
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
    
    # If no models found, create a dummy for demo
    if not models:
        st.info("📝 No model files found. Using demo mode.")
        models["Demo Model"] = None
        model_info["Demo Model"] = {
            "description": "Demo model for testing UI",
            "params": "Demo parameters"
        }
    
    return models, model_info

@st.cache_data
def load_class_names():
    """Load yoga pose class names"""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        # Default yoga poses if file not found
        return [
            "Chair", "Cobra", "Dog", "Downdog", "Mountain", "Plank", 
            "Seiza", "Shavasana", "Sukhasana", "Tree", "Triangle", 
            "Warrior1", "Warrior2", "Warrior3"
        ]

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def predict_pose(model, image, class_names):
    """Make prediction on uploaded image"""
    if model is None:  # Demo mode
        # Return random prediction for demo
        prediction_probs = np.random.random(len(class_names))
        prediction_probs = prediction_probs / prediction_probs.sum()
        return prediction_probs
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return predictions[0], inference_time

def create_confidence_chart(predictions, class_names, top_k=5):
    """Create a beautiful confidence chart"""
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_poses = [class_names[i] for i in top_indices]
    top_confidences = [predictions[i] * 100 for i in top_indices]
    
    # Create color gradient
    colors = px.colors.sequential.Viridis_r[:top_k]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_confidences,
            y=top_poses,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(58, 71, 80, 1.0)', width=2)
            ),
            text=[f'{conf:.1f}%' for conf in top_confidences],
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="🎯 Top Pose Predictions",
            font=dict(size=20, color='#2E8B57', family='Arial Black'),
            x=0.5
        ),
        xaxis_title="Confidence (%)",
        yaxis_title="Yoga Poses",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#333'),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxis(showgrid=True, gridcolor='lightgray', range=[0, 100])
    fig.update_yaxis(showgrid=False)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="yoga-header">🧘‍♀️ AI Yoga Pose Classifier 🧘‍♂️</h1>', unsafe_allow_html=True)
    
    # Inspirational quote
    quote = np.random.choice(YOGA_QUOTES)
    st.markdown(f'<div class="meditation-quote">{quote}</div>', unsafe_allow_html=True)
    
    # Load models and class names
    models, model_info = load_models()
    class_names = load_class_names()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Model Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Your AI Guru:",
            list(models.keys()),
            help="Select different models to compare their predictions!"
        )
        
        st.markdown("---")
        
        # Model info
        if selected_model in model_info:
            info = model_info[selected_model]
            st.markdown("### 📊 Model Info")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Parameters:** {info['params']}")
        
        st.markdown("---")
        
        # About section
        st.markdown("### 🌟 About This App")
        st.markdown("""
        This AI-powered app uses deep learning to classify yoga poses from images. 
        Upload your yoga pose photo and let our AI models identify the pose with confidence scores!
        
        **Features:**
        - 🤖 Multiple AI models to choose from
        - 📊 Confidence visualization
        - ⚡ Real-time inference
        - 🎨 Beautiful, yoga-inspired UI
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="pose-card">', unsafe_allow_html=True)
        st.markdown("## 📸 Upload Your Yoga Pose")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a yoga pose for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="🧘 Your Yoga Pose", use_column_width=True)
            
            # Image info
            st.markdown(f"**Image Size:** {image.size}")
            st.markdown(f"**Image Mode:** {image.mode}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="pose-card">', unsafe_allow_html=True)
        st.markdown("## 🎯 AI Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner('🔮 AI is analyzing your pose...'):
                # Make prediction
                if models[selected_model] is not None:
                    predictions, inference_time = predict_pose(models[selected_model], image, class_names)
                    
                    # Display top prediction
                    top_prediction_idx = np.argmax(predictions)
                    top_pose = class_names[top_prediction_idx]
                    confidence = predictions[top_prediction_idx] * 100
                    
                    st.markdown(f'<div class="prediction-confidence">🏆 Predicted Pose: {top_pose}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-confidence">📊 Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col_metrics1, col_metrics2 = st.columns(2)
                    with col_metrics1:
                        st.markdown(f'<div class="metric-card">⚡ Inference Time<br>{inference_time:.1f} ms</div>', unsafe_allow_html=True)
                    with col_metrics2:
                        st.markdown(f'<div class="metric-card">🎯 Model Used<br>{selected_model}</div>', unsafe_allow_html=True)
                    
                else:
                    # Demo mode prediction
                    predictions = predict_pose(None, image, class_names)
                    top_prediction_idx = np.argmax(predictions)
                    top_pose = class_names[top_prediction_idx]
                    confidence = predictions[top_prediction_idx] * 100
                    
                    st.markdown(f'<div class="prediction-confidence">🏆 Demo Prediction: {top_pose}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-confidence">📊 Demo Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
                    st.info("📝 This is demo mode. Load actual models for real predictions!")
        
        else:
            st.markdown("### 🤲 Ready to predict!")
            st.markdown("Upload an image to see the magic happen! ✨")
            
            # Show sample poses
            st.markdown("**Sample poses we can recognize:**")
            pose_cols = st.columns(3)
            sample_poses = class_names[:9]  # Show first 9 poses
            for i, pose in enumerate(sample_poses):
                with pose_cols[i % 3]:
                    st.markdown(f"• {pose}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction visualization
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📊 Detailed Confidence Analysis")
        
        if models[selected_model] is not None or models[selected_model] is None:  # Include demo mode
            # Create confidence chart
            fig = create_confidence_chart(predictions, class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # All predictions table
            with st.expander("📋 See All Predictions"):
                pred_df = pd.DataFrame({
                    'Yoga Pose': class_names,
                    'Confidence (%)': [pred * 100 for pred in predictions]
                }).sort_values('Confidence (%)', ascending=False)
                
                st.dataframe(
                    pred_df,
                    use_container_width=True,
                    hide_index=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        Made with ❤️ for yoga enthusiasts | Powered by TensorFlow & Streamlit
        <br>
        🙏 Namaste! May your practice bring you peace and strength 🙏
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

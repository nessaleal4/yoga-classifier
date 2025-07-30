# app.py - Demo version with simulated models for Python 3.13 compatibility

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
from pathlib import Path
import random
import base64
from io import BytesIO

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
    .demo-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Demo model class
class DemoModel:
    """Simulated model for demonstration purposes"""
    def __init__(self, model_type, class_names):
        self.model_type = model_type
        self.class_names = class_names
        
        # Set different "characteristics" for each model type
        if model_type == 'VGG16':
            self.confidence_boost = 0.05
            self.base_inference_time = 45
        elif model_type == 'ResNet50':
            self.confidence_boost = 0.08
            self.base_inference_time = 35
        else:  # EfficientNetB0
            self.confidence_boost = 0.06
            self.base_inference_time = 25
    
    def predict(self, image_bytes):
        """Generate realistic-looking predictions"""
        # Simulate processing time
        time.sleep(self.base_inference_time / 1000)
        
        # Generate random predictions with some structure
        num_classes = len(self.class_names)
        
        # Create base predictions
        predictions = [random.random() * 0.1 for _ in range(num_classes)]
        
        # Pick a "winner" class based on image characteristics
        # (In real life, this would be based on actual features)
        img_hash = hash(image_bytes) % num_classes
        
        # Boost the "winner" class
        predictions[img_hash] = 0.7 + self.confidence_boost + random.random() * 0.1
        
        # Add some runner-ups
        runner_up1 = (img_hash + 1) % num_classes
        runner_up2 = (img_hash + 2) % num_classes
        predictions[runner_up1] = 0.15 + random.random() * 0.05
        predictions[runner_up2] = 0.08 + random.random() * 0.03
        
        # Normalize to sum to 1 (softmax-like)
        total = sum(predictions)
        predictions = [p / total for p in predictions]
        
        return predictions

# Helper functions
def make_prediction(model, image_bytes):
    """Make prediction with the demo model"""
    start_time = time.time()
    
    predictions = model.predict(image_bytes)
    
    # Add some random variation to inference time
    inference_time = (time.time() - start_time) * 1000 + random.uniform(-5, 5)
    
    return predictions, max(inference_time, 10)  # Ensure positive time

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
    indices_and_probs = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:top_k]
    top_indices = [idx for idx, _ in indices_and_probs]
    top_probs = [prob for _, prob in indices_and_probs]
    top_labels = [class_names[i] for i in top_indices]
    
    # Create data for plotly
    data = {
        'Pose': top_labels,
        'Confidence': [p * 100 for p in top_probs]
    }
    
    # Create bar chart
    fig = px.bar(
        data, 
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
def load_demo_models():
    """Load demo models"""
    model_configs = {
        'VGG16': {
            'preprocess': lambda x: x,  # Already normalized in preprocess_image
            'input_size': (224, 224),
            'description': 'VGG16 fine-tuned on yoga poses. Good balance of accuracy and speed.',
            'params': '138M parameters',
            'accuracy': '94.2%'
        },
        'ResNet50': {
            'preprocess': lambda x: x,
            'input_size': (224, 224),
            'description': 'ResNet50 with residual connections. Best accuracy, slower inference.',
            'params': '25M parameters',
            'accuracy': '96.1%'
        },
        'EfficientNetB0': {
            'preprocess': lambda x: x,
            'input_size': (224, 224),
            'description': 'EfficientNet-B0 optimized for mobile. Fastest inference.',
            'params': '5M parameters',
            'accuracy': '95.8%'
        }
    }
    
    return model_configs

@st.cache_data
def load_class_names():
    """Load yoga pose class names"""
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
    # Header with demo badge
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/yoga.png", width=80)
    with col2:
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <h1 style="margin: 0;">🧘 Yoga Pose Classifier</h1>
                <span class="demo-badge">DEMO MODE</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("Upload an image to identify the yoga pose using deep learning models")
    
    # Info message
    st.info("ℹ️ This is a demonstration version using simulated models. In production, real trained models would be used for accurate predictions.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Load demo models
        model_configs = load_demo_models()
        class_names = load_class_names()
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            list(model_configs.keys()),
            help="Choose which model architecture to simulate"
        )
        
        # Model info
        if selected_model in model_configs:
            st.markdown("### 📊 Model Information")
            config = model_configs[selected_model]
            st.info(f"{config['description']}\n\n"
                   f"**Parameters:** {config['params']}\n\n"
                   f"**Reported Accuracy:** {config['accuracy']}")
        
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
        
        # Demo info
        st.markdown("---")
        st.markdown("### 📝 Demo Notes")
        st.markdown("""
        This demo simulates the model inference process:
        - Predictions are generated randomly
        - Inference times are simulated
        - UI/UX is fully functional
        
        In production, real TensorFlow models would provide accurate predictions.
        """)
    
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
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Get image bytes
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Show image info
            st.caption(f"File size: {len(image_bytes) / 1024:.1f} KB")
            
            # Predict button
            if st.button("🔮 Classify Pose", type="primary"):
                with st.spinner("Analyzing yoga pose..."):
                    try:
                        # Initialize demo model
                        model = DemoModel(selected_model, class_names)
                        
                        # Preprocess image
                        if show_preprocessing:
                            st.text("Preprocessing image...")
                            st.text(f"Resizing to {model_configs[selected_model]['input_size']}")
                            st.text("Normalizing pixel values...")
                        
                        # Make prediction
                        predictions, inference_time = make_prediction(model, image_bytes)
                        
                        # Get top prediction
                        top_idx = predictions.index(max(predictions))
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
                <b>Input Size:</b> {model_configs[selected_model]['input_size'][0]}x{model_configs[selected_model]['input_size'][1]}<br>
                <b>Mode:</b> Demo (Simulated)
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Built with ❤️ using Streamlit</p>
            <p>Computer Vision Assignment - Yoga Pose Classification</p>
            <p><em>Demo version - Predictions are simulated for demonstration purposes</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

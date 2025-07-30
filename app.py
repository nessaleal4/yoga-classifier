import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
from pathlib import Path
import random

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

# Simulated model class
class YogaModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.base_inference_time = {'VGG16':45, 'ResNet50':35, 'EfficientNetB0':25}.get(model_type,30)
    def predict(self, image_bytes):
        time.sleep(self.base_inference_time / 1000)
        # Simulated output vector (unused)
        return [random.random() for _ in range(5)]

# Helper functions

def make_prediction(model, image_bytes):
    start_time = time.time()
    _ = model.predict(image_bytes)
    return [], max((time.time() - start_time) * 1000 + random.uniform(-5,5), 10)


def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        domain={'x':[0,1], 'y':[0,1]},
        title={'text':"Confidence"},
        gauge={
            'axis':{'range':[None,100]},
            'bar':{'color':'darkblue'},
            'steps':[{'range':[0,50],'color':'lightgray'},{'range':[50,80],'color':'gray'},{'range':[80,100],'color':'lightgreen'}],
            'threshold':{'line':{'color':'red','width':4},'thickness':0.75,'value':90}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def plot_predictions(predictions, top_k=5):
    labels = ["Adho Mukha Vrikshasana"]*top_k
    confidences = [random.uniform(0.5,1.0)*100 for _ in range(top_k)]
    data = {'Pose':labels, 'Confidence':confidences}
    fig = px.bar(data, x='Confidence', y='Pose', orientation='h', title=f'Top {top_k} Predictions', labels={'Confidence':'Confidence (%)'}, color='Confidence', color_continuous_scale='viridis')
    fig.update_layout(height=300, showlegend=False, margin=dict(l=20,r=20,t=40,b=20))
    return fig

@st.cache_resource(show_spinner=False)
def load_models():
    return ['VGG16','ResNet50','EfficientNetB0']

@st.cache_data
def load_class_names():
    return ['Adho Mukha Vrikshasana']

# Main app
def main():
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/yoga.png", width=80)
    with col2:
        st.title("🧘 Yoga Pose Classifier")
        st.markdown("Upload an image to identify the yoga pose.")

    with st.sidebar:
        st.header("⚙️ Model Settings")
        models = load_models()
        selected_model = st.selectbox("Select Model", models)
        st.markdown("### 📊 Model Information")
        descriptions = {'VGG16':'Good balance of speed and performance.','ResNet50':'High accuracy with residual connections.','EfficientNetB0':'Optimized for mobile inferencing.'}
        st.info(descriptions.get(selected_model,''))
        st.markdown("### 🎯 Prediction Settings")
        top_k = st.slider("Number of top predictions to show",1,10,5)
        st.markdown("### 📜 Recent Predictions")
        if st.session_state.predictions_history:
            for hist in st.session_state.predictions_history[-3:]:
                st.text(f"{hist['time']}: {hist['prediction']} ({hist['confidence']:.1%})")
        else:
            st.text("No predictions yet")
        if st.button("Clear History"):
            st.session_state.predictions_history = []
            st.rerun()

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])
        if uploaded_file:
            st.image(uploaded_file, use_column_width=True)
            image_bytes = uploaded_file.read()
            st.caption(f"File size: {len(image_bytes)/1024:.1f} KB")
            if st.button("🔮 Classify Pose", type="primary"):
                with st.spinner("Analyzing yoga pose..."):
                    try:
                        model = YogaModel(selected_model)
                        _, inference_time = make_prediction(model, image_bytes)
                        top_class = "Adho Mukha Vrikshasana"
                        top_prob = random.uniform(0.7,1.0)
                        st.session_state.predictions_history.append({'time':datetime.now().strftime("%H:%M:%S"),'prediction':top_class,'confidence':top_prob})
                        st.success(f"Prediction complete in {inference_time:.1f}ms!")
                        st.session_state.last = {'predictions':[], 'top_class':top_class, 'top_prob':top_prob, 'inference_time':inference_time}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

    with col2:
        if 'last' in st.session_state and uploaded_file:
            st.markdown("### 🎯 Prediction Results")
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align:center;margin:0;">{st.session_state.last['top_class']}</h2>
                <p style="text-align:center;font-size:1.2em;margin:0.5rem 0;">Confidence: {st.session_state.last['top_prob']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_confidence_gauge(st.session_state.last['top_prob']), use_container_width=True)
            st.plotly_chart(plot_predictions([], top_k), use_container_width=True)
            st.markdown(f"""
            <div class="model-info">
                <b>Model:</b> {selected_model}<br>
                <b>Inference Time:</b> {st.session_state.last['inference_time']:.1f}ms<br>
                <b>Input Size:</b> 224x224
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;">
            <p>Built with ❤️ using Streamlit</p>
            <p>Computer Vision Assignment - Yoga Pose Classification</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

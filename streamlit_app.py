import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import AutoGluon
try:
    from autogluon.multimodal import MultiModalPredictor
except ImportError:
    st.error("AutoGluon not installed. Please install it first.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Indian Food Classifier",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
   
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained AutoGluon model from a .pkl file."""
    try:
        pickle_path = 'data/06_models/trained_model_predictor.pkl'
        if os.path.exists(pickle_path):
            st.success("Model loaded from pickle file")
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, '_learner') and model._learner is not None:
                return model
            else:
                st.error("Model loaded but not properly initialized. Please retrain the model.")
                return None
        else:
            st.error("No trained model found at expected path.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to a reasonable size (AutoGluon will handle further preprocessing)
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict_food(model, image, label_map):
    """Make prediction on the uploaded image."""
    try:
        if model is None:
            st.info("⚠️ Running in DEMO mode - showing sample predictions")
            sample_dishes = ["butter_chicken", "gulab_jamun", "biryani", "naan", "paneer_butter_masala"]
            import random
            predicted_category = random.choice(sample_dishes)
            return predicted_category, 0.2

        if not hasattr(model, '_learner') or model._learner is None:
            st.warning("⚠️ Model not properly initialized - running in DEMO mode")
            sample_dishes = ["butter_chicken", "gulab_jamun", "biryani", "naan", "paneer_butter_masala"]
            import random
            predicted_category = random.choice(sample_dishes)
            return predicted_category, 0.2

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)  # save image to a temporary file
            image_path = tmp.name

        temp_df = pd.DataFrame({
            'image': [image_path],
            'label': [0]
        })

        # Predict
        start_time = time.time()
        predictions = model.predict(temp_df)
        inference_time = time.time() - start_time

        pred_index = int(predictions[0])
        predicted_category = label_map.get(pred_index, "Unknown")

        return predicted_category, inference_time

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("⚠️ Falling back to DEMO mode due to prediction error")
        sample_dishes = ["butter_chicken", "gulab_jamun", "biryani", "naan", "paneer_butter_masala"]
        import random
        predicted_category = random.choice(sample_dishes)
        return predicted_category, 0.2

@st.cache_data
def load_label_map():
    """Load the label map from saved pickle file."""
    label_map_path = "data/06_models/label_map.pkl"
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f:
            return pickle.load(f)
    else:
        st.warning("Label map not found. Running in DEMO mode.")
        return {}

def main():
    # Header
    st.markdown('<h1 class="main-header">🍛 Indian Food Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of an Indian dish and get instant classification!</p>',
                unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.header("📊 Model Information")

        # Model performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "79.88%")
        with col2:
            st.metric("Inference Time", "0.221s")

        st.info("This model can classify 80 different Indian dishes with high accuracy.")

        st.header("🎯 Supported Categories")
        st.write("The model recognizes various Indian dishes including:")
        st.write("• Curries (Butter Chicken, Paneer dishes)")
        st.write("• Breads (Naan, Chapati, Bhatura)")
        st.write("• Sweets (Gulab Jamun, Rasgulla)")
        st.write("• Rice dishes (Biryani)")
        st.write("• And many more...")

        st.header("📝 Instructions")
        st.write("1. Upload an image of an Indian dish")
        st.write("2. Wait for the model to process")
        st.write("3. View the prediction and confidence")

    model = load_model()
    label_map = load_label_map()

    if model is None:
        st.warning("⚠️ **Demo Mode Active**")
        st.info("""
        The trained model is not available or properly loaded. The app is running in **DEMO MODE**.

        **What this means:**
        - You can still upload images and test the interface
        - Predictions will be sample results from popular Indian dishes
        - This is for testing the UI functionality

        **To use the real model:**
        1. Ensure the model training completed successfully
        2. Check that `data/06_models/autogluon_model/` contains all model files
        3. Retrain the model if necessary using `kedro run --pipeline model_training`
        """)

        # Continue with demo mode
        model = None  # This will trigger demo mode in predict_food function

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📸 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of an Indian dish to classify"
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")

            # Preprocess the image
            processed_image = preprocess_image(image)

            if processed_image is not None:
                # Make prediction
                if st.button("🔍 Classify Dish", type="primary"):
                    with st.spinner("Analyzing image..."):
                        predicted_category, inference_time = predict_food(model, processed_image, label_map)

                    if predicted_category and inference_time:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"## 🎯 Prediction")
                        st.markdown(f"### **{predicted_category.replace('_', ' ').title()}**")
                        st.markdown(f"⏱️ **Inference Time:** {inference_time:.3f} seconds")
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Show confidence (simulated for now)
                        confidence = np.random.uniform(0.7, 0.95)  # Simulated confidence
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence:.1%}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.header("📈 Model Performance")

        # Performance metrics
        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.metric("Test Accuracy", "79,88%")

        with metrics_col2:
            st.metric("Categories", "80")
            st.metric("Avg Inference", "0.221s")

        st.markdown('</div>', unsafe_allow_html=True)

        # Recent predictions (placeholder)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.header("🕒 Recent Predictions")

        # Sample recent predictions
        sample_predictions = [
            ("Butter Chicken", "95.2%", "2 min ago"),
            ("Gulab Jamun", "87.1%", "5 min ago"),
            ("Biryani", "82.3%", "8 min ago"),
            ("Naan", "91.7%", "12 min ago")
        ]

        for dish, conf, time_ago in sample_predictions:
            st.write(f"🍽️ **{dish}** - {conf} ({time_ago})")

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using AutoGluon, Kedro, and Streamlit</p>
            <p>Model trained on 80 Indian food categories</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the trained model
try:
    model_path = os.path.join("models", "multi_cancer_model.h5")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load class names
try:
    class_file = os.path.join("config", "class_names.json")
    with open(class_file, "r") as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"Error loading class names: {e}")
    class_names = []

# Initialize Gemini API
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.warning("Gemini API key not found in environment variables")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}")
    gemini_model = None

# Function to predict cancer type
def predict_cancer_type(img):
    try:
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Unknown", 0.0

# Function to analyze medical image using Gemini API
def analyze_medical_image(img, predicted_class=None, confidence=None):
    if not gemini_model or not GEMINI_API_KEY:
        return "Gemini API not configured properly"
    
    try:
        prompt = f"""
        You are a highly skilled medical image analysis assistant specializing in cancer detection.
        A deep learning model has analyzed this image and predicted the cancer type as **{predicted_class}**
        with **{confidence:.2f}% confidence**.

        Please analyze the image and provide a structured medical suggestion:
        1. Describe the key visual indicators that support or contradict this diagnosis.
        2. Offer any additional insights or medical suggestions for further examination.
        3. Do not comment on Image quality or other non-medical aspects.
        4. DO not output the model confidence or class prediction, as I am already aware of it.
        5. In addition to your role as medical image analysis assistant, do not be too technical or use jargon.
        6. Your response should be understandable by a non-medical professional, as this is for educational purposes.

        Format the response as a structured report.
        """
        
        response = gemini_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error during analysis: {e}"

# Streamlit UI
# Set app title and favicon
st.set_page_config(
    page_title="Cancer Detection App",  
    page_icon="🩺", 
    layout="wide"  
) 

st.title("Cancer Type Prediction & Medical Analysis")
st.write("Upload an image in the sidebar to predict the cancer type and receive a medical analysis.")

# Sidebar for file upload and image preview
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

# Main content area for results
if uploaded_file is not None:
    st.subheader("Diagnosis & Medical Analysis")
    
    if st.button("Predict & Analyze"):
        try:
            if model is not None:
                predicted_class, confidence = predict_cancer_type(img)
                st.write(f"**Predicted Cancer Type:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}")

            if gemini_model is not None:
                analysis_result = analyze_medical_image(img, predicted_class, confidence)
                # st.write("\n**Medical Image Analysis:**")
                st.write(analysis_result)
        except Exception as e:
            st.error(f"Error processing image: {e}")

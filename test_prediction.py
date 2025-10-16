import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyAQpgc1oFLCnyAjjkjuawSB1DHlDvozYsY"
genai.configure(api_key=GEMINI_API_KEY)

# Load the disease detection model
MODEL_PATH = "models/trained_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    print("Model summary:")
    model.summary()  # Print model architecture
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names
CLASS_NAME_PATH = "models/class_name.json"
if not os.path.exists(CLASS_NAME_PATH):
    raise FileNotFoundError(f"Class names file not found at {CLASS_NAME_PATH}.")
with open(CLASS_NAME_PATH, "r") as f:
    class_name = json.load(f)
    print("Class names:", class_name)

# Function to preprocess and predict the image
def model_prediction(test_image):
    try:
        # Load and preprocess the image
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize to match model input size
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch

        # Debug: Print input array shape and values
        print("Input array shape:", input_arr.shape)
        print("Input array min/max:", np.min(input_arr), np.max(input_arr))

        # Make prediction
        prediction = model.predict(input_arr)
        print("Raw prediction output:", prediction)  # Debug: Print raw prediction
        result_index = np.argmax(prediction)
        print("Predicted class index:", result_index)  # Debug: Print predicted index
        return class_name.get(str(result_index), "Unknown Disease")  # Return the disease name
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Routes and other functions remain the same...
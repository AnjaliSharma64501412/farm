import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyAQpgc1oFLCnyAjjkjuawSB1DHlDvozYsY"
genai.configure(api_key=GEMINI_API_KEY)

# ========== Disease Detection Setup ==========
# Disease classes
CLASS_NAMES = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
    "Corn_Cercospora_leaf_spot", "Corn_Common_rust", "Corn_Northern_Leaf_Blight", "Corn_healthy",
    "Grape_Black_rot", "Grape_Esca", "Grape_Leaf_blight", "Grape_healthy",
    "Orange_Haunglongbing", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Tomato_healthy"
]

# Load the disease detection model
MODEL_PATH = "models/crop_disease_model.h5"
IMAGE_SIZE = (256, 256)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Disease detection model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# ========== Soil Moisture Setup ==========
# Load dataset and train model
try:
    soil_df = pd.read_csv("Soil_Moisture_Data.csv")
    # Train Model
    knn = KNeighborsRegressor(n_neighbors=5)
    X = soil_df[["Latitude", "Longitude"]]
    y = soil_df["Soil_Moisture"]
    knn.fit(X, y)
    logger.info("Soil moisture model trained successfully")
except Exception as e:
    logger.error(f"Error setting up soil moisture model: {e}")
    knn = None

# ========== Helper Functions ==========
def model_prediction(test_image_path):
    try:
        # Load and preprocess the image
        img = Image.open(test_image_path)
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        disease_name = CLASS_NAMES[predicted_idx]
        
        return {
            "name": disease_name,
            "confidence": confidence,
            "is_healthy": "healthy" in disease_name.lower()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        chat_history = session.get("chat_history", [])
        
        # Append user's message
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})
        
        # Generate response
        response = model.generate_content(chat_history)
        bot_reply = response.text if response else "Sorry, I couldn't generate a response."
        
        # Update chat history
        chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})
        session["chat_history"] = chat_history
        
        return bot_reply
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"Error: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}

def format_disease_name(name):
    """Convert class name to human-readable format"""
    return name.replace('_', ' ').title()

# ========== Routes ==========
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/chatbot", methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('message')
        response = get_gemini_response(user_message)
        return jsonify({'response': response})
    return render_template('chatbot.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        ai_response = get_gemini_response(user_message)
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"response": "Chat history cleared."})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Invalid file type (PNG, JPG, JPEG, WEBP only)', 'error')
            return redirect(request.url)
        
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)
            
            # Create thumbnail
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            thumbnail_path = os.path.join(uploads_dir, 'thumbnail_' + filename)
            img.save(thumbnail_path)
            
            # Get prediction
            prediction = model_prediction(file_path)
            if "error" in prediction:
                raise Exception(prediction["error"])
            
            # Format results
            disease_name = format_disease_name(prediction["name"])
            confidence_percent = f"{prediction['confidence']*100:.2f}%"
            
            return render_template('disease.html',
                                prediction=disease_name,
                                confidence=confidence_percent,
                                confidence_width=f"{prediction['confidence']*100}%",
                                image_path=filename,
                                thumbnail_url='thumbnail_' + filename,
                                is_healthy=prediction["is_healthy"])
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('disease.html')

@app.route("/soil-moisture")
def soil_moisture_home():
    return render_template("sm.html")

@app.route("/get_soil_moisture", methods=["GET"])
def get_soil_moisture():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and Longitude required"}), 400

    if knn is None:
        return jsonify({"error": "Soil moisture model not available"}), 500

    try:
        moisture = knn.predict([[lat, lon]])[0]
        return jsonify({
            "latitude": lat,
            "longitude": lon,
            "soil_moisture": moisture
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
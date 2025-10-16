import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.neighbors import KNeighborsRegressor
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ========== Disease Detection Setup ==========
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

MODEL_PATH = "models/crop_disease_model.h5"
IMAGE_SIZE = (256, 256)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Disease detection model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load disease detection model: {e}")
else:
    logger.error(f"Model file not found at {MODEL_PATH}")

# ========== Soil Moisture Model Setup ==========
knn = None
try:
    soil_df = pd.read_csv("Soil_Moisture_Data.csv")
    knn = KNeighborsRegressor(n_neighbors=5)
    X = soil_df[["Latitude", "Longitude"]]
    y = soil_df["Soil_Moisture"]
    knn.fit(X, y)
    logger.info("Soil moisture model trained successfully.")
except Exception as e:
    logger.error(f"Error setting up soil moisture model: {e}")

# ========== Helper Functions ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}

def format_disease_name(name: str) -> str:
    return name.replace('_', ' ').title()

def model_prediction(image_path: str) -> dict:
    if model is None:
        return {"error": "Disease detection model is not available"}

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        disease_name = CLASS_NAMES[pred_idx]
        return {
            "name": disease_name,
            "confidence": confidence,
            "is_healthy": "healthy" in disease_name.lower()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

def get_gemini_response(user_input: str) -> str:
    try:
        if 'chat_session' not in session:
            model = genai.GenerativeModel('gemini-pro')
            session['chat_session'] = model.start_chat(history=[])
            session['chat_session'].send_message(
                "You are AgriBot, an agricultural assistant. "
                "Provide concise, accurate answers to farming questions. "
                "Specialize in crops, soil, pests, and farming techniques."
            )
        
        response = session['chat_session'].send_message(user_input)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm having trouble connecting to the knowledge base. Please try again later."

# ========== Routes ==========
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        if not user_message:
            flash('Please enter a message', 'error')
            return redirect(url_for('chatbot'))
        
        response = get_gemini_response(user_message)
        
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'sender': 'user',
            'message': user_message,
            'time': datetime.now().strftime('%I:%M %p')
        })
        session['conversation'].append({
            'sender': 'bot',
            'message': response,
            'time': datetime.now().strftime('%I:%M %p')
        })
        session.modified = True
        
        return redirect(url_for('chatbot'))
    
    if 'conversation' not in session:
        session['conversation'] = [{
            'sender': 'bot',
            'message': "Hello! I'm AgriBot, your agricultural assistant. How can I help you today?",
            'time': datetime.now().strftime('%I:%M %p')
        }]
    
    return render_template('chatbot.html', conversation=session['conversation'])

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session.pop('conversation', None)
    session.pop('chat_session', None)
    flash('Conversation cleared', 'info')
    return redirect(url_for('chatbot'))

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if model is None:
        flash('Disease detection service is currently unavailable.', 'error')
        return render_template('disease.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Allowed types: PNG, JPG, JPEG, WEBP.', 'error')
            return redirect(request.url)

        try:
            filename = secure_filename(file.filename)
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)

            # Create thumbnail
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            thumbnail_path = os.path.join(uploads_dir, f'thumbnail_{filename}')
            img.save(thumbnail_path)

            prediction = model_prediction(file_path)
            if "error" in prediction:
                raise Exception(prediction["error"])

            disease_name = format_disease_name(prediction["name"])
            confidence_percent = f"{prediction['confidence'] * 100:.2f}%"

            return render_template('disease.html',
                                   prediction=disease_name,
                                   confidence=confidence_percent,
                                   confidence_width=f"{prediction['confidence']*100}%",
                                   image_path=filename,
                                   thumbnail_url=f'thumbnail_{filename}',
                                   is_healthy=prediction["is_healthy"])
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('disease.html')

@app.route('/soil-moisture')
def soil_moisture_home():
    return render_template('sm.html')

@app.route('/get_soil_moisture', methods=['GET'])
def get_soil_moisture():
    if knn is None:
        return jsonify({"error": "Soil moisture model not available"}), 500

    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Latitude and Longitude required"}), 400

    try:
        moisture = knn.predict([[lat, lon]])[0]
        return jsonify({"latitude": lat, "longitude": lon, "soil_moisture": moisture})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
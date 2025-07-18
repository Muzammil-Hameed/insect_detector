from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Flask app setup
app = Flask(__name__)

# Constants
MODEL_PATH = 'insect_detector.h5'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = load_model(MODEL_PATH)

# Class labels
class_names = [
    "ants", "bees", "beetle", "catterpillar", "earthworms",
    "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil"
]

# Helper to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_insect(image_path):
    img = load_img(image_path, target_size=(128, 128))  # change to 128x128
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    return predicted_class, confidence


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error="No file found in request.")

    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            label, confidence = predict_insect(filepath)
            return render_template("index.html", filename=filename, label=label, confidence=confidence)
        except Exception as e:
            return render_template("index.html", error=f"Prediction error: {str(e)}")

    return render_template("index.html", error="Unsupported file type. Please upload JPG or PNG.")

if __name__ == "__main__":
    app.run(debug=True)

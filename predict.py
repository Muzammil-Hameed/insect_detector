import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Configurations
MODEL_PATH = 'insect_detector.h5'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = load_model(MODEL_PATH)

# Class labels (model training ke mutabiq)
class_names = [
    "ants", "bees", "beetle", "catterpillar", "earthworms",
    "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil"
]

# Prediction function
def predict_insect(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error="No file part in the request")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error="No file selected")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label, confidence = predict_insect(filepath)
    except Exception as e:
        return render_template("index.html", error=f"Error during prediction: {str(e)}")

    return render_template("index.html", filename=filename, label=label, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)

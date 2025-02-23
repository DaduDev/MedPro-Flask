from flask import Flask, request, render_template, jsonify
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
import json

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('medicine_dataset.csv')

# Load the new inference OCR model
model = load_model('ocr_inference_model.h5')

# Load the character mapping (num_to_char) from JSON
with open('num_to_char.json', 'r') as f:
    num_to_char = json.load(f)
# Convert keys from string to int
num_to_char = {int(k): v for k, v in num_to_char.items()}
NUM_CLASSES = len(num_to_char) + 1  # Blank token is at index NUM_CLASSES - 1

def decode_prediction(pred):
    """
    Decode the prediction using argmax and the character mapping.
    Skips the blank token (assumed to be at index NUM_CLASSES - 1).
    """
    blank_token = NUM_CLASSES - 1
    # Get the predicted indices for the first (and only) batch element
    pred_indices = np.argmax(pred, axis=-1)[0]
    out_str = ''
    for idx in pred_indices:
        if idx != blank_token:
            out_str += num_to_char.get(idx, '')
    return out_str

def process_prescription(image_bytes):
    """
    Preprocess the uploaded image to match the OCR model's input:
    - Decodes the image
    - Converts to grayscale
    - Resizes to (128, 32) [width x height]
    - Normalizes pixel values
    - Adds channel and batch dimensions
    Then performs prediction and decodes the result.
    """
    try:
        # Convert image bytes to a NumPy array and decode as grayscale
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image")

        # Resize image to match model input dimensions (width, height) = (128, 32)
        img = cv2.resize(img, (128, 32))
        # Normalize the image
        img = img.astype("float32") / 255.0
        # Add channel dimension (for grayscale) and batch dimension
        img = np.expand_dims(img, axis=-1)  # shape becomes (32, 128, 1)
        img = np.expand_dims(img, axis=0)    # shape becomes (1, 32, 128, 1)

        # Predict using the OCR model
        y_pred = model.predict(img)
        decoded_text = decode_prediction(y_pred)
        print("Decoded prescription text:", decoded_text)
        return decoded_text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def get_alternatives(medicine_name):
    """
    Searches the dataset for alternatives based on the given medicine name.
    """
    medicine_name = medicine_name.strip().lower()
    medicine_data = dataset[dataset['name'].str.lower().str.contains(medicine_name)]

    if medicine_data.empty:
        return {"substitutes": [], "sideEffects": []}

    substitutes = medicine_data.iloc[0][[f'substitute{i}' for i in range(5)]].dropna().tolist()
    side_effects = medicine_data.iloc[0][[f'sideEffect{i}' for i in range(3)]].dropna().tolist()

    return {"substitutes": substitutes, "sideEffects": side_effects}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_alternatives', methods=['POST'])
def get_alternatives_route():
    try:
        medicine_name = request.form.get('medicine_name', '').strip()
        prescription_file = request.files.get('prescription_image')

        # If an image is uploaded, process it using the OCR model
        if prescription_file:
            image_bytes = prescription_file.read()
            prescription_text = process_prescription(image_bytes)
            prescription_file.seek(0)  # For reusability if needed
            result = get_alternatives(prescription_text)

        # Otherwise, fall back to using the provided text input
        elif medicine_name:
            result = get_alternatives(medicine_name)
        else:
            return jsonify({"error": "Please provide either text or an image"}), 400

        return jsonify(result)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)

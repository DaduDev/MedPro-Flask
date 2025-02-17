from flask import Flask, request, render_template, jsonify
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
import pickle
import imutils

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('medicine_dataset.csv')
model = load_model('ocr.h5')
with open('label_binarizer.pkl', 'rb') as f:
    LB = pickle.load(f)

def get_alternatives(medicine_name):
    medicine_name = medicine_name.strip().lower()
    medicine_data = dataset[dataset['name'].str.lower().str.contains(medicine_name)]

    if medicine_data.empty:
        return {"substitutes": [], "sideEffects": []}

    substitutes = medicine_data.iloc[0][[f'substitute{i}' for i in range(5)]].dropna().tolist()
    side_effects = medicine_data.iloc[0][[f'sideEffect{i}' for i in range(3)]].dropna().tolist()

    return {"substitutes": substitutes, "sideEffects": side_effects}

def process_prescription(image_bytes):
    try:
        letters, _ = get_letters(image_bytes)
        word = get_word(letters)
        print(word)
        return word
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    letters = []
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y:y+h, x:x+w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)
            ypred = model.predict(thresh, verbose=0)
            ypred = LB.inverse_transform(ypred)
            letters.append(ypred[0])

    return letters, image

def get_word(letter):
    return "".join(letter).strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_alternatives', methods=['POST'])
def get_alternatives_route():
    try:
        medicine_name = request.form.get('medicine_name', '').strip()
        prescription_file = request.files.get('prescription_image')

        # Process image if uploaded
        if prescription_file:
            image_bytes = prescription_file.read()
            prescription_text = process_prescription(image_bytes)
            prescription_file.seek(0)  # Important for reusability
            result = get_alternatives(prescription_text)

        # Fallback to text input
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

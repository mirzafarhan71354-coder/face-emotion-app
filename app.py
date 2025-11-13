from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image data from the POST request
        data = request.json['image']
        # Decode base64 image
        image_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Analyze emotion
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result['dominant_emotion']

        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

# This line is important for Render to assign the port dynamically
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image data from frontend
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Analyze emotion using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result['dominant_emotion']

        return jsonify({'emotion': dominant_emotion})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Could not process the image'}), 500

if __name__ == "__main__":
    # For Render, bind to host 0.0.0.0 and port from environment
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


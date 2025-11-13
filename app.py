from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        emotion = "No face detected"

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)

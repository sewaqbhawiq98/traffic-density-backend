import os
import torch
import requests
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/mann123/yolo-traffic/resolve/main/best.pt"
MODEL_PATH = "best.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Model downloaded successfully.")

# Download and load model
download_model()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

@app.route('/')
def home():
    return 'YOLO Traffic Detection API is running!'

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)

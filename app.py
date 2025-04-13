import os
import torch
import requests
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Google Drive download for large file (>100MB) using session & confirm token
def download_from_google_drive(file_id, destination):
    print("📥 Downloading model from Google Drive...")
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print("✅ Model downloaded successfully.")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# MODEL SETUP
MODEL_ID = "1vaBqb6GuGOL9uOpAG7JTRr0UQboSePUB"  # your file ID
MODEL_PATH = "best.pt"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    download_from_google_drive(MODEL_ID, MODEL_PATH)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

@app.route('/')
def home():
    return '🚦 YOLO Traffic Detection API is Running!'

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
    port = int(os.environ.get("PORT", 5000))  # 5000 for local; $PORT for Render
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from firebase_admin import firestore, initialize_app, credentials
import os
import sys
import requests

# Setup path for YOLOv5 local model
sys.path.append("./yolo5")
sys.path.append(os.path.join(os.getcwd(), 'yolo5'))
from yolo5.models.common import DetectMultiBackend
from yolo5.utils.augmentations import letterbox

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase-service-account.json")
initialize_app(cred)
db = firestore.client()

# Ensure weights directory exists
weights_dir = os.path.join(os.getcwd(), "weights")
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, "best.pt")

# ✅ Direct download link from GoFile.io
download_url = "https://store1.gofile.io/download/M6Gq9z/best.pt"

# Download best.pt if not present
if not os.path.exists(model_path):
    print("Downloading best.pt from GoFile.io...")
    try:
        r = requests.get(download_url)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

# Load YOLOv5 model
device = 'cpu'
model = DetectMultiBackend(model_path, device=device)
model.eval()

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Traffic Detection API!"})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/detect', methods=['POST'])
def detect():
    video_file = request.files['video']
    video_path = "input_video.mp4"
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    vehicle_counts = {"CAR": 0, "BUS": 0, "TRUCK": 0, "BIKE": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, new_shape=(640, 640))[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        results = model(img)
        pred = results[0]

        for *box, conf, cls in pred:
            label = model.names[int(cls)].upper()
            if label in vehicle_counts:
                vehicle_counts[label] += 1

    cap.release()

    camera_id = request.form.get('cameraId', '1')
    db.collection('Camera').document(camera_id).set(vehicle_counts, merge=True)

    return jsonify({"message": "Detection complete", "vehicleCounts": vehicle_counts})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

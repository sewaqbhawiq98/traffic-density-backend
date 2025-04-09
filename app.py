from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from firebase_admin import firestore, initialize_app, credentials
import sys
import os


# Setup path for yolov5 local model
sys.path.append("./yolo5")
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
from models.common import DetectMultiBackend  # ← Keep this
# Only if DetectMultiBackend is inside yolo5/models/common.py

from yolov5.utils.augmentations import letterbox  # Updated import for letterbox

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase-service-account.json")
initialize_app(cred)
db = firestore.client()

# Load YOLOv5 model
model_path = 'best.pt'
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

        # Preprocess the frame
        img = letterbox(frame, new_shape=(640, 640))[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Run YOLO detection
        results = model(img)
        pred = results[0]

        for *box, conf, cls in pred:
            label = model.names[int(cls)].upper()
            if label in vehicle_counts:
                vehicle_counts[label] += 1

    cap.release()

    # Save to Firestore
    camera_id = request.form.get('cameraId', '1')
    db.collection('Camera').document(camera_id).set(vehicle_counts, merge=True)

    return jsonify({"message": "Detection complete", "vehicleCounts": vehicle_counts})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

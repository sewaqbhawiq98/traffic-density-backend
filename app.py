from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from firebase_admin import firestore, initialize_app, credentials
import urllib.request
import os
import sys
import gdown  
# Setup path for yolov5 local model
sys.path.append("./yolo5")
sys.path.append(os.path.join(os.getcwd(), 'yolo5'))
from yolo5.models.common import DetectMultiBackend
from yolo5.utils.augmentations import letterbox  # Updated import for letterbox

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase-service-account.json")
initialize_app(cred)
db = firestore.client()


# Download the model if it doesn't exist
model_url = 'https://drive.google.com/uc?id=1vaBqb6GuGOL9uOpAG7JTRr0UQboSePUB'  # your real file ID
model_path = 'best.pt'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Download best.pt from Google Drive if not present
weights_dir = os.path.join(os.getcwd(), "weights")
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, "best.pt")

# Replace this with your actual Google Drive file ID
file_id = "1vaBqb6GuGOL9uOpAG7JTRr0UQboSePUB"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(model_path):
    print("Downloading best.pt from Google Drive...")
    try:
        urllib.request.urlretrieve(download_url, model_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model weights: {e}")
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
  port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT isn't set
app.run(host='0.0.0.0', port=port)

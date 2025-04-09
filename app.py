import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore

from utils import letterbox  # Make sure this exists or define it here

# Init Firebase
cred = credentials.Certificate("serviceAccountKey.json")  # Put your key in this file
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
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
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float() / 255.0
            img = img.unsqueeze(0)

            results = model(img)[0]

            for *box, conf, cls in results:
                label = model.names[int(cls)].upper()
                if label in vehicle_counts:
                    vehicle_counts[label] += 1

        cap.release()

        camera_id = request.form.get('cameraId', '1')
        db.collection('Camera').document(camera_id).set(vehicle_counts, merge=True)

        return jsonify({"status": "success", "message": "Detection complete", "vehicleCounts": vehicle_counts})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

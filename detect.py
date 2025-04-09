import requests
import cv2
import torch
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Save the uploaded video file
        video_file = request.files['video']
        video_path = "input_video.mp4"
        video_file.save(video_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        vehicle_counts = {"CAR": 0, "BUS": 0, "TRUCK": 0, "BIKE": 0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            img = letterbox(frame, new_shape=(640, 640))[0]  # Resize with padding
            img = img.transpose((2, 0, 1))  # Convert HWC to CHW
            img = np.ascontiguousarray(img)  # Ensure memory is contiguous
            img = torch.from_numpy(img).float() / 255.0  # Normalize to [0, 1]
            img = img.unsqueeze(0)  # Add batch dimension

            # Run YOLO detection
            results = model(img)
            pred = results[0]

            # Parse predictions
            for *box, conf, cls in pred:
                label = model.names[int(cls)].upper()
                if label in vehicle_counts:
                    vehicle_counts[label] += 1

        cap.release()

        # Save results to Firestore
        camera_id = request.form.get('cameraId', '1')
        db.collection('Camera').document(camera_id).set(vehicle_counts, merge=True)

        return jsonify({"status": "success", "message": "Detection complete", "vehicleCounts": vehicle_counts})

    except Exception as e:
        # Handle errors and return the error message in the response
        print("Detection Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

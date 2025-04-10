from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import requests
import torch
from yolov5.models.common import DetectMultiBackend
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import sys
sys.path.insert(0, './yolov5')
from models.common import DetectMultiBackend  # or similar

app = FastAPI()

# Constants
GOFILE_FILE_ID = "263e24c3-f096-46d0-a65c-ba27fdd03652"
MODEL_PATH = "models/best.pt"
os.makedirs("models", exist_ok=True)

# Function to get GoFile direct download link
def download_model_from_gofile(file_id, save_path):
    print("[INFO] Fetching GoFile direct download link...")
    res = requests.get(f"https://api.gofile.io/getContent?contentId={file_id}&format=json")
    download_url = res.json()["data"]["contents"]["best.pt"]["link"]
    print(f"[INFO] Downloading model from: {download_url}")
    r = requests.get(download_url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    print("[INFO] Model download complete.")

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    download_model_from_gofile(GOFILE_FILE_ID, MODEL_PATH)

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
print("[INFO] Model loaded successfully.")

@app.get("/")
async def home():
    return {"message": "YOLOv5 API is running 🎯"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    img = np.array(image)
    results = model(img, size=640)
    boxes = results.pandas().xyxy[0].to_dict(orient="records")
    return {"detections": boxes}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import torch
from yolov5.models.common import DetectMultiBackend
import requests

app = FastAPI()

MODEL_URL = "https://your-file-hosting-link.com/best.pt"  # Use direct GoFile/Google Drive raw link
MODEL_PATH = "models/best.pt"

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
print("Model loaded.")

@app.get("/")
async def root():
    return {"status": "YOLO API is running"}

# You can add more endpoints for inference here

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

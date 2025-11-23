from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Auto-downloads if not present

@app.route("/")
def home():
    return {"message": "Object Detection API is running!"}

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Convert image bytes → numpy → cv2 format
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # YOLO detection
    results = model(frame)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            detections.append({
                "object": name,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(debug=True)

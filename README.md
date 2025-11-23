# Object Detection API (YOLO + Flask)

This API accepts an image and returns detected objects with bounding boxes.

## How to use
POST /detect
Upload file with name "image"

Example:
curl -X POST -F "image=@photo.jpg" https://your-app.onrender.com/detect

## Deploy to Render
1. Upload to GitHub
2. Go to Render → New Web Service
3. Start Command:
   gunicorn app:app

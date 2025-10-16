from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from frontend/backend

# Load OpenCV face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/detect", methods=["POST"])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({"message": "No image uploaded"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_list = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]

    return jsonify({
        "faceCount": len(faces),
        "faces": face_list
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

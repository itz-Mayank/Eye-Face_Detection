from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import onnxruntime as ort

# Define the preprocess_face function (same as saved in pipeline)
def preprocess_face(face_img):
    """Preprocess the face for ONNX model input."""
    face_resized = cv2.resize(face_img, (224, 224))  # Resize
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    face_normalized = face_rgb / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(face_normalized, axis=0).astype(np.float32)

# Load the saved pipeline (without function)
pipeline = joblib.load("model_pipeline.pkl")

# Extract components from the pipeline
onnx_model_path = pipeline['onnx_model_path']
input_name = pipeline['input_name']
output_name = pipeline['output_name']

# Load the ONNX model session
onnx_session = ort.InferenceSession(onnx_model_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Read and preprocess the image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    face_input = preprocess_face(image)

    # Run inference
    prediction = onnx_session.run([output_name], {input_name: face_input})[0][0]
    confidence = prediction.item()

    # Determine the label
    label = "Real" if confidence > 0.90 else "Spoof"
    confidence_percentage = confidence * 100 if label == "Real" else (1 - confidence) * 100

    return jsonify({"label": label, "confidence": confidence_percentage})

if __name__ == "__main__":
    app.run(debug=True)

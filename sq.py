import random
import cv2
import numpy as np
import onnxruntime as ort
import time
from threading import Thread
from gaze_tracking.gaze_tracking import GazeTracking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize GazeTracking
gaze = GazeTracking()

# Load ONNX model
onnx_session = ort.InferenceSession("model.onnx")
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Screen dimensions and dot parameters
screen_width, screen_height = 640, 480  # Reduced resolution for performance
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
current_dot_index = 0
dot_display_time = time.time()
dot_trace_success = 0

# Blink detection and gaze parameters
gaze_distance_threshold = 0.5
dot_display_interval = 3
blink_delay = 1
last_blink_time = time.time()

# Asynchronous frame grabbing
class FrameGrabber:
    def __init__(self, capture_device):
        self.capture_device = capture_device
        self.frame = None
        self.running = True
        self.thread = Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def update_frame(self):
        while self.running:
            ret, frame = self.capture_device.read()
            if ret:
                self.frame = cv2.resize(frame, (screen_width, screen_height))  # Resize for performance

    def get_frame(self):
        return self.frame

frame_grabber = FrameGrabber(cap)

while True:
    frame = frame_grabber.get_frame()
    if frame is None:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    annotated_frame = frame.copy()

    # Gaze tracking
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    # Process faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = gaze.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))  # Small faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))  # Small faces

    for (x, y, w, h) in faces:
        # Extract and preprocess face region
        face_img = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        face_input = np.expand_dims(face_rgb / 255.0, axis=0)

        # ONNX inference
        prediction = onnx_session.run([output_name], {input_name: face_input.astype(np.float32)})[0][0]
        liveness_label = "Real" if prediction > 0.75 else "Spoof"
        liveness_confidence = prediction if prediction > 0.75 else 1 - prediction
            # Convert liveness_confidence to scalar if it's a NumPy array
        if isinstance(liveness_confidence, np.ndarray):
            liveness_confidence = liveness_confidence.item()

        label_color = (0, 255, 0) if liveness_label == "Real" else (0, 0, 255)
        cv2.putText(annotated_frame, f"{liveness_label} ({liveness_confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # Dot tracing logic
        if current_dot_index < len(dots):
            dot_x, dot_y = dots[current_dot_index]
            cv2.circle(annotated_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)

            if gaze.is_center() and gaze.is_blinking() and time.time() - last_blink_time > blink_delay:
                if liveness_label == "Real":
                    dot_trace_success += 1
                    current_dot_index += 1
                    dot_display_time = time.time()
                    last_blink_time = time.time()

    # Status display
    if current_dot_index < len(dots):
        cv2.putText(annotated_frame, f"Trace the dot and blink! ({dot_trace_success}/{len(dots)})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(annotated_frame, "You are Damn REAL!", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Dot Tracing with Liveness Detection", annotated_frame)

    # Break loop on 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
frame_grabber.running = False
frame_grabber.thread.join()
cap.release()
cv2.destroyAllWindows()

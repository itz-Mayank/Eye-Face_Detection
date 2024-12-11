# PIPELINE LAGANE KE BAAD SAHI SE KAAM NHI KAR RHA HAI

import cv2
import numpy as np
import random
import time
import joblib
from gaze_tracking.gaze_tracking import GazeTracking
import onnxruntime as ort

# Constants
FRAME_SKIP_INITIAL = 3
MAX_PUPIL_DETECTION_FRAMES = 5
FACE_RESIZE_DIM = (224, 224)
MODEL_THRESHOLD = 0.5
REAL_COLOR = (0, 255, 0)  # Green
SPOOF_COLOR = (0, 0, 255)  # Red
DOT_DISPLAY_INTERVAL = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 480


class LivenessDetectionPipeline:
    """Pipeline for processing frames, performing liveness detection, and managing dot tracing."""

    def __init__(self, model_path, gaze_tracker):
        self.gaze_tracker = gaze_tracker
        self.model_path = model_path
        self.onnx_session = None  # Initialize as None
        self.input_name = None
        self.output_name = None

        # State variables for liveness detection
        self.pupil_detection_count = 0
        self.blink_count = 0
        self.is_eye_closed = False
        self.total_frames = 0
        self.real_count = 0
        self.spoof_count = 0
        self.frame_skip = FRAME_SKIP_INITIAL

        # State variables for dot tracing
        self.dots = [(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)) for _ in range(3)]
        self.current_dot_index = 0
        self.dot_display_time = time.time()
        self.dot_trace_success = 0

        # Initialize non-serializable objects
        self._init_inference_session()
        self._init_face_cascade()

    def _init_inference_session(self):
        """Initialize the ONNX runtime inference session."""
        self.onnx_session = ort.InferenceSession(self.model_path)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def _init_face_cascade(self):
        """Initialize the face cascade (excluded from serialization)."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_face(self, face_img):
        """Preprocess the face image for ONNX model."""
        face_resized = cv2.resize(face_img, FACE_RESIZE_DIM)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        face_normalized = face_rgb / 255.0
        return np.expand_dims(face_normalized, axis=0).astype(np.float32)

    def run_model(self, face_input):
        """Run ONNX model to predict liveness."""
        prediction = self.onnx_session.run([self.output_name], {self.input_name: face_input})[0][0]
        confidence = prediction.item()  # Ensure scalar
        if confidence > MODEL_THRESHOLD:
            return "Real", confidence * 100
        return "Spoof", (1 - confidence) * 100

    def handle_dot_tracing(self, annotated_frame):
        """Handle dot tracing logic."""
        if self.current_dot_index < len(self.dots):
            dot_x, dot_y = self.dots[self.current_dot_index]
            cv2.circle(annotated_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)  # Draw the dot

            # Check gaze and blink for validation
            if self.gaze_tracker.is_center() and self.gaze_tracker.is_blinking():
                self.dot_trace_success += 1
                self.current_dot_index += 1
                self.dot_display_time = time.time()

        if self.current_dot_index < len(self.dots):
            cv2.putText(annotated_frame, f"Trace the dot and blink! ({self.dot_trace_success}/{len(self.dots)})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "Dot Tracing Complete!", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    def detect_liveness_and_annotate(self, frame):
        """Process the frame, detect liveness, and annotate the frame."""
        self.total_frames += 1

        # Flip and resize for consistency
        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Refresh gaze tracking and annotate frame
        self.gaze_tracker.refresh(resized_frame)
        annotated_frame = self.gaze_tracker.annotated_frame()
        left_pupil = self.gaze_tracker.pupil_left_coords()
        right_pupil = self.gaze_tracker.pupil_right_coords()

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        liveness_label = "Spoof"
        liveness_confidence = 0.0

        for (x, y, w, h) in faces:
            # Extract face and preprocess
            face_img = gray[y:y + h, x:x + w]
            face_input = self.preprocess_face(face_img)

            # Run model periodically based on frame skip
            if self.total_frames % self.frame_skip == 0:
                liveness_label, liveness_confidence = self.run_model(face_input)

            # Draw face rectangle and label
            color = REAL_COLOR if liveness_label == "Real" else SPOOF_COLOR
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 4)
            cv2.putText(annotated_frame, f"{liveness_label} ({liveness_confidence:.1f}%)", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Handle dot tracing logic
        self.handle_dot_tracing(annotated_frame)

        # Update counts
        if liveness_label == "Real":
            self.real_count += 1
        else:
            self.spoof_count += 1

        return annotated_frame

    def __getstate__(self):
        """Exclude non-serializable objects from being pickled."""
        state = self.__dict__.copy()
        state["onnx_session"] = None  # Exclude ONNX session
        state["face_cascade"] = None  # Exclude face cascade
        return state

    def __setstate__(self, state):
        """Restore non-serializable objects after unpickling."""
        self.__dict__.update(state)
        self._init_inference_session()  # Reinitialize ONNX session
        self._init_face_cascade()  # Reinitialize face cascade

    def print_results(self):
        """Print final detection results."""
        real_percentage = (self.real_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        spoof_percentage = (self.spoof_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        print(f"\nReal Faces Detected: {self.real_count} ({real_percentage:.1f}%)")
        print(f"Spoof Faces Detected: {self.spoof_count} ({spoof_percentage:.1f}%)")


# Save and Load Pipeline
def save_pipeline(pipeline, filename="pipeline.joblib"):
    """Save the pipeline to a file."""
    joblib.dump(pipeline, filename)
    print(f"Pipeline saved to {filename}")


def load_pipeline(filename="pipeline.joblib"):
    """Load the pipeline from a file."""
    pipeline = joblib.load(filename)
    return pipeline


# Main loop
def main():
    pipeline = LivenessDetectionPipeline(
        "model.onnx",
        GazeTracking()
    )
    save_pipeline(pipeline)  # Save the pipeline for later use

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Process frame and display output
        annotated_frame = pipeline.detect_liveness_and_annotate(frame)
        cv2.imshow("Real vs. Spoof Detection with Dot Tracing", annotated_frame)

        # Exit on 'Esc'
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.print_results()


if __name__ == "__main__":
    main()

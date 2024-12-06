import cv2
import numpy as np
from gaze_tracking.gaze_tracking import GazeTracking
import onnxruntime as ort

# Constants
FRAME_SKIP_INITIAL = 3  # Initial frame skip count
MAX_PUPIL_DETECTION_FRAMES = 5
FACE_RESIZE_DIM = (224, 224)
MODEL_THRESHOLD = 0.5  # Liveness threshold
REAL_COLOR = (0, 255, 0)  # Green for "Real"
SPOOF_COLOR = (0, 0, 255)  # Red for "Spoof"

# Initialize GazeTracking for eye tracking
gaze = GazeTracking()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the ONNX model and initialize ONNX Runtime
onnx_session = ort.InferenceSession("model.onnx")
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables for tracking
frame_skip = FRAME_SKIP_INITIAL
pupil_detection_count = 0
blink_count = 0
is_eye_closed = False
total_frames = 0
real_count = 0
spoof_count = 0


def preprocess_face(face_img):
    """Preprocess the face for ONNX model input."""
    face_resized = cv2.resize(face_img, FACE_RESIZE_DIM)  # Resize
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    face_normalized = face_rgb / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(face_normalized, axis=0).astype(np.float32)


def run_model(face_input):
    """Run the ONNX model on the preprocessed face."""
    prediction = onnx_session.run([output_name], {input_name: face_input})[0][0]
    confidence = prediction.item()  # Ensure scalar
    if confidence > MODEL_THRESHOLD:
        return "Real", confidence * 100
    return "Spoof", (1 - confidence) * 100


def process_frame(frame, total_frames):
    """Process a single frame and return annotations and labels."""
    global pupil_detection_count, blink_count, is_eye_closed

    # Resize frame for faster processing
    frame = cv2.flip(frame, 1)
    # frame_resized = cv2.resize(frame, (640, 480))
    frame_resized = cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Analyze gaze and pupils
    gaze.refresh(frame_resized)
    annotated_frame = gaze.annotated_frame()
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    liveness_label = "Spoof"
    liveness_confidence = 0.0

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = gray[y:y + h, x:x + w]
        face_input = preprocess_face(face_img)

        # Run model every N frames
        if total_frames % frame_skip == 0:
            liveness_label, liveness_confidence = run_model(face_input)

        # Pupil and blink logic
        if left_pupil and right_pupil:
            pupil_detection_count += 1
            if pupil_detection_count >= MAX_PUPIL_DETECTION_FRAMES:
                liveness_label, liveness_confidence = "Real", 100.0

            # Detect blinks
            if gaze.is_blinking():
                if not is_eye_closed:
                    blink_count += 1
                    is_eye_closed = True
            else:
                is_eye_closed = False
        else:
            pupil_detection_count = 0

        # Draw rectangle and label
        frame_color = REAL_COLOR if liveness_label == "Real" else SPOOF_COLOR
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), frame_color, 4)
        cv2.putText(annotated_frame, f"{liveness_label} ({liveness_confidence:.1f}%)", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)

    # Annotate pupils
    if left_pupil and right_pupil:
        cv2.putText(annotated_frame, f"Left pupil: {left_pupil}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Right pupil: {right_pupil}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(annotated_frame, "Pupils not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated_frame, liveness_label


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    total_frames += 1

    # Skip frames dynamically based on activity
    if frame_skip > 1 and gaze.is_blinking():
        frame_skip = 1  # Reduce skip if blinking detected
    elif frame_skip > 1 and pupil_detection_count > 0:
        frame_skip = 2  # Moderate skip if pupils detected
    else:
        frame_skip = FRAME_SKIP_INITIAL  # Reset skip if no activity

    # Process the frame
    annotated_frame, liveness_label = process_frame(frame, total_frames)

    # Update counts
    if liveness_label == "Real":
        real_count += 1
    else:
        spoof_count += 1

    # Display the frame
    cv2.imshow("Real vs. Spoof Detection with Eye Movement", annotated_frame)

    # Exit on 'Esc'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print results
real_percentage = (real_count / total_frames) * 100 if total_frames > 0 else 0
spoof_percentage = (spoof_count / total_frames) * 100 if total_frames > 0 else 0
print(f"\nReal Faces Detected: {real_count} ({real_percentage:.1f}%)")
print(f"Spoof Faces Detected: {spoof_count} ({spoof_percentage:.1f}%)")

import random
import cv2
import numpy as np
from gaze_tracking.gaze_tracking import GazeTracking
import onnxruntime as ort
import time

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

# Variables for dot tracing
screen_width, screen_height = 640, 480
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
current_dot_index = 0
dot_trace_threshold = 15  # Number of frames the gaze must stay on a dot to consider it traced
dot_trace_count = 0
dot_trace_success = 0
dot_timer = time.time()  # Timer for the 3-second interval

# Blink detection variables
blink_count = 0
is_eye_closed = False

# Matrices to store spoof and real data
real_data_matrix = []
spoof_data_matrix = []

# Frame skip for performance
frame_skip = 5
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize the frame to a smaller resolution for faster processing
    frame = cv2.flip(frame, 1) 
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Skip frames for performance
    if total_frames % frame_skip != 0:
        total_frames += 1
        continue

    # Send the frame to GazeTracking for eye analysis
    gaze.refresh(frame_resized)
    annotated_frame = gaze.annotated_frame()

    # Get pupil coordinates
    current_left_pupil = gaze.pupil_left_coords()
    current_right_pupil = gaze.pupil_right_coords()

    # Check if the current dot is active and within the 3-second display time
    if current_dot_index < len(dots):
        dot_x, dot_y = dots[current_dot_index]

        # Draw the current dot in red
        cv2.circle(annotated_frame, (dot_x, dot_y), 10, (0, 0, 255), -1)

        # Calculate average gaze position if pupils are detected
        if current_left_pupil and current_right_pupil:
            avg_gaze_x = (current_left_pupil[0] + current_right_pupil[0]) / 2
            avg_gaze_y = (current_left_pupil[1] + current_right_pupil[1]) / 2

            # Check if gaze is near the current dot
            if abs(avg_gaze_x - dot_x) < 50 and abs(avg_gaze_y - dot_y) < 50:
                dot_trace_count += 1
                if dot_trace_count >= dot_trace_threshold:
                    # Check for a blink
                    if gaze.is_blinking():
                        blink_count += 1
                        dot_trace_success += 1
                        current_dot_index += 1  # Move to the next dot
                        dot_timer = time.time()  # Reset the timer
                        dot_trace_count = 0  # Reset trace count
            else:
                dot_trace_count = 0  # Reset trace count if gaze moves away

    # If all dots are traced
    if current_dot_index >= len(dots):
        liveness_label = "Real"
        liveness_confidence = 100.0
    else:
        liveness_label = "Spoof"
        liveness_confidence = 0.0

    # Store data in the correct matrix
    if liveness_label == "Real":
        real_data_matrix.append([liveness_label, liveness_confidence])
    else:
        spoof_data_matrix.append([liveness_label, liveness_confidence])

    # Draw rectangle around detected face (if applicable)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        frame_color = (0, 255, 0) if liveness_label == "Real" else (0, 0, 255)
        text_color = frame_color

        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), frame_color, 4)
        cv2.putText(annotated_frame, f"{liveness_label} ({liveness_confidence:.1f}%)", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Display the frame
    cv2.imshow("Real vs. Spoof Detection with Dot Tracing", annotated_frame)

    # Break loop on pressing 'Esc' key
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Update total frames count
    total_frames += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

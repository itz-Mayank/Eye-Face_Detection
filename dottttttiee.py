import random
import cv2
import numpy as np
from gaze_tracking.gaze_tracking import GazeTracking
import time

# Initialize GazeTracking
gaze = GazeTracking()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Screen dimensions and dot parameters
screen_width, screen_height = 640, 480
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
current_dot_index = 0
dot_display_time = time.time()
dot_trace_success = 0

# Gaze detection parameters
gaze_distance_threshold = 0.5  # Gaze direction confidence threshold
dot_display_interval = 3  # Time interval (in seconds) for displaying the next dot
blink_threshold = 5  # Number of frames to detect a blink
last_blink_time = time.time()
blink_delay = 1  # Minimum time between valid blinks

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to smartphone-compatible size
    annotated_frame = frame_resized.copy()

    # Update gaze tracking
    gaze.refresh(frame_resized)
    annotated_frame = gaze.annotated_frame()

    # Draw the current dot
    if current_dot_index < len(dots):
        dot_x, dot_y = dots[current_dot_index]
        cv2.circle(annotated_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)  # Red dot

        # Check if enough time has passed for current dot display
        if time.time() - dot_display_time >= dot_display_interval:
            # Check gaze direction relative to dot position
            if gaze.is_center():  # Assume user is looking at the dot if gaze is "center"
                # Detect blink to confirm tracing
                if gaze.is_blinking() and time.time() - last_blink_time > blink_delay:
                    dot_trace_success += 1
                    current_dot_index += 1
                    dot_display_time = time.time()  # Reset timer for the next dot
                    last_blink_time = time.time()  # Reset blink detection time

    # Display task status
    if current_dot_index < len(dots):
        cv2.putText(annotated_frame, f"Trace the dot and blink! ({dot_trace_success}/{len(dots)})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(annotated_frame, "You are Damn REAL!", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display annotated frame
    cv2.imshow("Static Dot Tracing with Blink Detection", annotated_frame)

    # Break loop on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

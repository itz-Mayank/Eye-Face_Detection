import cv2
import numpy as np
import time

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video feed from webcam
cap = cv2.VideoCapture(0)

# Parameters for liveness detection
motion_threshold = 3  # Minimum motion difference threshold
movement_detected = False

print("Starting camera... Please stay still.")

# Initialize previous frame for motion detection
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

start_time = time.time()
while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (face area)
        face_roi = gray_frame[y:y + h, x:x + w]
        prev_face_roi = prev_frame_gray[y:y + h, x:x + w]

        # Compare with the previous frame to detect motion
        diff = cv2.absdiff(face_roi, prev_face_roi)
        motion_score = np.sum(diff) / (w * h)  # Average motion per pixel

        # Check if motion exceeds the threshold
        if motion_score > motion_threshold:
            movement_detected = True

    # Display live or fake based on motion detection
    if movement_detected:
        cv2.putText(frame, "LIVE: Real Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "FAKE: Photo or Video Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Face Liveness Detection", frame)

    # Update the previous frame for comparison
    prev_frame_gray = gray_frame

    # Reset detection every few seconds to avoid false positives
    if time.time() - start_time > 2:
        movement_detected = False
        start_time = time.time()

    # Break on pressing ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
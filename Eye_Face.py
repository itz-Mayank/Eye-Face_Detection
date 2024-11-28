import cv2
import numpy as np
from gaze_tracking import GazeTracking
import tensorflow as tf

# Initialize GazeTracking for eye tracking
gaze = GazeTracking()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained liveness model (ensure you have a trained model file)
model = tf.keras.models.load_model("face_eye_liveness_model_lccfasd.h5")

# Initialize webcam (ensure webcam starts after model load)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables to track previous eye coordinates
previous_left_pupil = None
previous_right_pupil = None
pupil_detection_count = 0  # Counter to track how many frames have pupils detected
max_pupil_detection_frames = 5  # Number of frames pupils must be detected to be considered real

# Blink detection variables
blink_count = 0
is_eye_closed = False

# Matrices to store spoof and real data
real_data_matrix = []
spoof_data_matrix = []

# Tracking the total frames processed
total_frames = 0
real_count = 0
spoof_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Send the frame to GazeTracking for eye analysis
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    # Get pupil coordinates
    current_left_pupil = gaze.pupil_left_coords()
    current_right_pupil = gaze.pupil_right_coords()

    # Initialize the liveness detection variables
    liveness_label = "Spoof"
    liveness_confidence = 0.0

    # Check for faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (224, 224))  # Resize to model input size

        # Convert grayscale to RGB for the model
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

        # Normalize pixel values to [0, 1]
        face_normalized = face_rgb / 255.0

        # Add batch dimension
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict liveness
        prediction = model.predict(face_input)[0][0]  # Prediction should be a scalar, not an array

        # Ensure liveness_label and liveness_confidence are scalar and properly formatted
        if prediction > 0.5:
            liveness_label = "Real"
            liveness_confidence = (prediction * 100).item()  # Convert to percentage and ensure scalar
        else:
            liveness_label = "Spoof"
            liveness_confidence = ((1 - prediction) * 100).item()  # Convert to percentage and ensure scalar


        # Check pupil detection and track movement
        if current_left_pupil and current_right_pupil:
            # If both pupils are detected, classify as real
            pupil_detection_count += 1
            if pupil_detection_count >= max_pupil_detection_frames:
                liveness_label = "Real"
                liveness_confidence = 100.0

            # Blink detection (if eye closed for a frame)
            if gaze.is_blinking():
                if not is_eye_closed:  # Avoid counting the same blink multiple times
                    blink_count += 1
                    is_eye_closed = True
            else:
                is_eye_closed = False

            # Print the pupil coordinates and status in terminal
            print(f"Left pupil: {current_left_pupil}, Right pupil: {current_right_pupil}, {liveness_label}, Blink count: {blink_count}")
        else:
            # If no pupils detected, classify as spoof
            pupil_detection_count = 0
            liveness_label = "Spoof"
            liveness_confidence = 0.0

            # Print the pupil detection status
            print("Pupils not detected")

        # Store data in the correct matrix
        if liveness_label == "Real":
            real_data_matrix.append([liveness_label, liveness_confidence])
            real_count += 1
        else:
            spoof_data_matrix.append([liveness_label, liveness_confidence])
            spoof_count += 1

        # Draw rectangle around the face
        frame_color = (0, 255, 0) if liveness_label == "Real" else (0, 0, 255)  # Green for real, red for spoof
        text_color = frame_color

        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), frame_color, 4)

        # Display the label and confidence
        cv2.putText(annotated_frame, f"{liveness_label} ({liveness_confidence:.1f}%)", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Display pupil coordinates and status
    if current_left_pupil and current_right_pupil:
        cv2.putText(annotated_frame, f"Left pupil: {current_left_pupil}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Right pupil: {current_right_pupil}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(annotated_frame, "Pupils not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the frame with predictions
    cv2.imshow("Real vs. Spoof Detection with Eye Movement", annotated_frame)

    # Break loop on pressing 'Esc' key
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Update total frames count
    total_frames += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate and print real and spoof data matrices
real_percentage = (real_count / total_frames) * 100 if total_frames > 0 else 0
spoof_percentage = (spoof_count / total_frames) * 100 if total_frames > 0 else 0

print("\nReal Data Matrix (Overall Performance):")
print(f"Real: {real_count} ({real_percentage:.1f}%)")

print("\nSpoof Data Matrix (Pupils Not Detected):")
print(f"Spoof: {spoof_count} ({spoof_percentage:.1f}%)")

# Calculate and print overall accuracy
overall_accuracy = ((real_count + spoof_count) / total_frames) * 100 if total_frames > 0 else 0
overall_accuracy = min(100.0, overall_accuracy)  # Ensure accuracy doesn't exceed 100%
print(f"\nOverall Detection Accuracy: {overall_accuracy:.1f}%")

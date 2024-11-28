import cv2
import numpy as np
import tensorflow as tf

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if the cascades are loaded successfully
if face_cascade.empty():
    print("Error loading face cascade!")
    exit()
if eye_cascade.empty():
    print("Error loading eye cascade!")
    exit()

# Load the trained liveness model
model = tf.keras.models.load_model("face_eye_liveness_model.h5")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert frame to grayscale for Haar cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (128, 128))  # Resize to model input size

        # Convert grayscale to RGB for the model
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

        # Normalize pixel values to [0, 1]
        face_normalized = face_rgb / 255.0

        # Add batch dimension
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict liveness
        prediction = model.predict(face_input)[0][0]
        label = "Real" if prediction > 0.5 else "Spoof"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        # Convert confidence to a float for f-string formatting
        confidence = float(confidence)

        # Set frame and text color based on prediction
        face_color = (0, 255, 0) if label == "Real" else (0, 0, 255)  # Green for real, red for spoof
        text_color = face_color

        # Draw rectangle around the face with increased width
        cv2.rectangle(img, (x, y), (x + w, y + h), face_color, 4)
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Detect eyes within the detected face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (ex, ey, ew, eh) in eyes:
            # Set eye frame color based on prediction
            eye_color = (0, 255, 0) if label == "Real" else (0, 0, 255)

            # Draw rectangle around each eye with smaller frame width
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), eye_color, 2)

            # Approximate confidence for the eye
            eye_area = ew * eh
            face_area = w * h
            eye_confidence = (eye_area / face_area) * 100

            # Convert eye confidence to float for f-string formatting
            eye_confidence = float(eye_confidence)

            # Display "Eye" with confidence percentage in smaller text size
            cv2.putText(roi_color, f"Eye ({eye_confidence:.1f}%)", (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)

    # Display the frame with predictions and tracking
    cv2.imshow("Live Prediction with Face and Eye Tracking", img)

    # Break loop on pressing 'Esc' key
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

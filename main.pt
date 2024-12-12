import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the trained liveness model
model = tf.keras.models.load_model("face_eye_liveness_model.h5")

# Speech recognizer setup
recognizer = sr.Recognizer()

# Authentication variables
attempts = 0
authenticated = False
audio_words = []  # Stores unique words spoken by the user for authentication
final_result = "Authentication Incomplete"
blocked_reason = None

# Tracking metrics
predictions = []
ground_truth = []
eye_confidences = []
face_confidences = []

# Function to calculate and print metrics
def calculate_metrics():
    if predictions and ground_truth:
        acc = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)
        print(f"\nMetrics Summary:")
        print(f"  - Accuracy: {acc:.2f}")
        print(f"  - F1 Score: {f1:.2f}")
        print(f"  - Precision: {precision:.2f}")
        print(f"  - Recall: {recall:.2f}")
    else:
        print("\nNo predictions made during the session.")

    if eye_confidences:
        print(f"Average Eye Confidence: {np.mean(eye_confidences):.2f}%")
    if face_confidences:
        print(f"Average Face Confidence: {np.mean(face_confidences):.2f}%")

# Function to perform eye tracking
def perform_eye_tracking(frame, roi_gray, roi_color, eyes, face_area):
    if len(eyes) == 0:
        print("No eyes detected in the current frame.")
        return

    directions = ["Look Left", "Look Right", "Look Up", "Look Down"]
    for direction in directions:
        for (ex, ey, ew, eh) in eyes:
            # Calculate eye confidence based on size relative to face area
            eye_area = ew * eh
            eye_confidence = (eye_area / face_area) * 100
            eye_confidences.append(eye_confidence)

            # Draw rectangle around the eye
            eye_color = (0, 255, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), eye_color, 2)

        # Display direction prompt
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Eye Tracking Task", frame)
        time.sleep(2)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit if Esc is pressed
            break

# Function to perform audio authentication
def perform_audio_authentication():
    global attempts, authenticated, audio_words, final_result, blocked_reason

    print("Listening for verbal response...")
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for ambient noise
            audio = recognizer.listen(source, timeout=5)  # Listen with 10-second timeout
            speech_text = recognizer.recognize_google(audio).lower()
            print(f"User said: {speech_text}")

            # Add word to unique word list for validation
            if speech_text not in audio_words:
                audio_words.append(speech_text)

            # Check if three unique words have been spoken
            if len(audio_words) >= 3:
                authenticated = True
                final_result = "Authenticated"
                print("Audio Authentication Successful!")
                return True
            else:
                print(f"Current words: {audio_words} (Need 3 unique words)")
        except sr.WaitTimeoutError:
            print("No response detected. Please try again.")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Speech recognition service is unavailable.")

        # Increment failed attempts
        attempts += 1
        if attempts >= 3:
            final_result = "Access Blocked"
            blocked_reason = "Audio not recognized 3 times."
            print("\nAccess Blocked! Reason: Audio not recognized 3 times.")
            return False
    return True

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

    # Convert the frame to grayscale for Haar cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region and preprocess for the model
        face_img = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (64, 64))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        face_normalized = face_rgb / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict liveness
        prediction = model.predict(face_input)[0][0]
        label = "Real" if prediction > 0.5 else "Fake"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        # Track predictions and confidence
        predictions.append(1 if label == "Real" else 0)
        ground_truth.append(1)  # Assume all faces in the video are real for simplicity
        face_confidences.append(confidence)

        # Set frame and text colors
        face_color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        text_color = face_color

        # Draw face rectangle and display prediction
        cv2.rectangle(img, (x, y), (x + w, y + h), face_color, 4)
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        # Perform eye tracking if eyes are detected
        if len(eyes) >= 2:
            perform_eye_tracking(img, roi_gray, roi_color, eyes, face_area=w * h)

        # Perform audio authentication
        if not authenticated:
            if not perform_audio_authentication():
                # Block access and exit
                cv2.putText(img, "Access Blocked!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Face Liveness Detection with Audio", img)
                cv2.waitKey(3000)  # Display message for 3 seconds
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nFinal Result: {final_result}")
                print(f"Blocked Reason: {blocked_reason}")
                calculate_metrics()
                exit()

    # Display the frame
    cv2.imshow("Face Liveness Detection with Audio and Eye Tracking", img)

    # Break loop on pressing 'Esc'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print final result and metrics
print(f"\nFinal Result: {final_result}")
if blocked_reason:
    print(f"Blocked Reason: {blocked_reason}")
calculate_metrics()
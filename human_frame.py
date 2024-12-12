from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the path to the oval mask image
uploaded_img = os.path.join(os.path.dirname(__file__), 'image/Untitled design.jpg')

# Function to generate the oval mask
def create_human_mask():
    ref_image = cv2.imread(uploaded_img, cv2.IMREAD_GRAYSCALE)
    if ref_image is None:
        raise FileNotFoundError(f"Image not found at {uploaded_img}")
    
    height, width = 1080, 1920  # Full HD dimensions
    ref_image = cv2.resize(ref_image, (width, height))

    # Threshold the mask to create a binary mask
    _, mask = cv2.threshold(ref_image, 200, 255, cv2.THRESH_BINARY_INV)
    return mask

# Generate mask once
human_mask = create_human_mask()

# Function to draw dashed contours
def draw_dashed_contour(image, contours, color, thickness=2, dash_length=20, space_length=10):
    for contour in contours:
        for i in range(len(contour)):
            start_point = tuple(contour[i][0])
            end_point = tuple(contour[(i + 1) % len(contour)][0])
            distance = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))
            
            for j in range(0, distance, dash_length + space_length):
                start_dash = (
                    int(start_point[0] + j * (end_point[0] - start_point[0]) / distance),
                    int(start_point[1] + j * (end_point[1] - start_point[1]) / distance)
                )
                end_dash = (
                    int(start_point[0] + min(j + dash_length, distance) * (end_point[0] - start_point[0]) / distance),
                    int(start_point[1] + min(j + dash_length, distance) * (end_point[1] - start_point[1]) / distance)
                )
                cv2.line(image, start_dash, end_dash, color, thickness)

# Function to generate the video frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize frame to match the mask dimensions
        frame = cv2.resize(frame, (human_mask.shape[1], human_mask.shape[0]))

        # Apply a heavy blur to the entire frame
        blurred_frame = cv2.GaussianBlur(frame, (101, 101), 0)

        # Increase the whiteness of the blurred background
        whiteness = np.full_like(frame, 255)
        blurred_frame = cv2.addWeighted(blurred_frame, 0.3, whiteness, 0.7, 0)

        # Apply the human-shaped mask
        focused_area = cv2.bitwise_and(frame, frame, mask=human_mask)
        blurred_area = cv2.bitwise_and(blurred_frame, blurred_frame, mask=cv2.bitwise_not(human_mask))
        combined = cv2.add(focused_area, blurred_area)

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Check if any face is within the oval mask
        face_in_oval = False
        for (x, y, w, h) in faces:
            face_center = (x + w // 2, y + h // 2)
            if human_mask[face_center[1], face_center[0]] == 255:  # Check if face center is in the oval
                face_in_oval = True
                break

        # Add the oval outline with dashed style based on detection
        contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_color = (0, 255, 0) if face_in_oval else (0, 0, 255)  # Green for detected, Red for not detected
        draw_dashed_contour(combined, contours, outline_color, thickness=4)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
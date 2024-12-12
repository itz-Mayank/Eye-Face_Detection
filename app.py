from flask import Flask, render_template, Response
import cv2
import random
import time
from gaze_tracking.gaze_tracking import GazeTracking

app = Flask(__name__)

# Initialize GazeTracking and webcam
gaze = GazeTracking()
cap = cv2.VideoCapture(0)

# Check if the webcam is available
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Screen dimensions and dot parameters
screen_width, screen_height = 1280, 720
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
current_dot_index = 0
dot_display_time = time.time()
dot_trace_success = [0] * len(dots)  # Track blink counts for each dot

# Parameters
dot_display_interval = 3  # Time interval for each dot
blink_delay = 0.4  # Minimum time between valid blinks
last_blink_time = time.time()


def is_blinking():
    global last_blink_time
    if gaze.is_blinking():
        if time.time() - last_blink_time > blink_delay:
            last_blink_time = time.time()
            return True
    return False


def generate_frames():
    global current_dot_index, dot_display_time, dot_trace_success

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (screen_width, screen_height))

        # Update gaze tracking
        gaze.refresh(frame_resized)
        annotated_frame = gaze.annotated_frame()

        # Black screen for random dot display
        black_frame = frame_resized.copy()
        black_frame[:] = (0, 0, 0)

        # Draw the current dot
        if current_dot_index < len(dots):
            dot_x, dot_y = dots[current_dot_index]
            cv2.circle(black_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)  # Red dot

            # Check if the user looked at the dot and blinked
            if time.time() - dot_display_time >= dot_display_interval:
                if gaze.is_center() and is_blinking():
                    dot_trace_success[current_dot_index] += 1
                    current_dot_index += 1
                    dot_display_time = time.time()  # Reset timer for next dot
        else:
            cv2.putText(black_frame, "Verification Successful!", (450, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Overlay live webcam feed at the top-right corner
        inset_height, inset_width = 120, 160
        inset_feed = cv2.resize(annotated_frame, (inset_width, inset_height))
        black_frame[10:10 + inset_height, -10 - inset_width:-10] = inset_feed  # Top-right corner

        # Encode frame for web display
        ret, buffer = cv2.imencode('.jpg', black_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

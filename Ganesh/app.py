from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Circle parameters
circle_radius = 150
desired_face_width_range = (100, 190)

# Global variable to control camera feed
camera_active = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/another.html')
def another():
    return render_template('another.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        global camera_active
        cap = cv2.VideoCapture(0)

        while True:
            if not camera_active:
                break

            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            circle_center = (width // 2, height // 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            circle_color = (0, 0, 255)
            for (x, y, w, h) in faces:
                face_center = (x + w // 2, y + h // 2)
                face_width = w

                distance_to_center = np.sqrt((face_center[0] - circle_center[0]) ** 2 +
                                             (face_center[1] - circle_center[1]) ** 2)

                if distance_to_center <= circle_radius and desired_face_width_range[0] <= face_width <= desired_face_width_range[1]:
                    circle_color = (0, 255, 0)
                    break

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, circle_center, circle_radius, 255, -1)

            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            x_start = circle_center[0] - circle_radius
            x_end = circle_center[0] + circle_radius
            y_start = circle_center[1] - circle_radius
            y_end = circle_center[1] + circle_radius

            cropped_frame = masked_frame[y_start:y_end, x_start:x_end]
            output_frame = cv2.resize(cropped_frame, (300, 300))
            cv2.circle(output_frame, (150, 150), circle_radius, circle_color, thickness=5)

            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({"status": "Camera stopped"})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({"status": "Camera started"})

if __name__ == "__main__":
    app.run(debug=True)

import cv2
import random
import time
import threading
import numpy as np
import pandas as pd  # For handling the CSV file
from gaze_tracking.gaze_tracking import GazeTracking

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

# Load words from CSV
csv_file = "english_hindi_silly_words.csv"  # Replace with your CSV file path
try:
    words_df = pd.read_csv(csv_file)
    words_list = words_df['words'].tolist()  # Assumes a column named 'words'
except Exception as e:
    print(f"Error loading CSV file: {e}")
    words_list = ["word1", "word2", "word3", "word4", "word5"]  # Fallback words

# Create random dot locations and assign words
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
dot_words = random.sample(words_list, len(dots))

# Initialize dot tracking variables
current_dot_index = 0
dot_display_time = time.time()
dot_trace_success = [0] * len(dots)  # Track blink count for each dot

# Gaze detection parameters
dot_display_interval = 3  # Time interval (in seconds) for displaying the next dot
blink_delay = 0.4  # Minimum time between valid blinks
last_blink_time = time.time()

# Blink counter
blink_count = 0  # Variable to count blinks

# Flag to stop the tasks once completed
task_completed = False

# Blink detection improvement: detect blink using more precise timing
def is_blinking():
    global last_blink_time, blink_count
    if gaze.is_blinking():
        if time.time() - last_blink_time > blink_delay:
            last_blink_time = time.time()  # Update blink time after a valid blink
            blink_count += 1  # Increment the blink count
            return True
    return False

def gaze_and_blink_task():
    global current_dot_index, dot_display_time, dot_trace_success, last_blink_time, task_completed, blink_count

    print("Starting Gaze and Blink Task...")
    while True:
        if task_completed:  # Check if the task is completed
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame
        frame_resized = cv2.resize(frame, (720, 640))  # Resize to smartphone-compatible size
        annotated_frame = frame_resized.copy()

        # Update gaze tracking
        gaze.refresh(frame_resized)
        annotated_frame = gaze.annotated_frame()

        # Draw the current dot and word
        if current_dot_index < len(dots):
            dot_x, dot_y = dots[current_dot_index]
            word = dot_words[current_dot_index]
            cv2.circle(annotated_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)  # Red dot
            cv2.putText(annotated_frame, word, (dot_x - 40, dot_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Word under the dot

            # Check if enough time has passed for current dot display
            if time.time() - dot_display_time >= dot_display_interval:
                # Assume user is looking at the dot if gaze is "center"
                if gaze.is_center():
                    # Detect blink to confirm tracing
                    if is_blinking():
                        dot_trace_success[current_dot_index] += 1  # Increment blink count for the current dot
                        current_dot_index += 1
                        dot_display_time = time.time()  # Reset timer for the next dot

        # Display task status
        if current_dot_index < len(dots):
            cv2.putText(annotated_frame, f"Trace the dot and blink! ({sum(dot_trace_success)}/{len(dots)})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "You are Damn REAL!", (400, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Display blink count
        cv2.putText(annotated_frame, f"Blink Count: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display annotated frame
        cv2.imshow("Static Dot Tracing with Blink Detection", annotated_frame)

        # Break loop on pressing 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Mark task as completed
    task_completed = True
    if sum(dot_trace_success) == len(dots):  # Check if all dots were traced and blinked successfully
        print("Detection Successful: Detected Successfully!")
        for i, count in enumerate(dot_trace_success):
            print(f"Dot {i+1} blink count: {count}")
    else:
        print("Detection Failed: Not Detected! Try Again.")
    task_completed = True  # Signal that the task is completed

if __name__ == "__main__":
    # Run the gaze and blink task in a separate thread
    gaze_thread = threading.Thread(target=gaze_and_blink_task)

    gaze_thread.start()

    gaze_thread.join()

    print("Task completed.")

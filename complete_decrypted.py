import cv2
import random
import time
import threading
import csv
import numpy as np
from gaze_tracking.gaze_tracking import GazeTracking
import speech_recognition as sr

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

# Create random dot locations
dots = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(3)]
dot_texts = []

# Load text from CSV
with open(r'C:\Users\Mayank Meghwal\Desktop\SIH_FaceLivenessProject\Face live\english_hindi_silly_words.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    dot_texts = [row[0] for row in reader]  # Assuming we only use the English word for display

if len(dot_texts) < len(dots):
    print("Warning: Not enough words in CSV. Using placeholder text.")
    dot_texts += ["Placeholder"] * (len(dots) - len(dot_texts))

# Initialize dot tracking variables
current_dot_index = 0
dot_display_time = time.time()
dot_trace_success = [0] * len(dots)  # Track blink count for each dot

# Gaze detection parameters
dot_display_interval = 3  # Time interval (in seconds) for displaying the next dot
blink_delay = 0.4  # Minimum time between valid blinks
last_blink_time = time.time()

# Flag to stop the tasks once completed
task_completed = False

# Blink detection improvement: detect blink using more precise timing
def is_blinking():
    global last_blink_time
    if gaze.is_blinking():
        if time.time() - last_blink_time > blink_delay:
            last_blink_time = time.time()  # Update blink time after a valid blink
            return True
    return False

def gaze_and_blink_task():
    global current_dot_index, dot_display_time, dot_trace_success, last_blink_time, task_completed

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

        # Draw the current dot and display text
        if current_dot_index < len(dots):
            dot_x, dot_y = dots[current_dot_index]
            cv2.circle(annotated_frame, (dot_x, dot_y), 15, (0, 0, 255), -1)  # Red dot
            cv2.putText(annotated_frame, dot_texts[current_dot_index], (dot_x - 50, dot_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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

def speech_recognition_task():
    global task_completed

    recognizer = sr.Recognizer()

    def listen_for_word():
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("Listening...")

            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("Recognizing...")
                user_word = recognizer.recognize_google(audio, show_all=False).lower()
                return user_word
            except sr.WaitTimeoutError:
                print("Timeout: Please speak a bit louder or closer.")
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the audio. Please try again.")
            except sr.RequestError as e:
                print(f"There was an issue with the speech recognition service: {e}")

    while not task_completed:
        word = random.choice(dot_texts)
        print(f"Your word is: {word}")

        user_word = listen_for_word()
        if user_word:
            if user_word.strip().lower() == word.strip().lower():
                print("Correct!")
            else:
                print(f"Incorrect! The correct word was: {word}")

    print("Speech Recognition Task Completed.")
    task_completed = True  # Signal that the task is completed

if __name__ == "__main__":
    # Run both tasks in parallel
    gaze_thread = threading.Thread(target=gaze_and_blink_task)
    speech_thread = threading.Thread(target=speech_recognition_task)

    gaze_thread.start()
    speech_thread.start()

    gaze_thread.join()
    speech_thread.join()

    print("Both tasks completed.")

import cv2
import mediapipe as mp
import threading
import socket
import json
import time
import numpy as np


# Timer variables
timer_start = None
state = "Ready"  # States: Ready, Jab, Evaluate
knuckle_x_positions = []  # To store X-coordinates of the middle finger's knuckle

def calculate_angle_2d(a, b):
    # Calculate the angle between two 2D vectors
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def send_data(data):
    server_address = ('localhost', 8052)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        # Serialize list to JSON
        json_data = json.dumps({"array": data})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()


# Array to store the coordinates (x, y, z) for the neck, left wrist, and right wrist
body_coordinates = []

# Initialize Pose from Mediapipe (with separate models for each camera)
mp_pose = mp.solutions.pose
pose1 = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose2 = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Create global variables to store the frames
frame1 = None
frame2 = None

# Function to capture video from camera 1
def capture_camera_1():
    global frame1
    cap1 = cv2.VideoCapture(2)  # Camera 1
    while cap1.isOpened():
        success, frame = cap1.read()
        if not success:
            print("Ignoring empty camera frame from Camera 1.")
            continue
        frame1 = frame

# Function to capture video from camera 2
def capture_camera_2():
    global frame2
    cap2 = cv2.VideoCapture(1)  # Camera 2
    while cap2.isOpened():
        success, frame = cap2.read()
        if not success:
            print("Ignoring empty camera frame from Camera 2.")
            continue
        frame2 = frame

# Start threads for each camera
thread1 = threading.Thread(target=capture_camera_1)
thread2 = threading.Thread(target=capture_camera_2)
thread1.start()
thread2.start()

while True:
    # Only process frames when both frames are available
    if frame1 is not None and frame2 is not None:
        # Resize frames to reduce processing time (optional)
        frame1_resized = cv2.resize(frame1, (640, 480))  # Resize to 640x480
        frame2_resized = cv2.resize(frame2, (640, 480))  # Resize to 640x480

        # Process Camera 1
        frame_rgb1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
        results1 = pose1.process(frame_rgb1)
        annotated_frame1 = frame1_resized.copy()

        # Process Camera 2
        frame_rgb2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
        results2 = pose2.process(frame_rgb2)
        annotated_frame2 = frame2_resized.copy()

        if results1.pose_landmarks and results2.pose_landmarks:
            # Extract landmarks from both cameras
            landmarks1 = results1.pose_landmarks.landmark
            landmarks2 = results2.pose_landmarks.landmark

            # Frame dimensions for scaling
            frame_height1, frame_width1, _ = frame1_resized.shape
            frame_height2, frame_width2, _ = frame2_resized.shape

            # Calculate neck coordinates (midpoint of shoulders)
            left_shoulder1 = landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder1 = landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder2 = landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder2 = landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            neck_x = (left_shoulder2.x + right_shoulder2.x) / 2  # Camera 2
            neck_y = (left_shoulder2.y + right_shoulder2.y) / 2  # Camera 2
            neck_z = (left_shoulder1.x + right_shoulder1.x) / 2  # Camera 1

            # Left wrist coordinates
            left_wrist1 = landmarks1[mp_pose.PoseLandmark.LEFT_WRIST]
            left_wrist2 = landmarks2[mp_pose.PoseLandmark.LEFT_WRIST]
            left_x = left_wrist2.x  # Camera 2
            left_y = left_wrist2.y  # Camera 2
            left_z = left_wrist1.x  # Camera 1

            # Right wrist coordinates
            right_wrist1 = landmarks1[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_wrist2 = landmarks2[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_x = right_wrist2.x  # Camera 2
            right_y = right_wrist2.y  # Camera 2
            right_z = right_wrist1.x  # Camera 1

            # Update body_coordinates array
            body_coordinates = [
                [neck_x, neck_y, neck_z],       # Neck
                [left_x, left_y, left_z],       # Left wrist
                [right_x, right_y, right_z],    # Right wrist
            ]

            #Send data
            send_data(body_coordinates)


            # Draw landmarks and coordinates on combined frame
            for i, (x, y, z) in enumerate(body_coordinates):
                cx, cy = int(x * 640), int(y * 480)
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Neck: green; Wrists: blue
                label = "Neck" if i == 0 else ("Left Wrist" if i == 1 else "Right Wrist")

                # Draw circles
                cv2.circle(annotated_frame2, (cx, cy), 10, color, -1)

                # Add text
                text = f"{label}: ({x:.2f}, {y:.2f}, {z:.2f})"
                cv2.putText(annotated_frame2, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # Concatenate the two frames side by side
        combined_frame = cv2.hconcat([annotated_frame1, annotated_frame2])

        # Display the combined frame
        cv2.imshow("Pose Estimation (Both Cameras)", combined_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
thread1.join()
thread2.join()
cv2.destroyAllWindows()

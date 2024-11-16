import cv2
import mediapipe as mp
import threading
import numpy as np
import json
import socket
import time

# Global variables for frame and body coordinates
frame1 = None
frame2 = None
capture1_running = True
capture2_running = True

# Function to send data to Unity
def send_data(data):
    server_address = ('localhost', 8052)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(server_address)
        json_data = json.dumps({"array": data})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()

# Capture frames from Camera 1
def capture_camera_1():
    global frame1
    cap1 = cv2.VideoCapture(2)  # Camera 1
    while capture1_running:
        success, frame = cap1.read()
        if not success:
            print("Ignoring empty frame from Camera 1.")
            continue
        frame1 = frame
    cap1.release()  # Release camera when done

# Capture frames from Camera 2
def capture_camera_2():
    global frame2
    cap2 = cv2.VideoCapture(1)  # Camera 2
    while capture2_running:
        success, frame = cap2.read()
        if not success:
            print("Ignoring empty frame from Camera 2.")
            continue
        frame2 = frame
    cap2.release()  # Release camera when done

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose_body_1 = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_body_2 = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start the video capture threads
thread1 = threading.Thread(target=capture_camera_1)
thread2 = threading.Thread(target=capture_camera_2)
thread1.start()
thread2.start()

# Main loop
while True:
    if frame1 is not None and frame2 is not None:
        # Resize frames
        frame1_resized = cv2.resize(frame1, (640, 480))
        frame2_resized = cv2.resize(frame2, (640, 480))

        # Process frames with Mediapipe
        frame_rgb1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
        frame_rgb2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
        results_pose_1 = pose_body_1.process(frame_rgb1)
        results_pose_2 = pose_body_2.process(frame_rgb2)

        # Check if landmarks are detected
        if results_pose_1.pose_landmarks and results_pose_2.pose_landmarks:
            # Extract landmarks
            landmarks1 = results_pose_1.pose_landmarks.landmark
            landmarks2 = results_pose_2.pose_landmarks.landmark

            # Loop over all landmarks except face and hands
            body_landmarks_1 = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.MID_HIP
            ]
            body_landmarks_2 = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.MID_HIP
            ]

            # Draw landmarks on the frame
            for idx in body_landmarks_1:
                landmark = landmarks1[idx]
                x, y = int(landmark.x * frame1_resized.shape[1]), int(landmark.y * frame1_resized.shape[0])
                cv2.circle(frame1_resized, (x, y), 5, (0, 255, 0), -1)

            for idx in body_landmarks_2:
                landmark = landmarks2[idx]
                x, y = int(landmark.x * frame2_resized.shape[1]), int(landmark.y * frame2_resized.shape[0])
                cv2.circle(frame2_resized, (x, y), 5, (0, 255, 0), -1)

            # Prepare data for sending to Unity (only positions)
            body_coordinates = []
            for idx in body_landmarks_1:
                landmark = landmarks1[idx]
                body_coordinates.extend([landmark.x, landmark.y, landmark.z])

            for idx in body_landmarks_2:
                landmark = landmarks2[idx]
                body_coordinates.extend([landmark.x, landmark.y, landmark.z])

            # Send data to Unity
            send_data(body_coordinates)

        # Show combined frame
        combined_frame = cv2.hconcat([frame1_resized, frame2_resized])
        cv2.imshow("Combined Frame", combined_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture1_running = False
            capture2_running = False
            break

# Ensure threads finish execution
thread1.join()
thread2.join()
cv2.destroyAllWindows()

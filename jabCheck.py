import cv2
import mediapipe as mp
import time
import math
import numpy as np

import pyttsx3
import threading
message =[]

def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional, adjust as needed)
    engine.setProperty('rate', 150)  # Speed (default: 200 words per minute)
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

    # Get available voices
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # 0 for male, 1 for female

    # Speak the text
    engine.say(text)
    engine.runAndWait()

def tts_in_thread(text):
    # Create and start a thread for the TTS function
    tts_thread = threading.Thread(target=text_to_speech, args=(text,))
    tts_thread.start()
    return tts_thread

left_leg_posture = ""
right_leg_posture = ""
def calculate_angle_2d(v1, v2):
    """Calculate the angle (in degrees) between two 2D vectors."""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot_product / (mag_v1 * mag_v2)
    cos_angle = min(1.0, max(cos_angle, -1.0))  # Handle numerical issues
    return math.degrees(math.acos(cos_angle))

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Open video capture for both cameras
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

# Timer variables
timer_start = None
state = "Ready"  # States: Ready, Jab, Evaluate
knuckle_x_positions = []  # To store X-coordinates of the middle finger's knuckle
show_landmarks = True  # Toggle for showing landmarks

while cap1.isOpened() and cap2.isOpened():
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()

    if not success1 or not success2:
        print("Ignoring empty camera frames.")
        continue

    # Convert the frames to RGB
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Process the first frame for pose detection
    results1 = pose.process(frame1_rgb)

    # Process the second frame for hand detection
    results2 = hands.process(frame2_rgb)

    # Annotate the frames
    annotated_frame1 = frame1.copy()
    annotated_frame2 = frame2.copy()

    if show_landmarks:
        if results1.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results2.multi_hand_landmarks:
            for hand_landmarks in results2.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Timer logic
    if state == "Ready":
        if timer_start is None:
            timer_start = time.time()
        countdown = 3 - int(time.time() - timer_start)
        if countdown > 0:
            cv2.putText(annotated_frame1, f"Get ready to jab in: {countdown}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            state = "Jab"
            timer_start = time.time()
            knuckle_x_positions = []  # Reset knuckle positions

    elif state == "Jab":
        elapsed = time.time() - timer_start
        cv2.putText(annotated_frame1, "Jab Now!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results2.multi_hand_landmarks:
            for hand_landmarks in results2.multi_hand_landmarks:
                knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                knuckle_x_positions.append(knuckle.x)  # Record X-coordinate of the knuckle

        if results1.pose_landmarks:
            # Get landmarks for knee angle calculation
            left_hip = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calculate angles for both knees
            left_knee_angle = calculate_angle_2d(
                [left_knee.x - left_hip.x, left_knee.y - left_hip.y],
                [left_ankle.x - left_knee.x, left_ankle.y - left_knee.y]
            )
            right_knee_angle = calculate_angle_2d(
                [right_knee.x - right_hip.x, right_knee.y - right_hip.y],
                [right_ankle.x - right_knee.x, right_ankle.y - right_knee.y]
            )

            # Display knee angles on the frame
            cv2.putText(annotated_frame1, f"Left Knee: {left_knee_angle:.2f}째", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(annotated_frame1, f"Right Knee: {right_knee_angle:.2f}째", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if elapsed >= 2:
            state = "Evaluate"
            timer_start = None


    elif state == "Evaluate":

        # Evaluate pose and movement

        if results1.pose_landmarks:

            left_shoulder = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

            left_hip = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

            left_knee = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

            left_ankle = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_shoulder = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            right_hip = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            right_knee = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            right_ankle = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]


            left_elbow = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

            left_wrist = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]


            # Calculate angles for left and right knees

            shoulder_to_elbow = [left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y]
            elbow_to_wrist = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
            angle = calculate_angle_2d(shoulder_to_elbow, elbow_to_wrist)

            print(f"Jab angle: {angle}")
            if abs(angle) < 30:
                message.append("Good jab, timing is good")

            else:
                message.append("Bad jab, go quicker")

            left_knee_angle = calculate_angle_2d(

                [left_knee.x - left_hip.x, left_knee.y - left_hip.y],

                [left_ankle.x - left_knee.x, left_ankle.y - left_knee.y]

            )

            right_knee_angle = calculate_angle_2d(

                [right_knee.x - right_hip.x, right_knee.y - right_hip.y],

                [right_ankle.x - right_knee.x, right_ankle.y - right_knee.y]

            )

            # Determine good or bad posture for left leg

            GOOD_POSTURE_RANGE = (5, 10)  # Threshold range for straight legs

            if GOOD_POSTURE_RANGE[0] <= left_knee_angle <= GOOD_POSTURE_RANGE[1]:
                left_leg_posture = "Good left leg"
                left_color = (0, 255, 0)  # Green for good posture

            else:
                if left_knee_angle < GOOD_POSTURE_RANGE[0]:
                    message.append("Increase the angle of the left leg")

                if left_knee_angle > GOOD_POSTURE_RANGE[1]:
                    message.append("Decrease the angle of the left leg")


                left_leg_posture = "Bad left leg"

                left_color = (0, 0, 255)  # Red for bad posture

            # Determine good or bad posture for right leg

            if GOOD_POSTURE_RANGE[0] <= right_knee_angle <= GOOD_POSTURE_RANGE[1]:

                right_leg_posture = "Good right leg"

                right_color = (0, 255, 0)  # Green for good posture

            else:
                if right_knee_angle < GOOD_POSTURE_RANGE[0]:
                    message.append("Increase the angle of the right leg")
                if right_knee_angle > GOOD_POSTURE_RANGE[1]:

                    message.append("Decrease the angle of the right leg")
                right_leg_posture = "Bad right leg"

                right_color = (0, 0, 255)  # Red for bad posture

                tts_in_thread(message)

            # Annotate the frame with angles and posture evaluations

            cv2.putText(annotated_frame1, f"Left Knee: {left_knee_angle:.2f}째 ({left_leg_posture})", (50, 100),

                        cv2.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

            cv2.putText(annotated_frame1, f"Right Knee: {right_knee_angle:.2f}째 ({right_leg_posture})", (50, 150),

                        cv2.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)

        if len(knuckle_x_positions) >= 2:

            initial_x = knuckle_x_positions[0]

            final_x = knuckle_x_positions[-1]

            sampled_variations = [abs(x - initial_x) for x in knuckle_x_positions]

            max_variation = max(sampled_variations) / initial_x * 100

            print(f"Knuckle Movement Variance: {max_variation:.2f}%")

            if max_variation <= 100:

                print("Accurate Jab!")

            else:
                tts_in_thread("Bad jab, throw a straighter jab")
                print("Not Accurate Jab!")

        else:

            print("Not enough data to evaluate hand movement.")

        print(left_leg_posture)
        print(right_leg_posture)
        state = "Ready"

    # Combine the two frames horizontally
    combined_frame = np.hstack((cv2.resize(annotated_frame1, (640, 480)),
                                cv2.resize(annotated_frame2, (640, 480))))

    # Display the combined frame
    cv2.imshow('Combined Camera Feeds - Pose and Hand Tracking', combined_frame)

    # Break the loop if 'q' is pressed, toggle landmarks with 'l'
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        show_landmarks = not show_landmarks  # Toggle landmark visibility

# Release the video capture and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
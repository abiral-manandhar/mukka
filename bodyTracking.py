# import cv2
# import mediapipe as mp
# import time  # To manage the timer
# import math  # To use acos and degrees
# import numpy as np  # For combining frames
#
# def calculate_angle_2d(v1, v2):
#     dot_product = v1[0] * v2[0] + v1[1] * v2[1]
#     mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
#     mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
#     cos_angle = dot_product / (mag_v1 * mag_v2)
#     cos_angle = min(1.0, max(cos_angle, -1.0))  # Handle numerical issues
#     return math.degrees(math.acos(cos_angle))
#
# # Initialize MediaPipe solutions
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
#                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
#                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# mp_drawing = mp.solutions.drawing_utils  # To draw landmarks
#
# # Open video capture for both cameras
# cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)
#
# # Timer variables
# timer_start = None
# state = "Ready"  # States: Ready, Jab, Evaluate
#
# while cap1.isOpened() and cap2.isOpened():
#     success1, frame1 = cap1.read()
#     success2, frame2 = cap2.read()
#
#     if not success1 or not success2:
#         print("Ignoring empty camera frames.")
#         continue
#
#     # Convert the frames to RGB
#     frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
#     frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
#
#     # Process the first frame for pose detection
#     results1 = pose.process(frame1_rgb)
#
#     # Process the second frame for hand detection
#     results2 = hands.process(frame2_rgb)
#
#     # Annotate the frames
#     annotated_frame1 = frame1.copy()
#     annotated_frame2 = frame2.copy()
#
#     # Timer logic
#     if state == "Ready":
#         # Display "Get ready to jab" and start the 3-second countdown
#         if timer_start is None:
#             timer_start = time.time()
#         countdown = 3 - int(time.time() - timer_start)
#         if countdown > 0:
#             cv2.putText(annotated_frame1, f"Get ready to jab in: {countdown}", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             state = "Jab"
#             timer_start = time.time()
#
#     elif state == "Jab":
#         # Instruct the user to jab and start the 1-second timer
#         if timer_start is None:
#             timer_start = time.time()
#         elapsed = time.time() - timer_start
#         cv2.putText(annotated_frame1, "Jab Now!", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         if elapsed >= 5:
#             state = "Evaluate"
#             timer_start = None
#
#     elif state == "Evaluate":
#         # Pose-based angle evaluation
#         if results1.pose_landmarks:
#             left_shoulder = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             left_elbow = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#             left_wrist = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#
#             # Calculate vectors and angle
#             shoulder_to_elbow = [left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y]
#             elbow_to_wrist = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
#             angle = calculate_angle_2d(shoulder_to_elbow, elbow_to_wrist)
#
#             # Evaluate the angle
#             print(f"Jab angle: {angle}")
#             if abs(angle) < 30:
#                 print("Jab Good!")
#             else:
#                 print("Jab Bad!")
#
#         # Hand-based knuckle evaluation
#         if results2.multi_hand_landmarks:
#             for hand_landmarks in results2.multi_hand_landmarks:
#                 knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
#                 joint = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
#
#                 # Compare y-coordinates
#                 if knuckle.y < joint.y:
#                     print(knuckle.y, joint.y)
#                     print("Knuckle y-coordinate is greater than finger joint y-coordinate.")
#                 else:
#                     print(knuckle.y, joint.y)
#
#                     print("Knuckle y-coordinate is not greater than finger joint y-coordinate.")
#
#                 # Compare x-coordinates
#                 if knuckle.x == joint.x:
#                     print(knuckle.x, knuckle.y)
#
#                     print("Knuckle x-coordinate is the same as finger joint x-coordinate.")
#                 else:
#                     print(knuckle.x, knuckle.y)
#
#                     print("Knuckle x-coordinate is not the same as finger joint x-coordinate.")
#
#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(
#                     annotated_frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
#                 )
#
#         state = "Ready"  # Reset state for the next jab
#         timer_start = None
#
#     # Combine the two frames horizontally
#     combined_frame = np.hstack((cv2.resize(annotated_frame1, (640, 480)),
#                                 cv2.resize(annotated_frame2, (640, 480))))
#
#     # Display the combined frame
#     cv2.imshow('Combined Camera Feeds - Pose and Hand Tracking', combined_frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
#
# # Release the video capture and close windows
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import time
import math
import numpy as np

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

    # Timer logic
    if state == "Ready":
        # Display "Get ready to jab" and start the 3-second countdown
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
        # Instruct the user to jab and start the timer
        elapsed = time.time() - timer_start
        cv2.putText(annotated_frame1, "Jab Now!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results2.multi_hand_landmarks:
            for hand_landmarks in results2.multi_hand_landmarks:
                knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                knuckle_x_positions.append(knuckle.x)  # Record X-coordinate of the knuckle

        if elapsed >= 5:
            state = "Evaluate"
            timer_start = None

    elif state == "Evaluate":
        if len(knuckle_x_positions) >= 2:
            initial_x = knuckle_x_positions[0]
            final_x = knuckle_x_positions[-1]
            sampled_variations = [abs(x - initial_x) for x in knuckle_x_positions]
            max_variation = max(sampled_variations) / initial_x * 100

            print(f"Initial X: {initial_x:.3f}")
            print(f"Final X: {final_x:.3f}")
            print(f"Maximum Variation: {max_variation:.2f}%")

            if max_variation <= 20:
                print("Accurate Jab!")
            else:
                print("Not Accurate Jab!")
        else:
            print("Not enough data to evaluate.")

        state = "Ready"  # Reset state for the next jab

    # Combine the two frames horizontally
    combined_frame = np.hstack((cv2.resize(annotated_frame1, (640, 480)),
                                cv2.resize(annotated_frame2, (640, 480))))

    # Display the combined frame
    cv2.imshow('Combined Camera Feeds - Pose and Hand Tracking', combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()


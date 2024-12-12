import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Set up directories for saving images
data_dir = './data/dataset/'
os.makedirs(data_dir, exist_ok=True)
class_names = [f'left_{i}' for i in range(10)] + [f'right_{i}' for i in range(10)]

# Create subdirectories for each class
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters for collecting images
num_images = 150  # Number of images per class
image_size = (256, 256)  # Resize images to 512x512 pixels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    for i in range(10):
        for hand_type in ['left', 'right']:
            class_name = f"{hand_type}_{i}"
            print(f"Collecting images for class: {class_name}")
            count = 0
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image.")
                    break

                # Flip the image horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and find hands
                results = hands.process(image_rgb)

                # Check if any hands are detected
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Get the handedness
                        handedness = results.multi_handedness[idx].classification[0].label.lower()
                        if handedness != hand_type:
                            continue  # Skip if the detected hand does not match the current type

                        # Draw the hand landmarks and connections
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                        # Get bounding box coordinates
                        h, w, _ = frame.shape
                        x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                        y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                        x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
                        y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

                        # Expand the bounding box slightly for better cropping
                        margin = 20
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(w, x_max + margin)
                        y_max = min(h, y_max + margin)

                        # Crop the hand region
                        hand_roi = frame[y_min:y_max, x_min:x_max]

                        # Resize and save the cropped hand image
                        if hand_roi.size > 0:
                            resized_hand = cv2.resize(hand_roi, image_size)
                            save_path = os.path.join(data_dir, class_name, f"{count}.jpg")
                            cv2.imwrite(save_path, resized_hand)
                            count += 1

                # Display the frame with instructions
                cv2.putText(frame, f"Class: {class_name}, Image: {count + 1}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Collecting Images', frame)

                # Exit if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Add interval between classes
            print(f"Completed collection for class {class_name}. Taking a short break.")
            for j in range(5, 0, -1):
                print(f"Next class starting in {j} seconds...")
                time.sleep(1)

            if count < num_images:
                print(f"Collection for class {class_name} was interrupted.")
                break

# Release resources
cap.release()
cv2.destroyAllWindows()

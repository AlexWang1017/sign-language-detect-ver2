import cv2
import mediapipe as mp
import os
import numpy as np

# Set up directories for saving images
data_dir = './data/dataset/'
os.makedirs(data_dir, exist_ok=True)
class_names = ['0', '1', '2', '3', '4']

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
num_images = 100  # Number of images per class
image_size = (256, 256)  # Resize images to 256x256 pixels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    for class_name in class_names:
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

            # Draw hand annotations on the image
            frame.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Calculate bounding box for the hand region
                    H, W, _ = frame.shape
                    x_min = min(landmark.x for landmark in hand_landmarks.landmark) * W
                    y_min = min(landmark.y for landmark in hand_landmarks.landmark) * H
                    x_max = max(landmark.x for landmark in hand_landmarks.landmark) * W
                    y_max = max(landmark.y for landmark in hand_landmarks.landmark) * H

                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                    # Expand the bounding box slightly
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(W, x_max + margin)
                    y_max = min(H, y_max + margin)

                    # Crop the hand region
                    hand_region = frame[y_min:y_max, x_min:x_max]

                    # Resize and save the cropped hand region
                    resized_hand = cv2.resize(hand_region, image_size)
                    save_path = os.path.join(data_dir, class_name, f"{count}.jpg")
                    cv2.imwrite(save_path, resized_hand)
                    count += 1

            # Draw instructions on the frame
            cv2.putText(frame, f"Class: {class_name}, Image: {count}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collecting Images', frame)

            # Wait for user to press 's' to save the image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and results.multi_hand_landmarks:
                # Resize and save the image
                resized_frame = cv2.resize(frame, image_size)
                save_path = os.path.join(data_dir, class_name, f"{count}.jpg")
                cv2.imwrite(save_path, resized_frame)
                count += 1

            elif key == ord('q'):
                break

        if count < num_images:
            print(f"Collection for class {class_name} was interrupted.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import os
import tensorflow as tf
import mediapipe as mp
import time

# Load the trained model
data_dir = './data/dataset/'
model_path = os.path.join(data_dir, 'static_gesture_model.h5')
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['0', '1','2','3', '4','5','6','7','8','9']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5)

prev_time = 0

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks and make predictions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get bounding box coordinates
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

                # Extract the hand region and resize it to the model's input size
                hand_region = frame[y_min:y_max, x_min:x_max]
                if hand_region.size > 0:
                    hand_region_resized = cv2.resize(hand_region, (256, 256))  # Resize to match the model's input

                    # Normalize pixel values
                    hand_region_normalized = hand_region_resized / 255.0

                    # Add batch dimension for prediction
                    input_data = np.expand_dims(hand_region_normalized, axis=0)  # Shape: (1, 512, 512, 3)

                    # Make a prediction
                    try:
                        prediction = model.predict(input_data)
                        confidence = np.max(prediction[0])
                        if confidence > 0.5:
                            predicted_label = np.argmax(prediction[0])
                            predicted_character = class_names[predicted_label]

                            # Display the prediction
                            cv2.putText(frame, f"{predicted_character} ({confidence:.2f})", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Low Confidence", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Prediction Error: {e}")

        else:
            cv2.putText(frame, "No Hand Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Static Hand Gesture Recognition', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(10) == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
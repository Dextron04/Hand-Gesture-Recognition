import cv2
import numpy as np
import mediapipe as mp
import uuid
import os

# Open a connection to the webcam (usually, 0 represents the default camera)
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture and display video frames
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence=0.5, min_hand_presence_confidence=0.5) as hands:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Couldn't read frame.")
            break

        

        #BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set Flags
        image.flags.writeable = False

        #Detection
        results = hands.process(image)

        # Set Flags
        image.flags.writeable = True

        #RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        


        # Render detections
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        
        print(results)


        # Display the frame in a window
        cv2.imshow("Hand Tracking", image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

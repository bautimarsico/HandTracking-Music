import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the RGB image to detect hands
    results = hands.process(imgRGB)

    # If hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Iterate through detected landmarks
            for id, lm in enumerate(handLms.landmark):
                # Get image dimensions
                h, w, c = img.shape
                # Convert landmark coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)

                # If the landmark is the wrist (id == 0), draw a circle
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            # Draw hand landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow("image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

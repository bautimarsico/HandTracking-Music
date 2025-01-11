import cv2
import mediapipe as mp
import time
import numpy as np


class handDetector:

    def __init__ (self,mode = False,maxHands = 2,complexity = 1,detectionConf = 0.5,trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        
        # Initialize MediaPipe Hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self,img,draw = True):

        
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image to detect hands
        self.results = self.hands.process(imgRGB)

        # If hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Only draw landmarks if draw is True
                if draw:    
                    # Draw hand landmarks on the image
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                
        return img
    
    def findPosition(self,img,handNo = 0,draw = True):

        self.lmList = []
        if self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]            
            for id, lm in enumerate(myHand.landmark):
                    
                    # Get image dimensions
                    h, w, c = img.shape
                    # Convert landmark coordinates to pixel values
                    cx, cy = int(lm.x * w), int(lm.y * h)
              

                    self.lmList.append([id,cx,cy])


                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def findPointPosition(self,img,handNo = 0,identificator = 8,draw = True):

        self.lmList = []
        if self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]            
            for id, lm in enumerate(myHand.landmark):
                    
                if id == identificator:
                    # Get image dimensions
                    h, w, c = img.shape
                    # Convert landmark coordinates to pixel values
                    cx, cy = int(lm.x * w), int(lm.y * h)
              

                    return (cx,cy)

        else: 
            return None
    





    
def pointPosition (list,id):
    if len(list) > id:
        return (list[id][1],list[id][2])
    else:
        return None

def main():
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        # Capture frame-by-frame
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)

        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # Display the resulting frame
        cv2.imshow("image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(27) & 0xFF == ord('q'):
             break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

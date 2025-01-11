import cv2
import mediapipe as mp
import time
import numpy as np
import handTrackingModule as htm
import math
from scamp import *

session = Session()
piano = session.new_part("guitar")


def main():
   
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(maxHands=2)
    
    while True:
        
        # Capture frame-by-frame
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)

        # this give as a list with lists of the form [id,x,y]
        #lm0 = detector.findPosition(img,0,True)
        #lm1 = detector.findPosition(img,1,True)



        # position of a specific point  it requires img , hand number and identificator(landmark) 
        finger0 = detector.findPointPosition(img,handNo=0,identificator=4)
        finger1 = detector.findPointPosition(img,handNo=0,identificator=8)

        finger2 = detector.findPointPosition(img,handNo=1,identificator=4)
        finger3 = detector.findPointPosition(img,handNo=1,identificator=8)

        
     

        #line
        if finger0 != None and finger1 != None and finger2 != None and finger3 != None:
            

            line_size1 = math.sqrt((finger0[0]-finger1[0])**2+(finger0[1]-finger1[1])**2)
            line_size2 = math.sqrt((finger2[0]-finger3[0])**2+(finger2[1]-finger3[1])**2)
            piano.play_note(line_size1,0.5,int(line_size2)*0.005)
     

            cv2.line(img,finger0,finger1,(255,0,0),4)
            cv2.putText(img,"tone " + str(int(line_size1)),(finger1[0]-50,finger1[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

            cv2.line(img,finger2,finger3,(0,0,255),4)
            cv2.putText(img,"speed "+ str(int(line_size2)*0.001),finger3,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)


        

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

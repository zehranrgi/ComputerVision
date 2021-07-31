import cv2
import mediapipe as mp 
import time
import hand_track_module as htm

prevTime = 0  #previous
currTime = 0  #current
cap = cv2.VideoCapture("/dev/video0")

detect= htm.handDetect()

while True:
    opening, img = cap.read()
    img = detect.findHand(img)
    land_list = detect.findPosition(img)
    if len(land_list) != 0:
        print(land_list[4])

    currTime = time.time()
    fps = 1/(currTime-prevTime) 
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_DUPLEX,2, (255,0,255), 3)
    

    cv2.imshow("Image",img)
    cv2.waitKey(1)



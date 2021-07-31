import cv2
import time
import pose_module as ps 



# cap = cv2.VideoCapture("/home/zehranrgi/Videos/zoom_0.mp4")
cap = cv2.VideoCapture("/dev/video0")

#FOR FRAME RATE
prevTime = 0  #previous
currTime =  0  #current
detect = ps.poseDetect()

while True:

    opening, vid = cap.read()

    vid = detect.findPose(vid)

    landmarkList = detect.findPose(vid,draw=False)
    # if len(landmarkList) != 0:
        # print(landmarkList[0])
    # cv2.circle(vid, (landmarkList[0][1],landmarkList[0][2]), 20, (128,0,0), cv2.FILLED)

    currTime = time.time()
    fps = 1/(currTime-prevTime) 
    prevTime = currTime

    cv2.putText(vid, str(int(fps)), (10,50), cv2.FONT_HERSHEY_DUPLEX,2, (255,0,255), 3)
    

    cv2.imshow("Video",vid)
    cv2.waitKey(1)

import cv2
import mediapipe as mp 
import time

# cap = cv2.VideoCapture("/home/zehranrgi/Documents/Projects/hand_tracking/position/zoom_0.mp4")
cap = cv2.VideoCapture("/home/zehranrgi/Videos/zoom_0.mp4")

mpPosition = mp.solutions.pose

pose = mpPosition.Pose()  

mpDraw = mp.solutions.drawing_utils 

#FOR FRAME RATE
prevTime = 0  #previous
# currTime =  0  #current

while True:

    opening, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # print(results.pose_landmarks) 


    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks, mpPosition.POSE_CONNECTIONS) #we have to put poseLm why? because img is in the format BGR.


        for id, landmr, in enumerate(results.pose_landmarks.landmark):
            # print(id,landmr)


            h, w, c = img.shape #we will find the height, width and channels.
            cx, cy = int(landmr.x*w),int(landmr.y*h)

            print(id,cx,cy)
            if id ==0:
                cv2.circle(img, (cx,cy), 20, (128,0,0), cv2.FILLED)




        # mpDraw.draw_landmarks(img,poseLm, mpPose.POSE_CONNECTIONS) #we have to put poseLm why? because img is in the format BGR.






    
        
    currTime = time.time()
    fps = 1/(currTime-prevTime) 
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_DUPLEX,2, (255,0,255), 3)
       

    cv2.imshow("Video",img)
    cv2.waitKey(1)


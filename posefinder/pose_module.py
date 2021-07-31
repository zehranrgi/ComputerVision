import cv2
import mediapipe as mp 
import time

# cap = cv2.VideoCapture("/home/zehranrgi/Documents/Projects/hand_tracking/position/zoom_0.mp4")


class poseDetect():


    def __init__(self,mode=False, upperBody=False,smooth= True, detectCOnf=0.5,tracConf=0.5):



        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectCOnf=detectCOnf
        self.tracConf=tracConf



        self.mpPosition = mp.solutions.pose

        self.pose = self.mpPosition.Pose(self.mode,self.upperBody,self.smooth,
                                            self.detectCOnf,self.tracConf  )  

        self.mpDraw = mp.solutions.drawing_utils 


    def findPose(self,vid,draw=True):


        imgRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        
                
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(vid,self.results.pose_landmarks, self.mpPosition.POSE_CONNECTIONS) #we have to put poseLm why? because img is in the format BGR.

        return vid


    def getPosition(self,vid,draw=True):

        landmarkList = []

        if self.results.pose_landmarks:

        
            for id, landmr, in enumerate(self.results.pose_landmarks.landmark):
                # print(id,landmr)


                h, w, c = vid.shape #we will find the height, width and channels.
                cx, cy = int(landmr.x*w),int(landmr.y*h)
                landmarkList.append([id,cx,cy])
                if draw:
                    print(id,cx,cy)
                    if id ==0:
                        cv2.circle(vid, (cx,cy), 20, (128,0,0), cv2.FILLED)
        return landmarkList






def main():
    cap = cv2.VideoCapture("/home/zehranrgi/Videos/zoom_0.mp4")

    #FOR FRAME RATE
    prevTime = 0  #previous
    currTime =  0  #current
    detect = poseDetect()

    while True:

        opening, vid = cap.read()

        vid = detect.findPose(vid)

        landmarkList = detect.findPose(vid,draw=False)
        # print(landmarkList[0])
        # cv2.circle(vid, (landmarkList[0][1],landmarkList[0][2]), 20, (128,0,0), cv2.FILLED)

        currTime = time.time()
        fps = 1/(currTime-prevTime) 
        prevTime = currTime

        cv2.putText(vid, str(int(fps)), (10,50), cv2.FONT_HERSHEY_DUPLEX,2, (255,0,255), 3)
        

        cv2.imshow("Video",vid)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
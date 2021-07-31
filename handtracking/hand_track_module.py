import cv2
import mediapipe as mp 
import time



class handDetect():
    def __init__(self, mode=False, maxHand=2, detectionConf=0.5, trackConf=0.5):
    
        self.mode = mode
        self.maxHand = maxHand
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(self.mode,self.maxHand,self.detectionConf,self.trackConf)  #hands funct. shows the setting. for ex: how many hands etc.


        self.mpDraw = mp.solutions.drawing_utils #before that we can not see a frame when we open our cam.

    def findHand(self,img, draw=True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)  #if you show your hand, you can see the coordinates.


        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
           
                if draw: 
                    self.mpDraw.draw_landmarks(img,handLm, self.mpHands.HAND_CONNECTIONS) #we have to put handLm why? because img is in the format BGR.
        return img
    

    def findPosition(self,img, handNu=0,draw=True):

        land_list = []
        if self.results.multi_hand_landmarks:
            Hd = self.results.multi_hand_landmarks[handNu]
            for id, landmr, in enumerate(Hd.landmark):
                # print(id,landmr)


                h, w, c = img.shape #we will find the height, width and channels.
                cx, cy = int(landmr.x*w),int(landmr.y*h)
                land_list.append([id,cx,cy])
                # print(id,cx,cy)
                if draw:
                    cv2.circle(img, (cx,cy), 10, (128,0,0), cv2.FILLED)

            return land_list

def main():
#FOR FRAME RATE
    prevTime = 0  #previous
    currTime = 0  #current
    cap = cv2.VideoCapture("/dev/video0")

    detect=handDetect()

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


if __name__ == "__main__":
    main()


    

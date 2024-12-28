import time
import cv2
import mediapipe as mp
from cvModules.handDetector import HandDetector 
from cvModules.poseDetector import PoseDetector 
from cvModules.faceDetector import FaceDetector 
from cvModules.faceMeshDetector import FaceMeshDetector 
def mainHands():
    Ptime = 0
    count = 0
    cap = cv2.VideoCapture(0)
    lastFps = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)    
        imList = detector.findPosition(img)
        img, direction, amount = detector.findDirectionOfHand(img)
        #if len(imList) != 0:
        #    dedoCorazon, dedoIndice, dedoPulgar = imList[8], imList[4], imList[20]
#
        #    #distance2 = detector.findDistance(dedoCorazon, dedoIndice, img, draw=True, position = (10, 130))
        #    #distance1, img = detector.findDistance(dedoPulgar, dedoIndice, img)
        #    fingers, img = detector.findFingersUp(img, position = (200, 70))

        Ctime = time.time()
        fps = 1 / (Ctime - Ptime)
        Ptime = Ctime
        count += 1
        # print(count)
        cv2.putText(img, f'FPS: {str(int(lastFps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if count > 10:
            lastFps = fps
            count = 0
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    mainHands()
import mediapipe as mp
import cv2


class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)  # Ensure detectionCon is float
        self.trackCon = float(trackCon)          # Ensure trackCon is float
        self.modelComplexity = modelComplexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
    
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3, round = True, position = (10, 100)):    
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center point
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 # Pythagorean theorem
        if round:
            length = int(length)
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f'Distancia: {length}',position , cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )
        
        return length, img
    
    def findFingersUp(self, img, draw=True, position=(10, 100)):
        lmList = []
        if self.results.multi_hand_landmarks:
            for index, handLms in enumerate(self.results.multi_hand_landmarks):
                fingers = [0, 0, 0, 0, 0]
                # Thumb
                if handLms.landmark[4].y < handLms.landmark[3].y:
                    fingers[0] = 1
                # Index
                if handLms.landmark[8].y < handLms.landmark[6].y:
                    fingers[1] = 1
                # Middle
                if handLms.landmark[12].y < handLms.landmark[10].y:
                    fingers[2] = 1
                # Ring
                if handLms.landmark[16].y < handLms.landmark[14].y:
                    fingers[3] = 1
                # Pinky
                if handLms.landmark[20].y < handLms.landmark[18].y:
                    fingers[4] = 1
                lmList = fingers
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.putText(img, f'Dedos: {lmList}', (position[0], position[1] + 50 * index), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        return lmList, img
    def findDirectionOfHand(self, img, draw=True, position = (10, 100)):
        direction = ['', '', '']
        amount = [0, 0, 0]
        if self.results.multi_hand_landmarks:
            for index, handLms in enumerate(self.results.multi_hand_landmarks):
                indexFingerIndex = 8
                indexFinger = handLms.landmark[indexFingerIndex]
                indexFingerDipIndex = 7      
                indexFingerDip = handLms.landmark[7]
                # Izquierda, Derecha, Arriba, Abajo, Cerca, Lejos
                # Derecha
                xValues = list(map(lambda x: x.x, handLms.landmark))
                yValues = list(map(lambda x: x.y, handLms.landmark))
                zValues = list(map(lambda x: x.z, handLms.landmark))
                maxXIndex = xValues.index(max(xValues))
                maxYIndex = yValues.index(max(yValues))
                maxZIndex = zValues.index(max(zValues))
                minXIndex = xValues.index(min(xValues))
                minYIndex = yValues.index(min(yValues))
                minZIndex = zValues.index(min(zValues))
                indexFingerList = [indexFingerIndex, indexFinger.x, indexFinger.y, indexFinger.z]
                indexFingerDipList = [indexFingerDipIndex, indexFingerDip.x, indexFingerDip.y, indexFingerDip.z]
                
                def rounded(attribute):
                    # Dynamically access the 'variable' attribute using the variable passed to the function
                    return round((getattr(indexFingerDip, attribute) - getattr(indexFinger, attribute)) / self.findDistance(indexFingerList, indexFingerDipList, img, draw=False, round=False)[0], 2) * 100

                if maxXIndex == indexFingerIndex:
                    direction[0] = 'Izquierda'
                    amount[0] = rounded("x")  # Use the 'x' attribute dynamically
                    
                elif minXIndex == indexFingerIndex:
                    direction[0] = 'Derecha'
                    amount[0] = rounded("x")  # Use the 'x' attribute dynamically

                if maxYIndex == indexFingerIndex:
                    direction[1] = 'Abajo'
                    amount[1] = rounded("y")  # Use the 'y' attribute dynamically
                elif minYIndex == indexFingerIndex:
                    direction[1] = 'Arriba'
                    amount[1] = rounded("y")  # Use the 'y' attribute dynamically

                if maxZIndex == indexFingerIndex:
                    direction[2] = 'Lejos'
                    amount[2] = rounded("z")  # Use the 'z' attribute dynamically
                elif minZIndex == indexFingerIndex:
                    direction[2] = 'Cerca'
                    amount[2] = rounded("z")  # Use the 'z' attribute dynamically

                # Calculate speed
                print(amount)
                if draw:
                    h, w, c = img.shape
                    x, y = (int(indexFinger.x * w), int(indexFinger.y * h))
                    cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, f'Direccion: {direction}', (position[0], position[1] + 50 * index), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    cv2.putText(img, f'Cantidad: {amount}', (position[0], position[1] + 50 * index + 50), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        return img, direction, amount
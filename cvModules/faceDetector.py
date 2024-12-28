import mediapipe as mp
import cv2


class FaceDetector:
    def __init__(self, model = 0, detectionCon=0.5, ):
        self.model = model
        self.detectionCon = float(detectionCon)
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(model_selection = self.model,
                                              min_detection_confidence = self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        if self.results.detections:
            for _, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.detections:
            for _, detection in enumerate(self.results.detections):
                ih, iw, ic = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                lmList.append(bbox)
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return lmList

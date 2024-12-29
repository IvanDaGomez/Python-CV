import face_recognition
import cv2
import numpy as np

import os


# Load images on a path
def loadImages(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    if '.DS_Store' in myList:
        myList.remove('.DS_Store')
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames

# Find encodings of each image on the file
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Path to the images(Change it to your path)
path = 'faceRecognition/Images'

images, classNames = loadImages(path)
encodeListKnown = findEncodings(images)

print('Encoding Complete')

def main():

    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                matchPercentage = int(100 - (faceDis[matchIndex] * 100))
                y1, x2, y2, x1 = map(lambda face: face * 4, faceLoc)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{name} {matchPercentage}%', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _,frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(grayFrame)
    for face in faces:
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x1,y1),(x2,y2), (255,0,0), 2)
        landmarks = predictor(grayFrame, face)
        print(landmarks)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
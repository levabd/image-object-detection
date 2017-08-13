import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
face_ext_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalcatface_extended.xml')
face_cascade_alt = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')

eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if not type(faces) == np.ndarray:
        faces = face_ext_cascade.detectMultiScale(gray, 1.3, 5)

    if not type(faces) == np.ndarray:
        faces = face_cascade_alt.detectMultiScale(gray, 1.3, 5)

    if not type(faces) == np.ndarray:
        faces = face_cascade_alt2.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile = smile_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
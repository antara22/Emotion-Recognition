import numpy as np
import cv2
import dlib
import time

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
t0 = time.time()
path = 'new1.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = gray
faces = face_cascade.detectMultiScale(gray,1.03,5)

for (x, y, w, h) in faces:
    cv2.rectangle(img,(x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
    #vals = getDistances(landmarks,img.shape[1],img.shape[0])
    #test_data = np.float32(vals).reshape(1,len(vals))
    #result = svm.predict(test_data)
    #print (result)
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    t = (landmarks[30,0],landmarks[30,1])
    cv2.putText(img, 'Q', t,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                color=(0, 0, 255))
   
t1 = time.time()
print(t1-t0)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

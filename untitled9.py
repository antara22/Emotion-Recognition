# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:36:04 2018

"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:06:03 2018

"""

import os
import cv2
import dlib
import numpy as np
import time
from sklearn.metrics import confusion_matrix
emotions_tree = os.walk('Emotion')
images_tree = os.walk('cohn-kanade-images')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')




t_train_start = time.time()
existing_labels = []
label_locations = []
for root,dirs,files in emotions_tree:
    for file in files:
        label_locations.append((root, file))
        existing_labels.append(file[:17])

image_locations = []
for root,dirs,files in images_tree:
    for file in files:
        if file[:17] in existing_labels:
            image_locations.append((root, file))

def getEmotion(val):
    return{
        0 : 'neutral.png',
        1 : 'angry.png',
        2 : 'contempt.png',
        3 : 'disgust.png',
        4 : 'fear.png',
        5 : 'happy.png',
        6 : 'sadness.png',
        7 : 'surprise.png',
    }[val]

neutral = cv2.imread('Emojis/neutral.png',-1)
angry = cv2.imread('Emojis/angry.png',-1)
contempt = cv2.imread('Emojis/contempt.png',-1)
disgust = cv2.imread('Emojis/disgust.png',-1)
fear = cv2.imread('Emojis/fear.png',-1)
happy = cv2.imread('Emojis/happy.png',-1)
sadness = cv2.imread('Emojis/sadness.png',-1)
surprise = cv2.imread('Emojis/surprise.png',-1)

def getEmoji(val):
    return{
        0 : neutral,
        1 : angry,
        2 : contempt,
        3 : disgust,
        4 : fear,
        5 : happy,
        6 : sadness,
        7 : surprise,
    }[val]

def distanceFromPoint(pointa, pointb):
    return np.linalg.norm(np.array(pointa) - np.array(pointb))

def distanceFromCenter(point):
    return np.linalg.norm(np.array(point) - np.array((.5,.5)))

def getDistances(points,width,height):
    normalized_distances = []
    for point in points:
        norm_cur = (float(point[0,0])/float(width),float(point[0,1])/float(height))
        normalized_distances.append(distanceFromCenter(norm_cur))
    return np.float32(normalized_distances)

def normalize(points,width,height):
    normalized_points = []
    for point in points:
        norm_cur = (float(point[0, 0]) / float(width), float(point[0, 1]) / float(height))
        normalized_points.append(norm_cur)
    return np.float32(list(sum(normalized_points,())))

def normalizeFromPoint(points,nPoint):
    normalized_points = []
    distances = []
    for point in points:
        distances.append(distanceFromPoint(point,nPoint))
    max = np.amax(distances)
    for point in points:
        dx = point[0, 0] - nPoint[0, 0]
        dy = point[0, 1] - nPoint[0, 1]
        norm_cur = ( dx/max , dy/max)
        normalized_points.append(norm_cur)
    return np.float32(list(sum(normalized_points,())))

train_labels = []
count = 0
for path,file in label_locations:
    f = open(os.path.join(path,file))
    val = f.read()
    train_labels.append(float(val))
    count += 1

train_data = []
counter = 0
for path, file in image_locations:
    img = cv2.imread(os.path.join(path,file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,3)
    if len(face) != 0:
        x,y,w,h = face[0]
        rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        landmarks = np.matrix([[p.x, p.y] 
        for p in predictor(img, rect).parts()])
        norm = normalizeFromPoint(landmarks,landmarks[30])
        train_data.append(norm)
    else:
        print (train_labels.pop(counter))
        counter -=1
    counter += 1


print (len(train_labels))
print (len(train_data))

train_data_mat = np.float32(train_data).reshape(len(train_data),len(train_data[0]))
train_labels_mat = np.int32(train_labels).reshape(-1,len(train_labels))
training_data = train_data_mat[:int(len(train_data_mat)*0.8)] #get first 80% of file list
prediction_data = train_data_mat[-int(len(train_data_mat)*0.2):] #get last 20% of file list
training_labels = train_labels_mat[:int(len(train_labels_mat)*0.8)] #get first 80% of file list
prediction_labels = train_labels_mat[-int(len(train_labels_mat)*0.2):] #get last 20% of file list


svm = cv2.ml.SVM_create()
svm.setGamma(5)
svm.setC(50)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(train_data_mat, cv2.ml.ROW_SAMPLE, train_labels_mat)
svm.save('svm_data.dat')


t_train_end = time.time()
print ('Time taken to train {}'.format(t_train_end - t_train_start))



results = svm.predict(prediction_data)

correct = 0
count = 0
for val in results[1]:
    t1 = train_labels[count]
    t2 = val[0]
    if t1 == t2:
        correct += 1
    count += 1

print (float(correct)*100/len(results[1]))

cn = confusion_matrix(prediction_labels,results)

time_s = time.time()
counter = 0
max_frames = 30
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eframe = frame.copy()
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[int(y):int(y + h), int(x):int(x + w)]
        roi_color = frame[int(y):int(y + h), int(x):int(x + w)]
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y] 
        for p in predictor(frame, rect).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 3, color=(0, 255, 255))
        vals = normalizeFromPoint(landmarks,landmarks[30])
        test_data = np.float32(vals).reshape(-1, len(vals))
        result = svm.predict(test_data)
        emotion = result[1]
        emotion = emotion[0]
        emoji = getEmoji(int(emotion[0]))
        ycenter = int(y + h/2 - emoji.shape[0]/2)
        xcenter = int(x + w/ 2 - emoji.shape[1] / 2)
        try:
            for c in range(0,3):
                eframe[ycenter:ycenter+emoji.shape[0],xcenter:xcenter+emoji.shape[1],c] = \
                  emoji[:, :, c] * (emoji[:, :, 3] / 255) + eframe[ycenter:ycenter + emoji.shape[0],
                  xcenter:xcenter + emoji.shape[1], c] * (1 - emoji[:, :, 3] / 255)
        except ValueError:
            pass

    time_e = time.time()
    counter += 1
    sec = time_e - time_s
    fps = counter / sec
    if(counter > max_frames):
        counter = 0
        time_s = time.time()
    cv2.putText(frame, str(int(fps)), (0,frame.shape[0]-1), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
    cv2.putText(eframe, str(int(fps)), (0,eframe.shape[0]-1), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
    cv2.namedWindow('Regular Feed')
    cv2.imshow('Regular Feed',frame)
    cv2.namedWindow("Emoji Feed")
    cv2.imshow('Emoji Feed', eframe)

    #print fps
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
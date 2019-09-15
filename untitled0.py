# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:08:41 2018

@author: Subhodeep
"""

#this file contains a lot of code from prepData however there are plenty of addons to detect emotion, but not just prep data



# our bags for working with camera, video and markup
from imutils.video import VideoStream

from imutils import face_utils

import time

import argparse

import imutils

import time

import dlib

import cv2

import os

import sys

import numpy as np

#qimport tensor



# ------------------------ Define all parameters, arguments and global variables


# File prefix for the found person
# File prefix for the found person
# new filename detectedFilePrefix + "_" + fileName + "_" + faceNum
detectedFilePrefix = "det_"



# Instead of the pars argument -s, we will simply check for the function itself, the folder we have, or the file. The result of the check in the next global variable
singleImage = False



# We need a file that contains the training data. I immediately transfer it in default, so as not to indicate constantly
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",required=False,

	help="path to facial landmark predictor")

ap.add_argument("-i", "--imagePath", default="", required=False, help="path to image folder or image")

ap.add_argument("-d", "--drawCircles", action='store_true',	help="pass this flag if you want to draw points on aligned images")

args = vars(ap.parse_args())

 



# And here is our predictor, which will specify the points. First the face detector, and then the predictor of points
print("[INFO] load the detector and the predictor ...")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])





# Connect the camera if the path to the pictures is not specified

if (args["imagePath"]==""):

	print("[INFO] the camera has gone! ..")

	vs = VideoStream(-1).start()

	time.sleep(2.0)

else:

	print("[INFO] image mode is selected")



# ------------------------------- Define all the functions we need
	

# Let's check that we have one image in the path or a folder. It is necessary that the sweatом правильно определять директории для библиотеки os	

def checkIfSingleImage(path):

	image = cv2.imread(path)

	if (image is None):

		print('[INFO] Folder selected')

		return False

	else:

		print('[INFO] 1 file selected')

		return True



#if we want to save every face found

def addPrefix(picName):

# Since. picName contains the path, it must be split to add "det_" before the file name
	slashPos=picName.rfind("/")

# if it is in a folder, then break the path and add det_ there, otherwise just add det_
	if  (slashPos>-1):

		firstPart=picName[:slashPos+1]

		lastPart=picName[1+slashPos:]

		picName=firstPart+detectedFilePrefix+lastPart

		return picName

	else:

		picName=detectedFilePrefix+picName

		return picName



def detectAndMark(picture,picName,drawCircles):

	#as result of this function we will have ...



	print(picName+" is being proccessed")



	picName=addPrefix(picName)



		

	#to increase speed, WARNING decreases accuracy

	#picture = imutils.resize(picture, width=400)



# Then in the CB to work HOG
	#img = cv2.cvtColor(picture, cv2.COLOR_BGR2img)

	img = cv2.cvtColor(picture,cv2.COLOR_BGR2RGB)

 



	# and here works HOG with the detector

# If there are a lot of faces on the images, then you can give the second argument not 0, but 1, then the number of samples will be increased for a more detailed search of persons. Increases the time by 4 times

	rects = detector(img, 0)



	#to let user know of faces found

	curFaceNum = 0

	

# process each person
	if len(rects)>0:

		for rect in rects:

			features = []

			curFaceNum+=1

			# define all brands using the predictor

			shape = predictor(img, rect)

			

# This is for the coordinates of the points of the face in the form of an array
			shapeArr = face_utils.shape_to_np(shape)

	

# If we want to draw tags on a photo, then we draw
			if (drawCircles):

# Draw the labels
				for (x, y) in shapeArr:

					cv2.circle(img, (x, y), int((rect.right()-rect.left())/50), (255, 0, 0), -1)	

	

# Align the face
			full_det=dlib.full_object_detection()

			full_det=(shape)

			image=dlib.get_face_chip(img,full_det)

			

			#а теперь записываем в файл

			#cv2.imwrite(picName+"_"+str(curFaceNum)+".jpg",picture[rect.top():rect.bottom(),rect.left():rect.right()]) #оставлю этот вариант, чтобы была возможность сравнивать до выравнивая и после



			#we create set of features for each face

			for (x, y) in shapeArr:

				features.append(((x-rect.left())/(rect.right()-rect.left())))

				features.append(((y-rect.top())/(rect.bottom()-rect.top())))

			features=np.asarray(features)

			use_neural_network(features)

				

		print("{} faces detected".format(curFaceNum))

	else: 

		print("ops, no faces here")





def detectEmotion(frame):

# Resize to 400 pixels in width to make it more convenient and faster to work
	frame = imutils.resize(frame, width=400)

	#потом в ЧБ, чтобы работал HOG

	#img = cv2.cvtColor(frame, cv2.COLOR_BGR2img)

	img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

 

# and here works HOG with the detector
	rects = detector(img, 0)

	

# process each person
	for rect in rects:

# define all brands using the predictor
		shape = predictor(img, rect)

		shape = face_utils.shape_to_np(shape)	

		

		features = []

# Draw the labels
		for (x, y) in shape:

			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			features.append(((x-rect.left())/(rect.right()-rect.left())))

			features.append(((y-rect.top())/(rect.bottom()-rect.top())))

		

		features=np.asarray(features)

		tensor.use_neural_network(features)

		

	cv2.imshow("Frame", frame)



if __name__ == '__main__':	


	curTime=time.time()

	# ---- Part with camera

# We will not greatly reduce the code, let it be better understood

# If the images from the camera, then go to the endless cycle
	if (args["imagePath"]==""):

		#the video from the camera goes on endlessly

		while True:

			# take an image from the camera,

			frame = vs.read()

			

# looking for people
			detectEmotion(frame)

			

			# and this is our opportunity to exit the program

			key = cv2.waitKey(1) & 0xFF

		 

# so, pressing q, it is obligatory in English. layout, progarmma will be released
			if key == ord("q"):

				break

				

# close the window and turn off the camera
		cv2.destroyAllWindows()

		vs.stop()

	# ---- Part with files

# And if not from the face, then sort through all the files in the folderaqq

	else:

		

		#проверим, у нас папка или файл

		singleImage = checkIfSingleImage(args["imagePath"])

		#если выбран только файл, то просто укажем, что список файлов состоит из 1 файла

		if singleImage:

			filesList=[args["imagePath"]]

		#иначе работает с папкой и указываем полный список файлов

		else:

			directory=os.fsencode(args["imagePath"])

			filesList=os.listdir(directory)

		

		#А дальше перебираем все файлы, даже если он 1

		for file in filesList:

			filename=os.fsdecode(file)

			#Если файл не 1, то делаем поправку на путь

			if (not singleImage):

				filename=args["imagePath"]+"/"+filename

			picture=cv2.imread(filename)

			detectAndMark(picture,filename, args['drawCircles'])

		

	print('detectionTime is '+str(time.time()-curTime)+' secs')
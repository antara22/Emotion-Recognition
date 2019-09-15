# EmojiMapping
Detects a face and determines what emoji is most closely represented by the facial expressions

MainTest.py:  
This is the main python script, it contains svm training + live camera facial tracking/emotion predictions
 
Test.py:  
Basic opencv facial detection + dlib landmark detection test script

VideoTest.py:  
Basic opencv video testing script

System Requirements:  
Python 2.7._  
OpenCV 3.1 (I believe 3.0 and other versions may work as well)  
dlib 19.2.0 (this may require boost 1.6.0/cmake/visual C++ if you are on windows)  
cohn-kanade image database for svm training  
numpy  
(this list may be incomplete)

import numpy as np
import cv2 
from random import randrange
from google.colab.patches import cv2_imshow

trained_face_data=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

img=cv2.imread("/content/robert-downey-jr-CPD8HG.jpg")

grayscaling_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates=trained_face_data.detectMultiScale(grayscaling_img)
#print(face_coordinates)
(x,y,w,h)=face_coordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,125),4)

cv2_imshow(img)
cv2.waitKey(0)


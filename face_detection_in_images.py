#  code for only image 

import cv2
from random import randrange 

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in 
# reading image array like bigmatrix with bunch of numbers pixels 
# img = cv2.imread('RDJ.jpg')
img = cv2.imread('image\\1.png')
# img = cv2.imread('JT.jpg')


# convert to grayscale
grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


print(face_coordinates)

# Draw rectangles around the faces 
for ( x, y ,w ,h) in face_coordinates:
    cv2.rectangle(img, (x ,y), (x+w ,y+h), (randrange(128,256),randrange(256),randrange(256)), 2)


# for 1 face
# (x,y,w,h)=face_coordinates[0] # for 1 person image
# (x,y,w,h)=face_coordinates[1] # for 2 person image
# cv2.rectangle(img, (101 ,168), (394+101 ,394+168), (0,255,0), 2)

# show the image (waitkey() :  to pause the screen)
# imshow('TITLE',image)

# Display image with the faces
cv2.imshow('Face Detector',img)
cv2.waitKey()


print('Code Completed')

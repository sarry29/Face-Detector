# AI Face detection with python
#  code for videoo stream

import cv2
from random import randrange 

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# To capture video from webcam
# webcam=cv2.VideoCapture('C:\\mobile\\vn\\1.mp4')
webcam=cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # read the current frame
    successful_frame_read,frame = webcam.read()

    # convert to grayscale
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces 
    for ( x, y ,w ,h) in face_coordinates:
    # (x , y, w , h) in face_coordinates[0]
        cv2.rectangle(frame, (x ,y), (x+w ,y+h), (randrange(128,256),randrange(256),randrange(256)), 2)
        # cv2.rectangle(frame, (x ,y), (x+w ,y+h), (0,255,0), 2)

    # Display image with the faces
    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1)
    # wait for 1 millisec and hit key by itself

    #stop if Q key is pressed 
    if key==81 or key==113:
        break

# Relase the videocaptures
webcam.release()

print('Code Completed')
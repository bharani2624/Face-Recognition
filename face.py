#/home/abd/pam/pam/bin/python
# import face_recognition
# import cv2
# import numpy
# import pam
# face=face_recognition.load_image_file("/home/abd/pam/abd.jpg")
# known_face_encoding=face_recognition.face_encodings(face)[0]
# cap=cv2.VideoCapture(0)
# p=pam.pam()
# process_frame=True
# face_detected=False
# while True:
#     ret,frame=cap.read()
#     fast_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
#     rgb_frame=numpy.ascontiguousarray(fast_frame[:,:,::-1])#Converting From bgr(opencv format) to rgb(face_recognition)

#     if process_frame:
#         face_feature_location=face_recognition.face_locations(rgb_frame)
#         face_encodings=face_recognition.face_encodings(rgb_frame,face_feature_location)

#         for face_encoding in face_encodings:
#             matches=face_recognition.compare_faces([known_face_encoding],face_encoding)

#             if True in matches:
#                 print("System Unlocked")
#                 face_detected=True

#                 if p.authenticate('abd','2624'):
#                     print("Authentication Successful")
#                     exit()
#                 break
#     process_frame=not process_frame
#     if face_detected:
#         break
    
#     for (top,right,bottom,left) in face_feature_location:
#         top*=4
#         right*=4
#         bottom*=4
#         left*=4
#         cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
#     cv2.imshow('Face Recognition',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

#!/usr/bin/env python3

import face_recognition
import cv2
import numpy
import pam
import sys
import time
from playsound import playsound
def play():
    playsound("ItsHerTime.wav")
    sys.exit(0)
# Load the known face
face = face_recognition.load_image_file("/home/abd/pam/cap.jpg")
known_face_encoding = face_recognition.face_encodings(face)[0]

# Start capturing video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(2)  # Exit with code 2 for error

process_frame = True
face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        sys.exit(2) 

    fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = numpy.ascontiguousarray(fast_frame[:, :, ::-1])

    if process_frame:
        face_feature_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_feature_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

            if True in matches:
                sys.exit(0)
                play()
                time.sleep(1)
                print("Face recognized")
                face_detected = True

    process_frame = not process_frame

    if face_detected:
        break

print("Face not recognized")
sys.exit(1)

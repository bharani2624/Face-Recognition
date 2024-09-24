import cv2
import time
cap=cv2.VideoCapture(0)
time.sleep(1)
ret,image=cap.read()
if ret:
    cv2.imwrite(f'/home/abd/pam/dataset/cap{10}.jpg',image)
cap.release()

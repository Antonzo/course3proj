import cv2
import random
cap = cv2.VideoCapture(1)
v = random.randint
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("C:/Users/tonym/Desktop/ProjectX/calib_extra/img" + str(v(1, 10000)) + ".jpg" , frame)
        cv2.waitKey(100)
        
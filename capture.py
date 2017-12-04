import cv2
import numpy as np

clicked = False
def onMouse(event,x,y,flags,param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)


print('Showing camera feed. Click window or press any key to stop.')

count = 1
success,image = cameraCapture.read()
while success and cv2.waitKey(1) != 27:
    if clicked:
        cv2.imwrite('data/left0'+str(count)+'.png', image)
        count = count+1
        clicked = False
    cv2.imshow('MyWindow', image)
    success, image = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()

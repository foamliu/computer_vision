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

success,image = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.2 * dst.max()] = [0, 0, 255]

    cv2.imshow('MyWindow', image)
    #cv2.imshow('MyWindow', frame)
    success, image = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()

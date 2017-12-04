
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('fly.png',0)
surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(img,None)
print(len(kp))
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

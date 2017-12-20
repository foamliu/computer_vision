import numpy as np
import cv2
from matplotlib import pyplot as plt

bg = cv2.imread('data/box_in_scene.png', 0)
img = cv2.imread('data/yimeng.png')
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

M = np.array([[  4.41540342e-01,  -1.60499555e-01,   1.18716337e+02],
              [ -4.42837759e-04,   4.07985472e-01,   1.60960377e+02],
              [ -2.50978655e-04,  -3.38322580e-04,   1.00000000e+00]])

rows, cols = bg.shape
print(rows, cols)
img = cv2.warpPerspective(img, M, (cols, rows))

roi = bg[0:rows, 0:cols ]
ret, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img,img,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)

cv2.imshow('res',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('ORB')

image_0 = cv2.imread('data/slambook.jpg')

# Initiate ORB detector
orb = cv2.ORB_create(2000)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
kp_0 = orb.detect(image_0, None)
kp_0, des_0 = orb.compute(image_0, kp_0)

ret,image = cameraCapture.read()
while ret and cv2.waitKey(1) != 27:
    # find the keypoints with ORB
    kp = orb.detect(image, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)

    # Match descriptors.
    matches = bf.match(des,des_0)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # Draw first 40 matches.
    mat_img = cv2.drawMatches(image,kp,image_0,kp_0,good[:40],None,flags=2)

    cv2.imshow('MyWindow', mat_img)
    success, image = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()

"""
Rotated Star Image Detector
@author: Okan Yalçın
date:7.11.2020


"""

import cv2
import numpy as np


rotated_img = cv2.imread("Small_area_rotated.png",cv2.IMREAD_GRAYSCALE)
star_map = cv2.imread("StarMap.png",cv2.IMREAD_GRAYSCALE)

#ORB detector
orb = cv2.ORB_create()
kp1 , des1 = orb.detectAndCompute(rotated_img,None)
kp2 , des2 = orb.detectAndCompute(star_map,None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = False)
matches = bf.match(des1,des2)
matches=sorted(matches,key = lambda x:x.distance)

matching_result = cv2.drawMatches(rotated_img,kp1,star_map,kp2,matches[:50],None,flags=2)

cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
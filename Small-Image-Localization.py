"""
4DSight
Employement Project
Prepared by: Okan Yalçın
5.11.2020

"""

#Import necessary modules
import matplotlib.pyplot as plt
import cv2

#Load image as gray scaled
star_map = cv2.imread("StarMap.png",0)
small_image = cv2.imread("Small_area_rotated.png",0)

#Width and height of the template image stored
w,h = small_image.shape

#Square Difference Norm method applied to find the match.
res = cv2.matchTemplate(star_map,small_image,cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(res)


#Coordinates of the small image
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
print("Detected image coordinates:")
print("Top left corner:", top_left)
print("Top right corner:",(top_left[0]+w,top_left[1]))
print("Bottom right:",bottom_right)
print("Bottom left:",(bottom_right[0]-w,bottom_right[1]))


#Shows the result.
cv2.rectangle(star_map,top_left,bottom_right,255,2)
plt.subplot(121)
plt.imshow(res,cmap="gray")
plt.title("Detected Point")

plt.xticks([]),plt.yticks([])
plt.subplot(122)
plt.imshow(star_map,cmap="gray")
plt.title("Matched Result")
plt.show()
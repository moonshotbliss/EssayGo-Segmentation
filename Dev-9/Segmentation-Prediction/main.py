
#imports and libraries
from segmentation import lineSegment
import cv2
import numpy as np
import time
import cv2

#read import image
src = cv2.imread('images/image-3.jpg', cv2.IMREAD_UNCHANGED)

#percent by which the image is resized
scale_percent = 50

#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)


# Displaying the image
cv2.imshow("Resized image",output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#pass the image in lineSegment function
final = lineSegment(output)
print (final)

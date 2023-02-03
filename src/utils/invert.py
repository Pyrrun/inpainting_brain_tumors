import cv2
import os
import numpy as np

for mask in os.listdir('../../Masks/pconv'):
    image = cv2.imread(os.path.join('../../Masks/pconv',mask))
    inverted_image = cv2.bitwise_not(image)
    cv2.imwrite(os.path.join('../../Masks/inverted',mask),inverted_image)
#image = cv2.imread("image.png", 0)
#inverted_image = cv2.bitwise_not(image)
#cv2.imwrite("inverted.jpg", inverted)
#cv2.imshow("Original Image",image)
#cv2.imshow("Inverted Image",inverted_image)
